"""Read-only adapters for frozen splits and stored candidate evidence.

No fitting, graph construction, interval-to-membership assignment or numerical
metric evaluation is allowed here. All memberships come from saved tables.
"""

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from flir_pipeline.clustering.inspection import inspect_clustering
from flir_pipeline.data.identity import dataset_id_from_manifest
from flir_pipeline.data.local_images import declared_file
from flir_pipeline.detection.protocol import experiment_matrix, verify_plan
from flir_pipeline.explorer.data import load_cluster, load_split, with_split
from flir_pipeline.explorer.discovery import (
    checked_file,
    discover_runs,
    read_json,
    sha256,
)
from flir_pipeline.similarity.storage import stable_id
from flir_pipeline.splitting.base import SplitConfig, split_space_id


@dataclass
class Sources:
    """Watch every consumed byte before parsing, then again before publication."""

    files: dict[Path, str] = field(default_factory=dict)
    receipts: list[dict] = field(default_factory=list)

    def watch(self, path):
        path = Path(path).resolve()
        actual = sha256(path)
        if path in self.files and self.files[path] != actual:
            raise ValueError("Source changed during export")
        self.files[path] = actual
        return actual

    def artifact(self, directory, meta, identity):
        checks = meta.get("output_sha256", meta.get("output_checksums", {}))
        digest = self.watch(directory / "metadata.json")
        if read_json(directory / "metadata.json") != meta:
            raise ValueError("Source metadata changed during discovery")
        for name, expected in checks.items():
            if self.watch(declared_file(directory, name)) != expected:
                raise ValueError(f"Source checksum mismatch: {name}")
        if (directory / "receipt.json").is_file():
            self.watch(directory / "receipt.json")
        receipt = dict(
            artifact_id=identity,
            artifact_kind=meta["artifact_kind"],
            metadata_sha256=digest,
            output_checksums=checks,
        )
        existing = [r for r in self.receipts if r["artifact_id"] == identity]
        if existing and existing != [receipt]:
            raise ValueError("Ambiguous artifact identity")
        if not existing:
            self.receipts.append(receipt)

    def unchanged(self):
        if any(
            not p.is_file() or sha256(p) != digest for p, digest in self.files.items()
        ):
            raise ValueError("Source changed during export; publication cancelled")


def selected_runs(plan_directory, manifest_path, split_root, clustering_root, sources):
    """The frozen plan selects runs, never aliases or directory naming conventions."""
    sources.watch(manifest_path)
    sources.watch(plan_directory / "plan.json")
    sources.watch(plan_directory / "runtime_freeze.json")
    plan = verify_plan(plan_directory)
    freeze = read_json(plan_directory / "runtime_freeze.json")
    if (
        freeze["plan_id"] != plan["plan_id"]
        or stable_id(freeze["model_config"]) != freeze["model_config_id"]
    ):
        raise ValueError("Runtime freeze differs from selected plan")
    # Hardware freeze may resolve batch size; it cannot select a new experiment.
    from copy import deepcopy

    config = deepcopy(plan["identity"]["config"])
    config["train"]["batch"] = freeze["model_config"]["protocol_config"]["train"][
        "batch"
    ]
    if config != freeze["model_config"]["protocol_config"]:
        raise ValueError("Runtime freeze changes the scientific protocol")
    selected = plan["identity"]["splits"]
    matrix = experiment_matrix(selected, config["training_seeds"])
    if (
        matrix.empty
        or matrix.duplicated(["strategy", "split_seed", "detector_seed"]).any()
    ):
        raise ValueError("Empty or duplicate frozen experimental cells")
    manifest = pd.read_parquet(manifest_path)
    dataset = dataset_id_from_manifest(manifest)
    if (
        manifest.empty
        or manifest.content_id.isna().any()
        or not manifest.content_id.eq(manifest.image_sha256).all()
    ):
        raise ValueError("Manifest content IDs must preserve exact byte identity")
    # Discovery excludes partial/ambiguous identities. Any selected exclusion
    # therefore fails resolution, while abandoned unselected runs are irrelevant.
    split_runs, _ = discover_runs(split_root)
    cluster_runs, _ = discover_runs(clustering_root)
    split_lookup = {r.space_id: r for r in split_runs}
    cluster_lookup = {r.space_id: r for r in cluster_runs}
    splits, clusters = [], {}
    for spec in sorted(
        selected, key=lambda r: (r["strategy"], r["split_seed"], r["split_space_id"])
    ):
        run = split_lookup.get(spec["split_space_id"])
        if run is None:
            raise ValueError(f"Cannot resolve selected split: {spec['split_space_id']}")
        sources.artifact(run.directory, run.metadata, run.space_id)
        if (
            sources.files[(run.directory / "metadata.json").resolve()]
            != spec["split_metadata_sha256"]
        ):
            raise ValueError("Split metadata differs from frozen selection")
        payload = run.metadata["identity_payload"]
        SplitConfig(**payload["configuration"])
        if (
            split_space_id(payload) != run.space_id
            or run.dataset_id != dataset
            or spec["dataset_id"] != dataset
            or run.seed != spec["split_seed"]
            or run.clustering_space_id != spec["clustering_space_id"]
        ):
            raise ValueError("Frozen split identity/dataset/seed mismatch")
        if (run.strategy == "cluster_aware") != (run.clustering_space_id is not None):
            raise ValueError("Split clustering reference violates strategy")
        if run.strategy != "cluster_aware" and spec["strategy"] != run.strategy:
            raise ValueError("Frozen baseline strategy mismatch")
        split = load_split(run, manifest)
        # Also bind the saved unique-content assignment table, without running
        # split metric evaluators or solving any assignments.
        content_rows = pd.read_parquet(checked_file(run, "split_assignments.parquet"))
        if not content_rows.content_id.is_unique or set(content_rows.content_id) != set(
            manifest.content_id
        ):
            raise ValueError("Duplicate/unknown content in split assignment index")
        for row in content_rows.itertuples():
            presence = sorted(
                split.records.loc[
                    split.records.content_id.eq(row.content_id), "new_split"
                ].unique()
            )
            if (
                len(presence) == 1
                and row.new_split != presence[0]
                or len(presence) > 1
                and not pd.isna(row.new_split)
            ):
                raise ValueError(
                    "Saved content assignment flattens occurrence membership"
                )
            if run.strategy == "historical":
                import json

                if json.loads(row.split_membership_set) != presence:
                    raise ValueError("Historical membership set mismatch")
        for partition in ("train", "val", "test"):
            if (
                int(split.records.new_split.eq(partition).sum())
                != spec["record_counts"][partition]
            ):
                raise ValueError("Record counts differ from frozen plan")
        if run.clustering_space_id:
            cid = run.clustering_space_id
            if cid not in cluster_lookup:
                raise ValueError(f"Missing selected scientific clustering: {cid}")
            if cid not in clusters:
                cluster_run = cluster_lookup[cid]
                sources.artifact(cluster_run.directory, cluster_run.metadata, cid)
                inspect_clustering(cluster_run.directory)
                clusters[cid] = load_cluster(cluster_run, manifest)
            with_split(clusters[cid], split)
        splits.append((spec, split))
    sources.unchanged()
    return manifest, plan, splits, clusters


@dataclass
class CandidateEvidence:
    """Raw authoritative rows; IDs remain scoped by artifact, never reminted."""

    groups: list[dict] = field(default_factory=list)
    memberships: list[dict] = field(default_factory=list)
    boundaries: list[dict] = field(default_factory=list)
    reviews: list[dict] = field(default_factory=list)
    video_records: list[pd.DataFrame] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)


def _occurrence_binding(frame, manifest, signature, manifest_digest):
    if (
        signature["dataset_id"] != dataset_id_from_manifest(manifest)
        or signature["checksums"]["manifest_sha256"] != manifest_digest
    ):
        raise ValueError("Candidate evidence belongs to a different manifest")
    if not frame.frame_id.is_unique or not set(frame.frame_id) <= set(
        manifest.frame_id
    ):
        raise ValueError("Candidate occurrences reference unknown/duplicate records")
    expected = manifest.set_index("frame_id").loc[frame.frame_id]
    if list(expected.content_id) != list(frame.content_id):
        raise ValueError("Candidate occurrence content mapping changed")
    from flir_pipeline.data.temporal import audit_temporal_lineage

    lineage = (
        audit_temporal_lineage(manifest, max_frame_gap=0)
        .lineage.set_index("frame_id")
        .loc[frame.frame_id]
    )
    if list(frame.timeline_id) != list(lineage.possible_sequence):
        raise ValueError(
            "Candidate timeline differs from authoritative filename lineage"
        )
    actual_positions = frame.position.astype("Int64").reset_index(drop=True)
    expected_positions = lineage.possible_frame_index.where(
        lineage.order_reconstructable_from_name
    ).reset_index(drop=True)
    if not actual_positions.equals(expected_positions):
        raise ValueError(
            "Candidate positions differ from authoritative filename indices"
        )
    return expected


def experimental_candidates(
    directory, manifest, manifest_digest, sources, result, directories
):
    """Join stored component→core→content tables; never rebuild components."""
    from flir_pipeline.sequences.experiments.artifacts import inspect, tables

    raw_meta = read_json(directory / "metadata.json")
    sources.artifact(directory, raw_meta, raw_meta["artifact_id"])
    meta = inspect(directory)
    data = tables(directory)
    aid = meta["artifact_id"]
    occurrence = data["occurrences"]
    _occurrence_binding(
        occurrence, manifest, meta["identity"]["sources"]["input"], manifest_digest
    )
    if meta["artifact_kind"] == "sequence_structure_review_v1":
        native = meta["identity"]["sources"].get("native_producers", {})
        if native:
            from types import SimpleNamespace

            from flir_pipeline.sequences.experiments.native_evidence import (
                verify_native,
            )

            for producer in native.values():
                for name in producer["source_files"]:
                    sources.watch(
                        declared_file(Path(producer["source_artifact_path"]), name)
                    )
            signature = meta["identity"]["sources"]["input"]
            verify_native(
                directory,
                SimpleNamespace(
                    records=occurrence,
                    signature=signature,
                    family=signature["family"],
                    unchanged=sources.unchanged,
                ),
            )
        intervals = data["intervals"]
        from flir_pipeline.sequences.experiments.structure import Interval

        if not intervals.element_id.is_unique:
            raise ValueError("Duplicate interval ID")
        for raw in intervals.to_dict("records"):
            zone = Interval.model_validate(raw)
            if zone.kind == "boundary_zone":
                ids = occurrence.loc[
                    occurrence.timeline_id.eq(zone.timeline_id), "frame_id"
                ].tolist()
                if not ids:
                    raise ValueError("Boundary references an unknown timeline")
                result.boundaries.append(
                    dict(
                        evidence_artifact_id=aid,
                        element_id=zone.element_id,
                        upstream_timeline_id=zone.timeline_id,
                        record_ids=ids,
                        start=zone.start,
                        end=zone.end,
                        decision=zone.decision,
                        notes=zone.notes,
                    )
                )
        cores = data["cores"]
        membership = data["membership"]
        assigned = membership.loc[
            membership.target.eq("sequence_core") & membership.evaluation_mask
        ]
        if assigned.frame_id.duplicated().any() or not set(
            assigned.target_label
        ) <= set(cores.element_id):
            raise ValueError("Invalid stored core membership")
        by_id = occurrence.set_index("frame_id")
        for row in assigned.itertuples():
            if (
                row.frame_id not in by_id.index
                or by_id.loc[row.frame_id, "content_id"] != row.content_id
            ):
                raise ValueError("Unknown core content/record reference")
        for core in cores.sort_values("element_id").to_dict("records"):
            selected = assigned.loc[assigned.target_label.eq(core["element_id"])]
            result.groups.append(
                dict(
                    evidence_artifact_id=aid,
                    linkage_group_id=core["element_id"],
                    kind="candidate_core",
                    evidence_sources=[aid],
                    upstream_metadata=core,
                    review_state=core.get("decision"),
                )
            )
            for content, rows in selected.groupby("content_id", sort=True):
                result.memberships.append(
                    dict(
                        evidence_artifact_id=aid,
                        linkage_group_id=core["element_id"],
                        content_id=content,
                        record_ids=sorted(rows.frame_id),
                        role="core_member",
                        upstream_element_ids=[core["element_id"]],
                    )
                )
        return
    components, membership, elements = (
        data["diagnostic_components"],
        data["core_content_membership"],
        data["structure_elements"],
    )
    if (
        not components.core_id.is_unique
        or not components.diagnostic_only.eq(True).all()
        or not components.dependency_confirmed.eq(False).all()
        or membership.duplicated(["core_id", "content_id"]).any()
        or not elements.element_id.is_unique
        or set(components.core_id) != set(elements.element_id)
        or not set(membership.core_id) <= set(elements.element_id)
        or not set(membership.content_id) <= set(occurrence.content_id)
    ):
        raise ValueError("Invalid stored diagnostic component memberships")
    structure_ref = meta["identity"]["sources"]["structure"]
    matches = [
        p
        for p in directories
        if read_json(p / "metadata.json").get("artifact_id")
        == structure_ref["artifact_id"]
    ]
    if (
        len(matches) != 1
        or sha256(matches[0] / "metadata.json") != structure_ref["metadata_sha256"]
    ):
        raise ValueError(
            "Recurrence requires its exact existing structure artifact under evidence roots"
        )
    structure_meta = inspect(matches[0])
    if (
        structure_meta["artifact_kind"] != "sequence_structure_review_v1"
        or structure_meta["identity"]["sources"]["input"]
        != meta["identity"]["sources"]["input"]
    ):
        raise ValueError("Recurrence and structure source identities differ")
    stored_membership = tables(matches[0])["membership"]
    joined = membership.merge(components, on="core_id", validate="many_to_one")
    for gid, group in components.groupby("diagnostic_component_id", sort=True):
        result.groups.append(
            dict(
                evidence_artifact_id=aid,
                linkage_group_id=gid,
                kind="diagnostic_component",
                evidence_sources=[aid, structure_ref["artifact_id"]],
                review_state=None,
                upstream_metadata={
                    "core_ids": sorted(group.core_id),
                    "diagnostic_only": True,
                    "dependency_confirmed": False,
                },
            )
        )
        members = joined.loc[joined.diagnostic_component_id.eq(gid)]
        for content, rows in members.groupby("content_id", sort=True):
            # Join existing occurrence memberships, never assign an interval's
            # interior or drag other occurrences of a duplicate into this core.
            selected_records = stored_membership.loc[
                stored_membership.content_id.eq(content)
                & stored_membership.target_label.isin(rows.core_id)
                & stored_membership.evaluation_mask
            ]
            record_ids = sorted(set(selected_records.frame_id))
            if not record_ids:
                raise ValueError(
                    "Component content lacks a stored core occurrence membership"
                )
            result.memberships.append(
                dict(
                    evidence_artifact_id=aid,
                    linkage_group_id=gid,
                    content_id=content,
                    record_ids=record_ids,
                    role="core_member",
                    upstream_element_ids=sorted(rows.core_id),
                )
            )


def load_candidates(
    roots,
    manifest_path,
    manifest,
    sources,
    *,
    sequence_root=None,
    review_source_map=None,
):
    """Discover supported publications; unsupported completed evidence fails closed.

    Different corpora require explicit adapters, never fuzzy filename matching.
    Missing roots mean unavailable evidence and are recorded as limitations.
    """
    from flir_pipeline.linkage.base import ARTIFACT_KIND
    from flir_pipeline.linkage.review_model import KIND, MANUAL_COLUMNS
    from flir_pipeline.linkage.review_sources import load_candidate_context
    from flir_pipeline.sequences.experiments.artifacts import KINDS

    result = CandidateEvidence()
    directories = sorted(
        {
            p.parent.resolve()
            for root in roots
            for p in root.rglob("metadata.json")
            if not any(part.endswith(".partial") for part in p.relative_to(root).parts)
        }
    )
    manifest_digest = sources.watch(manifest_path)
    seen = set()
    for directory in directories:
        sources.watch(directory / "metadata.json")
        meta = read_json(directory / "metadata.json")
        kind = meta.get("artifact_kind")
        if kind in KINDS and kind not in {
            "sequence_recurrence_v1",
            "sequence_structure_review_v1",
        }:
            continue  # Fits/metrics/suites are not candidate membership sources.
        aid = meta.get("artifact_id")
        if not aid or aid in seen:
            raise ValueError("Missing/duplicate linkage artifact identity")
        seen.add(aid)
        if kind in {"sequence_recurrence_v1", "sequence_structure_review_v1"}:
            experimental_candidates(
                directory, manifest, manifest_digest, sources, result, directories
            )
        elif kind == ARTIFACT_KIND:
            if sequence_root is None:
                raise ValueError(
                    "Candidate linkage needs --sequence-root to resolve stored video occurrences"
                )
            matches = [
                p.parent
                for p in sequence_root.rglob("metadata.json")
                if read_json(p).get("artifact_id") == meta["sequence_set_id"]
            ]
            if len(matches) != 1:
                raise ValueError("Cannot uniquely resolve linkage sequence_set_id")
            seq = matches[0]
            sources.artifact(directory, meta, aid)
            seq_meta = read_json(seq / "metadata.json")
            sources.artifact(seq, seq_meta, seq_meta["artifact_id"])
            _, video, candidates, _, _ = load_candidate_context(
                directory, manifest_path, seq
            )
            result.video_records.append(video)
            for row in candidates.sort_values("candidate_id").to_dict("records"):
                gid = row["candidate_id"]
                result.groups.append(
                    dict(
                        evidence_artifact_id=aid,
                        linkage_group_id=gid,
                        kind="candidate_pair",
                        evidence_sources=[aid, seq_meta["artifact_id"]],
                        upstream_metadata=row,
                        review_state=None,
                    )
                )
                pair_members = {}
                for column, role, records in (
                    ("labeled_content_id", "query", manifest),
                    ("video_content_id", "candidate", video),
                ):
                    member = dict(
                        evidence_artifact_id=aid,
                        linkage_group_id=gid,
                        content_id=row[column],
                        record_ids=sorted(
                            records.loc[records.content_id.eq(row[column]), "frame_id"]
                        ),
                        role=role,
                        upstream_element_ids=[],
                    )
                    if row[column] in pair_members:
                        member["record_ids"] = sorted(
                            set(member["record_ids"])
                            | set(pair_members[row[column]]["record_ids"])
                        )
                        member["role"] = "query_and_candidate"
                    pair_members[row[column]] = member
                result.memberships.extend(pair_members.values())
        elif kind == KIND:
            if review_source_map is None:
                raise ValueError(
                    "Manual review evidence requires --review-source-map for source verification"
                )
            from flir_pipeline.linkage.review_aggregate_storage import load_source_map
            from flir_pipeline.linkage.review_storage import (
                inspect_snapshot,
                verify_review,
            )

            sources.watch(review_source_map)
            bindings = load_source_map(review_source_map)
            sources.artifact(directory, meta, aid)
            checked, rows, _ = inspect_snapshot(directory)
            paths = bindings.get(checked["calibration_id"])
            if paths is None:
                raise ValueError("Unresolved manual review calibration")
            if sources.watch(paths.labeled_manifest) != manifest_digest:
                raise ValueError(
                    "Manual review belongs to a different labeled manifest"
                )
            from dataclasses import asdict

            for path in asdict(paths).values():
                if isinstance(path, Path):
                    if path.is_dir():
                        # The existing reviewer supports both checksum-bearing
                        # publications and the external confirmed-manual v1.
                        # Snapshot either transport, then let its loader verify.
                        for source_file in path.rglob("*"):
                            if source_file.is_file():
                                sources.watch(source_file)
                    else:
                        sources.watch(path)
            quality = verify_review(directory, paths)
            if not quality.get("quality_valid") or not quality.get("source_bound"):
                raise ValueError(f"Manual review verification failed: {quality}")
            for row in rows.sort_values("review_query_id").to_dict("records"):
                result.reviews.append(
                    dict(
                        evidence_artifact_id=aid,
                        review_query_id=row["review_query_id"],
                        content_id=row["labeled_content_id"],
                        proposed_visual_dependency_group_id=row[
                            "proposed_visual_dependency_group_id"
                        ],
                        decision={k: row[k] for k in MANUAL_COLUMNS},
                    )
                )
        else:
            raise ValueError(
                f"Unsupported evidence kind {kind!r}; select explicit supported publication roots"
            )
    if not result.groups:
        result.limitations.append(
            "No candidate linkage/component memberships were supplied; empty tables are unavailable evidence, not absence of relationships."
        )
    sources.unchanged()
    return result
