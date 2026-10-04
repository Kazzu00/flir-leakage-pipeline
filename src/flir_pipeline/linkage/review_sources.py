"""Source binding for manual review without recomputing or promoting candidates."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from flir_pipeline.data.identity import dataset_id_from_manifest
from flir_pipeline.linkage.base import ARTIFACT_KIND
from flir_pipeline.linkage.base import SEMANTICS as LINKAGE_SEMANTICS
from flir_pipeline.linkage.review_model import (
    KEYS,
    canonical_json,
    frame_records,
    read_csv,
    require_strings,
    validate_sample,
)
from flir_pipeline.linkage.storage import OCCURRENCE_COLUMNS, OUTPUT_FILES
from flir_pipeline.sequences.construction import check_partition_invariants
from flir_pipeline.sequences.storage import SEQUENCE_KIND
from flir_pipeline.similarity.storage import file_sha256, read_json, stable_id


@dataclass(frozen=True)
class ReviewPaths:
    calibration_sample: Path
    linkage: Path
    labeled_manifest: Path
    sequence_set: Path
    visual_dependencies: Path
    membership_table: str = "membership.csv"


def artifact_files(directory, meta):
    """Verify all declared files safely, without following metadata outside root."""
    from flir_pipeline.data.local_images import declared_file

    if (
        not isinstance(meta.get("output_checksums"), dict)
        or not meta["output_checksums"]
    ):
        raise ValueError("Source publication needs output checksums")
    checks = {"metadata.json": file_sha256(directory / "metadata.json")}
    for name, expected in meta["output_checksums"].items():
        actual = file_sha256(declared_file(directory, name))
        if actual != expected:
            raise ValueError(f"Source output checksum mismatch: {name}")
        checks[name] = actual
    return checks


def load_linkage_context(paths):
    """Verify publication/lineage bindings; full numerical QA remains linkage verify."""
    labeled, sequences, candidates, occurrences, signature = load_candidate_context(
        paths.linkage, paths.labeled_manifest, paths.sequence_set
    )
    signature["calibration_sample_sha256"] = file_sha256(paths.calibration_sample)
    return labeled, sequences, candidates, occurrences, signature


def load_candidate_context(linkage, labeled_manifest, sequence_set):
    """Read existing candidate evidence without requiring a manual calibration.

    Shared by review and presentation exports. This checks stored identities,
    checksums and occurrence bindings, without recalculating encoder scores.
    """
    from types import SimpleNamespace

    paths = SimpleNamespace(
        linkage=linkage, labeled_manifest=labeled_manifest, sequence_set=sequence_set
    )
    linkage_meta = read_json(paths.linkage / "metadata.json")
    if (
        linkage_meta.get("artifact_kind") != ARTIFACT_KIND
        or linkage_meta.get("artifact_version") != 1
        or linkage_meta.get("ground_truth") is not False
        or linkage_meta.get("semantics") != LINKAGE_SEMANTICS
        or linkage_meta.get("artifact_id") != stable_id(linkage_meta["identity"])
        or set(linkage_meta["output_checksums"]) != set(OUTPUT_FILES)
    ):
        raise ValueError("Expected an intact candidate-only linkage publication")
    linkage_files = artifact_files(paths.linkage, linkage_meta)
    signature = linkage_meta["identity"]["sources"]
    labeled = pd.read_parquet(paths.labeled_manifest)
    pd.testing.assert_frame_equal(
        labeled.sort_values("frame_id").reset_index(drop=True),
        pd.read_parquet(paths.linkage / "labeled_occurrences.parquet"),
        check_exact=True,
    )
    if (
        file_sha256(paths.labeled_manifest)
        != signature["checksums"]["labeled_manifest"]
        or dataset_id_from_manifest(labeled) != linkage_meta["labeled_dataset_id"]
    ):
        raise ValueError("Labeled manifest differs from the linkage source")
    seq_meta = read_json(paths.sequence_set / "metadata.json")
    if (
        seq_meta["artifact_kind"] != SEQUENCE_KIND
        or seq_meta["artifact_id"] != linkage_meta["sequence_set_id"]
        or seq_meta["artifact_id"] != stable_id(seq_meta["identity"])
    ):
        raise ValueError("Sequence set differs from the linkage source")
    sequence_files = artifact_files(paths.sequence_set, seq_meta)
    if sequence_files != signature["checksums"]["sequence_set"]:
        raise ValueError("Sequence occurrence publication has changed since linkage")
    sequences = pd.read_parquet(paths.sequence_set / "occurrence_assignments.parquet")
    tables = {
        name: pd.read_parquet(paths.sequence_set / f"{name}.parquet")
        for name in (
            "boundaries",
            "sequence_instances",
            "occurrence_assignments",
            "dependency_edges",
            "dependency_support",
            "manual_review",
        )
    }
    check_partition_invariants(sequences, tables, seq_meta["artifact_id"])
    candidates = pd.read_parquet(paths.linkage / "content_candidates.parquet")
    occurrences = pd.read_parquet(paths.linkage / "candidate_occurrences.parquet")
    if (
        candidates.empty
        or not candidates.candidate_id.is_unique
        or candidates.duplicated(["labeled_content_id", "video_content_id"]).any()
        or not set(candidates.labeled_content_id) <= set(labeled.content_id)
        or not set(candidates.video_content_id) <= set(sequences.content_id)
    ):
        raise ValueError("Invalid candidate pair identities")
    expected = sequences.loc[
        sequences.content_id.isin(candidates.video_content_id), list(OCCURRENCE_COLUMNS)
    ].rename(columns={"content_id": "video_content_id", "frame_id": "video_frame_id"})
    expected = expected.sort_values(
        ["video_content_id", "video_id", "sample_index", "video_frame_id"]
    ).reset_index(drop=True)
    pd.testing.assert_frame_equal(occurrences, expected, check_exact=True)
    return (
        labeled,
        sequences,
        candidates,
        occurrences,
        {
            "labeled_manifest_sha256": file_sha256(paths.labeled_manifest),
            "linkage_files": linkage_files,
            "linkage_id": linkage_meta["artifact_id"],
            "sequence_files": sequence_files,
            "sequence_set_id": seq_meta["artifact_id"],
        },
    )


def visual_dependency_binding(paths: ReviewPaths, metadata, sequence_set_id):
    """Keep the real v1 producer contract separate from normalized publications.

    The confirmed-manual v1 producer has no artifact_id or output_checksums.
    Hash its exact external bytes on the consumer side; never inject producer
    declarations or rewrite the frozen source. Other producers retain the
    stricter existing requirement for a declared ID and output checksums.
    """
    from flir_pipeline.data.local_images import declared_file

    if (
        metadata.get("ground_truth") is not False
        or metadata.get("review_status") != "confirmed_manual_review"
        or metadata.get("manual_confirmation_complete") is not True
        or metadata.get("sequence_set_id") != sequence_set_id
        or metadata.get("split_created", False) is not False
        or metadata.get("sequence_instances_merged", False) is not False
    ):
        raise ValueError(
            "Visual dependency membership requires confirmed manual review, ground_truth=false and the bound sequence_set_id; no merged sequences or split"
        )
    if metadata.get("artifact_kind") == "confirmed_manual_visual_dependency_validation":
        # Branch on kind before checking version: an unsupported version must
        # fail here, never fall through to a looser/alternative contract.
        if (
            type(metadata.get("artifact_version")) is not int
            or metadata["artifact_version"] != 1
            or metadata.get("semantic_role")
            != "must_link_constraint_for_leakage_safe_split"
            or metadata.get("sequence_instances_merged") is not False
            or metadata.get("split_created") is not False
            or metadata.get("exact_duplicate_dependencies_preserved") is not True
        ):
            raise ValueError(
                "Invalid confirmed-manual visual dependency v1 semantics/version"
            )
        if paths.membership_table in {"metadata.json", "visual_dependency_groups.csv"}:
            raise ValueError("Select the distinct sequence membership table explicitly")
        checks = {
            name: file_sha256(declared_file(paths.visual_dependencies, name))
            for name in (
                "metadata.json",
                "visual_dependency_groups.csv",
                paths.membership_table,
            )
        }
        identity = {
            "identity_kind": "consumer_bound_visual_dependency_source_v1",
            "artifact_kind": metadata["artifact_kind"],
            "artifact_version": metadata["artifact_version"],
            "sequence_set_id": sequence_set_id,
            "source_checksums": checks,
        }
        return {
            "visual_dependency_files": checks,
            "visual_dependency_consumer_source_identity": identity,
            "visual_dependency_consumer_source_fingerprint": hashlib.sha256(
                canonical_json(identity).encode("utf-8")
            ).hexdigest(),
        }
    for key in ("artifact_kind", "artifact_id"):
        if not isinstance(metadata.get(key), str) or not metadata[key].strip():
            raise ValueError(f"Visual dependency source requires {key}")
    checks = artifact_files(paths.visual_dependencies, metadata)
    if paths.membership_table not in metadata["output_checksums"]:
        raise ValueError(
            "Membership table must be checksum-bound by its source metadata"
        )
    return {
        "visual_dependency_files": checks,
        "visual_dependency_artifact_id": metadata["artifact_id"],
    }


def load_review_sources(paths: ReviewPaths):
    """Consume explicit producer contracts, preserving complete membership rows."""
    from flir_pipeline.data.local_images import declared_file

    labeled, sequences, candidates, occurrences, signature = load_linkage_context(paths)
    metadata = read_json(paths.visual_dependencies / "metadata.json")
    binding = visual_dependency_binding(paths, metadata, signature["sequence_set_id"])
    table_path = declared_file(paths.visual_dependencies, paths.membership_table)
    if table_path.suffix == ".csv":
        membership = read_csv(table_path)
    elif table_path.suffix == ".parquet":
        membership = pd.read_parquet(table_path)
    else:
        raise ValueError("Membership table must be CSV or Parquet")
    require_strings(membership, ["sequence_id", "visual_dependency_group_id"])
    if not membership.sequence_id.is_unique or set(membership.sequence_id) != set(
        sequences.sequence_id
    ):
        raise ValueError(
            "Confirmed membership must cover each sequence exactly once, including singleton groups"
        )
    # Consume the confirmed memberships exactly. Combining visual and exact-copy
    # must-link constraints belongs to a future split protocol, not calibration.
    signature.update(
        {
            **binding,
            "membership_table": paths.membership_table,
        }
    )
    sample = validate_sample(read_csv(paths.calibration_sample))
    return sample, labeled, sequences, candidates, occurrences, membership, signature


def review_evidence(
    sample, labeled, sequences, candidates, occurrences, membership, calibration_id
):
    """Keep the proposed group explicit and every other candidate membership visible."""
    sample = validate_sample(sample)
    if not set(sample.labeled_content_id) <= set(labeled.content_id):
        raise ValueError("Unknown labeled query in calibration sample")
    if not set(sample[KEYS[1]]) <= set(membership.visual_dependency_group_id):
        raise ValueError("Unknown proposed visual dependency group")
    mapping = membership.set_index("sequence_id").visual_dependency_group_id
    sequences = sequences.assign(
        visual_dependency_group_id=sequences.sequence_id.map(mapping)
    )
    occurrences = occurrences.assign(
        visual_dependency_group_id=occurrences.sequence_id.map(mapping)
    )
    candidates = candidates[
        candidates.labeled_content_id.isin(sample.labeled_content_id)
    ].copy()
    occurrences = occurrences[
        occurrences.video_content_id.isin(candidates.video_content_id)
    ].copy()
    rows = []
    for query in sample.to_dict("records"):
        selected = candidates[
            candidates.labeled_content_id.eq(query[KEYS[0]])
        ].sort_values("video_content_id")
        available = occurrences[
            occurrences.video_content_id.isin(selected.video_content_id)
        ]
        if selected.empty or query[KEYS[1]] not in set(
            available.visual_dependency_group_id
        ):
            raise ValueError(
                "Proposed group must occur in the query's actual candidate evidence"
            )
        details = []
        for candidate in frame_records(selected):
            candidate["occurrences"] = frame_records(
                available[
                    available.video_content_id.eq(candidate["video_content_id"])
                ].sort_values(["video_id", "sample_index", "video_frame_id"])
            )
            details.append(candidate)
        rows.append(
            {
                **{name: query[name] for name in (*KEYS, "review_stratum")},
                "review_query_id": stable_id(
                    {
                        "calibration_id": calibration_id,
                        **{name: query[name] for name in KEYS},
                    }
                ),
                "sample_metadata_json": canonical_json(query),
                "candidate_details_json": canonical_json(details),
                "manual_decision": "",
                "manual_notes": "",
                "reviewer": "",
                "decision_source": "",
                "reviewed_at_utc": "",
            }
        )
    return (
        pd.DataFrame(rows),
        sequences,
        {
            "candidates": candidates.reset_index(drop=True),
            "candidate_occurrences": occurrences.reset_index(drop=True),
            "labeled_occurrences": labeled[
                labeled.content_id.isin(sample.labeled_content_id)
            ]
            .sort_values("frame_id")
            .reset_index(drop=True),
            "calibration_sample": sample,
            "visual_dependency_membership": membership.sort_values(
                "sequence_id"
            ).reset_index(drop=True),
        },
    )
