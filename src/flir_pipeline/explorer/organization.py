"""Normalized organization evidence export with staged, rollback-safe publication."""

import hashlib
import json
import shutil
import subprocess
import tempfile
import zipfile
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

from flir_pipeline.data.classes import class_name
from flir_pipeline.data.local_images import relative_posix_path
from flir_pipeline.data.temporal import audit_temporal_lineage
from flir_pipeline.detection.final_report import _publish, clean, write_payload
from flir_pipeline.explorer.discovery import checked_file, sha256
from flir_pipeline.explorer.frames import FrameReader
from flir_pipeline.explorer.organization_contract import (
    SCHEMA_VERSION,
    OrganizationExport,
    schema,
)
from flir_pipeline.explorer.organization_sources import (
    Sources,
    load_candidates,
    selected_runs,
)


def value(row, key):
    result = row.get(key)
    return None if result is None or pd.isna(result) else clean(result)


def table_rows(frame):
    return clean(frame.astype(object).where(frame.notna(), None).to_dict("records"))


def location(row):
    """Presentation namespace only; no source-video identity is inferred."""
    if value(row, "video_id"):
        return json.dumps(
            ["video", row["video_id"]], ensure_ascii=False, separators=(",", ":")
        )
    archive, family = value(row, "source_archive"), value(row, "possible_sequence")
    if archive and family:
        return json.dumps(
            ["filename", archive, family], ensure_ascii=False, separators=(",", ":")
        )
    return None


def occurrence_records(manifest, evidence):
    """Keep occurrences, including conflicting annotations and repeated positions."""
    lineage = audit_temporal_lineage(manifest, max_frame_gap=0).lineage.set_index(
        "frame_id"
    )
    raw = {r["frame_id"]: r for r in table_rows(manifest)}
    labeled = set(raw)
    for frame in evidence.video_records:
        for r in table_rows(frame):
            rid = r["frame_id"]
            if rid in labeled or rid in raw and raw[rid] != r:
                raise ValueError(
                    "Conflicting/duplicate record identity across evidence cohorts"
                )
            raw[rid] = r
    records = []
    for rid, r in sorted(raw.items()):
        is_labeled = rid in labeled
        if not is_labeled and r.get("image_sha256") != r["content_id"]:
            raise ValueError("Video evidence needs authoritative exact content bytes")
        if is_labeled:
            temporal = lineage.loc[rid]
            r = {
                **r,
                "possible_sequence": temporal.possible_sequence,
                "possible_frame_index": value(temporal, "possible_frame_index"),
            }
        member = (
            value(r, "source_member_path") if is_labeled else value(r, "image_path")
        )
        archive = value(r, "source_archive") if is_labeled else None
        if member:
            relative_posix_path(member)
        if archive:
            relative_posix_path(archive)
        classes = None
        if is_labeled and "classes_present" in r and r["classes_present"] is not None:
            classes = sorted(
                {int(c) for c in str(r["classes_present"]).split("|") if c}
            )
        records.append(
            dict(
                record_id=rid,
                content_id=r["content_id"],
                cohort="labeled" if is_labeled else "video_evidence",
                filename=PurePosixPath(member).name if member else None,
                source_archive=archive,
                source_video_id=None if is_labeled else value(r, "video_id"),
                timeline_id=location(r),
                frame_index=value(
                    r, "possible_frame_index" if is_labeled else "sample_index"
                ),
                source_frame_index_estimate=None
                if is_labeled
                else value(r, "source_frame_index_estimate"),
                timestamp_seconds=None if is_labeled else value(r, "timestamp_seconds"),
                temporal_source="filename_heuristic"
                if is_labeled
                else "sampled_video_grid",
                original_split=value(r, "original_split") if is_labeled else None,
                class_ids=classes,
                class_names=[class_name(c) for c in classes]
                if classes is not None
                else None,
                num_objects=value(r, "num_objects") if is_labeled else None,
                label_empty=value(r, "label_empty") if is_labeled else None,
                width=value(r, "width"),
                height=value(r, "height"),
            )
        )
    return records, raw


def consensus(rows, key):
    values = [row[key] for row in rows]
    return values[0] if values and all(v == values[0] for v in values) else None


def content_records(records, raw, clusters):
    by_content = defaultdict(list)
    representatives = {}
    for evidence in clusters.values():
        if evidence.full_artifact is None:
            continue
        for row in evidence.memberships.itertuples():
            representatives.setdefault(row.content_id, row.representative_frame_id)
    for r in records:
        by_content[r["content_id"]].append(r)
    contents = []
    for cid, rows in sorted(by_content.items()):
        representative = representatives.get(cid, rows[0]["record_id"])
        selected = next(r for r in rows if r["record_id"] == representative)
        annotated = [r for r in rows if r["cohort"] == "labeled"]
        # The manifest binds label bytes, not semantic equivalence. Different
        # hashes can reflect formatting/order alone; preserve the disagreement
        # without claiming a bounding-box conflict or choosing one annotation.
        annotation_hashes = {raw[r["record_id"]].get("label_sha256") for r in annotated}
        status = (
            "unavailable"
            if not annotation_hashes
            or None in annotation_hashes
            or "" in annotation_hashes
            else "identical_label_bytes"
            if len(annotation_hashes) == 1
            else "different_label_bytes"
        )
        temporal = consensus(rows, "temporal_source") or "unknown"
        single_timeline = consensus(rows, "timeline_id") is not None
        contents.append(
            dict(
                content_id=cid,
                canonical_filename=selected["filename"],
                representative_record_id=representative,
                source_video_id=consensus(rows, "source_video_id"),
                frame_index=consensus(rows, "frame_index") if single_timeline else None,
                source_frame_index_estimate=consensus(
                    rows, "source_frame_index_estimate"
                )
                if single_timeline
                else None,
                timestamp_seconds=consensus(rows, "timestamp_seconds")
                if single_timeline
                else None,
                temporal_source=temporal,
                record_ids=sorted(r["record_id"] for r in rows),
                class_ids=consensus(annotated, "class_ids")
                if status == "identical_label_bytes"
                else None,
                class_names=consensus(annotated, "class_names")
                if status == "identical_label_bytes"
                else None,
                annotation_consensus=status,
                preview_key=hashlib.sha256(cid.encode()).hexdigest(),
                width=consensus(rows, "width"),
                height=consensus(rows, "height"),
            )
        )
    return contents


def temporal_spans(records):
    grouped = defaultdict(list)
    for r in records:
        if r["timeline_id"]:
            grouped[r["timeline_id"]].append(r)
    result = []
    for tid, rows in sorted(grouped.items()):
        indices = [r["frame_index"] for r in rows if r["frame_index"] is not None]
        result.append(
            dict(
                timeline_id=tid,
                frame_index_min=min(indices) if indices else None,
                frame_index_max=max(indices) if indices else None,
                n_unique_contents=len({r["content_id"] for r in rows}),
            )
        )
    return result


def timelines(records, contents):
    previews = {c["content_id"]: c["preview_key"] for c in contents}
    groups = defaultdict(list)
    for r in records:
        if r["timeline_id"]:
            groups[r["timeline_id"]].append(r)
    result = []
    for tid, rows in sorted(groups.items()):
        points = defaultdict(list)
        for r in rows:
            points[r["frame_index"], r["timestamp_seconds"], r["content_id"]].append(
                r["record_id"]
            )
        namespace = json.loads(tid)
        result.append(
            dict(
                timeline_id=tid,
                source_video_id=rows[0]["source_video_id"],
                source_archive=rows[0]["source_archive"],
                inferred_family=namespace[2] if namespace[0] == "filename" else None,
                temporal_source=rows[0]["temporal_source"],
                points=[
                    dict(
                        frame_index=idx,
                        timestamp_seconds=seconds,
                        content_id=cid,
                        record_ids=sorted(ids),
                        preview_key=previews[cid],
                    )
                    for (idx, seconds, cid), ids in sorted(
                        points.items(),
                        key=lambda p: (p[0][0] is None, p[0][0] or 0, p[0][2]),
                    )
                ],
            )
        )
    return result


def scientific_tables(manifest, selected, clusters, evidence):
    records, raw = occurrence_records(manifest, evidence)
    contents = content_records(records, raw, clusters)
    result = dict(
        records=records,
        contents=contents,
        splits=[],
        split_memberships=[],
        clustering_configurations=[],
        clusters=[],
        cluster_memberships=[],
        linkage_groups=[],
        linkage_memberships=[],
        boundary_zones=[],
        reviews=evidence.reviews,
        timelines=timelines(records, contents),
        media=[],
    )
    for spec, split in selected:
        identity = {
            key: spec[key] for key in ("strategy", "split_seed", "split_space_id")
        }
        partitions = {}
        for partition in ("train", "val", "test"):
            rows = split.records.loc[split.records.new_split.eq(partition)].sort_values(
                "frame_id"
            )
            partitions[partition] = dict(
                n_records=len(rows), n_unique_contents=rows.content_id.nunique()
            )
            result["split_memberships"].extend(
                {
                    **identity,
                    "partition": partition,
                    "record_id": r.frame_id,
                    "content_id": r.content_id,
                }
                for r in rows.itertuples()
            )
        result["splits"].append(
            {
                **identity,
                "artifact_id": split.run.space_id,
                "source_strategy": split.run.strategy,
                "cluster_run_id": split.run.clustering_space_id,
                "partitions": partitions,
                "class_support": spec.get("class_counts"),
            }
        )
    for cid, cluster_evidence in sorted(clusters.items()):
        cluster = cluster_evidence.full_artifact
        meta = cluster.run.metadata if cluster is not None else {}
        memberships = cluster_evidence.memberships
        labels = memberships.cluster_id
        metrics = cluster.metrics if cluster is not None else None
        n_clusters, n_noise = len(set(labels) - {-1}), int(labels.eq(-1).sum())
        probabilities = {}
        diagnostics = {}
        if "membership_probabilities.npy" in meta.get("optional_files", []):
            from flir_pipeline.clustering.storage import quality_checks

            values = np.load(
                checked_file(cluster.run, "membership_probabilities.npy"),
                allow_pickle=False,
            )
            if not quality_checks(labels.to_numpy(), len(labels), values)[
                "quality_valid"
            ]:
                raise ValueError("Invalid stored clustering probabilities")
            probabilities = dict(
                zip(cluster.contents.content_id, values.tolist(), strict=True)
            )
        if "algorithm_diagnostics.npz" in meta.get("optional_files", []):
            with np.load(
                checked_file(cluster.run, "algorithm_diagnostics.npz"),
                allow_pickle=False,
            ) as stored:
                if set(stored.files) != {
                    "reachability",
                    "core_distances",
                    "ordering_rows",
                }:
                    raise ValueError("Unknown stored clustering diagnostics schema")
                reachability, distance, order = (
                    stored[k]
                    for k in ("reachability", "core_distances", "ordering_rows")
                )
            if (
                any(a.shape != (len(labels),) for a in (reachability, distance, order))
                or order.dtype.kind not in "iu"
                or sorted(order.tolist()) != list(range(len(labels)))
                or any(
                    np.isnan(a).any() or (a < 0).any() for a in (reachability, distance)
                )
            ):
                raise ValueError(
                    "Invalid stored clustering distance/ordering diagnostics"
                )
            positions = np.argsort(order)
            for i, content in enumerate(cluster.contents.content_id):
                diagnostics[content] = dict(
                    reachability=None
                    if np.isinf(reachability[i])
                    else float(reachability[i]),
                    reachability_infinite=bool(np.isinf(reachability[i])),
                    core_distance=None if np.isinf(distance[i]) else float(distance[i]),
                    core_distance_infinite=bool(np.isinf(distance[i])),
                    ordering_position=int(positions[i]),
                )
        if metrics is not None and (
            metrics["n_clusters_excluding_noise"] != n_clusters
            or abs(metrics["noise_fraction"] - n_noise / len(labels)) > 1e-12
        ):
            raise ValueError(
                "Stored clustering noise/count statistics differ from labels"
            )
        result["clustering_configurations"].append(
            dict(
                cluster_run_id=cid,
                source_kind=cluster_evidence.source_kind,
                full_clustering_artifact_available=cluster is not None,
                membership_consistency_verified=cluster is None,
                source_split_ids=sorted(cluster_evidence.source_membership_checksums),
                source_membership_checksums=dict(
                    sorted(cluster_evidence.source_membership_checksums.items())
                ),
                dataset_id=cluster_evidence.dataset_id,
                feature_space_id=meta.get("feature_space_id"),
                configuration_id=meta.get("configuration_id"),
                model_id=meta.get("model_id"),
                strategy_labels=sorted(
                    {
                        s["strategy"]
                        for s, _ in selected
                        if s["clustering_space_id"] == cid
                    }
                ),
                representation=cluster.run.representation
                if cluster is not None
                else None,
                reduction_space_id=meta.get("reduction_space_id"),
                reduction_seed=meta.get("reduction_seed"),
                extractor=cluster.run.encoder if cluster is not None else None,
                algorithm=cluster.run.algorithm if cluster is not None else None,
                parameters=meta.get("config"),
                effective_parameters=meta.get("effective_parameters"),
                n_clusters=n_clusters,
                n_noise=n_noise,
                noise_fraction=metrics["noise_fraction"]
                if metrics is not None
                else n_noise / len(labels),
            )
        )
        for label, members in memberships.groupby("cluster_id", sort=True):
            ids = set(members.content_id)
            member_records = [r for r in records if r["content_id"] in ids]
            videos = defaultdict(set)
            for r in member_records:
                if r["source_video_id"]:
                    videos[r["source_video_id"]].add(r["content_id"])
            result["clusters"].append(
                dict(
                    cluster_run_id=cid,
                    cluster_id=int(label),
                    size_unique_contents=len(ids),
                    is_noise=label == -1,
                    source_video_distribution={
                        k: len(v) for k, v in sorted(videos.items())
                    },
                    temporal_spans=temporal_spans(member_records),
                )
            )
            for content in sorted(ids):
                result["cluster_memberships"].append(
                    dict(
                        cluster_run_id=cid,
                        cluster_id=int(label),
                        content_id=content,
                        is_noise=label == -1,
                        probability=probabilities.get(content),
                        **diagnostics.get(content, {}),
                    )
                )
    by_record = {r["record_id"]: r for r in records}
    for m in sorted(
        evidence.memberships,
        key=lambda r: (
            r["evidence_artifact_id"],
            r["linkage_group_id"],
            r["content_id"],
        ),
    ):
        if (
            not m["record_ids"]
            or not set(m["record_ids"]) <= set(by_record)
            or any(
                by_record[r]["content_id"] != m["content_id"] for r in m["record_ids"]
            )
        ):
            raise ValueError("Broken candidate content/record reference")
        rows = [by_record[r] for r in m["record_ids"]]
        result["linkage_memberships"].append(
            {
                **m,
                "source_video_ids": sorted(
                    {r["source_video_id"] for r in rows if r["source_video_id"]}
                ),
                "frame_indices": sorted(
                    {r["frame_index"] for r in rows if r["frame_index"] is not None}
                ),
            }
        )
    for g in sorted(
        evidence.groups,
        key=lambda r: (r["evidence_artifact_id"], r["linkage_group_id"]),
    ):
        members = [
            m
            for m in result["linkage_memberships"]
            if (m["evidence_artifact_id"], m["linkage_group_id"])
            == (g["evidence_artifact_id"], g["linkage_group_id"])
        ]
        rows = [
            by_record[rid]
            for rid in sorted({r for m in members for r in m["record_ids"]})
        ]
        result["linkage_groups"].append(
            {
                **g,
                "member_count": len(members),
                "source_video_ids": sorted(
                    {r["source_video_id"] for r in rows if r["source_video_id"]}
                ),
                "temporal_spans": temporal_spans(rows),
            }
        )
    for b in sorted(
        evidence.boundaries, key=lambda r: (r["evidence_artifact_id"], r["element_id"])
    ):
        rows = [by_record[r] for r in b["record_ids"]]
        result["boundary_zones"].append(
            {k: v for k, v in b.items() if k != "record_ids"}
            | {
                "timeline_ids": sorted(
                    {r["timeline_id"] for r in rows if r["timeline_id"]}
                )
            }
        )
    return result, raw


def preview_media(
    contents, raw, manifest, output, *, include, data_root, video_images_root
):
    """One checksum-verified, compact JPEG per content; absence never drops rows."""
    from contextlib import ExitStack

    from PIL import Image, UnidentifiedImageError

    from flir_pipeline.features.image_source import ImageSource

    result = []
    with ExitStack() as stack:
        reader = (
            stack.enter_context(FrameReader(data_root, manifest))
            if include and data_root
            else None
        )
        video = (
            stack.enter_context(ImageSource(None, video_images_root))
            if include and video_images_root and video_images_root.is_dir()
            else None
        )
        for c in contents:
            row = dict(
                preview_key=c["preview_key"],
                content_id=c["content_id"],
                relative_path=None,
                width=None,
                height=None,
                checksum=None,
                media_available=False,
                unavailable_reason="source_unavailable" if include else "not_requested",
            )
            if include:
                r = raw[c["representative_record_id"]]
                try:
                    if reader and c["representative_record_id"] in reader.rows.index:
                        image = reader.image(c["representative_record_id"], width=256)
                    elif video and r.get("image_path"):
                        image = video.decode(
                            r["image_path"], r["image_sha256"]
                        ).convert("RGB")
                        image.thumbnail((256, 256), Image.Resampling.LANCZOS)
                    else:
                        result.append(row)
                        continue
                except (FileNotFoundError, KeyError):
                    result.append(row)
                    continue
                except (
                    ValueError,
                    OSError,
                    zipfile.BadZipFile,
                    UnidentifiedImageError,
                ):
                    row["unavailable_reason"] = "source_invalid"
                    result.append(row)
                    continue
                # Destination failures are fatal; only unavailable source media
                # may degrade. Staging ensures no old scientific export is lost.
                path = output / f"{c['preview_key']}.jpg"
                image.save(path, format="JPEG", quality=82, optimize=False)
                row.update(
                    relative_path=path.name,
                    width=image.width,
                    height=image.height,
                    checksum=sha256(path),
                    media_available=True,
                    unavailable_reason=None,
                )
            result.append(row)
    return result


def _overlap(a, b):
    return a == b or a in b.parents or b in a.parents


def destinations(output, media_output, protected, include):
    outputs = [output.resolve(), media_output.resolve()]
    if _overlap(*outputs) or any(
        _overlap(o, p.resolve()) for o in outputs for p in protected
    ):
        raise ValueError(
            "Export destinations must be disjoint from sources and each other"
        )
    repo = Path(__file__).resolve().parents[3]
    if media_output.resolve().is_relative_to(
        repo
    ) and not media_output.resolve().is_relative_to(repo / "artifacts"):
        raise ValueError(
            "Preview media inside the repository must remain under ignored artifacts/"
        )
    allowed = {f"{k}.json" for k in OrganizationExport.model_fields} | {
        f"schema/{SCHEMA_VERSION}.schema.json"
    }
    for directory, names in ((output, allowed), (media_output, None)):
        if directory == media_output and not include:
            continue
        if (
            directory.is_symlink()
            or directory.exists()
            and not directory.is_dir()
            or any(p.is_symlink() for p in directory.rglob("*"))
        ):
            raise ValueError("Export destination must be a directory without symlinks")
        existing = {
            p.relative_to(directory).as_posix()
            for p in directory.rglob("*")
            if p.is_file()
        }
        if names is not None and existing - names:
            raise ValueError("Export destination contains unrelated files")
        if names is None and existing:
            receipt = directory / "media-receipt.json"
            if not receipt.is_file():
                raise ValueError("Existing media lacks an ownership receipt")
            saved = json.loads(receipt.read_text(encoding="utf-8"))
            if (
                set(saved) != {"schema_version", "files"}
                or saved["schema_version"] != SCHEMA_VERSION
                or existing != {*saved["files"], "media-receipt.json"}
            ):
                raise ValueError("Media destination contains unrelated files")
            for name, digest in saved["files"].items():
                relative_posix_path(name)
                if sha256(directory / name) != digest:
                    raise ValueError(
                        "Existing preview changed; preserve for inspection"
                    )


def git_provenance():
    try:
        repo = Path(__file__).resolve().parents[3]
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repo,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def export_organization(
    *,
    manifest_path=Path("data/manifests/flir_canonical_candidate_v1.parquet"),
    plan_directory=Path("artifacts/detection/protocol"),
    split_root=Path("artifacts/splitting/runs"),
    clustering_root=Path("artifacts/clustering"),
    linkage_root=Path("artifacts/linkage"),
    evidence_roots=(),
    sequence_root=None,
    review_source_map=None,
    output=Path("exports/frontend/organization"),
    media_output=Path("artifacts/frontend/organization-media"),
    include_previews=False,
    data_root=None,
    video_images_root=None,
):
    """Export only existing scientific evidence. Callers must serialize writers."""
    protected = [
        manifest_path,
        plan_directory,
        split_root,
        clustering_root,
        linkage_root,
        *evidence_roots,
        *(
            p
            for p in (sequence_root, review_source_map, data_root, video_images_root)
            if p is not None
        ),
    ]
    destinations(output, media_output, protected, include_previews)
    sources = Sources()
    manifest, plan, splits, clusters = selected_runs(
        plan_directory, manifest_path, split_root, clustering_root, sources
    )
    evidence = load_candidates(
        [linkage_root, *evidence_roots],
        manifest_path,
        manifest,
        sources,
        sequence_root=sequence_root,
        review_source_map=review_source_map,
    )
    # Review source maps can add protected files outside the explicit CLI roots.
    destinations(output, media_output, [*protected, *sources.files], include_previews)
    tables, raw = scientific_tables(manifest, splits, clusters, evidence)
    stage = media_stage = None
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(
            tempfile.mkdtemp(prefix=f".{output.name}-stage-", dir=output.parent)
        )
        if include_previews:
            media_output.parent.mkdir(parents=True, exist_ok=True)
            media_stage = Path(
                tempfile.mkdtemp(
                    prefix=f".{media_output.name}-stage-", dir=media_output.parent
                )
            )
        tables["media"] = preview_media(
            tables["contents"],
            raw,
            manifest,
            media_stage,
            include=include_previews,
            data_root=data_root,
            video_images_root=video_images_root,
        )
        commit, dirty = git_provenance()
        header = dict(
            generated_at=datetime.now(UTC).isoformat(),
            source_commit=commit,
            source_dirty=dirty,
            dataset_id=plan["identity"]["splits"][0]["dataset_id"],
            manifest_sha256=sources.files[manifest_path.resolve()],
            plan_id=plan["plan_id"],
            plan_sha256=sources.files[(plan_directory / "plan.json").resolve()],
            runtime_freeze_sha256=sources.files[
                (plan_directory / "runtime_freeze.json").resolve()
            ],
            record_count=len(tables["records"]),
            unique_content_count=len(tables["contents"]),
            labeled_record_count=len(manifest),
            labeled_unique_content_count=manifest.content_id.nunique(),
            source_video_count=len(
                {
                    r["source_video_id"]
                    for r in tables["records"]
                    if r["source_video_id"]
                }
            ),
            timeline_count=len(tables["timelines"]),
            split_count=len(splits),
            strategy_count=len({s["strategy"] for s, _ in splits}),
            clustering_configuration_ids=sorted(clusters),
            evidence_sources=sorted(
                sources.receipts, key=lambda r: (r["artifact_kind"], r["artifact_id"])
            ),
            limitations=[
                "Stored identities, checksums and relationships verified; no scientific metrics or detector outcomes recomputed.",
                "This export does not verify completion of detector training; selection comes from the supplied frozen plan.",
                "Filename families are heuristic timelines, not authoritative source-video identities. Index gaps are not seconds.",
                "Cluster -1 denotes unassigned noise; it is not a sequence or an indivisible cluster.",
                "Manual review is external evidence, not ground truth. Candidate components are diagnostic only.",
                "Missing previews do not affect scientific memberships. Media stays local and requires separate synchronization.",
                "Counts and ranges summarize stored memberships only; no confidence scores or new boundaries are generated.",
                *evidence.limitations,
                *(
                    [
                        "The original clustering publication for one or more selected split identities is unavailable. "
                        "Exact content-to-cluster membership remains preserved in checksum-verified frozen split artifacts "
                        "and was verified identical across all selected split seeds. Original clustering diagnostics "
                        "and unavailable configuration fields were not reconstructed."
                    ]
                    if any(c.full_artifact is None for c in clusters.values())
                    else []
                ),
            ],
            terminology={
                "record": "historical or source-video occurrence; record_id preserves frame_id",
                "content": "unique exact visual bytes; duplicates retain all record IDs",
                "cluster": "algorithmic visual grouping, never sequence identity",
                "full_clustering_artifact": "the verified original clustering publication is available",
                "frozen_split_membership": "the original clustering publication is unavailable; exact labels consumed by split construction remain preserved and cross-seed verified in immutable selected split artifacts",
                "candidate_pair": "existing top-k candidate ID; not a newly constructed group/component",
                "candidate_core": "stored diagnostic core membership; never a confirmed sequence",
                "diagnostic_component": "existing component-to-core-to-content evidence; not ground truth",
                "timeline_id": "presentation namespace for an existing video ID or archive/filename family; not a scientific sequence ID",
                "temporal_span": "min/max observed indices within one timeline; no implied continuous coverage",
                "source_frame_index_estimate": "upstream estimate, never promoted to an exact decoder index",
                "media.relative_path": "relative to the separately synchronized organization-media root",
            },
            file_sha256={},
        )
        bundle = OrganizationExport.model_validate(
            clean({"manifest": header, **tables})
        )
        payload = bundle.model_dump(mode="json")
        for name, data in payload.items():
            if name != "manifest":
                write_payload(stage / f"{name}.json", data)
        (stage / "schema").mkdir()
        write_payload(stage / "schema" / f"{SCHEMA_VERSION}.schema.json", schema())
        payload["manifest"]["file_sha256"] = {
            p.relative_to(stage).as_posix(): sha256(p)
            for p in sorted(stage.rglob("*"))
            if p.is_file()
        }
        write_payload(stage / "manifest.json", payload["manifest"])
        # Validate serialized files too, so the schema and on-disk data describe
        # exactly the same bundle consumed later by the frontend.
        OrganizationExport.model_validate(
            {
                k: json.loads((stage / f"{k}.json").read_text(encoding="utf-8"))
                for k in OrganizationExport.model_fields
            }
        )
        publications = [(stage, output)]
        if include_previews:
            write_payload(
                media_stage / "media-receipt.json",
                dict(
                    schema_version=SCHEMA_VERSION,
                    files={
                        r["relative_path"]: r["checksum"]
                        for r in payload["media"]
                        if r["media_available"]
                    },
                ),
            )
            publications.insert(0, (media_stage, media_output))
        sources.unchanged()
        _publish(publications)
        return payload["manifest"]
    finally:
        # Only fresh, resolved sibling staging trees owned by this invocation.
        for path, destination in ((stage, output), (media_stage, media_output)):
            if (
                path is not None
                and path.exists()
                and path.resolve().parent == destination.resolve().parent
                and path.name.startswith(f".{destination.name}-stage-")
            ):
                shutil.rmtree(path)
