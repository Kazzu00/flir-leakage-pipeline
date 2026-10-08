"""Audit external visual variants without asserting their native video alignment."""

from __future__ import annotations

import json
import math
import re
import zipfile
from dataclasses import asdict
from pathlib import Path, PurePosixPath

import pandas as pd

from flir_pipeline.data.identity import dataset_id_from_manifest
from flir_pipeline.data.local_images import declared_file
from flir_pipeline.data.video_variant_contract import (
    MANIFEST_VERSION,
    IngestionConfig,
    digest_document,
    scientific_frame_id,
)
from flir_pipeline.data.zip_image_collection import (
    check_archive,
    image_properties,
    read_entry,
)
from flir_pipeline.utils.hashing import sha256_file

TABLE_DTYPES = {
    "archives": {
        "archive_key": "string",
        "path": "string",
        "sha256": "string",
        "size_bytes": "Int64",
        "declared_entries": "Int64",
        "inventoried_entries": "Int64",
        "uncompressed_bytes": "Int64",
        "archive_error": "string",
    },
    "entries": {
        "entry_id": "string",
        "archive_key": "string",
        "source_archive": "string",
        "member_ordinal": "Int64",
        "source_member_path": "string",
        "size_bytes": "Int64",
        "compressed_bytes": "Int64",
        "crc32": "Int64",
        "is_directory": "boolean",
        "is_image": "boolean",
        "image_sha256": "string",
        "read_error": "string",
        "width": "Int64",
        "height": "Int64",
        "channels": "Int64",
        "image_mode": "string",
        "image_format": "string",
        "image_decode_valid": "boolean",
        "image_error": "string",
    },
    "video_sources": {
        "video_key": "string",
        "source_video_id": "string",
        "path": "string",
        "sha256": "string",
        "size_bytes": "Int64",
        "declared_metadata_json": "string",
        "probe_status": "string",
        "probe_metadata_json": "string",
        "probe_error": "string",
    },
    "occurrences": {
        "entry_id": "string",
        "collection_id": "string",
        "series_id": "string",
        "possible_series_ids_json": "string",
        "observed_index": "Int64",
        "observation_id": "string",
        "frame_id": "string",
        "identity_status": "string",
        "content_id": "string",
        "image_sha256": "string",
        "source_archive": "string",
        "source_member_path": "string",
        "member_ordinal": "Int64",
        "width": "Int64",
        "height": "Int64",
        "channels": "Int64",
        "image_mode": "string",
        "image_format": "string",
        "image_decode_valid": "boolean",
        "image_error": "string",
        "read_error": "string",
    },
    "index_issues": {
        "series_id": "string",
        "issue_type": "string",
        "index_start": "Int64",
        "index_stop": "Int64",
        "index_step": "Int64",
        "entry_ids_json": "string",
        "description": "string",
    },
    "alignment_candidates": {
        "entry_id": "string",
        "frame_id": "string",
        "observed_index": "Int64",
        "candidate_video_key": "string",
        "candidate_source_video_id": "string",
        "candidate_timestamp_seconds": "Float64",
        "timestamp_rule": "string",
        "alignment_status": "string",
        "verification_scope": "string",
        "verified_native_frame_index": "Int64",
        "verified_timestamp_seconds": "Float64",
    },
    "observations": {
        "observation_id": "string",
        "series_id": "string",
        "video_key": "string",
        "description": "string",
        "observed_index": "Int64",
        "reported_timestamp_seconds": "Float64",
        "evidence_path": "string",
        "evidence_sha256": "string",
        "reviewer": "string",
        "reviewed_at": "string",
        "evidence_status": "string",
        "verification_scope": "string",
    },
}
MANIFEST_COLUMNS = {
    "manifest_version": "string",
    "frame_id": "string",
    "content_id": "string",
    "image_sha256": "string",
    "collection_id": "string",
    "series_id": "string",
    "observed_index": "Int64",
    "width": "Int64",
    "height": "Int64",
    "channels": "Int64",
    "image_mode": "string",
    "image_format": "string",
    "image_decode_valid": "boolean",
    "image_error": "string",
    "label_exists": "boolean",
    "label_sha256": "string",
    "original_split": "string",
}


def json_text(value) -> str:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False
    )


def table(name: str, rows: list[dict]) -> pd.DataFrame:
    schema = TABLE_DTYPES[name]
    return pd.DataFrame(rows, columns=list(schema)).astype(schema)


def records(frame: pd.DataFrame) -> list[dict]:
    return [
        {key: None if pd.isna(value) else value for key, value in row.items()}
        for row in frame.to_dict("records")
    ]


def input_files(config: IngestionConfig) -> dict:
    """Only explicitly declared files are inspected; no recursive discovery."""
    result = {f"archive:{a.archive_key}": a for a in config.archives}
    result.update({f"video:{v.video_key}": v for v in config.videos})
    if config.auxiliary_inventory is not None:
        result["auxiliary_inventory"] = config.auxiliary_inventory
    for observation in config.observations:
        if observation.evidence_path is not None:
            result[f"evidence:{observation.observation_id}"] = observation.evidence_path
    return result


def snapshot_sources(config: IngestionConfig, root: Path) -> dict:
    sources = {}
    for key, declaration in sorted(input_files(config).items()):
        relative = declaration if isinstance(declaration, str) else declaration.path
        path = declared_file(root, relative)
        digest = sha256_file(path)
        expected = None if isinstance(declaration, str) else declaration.expected_sha256
        if expected is not None and digest != expected:
            raise ValueError(f"Input checksum mismatch: {key}")
        sources[key] = {
            "path": relative,
            "sha256": digest,
            "size_bytes": path.stat().st_size,
        }
    return sources


def observe_sources(
    config: IngestionConfig, root: Path, *, ffprobe_bin: str | None = None
):
    """Read one bounded image at a time and retain every physical ZIP entry."""
    sources = snapshot_sources(config, root)
    archive_rows, entry_rows, video_rows = [], [], []
    for declaration in sorted(config.archives, key=lambda item: item.archive_key):
        bound = sources[f"archive:{declaration.archive_key}"]
        row = dict(
            archive_key=declaration.archive_key,
            **bound,
            declared_entries=None,
            inventoried_entries=0,
            uncompressed_bytes=None,
            archive_error=None,
        )
        try:
            with zipfile.ZipFile(declared_file(root, declaration.path), "r") as archive:
                infos = archive.infolist()
                row["declared_entries"] = len(infos)
                total = sum(info.file_size for info in infos)
                # Malformed ZIP64 declarations can exceed Parquet's int64 range.
                # Their exact total survives in the limit error, without overflow.
                row["uncompressed_bytes"] = total if total <= 2**63 - 1 else None
                check_archive(infos, config.limits)
                for ordinal, info in enumerate(infos):
                    is_directory = info.orig_filename.endswith("/")
                    is_image = (
                        not is_directory
                        and PurePosixPath(info.orig_filename).suffix.lower()
                        in config.image_extensions
                    )
                    entry = dict(
                        entry_id=digest_document(
                            [
                                "zip-entry-v1",
                                declaration.archive_key,
                                bound["sha256"],
                                ordinal,
                                info.orig_filename,
                            ]
                        ),
                        archive_key=declaration.archive_key,
                        source_archive=declaration.path,
                        member_ordinal=ordinal,
                        source_member_path=info.orig_filename,
                        size_bytes=info.file_size,
                        compressed_bytes=info.compress_size,
                        crc32=info.CRC,
                        is_directory=is_directory,
                        is_image=is_image,
                        image_sha256=None,
                        read_error=None,
                        image_decode_valid=False,
                    )
                    if is_image:
                        try:
                            content, digest = read_entry(archive, info, config.limits)
                            entry["image_sha256"] = digest
                            entry.update(
                                image_properties(
                                    content, config.limits, config.expected_image_format
                                )
                            )
                            del content
                        except (
                            ValueError,
                            OSError,
                            RuntimeError,
                            NotImplementedError,
                            zipfile.BadZipFile,
                        ) as error:
                            entry["read_error"] = str(error)
                    entry_rows.append(entry)
                    row["inventoried_entries"] += 1
        except (
            ValueError,
            OSError,
            RuntimeError,
            NotImplementedError,
            zipfile.BadZipFile,
        ) as error:
            row["archive_error"] = str(error)
        archive_rows.append(row)
    for video in sorted(config.videos, key=lambda item: item.video_key):
        bound = sources[f"video:{video.video_key}"]
        row = dict(
            video_key=video.video_key,
            **bound,
            source_video_id=digest_document(["source-video-bytes-v1", bound["sha256"]]),
            declared_metadata_json=json_text(video.declared_metadata),
            probe_status="not_requested",
            probe_metadata_json=None,
            probe_error=None,
        )
        if ffprobe_bin is not None:
            from flir_pipeline.data.video_frames import VideoFramesError, probe_video

            try:
                info = probe_video(
                    declared_file(root, video.path), ffprobe_bin, video.path
                )
                row.update(
                    probe_status="observed", probe_metadata_json=json_text(asdict(info))
                )
            except (ValueError, OSError, VideoFramesError) as error:
                row.update(probe_status="error", probe_error=str(error))
        video_rows.append(row)
    # A full second checksum pass binds reads to immutable input snapshots. Stat
    # equality alone would not establish that source bytes stayed unchanged.
    if snapshot_sources(config, root) != sources:
        raise ValueError("Source files changed during ingestion; publication refused")
    observed = {
        "archives": table("archives", archive_rows),
        "entries": table("entries", entry_rows),
        "video_sources": table("video_sources", video_rows),
    }
    return observed, sources


def derive_evidence(config: IngestionConfig, observed: dict, sources: dict):
    """Reconstruct logical identities and candidates solely from stored ZIP facts.

    Repeated named indices cannot supply unique frame_id to the existing manifest
    contract. Preserve their physical occurrences and withhold the whole scientific
    manifest, rather than choosing representatives or inventing suffixes.
    """
    rows, issues = [], []
    series_by_id = {s.series_id: s for s in config.series}

    def issue(
        series, kind, entries=(), start=None, stop=None, step=None, description=""
    ):
        issues.append(
            dict(
                series_id=series,
                issue_type=kind,
                index_start=start,
                index_stop=stop,
                index_step=step,
                entry_ids_json=json_text(sorted(entries)),
                description=description,
            )
        )

    for entry in records(observed["entries"]):
        if not entry["is_image"]:
            continue
        matches = [
            s
            for s in config.series
            if entry["archive_key"] in s.archive_keys
            and re.fullmatch(s.member_pattern, entry["source_member_path"])
        ]
        series = matches[0] if len(matches) == 1 else None
        index = None
        if series is not None:
            match = re.fullmatch(
                series.index_pattern, PurePosixPath(entry["source_member_path"]).name
            )
            if match is not None:
                token = match.group("index")
                if token is not None and token.isascii() and token.isdecimal():
                    # Bound before integer conversion, including pathological names.
                    canonical = token.lstrip("0") or "0"
                    if len(canonical) <= 19 and int(canonical) <= 2**63 - 1:
                        index = int(canonical)
        copied = {
            key: entry[key] for key in TABLE_DTYPES["occurrences"] if key in entry
        }
        rows.append(
            dict(
                **copied,
                collection_id=config.collection_id,
                series_id=series.series_id if series else None,
                possible_series_ids_json=json_text(
                    sorted(s.series_id for s in matches)
                ),
                observed_index=index,
                observation_id=None,
                frame_id=None,
                identity_status="unresolved",
                content_id=entry["image_sha256"],
            )
        )
        if series is None:
            issue(
                None,
                "ambiguous_series" if matches else "unassigned_series",
                [entry["entry_id"]],
            )
        elif index is None:
            issue(series.series_id, "unparsed_index", [entry["entry_id"]])
    for series in config.series:
        members = [row for row in rows if row["series_id"] == series.series_id]
        by_index = {}
        for row in members:
            if row["observed_index"] is not None:
                by_index.setdefault(row["observed_index"], []).append(row)
        if not members:
            issue(series.series_id, "empty_series")
        for index, group in sorted(by_index.items()):
            if len(group) > 1:
                issue(
                    series.series_id,
                    "duplicate_index",
                    [r["entry_id"] for r in group],
                    index,
                    index,
                    description="Logical observation is ambiguous; no physical entry is selected",
                )
            for row in group:
                row["observation_id"] = digest_document(
                    [MANIFEST_VERSION, config.collection_id, series.series_id, index]
                )
                if len(group) > 1:
                    row["identity_status"] = "ambiguous_index"
                elif row["image_sha256"] is not None:
                    row["identity_status"] = "identified"
                    row["frame_id"] = scientific_frame_id(
                        config.collection_id,
                        series.series_id,
                        index,
                        row["image_sha256"],
                    )
        indices = sorted(by_index)
        first, last = series.expected_index_start, series.expected_index_stop
        if first is None:
            first, last = (indices[0], indices[-1]) if indices else (None, None)
        if first is not None:
            step, cursor = series.expected_index_step, first
            for index in indices:
                if index < first or index > last or (index - first) % step:
                    issue(
                        series.series_id,
                        "off_grid_index",
                        [r["entry_id"] for r in by_index[index]],
                        index,
                        index,
                    )
                    continue
                if index > cursor:
                    issue(
                        series.series_id,
                        "missing_index_range",
                        start=cursor,
                        stop=index - step,
                        step=step,
                        description="declared_range"
                        if series.expected_index_start is not None
                        else "observed_interior_only",
                    )
                cursor = index + step
            if cursor <= last:
                issue(
                    series.series_id,
                    "missing_index_range",
                    start=cursor,
                    stop=last,
                    step=step,
                    description="declared_range"
                    if series.expected_index_start is not None
                    else "observed_interior_only",
                )
    occurrences = table("occurrences", sorted(rows, key=lambda row: row["entry_id"]))
    candidates = []
    video_ids = {
        v["video_key"]: v["source_video_id"] for v in records(observed["video_sources"])
    }
    for row in records(occurrences):
        series = series_by_id.get(row["series_id"])
        video_key = series.candidate_video_key if series else None
        seconds, rule = None, None
        if series and series.candidate_time and row["observed_index"] is not None:
            hypothesis = series.candidate_time
            seconds = (
                row["observed_index"] - hypothesis.index_origin
            ) / hypothesis.fps + hypothesis.offset_seconds
            rule = json_text(
                {
                    "rule": "(observed_index-index_origin)/fps+offset_seconds",
                    **hypothesis.model_dump(),
                }
            )
            if not math.isfinite(seconds) or seconds < 0:
                seconds = None
                issue(series.series_id, "invalid_candidate_time", [row["entry_id"]])
        candidates.append(
            dict(
                entry_id=row["entry_id"],
                frame_id=row["frame_id"],
                observed_index=row["observed_index"],
                candidate_video_key=video_key,
                candidate_source_video_id=video_ids.get(video_key),
                candidate_timestamp_seconds=seconds,
                timestamp_rule=rule,
                alignment_status="ambiguous"
                if row["identity_status"] != "identified"
                else "candidate"
                if video_key
                else "unverified",
                verification_scope="none",
                verified_native_frame_index=None,
                verified_timestamp_seconds=None,
            )
        )
    observations = []
    for observation in sorted(
        config.observations, key=lambda item: item.observation_id
    ):
        row = observation.model_dump()
        row["evidence_sha256"] = sources.get(
            f"evidence:{observation.observation_id}", {}
        ).get("sha256")
        observations.append(row)
    result = dict(
        observed,
        occurrences=occurrences,
        index_issues=table("index_issues", issues),
        alignment_candidates=table("alignment_candidates", candidates),
        observations=table("observations", observations),
    )
    archive_errors = int(observed["archives"].archive_error.notna().sum())
    available = bool(
        len(rows)
        and not archive_errors
        and occurrences.identity_status.eq("identified").all()
    )
    manifest = None
    if available:
        manifest_rows = []
        for row in sorted(records(occurrences), key=lambda item: item["frame_id"]):
            manifest_rows.append(
                {
                    **{key: row[key] for key in MANIFEST_COLUMNS if key in row},
                    "manifest_version": MANIFEST_VERSION,
                    "label_exists": False,
                    "label_sha256": "",
                    "original_split": "",
                }
            )
        # The scientific manifest deliberately excludes storage refs. Consumers
        # join frame_id -> occurrences -> entry_id; repackaging cannot alter it.
        manifest = pd.DataFrame(manifest_rows, columns=list(MANIFEST_COLUMNS)).astype(
            MANIFEST_COLUMNS
        )
    content_counts = occurrences.content_id.value_counts()
    summary = {
        "ingestion_completed": True,
        "scientific_manifest_available": available,
        "dataset_id": dataset_id_from_manifest(manifest) if available else None,
        "archive_count": len(observed["archives"]),
        "physical_entry_count": len(observed["entries"]),
        "image_occurrence_count": len(rows),
        "unique_content_count": int(occurrences.content_id.nunique()),
        "exact_duplicate_content_count": int(content_counts.gt(1).sum()),
        "duplicate_records": int(content_counts[content_counts.gt(1)].sum()),
        "redundant_records": int(
            occurrences.content_id.notna().sum() - len(content_counts)
        ),
        "unresolved_scientific_occurrences": int(occurrences.frame_id.isna().sum()),
        "archive_errors": archive_errors,
        "read_errors": int(occurrences.read_error.notna().sum()),
        "decode_errors": int(
            (
                occurrences.image_sha256.notna()
                & ~occurrences.image_decode_valid.fillna(False)
            ).sum()
        ),
        "index_issue_count": len(issues),
        "missing_index_count": sum(
            (i["index_stop"] - i["index_start"]) // i["index_step"] + 1
            for i in issues
            if i["issue_type"] == "missing_index_range"
        ),
        "index_completeness_scope": {
            s.series_id: "declared_range"
            if s.expected_index_start is not None
            else "observed_interior_only"
            for s in config.series
        },
        "alignment_verified": False,
        "alignment_verification_scope": "none",
        "verified_alignment_count": 0,
        "external_observation_count": len(observations),
        "labels_available": False,
        "detector_ready": False,
        "ground_truth": False,
        "sequences_created": False,
        "splits_created": False,
        "consumer_adapter_available": False,
    }
    summary["integrity_valid"] = bool(
        available and not issues and not summary["decode_errors"]
    )
    return result, manifest, summary
