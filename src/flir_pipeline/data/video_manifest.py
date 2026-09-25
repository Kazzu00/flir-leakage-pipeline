"""Occurrence/content identities for completed, read-only video samples."""

from __future__ import annotations

import hashlib
import io
import json
import math
import platform
import re
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd

from flir_pipeline.data.identity import dataset_id_from_manifest
from flir_pipeline.data.local_images import declared_file, relative_posix_path
from flir_pipeline.data.manifest import _git_commit, _image_properties, _stable_id
from flir_pipeline.data.video_frames import (
    FRAME_DTYPES,
    PRODUCER,
    PUBLICATION_MARKER,
    SCHEMA_VERSION,
    video_id_for,
)
from flir_pipeline.utils.hashing import sha256_file

MANIFEST_VERSION = "flir_video_samples_v1"


def _require_completed(root: Path) -> None:
    marker = root / PUBLICATION_MARKER
    if marker.exists() or marker.is_symlink():
        raise ValueError(
            "Incomplete video publication: .video-frames-publishing exists"
        )


def _positive_number(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"Invalid {name}: expected a finite positive number")
    return float(value)


def _validate_receipt(frames: pd.DataFrame, summary: dict) -> dict[str, dict]:
    """Bind every temporal row to the producer's grid, video facts and counts."""
    missing = set(FRAME_DTYPES) - set(frames.columns)
    if missing or frames.empty:
        raise ValueError(f"Empty frames.parquet or missing columns: {sorted(missing)}")
    required = [
        key for key, dtype in FRAME_DTYPES.items() if dtype not in {"Float64", "Int64"}
    ]
    if frames[required].isna().any().any():
        raise ValueError("Required sampling fields contain null values")
    if frames.duplicated(["video_id", "sample_index"]).any():
        raise ValueError("Duplicate temporal sample identity (video_id, sample_index)")
    if not frames.image_path.is_unique:
        raise ValueError("Duplicate image_path in frames.parquet")
    for name in frames.image_path:
        relative_posix_path(name)
    fps = _positive_number(summary["sample_fps"], "sample_fps")
    videos = summary["videos"]
    if not isinstance(videos, list) or not videos:
        raise ValueError("summary videos must be a nonempty list")
    if (
        type(summary["processed_videos"]) is not int
        or summary["processed_videos"] != len(videos)
        or type(summary["total_frames"]) is not int
        or summary["total_frames"] != len(frames)
    ):
        raise ValueError("summary video/frame counts disagree with frames.parquet")
    by_id = {}
    for video in videos:
        identity, source = video["video_id"], video["source_video"]
        relative_posix_path(source)
        count = video["extracted_frames"]
        if identity != video_id_for(source) or identity in by_id:
            raise ValueError("Invalid or duplicate summary video identity")
        if type(count) is not int or count <= 0:
            raise ValueError("Invalid extracted_frames in summary")
        if not re.fullmatch(r"[0-9a-f]{64}", video["source_sha256"]):
            raise ValueError("Invalid source_sha256 in summary")
        group = frames[frames.video_id.eq(identity)].sort_values("sample_index")
        if len(group) != count:
            raise ValueError(f"summary and frames.parquet counts disagree: {identity}")
        if not pd.api.types.is_integer_dtype(
            group.sample_index.dtype
        ) or not np.array_equal(group.sample_index, np.arange(count)):
            raise ValueError(f"Non-contiguous integer sample_index: {identity}")
        if not group.image_path.eq(
            [f"{identity}/frame_{i:06d}.jpg" for i in range(count)]
        ).all():
            raise ValueError(f"image_path disagrees with temporal identity: {identity}")
        if (
            not group.source_video.eq(source).all()
            or not group.sample_fps.eq(fps).all()
        ):
            raise ValueError(f"Sampling provenance disagrees with summary: {identity}")
        if not np.allclose(
            group.timestamp_seconds, np.arange(count) / fps, rtol=0, atol=1e-9
        ):
            raise ValueError(f"Invalid timestamp_seconds sampling grid: {identity}")
        for key in (
            "source_fps",
            "source_width",
            "source_height",
            "source_duration_seconds",
            "source_nb_frames",
            "codec_name",
        ):
            value = video[key]
            matches = (
                group[key].isna()
                if value is None
                else group[key].eq(value).fillna(False)
            )
            if not matches.all():
                raise ValueError(f"{key} disagrees with summary: {identity}")
        source_fps = video["source_fps"]
        if source_fps is None:
            valid_estimates = group.source_frame_index_estimate.isna().all()
        else:
            source_fps = _positive_number(source_fps, "source_fps")
            estimates = [math.floor(i / fps * source_fps + 0.5) for i in range(count)]
            valid_estimates = (
                group.source_frame_index_estimate.eq(estimates).fillna(False).all()
            )
        if not valid_estimates:
            raise ValueError(f"Invalid source_frame_index_estimate: {identity}")
        by_id[identity] = video
    if set(frames.video_id) != set(by_id):
        raise ValueError("summary and frames.parquet video sets disagree")
    return by_id


def build_video_manifest(frames_root: Path, output: Path, report_output: Path) -> dict:
    """Keep all occurrences, including corrupt JPEGs explicitly flagged for QA.

    frame_id hashes unit-separated version, path-based video ID, source-video
    SHA256, sample FPS in float.hex form, sample_index and JPEG SHA256. Thus an
    occurrence is portable but bound to its source bytes, grid and exact image.
    dataset_id uses the unchanged shared algorithm with label_sha256="" (absent).
    Outputs must be outside the read-only sampling root. No sequence is inferred.
    """
    root = frames_root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError("frames-root must be an existing directory")
    output, report_output = (
        output.expanduser().resolve(),
        report_output.expanduser().resolve(),
    )
    if output.is_relative_to(root) or report_output.is_relative_to(root):
        raise ValueError("Manifest and reports must be outside read-only frames-root")
    report_path = report_output / "summary.json"
    if output == report_path or report_output.is_relative_to(output):
        raise ValueError("Manifest output and report directory must not collide")
    _require_completed(root)
    summary_bytes = declared_file(root, "summary.json").read_bytes()
    parquet_bytes = declared_file(root, "frames.parquet").read_bytes()
    parquet_hash = hashlib.sha256(parquet_bytes).hexdigest()
    summary_hash = hashlib.sha256(summary_bytes).hexdigest()
    try:
        summary = json.loads(summary_bytes)
        if (
            summary["producer"] != PRODUCER
            or type(summary["schema_version"]) is not int
            or summary["schema_version"] != SCHEMA_VERSION
        ):
            raise ValueError("Unrecognized extract-video-frames producer/schema")
        if summary["frames_parquet_sha256"] != parquet_hash:
            raise ValueError("frames.parquet checksum mismatch")
        frames = pd.read_parquet(io.BytesIO(parquet_bytes))
        by_id = _validate_receipt(frames, summary)
        # Only producer-defined provenance enters this version; no ad hoc labels,
        # splits or inferred sequences may ride along through extra input columns.
        frames = frames[list(FRAME_DTYPES)]
    except (KeyError, TypeError, AttributeError, OverflowError) as error:
        raise ValueError(f"Invalid sampling receipt/schema: {error}") from error
    # Validate every declared destination before hashing or writing any output.
    for relative in frames.image_path:
        declared_file(root, relative)
    rows = []
    for sample in frames.sort_values(["video_id", "sample_index"]).to_dict("records"):
        content = declared_file(root, sample["image_path"]).read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        properties = _image_properties(content)
        if not properties["image_error"] and properties["image_format"] != "JPEG":
            properties["image_error"] = "Expected JPEG bytes from extract-video-frames"
        source_hash = by_id[sample["video_id"]]["source_sha256"]
        rows.append(
            {
                **sample,
                "frame_id": _stable_id(
                    MANIFEST_VERSION,
                    sample["video_id"],
                    source_hash,
                    float(sample["sample_fps"]).hex(),
                    str(sample["sample_index"]),
                    digest,
                ),
                "content_id": digest,
                "image_sha256": digest,
                "manifest_version": MANIFEST_VERSION,
                "source_video_sha256": source_hash,
                "source_type": "directory",
                "source_archive": "",
                # Compatibility aliases denote a root-relative local file, never a ZIP member.
                "source_member_path": sample["image_path"],
                "relative_image_path": sample["image_path"],
                "label_sha256": "",
                "label_exists": False,
                "original_split": "",
                "image_decode_valid": not properties["image_error"],
                **properties,
            }
        )
    manifest = pd.DataFrame(rows).astype(FRAME_DTYPES)
    counts = manifest.content_id.value_counts()
    manifest["duplicate_occurrence_count"] = manifest.content_id.map(counts)
    manifest["exact_duplicate"] = manifest.duplicate_occurrence_count.gt(1)
    manifest["duplicate_group_id"] = manifest.content_id.map(
        lambda value: f"duplicate-{value}" if counts[value] > 1 else ""
    )
    failures = manifest.loc[
        ~manifest.image_decode_valid, ["frame_id", "image_path", "image_error"]
    ]
    report = {
        "producer": "flir-pipeline.data.build-video-manifest",
        "manifest_version": MANIFEST_VERSION,
        "dataset_id": dataset_id_from_manifest(manifest),
        "total_records": len(manifest),
        "unique_content_ids": len(counts),
        "exact_duplicate_content_ids": int(counts.gt(1).sum()),
        "duplicate_records": int(manifest.exact_duplicate.sum()),
        "redundant_records": len(manifest) - len(counts),
        "records_per_video": manifest.video_id.value_counts().sort_index().to_dict(),
        "decode_failures": len(failures),
        "decode_failure_records": failures.to_dict("records"),
        "frames_parquet_sha256": parquet_hash,
        "sampling_summary_sha256": summary_hash,
        "read_only_source": True,
        "created_at": datetime.now(UTC).isoformat(),
        "python_version": platform.python_version(),
        "git_commit": _git_commit(),
        "implementation_sha256": sha256_file(Path(__file__)),
    }
    _require_completed(root)
    if (
        sha256_file(declared_file(root, "frames.parquet")) != parquet_hash
        or sha256_file(declared_file(root, "summary.json")) != summary_hash
    ):
        raise ValueError("Sampling metadata changed while building manifest")
    output.parent.mkdir(parents=True, exist_ok=True)
    report_output.mkdir(parents=True, exist_ok=True)
    # Exclusive temporary files and replace avoid writing through an existing hardlink.
    with NamedTemporaryFile(
        dir=output.parent, suffix=".parquet", delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        manifest.to_parquet(temporary, index=False)
        report["manifest_parquet_sha256"] = sha256_file(temporary)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    with NamedTemporaryFile(dir=report_output, suffix=".json", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        temporary.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        temporary.replace(report_path)
    finally:
        temporary.unlink(missing_ok=True)
    return report
