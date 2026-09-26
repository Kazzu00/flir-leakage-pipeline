"""Validate sampling-grid provenance without inferring scenes or capture times."""

from __future__ import annotations

import numpy as np
import pandas as pd

VIDEO_VERSION = "flir_video_samples_v1"
VIDEO_COLUMNS = (
    "manifest_version", "frame_id", "content_id", "image_sha256", "image_path",
    "video_id", "source_video", "source_video_sha256", "sample_index",
    "timestamp_seconds", "sample_fps", "source_frame_index_estimate",
)
VIDEO_SEMANTICS = {
    "temporal_source": "sampled_video_grid",
    "video_id": "source video identity; not sequence identity",
    "sample_index": "sampling-grid index; samples",
    "timestamp_seconds": "relative sampling-grid time; seconds; not capture time",
    "source_frame_index_estimate": "estimate; not an exact decoder index",
    "capture_timestamp": "unavailable/unverified",
    "sequence_identity": "unknown",
    "same_source_video": "shared source membership; does not mean same_sequence",
    "gap_rule": "independent minima over all same-video occurrence pairs",
    "cross_video_gaps": "null when source-video membership sets are disjoint",
}


def temporal_mode(manifest: pd.DataFrame) -> str:
    """Reject mixed/undeclared grids instead of silently using filename heuristics."""
    versions = manifest.get("manifest_version", pd.Series(dtype=str))
    if versions.eq(VIDEO_VERSION).any():
        if versions.isna().any() or not versions.eq(VIDEO_VERSION).all():
            raise ValueError("Mixed temporal manifest versions are unsupported")
        return "sampled_video_grid"
    if {"video_id", "sample_index", "timestamp_seconds"} & set(manifest):
        raise ValueError("Video temporal fields require flir_video_samples_v1")
    return "filename_heuristic"


def audit_video_temporal_lineage(manifest: pd.DataFrame) -> pd.DataFrame:
    """Keep every occurrence; validate the declared grid, not physical capture time.

    Upstream build-video-manifest binds the grid to the sampling receipt. This
    audit checks its internal consistency without opening or changing any images.
    Noncontiguous subsets are valid; duplicate grid positions are not.
    """
    if temporal_mode(manifest) != "sampled_video_grid":
        raise ValueError("Expected sampled_video_grid provenance")
    missing = set(VIDEO_COLUMNS) - set(manifest)
    if missing or manifest.empty:
        raise ValueError(f"Missing video provenance or empty manifest: {sorted(missing)}")
    if {"sequence_id", "possible_sequence", "sequence_key", "split_id", "new_split", "cluster_id", "group_id"} & set(manifest):
        raise ValueError("Sampling-grid provenance must not declare inferred sequences or assigned groups/splits")
    # These empty/false fields exist in the producer's unlabeled v1 manifest.
    # Reject conflicting annotations instead of silently discarding them.
    for name in ("original_split", "label_sha256"):
        if name in manifest and not manifest[name].fillna("").eq("").all():
            raise ValueError("flir_video_samples_v1 has no labels or assigned splits")
    if "label_exists" in manifest and not manifest.label_exists.eq(False).fillna(False).all():
        raise ValueError("flir_video_samples_v1 has no labels")
    records = manifest[list(VIDEO_COLUMNS)].copy()
    for name in VIDEO_COLUMNS[:8]:
        if not records[name].map(lambda v: isinstance(v, str) and bool(v.strip())).all():
            raise ValueError(f"Invalid video provenance identity: {name}")
    if (not records.frame_id.is_unique or not records.image_path.is_unique
            or records.duplicated(["video_id", "sample_index"]).any()
            or not records.content_id.eq(records.image_sha256).all()):
        raise ValueError("Invalid occurrence/content/grid identities")
    for name in ("sample_index", "source_frame_index_estimate"):
        values = records[name]
        if (not pd.api.types.is_integer_dtype(values.dtype)
                or values.dropna().lt(0).any()
                or (name == "sample_index" and values.isna().any())):
            raise ValueError(f"{name} must contain nonnegative integers")
        records[name] = values.astype("Int64")
    for name in ("timestamp_seconds", "sample_fps"):
        values = records[name]
        if (not pd.api.types.is_numeric_dtype(values.dtype)
                or pd.api.types.is_bool_dtype(values.dtype)
                or not np.isfinite(values.to_numpy(dtype=float, na_value=np.nan)).all()):
            raise ValueError(f"{name} must contain finite numbers")
        records[name] = values.astype("float64")
    if records.timestamp_seconds.lt(0).any() or records.sample_fps.le(0).any() or not np.allclose(
        records.timestamp_seconds, records.sample_index.to_numpy(dtype=float) / records.sample_fps,
        atol=1e-9, rtol=0,
    ):
        raise ValueError("timestamp_seconds must equal sample_index / sample_fps")
    for _, group in records.groupby("video_id", sort=False):
        if any(group[name].nunique() != 1 for name in ("source_video", "source_video_sha256", "sample_fps")):
            raise ValueError("Conflicting source identity or sampling FPS within a video")
    if records.groupby("source_video").video_id.nunique().gt(1).any():
        raise ValueError("Source video has conflicting video_id values")
    records["temporal_source"] = "sampled_video_grid"
    records["capture_timestamp_verified"] = False
    return records.sort_values("frame_id").reset_index(drop=True)
