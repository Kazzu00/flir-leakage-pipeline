"""Portable dataset identity shared by manifests, features and local reports."""

from __future__ import annotations

import hashlib
import json

import pandas as pd


def dataset_id_from_manifest(manifest: pd.DataFrame) -> str:
    """Identify the supplied records, never a report found in the working directory.

    Hash sorted (frame_id, image_sha256, label_sha256) tuples and the manifest
    version. This preserves the canonical v1 algorithm and ignores row order,
    absolute paths and historical split metadata. Annotation changes intentionally
    change dataset identity, although annotations never enter image encoders.
    Unversioned synthetic manifests retain the original feature fallback hash.
    """
    columns = ["frame_id", "image_sha256", "label_sha256"]
    missing = set(columns) - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest identity requires columns: {sorted(missing)}")
    if manifest[columns].isna().any().any():
        raise ValueError("Manifest identity columns must not contain null values")
    if not manifest["frame_id"].is_unique:
        raise ValueError("Manifest frame_id must identify unique occurrences")
    values = sorted(manifest[columns].astype(str).itertuples(index=False, name=None))
    payload = json.dumps(values, separators=(",", ":"), ensure_ascii=True)
    if "manifest_version" in manifest.columns and not manifest.empty:
        versions = manifest["manifest_version"].dropna().unique()
        if len(versions) != 1 or manifest["manifest_version"].isna().any():
            raise ValueError("Manifest must have exactly one non-null manifest_version")
        payload = f"{versions[0]}\x1f{payload}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
