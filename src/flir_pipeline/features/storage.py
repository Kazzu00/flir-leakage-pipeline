"""Content-level embedding storage, deterministic spaces and resumable extraction."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from flir_pipeline.data.identity import dataset_id_from_manifest
from flir_pipeline.features.base import FeatureExtractor, l2_normalize
from flir_pipeline.features.preprocessing import decode_zip_image


def feature_space_id(config: dict[str, Any]) -> str:
    """Return a 16-hex SHA256 prefix of canonical mathematical configuration.

    Sorted JSON includes model/revision, pooling, preprocessing and normalization.
    Device, batch size and timestamps do not change identity. Dataset membership
    and sampled content belong to the separate cache signature.
    """
    excluded = {"device", "batch_size", "created_at", "timestamp"}
    canonical = {key: value for key, value in config.items() if key not in excluded}
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _selected_content(manifest: pd.DataFrame, limit_content: int | None, seed: int) -> pd.DataFrame:
    unique = manifest.sort_values("content_id").drop_duplicates("content_id", keep="first")
    if limit_content is None:
        return unique.reset_index(drop=True)
    if limit_content < 1:
        raise ValueError("limit_content must be positive")
    if len(unique) <= limit_content:
        return unique.reset_index(drop=True)
    generator = np.random.default_rng(seed)
    selected_indices = generator.choice(len(unique), size=limit_content, replace=False)
    return unique.iloc[np.sort(selected_indices)].reset_index(drop=True)


def _atomic_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, path)


def _quality(raw: np.ndarray, normalized: np.ndarray, content_index: pd.DataFrame, record_index: pd.DataFrame) -> dict:
    """Check numerical health and both directions of the content/occurrence map."""
    shape_valid = raw.ndim == normalized.ndim == 2 and raw.shape == normalized.shape and len(raw) > 0
    norms = np.linalg.norm(normalized, axis=1) if normalized.ndim == 2 else np.array([])
    index_valid = (
        len(content_index) == len(raw)
        and content_index["content_id"].is_unique
        and not content_index["content_id"].isna().any()
        and np.array_equal(content_index["embedding_row"], np.arange(len(raw)))
        and record_index["frame_id"].is_unique
        and not record_index[["frame_id", "content_id"]].isna().any().any()
    )
    mapping_valid = False
    if index_valid and "embedding_row" in record_index:
        rows = content_index.set_index("content_id")["embedding_row"]
        expected = record_index["content_id"].map(rows).fillna(-1)
        mapping_valid = bool(
            expected.eq(record_index["embedding_row"]).all()
            and set(content_index["content_id"]) <= set(record_index["content_id"])
        )
    result = {
        "content_embedding_count": int(len(raw)),
        "embedding_dimension": int(raw.shape[1]) if raw.ndim == 2 else 0,
        "raw_has_nan": bool(np.isnan(raw).any()),
        "raw_has_inf": bool(np.isinf(raw).any()),
        "normalized_has_nan": bool(np.isnan(normalized).any()),
        "normalized_has_inf": bool(np.isinf(normalized).any()),
        "zero_norm_count": int((norms == 0).sum()),
        "l2_norm_valid": bool(np.allclose(norms[norms > 0], 1.0, atol=1e-5)),
        "content_id_unique": bool(content_index["content_id"].is_unique),
        "embedding_row_unique": bool(content_index["embedding_row"].is_unique),
        "record_frame_id_unique": bool(record_index["frame_id"].is_unique),
        "array_shape_valid": bool(shape_valid),
        "index_valid": bool(index_valid),
        "record_mapping_valid": mapping_valid,
        "quality_valid": bool(
            shape_valid
            and index_valid
            and mapping_valid
            and not np.isnan(raw).any()
            and not np.isinf(raw).any()
            and not np.isnan(normalized).any()
            and not np.isinf(normalized).any()
            and (norms == 0).sum() == 0
            and np.allclose(norms, 1.0, atol=1e-5)
        ),
    }
    if shape_valid and result["quality_valid"]:
        expected_l2, _ = l2_normalize(raw)
        result["quality_valid"] = bool(np.allclose(expected_l2, normalized, atol=1e-5))
    return result


def extract_to_store(
    manifest_path: Path,
    images_archive: Path,
    extractor: FeatureExtractor,
    output_root: Path = Path("artifacts/features"),
    dataset_id: str | None = None,
    batch_size: int = 8,
    limit_content: int | None = None,
    seed: int = 0,
) -> Path:
    """Store one raw/L2 float32 row per selected content_id from a read-only ZIP.

    The Parquet manifest supplies occurrences and source-member paths. Exact
    copies share a vector so future density clustering does not count them twice;
    record_index preserves all frame_id, using -1 for unsampled smoke contents.
    A single writer flushes each batch before atomically advancing its checkpoint.
    Resume requires the same dataset, space and selection. Metadata is written
    last as the completion marker. Existing valid artifacts are reused, never
    migrated in place. Return the final feature directory.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    manifest = pd.read_parquet(manifest_path)
    if manifest.empty:
        raise ValueError("Cannot extract features from an empty manifest")
    computed_dataset_id = dataset_id_from_manifest(manifest)
    dataset_id = dataset_id or computed_dataset_id
    selected = _selected_content(manifest, limit_content, seed)
    config = extractor.feature_space_config()
    space_id = feature_space_id(config)
    feature_dir = output_root / extractor.name / dataset_id / space_id
    feature_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = feature_dir / "metadata.json"
    signature = {
        "dataset_id": dataset_id,
        "feature_space_id": space_id,
        "content_ids": selected["content_id"].tolist(),
        "image_sha256": selected["image_sha256"].tolist(),
    }
    final_metadata = metadata_path if metadata_path.is_file() else None
    if final_metadata:
        existing = json.loads(final_metadata.read_text(encoding="utf-8"))
        if existing.get("cache_signature") != signature:
            raise RuntimeError("Existing feature cache does not match dataset/config selection")
        if not verify_feature_directory(feature_dir)["quality_valid"]:
            raise RuntimeError("Existing feature cache failed verification; preserve it for inspection")
        return feature_dir
    partial = feature_dir / ".partial"
    partial.mkdir(exist_ok=True)
    checkpoint_path = partial / "checkpoint.json"
    if checkpoint_path.is_file():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("signature") != signature:
            raise RuntimeError("Partial feature cache does not match dataset/config selection")
        completed = int(checkpoint["completed"])
        if checkpoint.get("embedding_dimension") != extractor.embedding_dimension:
            raise RuntimeError("Partial feature cache has a different embedding dimension")
    else:
        completed = 0
        _atomic_json(
            checkpoint_path,
            {"signature": signature, "completed": 0, "embedding_dimension": extractor.embedding_dimension},
        )
    total = len(selected)
    raw_path = partial / "embeddings_raw.npy"
    l2_path = partial / "embeddings_l2.npy"
    if not 0 <= completed <= total:
        raise RuntimeError("Partial feature checkpoint has an invalid completed count")
    if completed and not (raw_path.is_file() and l2_path.is_file()):
        raise RuntimeError("Partial feature arrays are missing; preserve this interrupted publication for inspection")
    mode = "r+" if raw_path.exists() and l2_path.exists() else "w+"
    raw = np.lib.format.open_memmap(
        raw_path,
        mode=mode,
        dtype=np.float32,
        shape=(total, extractor.embedding_dimension),
    )
    normalized = np.lib.format.open_memmap(
        l2_path,
        mode=mode,
        dtype=np.float32,
        shape=(total, extractor.embedding_dimension),
    )
    if raw.shape != (total, extractor.embedding_dimension) or normalized.shape != raw.shape:
        raise RuntimeError("Partial feature arrays do not match the checkpoint shape")
    with zipfile.ZipFile(images_archive) as archive:
        for start in tqdm(range(completed, total, batch_size), desc=f"Extracting {extractor.name}", unit="batch"):
            stop = min(start + batch_size, total)
            batch_images = [
                extractor.preprocess(decode_zip_image(archive, member_path))
                for member_path in selected.loc[start:stop - 1, "source_member_path"]
            ]
            batch_raw = np.asarray(extractor.encode_batch(batch_images), dtype=np.float32)
            if batch_raw.shape != (stop - start, extractor.embedding_dimension):
                raise ValueError(f"Extractor returned unexpected shape: {batch_raw.shape}")
            batch_l2, _ = l2_normalize(batch_raw)
            raw[start:stop] = batch_raw
            normalized[start:stop] = batch_l2
            raw.flush()
            normalized.flush()
            _atomic_json(
                checkpoint_path,
                {"signature": signature, "completed": stop, "embedding_dimension": extractor.embedding_dimension},
            )
    raw.flush()
    normalized.flush()
    del raw, normalized
    os.replace(raw_path, feature_dir / "embeddings_raw.npy")
    os.replace(l2_path, feature_dir / "embeddings_l2.npy")
    content_index = pd.DataFrame(
        {
            "embedding_row": np.arange(total, dtype=np.int64),
            "content_id": selected["content_id"].tolist(),
            "representative_frame_id": selected["frame_id"].tolist(),
            "image_sha256": selected["image_sha256"].tolist(),
            "source_archive": selected["source_archive"].tolist(),
            "source_member_path": selected["source_member_path"].tolist(),
        }
    )
    row_by_content = dict(
        zip(
            content_index["content_id"],
            content_index["embedding_row"],
            strict=True,
        )
    )
    record_index = manifest[["frame_id", "content_id"]].copy()
    record_index["embedding_row"] = record_index["content_id"].map(row_by_content).fillna(-1).astype(np.int64)
    content_index.to_parquet(feature_dir / "content_index.parquet", index=False)
    record_index.to_parquet(feature_dir / "record_index.parquet", index=False)
    raw_final = np.load(feature_dir / "embeddings_raw.npy", mmap_mode="r")
    l2_final = np.load(feature_dir / "embeddings_l2.npy", mmap_mode="r")
    quality = _quality(raw_final, l2_final, content_index, record_index)
    (feature_dir / "feature_quality.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    if not quality["quality_valid"]:
        raise RuntimeError("Extracted feature quality is invalid; metadata completion marker was not written")
    metadata = {
        **extractor.metadata(),
        "dataset_id": dataset_id,
        "feature_space_id": space_id,
        "feature_space_algorithm": "SHA256(sorted JSON feature-space config excluding device/batch_size)",
        "cache_signature": signature,
        "total_records": len(manifest),
        "unique_content_ids": int(manifest["content_id"].nunique()),
        "selected_content_ids": total,
        "seed": seed,
        "batch_size": batch_size,
        "python_version": platform.python_version(),
        "git_commit": _git_commit(),
        "created_at": datetime.now(UTC).isoformat(),
        "l2_normalized_available": True,
    }
    _atomic_json(metadata_path, metadata)
    shutil.rmtree(partial)
    return feature_dir


def verify_feature_directory(feature_dir: Path) -> dict:
    """Verify arrays, indexes and quality invariants for one feature directory."""
    raw = np.load(feature_dir / "embeddings_raw.npy", mmap_mode="r")
    normalized = np.load(feature_dir / "embeddings_l2.npy", mmap_mode="r")
    content_index = pd.read_parquet(feature_dir / "content_index.parquet")
    record_index = pd.read_parquet(feature_dir / "record_index.parquet")
    result = _quality(raw, normalized, content_index, record_index)
    result["metadata_exists"] = (feature_dir / "metadata.json").is_file()
    result["quality_valid"] = result["quality_valid"] and result["metadata_exists"]
    return result


def summarize_feature_directory(feature_dir: Path) -> dict:
    """Read metadata and quality without loading embeddings into RAM."""
    metadata = json.loads((feature_dir / "metadata.json").read_text(encoding="utf-8"))
    quality = json.loads((feature_dir / "feature_quality.json").read_text(encoding="utf-8"))
    return {"metadata": metadata, "quality": quality}


def verify_features_against_manifest(feature_dir: Path, manifest: pd.DataFrame) -> dict:
    """Check complete coverage of the supplied canonical dataset and model provenance.

    Ordinary verification allows -1 rows for smoke samples. Closure additionally
    requires every content exactly once, all historical occurrences with their
    original content mapping, matching metadata counts/identity and a resolved
    model commit. Equal numerical vectors are allowed for different contents.
    """
    result = verify_feature_directory(feature_dir)
    metadata = json.loads((feature_dir / "metadata.json").read_text(encoding="utf-8"))
    content = pd.read_parquet(feature_dir / "content_index.parquet")
    records = pd.read_parquet(feature_dir / "record_index.parquet")
    expected = manifest.set_index("frame_id")["content_id"].sort_index()
    actual = records.set_index("frame_id")["content_id"].sort_index()
    coverage = (
        manifest["frame_id"].is_unique
        and expected.equals(actual)
        and set(content["content_id"]) == set(manifest["content_id"])
        and records["embedding_row"].ge(0).all()
    )
    counts_valid = (
        metadata.get("dataset_id") == dataset_id_from_manifest(manifest)
        and metadata.get("total_records") == len(manifest)
        and metadata.get("selected_content_ids") == result["content_embedding_count"]
        and metadata.get("unique_content_ids") == manifest["content_id"].nunique()
        and metadata.get("embedding_dimension") == result["embedding_dimension"]
    )
    revision = metadata.get("resolved_model_revision")
    revision_valid = isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}", revision) is not None and metadata.get("model_revision") == revision
    result.update({
        "record_count": len(records), "full_record_content_coverage": bool(coverage),
        "metadata_matches_manifest": bool(counts_valid), "resolved_revision_valid": revision_valid,
        "full_dataset_valid": bool(result["quality_valid"] and coverage and counts_valid),
    })
    result["reproducible_full_dataset_valid"] = result["full_dataset_valid"] and revision_valid
    return result
