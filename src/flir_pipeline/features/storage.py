"""Content-level embedding storage, deterministic spaces and resumable extraction."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
from collections.abc import Callable
from contextlib import ExitStack, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from flir_pipeline.data.identity import dataset_id_from_manifest
from flir_pipeline.features.base import FeatureExtractor, l2_normalize
from flir_pipeline.features.image_source import ImageSource
from flir_pipeline.features.source_binding import (
    VideoVariantSourceBinding,
    prepare_video_variant_binding,
    validate_store_binding,
)
from flir_pipeline.utils.hashing import sha256_file


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


def _selected_content(manifest: pd.DataFrame, limit_content: int | None, seed: int, *, occurrence_tiebreak: bool = False) -> pd.DataFrame:
    columns = ["content_id", "frame_id"] if occurrence_tiebreak else "content_id"
    unique = manifest.sort_values(columns).drop_duplicates("content_id", keep="first")
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
    images_archive: Path | None = None,
    extractor: FeatureExtractor | None = None,
    output_root: Path = Path("artifacts/features"),
    dataset_id: str | None = None,
    batch_size: int = 8,
    limit_content: int | None = None,
    seed: int = 0,
    *,
    images_root: Path | None = None,
    variant_spec: Path | None = None,
    video_variant_ingestion: Path | None = None,
    input_root: Path | None = None,
    max_open_archives: int = 8,
    extractor_factory: Callable[[], FeatureExtractor] | None = None,
) -> Path:
    """Store one raw/L2 row per selected content from a read-only ZIP or directory.

    The Parquet manifest supplies occurrences and source-member paths. Exact
    copies share a vector so future density clustering does not count them twice;
    record_index preserves all frame_id, using -1 for unsampled smoke contents.
    A single writer flushes each batch before atomically advancing its checkpoint.
    Resume requires the same dataset, space and selection. Metadata is written
    last as the completion marker. Existing valid artifacts are reused, never
    migrated in place. Return the final feature directory.

    For zip_collection, pass the publication's original manifest.parquet together
    with video_variant_ingestion and input_root; do not pass historical image
    sources. Its variant is authoritative. Scientific IDs stay packaging-independent,
    while a versioned source_binding makes physical publication changes incompatible
    with resume/reuse. No labels or historical splits are required by the encoder.
    An optional extractor_factory defers model construction until source validation
    succeeds inside the same reader session, avoiding a second ZIP checksum pass.
    Supply either the existing extractor object or this factory, never both.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if extractor is None and extractor_factory is None:
        raise ValueError("An extractor is required")
    if extractor is not None and extractor_factory is not None:
        raise ValueError("Supply exactly one of extractor / extractor_factory")
    from flir_pipeline.data.variants import read_variant

    binding = None
    if video_variant_ingestion is not None:
        if input_root is None or images_archive is not None or images_root is not None:
            raise ValueError("Video variant ingestion requires input_root and excludes historical image sources")
        binding, variant = prepare_video_variant_binding(video_variant_ingestion, manifest_path, input_root)
        if variant_spec is not None and read_variant(variant_spec, manifest_path) != variant:
            raise ValueError("Requested variant differs from the ingestion declaration")
        if output_root.resolve().is_relative_to(Path(video_variant_ingestion).resolve()):
            raise ValueError("Feature output-root must be outside the ingestion publication")
        source_context = ImageSource.from_video_variant(video_variant_ingestion, input_root, max_open_archives=max_open_archives)
    else:
        if input_root is not None:
            raise ValueError("input_root requires video_variant_ingestion")
        variant = read_variant(variant_spec, manifest_path) if variant_spec else None
        source_context = ImageSource(images_archive, images_root)
    manifest = pd.read_parquet(manifest_path)
    if manifest.empty:
        raise ValueError("Cannot extract features from an empty manifest")
    selected = _selected_content(manifest, limit_content, seed, occurrence_tiebreak=binding is not None)
    if binding is not None:
        if (
            sha256_file(manifest_path) != binding.manifest_sha256
            or dataset_id_from_manifest(manifest) != binding.dataset_id
            or source_context.frame_ids != tuple(sorted(manifest.frame_id))
            or source_context.locate(selected.iloc[0].frame_id).artifact_id != binding.artifact_id
            or sha256_file(Path(video_variant_ingestion) / "metadata.json") != binding.metadata_sha256
            or sha256_file(Path(video_variant_ingestion) / "receipt.json") != binding.receipt_sha256
        ):
            raise ValueError("Ingestion source/manifest changed during preparation")
    with source_context as source:
        if source.root is not None and output_root.resolve().is_relative_to(source.root):
            raise ValueError("Feature output-root must be outside read-only images-root")
        if binding is None:
            source.validate(manifest if source.kind == "directory" else selected)
        if extractor_factory is not None:
            extractor = extractor_factory()
            if not isinstance(extractor, FeatureExtractor):
                raise TypeError("extractor_factory must return a FeatureExtractor")
        # The ingestion reader validates the complete ledger and ZIP fingerprints.
        # Selected pixels are checked by decode() exactly when used; cache reuse
        # does not need an exhaustive image replay of an unchanged ZIP publication.
        return _extract_to_store(
            manifest, selected, source, extractor, output_root, dataset_id, batch_size, seed, variant,
            binding,
        )


def _extract_to_store(
    manifest: pd.DataFrame, selected: pd.DataFrame, source: ImageSource,
    extractor: FeatureExtractor, output_root: Path, dataset_id: str | None,
    batch_size: int, seed: int, variant: dict | None = None,
    source_binding: VideoVariantSourceBinding | None = None,
) -> Path:
    """One storage/checkpoint implementation for all image transports."""
    computed_dataset_id = dataset_id_from_manifest(manifest)
    dataset_id = dataset_id or computed_dataset_id
    config = extractor.feature_space_config()
    space_id = feature_space_id(config)
    if variant and dataset_id != variant["dataset_id"]:
        raise ValueError("Dataset override differs from the declared variant")
    feature_root = output_root / extractor.name / dataset_id
    if variant:
        # Variant identity already binds dataset_id; avoid duplicating two long
        # hashes in Windows paths. The complete identities remain in metadata.
        feature_root = output_root / extractor.name / "variants" / variant["dataset_variant_id"]
    feature_dir = feature_root / space_id
    feature_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = feature_dir / "metadata.json"
    signature = {
        "dataset_id": dataset_id,
        "feature_space_id": space_id,
        "content_ids": selected["content_id"].tolist(),
        "image_sha256": selected["image_sha256"].tolist(),
    }
    if variant:
        signature["dataset_variant"] = variant
    if source_binding is not None:
        signature["source_binding"] = source_binding.model_dump(mode="json")
    final_metadata = metadata_path if metadata_path.is_file() else None
    if final_metadata:
        existing = json.loads(final_metadata.read_text(encoding="utf-8"))
        if existing.get("cache_signature") != signature:
            raise RuntimeError("Existing feature cache does not match dataset/config selection")
        if not verify_feature_directory(feature_dir)["quality_valid"]:
            raise RuntimeError("Existing feature cache failed verification; preserve it for inspection")
        return feature_dir
    partial = feature_dir / ".partial"
    if source_binding is not None and not (partial / "checkpoint.json").is_file():
        if any(p != partial for p in feature_dir.iterdir()) or partial.exists() and any(partial.iterdir()):
            raise RuntimeError("Unbound partial feature files are preserved; a compatible checkpoint is required")
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
    if source_binding is not None and raw_path.is_file() != l2_path.is_file():
        raise RuntimeError("Partial feature array pair is incomplete; preserve it for inspection")
    mode = "r+" if raw_path.exists() and l2_path.exists() else "w+"
    with ExitStack() as array_resources:
        raw = np.lib.format.open_memmap(
            raw_path,
            mode=mode,
            dtype=np.float32,
            shape=(total, extractor.embedding_dimension),
        )
        array_resources.callback(raw._mmap.close)
        normalized = np.lib.format.open_memmap(
            l2_path,
            mode=mode,
            dtype=np.float32,
            shape=(total, extractor.embedding_dimension),
        )
        array_resources.callback(normalized._mmap.close)
        if raw.shape != (total, extractor.embedding_dimension) or normalized.shape != raw.shape:
            raise RuntimeError("Partial feature arrays do not match the checkpoint shape")
        if source_binding is not None and (raw.dtype != np.float32 or normalized.dtype != np.float32):
            raise RuntimeError("Partial video variant feature arrays must be float32")
        path_column = source.path_column(selected)
        for start in tqdm(range(completed, total, batch_size), desc=f"Extracting {extractor.name}", unit="batch"):
            stop = min(start + batch_size, total)
            batch_images = [
                extractor.preprocess(source.decode(relative, expected))
                for relative, expected in selected.loc[start:stop - 1, [path_column, "image_sha256"]].itertuples(index=False, name=None)
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
            "source_archive": selected["source_archive"].tolist() if source.kind == "zip" else [""] * total,
            "source_member_path": selected[path_column].tolist(),
        }
    )
    if source.kind == "directory":
        content_index["source_type"] = "directory"
        content_index["image_path"] = selected[path_column].tolist()
    if source_binding is not None:
        # Physical locators are columns, never content/feature identity. A frame_id
        # reference must not be written into the legacy member-path column.
        physical = _physical_locations(source, selected.frame_id)
        for column in physical:
            content_index[column] = physical[column]
    row_by_content = dict(
        zip(
            content_index["content_id"],
            content_index["embedding_row"],
            strict=True,
        )
    )
    record_index = manifest[["frame_id", "content_id"]].copy()
    if source_binding is not None:
        # Preserve the original typed scientific fields, including declared
        # annotation/split absence. This is an index copy, not a new manifest.
        record_index = manifest.copy().reset_index(drop=True)
        physical = _physical_locations(source, record_index.frame_id)
        for column in physical:
            record_index[column] = physical[column]
    record_index["embedding_row"] = record_index["content_id"].map(row_by_content).fillna(-1).astype(np.int64)
    content_index.to_parquet(feature_dir / "content_index.parquet", index=False)
    record_index.to_parquet(feature_dir / "record_index.parquet", index=False)
    with _feature_arrays(feature_dir) as (raw_final, l2_final):
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
        "image_source_type": source.kind,
        "read_only_source": True,
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
    if variant:
        metadata["dataset_variant"] = variant
    if source_binding is not None:
        metadata["source_binding"] = source_binding.model_dump(mode="json")
        metadata["source_index_checksums"] = {
            name: sha256_file(feature_dir / name)
            for name in ("content_index.parquet", "record_index.parquet")
        }
        with _feature_arrays(feature_dir) as (raw_final, l2_final):
            if not _bound_store_valid(feature_dir, metadata, raw_final, l2_final, content_index, record_index):
                raise RuntimeError("Feature source binding/index validation failed; publication preserved")
    _atomic_json(metadata_path, metadata)
    shutil.rmtree(partial)
    return feature_dir


_PHYSICAL_COLUMNS = (
    "entry_id", "archive_key", "member_ordinal", "source_archive", "source_member_path",
)


def _physical_locations(source: ImageSource, frame_ids) -> pd.DataFrame:
    """Copy exact ledger locators; never derive a shard from scientific ordering."""
    locations = (source.locate(frame_id) for frame_id in frame_ids)
    return pd.DataFrame([
        {"source_type": "zip_collection", **{column: getattr(location, column) for column in _PHYSICAL_COLUMNS}}
        for location in locations
    ])


def _bound_store_valid(feature_dir, metadata, raw, normalized, content, records) -> bool:
    """Verify stored scientific/physical mapping without opening original sources.

    SHA256-bound index files protect all occurrences and representative locators.
    Live publication/source verification belongs to extract_to_store's session.
    Legacy stores have no such binding and keep their existing verification path.
    """
    try:
        binding = validate_store_binding(metadata)
        if raw.dtype != np.float32 or normalized.dtype != np.float32:
            return False
        if set(metadata["source_index_checksums"]) != {"content_index.parquet", "record_index.parquet"}:
            return False
        if any(sha256_file(feature_dir / name) != digest for name, digest in metadata["source_index_checksums"].items()):
            return False
        if (
            dataset_id_from_manifest(records) != binding.dataset_id
            or not records.manifest_version.eq(binding.manifest_version).all()
            or records.label_exists.isna().any() or records.label_exists.any()
            or not records.label_sha256.eq("").all()
            or not records.original_split.eq("").all()
            or not records.content_id.eq(records.image_sha256).all()
            or not content.content_id.eq(content.image_sha256).all()
            or metadata.get("total_records") != len(records)
            or metadata.get("unique_content_ids") != records.content_id.nunique()
            or metadata.get("selected_content_ids") != len(content)
            or raw.shape != (len(content), metadata.get("embedding_dimension"))
        ):
            return False
        signature = metadata["cache_signature"]
        if (
            signature.get("dataset_id") != binding.dataset_id
            or signature.get("feature_space_id") != metadata.get("feature_space_id")
            or signature.get("content_ids") != content.content_id.tolist()
            or signature.get("image_sha256") != content.image_sha256.tolist()
        ):
            return False
        for index in (content, records):
            if (
                index[list(_PHYSICAL_COLUMNS)].isna().any().any()
                or not index.source_type.eq("zip_collection").all()
                or not pd.api.types.is_integer_dtype(index.member_ordinal)
                or not index.member_ordinal.ge(0).all()
                or not index.entry_id.is_unique
                or index.duplicated(["archive_key", "member_ordinal"]).any()
            ):
                return False
            for key, archive in index[["archive_key", "source_archive"]].drop_duplicates().itertuples(index=False, name=None):
                if key not in binding.archive_keys or binding.sources[f"archive:{key}"].path != archive:
                    return False
        representatives = records.set_index("frame_id").loc[content.representative_frame_id]
        for column in ("content_id", "image_sha256", *_PHYSICAL_COLUMNS):
            if representatives[column].tolist() != content[column].tolist():
                return False
        return True
    except (ValueError, TypeError, KeyError, AttributeError, OSError):
        return False


@contextmanager
def _feature_arrays(feature_dir: Path):
    """Release read-only array handles even when QA or publication fails."""
    with ExitStack() as resources:
        raw = np.load(feature_dir / "embeddings_raw.npy", mmap_mode="r")
        resources.callback(raw._mmap.close)
        normalized = np.load(feature_dir / "embeddings_l2.npy", mmap_mode="r")
        resources.callback(normalized._mmap.close)
        yield raw, normalized


def verify_feature_directory(feature_dir: Path) -> dict:
    """Verify arrays and indexes with bounded, exception-safe read-only handles."""
    with _feature_arrays(feature_dir) as (raw, normalized):
        return _verify_feature_directory(feature_dir, raw, normalized)


def _verify_feature_directory(feature_dir: Path, raw: np.ndarray, normalized: np.ndarray) -> dict:
    """Verify arrays, indexes and quality invariants for one feature directory."""
    content_index = pd.read_parquet(feature_dir / "content_index.parquet")
    record_index = pd.read_parquet(feature_dir / "record_index.parquet")
    result = _quality(raw, normalized, content_index, record_index)
    result["metadata_exists"] = (feature_dir / "metadata.json").is_file()
    result["quality_valid"] = result["quality_valid"] and result["metadata_exists"]
    if result["metadata_exists"]:
        from flir_pipeline.data.variants import validate_variant

        metadata = json.loads((feature_dir / "metadata.json").read_text(encoding="utf-8"))
        variant = metadata.get("dataset_variant")
        signature = metadata.get("cache_signature", {})
        try:
            if variant is not None:
                validate_variant(variant, metadata.get("dataset_id"))
            result["dataset_variant_valid"] = (
                "cache_signature" not in metadata or signature.get("dataset_variant") == variant
            )
        except (ValueError, TypeError):
            result["dataset_variant_valid"] = False
        result["quality_valid"] = result["quality_valid"] and result["dataset_variant_valid"]
        if (
            "source_binding" in metadata or "source_binding" in signature
            or "source_index_checksums" in metadata
            or metadata.get("image_source_type") == "zip_collection"
        ):
            result["source_binding_valid"] = _bound_store_valid(
                feature_dir, metadata, raw, normalized, content_index, record_index
            )
            result["quality_valid"] = result["quality_valid"] and result["source_binding_valid"]
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
