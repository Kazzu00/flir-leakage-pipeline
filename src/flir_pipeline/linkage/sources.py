"""Read-only feature pairing and source-bound sequence occurrence lineage."""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from flir_pipeline.data.identity import dataset_id_from_manifest
from flir_pipeline.data.video_temporal import audit_video_temporal_lineage
from flir_pipeline.features.storage import (
    feature_space_id,
    verify_features_against_manifest,
)
from flir_pipeline.sequences.base import SEMANTICS as SEQUENCE_SEMANTICS
from flir_pipeline.sequences.base import SequenceConfig
from flir_pipeline.sequences.construction import (
    check_partition_invariants,
    construct_tables,
)
from flir_pipeline.sequences.storage import (
    FEATURE_FILES,
    SequenceSources,
    detection_identity,
)
from flir_pipeline.sequences.validation import (
    REVIEW_COLUMNS,
    VALIDATION_FILES,
    ReviewMetadata,
    ReviewRow,
)
from flir_pipeline.similarity.storage import file_sha256, read_json, stable_id

SEQUENCE_TABLES = (
    "temporal_scores",
    "candidate_events",
    "manual_review",
    "boundaries",
    "sequence_instances",
    "occurrence_assignments",
    "dependency_edges",
    "dependency_support",
)
SEQUENCE_FILES = (
    "metadata.json",
    "summary.json",
    *(f"{name}.parquet" for name in SEQUENCE_TABLES),
)


@dataclass(frozen=True)
class InputPaths:
    labeled_manifest: Path
    video_manifest: Path
    labeled_clip: Path
    labeled_dinov2: Path
    video_clip: Path
    video_dinov2: Path
    sequence_set: Path


@dataclass
class FeatureInput:
    ids: list[str]
    vectors: np.ndarray
    records: pd.DataFrame
    space: dict


@dataclass
class LinkageSources:
    paths: InputPaths
    labeled_records: pd.DataFrame
    video_records: pd.DataFrame
    features: dict[str, FeatureInput]
    signature: dict


def source_checksums(paths: InputPaths) -> dict:
    """Fixed filenames, never paths supplied by untrusted artifact metadata."""
    return {
        "labeled_manifest": file_sha256(paths.labeled_manifest),
        "video_manifest": file_sha256(paths.video_manifest),
        "features": {
            role: {
                name: file_sha256(getattr(paths, role) / name) for name in FEATURE_FILES
            }
            for role in ("labeled_clip", "labeled_dinov2", "video_clip", "video_dinov2")
        },
        "sequence_set": {
            name: file_sha256(paths.sequence_set / name) for name in SEQUENCE_FILES
        },
    }


def _feature(directory: Path, manifest: pd.DataFrame, encoder: str) -> FeatureInput:
    index = pd.read_parquet(directory / "content_index.parquet")
    records = pd.read_parquet(directory / "record_index.parquet")
    if not {"content_id", "embedding_row"} <= set(index) or not {
        "frame_id",
        "content_id",
        "embedding_row",
    } <= set(records):
        raise ValueError("Missing feature content/record-index identity columns")
    ids = sorted(manifest.content_id.unique().tolist())
    if (
        not index.content_id.is_unique
        or index.content_id.isna().any()
        or set(index.content_id) != set(ids)
    ):
        raise ValueError(
            "Feature content index must cover exactly the manifest unique content IDs"
        )
    for frame in (index, records):
        if (
            not pd.api.types.is_integer_dtype(frame.embedding_row.dtype)
            or frame.embedding_row.isna().any()
            or frame.embedding_row.lt(0).any()
        ):
            raise ValueError("Feature embedding rows must be nonnegative integers")
    vectors = np.load(
        directory / "embeddings_l2.npy", mmap_mode="r", allow_pickle=False
    )
    if (
        vectors.ndim != 2
        or vectors.shape[0] != len(ids)
        or vectors.shape[1] < 1
        or vectors.dtype != np.float32
    ):
        raise ValueError(
            "Expected a complete float32 (content_count, dimension) L2 array"
        )
    if not verify_features_against_manifest(directory, manifest)[
        "reproducible_full_dataset_valid"
    ]:
        raise ValueError(
            "Linkage requires complete valid raw/L2 features and a resolved model revision"
        )
    meta = read_json(directory / "metadata.json")
    pooling = "projected_pooler_output" if encoder == "clip" else "cls_token"
    if meta.get("extractor") != encoder or meta.get("pooling_strategy") != pooling:
        raise ValueError(
            f"Expected original {encoder} image representation with {pooling}"
        )
    config = {
        name: meta[name]
        for name in (
            "extractor",
            "model_id",
            "model_revision",
            "pooling_strategy",
            "preprocessing",
        )
    }
    config["normalization_policy"] = "raw_and_l2_float32"
    if meta["feature_space_id"] != feature_space_id(config):
        raise ValueError("Feature-space ID does not match mathematical configuration")
    rows = index.set_index("content_id").embedding_row.loc[ids].to_numpy(dtype=np.int64)
    # Each store may order its contents differently; align by IDs, never row position
    # from another dataset or encoder. O(N_content × D), not cross-pair storage.
    return FeatureInput(
        ids,
        vectors[rows],
        records,
        {
            "feature_space_id": meta["feature_space_id"],
            "feature_config": config,
            "embedding_dimension": vectors.shape[1],
        },
    )


def _labeled_manifest(path):
    manifest = pd.read_parquet(path)
    required = {
        "manifest_version",
        "frame_id",
        "content_id",
        "image_sha256",
        "label_sha256",
        "original_split",
    }
    if manifest.empty or not required <= set(manifest):
        raise ValueError(
            "Expected a nonempty canonical labeled manifest with occurrence/annotation lineage"
        )
    if not manifest.manifest_version.eq("flir_canonical_candidate_v1").all():
        raise ValueError("Labeled source requires flir_canonical_candidate_v1")
    for name in ("frame_id", "content_id", "image_sha256"):
        if (
            not manifest[name]
            .map(lambda value: isinstance(value, str) and bool(value.strip()))
            .all()
        ):
            raise ValueError(f"Invalid labeled identity: {name}")
    if (
        not manifest.frame_id.is_unique
        or not manifest.content_id.eq(manifest.image_sha256).all()
        or not manifest.original_split.isin(["train", "val", "test"]).all()
    ):
        raise ValueError("Invalid labeled occurrence/content/historical lineage")
    dataset_id_from_manifest(manifest)
    return manifest


def _sequence_records(paths, video_manifest, features, checksums):
    """Validate the existing publication as a source; never infer new sequences.

    This consumer checks all source/output hashes and reconstructs the committed
    partition from the stored confirmed review. The original external CSVs are
    not CLI inputs here; full boundary-detector/review verification remains the
    responsibility of `sequences verify` with its original validation directory.
    """
    directory = paths.sequence_set
    meta = read_json(directory / "metadata.json")
    identity = meta["identity"]
    if (
        meta["artifact_kind"] != "sequence_instance_set"
        or identity["artifact_kind"] != "sequence_instance_set"
        or meta["semantics"] != SEQUENCE_SEMANTICS
        or stable_id(identity) != meta["artifact_id"]
    ):
        raise ValueError("Expected an immutable reviewed sequence-instance set")
    required = set(SEQUENCE_FILES) - {"metadata.json"}
    if set(meta["output_checksums"]) != required or any(
        meta["output_checksums"][name] != checksums["sequence_set"][name]
        for name in required
    ):
        raise ValueError(
            "Sequence-set output checksums do not match its completed publication"
        )
    config = SequenceConfig.model_validate(identity["detection"]["config"])
    records = (
        audit_video_temporal_lineage(video_manifest)
        .sort_values(["video_id", "sample_index"])
        .reset_index(drop=True)
    )
    if not records.sample_fps.eq(1.0).all():
        raise ValueError("Reviewed sequence v1 requires a 1 FPS grid")
    for _, group in records.groupby("video_id", sort=False):
        if not np.array_equal(group.sample_index, np.arange(len(group))):
            raise ValueError("Sequence source requires contiguous occurrence grids")
    for encoder in ("clip", "dinov2"):
        mapping = (
            features[f"video_{encoder}"].records.set_index("frame_id").embedding_row
        )
        records[f"{encoder}_embedding_row"] = records.frame_id.map(mapping).astype(
            "int64"
        )
    signature = {
        "dataset_id": dataset_id_from_manifest(video_manifest),
        "manifest_sha256": checksums["video_manifest"],
        "feature_files": {
            encoder: checksums["features"][f"video_{encoder}"]
            for encoder in ("clip", "dinov2")
        },
        "feature_spaces": {
            encoder: features[f"video_{encoder}"].space
            for encoder in ("clip", "dinov2")
        },
    }
    source = SequenceSources(
        records,
        {},
        signature,
        (paths.video_manifest, paths.video_clip, paths.video_dinov2),
    )
    detection = detection_identity(source, config)
    if identity["detection"] != detection or identity["detection_id"] != stable_id(
        detection
    ):
        raise ValueError(
            "Sequence set is not bound to the supplied video manifest and feature stores"
        )
    validation = identity["validation"]
    review_meta = ReviewMetadata.model_validate(validation["metadata"])
    if (
        set(validation["files"]) != set(VALIDATION_FILES)
        or any(
            re.fullmatch(r"[0-9a-f]{64}", value) is None
            for value in validation["files"].values()
        )
        or hashlib.sha256(
            json.dumps(
                validation["files"], sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        != validation["validation_sha256"]
    ):
        raise ValueError("Invalid frozen confirmed-review checksum binding")
    # Read only the partition-side tables. Temporal scores are checksum-bound;
    # this is deliberately not another boundary detection run.
    tables = {
        name: pd.read_parquet(directory / f"{name}.parquet")
        for name in SEQUENCE_TABLES
        if name != "temporal_scores"
    }
    review = tables["manual_review"]
    if list(review) != list(REVIEW_COLUMNS) or not review.event_id.is_unique:
        raise ValueError("Invalid frozen manual-review schema or duplicate events")
    for row in review.to_dict("records"):
        if pd.isna(row["repeat_partner_sample_index"]):
            row["repeat_partner_sample_index"] = None
        ReviewRow.model_validate(row)
    if (
        review_meta.event_count != len(review)
        or review_meta.accepted_count != int(review.decision.eq("accept").sum())
        or review_meta.rejected_count != int(review.decision.eq("reject").sum())
        or review_meta.boundary_type_counts
        != review.boundary_type.value_counts().to_dict()
    ):
        raise ValueError(
            "Frozen review metadata counts disagree with its authoritative snapshot"
        )
    expected = construct_tables(
        records, tables["candidate_events"], review, validation, meta["artifact_id"]
    )
    for name, frame in expected.items():
        pd.testing.assert_frame_equal(tables[name], frame, check_exact=True)
    check_partition_invariants(records, tables, meta["artifact_id"])
    return tables["occurrence_assignments"], meta["artifact_id"]


def load_sources(paths: InputPaths) -> LinkageSources:
    initial = source_checksums(paths)
    labeled = _labeled_manifest(paths.labeled_manifest)
    video = pd.read_parquet(paths.video_manifest)
    audit_video_temporal_lineage(video)
    features = {
        f"{dataset}_{encoder}": _feature(
            getattr(paths, f"{dataset}_{encoder}"), manifest, encoder
        )
        for dataset, manifest in (("labeled", labeled), ("video", video))
        for encoder in ("clip", "dinov2")
    }
    for encoder in ("clip", "dinov2"):
        if features[f"labeled_{encoder}"].space != features[f"video_{encoder}"].space:
            raise ValueError(
                f"Mismatched {encoder} mathematical feature spaces across datasets"
            )
    occurrences, sequence_id = _sequence_records(paths, video, features, initial)
    signature = {
        "checksums": initial,
        "labeled_dataset_id": dataset_id_from_manifest(labeled),
        "video_dataset_id": dataset_id_from_manifest(video),
        "feature_spaces": {
            encoder: features[f"video_{encoder}"].space
            for encoder in ("clip", "dinov2")
        },
        "sequence_set_id": sequence_id,
    }
    if source_checksums(paths) != initial:
        raise ValueError("Linkage sources changed while loading; use immutable inputs")
    return LinkageSources(
        paths,
        labeled.sort_values("frame_id").reset_index(drop=True),
        occurrences,
        features,
        signature,
    )


def assert_sources_unchanged(source: LinkageSources):
    if source_checksums(source.paths) != source.signature["checksums"]:
        raise ValueError("Linkage sources changed during execution")
