"""Verified content-level reductions, immutable outputs and source-bound evaluation."""

from __future__ import annotations

import hashlib
import inspect
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

from flir_pipeline.features.storage import verify_features_against_manifest
from flir_pipeline.reduction.base import (
    ReductionConfig,
    configuration_id,
    reduction_space_id,
)
from flir_pipeline.reduction.metrics import (
    EvaluationReference,
    evaluate_coordinates,
    neighbor_order,
    neighborhood_table,
)
from flir_pipeline.reduction.reducers import implementation_versions, make_reducer
from flir_pipeline.similarity.storage import (
    FEATURE_FILES,
    execution_provenance,
    file_sha256,
    read_json,
    verify_similarity_directory,
    write_json,
)

RUN_FILES = ("coordinates.npy", "content_index.parquet", "nearest_neighbors.parquet",
             "neighborhood_preservation.parquet", "metrics.json", "quality.json", "feature_metadata.json")
SIMILARITY_INPUT_FILES = ("cosine_similarity.npy", "nearest_neighbors.parquet", "metadata.json")


def reduction_provenance() -> dict:
    previous = execution_provenance()
    previous["reused_similarity_source_sha256"] = previous.pop("source_sha256")
    previous["source_sha256"] = {p.name: file_sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    return previous


@dataclass
class ReductionInputs:
    feature_directory: Path
    similarity_directory: Path
    feature: dict
    similarity: dict
    content_index: pd.DataFrame
    embeddings: np.ndarray
    cosine: np.ndarray
    original_neighbors: pd.DataFrame
    signatures: dict

    def reference(self, config: ReductionConfig) -> EvaluationReference:
        return EvaluationReference.from_similarity(self.cosine, self.content_index.content_id.tolist(),
                                                   self.original_neighbors, config.distance_sample_size,
                                                   config.distance_sample_seed)


def load_inputs(feature_directory: Path, similarity_directory: Path, manifest_path: Path) -> ReductionInputs:
    manifest = pd.read_parquet(manifest_path)
    if not verify_features_against_manifest(feature_directory, manifest)["reproducible_full_dataset_valid"]:
        raise ValueError("Reduction requires verified full canonical feature coverage")
    if not verify_similarity_directory(similarity_directory, feature_directory, manifest_path)["quality_valid"]:
        raise ValueError("Original similarity reference must match the complete features and manifest")
    feature, similarity = read_json(feature_directory/"metadata.json"), read_json(similarity_directory/"metadata.json")
    if any(feature[key] != similarity[key] for key in ("dataset_id", "feature_space_id", "extractor")):
        raise ValueError("Feature and similarity spaces do not correspond")
    ids = pd.read_parquet(feature_directory/"content_index.parquet")
    return ReductionInputs(
        feature_directory, similarity_directory, feature, similarity, ids,
        np.load(feature_directory/"embeddings_l2.npy", mmap_mode="r", allow_pickle=False),
        np.load(similarity_directory/"cosine_similarity.npy", mmap_mode="r", allow_pickle=False),
        pd.read_parquet(similarity_directory/"nearest_neighbors.parquet"),
        {"features": {name: file_sha256(feature_directory/name) for name in FEATURE_FILES},
         "similarity": {name: file_sha256(similarity_directory/name) for name in SIMILARITY_INPUT_FILES}},
    )


def coordinate_quality(coordinates: np.ndarray, n: int, dimension: int) -> dict:
    """Detect full collapse and report rank/duplicates without requiring unique rows."""
    shape_valid = coordinates.shape == (n, dimension) and n > 1
    dtype_valid = np.issubdtype(coordinates.dtype, np.floating)
    finite = bool(np.isfinite(coordinates).all())
    if not shape_valid or not dtype_valid or not finite:
        return {"shape_valid": shape_valid, "floating_dtype": dtype_valid, "finite": finite, "quality_valid": False}
    centered = coordinates.astype(np.float64)-coordinates.mean(axis=0, dtype=np.float64)
    rank = int(np.linalg.matrix_rank(centered))
    unique = len(np.unique(coordinates, axis=0))
    return {"shape_valid": True, "floating_dtype": True, "finite": True,
            "full_collapse": rank == 0, "centered_rank": rank,
            "rank_deficient_warning": rank < dimension, "unique_coordinate_rows": unique,
            "duplicate_coordinate_rows": n-unique, "coordinate_span": np.ptp(coordinates, axis=0).tolist(),
            "quality_valid": rank > 0}


def run_to_store(inputs: ReductionInputs, config: ReductionConfig,
                 output_root: Path = Path("artifacts/reduction"),
                 reference: EvaluationReference | None = None) -> Path:
    """Fit one complete content space; finished runs are resumable grid checkpoints.

    No optimizer state is fabricated. Incomplete publications are preserved and
    refused. A failure during fitting leaves no completed reduction directory.
    """
    config.validate_population(*inputs.embeddings.shape)
    versions = implementation_versions(config.method)
    feature = inputs.feature
    run_id = reduction_space_id(feature["dataset_id"], feature["feature_space_id"], config, versions)
    output = output_root/feature["extractor"]/feature["dataset_id"]/feature["feature_space_id"]/config.method/run_id
    if output.exists():
        if not (output/"metadata.json").is_file():
            raise ValueError("Incomplete reduction preserved; use a separate output root")
        meta = read_json(output/"metadata.json")
        if meta.get("input_signatures") != inputs.signatures or not verify_reduction(output)["quality_valid"]:
            raise ValueError("Reduction cache mismatches input fingerprints or failed verification")
        return output
    reference = reference or inputs.reference(config)
    n = len(inputs.content_index)
    if (reference.content_ids != inputs.content_index.content_id.tolist()
            or reference.sample_seed != config.distance_sample_seed
            or len(reference.sample_a) != min(config.distance_sample_size, n*(n-1)//2)):
        raise ValueError("Evaluation reference does not match the configured content order or sampling")
    started = time.perf_counter()
    reducer = make_reducer(config)
    result = reducer.fit_transform(inputs.embeddings)
    backend_seconds = time.perf_counter()-started
    quality = coordinate_quality(result.coordinates, len(inputs.content_index), config.output_dimension)
    if not quality["quality_valid"]:
        raise ValueError("Reducer produced invalid or completely collapsed coordinates")
    evaluation_start = time.perf_counter()
    metrics, neighbors, preservation = evaluate_coordinates(result.coordinates, reference, config.evaluation_ks)
    evaluation_seconds = time.perf_counter()-evaluation_start
    source_fields = ("dataset_id", "feature_space_id", "extractor", "model_id", "model_revision", "resolved_model_revision", "embedding_dimension", "pooling_strategy")
    snapshot = {key: feature[key] for key in source_fields}
    output.mkdir(parents=True, exist_ok=False)
    np.save(output/"coordinates.npy", result.coordinates, allow_pickle=False)
    inputs.content_index.to_parquet(output/"content_index.parquet", index=False)
    neighbors.to_parquet(output/"nearest_neighbors.parquet", index=False)
    preservation.to_parquet(output/"neighborhood_preservation.parquet", index=False)
    write_json(output/"metrics.json", metrics)
    write_json(output/"quality.json", quality)
    write_json(output/"feature_metadata.json", snapshot)
    library_module = inspect.getmodule(reducer._fit)
    metadata = {
        **snapshot, **reduction_provenance(), **result.metadata,
        "artifact_kind": "reduction_run", "method": config.method, "N": len(inputs.content_index),
        "input_dimension": inputs.embeddings.shape[1], "output_dimension": config.output_dimension,
        "coordinate_dtype": str(result.coordinates.dtype), "seed": config.seed,
        "input_representation": config.input_representation, "preprocessing": config.preprocessing,
        "hyperparameters": config.hyperparameters, "config": asdict(config),
        "reduction_space_id": run_id, "configuration_id": configuration_id(config),
        "library": "scikit-learn" if config.method == "tsne" else "pacmap",
        "library_version": versions["scikit-learn" if config.method == "tsne" else "pacmap"],
        "implementation_versions": versions, "adapter_source_sha256": file_sha256(Path(library_module.__file__)),
        "reference_similarity_space_id": inputs.similarity["similarity_space_id"],
        "input_signatures": inputs.signatures, "backend_total_seconds": backend_seconds,
        "evaluation_seconds": evaluation_seconds,
        "distance_sample_sha256": hashlib.sha256(np.column_stack([reference.sample_a, reference.sample_b]).astype(np.int64).tobytes()).hexdigest(),
        "output_sha256": {name: file_sha256(output/name) for name in RUN_FILES},
    }
    write_json(output/"metadata.json", metadata)
    if not verify_reduction(output)["quality_valid"]:
        raise ValueError("Written reduction failed verification; retained for inspection")
    return output


def verify_reduction(directory: Path, inputs: ReductionInputs | None = None,
                     reference: EvaluationReference | None = None) -> dict:
    """Check coordinates, identities, ranks and checksums; optionally recompute metrics."""
    checks = {"metadata_exists": (directory/"metadata.json").is_file()}
    try:
        meta = read_json(directory/"metadata.json")
        config = ReductionConfig(**meta["config"])
        ids = pd.read_parquet(directory/"content_index.parquet")
        coords = np.load(directory/"coordinates.npy", allow_pickle=False)
        source = read_json(directory/"feature_metadata.json")
        quality = coordinate_quality(coords, meta["N"], config.output_dimension)
        checks["coordinates_valid"] = quality["quality_valid"]
        checks["quality_snapshot_matches"] = quality == read_json(directory/"quality.json")
        checks["metadata_consistent"] = (meta["artifact_kind"] == "reduction_run" and meta["method"] == config.method
                                         and meta["seed"] == config.seed and meta["hyperparameters"] == config.hyperparameters
                                         and meta["preprocessing"] == config.preprocessing and meta["input_representation"] == config.input_representation
                                         and meta["output_dimension"] == config.output_dimension and meta["coordinate_dtype"] == str(coords.dtype)
                                         and meta["input_dimension"] == source["embedding_dimension"] and all(meta.get(k) == v for k, v in source.items()))
        checks["index_valid"] = bool(len(ids) == meta["N"] and ids.content_id.is_unique and ids.content_id.notna().all() and np.array_equal(ids.embedding_row, np.arange(len(ids))))
        checks["identity_valid"] = meta["reduction_space_id"] == reduction_space_id(meta["dataset_id"], meta["feature_space_id"], config, meta["implementation_versions"]) and meta["configuration_id"] == configuration_id(config)
        checks["output_checksums_valid"] = set(meta["output_sha256"]) == set(RUN_FILES) and all(file_sha256(directory/name) == meta["output_sha256"][name] for name in RUN_FILES)
        distances = squareform(pdist(coords.astype(np.float64)))
        ordered = neighbor_order(distances, ids.content_id.tolist())
        expected = neighborhood_table(ordered, ids.content_id.tolist(), distances, max(config.evaluation_ks))
        actual = pd.read_parquet(directory/"nearest_neighbors.parquet")
        checks["neighbors_valid"] = expected.equals(actual)
        metrics = read_json(directory/"metrics.json")
        checks["metrics_valid"] = all(isinstance(metrics[f"{label}@{k}"], (float, int)) and 0 <= metrics[f"{label}@{k}"] <= 1 for k in config.evaluation_ks for label in ("trustworthiness", "continuity"))
        if inputs is not None:
            checks["source_inputs_match"] = meta["input_signatures"] == inputs.signatures and ids.equals(inputs.content_index)
            checks["source_identities_match"] = all(meta[key] == inputs.feature[key] for key in ("dataset_id", "feature_space_id", "extractor"))
            reference = reference or inputs.reference(config)
            expected_metrics, _, expected_preservation = evaluate_coordinates(coords, reference, config.evaluation_ks)
            checks["metrics_recomputed"] = metrics == expected_metrics
            checks["preservation_recomputed"] = expected_preservation.equals(pd.read_parquet(directory/"neighborhood_preservation.parquet"))
            checks["pair_sample_matches"] = meta["distance_sample_sha256"] == hashlib.sha256(np.column_stack([reference.sample_a, reference.sample_b]).astype(np.int64).tobytes()).hexdigest()
        checks["quality_valid"] = all(checks.values())
    except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError) as error:
        checks.update(quality_valid=False, error_type=type(error).__name__)
    return checks
