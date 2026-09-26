"""Source-bound content assignments and exact, reproducible posterior evaluation."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

from flir_pipeline.clustering.algorithms import (
    effective_parameters,
    fit_clustering,
    implementation_versions,
)
from flir_pipeline.clustering.base import ClusteringConfig, clustering_space_id
from flir_pipeline.clustering.metrics import (
    EvaluationContext,
    evaluate_clustering,
    validate_labels,
)
from flir_pipeline.reduction.benchmark import verify_benchmark
from flir_pipeline.reduction.storage import ReductionInputs, load_inputs
from flir_pipeline.similarity.storage import (
    execution_provenance,
    file_sha256,
    read_json,
    write_json,
)

RUN_FILES = ("cluster_labels.npy", "content_index.parquet", "cluster_summary.parquet", "metrics.json", "quality.json")
VIDEO_EVALUATION_PROTOCOL = "exact_original_euclidean_cosine_video_unknown_sequences_v2"
VIDEO_UNAVAILABLE_METRICS = ["sequence_coherence", "sequence_temporal_recall", "historical_splits"]


def clustering_provenance() -> dict:
    source = execution_provenance()
    source["similarity_source_sha256"] = source.pop("source_sha256")
    source["source_sha256"] = {p.name: file_sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    return source


@dataclass
class ClusterSpace:
    representation: str
    seed: int | None
    reduction_space_id: str | None
    values: np.ndarray
    distances: np.ndarray
    signatures: dict


@dataclass
class ClusteringFamily:
    source: ReductionInputs
    context: EvaluationContext
    spaces: dict[tuple[str, int | None], ClusterSpace]
    benchmark_directory: Path


def load_family(feature_directory: Path, similarity_directory: Path,
                manifest_path: Path, reduction_benchmark: Path) -> ClusteringFamily:
    """Verify full features/similarity and the selected reduction family across seeds."""
    source = load_inputs(feature_directory, similarity_directory, manifest_path)
    if not verify_benchmark(reduction_benchmark)["quality_valid"]:
        raise ValueError("Reduction benchmark failed verification")
    meta = read_json(reduction_benchmark/"metadata.json")
    if meta["input_signatures"] != source.signatures or any(meta[k] != source.feature[k] for k in ("dataset_id", "feature_space_id", "extractor")):
        raise ValueError("Reduction benchmark does not match the full source space")
    original_distances = squareform(pdist(source.embeddings.astype(np.float64), metric="euclidean"))
    context = EvaluationContext.create(source.content_index.content_id.tolist(), original_distances,
                                       source.cosine, source.original_neighbors,
                                       pd.read_parquet(similarity_directory/"content_provenance.parquet"))
    spaces = {("original_l2", None): ClusterSpace("original_l2", None, None, source.embeddings, original_distances, source.signatures)}
    candidates, runs = pd.read_csv(reduction_benchmark/"candidates.csv"), pd.read_csv(reduction_benchmark/"runs.csv")
    if set(candidates.method) != {"tsne", "pacmap"} or len(candidates) != 2:
        raise ValueError("One committed reduction candidate per method is required")
    for candidate in candidates.itertuples():
        selected = runs.loc[(runs.method == candidate.method) & (runs.configuration_id == candidate.configuration_id)]
        if sorted(selected.seed.tolist()) != [0, 1, 2]:
            raise ValueError("Reduction candidate requires seeds 0/1/2")
        for row in selected.itertuples():
            directory = reduction_benchmark.parents[1]/row.method/row.reduction_space_id
            run_meta = read_json(directory/"metadata.json")
            if run_meta["input_signatures"] != source.signatures or not pd.read_parquet(directory/"content_index.parquet").equals(source.content_index):
                raise ValueError("Selected reduction content order/source does not match")
            values = np.load(directory/"coordinates.npy", mmap_mode="r", allow_pickle=False)
            if values.shape != (len(source.content_index), 2) or not np.isfinite(values).all():
                raise ValueError("The current clustering protocol requires full 2D reduction coordinates")
            signatures = {**source.signatures, "reduction": {p: file_sha256(directory/p) for p in ("coordinates.npy", "metadata.json", "content_index.parquet")}}
            spaces[(row.method, int(row.seed))] = ClusterSpace(row.method, int(row.seed), row.reduction_space_id,
                                                              values, squareform(pdist(values.astype(np.float64))), signatures)
    return ClusteringFamily(source, context, spaces, reduction_benchmark)


def quality_checks(labels: np.ndarray, n: int, probabilities: np.ndarray | None = None) -> dict:
    validate_labels(labels, n)
    quality = {"labels_shape_valid": True, "integer_labels": True, "noise_preserved": True,
               "nonnegative_cluster_ids": True, "membership_probabilities_valid": True}
    if probabilities is not None:
        quality["membership_probabilities_valid"] = bool(probabilities.shape == (n,) and np.isfinite(probabilities).all()
                                                           and ((probabilities >= 0) & (probabilities <= 1)).all()
                                                           and (probabilities[labels == -1] == 0).all())
    quality["quality_valid"] = all(quality.values())
    return quality


def run_to_store(family: ClusteringFamily, space: ClusterSpace, config: ClusteringConfig,
                 output_root: Path = Path("artifacts/clustering")) -> Path:
    parameters = effective_parameters(config, space.distances)
    versions = implementation_versions()
    feature = family.source.feature
    run_id = clustering_space_id(feature["dataset_id"], feature["feature_space_id"], space.representation,
                                 space.reduction_space_id, config, parameters, versions)
    representation_id = space.reduction_space_id or feature["feature_space_id"]
    output = output_root/feature["extractor"]/feature["dataset_id"]/representation_id/config.algorithm/run_id
    if output.exists():
        if not verify_run(output)["quality_valid"] or read_json(output/"metadata.json")["input_signatures"] != space.signatures:
            raise ValueError("Incomplete or incompatible clustering publication preserved; choose a separate root")
        return output
    result = fit_clustering(space.values, family.context.content_ids, config, parameters)
    quality = quality_checks(result.labels, len(space.values), result.probabilities)
    if not quality["quality_valid"]:
        raise ValueError("Clustering outputs failed quality checks")
    started = time.perf_counter()
    metrics, summary = evaluate_clustering(result.labels, family.context, space.distances)
    evaluation_seconds = time.perf_counter()-started
    output.mkdir(parents=True, exist_ok=False)
    np.save(output/"cluster_labels.npy", result.labels, allow_pickle=False)
    family.source.content_index.to_parquet(output/"content_index.parquet", index=False)
    summary.to_parquet(output/"cluster_summary.parquet", index=False)
    write_json(output/"metrics.json", metrics)
    write_json(output/"quality.json", quality)
    optional_files = []
    if result.probabilities is not None:
        np.save(output/"membership_probabilities.npy", result.probabilities, allow_pickle=False)
        optional_files.append("membership_probabilities.npy")
    if result.diagnostics:
        np.savez(output/"algorithm_diagnostics.npz", **result.diagnostics)
        optional_files.append("algorithm_diagnostics.npz")
    metadata = {**clustering_provenance(), **result.metadata, "artifact_kind": "clustering_run",
                "dataset_id": feature["dataset_id"], "feature_space_id": feature["feature_space_id"],
                "extractor": feature["extractor"], "model_id": feature["model_id"], "N": len(space.values),
                "input_dimension": space.values.shape[1], "representation": space.representation,
                "reduction_space_id": space.reduction_space_id, "reduction_seed": space.seed,
                "clustering_space_id": run_id, "configuration_id": config.configuration_id,
                "algorithm": config.algorithm, "config": asdict(config), "implementation_versions": versions,
                "input_signatures": space.signatures, "evaluation_seconds": evaluation_seconds,
                "evaluation_protocol": (VIDEO_EVALUATION_PROTOCOL
                                        if metrics.get("temporal_provenance_mode") == "sampled_video_grid"
                                        else "exact_original_euclidean_cosine_posterior_v1"),
                "noise_policy": "minus_one_unchanged; exclude_noise_queries; temporal_denominator_all_eligible_pairs",
                "record_mapping": "source feature record_index, fingerprint retained; no split assignment",
                "optional_files": optional_files, "output_sha256": {name: file_sha256(output/name) for name in (*RUN_FILES, *optional_files)}}
    if metrics.get("temporal_provenance_mode") == "sampled_video_grid":
        metadata.update(provenance_mode="sampled_video_grid",
                        unavailable_metric_groups=VIDEO_UNAVAILABLE_METRICS,
                        noise_policy="minus_one_unchanged; exclude_noise_queries; sequence_temporal_denominator_unavailable")
    write_json(output/"metadata.json", metadata)
    if not verify_run(output)["quality_valid"]:
        raise ValueError("Published clustering failed verification and was preserved")
    return output


def verify_run(directory: Path, family: ClusteringFamily | None = None) -> dict:
    checks = {"metadata_exists": (directory/"metadata.json").is_file()}
    try:
        meta = read_json(directory/"metadata.json")
        config = ClusteringConfig(**meta["config"])
        labels = np.load(directory/"cluster_labels.npy", allow_pickle=False)
        probabilities = np.load(directory/"membership_probabilities.npy", allow_pickle=False) if "membership_probabilities.npy" in meta["optional_files"] else None
        quality = quality_checks(labels, meta["N"], probabilities)
        checks["quality_snapshot_matches"] = quality["quality_valid"] and quality == read_json(directory/"quality.json")
        files = {*RUN_FILES, *meta["optional_files"]}
        checks["output_checksums_valid"] = files == set(meta["output_sha256"]) and all(file_sha256(directory/name) == meta["output_sha256"][name] for name in files)
        index = pd.read_parquet(directory/"content_index.parquet")
        checks["index_valid"] = bool(len(index) == meta["N"] and index.content_id.is_unique and index.content_id.notna().all() and np.array_equal(index.embedding_row, np.arange(len(index))))
        checks["identity_valid"] = meta["clustering_space_id"] == clustering_space_id(meta["dataset_id"], meta["feature_space_id"], meta["representation"], meta["reduction_space_id"], config, meta["effective_parameters"], meta["implementation_versions"])
        checks["metadata_consistent"] = meta["artifact_kind"] == "clustering_run" and meta["algorithm"] == config.algorithm and meta["configuration_id"] == config.configuration_id
        summary, metrics = pd.read_parquet(directory/"cluster_summary.parquet"), read_json(directory/"metrics.json")
        if meta.get("provenance_mode") == "sampled_video_grid" or metrics.get("temporal_provenance_mode") == "sampled_video_grid" or meta.get("evaluation_protocol") == VIDEO_EVALUATION_PROTOCOL:
            temporal_keys = [f"{name}@{k}" for name in ("temporal_recall", "temporal_pairs", "temporal_retained_pairs") for k in (1, 5, 10)]
            unknown_keys = ["weighted_dominant_sequence_fraction", "weighted_sequence_entropy_bits", "clustered_sequence_coverage", *temporal_keys]
            unknown_keys += ["historical_multisplit_clusters", "historical_unknown_clusters",
                             *(f"historical_{split}_only_clusters" for split in ("train", "val", "test"))]
            checks["video_unavailable_metrics_explicit"] = bool(
                meta.get("provenance_mode") == metrics.get("temporal_provenance_mode") == "sampled_video_grid"
                and meta.get("evaluation_protocol") == VIDEO_EVALUATION_PROTOCOL
                and meta.get("unavailable_metric_groups") == VIDEO_UNAVAILABLE_METRICS
                and meta.get("noise_policy") == "minus_one_unchanged; exclude_noise_queries; sequence_temporal_denominator_unavailable"
                and metrics.get("historical_split_evaluation") == "unavailable"
                and metrics.get("temporal_evaluation") == "sequence identity unknown; historical recall unavailable"
                and all(key in metrics and metrics[key] is None for key in unknown_keys)
                and summary.known_sequence_members.eq(0).all()
                and summary[["sequence_coverage", "sequence_count", "dominant_sequence_fraction", "sequence_entropy_bits", "historical_split_memberships", "historical_split_count"]].isna().all().all()
                and summary.historical_split_provenance_complete.eq(False).all())
        checks["summary_coverage_valid"] = (set(summary.cluster_id) == set(labels)-{-1} and summary.cluster_id.is_unique
                                            and all(r.n_members == int((labels == r.cluster_id).sum()) and r.medoid_content_id in set(index.loc[labels == r.cluster_id, "content_id"]) for r in summary.itertuples()))
        checks["population_metrics_valid"] = metrics["total_points"] == len(labels) and metrics["clustered_points"] == int((labels >= 0).sum()) and metrics["n_clusters_excluding_noise"] == len(summary)
        if family is not None:
            space = family.spaces[(meta["representation"], meta["reduction_seed"])]
            checks["source_matches"] = index.equals(family.source.content_index) and meta["input_signatures"] == space.signatures and meta["reduction_space_id"] == space.reduction_space_id
            checks["effective_parameters_recomputed"] = effective_parameters(config, space.distances) == meta["effective_parameters"]
            expected_metrics, expected_summary = evaluate_clustering(labels, family.context, space.distances)
            checks["metrics_recomputed"] = expected_metrics == metrics
            pd.testing.assert_frame_equal(expected_summary, summary, check_dtype=False)
            checks["summary_and_medoids_recomputed"] = True
        checks["quality_valid"] = all(checks.values())
    except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError, AssertionError) as error:
        checks.update(quality_valid=False, error_type=type(error).__name__)
    return checks
