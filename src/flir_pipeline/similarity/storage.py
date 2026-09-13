"""Versioned local cosine experiments, source fingerprints and executable QA."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from flir_pipeline.data.identity import dataset_id_from_manifest
from flir_pipeline.features.storage import verify_features_against_manifest
from flir_pipeline.similarity.cosine import (
    compute_cosine_similarity,
    compute_topk_neighbors,
    distribution_summary,
    matrix_quality,
    summarize_topk,
)
from flir_pipeline.similarity.temporal import (
    analyze_temporal_neighbors,
    annotate_pairs,
    build_content_provenance,
    summarize_pair_relations,
)


@dataclass(frozen=True)
class SimilarityConfig:
    """Only scientific/analysis settings enter the reproducible experiment ID."""

    metric: str = "cosine"
    top_k: int = 20
    pairwise_dtype: str = "float32"
    quantiles: tuple[float, ...] = (.9, .95, .975, .99, .995, .999)
    frame_delta_upper_bounds: tuple[int, ...] = (0, 1, 5, 10, 25, 50, 100)
    numerical_atol: float = 1e-5
    near_unit_atol: float = 1e-6
    algorithm_version: str = "content_cosine_v1"
    tie_break: str = "content_id_ascending"
    temporal_rule: str = "all_occurrences_consensus_archive_sequence_index"
    cross_split_rule: str = "exists_known_unequal_occurrence_splits"

    def __post_init__(self):
        object.__setattr__(self, "quantiles", tuple(self.quantiles))
        object.__setattr__(self, "frame_delta_upper_bounds", tuple(self.frame_delta_upper_bounds))
        if self.metric != "cosine" or self.pairwise_dtype != "float32":
            raise ValueError("Only cosine over float32 L2 embeddings is implemented")
        if type(self.top_k) is not int or self.top_k < 1:
            raise ValueError("top_k must be a positive integer")
        if not self.quantiles or tuple(sorted(set(self.quantiles))) != self.quantiles or not all(0 < q < 1 for q in self.quantiles):
            raise ValueError("quantiles must be strictly increasing values in (0,1)")
        if not self.frame_delta_upper_bounds or self.frame_delta_upper_bounds[0] != 0 or tuple(sorted(set(self.frame_delta_upper_bounds))) != self.frame_delta_upper_bounds or not all(type(v) is int and v >= 0 for v in self.frame_delta_upper_bounds):
            raise ValueError("frame delta bounds must be increasing nonnegative integers starting at 0")
        if not 0 < self.numerical_atol <= 1e-4 or not 0 < self.near_unit_atol <= self.numerical_atol:
            raise ValueError("Invalid numerical or near-unit diagnostic tolerance")
        fixed = (self.algorithm_version, self.tie_break, self.temporal_rule, self.cross_split_rule)
        if fixed != ("content_cosine_v1", "content_id_ascending", "all_occurrences_consensus_archive_sequence_index", "exists_known_unequal_occurrence_splits"):
            raise ValueError("Unsupported analysis rule/version")


def stable_id(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()[:16]


def similarity_space_id(dataset_id: str, feature_space_id: str, config: SimilarityConfig) -> str:
    """Identify a dataset/feature/analysis configuration independent of runtime paths."""
    return stable_id({"dataset_id": dataset_id, "feature_space_id": feature_space_id, "config": asdict(config)})


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def execution_provenance() -> dict:
    """Record the actual worktree, including source fingerprints before a commit."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "unknown", None
    return {"git_commit": commit, "git_worktree_dirty": dirty,
            "python_version": platform.python_version(), "numpy_version": np.__version__,
            "pandas_version": pd.__version__, "created_at": datetime.now(UTC).isoformat(),
            "source_sha256": {p.name: file_sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))}}


TABLE_NAMES = ("sequence_similarity", "quantile_candidates", "frame_delta_similarity", "historical_split_similarity", "topk_global_summary")
ARTIFACT_NAMES = ("cosine_similarity.npy", "nearest_neighbors.parquet", "content_index.parquet", "content_provenance.parquet", "record_provenance.parquet", "pair_analysis.parquet", "topk_content_summary.parquet", "near_unit_pairs.parquet", "similarity_summary.json", "feature_metadata.json", "quality.json", *(f"{name}.csv" for name in TABLE_NAMES))
FEATURE_FILES = ("embeddings_l2.npy", "content_index.parquet", "record_index.parquet", "metadata.json")
PROVENANCE_COLUMNS = ("frame_id", "content_id", "source_archive", "source_member_path", "original_split", "possible_sequence", "possible_frame_index", "temporal_inference_confidence")


def source_signature(feature_directory: Path, manifest: pd.DataFrame) -> dict:
    # Temporal/split changes need cache invalidation even though they never enter
    # dot products and the existing dataset identity does not encode that metadata.
    relevant = manifest.reindex(columns=PROVENANCE_COLUMNS).sort_values("frame_id")
    return {"feature_files": {name: file_sha256(feature_directory/name) for name in FEATURE_FILES},
            "posthoc_manifest_sha256": hashlib.sha256(relevant.to_json(orient="records").encode()).hexdigest()}


def compute_to_store(feature_directory: Path, manifest_path: Path, config: SimilarityConfig,
                     output_root: Path = Path("artifacts/similarity")) -> Path:
    """Verify full features, compute once per content, then join posterior metadata.

    Completed caches must match input fingerprints and pass QA before reuse.
    Partial directories are preserved and refused. One writer per experiment;
    metadata is the completion marker written after all outputs are saved.
    """
    manifest = pd.read_parquet(manifest_path)
    if not verify_features_against_manifest(feature_directory, manifest)["reproducible_full_dataset_valid"]:
        raise ValueError("Similarity requires a verified complete feature space and resolved revision")
    source = read_json(feature_directory/"metadata.json")
    config_id = similarity_space_id(source["dataset_id"], source["feature_space_id"], config)
    output = output_root/source["extractor"]/source["dataset_id"]/source["feature_space_id"]/config_id
    signature = source_signature(feature_directory, manifest)
    if output.exists():
        if not (output/"metadata.json").is_file():
            raise ValueError("Incomplete similarity directory preserved; use a separate output root")
        metadata = read_json(output/"metadata.json")
        if metadata.get("input_signature") != signature or not verify_similarity_directory(output)["quality_valid"]:
            raise ValueError("Existing similarity cache mismatches inputs or failed QA; preserved")
        return output
    content_index = pd.read_parquet(feature_directory/"content_index.parquet")
    n = len(content_index)
    if config.top_k >= n:
        raise ValueError("top_k must be smaller than the content count")
    # Numerical inputs stop here: one original L2 vector per unique content.
    embeddings = np.load(feature_directory/"embeddings_l2.npy", mmap_mode="r", allow_pickle=False)
    matrix = compute_cosine_similarity(embeddings, config.numerical_atol)
    neighbors = compute_topk_neighbors(matrix, content_index.content_id.tolist(), config.top_k)
    a, b = np.triu_indices(n, k=1)
    pairs = pd.DataFrame({"query_row": a.astype(np.int32), "neighbor_row": b.astype(np.int32), "cosine_similarity": matrix[a, b]})
    expected_pairs = n*(n-1)//2
    assert len(pairs) == expected_pairs
    # Temporal lineage and split sets enter only after similarities/ranks exist.
    contents, records = build_content_provenance(manifest, content_index)
    pairs = annotate_pairs(pairs, contents)
    neighbors = annotate_pairs(neighbors, contents)
    for side, index_col in (("query", "query_row"), ("neighbor", "neighbor_row")):
        neighbors[f"{side}_split_membership_set"] = contents.split_membership_set.to_numpy()[neighbors[index_col]]
    per_content, topk_summary = summarize_topk(neighbors)
    tables = summarize_pair_relations(pairs, config.quantiles, config.frame_delta_upper_bounds)
    tables["topk_global_summary"] = topk_summary
    near_unit = pairs.loc[np.abs(pairs.cosine_similarity-1) <= config.near_unit_atol].copy()
    for side, index_col in (("query", "query_row"), ("neighbor", "neighbor_row")):
        for key in ("content_id", "image_sha256"):
            near_unit[f"{side}_{key}"] = contents[key].to_numpy()[near_unit[index_col]]
    quality = {**matrix_quality(matrix, n, config.numerical_atol), "unique_pair_count": len(pairs),
               "expected_unique_pair_count": expected_pairs, "neighbor_count": len(neighbors)}
    summary = {
        "global_similarity": distribution_summary(pairs.cosine_similarity.to_numpy()),
        "temporal_neighbors": analyze_temporal_neighbors(neighbors),
        "temporal_coverage": {"contents": n, "sequence_known": int(contents.sequence_provenance_valid.sum()),
                              "frame_index_known": int(contents.frame_index_valid.sum()),
                              "ambiguous_or_unknown_sequence": int((~contents.sequence_provenance_valid).sum()),
                              "verified_timestamps": 0, "same_sequence_pairs_with_delta": int(pairs.frame_delta.notna().sum()),
                              "unknown_sequence_pairs": int(pairs.same_sequence.isna().sum())},
        "historical_membership": contents.split_membership_set.value_counts().to_dict(),
        "near_unit_pair_count": len(near_unit),
        "near_unit_atol": config.near_unit_atol,
        "std_ddof": 0, "quantile_method": "linear", "quantile_selection": "cosine >= threshold; retain ties",
        "pair_unit": "unordered distinct content pair, i<j", "neighbor_unit": "directed content-to-content edge",
    }
    source_fields = ("dataset_id", "feature_space_id", "extractor", "model_id", "model_revision", "resolved_model_revision", "embedding_dimension", "pooling_strategy")
    feature_snapshot = {key: source[key] for key in source_fields}
    output.mkdir(parents=True, exist_ok=False)
    np.save(output/"cosine_similarity.npy", matrix, allow_pickle=False)
    for filename, frame in (("nearest_neighbors", neighbors), ("content_index", content_index), ("content_provenance", contents),
                            ("record_provenance", records), ("pair_analysis", pairs), ("topk_content_summary", per_content), ("near_unit_pairs", near_unit)):
        frame.to_parquet(output/f"{filename}.parquet", index=False)
    for name, table in tables.items():
        table.to_csv(output/f"{name}.csv", index=False)
    write_json(output/"feature_metadata.json", feature_snapshot)
    write_json(output/"similarity_summary.json", summary)
    write_json(output/"quality.json", quality)
    metadata = {**feature_snapshot, **execution_provenance(), "similarity_space_id": config_id,
                "metric": config.metric, "top_k": config.top_k, "content_count": n,
                "config": asdict(config), "input_signature": signature,
                "output_sha256": {name: file_sha256(output/name) for name in ARTIFACT_NAMES}}
    write_json(output/"metadata.json", metadata)
    if not verify_similarity_directory(output)["quality_valid"]:
        raise ValueError("Written similarity artifacts failed verification; preserved for inspection")
    return output


def verify_similarity_directory(directory: Path, feature_directory: Path | None = None,
                                manifest_path: Path | None = None) -> dict:
    """Recompute QA, including actual top-k ranks and immutable artifact fingerprints.

    Optional source inputs additionally bind the matrix to the original L2 arrays
    and posterior manifest. Standalone verification uses stored lineage/snapshots;
    it never interprets a metadata flag as proof of numerical validity.
    """
    result = {"metadata_exists": (directory/"metadata.json").is_file()}
    try:
        meta = read_json(directory/"metadata.json")
        config = SimilarityConfig(**meta["config"])
        ids = pd.read_parquet(directory/"content_index.parquet")
        contents = pd.read_parquet(directory/"content_provenance.parquet")
        records = pd.read_parquet(directory/"record_provenance.parquet")
        neighbors = pd.read_parquet(directory/"nearest_neighbors.parquet")
        pairs = pd.read_parquet(directory/"pair_analysis.parquet")
        matrix = np.load(directory/"cosine_similarity.npy", mmap_mode="r", allow_pickle=False)
        source = read_json(directory/"feature_metadata.json")
        n = len(ids)
        result.update(matrix_quality(matrix, n, config.numerical_atol))
        result["dtype_valid"] = matrix.dtype == np.float32
        result["metadata_consistent"] = (meta["content_count"] == n and meta["metric"] == config.metric and meta["top_k"] == config.top_k
                                         and all(meta.get(key) == value for key, value in source.items()))
        result["similarity_space_id_valid"] = meta["similarity_space_id"] == similarity_space_id(meta["dataset_id"], meta["feature_space_id"], config)
        result["content_ids_aligned"] = bool(ids.content_id.is_unique and ids.content_id.notna().all()
                                             and np.array_equal(ids.embedding_row, np.arange(n))
                                             and contents.content_id.equals(ids.content_id) and contents.embedding_row.equals(ids.embedding_row))
        result["record_mapping_valid"] = bool(records.frame_id.is_unique and records.frame_id.notna().all()
                                               and set(records.content_id) == set(ids.content_id)
                                               and records.content_id.map(ids.set_index("content_id").embedding_row).equals(records.embedding_row))
        required = ["query_row", "neighbor_row", "query_content_id", "neighbor_rank", "neighbor_content_id", "cosine_similarity"]
        expected = compute_topk_neighbors(matrix, ids.content_id.tolist(), config.top_k)
        actual = neighbors.sort_values(["query_row", "neighbor_rank"]).reset_index(drop=True)
        result["topk_count_valid"] = len(neighbors) == n*config.top_k
        result["no_self_neighbors"] = bool(neighbors.query_content_id.ne(neighbors.neighbor_content_id).all())
        result["topk_ranks_values_and_order_valid"] = actual[required].equals(expected[required])
        a, b = np.triu_indices(n, 1)
        result["unique_pairs_valid"] = bool(len(pairs) == n*(n-1)//2 and np.array_equal(pairs.query_row, a)
                                            and np.array_equal(pairs.neighbor_row, b) and np.array_equal(pairs.cosine_similarity, matrix[a, b]))
        relation_columns = ["same_sequence", "frame_delta", "historical_cross_split", "split_provenance_complete", "provenance_complete"]
        result["posterior_relations_valid"] = (annotate_pairs(pairs, contents)[relation_columns].equals(pairs[relation_columns])
                                                and annotate_pairs(neighbors, contents)[relation_columns].equals(neighbors[relation_columns]))
        result["artifact_checksums_valid"] = set(meta["output_sha256"]) == set(ARTIFACT_NAMES) and all(file_sha256(directory/name) == meta["output_sha256"][name] for name in ARTIFACT_NAMES)
        if feature_directory is not None:
            result["source_feature_files_match"] = all(file_sha256(feature_directory/name) == meta["input_signature"]["feature_files"][name] for name in FEATURE_FILES)
            x = np.load(feature_directory/"embeddings_l2.npy", allow_pickle=False)
            result["matrix_matches_source_embeddings"] = bool(np.allclose(matrix, compute_cosine_similarity(x, config.numerical_atol), atol=config.numerical_atol, rtol=0))
        if manifest_path is not None:
            manifest = pd.read_parquet(manifest_path)
            result["canonical_dataset_id_matches"] = dataset_id_from_manifest(manifest) == meta["dataset_id"]
            expected_contents, expected_records = build_content_provenance(manifest, ids)
            result["canonical_provenance_matches"] = expected_contents.equals(contents) and expected_records.equals(records)
        result["quality_valid"] = all(result.values())
    except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError) as error:
        result.update({"quality_valid": False, "error_type": type(error).__name__})
    return result
