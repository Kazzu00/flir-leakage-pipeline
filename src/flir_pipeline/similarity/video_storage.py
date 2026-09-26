"""Version-2 video similarity: exact compact pair arrays and streamed Parquet.

No all-pairs DataFrame or triangular index arrays are constructed. For P=N(N-1)/2
the working pair payload is 21P bytes (float32 cosine, bool membership, int64
sample gap, float64 time gap). Quantile workspaces are temporary, exact float64;
the buffers are bounded, but allocator/BLAS/Arrow overhead and real RSS/runtime
still require measurement on the target machine.
"""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from flir_pipeline.data.identity import dataset_id_from_manifest
from flir_pipeline.data.video_temporal import VIDEO_SEMANTICS
from flir_pipeline.features.storage import (
    verify_feature_directory,
    verify_features_against_manifest,
)
from flir_pipeline.similarity.cosine import (
    compute_cosine_similarity,
    compute_topk_neighbors,
    distribution_summary,
    matrix_quality,
    summarize_topk,
)
from flir_pipeline.similarity.storage import (
    FEATURE_FILES,
    SimilarityConfig,
    config_payload,
    execution_provenance,
    file_sha256,
    read_json,
    similarity_space_id,
    source_signature,
    write_json,
)
from flir_pipeline.similarity.video_temporal import (
    VideoRelations,
    build_video_content_provenance,
)

TABLES = ("source_video_similarity", "sample_index_gap_similarity",
          "timestamp_gap_seconds_similarity", "quantile_candidates", "topk_global_summary")
BASE_FILES = ("cosine_similarity.npy", "nearest_neighbors.parquet", "content_index.parquet",
              "content_provenance.parquet", "record_provenance.parquet",
              "topk_content_summary.parquet", "near_unit_pairs.parquet", "similarity_summary.json",
              "feature_metadata.json", "quality.json", *(f"{name}.csv" for name in TABLES))
PAIR_SCHEMA = pa.schema([
    ("query_row", pa.int32()), ("neighbor_row", pa.int32()),
    ("cosine_similarity", pa.float32()), ("same_source_video", pa.bool_()),
    ("min_sample_index_gap", pa.int64()), ("min_timestamp_gap_seconds", pa.float64()),
    ("query_content_id", pa.string()), ("neighbor_content_id", pa.string()),
])
FEATURE_SNAPSHOT_FIELDS = ("dataset_id", "feature_space_id", "extractor", "model_id", "model_revision",
                           "resolved_model_revision", "embedding_dimension", "pooling_strategy")


def artifact_names(config):
    return (*BASE_FILES, *(("pair_analysis.parquet",) if config.pair_storage == "full_streamed_v1" else ()))


def compact_pairs(matrix, relations):
    """Enumerate i<j in deterministic row order using only O(N) index workspace."""
    n = len(matrix)
    count = n * (n-1) // 2
    scores = np.empty(count, dtype=np.float32)
    same = np.empty(count, dtype=bool)
    samples = np.empty(count, dtype=np.int64)
    seconds = np.empty(count, dtype=np.float64)
    start = 0
    for query in range(n-1):
        targets = np.arange(query+1, n)
        stop = start + len(targets)
        scores[start:stop] = matrix[query, query+1:]
        same[start:stop], samples[start:stop], seconds[start:stop] = relations.row(query, targets)
        start = stop
    return scores, same, samples, seconds


def pair_row(query, n, ids, arrays, start):
    targets = np.arange(query+1, n, dtype=np.int32)
    stop = start + len(targets)
    scores, same, samples, seconds = (x[start:stop] for x in arrays)
    gaps = pd.array(samples, dtype="Int64")
    gaps[~same] = pd.NA
    return pd.DataFrame({"query_row": np.full(len(targets), query, dtype=np.int32),
                         "neighbor_row": targets, "cosine_similarity": scores,
                         "same_source_video": same, "min_sample_index_gap": gaps,
                         "min_timestamp_gap_seconds": seconds,
                         "query_content_id": ids[query], "neighbor_content_id": ids[targets]})


def write_pair_tables(output, config, ids, arrays):
    """Even a near-unit cohort containing every pair is written in bounded rows."""
    with ExitStack() as stack:
        near = stack.enter_context(pq.ParquetWriter(output/"near_unit_pairs.parquet", PAIR_SCHEMA))
        full = (stack.enter_context(pq.ParquetWriter(output/"pair_analysis.parquet", PAIR_SCHEMA))
                if config.pair_storage == "full_streamed_v1" else None)
        start = 0
        for query in range(len(ids)-1):
            stop = start + len(ids)-query-1
            mask = np.abs(arrays[0][start:stop]-1) <= config.near_unit_atol
            if full is not None or mask.any():
                frame = pair_row(query, len(ids), ids, arrays, start)
                if full is not None:
                    full.write_table(pa.Table.from_pandas(frame, schema=PAIR_SCHEMA, preserve_index=False))
                if mask.any():
                    near.write_table(pa.Table.from_pandas(frame.loc[mask], schema=PAIR_SCHEMA, preserve_index=False))
            start = stop


def gap_summary(scores, gaps, known, bounds, unit):
    rows = []
    lower = None
    for upper in (*bounds, None):
        mask = known.copy()
        if lower is not None:
            mask &= gaps > lower
        if upper is not None:
            mask &= gaps <= upper
        rows.append({"lower_exclusive": lower, "upper_inclusive": upper, "unit": unit,
                     **distribution_summary(scores[mask])})
        lower = upper
    return pd.DataFrame(rows)


def summarize(arrays, neighbors, config, n, record_count):
    scores, same, samples, seconds = arrays
    thresholds = np.quantile(scores.astype(np.float64), config.quantiles, method="linear")
    quantiles = []
    for q, threshold in zip(config.quantiles, thresholds, strict=True):
        # Fix the comparison loop to float64: NumPy scalar-promotion rules must
        # not round a linear quantile threshold back to the float32 score grid.
        mask = np.greater_equal(scores, threshold, signature="dd->?")
        count = int(mask.sum())
        shared = int(np.count_nonzero(mask & same))
        quantiles.append({"quantile": q, "top_percentage": 100*(1-q), "threshold_cosine": float(threshold),
                          "pair_count": count, "same_source_video_count": shared,
                          "disjoint_source_video_count": count-shared})
    per_content, topk = summarize_topk(neighbors)
    tables = {
        "source_video_similarity": pd.DataFrame([
            {"relation": name, **distribution_summary(scores[mask])}
            for name, mask in (("same_source_video", same), ("disjoint_source_videos", ~same))]),
        "sample_index_gap_similarity": gap_summary(scores, samples, same, config.sample_index_gap_upper_bounds, "samples"),
        "timestamp_gap_seconds_similarity": gap_summary(scores, seconds, same, config.timestamp_gap_seconds_upper_bounds, "seconds"),
        "quantile_candidates": pd.DataFrame(quantiles), "topk_global_summary": topk,
    }
    temporal_neighbors = {}
    for name, rows in (("topk", neighbors), ("rank1", neighbors.loc[neighbors.neighbor_rank == 1])):
        shared = rows.same_source_video
        temporal_neighbors[name] = {"count": len(rows), "same_source_video_count": int(shared.sum()),
                                    "disjoint_source_video_count": int((~shared).sum()),
                                    "sample_index_gap": distribution_summary(rows.loc[shared, "min_sample_index_gap"].to_numpy(dtype=float)),
                                    "timestamp_gap_seconds": distribution_summary(rows.loc[shared, "min_timestamp_gap_seconds"].to_numpy())}
    summary = {"global_similarity": distribution_summary(scores), "temporal_neighbors": temporal_neighbors,
               "temporal_coverage": {"contents": n, "occurrences": record_count,
                                     "same_source_video_pairs": int(same.sum()),
                                     "disjoint_source_video_pairs": int((~same).sum()),
                                     "verified_capture_timestamps": 0, "sequence_identity": "unknown"},
               "near_unit_pair_count": int(np.count_nonzero(np.abs(scores-1) <= config.near_unit_atol)),
               "near_unit_atol": config.near_unit_atol, "std_ddof": 0,
               "quantile_method": "linear", "quantiles_exact": True,
               "quantile_selection": "cosine >= threshold; retain ties",
               "pair_unit": "unordered distinct content pair, i<j",
               "neighbor_unit": "directed content-to-content edge", "provenance_semantics": VIDEO_SEMANTICS}
    return summary, tables, per_content


def compute_video_to_store(feature_directory, manifest_path, config, output_root):
    if config.algorithm_version != "content_cosine_v2":
        raise ValueError("Video storage requires an explicit content_cosine_v2 configuration")
    manifest = pd.read_parquet(manifest_path)
    if not verify_features_against_manifest(feature_directory, manifest)["reproducible_full_dataset_valid"]:
        raise ValueError("Similarity requires verified complete features and resolved revision")
    source = read_json(feature_directory/"metadata.json")
    identity = similarity_space_id(source["dataset_id"], source["feature_space_id"], config)
    output = output_root/source["extractor"]/source["dataset_id"]/source["feature_space_id"]/identity
    signature = source_signature(feature_directory, manifest)
    index = pd.read_parquet(feature_directory/"content_index.parquet")
    contents, records = build_video_content_provenance(manifest, index)
    if output.exists():
        if not (output/"metadata.json").is_file():
            raise ValueError("Incomplete similarity directory preserved; use a separate output root")
        if read_json(output/"metadata.json").get("input_signature") != signature or not verify_video_similarity(output, feature_directory, manifest_path)["quality_valid"]:
            raise ValueError("Existing similarity cache mismatches inputs or failed QA; preserved")
        return output
    n = len(index)
    if config.top_k >= n:
        raise ValueError("top_k must be smaller than the content count")
    x = np.load(feature_directory/"embeddings_l2.npy", mmap_mode="r", allow_pickle=False)
    matrix = compute_cosine_similarity(x, config.numerical_atol)
    neighbors = compute_topk_neighbors(matrix, index.content_id.tolist(), config.top_k)
    relations = VideoRelations(records, n)
    neighbors = relations.annotate(neighbors)
    arrays = compact_pairs(matrix, relations)
    summary, tables, per_content = summarize(arrays, neighbors, config, n, len(records))
    quality = {**matrix_quality(matrix, n, config.numerical_atol),
               "unique_pair_count": len(arrays[0]), "expected_unique_pair_count": n*(n-1)//2,
               "neighbor_count": len(neighbors), "record_count": len(records)}
    snapshot = {key: source[key] for key in FEATURE_SNAPSHOT_FIELDS}
    output.mkdir(parents=True, exist_ok=False)
    np.save(output/"cosine_similarity.npy", matrix, allow_pickle=False)
    for name, frame in (("nearest_neighbors", neighbors), ("content_index", index),
                        ("content_provenance", contents), ("record_provenance", records),
                        ("topk_content_summary", per_content)):
        frame.to_parquet(output/f"{name}.parquet", index=False)
    write_pair_tables(output, config, index.content_id.to_numpy(), arrays)
    for name, table in tables.items():
        table.to_csv(output/f"{name}.csv", index=False)
    write_json(output/"feature_metadata.json", snapshot)
    write_json(output/"similarity_summary.json", summary)
    write_json(output/"quality.json", quality)
    metadata = {**snapshot, **execution_provenance(), "similarity_space_id": identity,
                "data_provenance_source_sha256": file_sha256(Path(__file__).parents[1]/"data"/"video_temporal.py"),
                "artifact_schema": "content_cosine_v2", "provenance_mode": config.provenance_mode,
                "provenance_semantics": VIDEO_SEMANTICS, "pair_storage": config.pair_storage,
                "quantiles_exact": True, "metric": config.metric, "top_k": config.top_k,
                "content_count": n, "record_count": len(records), "config": config_payload(config),
                "input_signature": signature,
                "output_sha256": {name: file_sha256(output/name) for name in artifact_names(config)}}
    write_json(output/"metadata.json", metadata)
    # Release O(P) buffers before the independent reconstruction in verification.
    del arrays, matrix
    if not verify_video_similarity(output)["quality_valid"]:
        raise ValueError("Written video similarity artifacts failed verification; preserved")
    return output


def _frames_equal(left, right, exact=False):
    try:
        pd.testing.assert_frame_equal(left, right, check_dtype=False, check_exact=exact, rtol=1e-12, atol=1e-12)
        return True
    except AssertionError:
        return False


def verify_pair_file(path, arrays, ids, near_atol=None):
    """Check every persisted pair in bounded batches, including ordering/coverage."""
    n, count, previous = len(ids), 0, -1
    parquet = pq.ParquetFile(path)
    if not parquet.schema_arrow.equals(PAIR_SCHEMA, check_metadata=False):
        return False
    for batch in parquet.iter_batches(batch_size=65536):
        # Preserve nullable integer gaps without a lossy float64 round trip.
        frame = batch.to_pandas(types_mapper=lambda dtype: pd.Int64Dtype() if pa.types.is_int64(dtype) else None)
        a, b = frame.query_row.to_numpy(), frame.neighbor_row.to_numpy()
        if not ((a >= 0) & (a < b) & (b < n)).all():
            return False
        a = a.astype(np.int64)
        offsets = a * (2*n-a-1)//2 + b-a-1
        if not (np.diff(offsets) > 0).all() or offsets[0] <= previous:
            return False
        scores, same, samples, seconds = (values[offsets] for values in arrays)
        expected_gaps = pd.array(samples, dtype="Int64")
        expected_gaps[~same] = pd.NA
        expected = pd.DataFrame({"query_row": a, "neighbor_row": b, "cosine_similarity": scores,
                                 "same_source_video": same, "min_sample_index_gap": expected_gaps,
                                 "min_timestamp_gap_seconds": seconds,
                                 "query_content_id": ids[a], "neighbor_content_id": ids[b]})
        if not _frames_equal(frame, expected, exact=True):
            return False
        if near_atol is not None and not (np.abs(scores-1) <= near_atol).all():
            return False
        count += len(frame)
        previous = offsets[-1]
    expected_count = len(arrays[0]) if near_atol is None else int(np.count_nonzero(np.abs(arrays[0]-1) <= near_atol))
    return count == expected_count


def verify_video_similarity(directory: Path, feature_directory=None, manifest_path=None):
    result = {"metadata_exists": (directory/"metadata.json").is_file()}
    try:
        meta = read_json(directory/"metadata.json")
        config = SimilarityConfig(**meta["config"])
        if config.algorithm_version != "content_cosine_v2":
            raise ValueError("Expected v2")
        files = artifact_names(config)
        result["artifact_checksums_valid"] = set(meta["output_sha256"]) == set(files) and all(
            file_sha256(directory/name) == meta["output_sha256"][name] for name in files)
        index = pd.read_parquet(directory/"content_index.parquet")
        contents = pd.read_parquet(directory/"content_provenance.parquet")
        records = pd.read_parquet(directory/"record_provenance.parquet")
        expected_contents, expected_records = build_video_content_provenance(records, index)
        result["occurrence_provenance_valid"] = contents.equals(expected_contents) and records.equals(expected_records)
        n = len(index)
        source = read_json(directory/"feature_metadata.json")
        result["metadata_consistent"] = (
            set(source) == set(FEATURE_SNAPSHOT_FIELDS)
            and meta["artifact_schema"] == "content_cosine_v2" and meta["content_count"] == n
            and meta["record_count"] == len(records) and meta["metric"] == config.metric
            and meta["top_k"] == config.top_k and meta["pair_storage"] == config.pair_storage
            and meta["provenance_mode"] == config.provenance_mode
            and meta["provenance_semantics"] == VIDEO_SEMANTICS and meta["quantiles_exact"] is True
            and all(meta.get(key) == value for key, value in source.items()))
        result["similarity_space_id_valid"] = meta["similarity_space_id"] == similarity_space_id(meta["dataset_id"], meta["feature_space_id"], config)
        matrix = np.load(directory/"cosine_similarity.npy", mmap_mode="r", allow_pickle=False)
        result.update(matrix_quality(matrix, n, config.numerical_atol))
        result["dtype_valid"] = matrix.dtype == np.float32
        relations = VideoRelations(expected_records, n)
        expected_neighbors = relations.annotate(compute_topk_neighbors(matrix, index.content_id.tolist(), config.top_k))
        neighbors = pd.read_parquet(directory/"nearest_neighbors.parquet")
        result["topk_and_posterior_valid"] = neighbors.equals(expected_neighbors)
        arrays = compact_pairs(matrix, relations)
        summary, tables, per_content = summarize(arrays, expected_neighbors, config, n, len(records))
        result["summary_valid"] = summary == read_json(directory/"similarity_summary.json")
        result["summary_tables_valid"] = all(_frames_equal(pd.read_csv(directory/f"{name}.csv"), table) for name, table in tables.items())
        result["content_summary_valid"] = per_content.equals(pd.read_parquet(directory/"topk_content_summary.parquet"))
        quality = {**matrix_quality(matrix, n, config.numerical_atol), "unique_pair_count": len(arrays[0]),
                   "expected_unique_pair_count": n*(n-1)//2, "neighbor_count": len(expected_neighbors), "record_count": len(records)}
        result["quality_snapshot_valid"] = quality == read_json(directory/"quality.json")
        result["near_unit_pairs_valid"] = verify_pair_file(directory/"near_unit_pairs.parquet", arrays, index.content_id.to_numpy(), config.near_unit_atol)
        if config.pair_storage == "full_streamed_v1":
            result["full_pairs_valid"] = verify_pair_file(directory/"pair_analysis.parquet", arrays, index.content_id.to_numpy())
        if feature_directory is not None:
            result["source_feature_files_match"] = set(meta["input_signature"]["feature_files"]) == set(FEATURE_FILES) and all(file_sha256(feature_directory/name) == meta["input_signature"]["feature_files"][name] for name in FEATURE_FILES)
            feature_meta = read_json(feature_directory/"metadata.json")
            result["feature_snapshot_matches_source"] = source == {key: feature_meta[key] for key in FEATURE_SNAPSHOT_FIELDS}
            result["content_index_matches_source"] = index.equals(pd.read_parquet(feature_directory/"content_index.parquet"))
            mapping = ["frame_id", "content_id", "embedding_row"]
            feature_records = pd.read_parquet(feature_directory/"record_index.parquet")
            result["record_mapping_matches_source"] = records[mapping].equals(feature_records[mapping].sort_values("frame_id").reset_index(drop=True))
            if manifest_path is None:
                result["source_feature_quality_valid"] = verify_feature_directory(feature_directory)["quality_valid"]
            x = np.load(feature_directory/"embeddings_l2.npy", mmap_mode="r", allow_pickle=False)
            # Reuse the validated cosine contract, release pair buffers first.
            del arrays
            result["matrix_matches_source_embeddings"] = bool(np.allclose(matrix, compute_cosine_similarity(x, config.numerical_atol), atol=config.numerical_atol, rtol=0))
        if manifest_path is not None:
            manifest = pd.read_parquet(manifest_path)
            c, r = build_video_content_provenance(manifest, index)
            result["canonical_dataset_id_matches"] = dataset_id_from_manifest(manifest) == meta["dataset_id"]
            result["canonical_provenance_matches"] = c.equals(contents) and r.equals(records)
            if feature_directory is not None:
                result["input_signature_matches"] = source_signature(feature_directory, manifest) == meta["input_signature"]
                result["source_full_coverage_valid"] = verify_features_against_manifest(feature_directory, manifest)["reproducible_full_dataset_valid"]
        result["quality_valid"] = all(result.values())
    except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError, OverflowError) as error:
        result.update(quality_valid=False, error_type=type(error).__name__)
    return result
