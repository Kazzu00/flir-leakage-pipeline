"""Exact synthetic video provenance and bounded similarity; no real-data claims."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from scipy.spatial.distance import pdist, squareform
from typer.testing import CliRunner

from flir_pipeline.cli import app
from flir_pipeline.data.identity import dataset_id_from_manifest
from flir_pipeline.data.video_temporal import (
    audit_video_temporal_lineage,
    temporal_mode,
)
from flir_pipeline.similarity.cosine import (
    compute_cosine_similarity,
    compute_topk_neighbors,
)
from flir_pipeline.similarity.storage import (
    SimilarityConfig,
    compute_to_store,
    config_payload,
    file_sha256,
    read_json,
    similarity_space_id,
    source_signature,
    stable_id,
    verify_similarity_directory,
    write_json,
)
from flir_pipeline.similarity.video_temporal import (
    VideoRelations,
    build_video_content_provenance,
)


def video_config(**overrides):
    return SimilarityConfig(**{
        "algorithm_version": "content_cosine_v2", "provenance_mode": "sampled_video_grid",
        "temporal_rule": "all_occurrences_min_same_source_video_gaps_v1",
        "cross_split_rule": "unavailable", "pair_storage": "summary_only_v1", "top_k": 3,
        **overrides,
    })


def video_features(tmp_path, n=5):
    feature = tmp_path/"features"
    feature.mkdir(parents=True)
    # A repeats in one video and across videos; A/B minima must use the later A.
    facts = [(0, "v1", 0, 2.), (0, "v1", 9, 2.), (0, "v2", 80, 4.),
             (1, "v1", 10, 2.), (1, "v2", 5, 4.),
             (2, "v3", 1, .5), (3, "v3", 3, .5), (4, "v4", 0, 3.)]
    facts += [(i, "v1", 20+i, 2.) for i in range(5, n)]
    rows = [{"manifest_version": "flir_video_samples_v1", "frame_id": f"f{i:03d}",
             "content_id": f"c{content:03d}", "image_sha256": f"c{content:03d}",
             "label_sha256": "", "video_id": video, "source_video": f"{video}.mp4",
             "source_video_sha256": video[1:]*64, "sample_index": sample,
             "timestamp_seconds": sample/fps, "sample_fps": fps,
             "source_frame_index_estimate": round(sample/fps*25), "image_path": f"{video}/{sample}.jpg"}
            for i, (content, video, sample, fps) in enumerate(facts)]
    manifest = pd.DataFrame(rows).astype({"sample_index": "Int64", "source_frame_index_estimate": "Int64"})
    index = manifest.drop_duplicates("content_id").sort_values("content_id")[["content_id", "image_sha256", "frame_id", "image_path"]]
    index = index.rename(columns={"frame_id": "representative_frame_id"}).assign(embedding_row=np.arange(n))
    index.to_parquet(feature/"content_index.parquet", index=False)
    manifest[["frame_id", "content_id"]].assign(embedding_row=manifest.content_id.map(index.set_index("content_id").embedding_row)).to_parquet(feature/"record_index.parquet", index=False)
    x = np.random.default_rng(0).normal(size=(n, 8)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    x[1] = x[0]  # Distinct content, near-unit vectors must stay discoverable.
    for name in ("embeddings_raw", "embeddings_l2"):
        np.save(feature/f"{name}.npy", x)
    meta = {"extractor": "synthetic", "model_id": "synthetic/video-test", "model_revision": "a"*40,
            "resolved_model_revision": "a"*40, "feature_space_id": "synthetic-video",
            "pooling_strategy": "synthetic", "dataset_id": dataset_id_from_manifest(manifest),
            "total_records": len(manifest), "selected_content_ids": n, "unique_content_ids": n,
            "embedding_dimension": x.shape[1]}
    write_json(feature/"metadata.json", meta)
    path = tmp_path/"manifest.parquet"
    manifest.to_parquet(path, index=False)
    return feature, path


def test_occurrences_never_become_sequences_or_representative_times(tmp_path):
    feature, path = video_features(tmp_path)
    manifest, index = pd.read_parquet(path), pd.read_parquet(feature/"content_index.parquet")
    assert temporal_mode(manifest) == "sampled_video_grid"
    contents, records = build_video_content_provenance(manifest, index)
    assert len(contents) == 5 and len(records) == 8
    assert contents.occurrence_count.tolist() == [3, 2, 1, 1, 1]
    assert records.loc[records.content_id.eq("c000"), "embedding_row"].eq(0).all()
    assert json.loads(contents.iloc[0].video_id_membership_set) == ["v1", "v2"]
    assert [r["sample_index"] for r in json.loads(contents.iloc[0].sampling_occurrences)] == [0, 9, 80]
    pd.testing.assert_series_equal(records.sample_index, manifest.sample_index)
    pd.testing.assert_series_equal(records.timestamp_seconds, manifest.timestamp_seconds)
    for frame in (contents, records):
        assert not {"sequence_id", "sequence_key", "possible_sequence", "frame_delta", "original_split", "split_mask", "labels"} & set(frame)
        assert frame.temporal_source.eq("sampled_video_grid").all()
        assert not frame.capture_timestamp_verified.any()
    # Feature representative order/choice never supplies time or memberships.
    swapped = index.assign(representative_frame_id="arbitrary")
    pd.testing.assert_frame_equal(contents, build_video_content_provenance(manifest.iloc[::-1], swapped)[0])


def test_exact_minimum_relations_use_all_occurrences_and_explicit_units(tmp_path):
    feature, path = video_features(tmp_path)
    c, records = build_video_content_provenance(pd.read_parquet(path), pd.read_parquet(feature/"content_index.parquet"))
    relations = VideoRelations(records, len(c))
    pairs = pd.DataFrame({"query_row": [0, 0, 2, 4, 1], "neighbor_row": [1, 2, 3, 0, 0]})
    result = relations.annotate(pairs)
    assert result.same_source_video.tolist() == [True, False, True, False, True]
    assert result.min_sample_index_gap.iloc[[0, 2, 4]].tolist() == [1, 2, 1]
    assert result.min_timestamp_gap_seconds.iloc[[0, 2, 4]].tolist() == [.5, 4., .5]
    assert result.min_sample_index_gap.iloc[[1, 3]].isna().all()
    assert result.min_timestamp_gap_seconds.iloc[[1, 3]].isna().all()
    # Minima can arise in different shared videos with different FPS.
    edited = pd.read_parquet(path)
    edited.loc[edited.frame_id.eq("f002"), "sample_index"] = 8
    edited.loc[edited.frame_id.eq("f002"), "timestamp_seconds"] = 8/100
    edited.loc[edited.video_id.eq("v2"), "sample_fps"] = 100.
    edited.loc[edited.frame_id.eq("f004"), "timestamp_seconds"] = 5/100
    _, r = build_video_content_provenance(edited, pd.read_parquet(feature/"content_index.parquet"))
    row = VideoRelations(r, len(c)).annotate(pairs).iloc[0]
    assert row.min_sample_index_gap == 1
    assert row.min_timestamp_gap_seconds == pytest.approx(.03)


@pytest.mark.parametrize("change", ["fraction", "negative", "timestamp", "fps", "duplicate", "sequence", "mixed", "undeclared", "source", "missing"])
def test_invalid_grids_fail_without_inventing_provenance(tmp_path, change):
    _, path = video_features(tmp_path)
    m = pd.read_parquet(path)
    if change == "fraction":
        m["sample_index"] = m.sample_index.astype(float) + .1
    elif change == "negative":
        m.loc[0, "sample_index"] = -1
    elif change == "timestamp":
        m.loc[0, "timestamp_seconds"] = 1.
    elif change == "fps":
        m.loc[0, "sample_fps"] = np.inf
    elif change == "duplicate":
        m.loc[1, ["sample_index", "timestamp_seconds"]] = [0, 0.]
    elif change == "sequence":
        m["sequence_id"] = m.video_id
    elif change == "mixed":
        m.loc[0, "manifest_version"] = "historical"
    elif change == "undeclared":
        m = m.drop(columns="manifest_version")
    elif change == "source":
        m.loc[0, "source_video"] = "wrong.mp4"
    else:
        m = m.drop(columns="sample_fps")
    with pytest.raises(ValueError):
        audit_video_temporal_lineage(m)


def test_scalable_and_full_modes_are_exact_without_triangular_pandas(tmp_path, monkeypatch):
    feature, path = video_features(tmp_path)
    # Forbid the historical full triangle allocation in both compute and verifier.
    def forbidden(*args, **kwargs):
        raise AssertionError("Full triangular indices/pair DataFrame are forbidden")
    monkeypatch.setattr(np, "triu_indices", forbidden)
    import flir_pipeline.similarity.storage as storage
    monkeypatch.setattr(storage, "annotate_pairs", forbidden)
    import flir_pipeline.similarity.video_storage as video_storage
    original = video_storage.pair_row
    def bounded(query, n, ids, arrays, start):
        result = original(query, n, ids, arrays, start)
        assert len(result) <= n-1
        return result
    monkeypatch.setattr(video_storage, "pair_row", bounded)
    outputs = [compute_to_store(feature, path, video_config(pair_storage=policy), tmp_path/policy)
               for policy in ("summary_only_v1", "full_streamed_v1")]
    assert not (outputs[0]/"pair_analysis.parquet").exists()
    assert outputs[0].name != outputs[1].name
    for output in outputs:
        assert verify_similarity_directory(output, feature, path)["quality_valid"]
    assert read_json(outputs[0]/"similarity_summary.json") == read_json(outputs[1]/"similarity_summary.json")
    pairs = pd.read_parquet(outputs[1]/"pair_analysis.parquet")
    assert len(pairs) == 10
    near = pd.read_parquet(outputs[0]/"near_unit_pairs.parquet")
    assert near[["query_content_id", "neighbor_content_id"]].values.tolist() == [["c000", "c001"]]
    # Explicit brute-force occurrence oracle, independent of VideoRelations.
    records = pd.read_parquet(path)
    for pair in pairs.itertuples():
        left = records[records.content_id.eq(pair.query_content_id)]
        right = records[records.content_id.eq(pair.neighbor_content_id)]
        deltas = [(abs(a.sample_index-b.sample_index), abs(a.timestamp_seconds-b.timestamp_seconds))
                  for a in left.itertuples() for b in right.itertuples() if a.video_id == b.video_id]
        assert pair.same_source_video == bool(deltas)
        if deltas:
            assert pair.min_sample_index_gap == min(d[0] for d in deltas)
            assert pair.min_timestamp_gap_seconds == min(d[1] for d in deltas)
        else:
            assert pd.isna(pair.min_sample_index_gap) and pd.isna(pair.min_timestamp_gap_seconds)
    summary = read_json(outputs[0]/"similarity_summary.json")
    values = pairs.cosine_similarity.to_numpy(dtype=np.float64)
    assert summary["global_similarity"]["mean"] == values.mean()
    assert summary["global_similarity"]["std"] == values.std()
    assert summary["global_similarity"]["median"] == np.quantile(values, .5)
    for r in pd.read_csv(outputs[0]/"quantile_candidates.csv").itertuples():
        threshold = np.quantile(values, r.quantile, method="linear")
        assert r.threshold_cosine == pytest.approx(threshold, abs=1e-15)
        assert r.pair_count == np.count_nonzero(values >= threshold)


def test_cosine_topk_metadata_independence_and_cache_fingerprints(tmp_path):
    feature, path = video_features(tmp_path)
    config = video_config()
    output = compute_to_store(feature, path, config, tmp_path/"one")
    before = (output/"metadata.json").read_bytes()
    assert compute_to_store(feature, path, config, tmp_path/"one") == output
    assert (output/"metadata.json").read_bytes() == before
    x = np.load(feature/"embeddings_l2.npy")
    matrix = np.load(output/"cosine_similarity.npy")
    np.testing.assert_array_equal(matrix, compute_cosine_similarity(x))
    raw_neighbors = compute_topk_neighbors(matrix, pd.read_parquet(feature/"content_index.parquet").content_id.tolist(), 3)
    pd.testing.assert_frame_equal(raw_neighbors, pd.read_parquet(output/"nearest_neighbors.parquet")[raw_neighbors.columns])
    m = pd.read_parquet(path)
    m["sample_index"] += 100
    m["timestamp_seconds"] = m.sample_index / m.sample_fps
    m.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="mismatches inputs"):
        compute_to_store(feature, path, config, tmp_path/"one")
    assert not verify_similarity_directory(output, manifest_path=path)["quality_valid"]
    updated = compute_to_store(feature, path, config, tmp_path/"two")
    np.testing.assert_array_equal(matrix, np.load(updated/"cosine_similarity.npy"))
    pd.testing.assert_frame_equal(raw_neighbors, pd.read_parquet(updated/"nearest_neighbors.parquet")[raw_neighbors.columns])
    with pytest.raises(ValueError, match="provenance mismatch"):
        compute_to_store(feature, path, SimilarityConfig(top_k=3), tmp_path/"historical")


def test_config_versioning_and_frozen_v1_identity():
    root = Path(__file__).resolve().parents[1]
    old = {"metric": "cosine", "top_k": 20, "pairwise_dtype": "float32",
           "quantiles": (.9, .95, .975, .99, .995, .999), "frame_delta_upper_bounds": (0, 1, 5, 10, 25, 50, 100),
           "numerical_atol": 1e-5, "near_unit_atol": 1e-6, "algorithm_version": "content_cosine_v1",
           "tie_break": "content_id_ascending", "temporal_rule": "all_occurrences_consensus_archive_sequence_index",
           "cross_split_rule": "exists_known_unequal_occurrence_splits"}
    assert config_payload(SimilarityConfig()) == old
    assert similarity_space_id("d", "f", SimilarityConfig()) == stable_id({"dataset_id": "d", "feature_space_id": "f", "config": old})
    for name in ("dinov2", "clip"):
        cfg = SimilarityConfig(**yaml.safe_load((root/f"configs/similarity/{name}_video_research.yaml").read_text()))
        assert cfg == video_config(top_k=20)
        assert "frame_delta_upper_bounds" not in config_payload(cfg)
    cfg = video_config()
    assert similarity_space_id("d", "f", cfg) != similarity_space_id("d", "f", SimilarityConfig(top_k=3))
    for other in (replace(cfg, pair_storage="full_streamed_v1"), replace(cfg, timestamp_gap_seconds_upper_bounds=(0, .5, 2))):
        assert similarity_space_id("d", "f", cfg) != similarity_space_id("d", "f", other)
    for changes in ({"algorithm_version": "content_cosine_v1"}, {"pair_storage": "full_v1"},
                    {"temporal_rule": "all_occurrences_consensus_archive_sequence_index"},
                    {"sample_index_gap_upper_bounds": (0, .5)}, {"timestamp_gap_seconds_upper_bounds": (0, np.inf)}):
        with pytest.raises(ValueError):
            replace(cfg, **changes)


def test_grid_signature_preserves_float64_precision_and_unknown_estimates(tmp_path):
    feature, path = video_features(tmp_path)
    manifest = pd.read_parquet(path)
    before = source_signature(feature, manifest)
    manifest.loc[1, "timestamp_seconds"] = np.nextafter(manifest.loc[1, "timestamp_seconds"], np.inf)
    assert source_signature(feature, manifest) != before
    manifest["source_frame_index_estimate"] = pd.array([pd.NA]*len(manifest), dtype="Int64")
    records = audit_video_temporal_lineage(manifest)
    assert records.source_frame_index_estimate.isna().all()
    manifest.to_parquet(path, index=False)
    output = compute_to_store(feature, path, video_config(), tmp_path/"similarity")
    assert verify_similarity_directory(output, feature, path)["quality_valid"]


def test_incomplete_v2_cache_is_preserved(tmp_path):
    feature, path = video_features(tmp_path)
    meta = read_json(feature/"metadata.json")
    config = video_config()
    identity = similarity_space_id(meta["dataset_id"], meta["feature_space_id"], config)
    output = tmp_path/"similarity"/meta["extractor"]/meta["dataset_id"]/meta["feature_space_id"]/identity
    output.mkdir(parents=True)
    sentinel = output/"interrupted.partial"
    sentinel.write_bytes(b"synthetic resumable state")
    with pytest.raises(ValueError, match="Incomplete similarity directory preserved"):
        compute_to_store(feature, path, config, tmp_path/"similarity")
    assert sentinel.read_bytes() == b"synthetic resumable state"


@pytest.mark.parametrize("corruption", ["matrix", "record", "content", "neighbor", "summary", "near", "policy", "full"])
def test_verifier_detects_corruption_even_with_updated_checksum(tmp_path, corruption):
    feature, manifest = video_features(tmp_path)
    config = video_config(pair_storage="full_streamed_v1")
    output = compute_to_store(feature, manifest, config, tmp_path/"similarity")
    filename = {"matrix": "cosine_similarity.npy", "record": "record_provenance.parquet",
                "content": "content_provenance.parquet", "neighbor": "nearest_neighbors.parquet",
                "summary": "similarity_summary.json", "near": "near_unit_pairs.parquet",
                "policy": "metadata.json", "full": "pair_analysis.parquet"}[corruption]
    target = output/filename
    if corruption == "matrix":
        x = np.load(target)
        x[0, 1] = np.nan
        np.save(target, x)
    elif corruption == "policy":
        meta = read_json(target)
        meta["pair_storage"] = "summary_only_v1"
        write_json(target, meta)
    elif corruption == "summary":
        summary = read_json(target)
        summary["global_similarity"]["mean"] = 99.
        write_json(target, summary)
    else:
        frame = pd.read_parquet(target)
        if corruption == "record":
            frame.loc[0, "timestamp_seconds"] += 1
        elif corruption == "content":
            frame.loc[0, "sampling_occurrences"] = "[]"
        elif corruption == "neighbor":
            frame.loc[0, "min_sample_index_gap"] = 999
        else:
            frame = frame.iloc[1:]
        frame.to_parquet(target, index=False)
    meta = read_json(output/"metadata.json")
    if filename in meta["output_sha256"]:
        meta["output_sha256"][filename] = file_sha256(target)
        write_json(output/"metadata.json", meta)
    assert not verify_similarity_directory(output)["quality_valid"]


def test_empty_and_all_near_unit_cohorts(tmp_path):
    for label, x in (("empty", np.eye(5, dtype=np.float32)), ("all", np.ones((5, 1), dtype=np.float32))):
        feature, path = video_features(tmp_path/label)
        for name in ("embeddings_raw", "embeddings_l2"):
            np.save(feature/f"{name}.npy", x)
        meta = read_json(feature/"metadata.json")
        meta["embedding_dimension"] = x.shape[1]
        write_json(feature/"metadata.json", meta)
        out = compute_to_store(feature, path, video_config(), tmp_path/label/"output")
        assert len(pd.read_parquet(out/"near_unit_pairs.parquet")) == (0 if label == "empty" else 10)
        assert verify_similarity_directory(out)["quality_valid"]


def test_cli_and_explicit_report_and_split_boundaries(tmp_path):
    feature, path = video_features(tmp_path)
    config = tmp_path/"config.yaml"
    config.write_text(yaml.safe_dump(config_payload(video_config())))
    runner = CliRunner()
    result = runner.invoke(app, ["similarity", "compute", "--feature-directory", str(feature), "--manifest", str(path), "--config", str(config), "--output-root", str(tmp_path/"output")])
    assert result.exit_code == 0, result.output
    output = next((tmp_path/"output").rglob("metadata.json")).parent
    for command in ("verify", "summary"):
        result = runner.invoke(app, ["similarity", command, str(output)])
        assert result.exit_code == 0, result.output
    from flir_pipeline.similarity.reporting import generate_similarity_report
    from flir_pipeline.splitting.storage import load_inputs
    with pytest.raises(ValueError, match="historical v1 only"):
        generate_similarity_report(output, output, tmp_path/"none", tmp_path/"no-zip")
    spec = tmp_path/"inputs.yaml"
    spec.write_text(yaml.safe_dump({"manifest": str(path), "encoders": {"dinov2": {}, "clip": {}}}))
    with pytest.raises(ValueError, match="out of scope"):
        load_inputs(spec, tmp_path/"none", tmp_path/"no-labels")


def test_reduction_and_clustering_accept_video_without_full_pairs(tmp_path):
    pytest.importorskip("sklearn")
    from flir_pipeline.clustering.base import ClusteringConfig
    from flir_pipeline.clustering.metrics import EvaluationContext
    from flir_pipeline.clustering.storage import (
        ClusteringFamily,
        ClusterSpace,
        run_to_store,
        verify_run,
    )
    from flir_pipeline.reduction.base import ReductionConfig
    from flir_pipeline.reduction.storage import load_inputs, verify_reduction
    from flir_pipeline.reduction.storage import run_to_store as reduce_to_store

    feature, path = video_features(tmp_path, n=48)
    output = compute_to_store(feature, path, video_config(top_k=20), tmp_path/"similarity")
    inputs = load_inputs(feature, output, path)
    assert not (output/"pair_analysis.parquet").exists()
    config = ReductionConfig("tsne", hyperparameters={"perplexity": 5, "max_iter": 300})
    reduction = reduce_to_store(inputs, config, tmp_path/"reduction")
    assert verify_reduction(reduction, inputs)["quality_valid"]
    xy = np.load(reduction/"coordinates.npy")
    provenance = pd.read_parquet(output/"content_provenance.parquet")
    original_distances = squareform(pdist(inputs.embeddings.astype(float)))
    context = EvaluationContext.create(inputs.content_index.content_id.tolist(), original_distances, inputs.cosine, inputs.original_neighbors, provenance)
    space = ClusterSpace("tsne", 0, read_json(reduction/"metadata.json")["reduction_space_id"], xy, squareform(pdist(xy.astype(float))), inputs.signatures)
    family = ClusteringFamily(inputs, context, {("tsne", 0): space}, tmp_path)
    clustering = run_to_store(family, space, ClusteringConfig("dbscan", {"min_samples": 3, "eps_quantile": .5}), tmp_path/"clustering")
    assert verify_run(clustering, family)["quality_valid"]
    cli = CliRunner().invoke(app, ["clustering", "verify", str(clustering)])
    assert cli.exit_code == 0, cli.output
    metrics = read_json(clustering/"metrics.json")
    assert metrics["temporal_recall@5"] is None
    assert metrics["historical_multisplit_clusters"] is None
    assert metrics["temporal_provenance_mode"] == "sampled_video_grid"
    assert "sequence_id" not in provenance
    assert "video_unknown_sequences_v2" in read_json(clustering/"metadata.json")["evaluation_protocol"]
    assert read_json(clustering/"metadata.json")["unavailable_metric_groups"] == ["sequence_coherence", "sequence_temporal_recall", "historical_splits"]
    metrics["temporal_recall@5"] = 0.0
    write_json(clustering/"metrics.json", metrics)
    meta = read_json(clustering/"metadata.json")
    meta["output_sha256"]["metrics.json"] = file_sha256(clustering/"metrics.json")
    write_json(clustering/"metadata.json", meta)
    checks = verify_run(clustering)
    assert checks["output_checksums_valid"]
    assert not checks["video_unavailable_metrics_explicit"] and not checks["quality_valid"]


@pytest.mark.parametrize("field,value", [
    ("timestamp_seconds", -1e-10), ("manifest_version", pd.NA),
    ("original_split", "train"), ("label_sha256", "invented-label"),
    ("label_exists", True), ("sequence_key", "v1"), ("new_split", "train"),
])
def test_reject_incompatible_video_declarations(tmp_path, field, value):
    _, path = video_features(tmp_path)
    manifest = pd.read_parquet(path)
    if field not in manifest:
        manifest[field] = value
    else:
        manifest.loc[0, field] = value
    with pytest.raises(ValueError):
        audit_video_temporal_lineage(manifest)


def test_empty_near_unit_parquet_still_requires_declared_schema(tmp_path):
    feature, path = video_features(tmp_path)
    x = np.eye(5, 8, dtype=np.float32)
    for name in ("embeddings_raw", "embeddings_l2"):
        np.save(feature/f"{name}.npy", x)
    output = compute_to_store(feature, path, video_config(), tmp_path/"similarity")
    filename = "near_unit_pairs.parquet"
    assert pd.read_parquet(output/filename).empty
    pd.DataFrame({"unrelated": pd.Series(dtype=str)}).to_parquet(output/filename, index=False)
    meta = read_json(output/"metadata.json")
    meta["output_sha256"][filename] = file_sha256(output/filename)
    write_json(output/"metadata.json", meta)
    checks = verify_similarity_directory(output)
    assert checks["artifact_checksums_valid"]
    assert not checks["near_unit_pairs_valid"] and not checks["quality_valid"]


@pytest.mark.parametrize("corruption", ["empty_snapshot", "pooling", "source_rows", "raw"])
def test_verification_binds_representation_and_mapping_to_actual_sources(tmp_path, corruption):
    feature, manifest = video_features(tmp_path)
    output = compute_to_store(feature, manifest, video_config(), tmp_path/"similarity")
    meta = read_json(output/"metadata.json")
    if corruption in {"empty_snapshot", "pooling"}:
        snapshot = read_json(output/"feature_metadata.json")
        if corruption == "empty_snapshot":
            snapshot = {}
        else:
            snapshot["pooling_strategy"] = meta["pooling_strategy"] = "wrong"
        write_json(output/"feature_metadata.json", snapshot)
        meta["output_sha256"]["feature_metadata.json"] = file_sha256(output/"feature_metadata.json")
    elif corruption == "source_rows":
        # Equal vectors can hide a swapped mapping if QA checks only the cosine.
        index = pd.read_parquet(feature/"content_index.parquet").iloc[[1, 0, 2, 3, 4]].reset_index(drop=True)
        index["embedding_row"] = np.arange(len(index))
        index.to_parquet(feature/"content_index.parquet", index=False)
        records = pd.read_parquet(feature/"record_index.parquet")
        records["embedding_row"] = records.content_id.map(index.set_index("content_id").embedding_row)
        records.to_parquet(feature/"record_index.parquet", index=False)
        meta["input_signature"] = source_signature(feature, pd.read_parquet(manifest))
    else:
        raw = np.load(feature/"embeddings_raw.npy")
        raw[0, 0] = np.nan
        np.save(feature/"embeddings_raw.npy", raw)
    write_json(output/"metadata.json", meta)
    if corruption == "empty_snapshot":
        assert not verify_similarity_directory(output)["quality_valid"]
    checks = verify_similarity_directory(output, feature, manifest)
    assert checks["artifact_checksums_valid"]
    assert not checks["quality_valid"]
    if corruption == "pooling":
        assert not checks["feature_snapshot_matches_source"]
    if corruption == "source_rows":
        assert checks["matrix_matches_source_embeddings"]
        assert not checks["content_index_matches_source"]
        assert not checks["record_mapping_matches_source"]
    if corruption == "raw":
        assert not checks["source_full_coverage_valid"]
    with pytest.raises(ValueError):
        compute_to_store(feature, manifest, video_config(), tmp_path/"similarity")


@pytest.mark.parametrize("pattern", ["all_noise", "one_cluster", "singletons"])
def test_video_clustering_degenerate_cases_preserve_unavailable_metrics(tmp_path, pattern):
    pytest.importorskip("sklearn")
    from flir_pipeline.clustering.metrics import EvaluationContext, evaluate_clustering
    from flir_pipeline.clustering.selection import pareto_mask

    feature, path = video_features(tmp_path, n=24)
    x = np.load(feature/"embeddings_l2.npy")
    contents, _ = build_video_content_provenance(pd.read_parquet(path), pd.read_parquet(feature/"content_index.parquet"))
    matrix = compute_cosine_similarity(x)
    neighbors = compute_topk_neighbors(matrix, contents.content_id.tolist(), 20)
    distances = squareform(pdist(x.astype(float)))
    context = EvaluationContext.create(contents.content_id.tolist(), distances, matrix, neighbors, contents)
    labels = {"all_noise": np.full(24, -1), "one_cluster": np.zeros(24), "singletons": np.arange(24)}[pattern].astype(np.int32)
    metrics, summary = evaluate_clustering(labels, context, distances)
    assert metrics["temporal_recall@5"] is None
    assert metrics["clustered_sequence_coverage"] is None
    assert metrics["historical_multisplit_clusters"] is None
    assert summary.sequence_count.isna().all()
    assert summary.historical_split_count.isna().all()
    assert context.temporal_pairs == {}
    _, omitted = pareto_mask(pd.DataFrame([metrics]), {"temporal_recall@5": "max", "noise_fraction": "min"})
    assert omitted == ["temporal_recall@5"]


def test_quantile_threshold_between_adjacent_float32_values_keeps_exact_cohort(tmp_path):
    from flir_pipeline.similarity.video_storage import compact_pairs, summarize

    feature, path = video_features(tmp_path)
    contents, records = build_video_content_provenance(pd.read_parquet(path), pd.read_parquet(feature/"content_index.parquet"))
    lo, hi = np.float32(.5), np.nextafter(np.float32(.5), np.float32(1))
    matrix = np.full((5, 5), lo, dtype=np.float32)
    np.fill_diagonal(matrix, 1)
    matrix[0, 1] = matrix[1, 0] = hi
    relations = VideoRelations(records, 5)
    neighbors = relations.annotate(compute_topk_neighbors(matrix, contents.content_id.tolist(), 3))
    arrays = compact_pairs(matrix, relations)
    _, tables, _ = summarize(arrays, neighbors, video_config(quantiles=(.9,)), 5, len(records))
    row = tables["quantile_candidates"].iloc[0]
    assert float(lo) < row.threshold_cosine < float(hi)
    assert row.pair_count == 1


def test_full_and_near_unit_verification_crosses_parquet_batch_boundary(tmp_path):
    import pyarrow.parquet as pq

    n = 370  # P=68265, more than the verifier's 65536-row batch.
    feature, manifest = video_features(tmp_path, n=n)
    x = np.zeros((n, 8), dtype=np.float32)
    x[:, 0] = 1
    for name in ("embeddings_raw", "embeddings_l2"):
        np.save(feature/f"{name}.npy", x)
    output = compute_to_store(feature, manifest, video_config(pair_storage="full_streamed_v1"), tmp_path/"similarity")
    assert verify_similarity_directory(output, feature, manifest)["quality_valid"]
    for name in ("near_unit_pairs", "pair_analysis"):
        parquet = pq.ParquetFile(output/f"{name}.parquet")
        assert parquet.metadata.num_rows == n*(n-1)//2
        assert max(parquet.metadata.row_group(i).num_rows for i in range(parquet.num_row_groups)) <= n-1
        assert [batch.num_rows for batch in parquet.iter_batches(batch_size=65536)] == [65536, 2729]


@pytest.mark.parametrize("corruption", ["row_dtype", "missing_hash"])
def test_content_provenance_rejects_invalid_index_schema(tmp_path, corruption):
    feature, path = video_features(tmp_path)
    index = pd.read_parquet(feature/"content_index.parquet")
    if corruption == "row_dtype":
        index["embedding_row"] = index.embedding_row.astype(float)
    else:
        index["image_sha256"] = index.image_sha256.astype("string")
        index.loc[0, "image_sha256"] = pd.NA
    with pytest.raises(ValueError, match="ordered unique content coverage"):
        build_video_content_provenance(pd.read_parquet(path), index)
