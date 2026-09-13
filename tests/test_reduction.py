"""Scientific invariants on small synthetic inputs; no data, models or network."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import pdist, squareform
from typer.testing import CliRunner

from flir_pipeline.cli import app
from flir_pipeline.data.identity import dataset_id_from_manifest
from flir_pipeline.reduction.base import (
    ReductionConfig,
    configuration_id,
    load_grid,
    reduction_space_id,
)
from flir_pipeline.reduction.benchmark import (
    benchmark_to_store,
    select_candidates,
    verify_benchmark,
)
from flir_pipeline.reduction.metrics import (
    EvaluationReference,
    evaluate_coordinates,
    neighbor_order,
    summarize_stability,
    trustworthiness_continuity,
)
from flir_pipeline.reduction.reducers import make_reducer
from flir_pipeline.reduction.storage import (
    coordinate_quality,
    load_inputs,
    run_to_store,
    verify_reduction,
)
from flir_pipeline.similarity.cosine import (
    compute_cosine_similarity,
    compute_topk_neighbors,
)
from flir_pipeline.similarity.storage import SimilarityConfig, compute_to_store


def vectors(n=64, d=8):
    x = np.random.default_rng(42).normal(size=(n, d)).astype(np.float32)
    return x/np.linalg.norm(x, axis=1, keepdims=True)


def synthetic_inputs(tmp_path):
    source = tmp_path/"features"
    source.mkdir(parents=True)
    x = vectors()
    n = len(x)
    rows = pd.DataFrame({"content_id": [f"c{i:03d}" for i in range(n)], "embedding_row": np.arange(n),
                         "representative_frame_id": [f"f{i:03d}" for i in range(n)],
                         "image_sha256": [f"synthetic-image-{i}" for i in range(n)],
                         "source_archive": "synthetic.zip", "source_member_path": [f"{i}.png" for i in range(n)]})
    manifest = rows.rename(columns={"representative_frame_id": "frame_id"}).drop(columns="embedding_row")
    manifest = manifest.assign(label_sha256="synthetic-label", original_split="train", possible_sequence="seq", possible_frame_index=np.arange(n), temporal_inference_confidence="medium")
    duplicate = manifest.iloc[[0]].copy().assign(frame_id="duplicate", original_split="val")
    manifest = pd.concat([manifest, duplicate], ignore_index=True)
    manifest_path = tmp_path/"manifest.parquet"
    manifest.to_parquet(manifest_path, index=False)
    rows.to_parquet(source/"content_index.parquet", index=False)
    manifest[["frame_id", "content_id"]].assign(embedding_row=[*range(n), 0]).to_parquet(source/"record_index.parquet", index=False)
    np.save(source/"embeddings_raw.npy", x)
    np.save(source/"embeddings_l2.npy", x)
    meta = {"dataset_id": dataset_id_from_manifest(manifest), "feature_space_id": "synthetic-feature",
            "extractor": "synthetic", "model_id": "synthetic/unit-vectors", "embedding_dimension": x.shape[1],
            "pooling_strategy": "synthetic", "model_revision": "a"*40, "resolved_model_revision": "a"*40,
            "selected_content_ids": n, "unique_content_ids": n, "total_records": len(manifest)}
    (source/"metadata.json").write_text(json.dumps(meta))
    similarity = compute_to_store(source, manifest_path, SimilarityConfig(), tmp_path/"similarity")
    return load_inputs(source, similarity, manifest_path), manifest_path


def test_grid_identity_and_preprocessing_contract():
    root = Path(__file__).resolve().parents[1]
    for method in ("tsne", "pacmap"):
        grid = load_grid(root/f"configs/reduction/{method}_research.yaml")
        assert len(grid) == 9
        assert len({configuration_id(c) for c in grid}) == 3
        assert set(c.seed for c in grid) == {0, 1, 2}
        config = grid[0]
        value = reduction_space_id("dataset", "feature", config, {"library": "1"})
        assert value == reduction_space_id("dataset", "feature", ReductionConfig(**json.loads(json.dumps(config.__dict__))), {"library": "1"})
        assert value != reduction_space_id("dataset", "feature", replace(config, seed=10), {"library": "1"})
        assert value != reduction_space_id("dataset", "feature", config, {"library": "2"})
        assert configuration_id(config) == configuration_id(replace(config, seed=10))
    with pytest.raises(ValueError, match="preliminary PCA"):
        ReductionConfig("tsne", preprocessing="pca50")
    with pytest.raises(ValueError, match="apply_pca=false"):
        ReductionConfig("pacmap", hyperparameters={"apply_pca": True})
    with pytest.raises(TypeError):
        ReductionConfig("tsne", device="cpu")
    with pytest.raises(ValueError, match="smaller than N"):
        ReductionConfig("tsne", hyperparameters={"perplexity": 64}).validate_population(64, 8)
    with pytest.raises(ValueError, match="silently adjust"):
        ReductionConfig("pacmap", hyperparameters={"n_neighbors": 20}).validate_population(64, 8)


def test_grid_rejects_unknown_keys_duplicate_runs_and_excessive_search(tmp_path):
    path = tmp_path/"config.yaml"
    path.write_text('method: tsne\nseeds: [0, 0]\n')
    with pytest.raises(ValueError, match="distinct seeds"):
        load_grid(path)
    path.write_text('method: tsne\ngrid:\n  perplexity: [5, 10, 20, 30]\n')
    with pytest.raises(ValueError, match="at most nine"):
        load_grid(path)
    path.write_text('method: tsne\nhyperparameters:\n  class_labels: true\n')
    with pytest.raises(ValueError, match="hyperparameter"):
        load_grid(path)


def test_exact_intrusion_and_omission_penalties():
    original = np.array([[0], [1], [4], [10]], dtype=float)
    reduced = np.array([[0], [5], [6], [7]], dtype=float)
    ids = ["a", "b", "c", "d"]
    a = neighbor_order(squareform(pdist(original)), ids)
    b = neighbor_order(squareform(pdist(reduced)), ids)
    scores = trustworthiness_continuity(a, b, (1,))
    assert scores == {"trustworthiness@1": .875, "continuity@1": .75}
    assert trustworthiness_continuity(a, a, (1,)) == {"trustworthiness@1": 1., "continuity@1": 1.}
    with pytest.raises(ValueError, match="k < N/2"):
        trustworthiness_continuity(a, b, (2,))


def test_trustworthiness_and_continuity_match_sklearn_reference():
    sklearn = pytest.importorskip("sklearn.manifold")
    x, y = vectors(), np.random.default_rng(1).normal(size=(64, 2))
    ids = [str(i) for i in range(len(x))]
    distances = squareform(pdist(x.astype(np.float64), metric="cosine"))
    original_order = neighbor_order(distances, ids)
    reduced_order = neighbor_order(squareform(pdist(y)), ids)
    scores = trustworthiness_continuity(original_order, reduced_order, (5, 10, 20))
    for k in (5, 10, 20):
        assert scores[f"trustworthiness@{k}"] == pytest.approx(sklearn.trustworthiness(x, y, metric="cosine", n_neighbors=k), abs=1e-12)
        # Cosine and Euclidean rankings coincide on these L2 vectors; no ties here.
        assert scores[f"continuity@{k}"] == pytest.approx(sklearn.trustworthiness(y, x, n_neighbors=k), abs=1e-12)


def test_preservation_stability_and_pair_sampling_are_geometry_and_row_invariant():
    x = vectors()
    ids = [f"c{i:03d}" for i in range(len(x))]
    cosine = compute_cosine_similarity(x)
    original = compute_topk_neighbors(cosine, ids)
    reference = EvaluationReference.from_similarity(cosine, ids, original, 100, 2)
    y = np.random.default_rng(4).normal(size=(64, 2))
    scores, a, per_content = evaluate_coordinates(y, reference, (5, 10, 20))
    moved = 3*y[:, [1, 0]]*np.array([-1, 1])+[100, -10]
    scores_b, b, _ = evaluate_coordinates(moved, reference, (5, 10, 20))
    assert scores == scores_b
    assert len(per_content) == 64*3 and scores["distance_pair_count"] == 100
    _, stability = summarize_stability([a, b], [0, 1], (5, 10, 20))
    assert stability["mean"].eq(1).all()
    order = np.random.default_rng(3).permutation(len(x))
    shuffled = EvaluationReference.from_similarity(cosine[np.ix_(order, order)], [ids[i] for i in order], original, 100, 2)
    pairs = [(ids[a], ids[b]) for a, b in zip(reference.sample_a, reference.sample_b, strict=True)]
    other = [(shuffled.content_ids[a], shuffled.content_ids[b]) for a, b in zip(shuffled.sample_a, shuffled.sample_b, strict=True)]
    assert pairs == other
    with pytest.raises(ValueError, match="distinct aligned runs"):
        summarize_stability([a, b], [0, 0], (5,))


def test_coordinate_qa_allows_repeated_positions_and_flags_collapse():
    y = np.array([[0, 0], [0, 0], [1, 0], [0, 1]], dtype=np.float32)
    q = coordinate_quality(y, 4, 2)
    assert q["quality_valid"] and q["duplicate_coordinate_rows"] == 1
    assert not coordinate_quality(np.zeros((4, 2)), 4, 2)["quality_valid"]
    assert not coordinate_quality(np.full((4, 2), np.nan), 4, 2)["quality_valid"]
    assert not coordinate_quality(y, 5, 2)["quality_valid"]


@pytest.mark.parametrize("method", ["tsne", "pacmap"])
def test_real_small_backends_repeat_seed_and_leave_input_unchanged(method):
    pytest.importorskip("sklearn")
    if method == "pacmap":
        pytest.importorskip("pacmap")
    x = vectors()
    saved = x.copy()
    params = {"perplexity": 10, "max_iter": 350} if method == "tsne" else {"num_iters": [10, 10, 20]}
    reducer = make_reducer(ReductionConfig(method, hyperparameters=params))
    first, second = reducer.fit_transform(x), reducer.fit_transform(x)
    assert first.coordinates.shape == (64, 2)
    assert coordinate_quality(first.coordinates, 64, 2)["quality_valid"]
    np.testing.assert_array_equal(x, saved)
    np.testing.assert_allclose(first.coordinates, second.coordinates, rtol=0, atol=1e-6)
    assert first.metadata["preprocessing_effective"]["input_dimension_used"] == 8
    with pytest.raises(ValueError, match="original float32 L2"):
        reducer.fit_transform(2*x)


def test_storage_source_binding_and_three_seed_benchmark(tmp_path):
    pytest.importorskip("sklearn")
    inputs, manifest = synthetic_inputs(tmp_path)
    config = ReductionConfig("tsne", hyperparameters={"perplexity": 10, "max_iter": 350})
    output = run_to_store(inputs, config, tmp_path/"reductions")
    assert verify_reduction(output, inputs)["quality_valid"]
    before = (output/"metadata.json").read_bytes()
    assert run_to_store(inputs, config, tmp_path/"reductions") == output
    assert before == (output/"metadata.json").read_bytes()
    assert len(pd.read_parquet(output/"content_index.parquet")) == 64
    assert not (output/"record_index.parquet").exists()
    cli = CliRunner().invoke(app, ["reduction", "verify", str(output)])
    assert cli.exit_code == 0, cli.output
    benchmark = benchmark_to_store(inputs, [replace(config, seed=i) for i in range(3)], tmp_path/"reductions")
    assert verify_benchmark(benchmark, inputs)["quality_valid"]
    candidate = pd.read_csv(benchmark/"candidates.csv").iloc[0]
    assert candidate.seed == 0 and candidate.reduction_space_id == output.name
    assert len(pd.read_csv(benchmark/"runs.csv")) == 3
    # Changed posterior metadata has no place in the reducer API or coordinate input.
    modified = pd.read_parquet(manifest).assign(original_split="test", possible_sequence="different")
    assert np.array_equal(inputs.embeddings, np.load(inputs.feature_directory/"embeddings_l2.npy"))
    assert modified.content_id.nunique() == len(inputs.embeddings)


@pytest.mark.parametrize("corruption", ["coordinates", "index", "identity", "neighbors", "metrics"])
def test_verifier_rejects_corrupted_artifacts(tmp_path, corruption):
    pytest.importorskip("sklearn")
    inputs, _ = synthetic_inputs(tmp_path)
    output = run_to_store(inputs, ReductionConfig("tsne", hyperparameters={"perplexity": 10, "max_iter": 350}), tmp_path/"reductions")
    if corruption == "coordinates":
        y = np.load(output/"coordinates.npy")
        y[0, 0] = np.nan
        np.save(output/"coordinates.npy", y)
    elif corruption == "index":
        path = output/"content_index.parquet"
        frame = pd.read_parquet(path)
        frame.loc[1, "content_id"] = frame.loc[0, "content_id"]
        frame.to_parquet(path, index=False)
    elif corruption == "identity":
        path = output/"metadata.json"
        meta = json.loads(path.read_text())
        meta["feature_space_id"] = "different"
        path.write_text(json.dumps(meta))
    elif corruption == "neighbors":
        path = output/"nearest_neighbors.parquet"
        frame = pd.read_parquet(path)
        frame.loc[0, "neighbor_rank"] = 2
        frame.to_parquet(path, index=False)
    else:
        path = output/"metrics.json"
        metrics = json.loads(path.read_text())
        metrics["trustworthiness@5"] = 1.1
        path.write_text(json.dumps(metrics))
    assert not verify_reduction(output)["quality_valid"]


def test_candidate_selection_keeps_tradeoffs_and_uses_fixed_seed():
    rows, stability = [], []
    for label, values in [("a", [.98, .98, .8, .7, .5]), ("b", [.97, .97, .75, .95, .8]), ("dominated", [.9, .9, .5, .5, .4])]:
        for seed in range(3):
            rows.append({"method": "tsne", "configuration_id": label, "label": label, "seed": seed,
                         "reduction_space_id": f"{label}-{seed}", "fit_seconds": 1., "quality_valid": True, "rank_deficient_warning": False,
                         "trustworthiness@5": values[0], "continuity@5": values[1], "jaccard@5_mean_jaccard": values[2], "spearman_distance": values[4]})
        stability.append({"method": "tsne", "configuration_id": label, "k": 5, "mean": values[3]})
    summary, selected = select_candidates(pd.DataFrame(rows), pd.DataFrame(stability), (5,))
    assert not summary.set_index("configuration_id").loc["dominated", "pareto_nondominated"]
    assert selected.seed.eq(0).all()
    assert selected.reduction_space_id.iloc[0] in ("a-0", "b-0")
    missing = pd.DataFrame(rows)
    missing.loc[(missing.configuration_id == "a") & (missing.seed == 1), "spearman_distance"] = np.nan
    incomplete, chosen = select_candidates(missing, pd.DataFrame(stability), (5,))
    assert not incomplete.set_index("configuration_id").loc["a", "eligible"]
    assert chosen.reduction_space_id.iloc[0] == "b-0"
    with pytest.raises(ValueError, match="exactly seeds"):
        select_candidates(pd.DataFrame(rows).loc[lambda x: x.seed != 2], pd.DataFrame(stability), (5,))


def test_posterior_temporal_selection_and_memberships():
    from flir_pipeline.reduction.visualization import split_category, temporal_window

    provenance = pd.DataFrame({"content_id": [f"c{i}" for i in range(9)],
                               "sequence_key": ["b", "a", "a", "a", "a", "a", "a", "a", "a"],
                               "frame_index": [1, 1, 2, 3, 9, 10, 11, 12, 12]})
    first = temporal_window(provenance, size=2)
    assert first.frame_index.tolist() == [1, 2]
    shuffled = provenance.sample(frac=1, random_state=2).assign(original_split="test", x=999.)
    assert temporal_window(shuffled, size=2).content_id.tolist() == first.content_id.tolist()
    assert split_category('["val", "train"]') == "train+val"
    assert split_category('["test"]') == "test only"
    assert split_category('[]') == "unknown"
    with pytest.raises(ValueError, match="Unrecognized"):
        split_category('["invented"]')
