"""Offline density-clustering invariants on separated synthetic contents and noise."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import pdist, squareform
from typer.testing import CliRunner

from flir_pipeline.cli import app
from flir_pipeline.clustering.algorithms import (
    effective_parameters,
    fit_clustering,
    k_distances,
)
from flir_pipeline.clustering.base import (
    ClusteringConfig,
    clustering_space_id,
    load_grid,
)
from flir_pipeline.clustering.experiments import (
    comparison_to_store,
    screening_to_store,
    verify_collection,
)
from flir_pipeline.clustering.metrics import (
    EvaluationContext,
    assignment_agreement,
    cluster_summary,
    evaluate_clustering,
    silhouette_without_noise,
)
from flir_pipeline.clustering.selection import (
    parameter_neighbors,
    pareto_mask,
    shortlist_screening,
)
from flir_pipeline.clustering.storage import (
    ClusteringFamily,
    ClusterSpace,
    run_to_store,
    verify_run,
)
from flir_pipeline.clustering.visualization import exemplar_clusters, exemplar_members
from flir_pipeline.reduction.storage import ReductionInputs
from flir_pipeline.similarity.cosine import (
    compute_cosine_similarity,
    compute_topk_neighbors,
)
from flir_pipeline.similarity.storage import file_sha256, read_json, write_json


@pytest.fixture
def family(tmp_path):
    rng = np.random.default_rng(4)
    x = np.concatenate([center+rng.normal(0, .025, (25, 8)) for center in np.eye(8)[:3]] + [rng.normal(size=(15, 8))]).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    n = len(x)
    ids = [f"synthetic-{i:03d}" for i in range(n)]
    index = pd.DataFrame({"content_id": ids, "embedding_row": np.arange(n)})
    provenance = index.assign(sequence_key=["a"]*45+["b"]*45, sequence_id=["a"]*45+["b"]*45,
                              sequence_provenance_valid=True, frame_index=list(range(45))*2, frame_index_valid=True,
                              split_mask=[3]+[1]*89, split_membership_complete=True)
    cosine = compute_cosine_similarity(x)
    neighbors = compute_topk_neighbors(cosine, ids, 20)
    distance = squareform(pdist(x.astype(np.float64)))
    context = EvaluationContext.create(ids, distance, cosine, neighbors, provenance)
    source = ReductionInputs(tmp_path, tmp_path, {"dataset_id": "synthetic-data", "feature_space_id": "synthetic-feature",
                                                "extractor": "synthetic", "model_id": "synthetic/unit-blobs"}, {}, index, x, cosine, neighbors,
                             {"features": {"synthetic": "unchanged"}})
    spaces = {("original_l2", None): ClusterSpace("original_l2", None, None, x, distance, source.signatures)}
    for representation in ("tsne", "pacmap"):
        for seed, scale in enumerate((1., 3., 7.)):
            y = x[:, :2]*scale
            spaces[(representation, seed)] = ClusterSpace(representation, seed, f"synthetic-{representation}-{seed}", y,
                                                         squareform(pdist(y.astype(np.float64))), {**source.signatures, "reduction": f"synthetic-{seed}"})
    return ClusteringFamily(source, context, spaces, tmp_path)


def test_config_identity_and_bounded_grids():
    root = Path(__file__).resolve().parents[1]
    for algorithm, count in (("dbscan", 18), ("optics", 27), ("hdbscan", 12)):
        assert len(load_grid(root/f"configs/clustering/{algorithm}_research.yaml")) == count
    config = ClusteringConfig("dbscan")
    arguments = ("dataset", "feature", "original_l2", None, config, {"eps": .5}, {"sklearn": "test"})
    value = clustering_space_id(*arguments)
    assert value == clustering_space_id(*arguments)
    assert value != clustering_space_id("dataset", "feature", "tsne", "reduction", *arguments[4:])
    assert config.configuration_id != replace(config, hyperparameters={"eps_quantile": .95}).configuration_id
    with pytest.raises(ValueError):
        ClusteringConfig("kmeans")
    with pytest.raises(ValueError):
        ClusteringConfig("dbscan", {"eps": .5})
    with pytest.raises(ValueError):
        ClusteringConfig("optics", {"min_samples": 1})


def test_k_distance_includes_self_and_handles_zero_distance_ties():
    distances = squareform(pdist(np.array([[0.], [0.], [2.], [5.]])))
    np.testing.assert_array_equal(k_distances(distances, 2), [0., 0., 2., 3.])
    np.testing.assert_array_equal(k_distances(distances, 3), [2., 2., 2., 5.])
    config = ClusteringConfig("dbscan", {"min_samples": 3, "eps_quantile": .5})
    assert effective_parameters(config, distances)["eps"] == 2
    assert effective_parameters(config, 8*distances)["eps"] == 16
    with pytest.raises(ValueError, match="Nonpositive"):
        effective_parameters(ClusteringConfig("dbscan", {"min_samples": 2, "eps_quantile": .1}), distances)


@pytest.mark.parametrize("algorithm", ["dbscan", "optics", "hdbscan"])
def test_algorithms_keep_noise_and_ignore_row_permutation(family, algorithm):
    pytest.importorskip("sklearn")
    config = ClusteringConfig(algorithm, {"min_samples": 5, **({"eps_quantile": .8} if algorithm == "dbscan" else {"min_cluster_size": 10})})
    space = family.spaces[("original_l2", None)]
    p = effective_parameters(config, space.distances)
    result = fit_clustering(space.values, family.context.content_ids, config, p)
    assert len(set(result.labels)-{-1}) >= 3
    assert (result.labels == -1).any()
    assert result.labels.shape == (90,)
    before = space.values.copy()
    perm = np.random.default_rng(7).permutation(90)
    reordered = fit_clustering(space.values[perm], np.asarray(family.context.content_ids)[perm].tolist(), config, p)
    np.testing.assert_array_equal(reordered.labels[np.argsort(perm)], result.labels)
    np.testing.assert_array_equal(space.values, before)
    if algorithm == "hdbscan":
        assert result.probabilities is not None
        assert (result.probabilities[result.labels == -1] == 0).all()


def test_exact_medoid_cosine_and_posterior_summary(family):
    labels = np.full(90, -1, dtype=np.int32)
    labels[:3] = 0
    distances = family.context.original_distances.copy()
    distances[:3, :3] = [[0, 1, 3], [1, 0, 2], [3, 2, 0]]
    cosine = family.context.cosine.copy()
    cosine[:3, :3] = [[1, .8, .6], [.8, 1, .9], [.6, .9, 1]]
    context = replace(family.context, original_distances=distances, cosine=cosine)
    summary = cluster_summary(labels, context).iloc[0]
    assert summary.medoid_content_id == family.context.content_ids[1]
    assert summary.intra_pair_count == 3
    assert summary.mean_intra_cosine == pytest.approx((.8+.6+.9)/3)
    assert summary.dominant_sequence_fraction == 1 and summary.sequence_entropy_bits == 0
    assert summary.historical_split_memberships == '["train", "val"]'
    assert summary.n_members == 3


def test_noise_metrics_and_temporal_denominator(family):
    labels = np.full(90, -1, dtype=np.int32)
    labels[:2] = 0
    pairs = (np.array([0, 1, 2]), np.array([1, 2, 3]))
    context = replace(family.context, temporal_pairs={k: pairs for k in (1, 5, 10)})
    metrics, table = evaluate_clustering(labels, context, family.spaces[("original_l2", None)].distances)
    assert metrics["temporal_recall@1"] == 1/3
    assert metrics["visual_eligible_edges@10"] == 20
    assert metrics["visual_query_coverage"] == 2/90
    assert metrics["single_cluster"] and not metrics["all_noise"] and metrics["nearly_all_noise"]
    assert metrics["silhouette_original_space"] is None
    assert table.n_members.sum() == 2
    empty, summary = evaluate_clustering(np.full(90, -1, dtype=np.int32), context, context.original_distances)
    assert empty["all_noise"] and empty["n_clusters_excluding_noise"] == 0 and summary.empty
    assert empty["temporal_recall@1"] == 0 and empty["visual_neighbor_coherence@10"] is None


def test_silhouette_excludes_noise_and_uses_original_space(family):
    sklearn = pytest.importorskip("sklearn.metrics")
    labels = np.array([0]*25+[1]*25+[2]*25+[-1]*15)
    expected = sklearn.silhouette_score(family.source.embeddings[:75].astype(np.float64), labels[:75])
    assert silhouette_without_noise(family.context.original_distances, labels) == pytest.approx(expected)
    metrics, _ = evaluate_clustering(labels, family.context, family.spaces[("pacmap", 0)].distances)
    assert metrics["silhouette_original_space"] == pytest.approx(expected)
    assert metrics["silhouette_clustering_space"] != pytest.approx(expected)


def test_ari_ami_distinguish_noise_and_trivial_intersections():
    pytest.importorskip("sklearn")
    a = np.array([-1]*80+[0]*5+[1]*5)
    b = np.array([-1]*80+[0, 1]*5)
    result = assignment_agreement(a, b)
    assert result["all_points_ari"] > result["common_clustered_ari"]
    assert result["all_points_ami"] > result["common_clustered_ami"]
    assert result["common_clustered_n"] == 10
    empty = assignment_agreement(np.array([-1, -1]), np.array([-1, -1]))
    assert empty["all_points_ari"] == 1 and empty["all_points_trivial"]
    assert empty["common_clustered_ari"] is None and empty["common_clustered_n"] == 0
    permuted = assignment_agreement(np.array([0, 0, 1, 1, -1]), np.array([9, 9, 8, 8, -1]))
    assert permuted["common_clustered_ari"] == permuted["common_clustered_ami"] == 1


def test_pareto_and_shortlist_do_not_use_classes_or_historical_splits():
    values = pd.DataFrame({"a": [1., 2., 1.], "b": [1., 2., .5]})
    mask, omitted = pareto_mask(values, {"a": "max", "b": "max", "missing": "max"})
    assert mask.tolist() == [False, True, False] and omitted == ["missing"]
    rows = pd.DataFrame([{"encoder": "synthetic", "representation": "original_l2", "algorithm": "dbscan",
                          "configuration_id": str(i), "n_clusters_excluding_noise": 3, "silhouette_original_space": v,
                          "weighted_mean_intra_cluster_similarity": v, "visual_neighbor_coherence@10": v,
                          "temporal_recall@5": v, "noise_fraction": 1-v, "largest_cluster_fraction": 1-v} for i, v in enumerate((.5, .7, .9))])
    _, selected, _ = shortlist_screening(rows)
    _, altered, _ = shortlist_screening(rows.assign(class_purity=[1, 1, 0], historical_multisplit_clusters=[0, 0, 99]))
    assert selected.configuration_id.tolist() == altered.configuration_id.tolist() == ["2"]


def test_parameter_neighbors_change_one_prescribed_coordinate():
    candidate = ClusteringConfig("hdbscan", {"min_cluster_size": 20, "min_samples": 10})
    pool = {f"{a}-{b}": ClusteringConfig("hdbscan", {"min_cluster_size": a, "min_samples": b}) for a in (10, 20, 30) for b in (5, 10, 20)}
    assert {identity for identity, _ in parameter_neighbors(candidate, pool)} == {"10-10", "30-10", "20-5", "20-20"}


@pytest.mark.parametrize("corruption", ["labels", "index", "medoid", "metadata", "metrics"])
def test_storage_verification_rejects_corruption(family, tmp_path, corruption):
    pytest.importorskip("sklearn")
    path = run_to_store(family, family.spaces[("original_l2", None)], ClusteringConfig("hdbscan", {"min_cluster_size": 10}), tmp_path/"outputs")
    assert verify_run(path, family)["quality_valid"]
    if corruption == "labels":
        labels = np.load(path/"cluster_labels.npy")
        labels[0] = -2
        np.save(path/"cluster_labels.npy", labels)
    elif corruption in ("index", "medoid"):
        filename = "content_index.parquet" if corruption == "index" else "cluster_summary.parquet"
        table = pd.read_parquet(path/filename)
        column = "content_id" if corruption == "index" else "medoid_content_id"
        table.loc[0, column] = "absent"
        table.to_parquet(path/filename, index=False)
    else:
        filename = f"{corruption}.json"
        value = read_json(path/filename)
        value["clustering_space_id" if corruption == "metadata" else "clustered_points"] = "invalid"
        write_json(path/filename, value)
    assert not verify_run(path, family)["quality_valid"]


def test_screening_comparison_and_cli_roundtrip(family, tmp_path):
    pytest.importorskip("sklearn")
    configs = [ClusteringConfig("dbscan", {"min_samples": 5, "eps_quantile": q}) for q in (.7, .8, .9)]
    configs += [ClusteringConfig("optics", {"min_samples": 5, "min_cluster_size": 10, "xi": xi}) for xi in (.03, .05, .1)]
    configs += [ClusteringConfig("hdbscan", {"min_samples": ms, "min_cluster_size": size}) for ms in (3, 5) for size in (10, 15)]
    root = tmp_path/"clustering"
    screen = screening_to_store({"synthetic": family}, configs, root)
    assert verify_collection(screen, {"synthetic": family})["quality_valid"]
    comparison = comparison_to_store(screen, {"synthetic": family}, root)
    assert verify_collection(comparison, {"synthetic": family})["quality_valid"]
    assert len(pd.read_csv(screen/"screening.csv")) == 30
    assert read_json(comparison/"summary.json")["additional_seed_runs"] <= 36
    cli = CliRunner().invoke(app, ["clustering", "summary", str(comparison)])
    assert cli.exit_code == 0, cli.output
    assert not list(root.rglob("split_id*"))
    with pytest.raises(ValueError, match="same artifact root"):
        comparison_to_store(screen, {"synthetic": family}, tmp_path/"other")
    spaces = dict(family.spaces)
    spaces[("tsne", 0)] = replace(spaces[("tsne", 0)], reduction_space_id="different-reduction")
    with pytest.raises(ValueError, match="reduction sources"):
        comparison_to_store(screen, {"synthetic": replace(family, spaces=spaces)}, root)
    # Even coherently updated checksums cannot validate an invented stability
    # aggregate: the verifier reconstructs it from the actual assignment pairs.
    path = comparison/"evaluated_shortlist.csv"
    table = pd.read_csv(path)
    table.loc[0, "parameters_all_points_ari_mean"] = .123456789
    table.to_csv(path, index=False)
    meta = read_json(comparison/"metadata.json")
    meta["output_sha256"][path.name] = file_sha256(path)
    write_json(comparison/"metadata.json", meta)
    assert not verify_collection(comparison)["quality_valid"]


def test_exemplars_use_original_medoid_and_keep_noise_separate(family):
    labels = np.array([0]*20+[1]*25+[2]*30+[-1]*15)
    summary = cluster_summary(labels, family.context)
    roles = exemplar_clusters(summary)
    assert roles[:3] == [("Pequeño", 0), ("Mediano", 1), ("Grande", 2)]
    assert all(cluster != -1 for _, cluster in roles)
    medoid = family.context.content_ids.index(summary.iloc[0].medoid_content_id)
    members = exemplar_members(labels, 0, medoid, family.context)
    assert members[0] == medoid and len(set(members)) == 4
    assert all(labels[i] == 0 for i in members)
    farthest = max(np.flatnonzero(labels == 0), key=lambda i: family.context.original_distances[medoid, i])
    assert farthest in members
    with pytest.raises(ValueError, match="Medoid"):
        exemplar_members(labels, 0, 85, family.context)


def test_clustering_notebook_is_source_only_and_has_all_review_sections():
    import json

    notebook = json.loads(Path("notebooks/clustering_review.ipynb").read_text(encoding="utf-8"))
    headings = [line for cell in notebook["cells"] if cell["cell_type"] == "markdown" for line in "".join(cell["source"]).splitlines() if line.startswith("## ")]
    assert len(headings) == 18
    for number, heading in enumerate(headings, 1):
        assert heading.startswith(f"## {number}. ")
    for cell in notebook["cells"]:
        assert not cell.get("outputs") and cell.get("execution_count") is None
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), "clustering notebook", "exec")
