"""Exact regression and matrix-lifetime checks, using only synthetic contents."""

import weakref
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import test_clustering
import yaml
from clustering_memory_reference import configs, digest, snapshot
from scipy.spatial.distance import pdist, squareform
from test_reduction import synthetic_inputs

from flir_pipeline.clustering import algorithms
from flir_pipeline.clustering.algorithms import (
    effective_parameters,
    implementation_versions,
    k_distances,
)
from flir_pipeline.clustering.base import ClusteringConfig
from flir_pipeline.clustering.distances import (
    EuclideanDistances,
    distance_matrix,
    retain_distances,
)
from flir_pipeline.clustering.experiments import (
    comparison_to_store,
    load_families,
    screening_to_store,
    verify_collection,
)
from flir_pipeline.clustering.metrics import (
    EvaluationContext,
    cluster_summary,
    evaluate_clustering,
    silhouette_without_noise,
)
from flir_pipeline.clustering.storage import load_family, run_to_store, verify_run
from flir_pipeline.clustering.visualization import exemplar_members
from flir_pipeline.reduction.base import ReductionConfig
from flir_pipeline.reduction.benchmark import benchmark_to_store
from flir_pipeline.similarity.storage import file_sha256, read_json


@pytest.fixture
def family(tmp_path):
    return test_clustering.family.__wrapped__(tmp_path)


def lazy_family(eager):
    spaces = {key: replace(space, distances=EuclideanDistances(space.values))
              for key, space in eager.spaces.items()}
    context = replace(eager.context, original_distances=spaces[("original_l2", None)].distances)
    return replace(eager, spaces=spaces, context=context)


def assert_released(*families):
    assert all(not space.distances.is_materialized
               for family in families for space in family.spaces.values())


def watch_matrices(monkeypatch, families):
    """Track actual array lifetimes, not just whether a cache field was cleared."""
    names = {id(space.distances): (encoder, *key)
             for encoder, family in families.items() for key, space in family.spaces.items()}
    allocations = []
    original = EuclideanDistances.materialize

    @contextmanager
    def watched(self):
        allocated = not self.is_materialized
        with original(self) as matrix:
            if allocated:
                allocations.append((names[id(self)], weakref.ref(matrix)))
                live = [key for key, ref in allocations if ref() is not None]
                assert len(live) <= 2
                assert len({key[0] for key in live}) == 1
                assert len([key for key in live if key[1] != "original_l2"]) <= 1
            yield matrix

    monkeypatch.setattr(EuclideanDistances, "materialize", watched)
    return allocations


def test_scoped_distance_lifetime_is_lazy_exact_nested_and_exception_safe():
    values = np.random.default_rng(27).normal(size=(17, 6)).astype(np.float32)
    expected = squareform(pdist(values.astype(np.float64)))
    source = EuclideanDistances(values)
    assert source.shape == (17, 17) and len(source) == 17
    assert not source.is_materialized
    with pytest.raises(RuntimeError, match="synthetic failure"), retain_distances(source, source):
        assert not source.is_materialized
        with distance_matrix(source) as first:
            np.testing.assert_array_equal(first, expected)
            reference = weakref.ref(first)
            with distance_matrix(source) as second:
                assert first is second
            del first, second
        assert reference() is not None
        raise RuntimeError("synthetic failure")
    assert not source.is_materialized and reference() is None
    with distance_matrix(source) as recomputed:
        np.testing.assert_array_equal(recomputed, expected)
    assert not source.is_materialized


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_block_k_distances_match_eager_partition_without_retaining_matrix(monkeypatch, dtype):
    values = np.random.default_rng(3).normal(size=(29, 5))
    values[1] = values[0]
    matrix = squareform(pdist(values)).astype(dtype)
    before = matrix.copy()
    monkeypatch.setattr(algorithms, "K_DISTANCE_BLOCK_ROWS", 4)
    for minimum in (2, 3, 7, 29):
        eager = matrix.copy()
        np.fill_diagonal(eager, np.inf)
        expected = np.partition(eager, minimum-2, axis=1)[:, minimum-2]
        actual = k_distances(matrix, minimum)
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == expected.dtype
        assert actual.flags.owndata and actual.base is None
    np.testing.assert_array_equal(matrix, before)
    matrix[-1, 0] = np.nan
    with pytest.raises(ValueError, match="Invalid distances"):
        k_distances(matrix, 3)


@pytest.mark.parametrize("pattern", ["noise", "single", "singletons", "mixed", "medoid_tie"])
def test_lazy_metrics_and_exemplars_equal_eager_including_degeneracy(family, pattern):
    lazy = lazy_family(family)
    labels = {
        "noise": [-1]*90, "single": [0]*90, "singletons": list(range(90)),
        "mixed": [0]*25+[1]*25+[2]*25+[-1]*15,
        "medoid_tie": [0]*2+[-1]*88,
    }[pattern]
    labels = np.asarray(labels, dtype=np.int32)
    key = ("pacmap", 2)
    expected, expected_table = evaluate_clustering(labels, family.context, family.spaces[key].distances)
    actual, actual_table = evaluate_clustering(labels, lazy.context, lazy.spaces[key].distances)
    assert actual == expected
    pd.testing.assert_frame_equal(actual_table, expected_table, check_exact=True)
    if pattern == "medoid_tie":
        assert actual_table.iloc[0].medoid_content_id == min(family.context.content_ids[:2])
    for row in expected_table.itertuples():
        medoid = family.context.content_ids.index(row.medoid_content_id)
        assert exemplar_members(labels, row.cluster_id, medoid, lazy.context) == exemplar_members(labels, row.cluster_id, medoid, family.context)
    assert_released(lazy)


def test_only_requested_spaces_materialize_and_cache_source_checks_survive(family, tmp_path, monkeypatch):
    lazy = lazy_family(family)
    allocations = watch_matrices(monkeypatch, {"synthetic": lazy})
    space = lazy.spaces[("tsne", 1)]
    config = ClusteringConfig("hdbscan", {"min_samples": 5, "min_cluster_size": 10})
    effective_parameters(config, space.distances)
    assert not allocations  # OPTICS/HDBSCAN need population, not a matrix, here.
    path = run_to_store(lazy, space, config, tmp_path/"runs")
    assert_released(lazy)
    assert {name[1:] for name, _ in allocations} == {("original_l2", None), ("tsne", 1)}
    assert all(ref() is None for _, ref in allocations)
    before = {p.name: file_sha256(p) for p in path.iterdir()}
    allocations.clear()
    assert run_to_store(lazy, space, config, tmp_path/"runs") == path
    assert not allocations  # Valid HDBSCAN publication reuse needs no distances.
    assert before == {p.name: file_sha256(p) for p in path.iterdir()}
    assert verify_run(path, lazy)["quality_valid"]
    assert_released(lazy)
    changed = replace(space, signatures={**space.signatures, "reduction": "changed"})
    incompatible = replace(lazy, spaces={**lazy.spaces, ("tsne", 1): changed})
    assert not verify_run(path, incompatible)["source_matches"]
    with pytest.raises(ValueError, match="incompatible"):
        run_to_store(incompatible, changed, config, tmp_path/"runs")
    assert_released(lazy)
    assert all(ref() is None for _, ref in allocations)


def test_screening_comparison_match_head_737e588_and_release_between_seeds(family, tmp_path, monkeypatch):
    lazy = lazy_family(family)
    families = {"synthetic": lazy}
    allocations = watch_matrices(monkeypatch, families)
    root = tmp_path/"lazy"
    screen = screening_to_store(families, configs(), root)
    assert_released(lazy)
    names = [key[1:] for key, _ in allocations]
    assert names.count(("original_l2", None)) == 3  # Reused across configs, released per representation.
    assert names.count(("tsne", 0)) == names.count(("pacmap", 0)) == 1
    assert all(seed in (None, 0) for _, seed in names)
    assert all(ref() is None for _, ref in allocations)
    comparison = comparison_to_store(screen, families, root)
    assert_released(lazy)
    result = snapshot(screen, comparison)
    oracle = read_json(Path(__file__).parent/"fixtures"/"clustering_memory_737e588.json")
    if implementation_versions() == oracle["versions"]:
        assert set(result["runs"]) == set(oracle["runs"])
        for key, run in result["runs"].items():
            for field, value in run.items():
                assert digest(value) == oracle["runs"][key][field], (key, field)
        for key, value in result["tables"].items():
            assert digest(value) == oracle["tables"][key], key
    else:
        # Library versions are already part of clustering identity. Compare the
        # legacy eager input path under this environment, never rewrite the oracle.
        eager_screen = screening_to_store({"synthetic": family}, configs(), tmp_path/"eager")
        eager_compare = comparison_to_store(eager_screen, {"synthetic": family}, tmp_path/"eager")
        assert result == snapshot(eager_screen, eager_compare)
    assert verify_collection(screen, families)["quality_valid"]
    assert verify_collection(comparison, families)["quality_valid"]
    assert_released(lazy)
    assert all(ref() is None for _, ref in allocations)
    # Signature-only comparison/cache validation must not request any matrices.
    allocations.clear()
    assert comparison_to_store(screen, families, root) == comparison
    assert not allocations
    changed = replace(lazy.spaces[("pacmap", 0)], reduction_space_id="changed")
    incompatible = replace(lazy, spaces={**lazy.spaces, ("pacmap", 0): changed})
    with pytest.raises(ValueError, match="reduction sources"):
        comparison_to_store(screen, {"synthetic": incompatible}, root)
    assert not allocations


def test_two_encoders_and_failing_screening_never_retain_previous_spaces(family, tmp_path, monkeypatch):
    from flir_pipeline.clustering import storage

    families = {name: lazy_family(replace(family, source=replace(family.source, feature={
        **family.source.feature, "extractor": name, "feature_space_id": f"synthetic-{name}-feature"})))
                for name in ("dinov2", "clip")}
    allocations = watch_matrices(monkeypatch, families)
    config = ClusteringConfig("hdbscan", {"min_samples": 5, "min_cluster_size": 10})
    screen = screening_to_store(families, [ClusteringConfig("dbscan", {"min_samples": 5, "eps_quantile": .8}), config], tmp_path/"two")
    assert verify_collection(screen, families)["quality_valid"]
    assert_released(*families.values())
    assert all(ref() is None for _, ref in allocations)

    def fail(*args):
        raise RuntimeError("synthetic fit failure")

    monkeypatch.setattr(storage, "fit_clustering", fail)
    with pytest.raises(RuntimeError, match="synthetic fit failure"):
        screening_to_store(families, [ClusteringConfig("dbscan", {"min_samples": 5, "eps_quantile": .8})], tmp_path/"failure")
    assert_released(*families.values())
    assert all(ref() is None for _, ref in allocations)


def test_load_families_keeps_existing_reductions_read_only_and_distances_lazy(tmp_path, monkeypatch):
    pytest.importorskip("pacmap")
    from flir_pipeline.clustering import distances

    inputs, manifest = synthetic_inputs(tmp_path)
    grid = [ReductionConfig(method, seed=seed, hyperparameters=params)
            for method, params in (("tsne", {"perplexity": 10, "max_iter": 350}),
                                   ("pacmap", {"num_iters": [10, 10, 20]}))
            for seed in (0, 1, 2)]
    root = tmp_path/"reductions"
    benchmark = benchmark_to_store(inputs, grid, root)
    before = {p.relative_to(root): file_sha256(p) for p in root.rglob("*") if p.is_file()}

    def reject(*args, **kwargs):
        raise AssertionError("Loading a clustering family must not materialize distances")

    monkeypatch.setattr(distances, "pdist", reject)
    spec = tmp_path/"inputs.yaml"
    spec.write_text(yaml.safe_dump({"manifest": str(manifest), "encoders": {"synthetic": {
        "feature_directory": str(inputs.feature_directory), "similarity_directory": str(inputs.similarity_directory),
        "reduction_benchmark": str(benchmark)}}}), encoding="utf-8")
    loaded = load_families(spec)["synthetic"]
    assert len(loaded.spaces) == 7
    assert_released(loaded)
    assert loaded.context.original_distances is loaded.spaces[("original_l2", None)].distances
    assert loaded.source.similarity["config"]["algorithm_version"] == "content_cosine_v1"
    assert all(isinstance(space.values, np.memmap) for space in loaded.spaces.values())
    assert before == {p.relative_to(root): file_sha256(p) for p in root.rglob("*") if p.is_file()}
    # Full upstream verification is still active, even though clustering is lazy.
    del loaded  # Release read-only Windows file mappings before synthetic corruption.
    coordinate = next(root.rglob("coordinates.npy"))
    values = np.load(coordinate)
    values[0, 0] += 1
    np.save(coordinate, values)
    with pytest.raises(ValueError, match="benchmark failed verification"):
        load_family(inputs.feature_directory, inputs.similarity_directory, manifest, benchmark)


def test_video_ids_never_become_sequences_or_selection_metrics(family):
    from flir_pipeline.clustering.selection import shortlist_screening

    lazy = lazy_family(family)
    provenance = pd.DataFrame({"content_id": family.context.content_ids,
                               "temporal_source": "sampled_video_grid", "video_id": ["v1"]*45+["v2"]*45})
    labels = np.asarray([0]*25+[1]*25+[2]*25+[-1]*15, dtype=np.int32)
    results = []
    for video_ids in (provenance.video_id, ["same-source"]*90, [f"v{i}" for i in range(90)]):
        context = EvaluationContext.create(lazy.context.content_ids, lazy.context.original_distances,
                                           lazy.context.cosine, lazy.context.neighbors,
                                           provenance.assign(video_id=video_ids))
        assert context.temporal_pairs == {}
        metrics, table = evaluate_clustering(labels, context, lazy.spaces[("tsne", 0)].distances)
        assert metrics["temporal_recall@5"] is None
        assert metrics["weighted_dominant_sequence_fraction"] is None
        assert metrics["clustered_sequence_coverage"] is None
        assert metrics["historical_multisplit_clusters"] is None
        assert table.sequence_count.isna().all() and table.historical_split_count.isna().all()
        _, _, omitted = shortlist_screening(pd.DataFrame([{
            "encoder": "synthetic", "representation": "tsne", "algorithm": "dbscan", "configuration_id": "test", **metrics}]))
        assert "temporal_recall@5" in str(omitted)
        results.append(metrics)
    assert results[0] == results[1] == results[2]
    assert_released(lazy)


def test_degenerate_silhouette_and_empty_summary_need_no_distances(family, monkeypatch):
    from flir_pipeline.clustering import distances

    lazy = lazy_family(family)

    def reject(*args, **kwargs):
        raise AssertionError("No distance is needed for a degenerate silhouette or empty summary")

    monkeypatch.setattr(distances, "pdist", reject)
    for labels in (np.full(90, -1), np.zeros(90, dtype=int), np.arange(90)):
        assert silhouette_without_noise(lazy.context.original_distances, labels) is None
    assert cluster_summary(np.full(90, -1), lazy.context).empty
