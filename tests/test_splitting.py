"""Offline scientific invariants for atomic splitting and posterior evaluation."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from flir_pipeline.cli import app
from flir_pipeline.data.identity import dataset_id_from_manifest
from flir_pipeline.similarity.cosine import compute_topk_neighbors
from flir_pipeline.similarity.storage import file_sha256, read_json, write_json
from flir_pipeline.splitting.base import (
    SplitConfig,
    identity_payload,
    split_space_id,
    targets_from_manifest,
)
from flir_pipeline.splitting.construction import (
    aggregate_groups,
    make_groups,
    milp_assignment,
    propagate,
    random_assignment,
    record_statistics,
)
from flir_pipeline.splitting.metrics import (
    balance_metrics,
    cluster_fracture,
    residual_similarity,
    temporal_metrics,
    verify_assignments,
)
from flir_pipeline.splitting.selection import (
    DIVERSITY_COLUMNS,
    select_clustering_candidates,
    select_final_candidates,
)
from flir_pipeline.splitting.storage import (
    SimilaritySource,
    SplitInputs,
    build_run,
    verify_split,
)


@pytest.fixture
def source():
    ids = [f"c{i:02d}" for i in range(15)]
    rows, boxes = [], []
    for i in range(16):
        cid = ids[i % 15]
        count = 0 if i % 6 == 0 else 2
        split = ("train", "val", "test")[i % 3]
        if i == 15:
            split = "test"  # exact content with conflicting label, in another historical split
        rows.append({"frame_id": f"f{i:02d}", "content_id": cid, "original_split": split,
                     "image_sha256": cid, "label_sha256": f"label{i}", "num_objects": count, "label_empty": count == 0})
        boxes.extend({"frame_id": f"f{i:02d}", "class_id": i % 5} for _ in range(count))
    manifest = pd.DataFrame(rows)
    stats = record_statistics(manifest, pd.DataFrame(boxes))
    rng = np.random.default_rng(7)
    x = rng.normal(size=(15, 5)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    matrix = x @ x.T
    matrix = np.clip(matrix, -1, 1)
    np.fill_diagonal(matrix, 1)
    pair_values = matrix[np.triu_indices(15, 1)]
    thresholds = pd.DataFrame([{"quantile": q, "top_percentage": 100*(1-q), "threshold_cosine": float(np.quantile(pair_values, q)),
                                "pair_count": int((pair_values >= np.float32(np.quantile(pair_values, q))).sum())}
                               for q in (.9, .95, .975, .99, .995, .999)])
    provenance = pd.DataFrame({"content_id": ids, "sequence_key": ["sequence_a"]*10+["sequence_b"]*5,
                               "sequence_provenance_valid": True, "frame_index": list(range(10))+list(range(5)), "frame_index_valid": True})
    similarity = SimilaritySource(ids, matrix, compute_topk_neighbors(matrix, ids, 14), thresholds, provenance, {"similarity_space_id": "synthetic"})
    return SplitInputs(manifest, stats, dataset_id_from_manifest(manifest), {"dinov2": similarity, "clip": similarity}, pd.DataFrame(), {}, {}, {})


def test_record_counts_preserve_duplicate_annotation_conflicts(source):
    groups = make_groups(sorted(source.manifest.content_id.unique()))
    units = aggregate_groups(source.statistics, groups)
    duplicate = units.set_index("group_id").loc["content:c00"]
    assert duplicate.record_count == 2 and duplicate.content_count == 1
    assert duplicate.class_0 == 2 and duplicate.empty_annotation_count == 1
    assert units.record_count.sum() == len(source.manifest)
    config = SplitConfig()
    targets = targets_from_manifest(source.manifest, config)
    np.testing.assert_allclose(targets, [5/16, 5/16, 6/16])


def test_noise_and_clusters_remain_distinct_atomic_units(source):
    labels = pd.Series([0]*5 + [1]*4 + [-1]*6, index=source.similarities["clip"].content_ids)
    groups = make_groups(labels.index.tolist(), labels)
    assert groups.loc[groups.cluster_id == -1, "group_id"].nunique() == 6
    assert groups.loc[groups.cluster_id == 0, "group_id"].nunique() == 1
    assignment = {g: ("train", "val", "test")[i % 3] for i, g in enumerate(sorted(groups.group_id.unique()))}
    contents, records = propagate(source.statistics, groups, assignment)
    quality = verify_assignments(contents, records, source.manifest, "cluster_aware", labels)
    assert quality["exact_duplicate_cross_split_count"] == quality["cluster_fracture_count"] == 0
    assert records.groupby("content_id").new_split.nunique().max() == 1
    fractured = contents.copy()
    fractured.loc[0, "new_split"] = "test" if contents.loc[0, "new_split"] != "test" else "val"
    bad_records = records.copy()
    bad_records["new_split"] = bad_records.content_id.map(fractured.set_index("content_id").new_split)
    with pytest.raises(ValueError, match="fracture"):
        verify_assignments(fractured, bad_records, source.manifest, "cluster_aware", labels)


def test_random_is_deterministic_and_targets_records(source):
    groups = make_groups(source.similarities["clip"].content_ids)
    units = aggregate_groups(source.statistics, groups)
    ratios = targets_from_manifest(source.manifest, SplitConfig())
    a, _ = random_assignment(units, ratios, 2)
    b, _ = random_assignment(units.sample(frac=1, random_state=9), ratios, 2)
    assert a == b
    assert a != random_assignment(units, ratios, 3)[0]
    contents, records = propagate(source.statistics, groups, a)
    verify_assignments(contents, records, source.manifest, "random_content")
    summary, table = balance_metrics(records, source.statistics, ratios)
    assert max(s["absolute_record_deviation"] for s in summary["sizes"]) <= 2
    assert len(table) == 15 and set(table.class_id) == set(range(5))


def test_milp_seeded_profiles_are_reproducible(source):
    ids = source.similarities["clip"].content_ids
    labels = pd.Series([0]*3 + [1]*3 + [-1]*9, index=ids)
    groups = make_groups(ids, labels)
    units = aggregate_groups(source.statistics, groups)
    config = SplitConfig(node_limit=8)
    ratios = targets_from_manifest(source.manifest, config)
    a, first = milp_assignment(units, ratios, 0, config)
    b, second = milp_assignment(units.sample(frac=1, random_state=3), ratios, 0, config)
    assert a == b and first["objective"] == second["objective"]
    assert not first["unique_optimum_proven"]
    c, r = propagate(source.statistics, groups, a)
    assert verify_assignments(c, r, source.manifest, "cluster_aware", labels)["quality_valid"]


def test_milp_unknown_stopping_code_requires_primal_feasibility(source, monkeypatch):
    import flir_pipeline.splitting.construction as construction
    units = aggregate_groups(source.statistics, make_groups(source.similarities["clip"].content_ids))
    real_solver = construction.milp
    def newer_highs_status(*args, **kwargs):
        result = real_solver(*args, **kwargs)
        result.status, result.success = 4, False
        result.message = "HiGHS Status 16: Solution limit reached"
        return result
    monkeypatch.setattr(construction, "milp", newer_highs_status)
    _, result = milp_assignment(units, np.array([.5,.25,.25]), 0, SplitConfig())
    assert result["primal_feasibility_verified"] and not result["optimal_within_tolerance"]
    def invalid_primal(*args, **kwargs):
        result = newer_highs_status(*args, **kwargs)
        result.x[0] = -100
        return result
    monkeypatch.setattr(construction, "milp", invalid_primal)
    with pytest.raises(ValueError, match="primal feasibility"):
        milp_assignment(units, np.array([.5,.25,.25]), 0, SplitConfig())


@pytest.mark.parametrize("corruption", ["missing_content", "extra_record", "wrong_lineage", "duplicate_cross_split", "invalid_split"])
def test_verifier_rejects_corrupt_assignments(source, corruption):
    groups = make_groups(source.similarities["clip"].content_ids)
    units = aggregate_groups(source.statistics, groups)
    a, _ = random_assignment(units, np.array([.5,.25,.25]), 0)
    c, r = propagate(source.statistics, groups, a)
    if corruption == "missing_content":
        c = c.iloc[1:]
    elif corruption == "extra_record":
        r = pd.concat([r, r.iloc[[0]]], ignore_index=True)
    elif corruption == "wrong_lineage":
        r.loc[0, "content_id"] = "c01"
    elif corruption == "duplicate_cross_split":
        r.loc[r.frame_id == "f15", "new_split"] = "val" if r.loc[r.frame_id == "f00", "new_split"].iloc[0] != "val" else "test"
    else:
        c.loc[0, "new_split"] = "validation"
    with pytest.raises((ValueError, AssertionError)):
        verify_assignments(c, r, source.manifest, "random_content")


def test_full_cross_nn_goes_beyond_stored_neighbors_and_retains_quantile_ties():
    matrix = np.array([[1,.99,.8,.2],[.99,1,.7,.3],[.8,.7,1,.95],[.2,.3,.95,1]], dtype=np.float32)
    ids = list("abcd")
    neighbors = compute_topk_neighbors(matrix, ids, 1)
    thresholds = pd.DataFrame([{"quantile": .5, "top_percentage": 50, "threshold_cosine": .8, "pair_count": 3}])
    metrics, pairs, nn = residual_similarity(matrix, neighbors, thresholds, np.array([1,1,2,4], dtype=np.int8), ids)
    assert nn.iloc[0].neighbor_content_id == "c"  # rank-1 b is inside train
    assert np.isclose(nn.iloc[0].cross_split_nn_similarity, .8)
    assert metrics["topk"]["1"]["fraction"] == .5
    assert pairs.iloc[0].cross_split_count == 2
    assert np.isclose(metrics["cross_split_nn"]["mean"], .85)


def test_historical_membership_sets_exclude_self_exact_pairs(source, tmp_path):
    run = build_run(source, SplitConfig(strategy="historical"), 0, tmp_path)
    q = read_json(run/"quality.json")
    assert q["exact_duplicate_cross_split_count"] == 1 and q["historical_exception"]
    c = pd.read_parquet(run/"split_assignments.parquet").set_index("content_id")
    assert pd.isna(c.loc["c00", "new_split"])
    assert c.loc["c00", "split_membership_set"] == '["test", "train"]'
    for e in ("dinov2", "clip"):
        nn = pd.read_parquet(run/f"cross_split_nn_{e}.parquet")
        assert (nn.content_id != nn.neighbor_content_id).all()
    assert verify_split(run, source)["quality_valid"]


def test_temporal_fragmentation_unknowns_and_cluster_fracture():
    provenance = pd.DataFrame({"sequence_key": ["a","a","a","b","b",""],
                               "sequence_provenance_valid": [True]*5+[False], "frame_index": [0,1,5,0,1,None]})
    masks = np.array([1,2,4,1,1,4], dtype=np.int8)
    summary, pairs = temporal_metrics(provenance, masks)
    assert summary["sequence_split_counts"] == {"1": 1, "2": 0, "3": 1}
    assert summary["unknown_sequence_content_count"] == 1
    assert pairs.iloc[0].pair_count == 2 and pairs.iloc[0].cross_split_count == 1
    assert pairs.iloc[1].pair_count == 4 and pairs.iloc[1].cross_split_count == 3
    fracture = cluster_fracture(np.array([0,0,-1,1,1,-1]), masks)
    assert fracture["cluster_count"] == 2 and fracture["fracture_rate"] == .5


def test_identity_seed_targets_and_solver_settings(source):
    config = SplitConfig()
    ratios = targets_from_manifest(source.manifest, config)
    payload = identity_payload(source.dataset_id, config, ratios, 0, "synthetic_cluster")
    original = split_space_id(payload)
    assert original == split_space_id(identity_payload(source.dataset_id, replace(config, seeds=(7,8)), ratios, 0, "synthetic_cluster"))
    assert original != split_space_id(identity_payload(source.dataset_id, config, ratios, 1, "synthetic_cluster"))
    assert original != split_space_id(identity_payload(source.dataset_id, replace(config, node_limit=1), ratios, 0, "synthetic_cluster"))
    assert original != split_space_id(identity_payload(source.dataset_id, config, np.array([.5,.25,.25]), 0, "synthetic_cluster"))


def test_storage_recomputes_semantics_even_after_checksum_repair(source, tmp_path):
    config = SplitConfig(strategy="random_content")
    run = build_run(source, config, 0, tmp_path)
    assert build_run(source, config, 0, tmp_path) == run
    assert verify_split(run, source)["quality_valid"]
    summary = read_json(run/"split_summary.json")
    summary["visual"]["dinov2"]["cross_split_nn"]["mean"] = -123
    write_json(run/"split_summary.json", summary)
    meta = read_json(run/"metadata.json")
    meta["output_sha256"]["split_summary.json"] = file_sha256(run/"split_summary.json")
    write_json(run/"metadata.json", meta)
    assert not verify_split(run, source)["quality_valid"]


def test_candidate_selection_is_order_invariant_and_covers_algorithms():
    rows = []
    for e in ("clip", "dinov2"):
        for rep in ("original_l2", "pacmap", "tsne"):
            for i, alg in enumerate(("dbscan", "optics", "hdbscan")):
                row = {k: (i+1)/4 for k in DIVERSITY_COLUMNS}
                row.update(encoder=e, representation=rep, algorithm=alg, clustering_space_id=f"{e}_{rep}_{i}",
                           pareto_stage_b=True, n_clusters_excluding_noise=(i+1)*3)
                rows.append(row)
    frame = pd.DataFrame(rows)
    a = select_clustering_candidates(frame)
    b = select_clustering_candidates(frame.sample(frac=1, random_state=9))
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == 12 and set(a.algorithm) == {"dbscan", "optics", "hdbscan"}


def test_final_pareto_uses_worst_seed_and_rejects_invalid_coverage():
    rows = []
    for candidate, loss in (("a", .1),("b",.2),("c",.01)):
        for seed in range(5):
            rows.append({"strategy":"cluster_aware", "clustering_space_id":candidate, "candidate_label":candidate,
                         "seed":seed,"split_space_id":f"{candidate}{seed}", "quality_valid":True,
                         "all_classes_covered": candidate != "c", "exact_duplicate_cross_split_count":0,
                         "cluster_fracture_count":0,"max_relative_record_deviation":.01,
                         "class_deviation_pp":loss,"dinov2_nn_mean":loss,"clip_nn_mean":loss,
                         "dinov2_top001_fraction":loss,"clip_top001_fraction":loss,"temporal_at5":loss})
    table, final = select_final_candidates(pd.DataFrame(rows))
    assert table.set_index("clustering_space_id").pareto.to_dict() == {"a":True,"b":False,"c":False}
    assert final.representative_split_space_id.tolist() == ["a0"]


@pytest.mark.parametrize("kwargs", [{"noise_policy":"similarity-components"}, {"target_ratios":[.7,.2,.2]}, {"seeds":[0,0]}, {"node_limit":0}])
def test_unapproved_or_invalid_config_rejected(kwargs):
    with pytest.raises(ValueError):
        SplitConfig(**kwargs)


def test_cli_surface_and_local_privacy():
    runner = CliRunner()
    for command in ("build", "baseline", "evaluate", "compare", "summary", "verify", "export-lists"):
        assert runner.invoke(app, ["splitting", command, "--help"]).exit_code == 0
    assert Path("configs/splits/random_baseline.yaml").is_file()


def test_export_only_references_existing_images_and_rejects_escape(source, tmp_path):
    from flir_pipeline.splitting.experiments import export_image_lists
    run = build_run(source, SplitConfig(strategy="random_content"), 0, tmp_path/"artifacts")
    images = tmp_path/"materialized"
    images.mkdir()
    manifest = source.manifest.copy()
    manifest["relative_image_path"] = manifest.frame_id + ".png"
    for name in manifest.relative_image_path:
        (images/name).write_bytes(b"synthetic placeholder; exporter does not decode")
    manifest_path = tmp_path/"manifest.parquet"
    manifest.to_parquet(manifest_path,index=False)
    before = {p.name:file_sha256(p) for p in images.iterdir()}
    export_image_lists(run,manifest_path,images,tmp_path/"lists")
    lines = [line for p in (tmp_path/"lists").glob("*.txt") for line in p.read_text().splitlines()]
    assert len(lines) == 16 and len(set(lines)) == 16
    assert before == {p.name:file_sha256(p) for p in images.iterdir()}
    manifest.loc[0,"relative_image_path"] = "../outside.png"
    (tmp_path/"outside.png").write_bytes(b"outside")
    manifest.to_parquet(manifest_path,index=False)
    with pytest.raises(ValueError,match="safe paths"):
        export_image_lists(run,manifest_path,images,tmp_path/"bad_lists")
    assert not (tmp_path/"bad_lists").exists()
