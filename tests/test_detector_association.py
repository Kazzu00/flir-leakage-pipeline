"""Offline evidence gates with real synthetic artifact verification, no models."""

import copy
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from flir_pipeline.detection.association import (
    ASSOCIATION_COLUMNS,
    COMPLETE,
    PARTIAL,
    PENDING,
    POPULATIONS,
    frozen_context,
    join_run,
    load_evidence,
    load_registry,
    prespecified_table,
    select_view,
    strategy_summaries,
    validate_binding,
    view_label,
)
from flir_pipeline.detection.association_plot import association_figure
from flir_pipeline.detection.metrics import (
    ImageStats,
    bootstrap,
    evaluate,
    save_image_stats,
)
from flir_pipeline.detection.protocol import detector_run_id, load_config
from flir_pipeline.detection.reporting import generate_report
from flir_pipeline.similarity.storage import (
    file_sha256,
    read_json,
    stable_id,
    write_json,
)


@pytest.fixture
def project(tmp_path):
    artifacts = tmp_path/"detection"
    protocol = artifacts/"protocol"
    protocol.mkdir(parents=True)
    splits = []
    for strategy in ("historical", "random_content", "C10", "C12"):
        for seed in ([0] if strategy == "historical" else range(5)):
            sid = f"synthetic-{strategy}-{seed}"
            splits.append({"strategy": strategy, "split_seed": seed, "split_space_id": sid,
                           "dataset_id": "synthetic", "record_counts": {"train": 1, "val": 1, "test": 1},
                           "class_counts": [{"split": "test", "class_id": c, "instances": 1} for c in range(5)],
                           "split_metadata_sha256": "synthetic-source",
                           "context": {"exact_duplicate_cross_split_count": 0, "dinov2_nn_mean": .6+seed*.01,
                                       "clip_nn_mean": .8+seed*.01, "dinov2_top001_pairs": 6+seed,
                                       "clip_top001_pairs": 7+seed, "temporal_at5": .2, "class_deviation_pp": .5}})
    config = load_config(Path("configs/detection/yolo11n.yaml"))
    config["evaluation"]["bootstrap_resamples"] = 2  # bounded synthetic test protocol
    identity = {"config": config, "splits": splits}
    plan = {"identity": identity, "plan_id": stable_id(identity)}
    write_json(protocol/"plan.json", plan)
    frozen_config = copy.deepcopy(config)
    frozen_config["train"]["batch"] = 2
    model = {"protocol_config": frozen_config, "weights_sha256": "synthetic-pretrained"}
    freeze = {"plan_id": plan["plan_id"], "model_config": model, "model_config_id": stable_id(model)}
    write_json(protocol/"runtime_freeze.json", freeze)
    pd.DataFrame({"candidate_label": ["C10", "C12"]}).to_csv(protocol/"candidate_audit.csv", index=False)
    return artifacts, protocol, plan, freeze


def publish(project, split_index=0, seed=42):
    """Synthetic completion artifacts; numerical metrics verify against captured stats."""
    artifacts, _, plan, freeze = project
    split = plan["identity"]["splits"][split_index]
    sid = split["split_space_id"]
    run_id = detector_run_id(sid, freeze["model_config"], seed)
    run = artifacts/"runs"/run_id
    run.mkdir(parents=True)
    view = artifacts/"views"/sid
    view.mkdir(parents=True, exist_ok=True)
    frame_id = f"frame-{sid}"
    pd.DataFrame([{"frame_id": frame_id, "new_split": "test", "num_objects": 5}]).to_parquet(view/"records.parquet", index=False)
    write_json(view/"materialization.json", {"split_metadata_sha256": split["split_metadata_sha256"],
                                           "files": {"records.parquet": file_sha256(view/"records.parquet")}})
    images = [ImageStats(frame_id, np.ones(5)*.9, np.arange(5), np.ones((5, 10), dtype=bool),
                        np.arange(5), np.ones(5, dtype=int), np.ones(5, dtype=int))]
    overall, classes = evaluate(images)
    config = freeze["model_config"]["protocol_config"]
    write_json(run/"metrics.json", {"overall": overall, "definition": config["evaluation"],
                                   "test_threshold_optimized": False, "checkpoint_selection": "validation_only"})
    classes.to_parquet(run/"metrics_per_class.parquet", index=False)
    save_image_stats(images, run/"test_image_stats.npz")
    intervals, draws = bootstrap(images, 2, config["evaluation"]["bootstrap_seed"])
    write_json(run/"bootstrap.json", intervals)
    draws.to_parquet(run/"bootstrap_samples.parquet", index=False)
    write_json(run/"training_summary.json", {"epochs_completed": config["train"]["epochs"],
                                           "initial_weights_sha256": freeze["model_config"]["weights_sha256"],
                                           "effective_arguments": {**config["train"], "seed": seed}})
    (run/"train/weights").mkdir(parents=True)
    for name in ("weights/best.pt", "results.csv", "args.yaml"):
        (run/"train"/name).write_text("synthetic test fixture", encoding="utf-8")
    files = ["metrics.json", "metrics_per_class.parquet", "bootstrap.json", "bootstrap_samples.parquet",
             "test_image_stats.npz", "training_summary.json", "train/weights/best.pt", "train/results.csv", "train/args.yaml"]
    meta = {"detector_run_id": run_id, "plan_id": plan["plan_id"],
            "identity": {"split_space_id": sid, "training_seed": seed, "model_configuration": freeze["model_config"]},
            "strategy": split["strategy"], "split_seed": split["split_seed"], "detector_seed": seed,
            "dataset_id": split["dataset_id"], "context": split["context"], "state": "COMPLETE", "test_tuning": False,
            "materialization_sha256": file_sha256(view/"materialization.json"),
            "output_sha256": {n: file_sha256(run/n) for n in files}}
    write_json(run/"metadata.json", meta)
    return run


def test_pending_matrix_and_pilots_never_enter_scientific_table(project):
    artifacts, protocol, _, _ = project
    pilot = artifacts/"small_pilot/historical"
    pilot.mkdir(parents=True)
    write_json(pilot/"pilot.json", {"scientific_result": False})
    copied = artifacts/"runs/copied-pilot"
    copied.mkdir(parents=True)
    write_json(copied/"metadata.json", {"state": "SMALL_PILOT_VALIDATED", "scientific_result": False})
    evidence = load_evidence(protocol, artifacts)
    assert evidence.state == PENDING and evidence.excluded_pilots == 2
    assert len(evidence.matrix) == 48 and evidence.matrix.status.eq(PENDING).all()
    assert evidence.associations.empty and set(ASSOCIATION_COLUMNS) <= set(evidence.associations)
    assert len(evidence.context) == 16 and evidence.context.dinov2_nn_mean.notna().all()
    assert evidence.context.dinov2_top001_fraction.isna().all()
    assert strategy_summaries(evidence) == {}
    assert prespecified_table(evidence, load_registry()).n_runs.eq(0).all()
    fig = association_figure(evidence, load_registry())
    text = " ".join(t.get_text() for ax in fig.axes for t in ax.texts)
    assert "0/48 controlled runs" in text and "PENDING COMPUTE" in text
    assert "Heavy Machinery" in text and not fig.axes[0].collections
    plt.close(fig)


def test_partial_and_running_do_not_publish_metrics(project):
    artifacts, protocol, plan, freeze = project
    publish(project)
    split = plan["identity"]["splits"][1]
    run = artifacts/"runs"/detector_run_id(split["split_space_id"], freeze["model_config"], 43)
    run.mkdir(parents=True)
    write_json(run/"state.json", {"state": "TRAINING", "identity": {"split_space_id": split["split_space_id"],
                               "model_configuration": freeze["model_config"], "training_seed": 43}})
    evidence = load_evidence(protocol, artifacts)
    assert evidence.state == PARTIAL and evidence.fairness["completed_runs"] == 1
    assert evidence.matrix.status.value_counts().to_dict() == {PENDING: 46, "COMPLETE": 1, "RUNNING": 1}
    assert evidence.associations.empty
    assert select_view(evidence, "map50_95", "dinov2_nn_mean", -1).empty


def test_complete_controlled_comparison_schema_filtering_and_summary(project):
    artifacts, protocol, _, _ = project
    for split in range(16):
        for seed in (42, 43, 44):
            publish(project, split, seed)
    evidence = load_evidence(protocol, artifacts)
    assert evidence.state == COMPLETE and evidence.matrix.status.eq("COMPLETE").all()
    assert len(evidence.associations) == 288 and set(ASSOCIATION_COLUMNS) <= set(evidence.associations)
    assert not evidence.associations.duplicated(["detector_run_id", "class_id"]).any()
    for cid in POPULATIONS:
        selected = select_view(evidence, "recall", "dinov2_nn_mean", cid)
        assert len(selected) == 48 and selected.class_name.eq(POPULATIONS[cid]).all()
    machinery = select_view(evidence, "recall", "clip_nn_mean", 4, "C10")
    assert len(machinery) == 15 and machinery.y.eq(1).all()
    assert prespecified_table(evidence, load_registry()).n_runs.eq(48).all()
    summaries = strategy_summaries(evidence)
    assert summaries["strategy_summary"].n_detector_runs.tolist() == [3, 15, 15, 15]
    historical = summaries["split_seed_summary"].query("strategy == 'historical'")
    assert historical["std"].isna().all()  # one split, not zero variation
    fig = association_figure(evidence, load_registry())
    assert len(fig.axes) == 7
    assert all(sum(len(c.get_offsets()) for c in ax.collections) == 48 for ax in fig.axes)
    plt.close(fig)
    output = artifacts.parent/"report"
    metadata = generate_report(protocol, artifacts, output)
    assert metadata["association_state"] == COMPLETE and metadata["association_rows"] == 288
    assert len(pd.read_csv(output/"tables/detector_residual_associations.csv")) == 288
    # An incomplete rebuild must not expose stale CSVs from the earlier complete build.
    one = next((artifacts/"runs").iterdir())
    assert one.resolve().parent == (artifacts/"runs").resolve()
    shutil.rmtree(one)  # disposable synthetic fixture only
    metadata = generate_report(protocol, artifacts, output)
    assert metadata["association_state"] == PARTIAL
    assert not (output/"tables/detector_residual_associations.csv").exists()
    assert not (output/"tables/run_metrics.csv").exists()


@pytest.mark.parametrize("change", ["split", "context", "strategy", "duplicate", "checksum", "pilot", "metric_definition", "bootstrap"])
def test_invalid_binding_duplicate_and_corruption_rejected(project, change):
    artifacts, protocol, _, _ = project
    run = publish(project)
    meta = read_json(run/"metadata.json")
    if change == "split":
        meta["identity"]["split_space_id"] = "wrong-split"
    elif change == "context":
        meta["context"]["dinov2_nn_mean"] = .99
    elif change == "strategy":
        meta["strategy"] = "C12"
    elif change == "duplicate":
        shutil.copytree(run, artifacts/"runs/duplicate")
    elif change == "checksum":
        (run/"metrics.json").write_text("{}")
    elif change == "pilot":
        meta["scientific_result"] = False
    elif change in {"metric_definition", "bootstrap"}:
        name = "metrics.json" if change == "metric_definition" else "bootstrap.json"
        value = read_json(run/name)
        if change == "metric_definition":
            value["definition"]["precision_recall_confidence"] = .5
        else:
            value["seed"] += 1
        write_json(run/name, value)
        meta["output_sha256"][name] = file_sha256(run/name)
    write_json(run/"metadata.json", meta)
    evidence = load_evidence(protocol, artifacts)
    assert not evidence.complete and evidence.associations.empty
    assert evidence.fairness["completed_runs"] == 0
    if change != "pilot":
        assert "INVALID" in evidence.matrix.status.values
        assert evidence.issues


def test_explicit_mismatched_split_rejection_and_missing_metrics(project):
    _, _, plan, freeze = project
    run = publish(project)
    meta = read_json(run/"metadata.json")
    with pytest.raises(ValueError, match="split_space_id mismatch"):
        validate_binding(meta, plan["identity"]["splits"][1], plan, freeze)
    overall = read_json(run/"metrics.json")["overall"]
    classes = pd.read_parquet(run/"metrics_per_class.parquet")
    classes.loc[classes.class_id.eq(4), ["support", "precision", "recall", "map50", "map50_95"]] = [0, None, None, None, None]
    overall.update(supported_classes=4, precision=None, recall=None, map50=None, map50_95=None)
    rows = join_run(meta, overall, classes, frozen_context(plan))
    assert rows.loc[rows.class_id.isin([-1, 4]), "recall"].isna().all()
    classes.loc[classes.class_id.eq(4), "recall"] = 0
    with pytest.raises(ValueError, match="undefined"):
        join_run(meta, overall, classes, frozen_context(plan))
    meta["scientific_result"] = False
    with pytest.raises(ValueError, match="Small pilots"):
        join_run(meta, overall, classes, frozen_context(plan))


def test_prespecified_vs_exploratory_is_exact_and_not_outcome_selected():
    registry = load_registry()
    assert len(registry) == 7
    for row in registry.itertuples():
        assert view_label(registry, row.detector_metric, row.residual_metric, row.class_id) == "PRESPECIFIED ASSOCIATION"
    assert view_label(registry, "recall", "dinov2_nn_mean", -1) == "EXPLORATORY VIEW"
    assert view_label(registry, "recall", "dinov2_nn_mean", 0) == "EXPLORATORY VIEW"
    assert view_label(registry, "map50_95", "class_deviation_pp", -1) == "EXPLORATORY VIEW"


def test_optional_fractions_are_copied_only_from_bound_stored_cohorts(project, tmp_path):
    _, _, plan, _ = project
    plan = copy.deepcopy(plan)
    split = plan["identity"]["splits"][0]
    directory = tmp_path/"splits"/split["split_space_id"]
    directory.mkdir(parents=True)
    name = "quantile_pairs_dinov2.parquet"
    pd.DataFrame([{"quantile": .999, "threshold_cosine": .91, "cross_split_count": 6,
                   "cross_split_fraction": .006}]).to_parquet(directory/name, index=False)
    write_json(directory/"metadata.json", {"split_space_id": split["split_space_id"], "output_sha256": {name: file_sha256(directory/name)}})
    split["split_metadata_sha256"] = file_sha256(directory/"metadata.json")
    context = frozen_context(plan, tmp_path/"splits")
    assert context.dinov2_top001_fraction.iloc[0] == .006
    assert context.clip_top001_fraction.isna().all()
    (directory/name).write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        frozen_context(plan, tmp_path/"splits")
