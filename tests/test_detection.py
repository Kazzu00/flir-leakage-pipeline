"""Offline scientific invariants; no torch, Ultralytics, downloads or real data."""

import copy
import hashlib
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from PIL import Image
from typer.testing import CliRunner

from flir_pipeline.cli import app
from flir_pipeline.data.identity import dataset_id_from_manifest
from flir_pipeline.detection.materialization import (
    materialize,
    plan_records,
    verify_view,
)
from flir_pipeline.detection.metrics import (
    ImageStats,
    aggregate_runs,
    bootstrap,
    evaluate,
    load_image_stats,
    save_image_stats,
)
from flir_pipeline.detection.protocol import (
    detector_run_id,
    experiment_matrix,
    load_config,
    verify_plan,
)
from flir_pipeline.detection.runtime import (
    fair_comparison,
    parse_training_history,
    pilot_subset,
)
from flir_pipeline.similarity.storage import file_sha256, stable_id, write_json


def image_stats(frame, classes=(0, 1, 2, 3, 4), correct=True):
    classes = np.asarray(classes, dtype=int)
    counts = np.bincount(classes, minlength=5)
    return ImageStats(frame, np.full(len(classes), .9), classes, np.full((len(classes), 10), correct, dtype=bool), classes,
                      counts, counts if correct else counts*0)


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    rows = []
    with zipfile.ZipFile(tmp_path/"Imagenes.zip", "w") as images, zipfile.ZipFile(tmp_path/"Etiquetas.zip", "w") as labels:
        for i in range(9):
            stream = io.BytesIO()
            Image.new("RGB", (16, 16), (i%8*20, 0, 0)).save(stream, format="PNG")
            content = stream.getvalue()
            annotation = b"" if i == 8 else f"{i%5} 0.5 0.5 0.25 0.25\n".encode()
            image_path, label_path = f"images/f{i}.png", f"labels/f{i}.txt"
            images.writestr(image_path, content)
            labels.writestr(label_path, annotation)
            ih, lh = hashlib.sha256(content).hexdigest(), hashlib.sha256(annotation).hexdigest()
            rows.append({"frame_id": f"f{i}", "content_id": ih, "image_sha256": ih, "label_sha256": lh,
                         "source_archive": "Imagenes.zip", "source_member_path": image_path, "relative_label_path": label_path,
                         "original_split": ("train", "val", "test")[i%3], "label_exists": True, "label_valid": True,
                         "image_decode_valid": True, "label_empty": not annotation, "num_objects": int(bool(annotation))})
    manifest = pd.DataFrame(rows)
    manifest_path = tmp_path/"manifest.parquet"
    manifest.to_parquet(manifest_path, index=False)
    split_dir = tmp_path/"split"
    split_dir.mkdir()
    assignments = manifest[["frame_id", "content_id", "original_split"]].rename(columns={"original_split": "new_split"})
    assignments.to_parquet(split_dir/"record_split_assignments.parquet", index=False)
    write_json(split_dir/"metadata.json", {"split_space_id": "synthetic_historical", "identity_payload": {
        "dataset_id": dataset_id_from_manifest(manifest), "configuration": {"strategy": "historical"}},
        "input_signatures": {"manifest": file_sha256(manifest_path)}})
    # Source split verification itself is covered by test_splitting; here exercise
    # byte-preserving materialization with real synthetic ZIP members.
    monkeypatch.setattr("flir_pipeline.detection.materialization.verify_split", lambda _: {"quality_valid": True})
    return tmp_path, manifest, assignments, split_dir, manifest_path


def test_materialization_preserves_conflicting_occurrence_labels_and_empty(dataset):
    root, manifest, assignments, source, manifest_path = dataset
    before = {n: file_sha256(root/n) for n in ("Imagenes.zip", "Etiquetas.zip")}
    view = materialize(source, manifest_path, root, root/"generated")
    receipt = verify_view(view)
    assert receipt["counts"] == {"train": 3, "val": 3, "test": 3}
    assert receipt["empty_labels"] == 1 and receipt["objects"] == 8
    assert before == {n: file_sha256(root/n) for n in before}
    config = yaml.safe_load((view/"dataset.yaml").read_text())
    assert config["names"][4] == "Heavy Machinery" and config["nc"] == 5
    records = pd.read_parquet(view/"records.parquet").set_index("frame_id")
    first, duplicate = Path(records.loc["f0", "image_path"]), Path(records.loc["f8", "image_path"])
    assert first.read_bytes() == duplicate.read_bytes()
    assert (duplicate.parent.parent/"labels/f8.txt").read_bytes() == b""
    assert (first.parent.parent/"labels/f0.txt").read_bytes() != b""
    assert materialize(source, manifest_path, root, root/"generated") == view
    duplicate.write_bytes(b"changed")
    with pytest.raises(ValueError, match="bytes changed"):
        verify_view(view)


def test_new_splits_reject_exact_overlap_and_incomplete_membership(dataset):
    _, manifest, assignments, _, _ = dataset
    with pytest.raises(ValueError, match="Exact content"):
        plan_records(manifest, assignments, "random_content")
    with pytest.raises(ValueError, match="coverage"):
        plan_records(manifest, assignments.iloc[:-1], "historical")
    changed = assignments.copy()
    changed.loc[0, "new_split"] = "val"
    with pytest.raises(ValueError, match="Historical"):
        plan_records(manifest, changed, "historical")


def test_materialization_rejects_wrong_manifest_and_source_mutation(dataset):
    root, manifest, _, source, manifest_path = dataset
    manifest.loc[0, "label_sha256"] = "wrong"
    manifest.to_parquet(manifest_path, index=False)
    with pytest.raises(ValueError, match="Manifest"):
        materialize(source, manifest_path, root, root/"generated")


def test_pilot_selection_stays_in_partition_and_preserves_present_classes(dataset):
    root, _, _, source, manifest_path = dataset
    view = materialize(source, manifest_path, root, root/"generated")
    records = pd.read_parquet(view/"records.parquet")
    selected = pilot_subset(records, 6, 42)
    assert len(selected) == 6 and set(selected.frame_id) <= set(records.frame_id)
    pd.testing.assert_frame_equal(selected, pilot_subset(records.sample(frac=1, random_state=9), 6, 42))


def test_run_identity_changes_with_science_not_order():
    config = {"weights_sha256": "synthetic", "train": {"batch": 2, "epochs": 50}}
    a = detector_run_id("split-a", config, 42)
    assert a == detector_run_id("split-a", dict(reversed(list(config.items()))), 42)
    assert a != detector_run_id("split-b", config, 42)
    assert a != detector_run_id("split-a", config, 43)
    changed = copy.deepcopy(config)
    changed["train"]["batch"] = 4
    assert a != detector_run_id("split-a", changed, 42)


def test_configuration_and_plan_tampering(tmp_path):
    config = load_config(Path("configs/detection/yolo11n.yaml"))
    assert config["train"]["batch"] is None  # the probe, not an assumed VRAM size, decides
    assert config["compute"]["cpu_stage_b_automatic"] is False
    identity = {"config": config}
    write_json(tmp_path/"plan.json", {"plan_id": stable_id(identity), "identity": identity})
    verify_plan(tmp_path)
    identity["config"]["train"]["epochs"] += 1
    write_json(tmp_path/"plan.json", {"plan_id": "unchanged", "identity": identity})
    with pytest.raises(ValueError, match="identity"):
        verify_plan(tmp_path)


def test_metrics_complete_failure_and_unsupported_class():
    perfect, table = evaluate([image_stats("a")])
    assert perfect["precision"] == perfect["recall"] == 1
    assert perfect["map50_95"] == pytest.approx(.995)
    assert table.class_name.iloc[4] == "Heavy Machinery"
    wrong, _ = evaluate([image_stats("b", correct=False)])
    assert all(wrong[m] == 0 for m in ("precision", "recall", "map50", "map50_95"))
    unsupported, classes = evaluate([image_stats("c", classes=(0, 1))])
    assert unsupported["map50"] is None and pd.isna(classes.iloc[4].recall)
    empty_prediction = image_stats("d")
    empty_prediction.confidence = np.empty(0)
    empty_prediction.predicted_class = np.empty(0, dtype=int)
    empty_prediction.true_positive = np.empty((0, 10), dtype=bool)
    empty_prediction.fixed_predictions[:] = 0
    empty_prediction.fixed_true_positives[:] = 0
    empty_prediction.validate()
    assert evaluate([empty_prediction])[0]["recall"] == 0


def test_metric_stats_roundtrip_and_reject_overmatching(tmp_path):
    path = tmp_path/"stats.npz"
    save_image_stats([image_stats("a"), image_stats("b", correct=False)], path)
    assert evaluate(load_image_stats(path))[0] == evaluate([image_stats("a"), image_stats("b", correct=False)])[0]
    sample = image_stats("c")
    sample.fixed_true_positives[0] = 2
    with pytest.raises(ValueError, match="matching"):
        sample.validate()
    with pytest.raises(ValueError, match="Duplicate"):
        save_image_stats([image_stats("a"), image_stats("a")], path)


def test_bootstrap_recomputes_pooled_ap_and_is_reproducible():
    samples = [image_stats("a"), image_stats("b", correct=False), image_stats("empty", classes=())]
    ci, draws = bootstrap(samples, 40, 12)
    same, draws_again = bootstrap(samples, 40, 12)
    assert ci == same
    pd.testing.assert_frame_equal(draws, draws_again)
    assert not draws.equals(bootstrap(samples, 40, 13)[1])
    assert len(draws) == 40*6 and any(r["valid_resamples"] < 40 for r in ci["intervals"])
    assert len(set(draws.loc[draws.class_id == -1, "map50"].dropna())) > 2


def test_hierarchical_aggregation_does_not_mix_variance_sources():
    rows = [{"strategy": "random_content", "split_seed": split, "detector_seed": seed, "class_id": -1,
             **dict.fromkeys(("precision", "recall", "map50", "map50_95"), value)}
            for split, seed, value in ((0, 42, .1), (0, 43, .3), (1, 42, .7), (1, 43, .9))]
    tables = aggregate_runs(pd.DataFrame(rows))
    within = tables["training_seed_summary"]
    assert within["std"].iloc[0] == pytest.approx(np.std([.1, .3], ddof=1))
    assert tables["split_seed_summary"]["std"].iloc[0] == pytest.approx(np.std([.2, .8], ddof=1))
    with pytest.raises(ValueError, match="Duplicate"):
        aggregate_runs(pd.DataFrame(rows+rows[:1]))


def test_matrix_and_fairness_gate_complete_cells():
    splits = [{"strategy": "historical", "split_seed": 0, "split_space_id": "h"}]
    splits += [{"strategy": strategy, "split_seed": seed, "split_space_id": f"{strategy}-{seed}"}
               for strategy in ("random_content", "C10", "C12") for seed in range(5)]
    matrix = experiment_matrix(splits, [42, 43, 44])
    assert len(matrix) == 48 and matrix.pilot.sum() == 4
    metas = [{"identity": {"split_space_id": r.split_space_id, "training_seed": r.detector_seed, "model_configuration": {"model": "same"}},
              "detector_seed": r.detector_seed, "state": "COMPLETE", "test_tuning": False, "dataset_id": "synthetic"} for r in matrix.itertuples()]
    assert fair_comparison(metas, matrix)["controlled"]
    assert fair_comparison([], matrix)["errors"] == ["missing_or_duplicate_experimental_cells"]
    assert not fair_comparison(metas[:-1], matrix)["controlled"]
    changed = copy.deepcopy(metas)
    changed[0]["identity"]["model_configuration"]["model"] = "different"
    assert not fair_comparison(changed, matrix)["controlled"]
    changed = copy.deepcopy(metas)
    changed[0]["test_tuning"] = True
    assert not fair_comparison(changed, matrix)["controlled"]


def test_training_parser_checks_budget_and_validation_selection(tmp_path):
    path = tmp_path/"results.csv"
    pd.DataFrame({"epoch": [1, 2, 3], "metrics/mAP50-95(B)": [.2, .4, .4]}).to_csv(path, index=False)
    summary = parse_training_history(path, 3)
    assert summary["epochs_completed"] == 3 and summary["best_epoch_from_rounded_csv"] == 3
    with pytest.raises(ValueError, match="epoch budget"):
        parse_training_history(path, 50)


def test_detector_cli_help_is_offline():
    result = CliRunner().invoke(app, ["detection", "--help"])
    assert result.exit_code == 0 and "pilot-small" in result.stdout and "freeze" in result.stdout


def test_cpu_stage_b_gate_precedes_training(tmp_path, monkeypatch):
    from flir_pipeline.detection.runtime import execute_matrix

    identity = {"config": {}, "splits": []}
    pid = stable_id(identity)
    write_json(tmp_path/"plan.json", {"plan_id": pid, "identity": identity})
    config = {"device": "cpu"}
    write_json(tmp_path/"runtime_freeze.json", {"plan_id": pid, "model_config": config, "model_config_id": stable_id(config)})
    monkeypatch.setattr("flir_pipeline.detection.runtime.run_one", lambda *args: pytest.fail("CPU gate allowed training"))
    with pytest.raises(ValueError, match="Automatic Stage B on CPU is disabled"):
        execute_matrix(tmp_path, tmp_path/"artifacts")
