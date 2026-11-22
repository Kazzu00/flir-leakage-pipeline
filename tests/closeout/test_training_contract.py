"""Exact missing-count contract through native materialization and YOLO boundary."""

from pathlib import Path

import pandas as pd
import pytest
from PIL import Image
from test_oe4_authorization import operation as operation

from flir_pipeline.closeout import oe4_runtime, training
from flir_pipeline.closeout.io import read_json
from flir_pipeline.closeout.oe4 import AUG_ZERO, REQUIRED, aug_off
from flir_pipeline.similarity.storage import file_sha256


def fixture_rows(tmp_path):
    rows = []
    for i in range(6):
        image, label = tmp_path / f"i{i}.png", tmp_path / f"i{i}.txt"
        Image.new("RGB", (32, 32), (i, 0, 0)).save(image)
        label.write_text("".join(f"{c} .5 .5 .1 .1\n" for c in range(i % 3)))
        rows.append(
            {
                "frame_id": f"frame_{i}",
                "basename": f"original_{i}",
                "new_split": "train" if i < 3 else "val",
                "image_path": str(image),
                "label_path": str(label),
                "image_sha256": file_sha256(image),
                "label_sha256": file_sha256(label),
            }
        )
    return pd.DataFrame(rows)


def test_missing_num_objects_is_derived_from_verified_labels(tmp_path):
    rows = fixture_rows(tmp_path)
    assert "num_objects" not in rows
    prepared = training.prepare(
        rows, tmp_path / "out", {"epochs": 50}, tmp_path / "yolo11m.pt"
    )
    assert prepared["counts"] == {f"frame_{i}": i % 3 for i in range(6)}
    assert "num_objects" not in rows  # Native write_view enriches its copy only.
    assert (
        pd.read_parquet(
            prepared["dataset"].parent / "records.parquet"
        ).num_objects.sum()
        == 6
    )
    assert not read_json(tmp_path / "out/preparation.json")["training_executed"]


def test_duplicate_occurrences_rejected_but_content_copies_retained(tmp_path):
    rows = fixture_rows(tmp_path)
    duplicate = pd.concat([rows, rows.iloc[:1]])
    with pytest.raises(ValueError, match="unique occurrences"):
        training.prepare(duplicate, tmp_path / "bad", {}, tmp_path / "weights")
    # Distinct historical occurrences may contain identical bytes.
    rows.loc[1, "image_sha256"] = rows.loc[0, "image_sha256"]
    Path(rows.loc[1, "image_path"]).write_bytes(
        Path(rows.loc[0, "image_path"]).read_bytes()
    )
    prepared = training.prepare(rows, tmp_path / "valid", {}, tmp_path / "weights")
    assert len(prepared["counts"]) == 6


@pytest.mark.parametrize("column", sorted(training.REQUIRED_COLUMNS))
def test_every_consumed_source_field_checked(tmp_path, column):
    rows = fixture_rows(tmp_path).drop(columns=[column])
    with pytest.raises(ValueError, match="Missing development fields"):
        training.prepare(rows, tmp_path / "out", {}, tmp_path / "weights")


@pytest.mark.parametrize("corruption", ["labels", "images", "syntax"])
def test_actual_yolo_bytes_and_images_rejected_on_corruption(tmp_path, corruption):
    rows = fixture_rows(tmp_path)
    if corruption == "images":
        Path(rows.iloc[0].image_path).write_bytes(b"corrupt")
    else:
        Path(rows.iloc[0].label_path).write_text("invalid YOLO")
        if corruption == "syntax":
            rows.loc[0, "label_sha256"] = file_sha256(Path(rows.iloc[0].label_path))
    with pytest.raises((ValueError, OSError)):
        training.prepare(rows, tmp_path / "out", {}, tmp_path / "weights")


def pinned_baseline():
    pytest.importorskip("ultralytics")
    from ultralytics.cfg import DEFAULT_CFG_DICT

    return {**DEFAULT_CFG_DICT, **REQUIRED, "amp": False, "workers": 0, "cache": False}


def test_actual_pinned_config_loader_and_complete_invocation_boundary(
    tmp_path, monkeypatch
):
    baseline = pinned_baseline()
    prepared = training.prepare(
        fixture_rows(tmp_path),
        tmp_path / "out",
        aug_off(baseline),
        tmp_path / "yolo11m.pt",
    )
    result = training.check_cpu(prepared, baseline)
    assert result["state"] == "READY_TO_INVOKE_TRAIN"
    assert all(result["effective_augmentation"][key] == 0 for key in AUG_ZERO)
    assert result["effective_augmentation"]["auto_augment"] is None
    assert result["effective_augmentation"]["multi_scale"] is False

    class InvocationReached(Exception):
        pass

    class Boundary:
        task = "detect"
        model = type("Model", (), {"yaml": {"scale": "m"}})()

        def __init__(self, checkpoint):
            assert checkpoint == prepared["settings"]["model"]
            self.callbacks = {}

        def add_callback(self, event, callback):
            self.callbacks[event] = callback

        def train(self, **arguments):
            assert arguments == prepared["settings"]
            assert arguments["epochs"] == 50 and arguments["seed"] == 42
            assert {
                "on_train_start",
                "on_train_batch_end",
                "on_train_epoch_start",
            } <= set(self.callbacks)
            raise InvocationReached("Reached exact model.train boundary; no training")

    import ultralytics

    monkeypatch.setattr(ultralytics, "YOLO", Boundary)
    with pytest.raises(InvocationReached):
        oe4_runtime.execute_prepared(
            prepared, {"policy": "fixture", "augmentation": "AUG_OFF"}, lambda: None
        )
    assert read_json(tmp_path / "out/receipt.json")["state"] == "FAILED_TRAINING"


def test_upstream_duplicate_annotation_drop_detected_before_gpu(tmp_path):
    baseline = pinned_baseline()
    rows = fixture_rows(tmp_path)
    label = Path(rows.iloc[1].label_path)
    label.write_text("0 .5 .5 .1 .1\n0 .5 .5 .1 .1\n")
    rows.loc[1, "label_sha256"] = file_sha256(label)
    prepared = training.prepare(
        rows, tmp_path / "out", aug_off(baseline), tmp_path / "weights"
    )
    with pytest.raises(ValueError, match="canonical annotation support"):
        training.check_cpu(prepared, baseline)


@pytest.mark.parametrize(
    "index,policy", [(0, "historica"), (1, "aleatoria"), (2, "c10")]
)
def test_worker_uses_each_frozen_manifest_and_actual_preparation(
    operation, monkeypatch, index, policy
):
    pytest.importorskip("ultralytics")
    config, cfg, _ = operation
    monkeypatch.setattr(oe4_runtime, "gpu_preflight", lambda: {"test_gpu_stub": True})

    def boundary(prepared, state, verify_after):
        assert state["policy"] == policy and state["augmentation"] == "AUG_OFF"
        assert prepared["counts"] == {"train_frame": 1, "val_frame": 1}
        assert prepared["output"] == Path(cfg["aug_off_output"]) / policy
        assert prepared["settings"]["model"] == cfg["initial_weights"]
        assert (
            prepared["settings"]["seed"] == 42 and prepared["settings"]["epochs"] == 50
        )
        verify_after()
        return {"state": "INVOCATION_BOUNDARY_VALIDATED"}

    monkeypatch.setattr(oe4_runtime, "execute_prepared", boundary)
    assert oe4_runtime.train(config, index)["state"] == "INVOCATION_BOUNDARY_VALIDATED"
