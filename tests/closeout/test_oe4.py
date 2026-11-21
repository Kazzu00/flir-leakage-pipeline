from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from flir_pipeline.closeout.common_test import (
    development_records,
    overlap_audit,
    write_test_view,
)
from flir_pipeline.closeout.io import digest, seal, verify
from flir_pipeline.closeout.oe4 import AUG_ZERO, REQUIRED, aug_off, load_config
from flir_pipeline.closeout.oe4_runtime import (
    filter_shared,
    gpu_preflight,
    metric_exports,
)
from flir_pipeline.detection.common_executor import NAMES
from flir_pipeline.detection.metrics import ImageStats


def baseline():
    return {
        **REQUIRED,
        **dict.fromkeys(AUG_ZERO, 0),
        "mosaic": 1.0,
        "fliplr": 0.5,
        "close_mosaic": 10,
        "auto_augment": "randaugment",
        "copy_paste_mode": "flip",
        "erasing": 0.4,
        "multi_scale": False,
        "lr0": 0.01,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,
        "nbs": 64,
        "patience": 100,
        "amp": False,
        "deterministic": True,
        "workers": 0,
        "cache": False,
        "rect": False,
        "fraction": 1.0,
        "pretrained": True,
        "freeze": None,
        "single_cls": False,
        "box": 7.5,
        "cls": 0.5,
        "dfl": 1.5,
        "time": None,
        "cfg": None,
        "val": True,
        "save": True,
        "classes": None,
        "augment": False,
        "model": "yolo11m.pt",
        "data": "development/dataset.yaml",
    }


def test_aug_off_preserves_all_nonaugmentation_controls():
    source = baseline()
    settings = aug_off(source)
    assert all(settings[k] == 0 for k in AUG_ZERO)
    assert settings["auto_augment"] is None
    assert settings["copy_paste_mode"] == "flip"  # Inert selector, probability zero.
    for key, value in source.items():
        if key not in {*AUG_ZERO, "auto_augment", "model", "data", "resume"}:
            assert settings[key] == value
    assert source["mosaic"] == 1
    del source["bgr"]
    with pytest.raises(ValueError, match="Missing recorded"):
        aug_off(source)


@pytest.mark.parametrize(
    "key,value",
    [
        ("epochs", 49),
        ("seed", 0),
        ("resume", True),
        ("time", 1),
        ("classes", [0, 1]),
        ("cfg", "override.yaml"),
        ("pretrained", False),
    ],
)
def test_invalid_baseline_rejected(key, value):
    with pytest.raises(ValueError):
        aug_off({**baseline(), key: value})


def test_gpu_never_runs_on_login(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(ValueError, match="Slurm"):
        gpu_preflight()


def test_both_ground_truth_and_predictions_filtered():
    item = ImageStats(
        "one",
        np.array([0.9, 0.8, 0.7]),
        np.array([0, 1, 4]),
        np.ones((3, 10), dtype=bool),
        np.array([0, 1, 4]),
        np.array([1, 1, 0, 0, 1]),
        np.array([1, 1, 0, 0, 1]),
    )
    subset = filter_shared([item])[0]
    assert subset.predicted_class.tolist() == subset.target_class.tolist() == [0, 1]
    assert subset.fixed_predictions.tolist() == [1, 1, 0, 0, 0]
    result, classes, shared = metric_exports([item])
    assert shared.support.tolist() == [1, 1]
    assert result["shared_01_macro"]["precision"] == 1
    assert len(classes) == 5
    assert classes.loc[classes.class_id.eq(4), "class_name"].iloc[0] == "SDZI"
    assert item.target_class.tolist() == [0, 1, 4]


def test_overlap_does_not_claim_temporal_leakage():
    test = pd.DataFrame(
        [
            {
                "frame_id": "a",
                "content_id": "x",
                "possible_sequence": "seq",
                "possible_frame_index": 10,
            }
        ]
    )
    dev = pd.DataFrame(
        [
            {
                "frame_id": "b",
                "content_id": "y",
                "possible_sequence": "seq",
                "possible_frame_index": 11,
            }
        ]
    )
    report = overlap_audit(test, dev, [])
    assert report["hard_identity_disjoint"] and len(report["temporal_candidates"]) == 1
    assert report["independence_proven"] is False
    groups = pd.DataFrame(
        [{"frame_id": "a", "group_id": "same"}, {"frame_id": "b", "group_id": "same"}]
    )
    assert not overlap_audit(test, dev, [], protection_groups=groups)[
        "hard_identity_disjoint"
    ]


def test_materialized_view_and_subset_are_portable_and_bound(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "image.png").write_bytes(b"synthetic image")
    (source / "label.txt").write_bytes(b"0 0.5 0.5 0.1 0.1\n4 0.5 0.5 0.1 0.1\n")
    row = {
        "frame_id": "a",
        "image_path": "image.png",
        "label_path": "label.txt",
        "image_sha256": digest((source / "image.png").read_bytes()),
        "label_sha256": digest((source / "label.txt").read_bytes()),
    }
    dest = tmp_path / "view"
    records = write_test_view([row], dest, source, [0, 1])
    assert records[0]["num_objects"] == 1
    assert "path" not in yaml.safe_load((dest / "dataset.yaml").read_text())
    assert (dest / "test.txt").read_text() == "./images/test/a.png\n"
    seal(dest, {"state": "test"})
    verify(dest)
    (dest / "labels/test/a.txt").write_text("")
    with pytest.raises(ValueError, match="checksum"):
        verify(dest)


def test_development_origin_cannot_be_guessed(tmp_path):
    root = tmp_path / "development"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    (root / "images/a.png").write_bytes(b"a")
    (root / "train.txt").write_text("./images/a.png\n")
    (root / "val.txt").write_text("./images/a.png\n")
    (root / "dataset.yaml").write_text(
        yaml.safe_dump(
            {"train": "train.txt", "val": "val.txt", "nc": 5, "names": NAMES}
        )
    )
    with pytest.raises(ValueError, match="Unmapped"):
        development_records(
            root / "dataset.yaml", pd.DataFrame([{"frame_id": "other"}])
        )


def test_committed_config_loads():
    cfg = load_config(Path("configs/closeout/oe4_20261115.yaml"))
    assert cfg["evaluation"]["augment"] is False


@pytest.mark.parametrize("epoch", [-1, 49])
def test_last_checkpoint_final_stripping_is_not_mistaken_for_missing_epoch(epoch):
    from flir_pipeline.closeout.oe4 import check_last_checkpoint_metadata

    history = pd.DataFrame(
        {"epoch": list(range(1, 51)), "metrics/mAP50-95(B)": [0.0] * 50}
    )
    checkpoint = {
        "version": "8.3.203",
        "epoch": epoch,
        "train_results": history.to_dict("list"),
        "train_args": baseline(),
    }
    check_last_checkpoint_metadata(checkpoint, history)
    checkpoint["train_results"]["epoch"][-1] = 49
    with pytest.raises(ValueError, match="50 completed"):
        check_last_checkpoint_metadata(checkpoint, history)


def test_preflight_collects_independent_blockers_and_can_retry(tmp_path, monkeypatch):
    from flir_pipeline.closeout import oe4

    cfg = load_config(Path("configs/closeout/oe4_20261115.yaml"))
    for key in (
        "publication",
        "source_root",
        "freeze",
        "common_test",
        "initial_weights",
        "weights_receipt",
    ):
        cfg[key] = str(tmp_path / key)
    cfg["runs"] = {policy: str(tmp_path / policy) for policy in oe4.POLICIES}
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(cfg, sort_keys=False))
    monkeypatch.setattr(oe4, "code_hashes", lambda: {"synthetic.py": "hash"})
    result = oe4.preflight(config)
    assert result["state"] == "BLOCKED"
    assert len(result["blockers"]) >= 5
    assert not (tmp_path / "freeze").exists()
    assert len(list(tmp_path.glob("freeze.blocked-*/receipt.json"))) == 1
    oe4.preflight(config)
    assert len(list(tmp_path.glob("freeze.blocked-*/receipt.json"))) == 2


def test_wrong_protocol_authorization_never_reaches_gpu(tmp_path, monkeypatch):
    from flir_pipeline.closeout import oe4
    from flir_pipeline.detection import admission

    cfg = load_config(Path("configs/closeout/oe4_20261115.yaml"))
    monkeypatch.setattr(oe4, "load_config", lambda *args: cfg)
    monkeypatch.setattr(oe4, "verify_freeze", lambda *args: {"state": "READY"})
    monkeypatch.setattr(admission, "authorized", lambda *args: {"protocol": "wrong"})
    with pytest.raises(ValueError, match="protocol/freeze"):
        oe4.authorize(tmp_path / "config", "train")
