"""Admission and train/val isolation regression checks; no FAC/GPU required."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from flir_pipeline.cli import app
from flir_pipeline.detection import common_admission as admission
from flir_pipeline.detection import common_training as training
from flir_pipeline.similarity.storage import file_sha256


def test_missing_external_trust_anchor_denies_signature(tmp_path, monkeypatch):
    authorization = tmp_path / "approval.json"
    signature = tmp_path / "approval.sig"
    signers = tmp_path / "signers"
    authorization.write_text("{}", encoding="utf-8")
    signature.write_text("invalid", encoding="utf-8")
    signers.write_text("reviewer ssh-ed25519 invalid", encoding="utf-8")
    monkeypatch.delenv("FLIR_TRUSTED_SIGNERS_SHA256", raising=False)
    with pytest.raises(ValueError, match="FLIR_TRUSTED_SIGNERS_SHA256"):
        admission.verify_signature(authorization, signature, signers, "reviewer")


def test_gate_rejects_forged_review_without_signature(tmp_path, monkeypatch):
    for name in ("approval.json", "approval.sig", "signers"):
        (tmp_path / name).write_text("invalid", encoding="utf-8")
    monkeypatch.setenv("FLIR_TRUSTED_SIGNERS_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="trust anchor"):
        admission.verify_signature(
            tmp_path / "approval.json", tmp_path / "approval.sig",
            tmp_path / "signers", "reviewer",
        )


def _fixture_approval(tmp_path: Path, monkeypatch):
    plan = tmp_path / "plan"
    preparation = tmp_path / "prepared"
    plan.mkdir()
    preparation.mkdir()
    (plan / "receipt.json").write_text("plan", encoding="utf-8")
    (preparation / "preparation.json").write_text("prep", encoding="utf-8")
    checkpoint = tmp_path / "official.pt"
    checkpoint.write_bytes(b"dummy checkpoint, never loaded")
    sha = file_sha256(checkpoint)
    frame_set = "sample-id"
    run = {
        "run_id": "run-id",
        "identity": {
            "records_id": frame_set,
            "initialization": {"sha256": sha},
            "detector_seed": 42,
            "configuration": {
                "train": {"epochs": 50, "imgsz": 640, "optimizer": "SGD"},
                "torch_version": "2.7.1",
                "ultralytics_version": "8.3.203",
            },
        },
    }
    monkeypatch.setattr(
        admission,
        "verify_executor_plan",
        lambda _: {"plan_id": "plan-id", "source_id": "src-id", "runs": [run]},
    )
    monkeypatch.setattr(
        admission,
        "verify_preparation",
        lambda _: {
            "source_id": "src-id", "test_materialized": False,
            "schema_version": admission.VERSION,
            "views": {frame_set: {"test_id": "test-id"}},
        },
    )
    monkeypatch.setattr(admission, "verify_signature", lambda *args: None)
    approval = {
        "schema_version": admission.ADMISSION_VERSION,
        "decision": "approved_for_real_training",
        "reviewer_identity": "reviewer",
        "review_date": "2026-11-09",
        "rationale": "Explicit documented, independently reviewed methodological decision.",
        "decisions": {field: True for field in admission.REQUIRED_DECISIONS},
        "plan_id": "plan-id",
        "plan_receipt_sha256": file_sha256(plan / "receipt.json"),
        "preparation_receipt_sha256": file_sha256(preparation / "preparation.json"),
        "frozen_test_id": "test-id",
        "checkpoint_sha256": sha,
        "initialization_source": "official_ultralytics_yolo11m",
        "class_names": {str(k): v for k, v in admission.NAMES.items()},
        "freeze": {
            "batch": 1, "device": "0", "epochs": 50, "imgsz": 640,
            "optimizer": "SGD", "torch": "2.7.1",
            "torchvision": "0.22.1", "ultralytics": "8.3.203",
            "amp": False, "tf32": False, "validated_non_test_pilot": True,
            "ultralytics_defaults_sha256": "a" * 64,
            "pilot_receipt_sha256": "b" * 64,
        },
    }
    signed = tmp_path / "approval.json"
    signed.write_text(json.dumps(approval), encoding="utf-8")
    signature = tmp_path / "approval.sig"
    signature.write_text("fixture", encoding="utf-8")
    signers = tmp_path / "signers"
    signers.write_text("fixture", encoding="utf-8")
    return plan, preparation, checkpoint, signed, signature, signers, approval


def test_signed_plan_and_data_cross_binding(tmp_path, monkeypatch):
    a = _fixture_approval(tmp_path, monkeypatch)
    checked = admission.verify_admission(*a[:6], "reviewer")
    assert checked["frozen_test_id"] == "test-id"
    assert checked["checkpoint_sha256"] == file_sha256(a[2])
    changed = a[-1]
    changed["frozen_test_id"] = "other"
    a[3].write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="Common test identity"):
        admission.verify_admission(*a[:6], "reviewer")


def test_unapproved_resume_does_not_create_artifacts(tmp_path, monkeypatch):
    a = _fixture_approval(tmp_path, monkeypatch)
    approval = admission.verify_admission(*a[:6], "reviewer")
    monkeypatch.setattr(training, "verify_admission", lambda *args: approval)
    output = tmp_path / "real-output"
    with pytest.raises(ValueError, match="Resume not yet admitted"):
        training.train_one(*a[:2], output, 0, *a[2:6], "reviewer", resume=True)
    assert not output.exists()


@pytest.mark.parametrize("command", ["train-approved", "verify-approved-run"])
def test_gated_cli_help(command):
    result = CliRunner().invoke(
        app, ["detection", "common-executor", command, "--help"]
    )
    assert result.exit_code == 0, result.output
