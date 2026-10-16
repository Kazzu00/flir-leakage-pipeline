"""Offline scientific invariants for the isolated provisional executor."""

from pathlib import Path

import pandas as pd
import pytest
from test_fixed_test import audited as audited
from typer.testing import CliRunner

from flir_pipeline.cli import app
from flir_pipeline.detection import common_executor as executor
from flir_pipeline.detection import common_runtime as runtime
from flir_pipeline.detection.protocol import load_config
from flir_pipeline.similarity.storage import read_json, write_json
from flir_pipeline.splitting.labeled import load_labeled_audit

CONFIG = Path("configs/detection/common_test_yolo11m_v2.yaml")


@pytest.fixture
def records():
    return pd.DataFrame(
        [
            {
                "frame_id": "a",
                "content_id": "a",
                "label_sha256": "la",
                "new_split": "train",
            },
            {
                "frame_id": "b",
                "content_id": "b",
                "label_sha256": "lb",
                "new_split": "val",
            },
            {
                "frame_id": "c",
                "content_id": "c",
                "label_sha256": "lc",
                "new_split": "test",
            },
            {
                "frame_id": "d",
                "content_id": "c",
                "label_sha256": "ld",
                "new_split": "quarantine",
            },
        ]
    )


def matrix(records):
    entries, assignments = [], {}
    for pop in ("historical", "expanded"):
        for policy in ("S1", "S2", "S3"):
            for seed in range(5):
                name = f"{pop}-{policy}-{seed}"
                entries.append(
                    {
                        "arm": name,
                        "population": pop,
                        "policy": policy,
                        "seed": seed,
                        "candidate_id": name,
                    }
                )
                rows = records.copy()
                # Exactly the five historical S2/S3 pairs are equivalent.
                effective = "S2" if pop == "historical" and policy == "S3" else policy
                rows["provenance"] = f"{pop}-{effective}-{seed}"
                assignments[name] = rows
    return {"runs": entries}, assignments


def test_config_independent_and_legacy_strict():
    config = executor.load_executor_config(CONFIG)
    old = load_config(Path("configs/detection/yolo11n.yaml"))
    assert config["train"] == old["train"]
    assert config["evaluation"] == old["evaluation"]
    assert old["model"] == "yolo11n.pt"
    with pytest.raises(ValueError, match="Unsupported"):
        load_config(CONFIG)


def test_test_conservation_and_quarantine(records):
    expected = records.loc[records.new_split == "test"].drop(columns="new_split")
    active = executor.active_records(records, expected)
    assert active.frame_id.tolist() == ["a", "b", "c"]
    assert executor.records_identity(records) == executor.records_identity(
        records.iloc[::-1]
    )
    changed = records.copy()
    changed.loc[changed.frame_id == "c", "label_sha256"] = "tampered"
    with pytest.raises(AssertionError):
        executor.active_records(changed, expected)
    changed = records.copy()
    changed.loc[changed.frame_id == "a", "content_id"] = "c"
    with pytest.raises(ValueError, match="crosses"):
        executor.active_records(changed, expected)


def test_exact_dedup_and_identity(records):
    summary, assignments = matrix(records)
    config = executor.load_executor_config(CONFIG)
    plan = executor.plan_runs(summary, assignments, config, "source")
    assert plan["arm_count"] == 30
    assert plan["experimental_identities"] == 90
    assert plan["execution_count"] == 75
    assert sum(len(r["aliases"]) == 2 for r in plan["runs"]) == 15
    assert plan == executor.plan_runs(summary, assignments, config, "source")
    changed = executor.plan_runs(summary, assignments, config, "source", "a" * 64)
    assert set(r["run_id"] for r in plan["runs"]).isdisjoint(
        r["run_id"] for r in changed["runs"]
    )
    assignments["historical-S3-0"].loc[0, "label_sha256"] = "changed"
    assert (
        executor.plan_runs(summary, assignments, config, "source")["execution_count"]
        == 78
    )


def test_dry_run_no_runtime_and_tamper(records, tmp_path, monkeypatch):
    summary, assignments = matrix(records)
    publication = tmp_path / "publication"
    publication.mkdir()
    write_json(publication / "receipt.json", {"synthetic": True})
    monkeypatch.setattr(
        executor, "load_publication", lambda p: (summary, assignments, None)
    )
    output = tmp_path / "plan"
    plan = executor.dry_run(publication, CONFIG, output)
    assert executor.verify_executor_plan(output) == plan
    assert not (output / "store").exists()
    with pytest.raises(ValueError, match="exists"):
        executor.dry_run(publication, CONFIG, output)
    (output / "plan.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        executor.verify_executor_plan(output)


def test_publication_gate_cannot_be_skipped(tmp_path, monkeypatch):
    write_json(tmp_path / "receipt.json", {"synthetic": True})
    called = []

    def reject(path):
        called.append(path)
        raise ValueError("source tampered")

    monkeypatch.setattr(executor, "verify_fixed_test", reject)
    with pytest.raises(ValueError, match="tampered"):
        executor.load_publication(tmp_path)
    assert called == [tmp_path]


def test_exact_test_count_required(tmp_path, monkeypatch):
    write_json(tmp_path / "receipt.json", {"synthetic": True})
    monkeypatch.setattr(executor, "verify_fixed_test", lambda p: None)
    write_json(
        tmp_path / "summary.json", {"configurations": 30, "failed_configurations": 0}
    )
    path = tmp_path / "sources/proposal"
    path.mkdir(parents=True)
    pd.DataFrame({"frame_id": ["test"]}).to_parquet(path / "test_records.parquet")
    with pytest.raises(ValueError, match="372"):
        executor.load_publication(tmp_path)


def test_prepare_original_bytes_and_no_test_exposure(audited, tmp_path, monkeypatch):
    import shutil

    audit, root, _ = audited
    source = load_labeled_audit(audit)
    rows = source.manifest.copy()
    rows["new_split"] = rows.original_split.fillna("train")
    protected = set(rows.loc[rows.new_split == "test", "content_id"])
    rows.loc[
        rows.content_id.isin(protected) & (rows.new_split != "test"), "new_split"
    ] = "quarantine"
    test = rows.loc[rows.new_split == "test"].drop(columns="new_split")
    publication = tmp_path / "publication"
    shutil.copytree(audit, publication / "sources/audit")
    write_json(publication / "receipt.json", {"fixture": True})
    summary = {"runs": [{"arm": "one"}, {"arm": "alias"}]}
    monkeypatch.setattr(
        executor,
        "load_publication",
        lambda p: (summary, {"one": rows, "alias": rows}, test),
    )
    output = tmp_path.parent / (tmp_path.name + "-prepared")
    result = executor.prepare(publication, root, output)
    assert len(result["views"]) == 1
    assert not result["test_materialized"]
    assert not list(output.rglob("test.txt"))
    assert executor.verify_preparation(output) == result
    image_names = {p.stem for p in (output / "store/images").iterdir()}
    assert image_names == set(
        rows.loc[rows.new_split.isin(["train", "val"]), "frame_id"]
    )
    next((output / "store/labels").iterdir()).write_bytes(b"tamper")
    with pytest.raises(ValueError, match="checksum"):
        executor.verify_preparation(output)


@pytest.mark.parametrize("resume", [False, True])
def test_real_training_and_resume_fail_before_io(tmp_path, resume):
    output = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="admission_contract"):
        executor.train(tmp_path / "approved.json", output, 0, resume)
    assert not output.exists()


@pytest.mark.parametrize(
    "command",
    [
        "dry-run",
        "prepare",
        "verify",
        "status",
        "preflight",
        "synthetic-smoke",
        "probe",
        "train",
    ],
)
def test_cli_help(command):
    result = CliRunner().invoke(
        app, ["detection", "common-executor", command, "--help"]
    )
    assert result.exit_code == 0, result.output


def test_cli_resume_cannot_override_admission(tmp_path):
    result = CliRunner().invoke(
        app,
        [
            "detection",
            "common-executor",
            "train",
            str(tmp_path),
            str(tmp_path / "out"),
            "--resume",
        ],
    )
    assert result.exit_code == 1
    assert "BLOCKED" in result.output


def test_synthetic_failure_receipted_without_gpu(tmp_path, monkeypatch):
    def fail(*args):
        raise ValueError("cuda_required_no_cpu_fallback")

    monkeypatch.setattr(runtime, "preflight", fail)
    output = tmp_path / "operation"
    with pytest.raises(ValueError, match="cuda_required"):
        runtime.synthetic_smoke(CONFIG, output)
    result = runtime.verify_operation(output)
    assert result["state"] == "failed"
    assert not result["scientific_training_executed"]
    with pytest.raises(ValueError, match="Fresh"):
        runtime.synthetic_smoke(CONFIG, output)
    state = read_json(output / "state.json")
    state["state"] = "completed"
    write_json(output / "state.json", state)
    with pytest.raises(ValueError, match="checksum"):
        runtime.verify_operation(output)


def test_synthetic_runtime_contract_with_fake_backend(tmp_path, monkeypatch):
    """Exercise orchestration and output verification without optional packages."""
    import sys
    from types import SimpleNamespace

    observed = {}

    class FakeYOLO:
        def __init__(self, model):
            assert model == "yolo11m.yaml"

        def add_callback(self, event, callback):
            assert event == "on_train_batch_end"
            self.callback = callback

        def train(self, **kwargs):
            observed.update(kwargs)
            self.callback(SimpleNamespace(batch_size=kwargs["batch"]))
            path = Path(kwargs["project"]) / "training"
            (path / "weights").mkdir(parents=True)
            for name in ("best.pt", "last.pt"):
                (path / "weights" / name).write_bytes(b"synthetic fake checkpoint")
            (path / "results.csv").write_text(
                "epoch,metrics/mAP50-95(B)\n1,0.01\n", encoding="utf-8"
            )

    monkeypatch.setattr(runtime, "preflight", lambda *a: {"synthetic_fake": True})
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeYOLO))
    output = tmp_path / "fake-operation"
    state = runtime.synthetic_smoke(CONFIG, output, cpu=True)
    assert state["state"] == "completed"
    assert observed["device"] == "cpu"
    assert observed["imgsz"] == 640 and observed["epochs"] == 1
    assert observed["pretrained"] is False
    assert (
        runtime.verify_operation(output)["checkpoint_sha256"]
        == state["checkpoint_sha256"]
    )
    assert not list(output.rglob("test*"))
    (output / "training/weights/last.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        runtime.verify_operation(output)


def test_preflight_rejects_missing_cuda_without_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "configure_runtime", lambda: None)
    monkeypatch.setattr(
        runtime,
        "environment",
        lambda: {
            "packages": {
                "torch": "2.7.1+cu118",
                "torchvision": "0.22.1+cu118",
                "ultralytics": "8.3.203",
            },
            "cuda_available": False,
        },
    )
    monkeypatch.setattr(runtime, "provenance", lambda: {"synthetic_fake": True})
    with pytest.raises(ValueError, match="no_cpu_fallback"):
        runtime.preflight(CONFIG, tmp_path / "preflight.json")
    assert read_json(tmp_path / "preflight.json")["state"] == "failed"
