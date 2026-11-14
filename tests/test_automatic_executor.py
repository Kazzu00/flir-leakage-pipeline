"""Offline fixtures only: technical policy, immutable attempts and common test.

Checkpoint bytes here are test doubles, never deserialized or represented as GPU
evidence. The small ZIP audit reuses the existing source/publication verifier tests.
"""

import copy
import io
import platform
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from test_fixed_test import audited as audited
from typer.testing import CliRunner

from flir_pipeline.cli import app
from flir_pipeline.detection import common_admission as admission
from flir_pipeline.detection import common_evaluation as evaluation
from flir_pipeline.detection import common_executor as executor
from flir_pipeline.detection import common_training as training
from flir_pipeline.detection.common_test_storage import signatures
from flir_pipeline.detection.metrics import ImageStats
from flir_pipeline.detection.scientific_executor import seal, write_view
from flir_pipeline.similarity.storage import (
    file_sha256,
    read_json,
    stable_id,
    write_json,
)
from flir_pipeline.splitting.labeled import load_labeled_audit

CONFIG = Path("configs/detection/common_test_yolo11m_v2.yaml")
ENV = {
    "python": platform.python_version(),
    "packages": {
        "torch": "2.7.1+cu118",
        "torchvision": "0.22.1+cu118",
        "ultralytics": "8.3.203",
    },
    "cuda": "11.8",
    "cuda_available": True,
    "gpu": "synthetic fixture",
}


def matrix(rows):
    arms = [
        {
            "arm": f"{p}-{s}-{i}",
            "population": p,
            "policy": s,
            "seed": i,
            "candidate_id": f"{p}-{s}-{i}",
        }
        for p in ("historical", "expanded")
        for s in ("S1", "S2", "S3")
        for i in range(5)
    ]
    return {"runs": arms}, {a["arm"]: rows.copy() for a in arms}


def test_global_quarantine_is_population_intersection():
    rows = pd.DataFrame(
        [
            {
                "frame_id": f"{split}-{i}",
                "content_id": f"{split}-{i}",
                "image_sha256": f"{split}-{i}",
                "new_split": split,
            }
            for split, size in (
                ("test", 372),
                ("quarantine", 142),
                ("train", 2),
                ("val", 2),
            )
            for i in range(size)
        ]
    )
    summary, assignments = matrix(rows)
    for arm in summary["runs"]:
        if arm["population"] == "historical":
            assignments[arm["arm"]] = rows.loc[rows.frame_id != "quarantine-141"].copy()
    test = rows.loc[rows.new_split == "test"].drop(columns="new_split")
    assert (
        admission.check_memberships(summary, assignments, test)[
            "quarantine_occurrences"
        ]
        == 142
    )
    assignments["historical-S1-0"].loc[
        lambda d: d.frame_id == "quarantine-0", "new_split"
    ] = "train"
    with pytest.raises(ValueError, match="quarantine"):
        admission.check_memberships(summary, assignments, test)


@pytest.fixture
def bundle(audited, tmp_path, monkeypatch):
    audit, sources, _ = audited
    rows = load_labeled_audit(audit).manifest.copy()
    rows["new_split"] = rows.original_split.fillna("train")
    test_contents = set(rows.loc[rows.new_split == "test", "content_id"])
    rows.loc[
        rows.content_id.isin(test_contents) & (rows.new_split != "test"), "new_split"
    ] = "quarantine"
    test = rows.loc[rows.new_split == "test"].drop(columns="new_split")
    summary, assignments = matrix(rows)
    publication = tmp_path / "publication"
    shutil.copytree(audit, publication / "sources/audit")
    (publication / "sources/proposal").mkdir()
    test.to_parquet(publication / "sources/proposal/test_records.parquet", index=False)
    write_json(publication / "receipt.json", {"synthetic_publication_boundary": True})

    def loader(p):
        return summary, assignments, test

    # Full publication verifier has its own offline suite; this fixture replaces
    # only its boundary and cohort sizes, not archive/plan/view/annotation hashes.
    monkeypatch.setattr(executor, "load_publication", loader)
    monkeypatch.setattr(admission, "load_publication", loader)
    monkeypatch.setattr(
        admission,
        "check_memberships",
        lambda *a: {
            "test_occurrences": len(test),
            "test_id": executor.records_identity(test),
            "quarantine_occurrences": 1,
            "arms": 30,
        },
    )
    output = tmp_path.parent / f"{tmp_path.name}-automatic"
    plan, prepared, gate = output / "plan", output / "prepared", output / "gate.json"
    executor.dry_run(publication, CONFIG, plan)
    executor.prepare(publication, sources, prepared)
    admission.admit(publication, plan, prepared, CONFIG, sources, gate)
    weights = output / "yolo11m.pt"
    weights.write_bytes(b"synthetic fixture; not a model")
    weights_receipt = output / "weights.json"

    class Response(io.BytesIO):
        def geturl(self):
            return admission.OFFICIAL_URL

    monkeypatch.setattr(
        admission.urllib.request,
        "urlopen",
        lambda *a, **kw: Response(weights.read_bytes()),
    )
    admission.verify_official_weights(weights, weights_receipt)
    probe = output / "probe"
    (probe / "training/weights").mkdir(parents=True)
    for name in ("last.pt", "best.pt"):
        (probe / "training/weights" / name).write_bytes(b"synthetic probe checkpoint")
    pd.DataFrame({"epoch": [1], "metrics/mAP50-95(B)": [0.0]}).to_csv(
        probe / "training/results.csv", index=False
    )
    config = executor.load_executor_config(CONFIG)
    write_json(
        probe / "state.json",
        {
            "schema_version": executor.VERSION,
            "mode": "synthetic-smoke",
            "state": "completed",
            "identity": {"config": config, "batch": 1, "device": "0"},
            "actual_epochs": 1,
            "observed_batches": [1],
            "environment": {
                "environment": ENV,
                "state": "completed",
                "blockers": [],
                "config_id": stable_id(config),
            },
            "checkpoint_sha256": {
                n: file_sha256(probe / "training/weights" / n)
                for n in ("last.pt", "best.pt")
            },
        },
    )
    write_json(probe / "receipt.json", {"artifacts": signatures(probe)})
    return {
        "output": output,
        "gate": gate,
        "weights": weights_receipt,
        "probe": probe,
        "plan": plan,
        "prepared": prepared,
        "publication": publication,
        "sources": sources,
    }


def checkpoint_fixture(ctx, run, path, complete=True):
    """Build receipt-bound file fixtures without loading or training any model."""
    rows = pd.read_parquet(
        ctx["prepared"] / "views" / run["records_id"] / "records.parquet"
    )
    rows["label_path"] = rows.image_path.map(
        lambda p: str(Path(p).parent.parent / "labels" / f"{Path(p).stem}.txt")
    )
    dataset = write_view(rows, path / "development")
    (path / "train/weights").mkdir(parents=True)
    for name in ("last.pt", "best.pt"):
        (path / "train/weights" / name).write_bytes(b"checkpoint test double")
    args = {
        **ctx["config"]["train"],
        "data": str(dataset),
        "seed": run["detector_seed"],
        "split": "val",
    }
    (path / "train/args.yaml").write_text(yaml.safe_dump(args), encoding="utf-8")
    epochs = ctx["config"]["train"]["epochs"] if complete else 1
    pd.DataFrame(
        {"epoch": range(1, epochs + 1), "metrics/mAP50-95(B)": [0.0] * epochs}
    ).to_csv(path / "train/results.csv", index=False)
    return seal(
        path,
        {
            "state": "completed" if complete else "interrupted",
            "run": run,
            "admission_sha256": ctx["admission_sha256"],
            "environment": ENV,
            "test_used": False,
            "checkpoint_selection": "validation_only",
            "checkpoint_sha256": {
                n: file_sha256(path / "train/weights" / n)
                for n in ("best.pt", "last.pt")
            },
        },
    )


def operational_freeze(bundle):
    ctx = training.context(bundle["gate"], bundle["weights"], 1, ENV, "pilot")
    pilot = bundle["output"] / "pilot/attempt-0000"
    checkpoint_fixture(ctx, ctx["runs"][0], pilot)
    freeze = bundle["output"] / "freeze.json"
    training.freeze(bundle["gate"], bundle["weights"], bundle["probe"], pilot, freeze)
    return freeze


def test_automatic_admission_original_bytes_and_derived_count(bundle):
    gate = admission.verify_admission(bundle["gate"])
    assert gate["human_review_asserted"] is False
    assert gate["annotation_policy"] == "original_bytes_per_occurrence"
    assert gate["counts"]["experimental_identities"] == 90
    assert (
        gate["counts"]["executions"] == 3
    )  # derived, never the production constant 75
    assert not list(bundle["prepared"].rglob("test.txt"))
    label = next((bundle["prepared"] / "store/labels").iterdir())
    label.write_text("0 .5 .5 .1 .1\n")
    with pytest.raises(ValueError, match="checksum"):
        admission.verify_admission(bundle["gate"])


def test_original_archive_tamper_rejected(bundle):
    with (bundle["sources"] / "images.zip").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="ZIP checksum"):
        admission.verify_admission(bundle["gate"])


def test_official_weights_reject_unrelated_bytes(tmp_path, monkeypatch):
    weights = tmp_path / "yolo11m.pt"
    weights.write_bytes(b"fine tuned weights")

    class Response(io.BytesIO):
        def geturl(self):
            return admission.OFFICIAL_URL

    monkeypatch.setattr(
        admission.urllib.request, "urlopen", lambda *a, **kw: Response(b"official")
    )
    with pytest.raises(ValueError, match="differs from official"):
        admission.verify_official_weights(weights, tmp_path / "receipt.json")
    assert not (tmp_path / "receipt.json").exists()


def test_probe_pilot_freeze_and_source_change(bundle, monkeypatch):
    freeze = operational_freeze(bundle)
    ctx = training.frozen_context(freeze)
    assert len(ctx["runs"]) == 3
    assert ctx["config"]["train"]["epochs"] == 50
    assert ctx["config"]["train"]["imgsz"] == 640
    assert ctx["admission"]["variant"]["kind"] == "original"
    monkeypatch.setattr(training, "source_identity", lambda: {"changed": "code"})
    with pytest.raises(ValueError, match="pilot|freeze"):
        training.frozen_context(freeze)


def test_probe_is_not_pilot_and_batch_cannot_drift(bundle):
    state = read_json(bundle["probe"] / "state.json")
    state["observed_batches"] = [1] * 10
    write_json(bundle["probe"] / "state.json", state)
    artifacts = signatures(bundle["probe"])
    artifacts.pop("receipt.json")
    write_json(bundle["probe"] / "receipt.json", {"artifacts": artifacts})
    assert (
        training.probe_evidence(
            bundle["probe"], executor.load_executor_config(CONFIG), 1
        )
        == ENV
    )
    with pytest.raises(ValueError, match="probe"):
        training.probe_evidence(
            bundle["probe"], executor.load_executor_config(CONFIG), 2
        )
    with pytest.raises(FileNotFoundError):
        training.freeze(
            bundle["gate"],
            bundle["weights"],
            bundle["probe"],
            bundle["probe"],
            bundle["output"] / "freeze.json",
        )


def test_safe_attempt_idempotence_resume_and_checkpoint_tamper(bundle):
    ctx = training.context(bundle["gate"], bundle["weights"], 1, ENV, "matrix")
    run = ctx["runs"][0]
    root = bundle["output"] / run["run_id"]
    attempt = root / "attempt-0000"
    checkpoint_fixture(ctx, run, attempt)
    assert training.execute_locked(ctx, run, root, False)["state"] == "completed"
    assert training.execute_locked(ctx, run, root, True)["state"] == "completed"
    changed = {**run, "detector_seed": 43}
    with pytest.raises(ValueError, match="identity"):
        training.execute_locked(ctx, changed, root, True)
    (attempt / "train/weights/last.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        training.execute_locked(ctx, run, root, True)


def test_interrupted_requires_resume_and_preserves_unverified(bundle):
    ctx = training.context(bundle["gate"], bundle["weights"], 1, ENV, "matrix")
    run = ctx["runs"][0]
    root = bundle["output"] / run["run_id"]
    checkpoint_fixture(ctx, run, root / "attempt-0000", complete=False)
    with pytest.raises(ValueError, match="explicit --resume"):
        training.execute_locked(ctx, run, root, False)
    (root / "attempt-0000/unbound.txt").write_text("unverified write after receipt")
    with pytest.raises(ValueError, match="Unbound"):
        training.execute_locked(ctx, run, root, True)


def test_test_release_requires_all_runs_and_identical_view(bundle):
    freeze = operational_freeze(bundle)
    ctx = training.frozen_context(freeze)
    root = bundle["output"] / "matrix"
    release = bundle["output"] / "release.json"
    checkpoint_fixture(
        ctx, ctx["runs"][0], root / ctx["runs"][0]["run_id"] / "attempt-0000"
    )
    with pytest.raises(ValueError, match="Entire matrix"):
        evaluation.release(freeze, root, release)
    assert not release.exists()
    for run in ctx["runs"][1:]:
        checkpoint_fixture(ctx, run, root / run["run_id"] / "attempt-0000")
    evaluation.release(freeze, root, release)
    test = bundle["output"] / "test-view"
    evaluation.prepare_test(release, test)
    permission, verified = evaluation.evaluation_context(release)
    evaluation.verify_test_view(test, permission, verified)
    label = next((test / "view/labels").iterdir())
    label.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        evaluation.verify_test_view(test, permission, verified)


def test_geometry_anomaly_retained_without_correction(tmp_path):
    from PIL import Image

    image, label = tmp_path / "a.png", tmp_path / "a.txt"
    Image.new("RGB", (8, 8)).save(image)
    label.write_bytes(b"4 .99 .5 .1 .2\n")
    row = {
        "frame_id": "a",
        "image_path": str(image),
        "label_path": str(label),
        "image_sha256": file_sha256(image),
        "label_sha256": file_sha256(label),
        "new_split": "test",
    }
    rows = pd.DataFrame([row])
    with pytest.raises(ValueError, match="annotation"):
        write_view(rows, tmp_path / "strict")
    write_view(rows, tmp_path / "current", strict_geometry=False)
    assert (tmp_path / "current/labels/a.txt").read_bytes() == label.read_bytes()


def sample(name):
    return ImageStats(
        name,
        np.full(5, 0.9),
        np.arange(5),
        np.ones((5, 10), bool),
        np.arange(5),
        np.ones(5, int),
        np.ones(5, int),
    )


def test_paired_identical_runs_are_zero_not_independent_replicates():
    images = [sample("a"), sample("b"), sample("c")]
    result = evaluation.paired_difference(
        images, images, ["duplicate", "duplicate", "other"], 10, 42, 0.95
    )
    assert all(
        r["difference_right_minus_left"] == r["lower"] == r["upper"] == 0
        for r in result.values()
    )
    assert result == evaluation.paired_difference(
        images, images, ["duplicate", "duplicate", "other"], 10, 42, 0.95
    )
    with pytest.raises(ValueError, match="identical images"):
        evaluation.paired_difference(
            images, images[::-1], ["a", "b", "c"], 10, 42, 0.95
        )


@pytest.mark.parametrize(
    "package,version",
    [("torch", "2.7.1+cu126"), ("torchvision", "0.23.0"), ("ultralytics", "8.4.0")],
)
def test_incompatible_environment(package, version):
    env = copy.deepcopy(ENV)
    env["packages"][package] = version
    with pytest.raises(ValueError, match="Incompatible"):
        training.compatible_environment(env)


@pytest.mark.parametrize(
    "command",
    [
        "admit",
        "verify-weights",
        "pilot",
        "freeze",
        "admission-check",
        "release-test",
        "prepare-test",
        "evaluate",
        "compare",
    ],
)
def test_automatic_cli_help(command):
    result = CliRunner().invoke(
        app, ["detection", "common-executor", command, "--help"]
    )
    assert result.exit_code == 0, result.output
    assert "SIGNATURE" not in result.output


def test_export_integrity_recomputed_after_resealing(tmp_path):
    from flir_pipeline.detection.metrics import evaluate, save_image_stats

    ctx = {"admission_sha256": "fixture"}
    run = {
        "run_id": "fixture-model",
        "detector_seed": 42,
        "aliases": [{"policy": "S1"}],
    }
    train, test, output = (tmp_path / name for name in ("train", "test", "evaluation"))
    train.mkdir()
    (test / "view").mkdir(parents=True)
    output.mkdir()
    write_json(train / "execution.json", {"synthetic_boundary": True})
    write_json(test / "execution.json", {"synthetic_boundary": True})
    pd.DataFrame([{"frame_id": "a", "num_objects": 5}]).to_parquet(
        test / "view/records.parquet"
    )
    images = [sample("a")]
    overall, classes = evaluate(images)
    classes["class_name"] = classes.class_id.map(executor.NAMES)
    classes.to_parquet(output / "metrics_per_class.parquet", index=False)
    write_json(output / "metrics.json", {"overall": overall})
    save_image_stats(images, output / "test_image_stats.npz")
    predictions = [
        {
            "frame_id": "a",
            "detection": c,
            "class_id": c,
            "confidence": 0.9,
            "x1": 0.0,
            "y1": 0.0,
            "x2": 1.0,
            "y2": 1.0,
            "coordinates": "validator_input_pixels_xyxy",
            "input_height": 640,
            "input_width": 640,
        }
        for c in range(5)
    ]
    evaluation.export_details(output, run, images, predictions)
    state = {
        "state": "completed",
        "run": run,
        "admission_sha256": "fixture",
        "training_execution_sha256": file_sha256(train / "execution.json"),
        "sealed_test_sha256": file_sha256(test / "execution.json"),
        "checkpoint_selection": "validation_only",
        "test_tuning": False,
    }
    seal(output, state)
    assert (
        evaluation.verify_result(output, ctx, run, train, test)["state"] == "completed"
    )
    rows = pd.read_parquet(output / "predictions.parquet")
    rows.loc[0, "confidence"] = 0.1
    rows.to_parquet(output / "predictions.parquet", index=False)
    seal(output, state)
    with pytest.raises(ValueError, match="Prediction export"):
        evaluation.verify_result(output, ctx, run, train, test)


def test_runtime_dispatch_preserves_resume_identity(bundle, monkeypatch):
    import sys
    from types import SimpleNamespace

    ctx = training.context(bundle["gate"], bundle["weights"], 1, ENV, "matrix")
    run = ctx["runs"][0]
    root = bundle["output"] / run["run_id"]
    previous = root / "attempt-0000"
    checkpoint_fixture(ctx, run, previous, complete=False)
    before = signatures(previous)
    calls = []

    def runtime(ctx_arg, run_arg, destination, previous_arg):
        calls.append((ctx_arg, run_arg, destination, previous_arg))
        return {"state": "test_double_no_training"}

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(settings=object()))
    monkeypatch.setitem(
        sys.modules,
        "flir_pipeline.detection.scientific_runtime",
        SimpleNamespace(run_training=runtime),
    )
    assert (
        training.execute_locked(ctx, run, root, True)["state"]
        == "test_double_no_training"
    )
    assert calls == [(ctx, run, root / "attempt-0001", previous)]
    assert signatures(previous) == before


def test_automatic_train_cli_routes_without_signature(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        executor,
        "train",
        lambda *args: calls.append(args) or {"state": "checked_test_double"},
    )
    result = CliRunner().invoke(
        app,
        [
            "detection",
            "common-executor",
            "train",
            str(tmp_path),
            str(tmp_path / "out"),
            "--technical-freeze",
            str(tmp_path / "freeze.json"),
            "--resume",
        ],
    )
    assert result.exit_code == 0, result.output
    assert calls[0][3] is True
    assert calls[0][4:8] == (None, None, None, None)
    assert calls[0][8] == tmp_path / "freeze.json"


def test_python_patch_version_is_bound_to_probe(monkeypatch):
    from flir_pipeline.detection import scientific_runtime

    env = copy.deepcopy(ENV)
    env["python"] = "3.12.0"
    with pytest.raises(ValueError, match="Python 3.11"):
        training.compatible_environment(env)
    monkeypatch.setattr(scientific_runtime, "configure_runtime", lambda: None)
    monkeypatch.setattr(scientific_runtime, "environment", lambda: copy.deepcopy(ENV))
    ctx = {
        "admission": {
            "policy": admission.POLICY,
            "environment": {"python": "3.11.99", "signature": ENV},
        }
    }
    with pytest.raises(ValueError, match="Frozen Python"):
        scientific_runtime.frozen_environment(ctx)


def test_legacy_blocker_text_is_not_a_scientific_parameter():
    rows = pd.DataFrame([{"frame_id": "a", "new_split": "train"}])
    summary, assignments = matrix(rows)
    plan = executor.plan_runs(
        summary, assignments, executor.load_executor_config(CONFIG), "fixture-source"
    )
    historical = copy.deepcopy(plan)
    for run in historical["runs"]:
        run["blockers"] = ["historical human blocker wording"]
    historical["plan_id"] = stable_id(
        {k: v for k, v in historical.items() if k != "plan_id"}
    )
    assert historical["plan_id"] != plan["plan_id"]
    assert admission.plan_contract(historical) == admission.plan_contract(plan)
    assert not any(
        "human" in blocker or "adjudication" in blocker
        for run in plan["runs"]
        for blocker in run["blockers"]
    )
    historical["runs"][0]["identity"]["detector_seed"] = 999
    assert admission.plan_contract(historical) != admission.plan_contract(plan)
