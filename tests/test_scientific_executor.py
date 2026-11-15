"""Offline admission, lineage and runtime boundaries; never trains a model."""

import json
import shutil
import subprocess
import sys
import types
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
import yaml
from PIL import Image

from flir_pipeline.detection import admission as auth
from flir_pipeline.detection import common_executor as provisional
from flir_pipeline.detection import scientific_executor as executor
from flir_pipeline.detection import scientific_runtime as runtime
from flir_pipeline.similarity.storage import file_sha256, read_json, write_json


def binding(path):
    return {"path": str(path.resolve()), "sha256": file_sha256(path)}


def window(purpose="train"):
    now = datetime.now(UTC)
    return {
        "schema_version": auth.NAMESPACE,
        "purpose": purpose,
        "reviewer": "synthetic reviewer",
        "not_before": (now - timedelta(hours=1)).isoformat(),
        "expires_at": (now + timedelta(hours=1)).isoformat(),
    }


def test_real_openssh_signature_and_tamper(tmp_path, monkeypatch):
    executable = shutil.which("ssh-keygen")
    assert executable, (
        "OpenSSH is required for the offline authorization integration test"
    )
    key = tmp_path / "key"
    subprocess.run(
        [executable, "-q", "-t", "ed25519", "-N", "", "-f", str(key)],
        check=True,
        capture_output=True,
    )
    trust = tmp_path / "allowed_signers"
    trust.write_text(
        "flir-scientific-approver " + key.with_suffix(".pub").read_text(),
        encoding="utf-8",
    )
    document = tmp_path / "admission.json"
    write_json(document, window())
    subprocess.run(
        [executable, "-Y", "sign", "-f", str(key), "-n", auth.NAMESPACE, str(document)],
        check=True,
        capture_output=True,
    )
    monkeypatch.setattr(auth, "system_trust", lambda: trust)
    monkeypatch.setattr(auth, "SSH_KEYGEN", executable)
    signature = Path(str(document) + ".sig")
    assert (
        auth.authorized(document, signature, "train")["reviewer"]
        == "synthetic reviewer"
    )
    with pytest.raises(ValueError, match="scope"):
        auth.authorized(document, signature, "evaluate")
    document.write_text('{"approved":true}', encoding="utf-8")
    with pytest.raises(ValueError, match="signature"):
        auth.authorized(document, signature, "train")


def test_unsigned_rejected_before_io(tmp_path, monkeypatch):
    monkeypatch.setattr(
        executor, "context", lambda *a: pytest.fail("must not inspect inputs")
    )
    with pytest.raises(ValueError, match="BLOCKED"):
        provisional.train(tmp_path, tmp_path / "out", 0)
    assert not (tmp_path / "out").exists()


def test_missing_authority_is_not_self_approval(tmp_path, monkeypatch):
    document = tmp_path / "approved.json"
    write_json(document, {**window(), "approved": True})
    monkeypatch.setattr(
        auth,
        "system_trust",
        lambda: (_ for _ in ()).throw(ValueError("BLOCKED authority")),
    )
    with pytest.raises(ValueError, match="authority"):
        auth.authorized(document, tmp_path / "signature", "train")


def test_expired_and_wrong_scope(monkeypatch, tmp_path):
    data = window()
    data["expires_at"] = (datetime.now(UTC) - timedelta(days=1)).isoformat()
    monkeypatch.setattr(auth, "verify_signature", lambda *a: json.dumps(data).encode())
    with pytest.raises(ValueError, match="expired"):
        auth.authorized(tmp_path, tmp_path, "train")


def test_readable_test_root_rejected(tmp_path):
    with pytest.raises(ValueError, match="protected"):
        auth.check_test_isolation({"test": {"protected_roots": [str(tmp_path)]}})


@pytest.fixture
def approved_payload(tmp_path, monkeypatch):
    review = tmp_path / "review.json"
    write_json(review, {"synthetic": True})
    weights = tmp_path / "yolo11m.pt"
    weights.write_bytes(b"synthetic; never loaded")
    payload = {
        **window(),
        "decisions": {name: binding(review) for name in auth.DECISIONS},
        "test": {"status": "APPROVED", "records_id": "synthetic-test"},
        "population": "expanded",
        "split_seeds": [0],
        "detector_seeds": [42, 43, 44],
        "policies": ["S1", "S2", "S3"],
        "max_runs": 9,
        "batch": 1,
        "variant": {"kind": "original", "id": "A", "version": "synthetic"},
        "weights": {
            **binding(weights),
            "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt",
        },
    }
    monkeypatch.setattr(auth, "authorized", lambda *a: payload)
    monkeypatch.setattr(auth, "verify_checkout", lambda *a: None)
    return payload


@pytest.mark.parametrize(
    "key,value",
    [
        ("batch", 0),
        ("max_runs", 10),
        ("population", "historical"),
        ("detector_seeds", [42]),
        ("policies", ["S1"]),
    ],
)
def test_approval_cannot_weaken_matrix(tmp_path, approved_payload, key, value):
    approved_payload[key] = value
    with pytest.raises(ValueError):
        auth.training_admission(tmp_path, tmp_path)


def test_approval_requires_decisions_official_weights_and_test(
    tmp_path, approved_payload
):
    assert auth.training_admission(tmp_path, tmp_path)["max_runs"] == 9
    approved_payload["test"]["status"] = "PROPOSED"
    with pytest.raises(ValueError, match="Provisional"):
        auth.training_admission(tmp_path, tmp_path)
    approved_payload["test"]["status"] = "APPROVED"
    approved_payload["weights"]["url"] = "local-JP-FLIR"
    with pytest.raises(ValueError, match="Official"):
        auth.training_admission(tmp_path, tmp_path)
    approved_payload["decisions"].pop("annotations")
    with pytest.raises(ValueError, match="decisions"):
        auth.training_admission(tmp_path, tmp_path)


def test_unannotated_video_is_rejected(tmp_path, approved_payload):
    approved_payload["variant"]["kind"] = "unannotated_video"
    with pytest.raises(ValueError, match="supervised"):
        auth.training_admission(tmp_path, tmp_path)


def test_primary_matrix_uses_existing_memberships(
    tmp_path, rows, approved_payload, monkeypatch
):
    from test_common_executor import matrix

    from flir_pipeline.detection.common_test_storage import signatures
    from flir_pipeline.similarity.storage import stable_id

    summary, assignments = matrix(rows)
    config_path = Path("configs/detection/common_test_yolo11m_v2.yaml").resolve()
    config = provisional.load_executor_config(config_path)
    plan = provisional.plan_runs(summary, assignments, config, "synthetic-source")
    plan_dir = tmp_path / "inputs/plan"
    plan_dir.mkdir(parents=True)
    write_json(plan_dir / "plan.json", plan)
    write_json(plan_dir / "receipt.json", {"artifacts": signatures(plan_dir)})
    prepared = tmp_path / "inputs/prepared"
    prepared.mkdir()
    receipt = {"source_id": "synthetic-source", "views": {}}
    for arm, table in assignments.items():
        identity = provisional.records_identity(table)
        receipt["views"].setdefault(
            identity, {"aliases": [], "test_id": "synthetic-test"}
        )["aliases"].append(arm)
    write_json(prepared / "preparation.json", receipt)
    approved_payload.update(
        plan=binding(plan_dir / "plan.json"),
        preparation=binding(prepared / "preparation.json"),
        config=binding(config_path),
        output_root=str(tmp_path / "outputs"),
    )
    review_dir = tmp_path / "approvals"
    review_dir.mkdir()
    document = review_dir / "training.json"
    write_json(document, approved_payload)
    monkeypatch.setattr(executor, "verify_preparation", lambda *a: receipt)
    ctx = executor.context(document, tmp_path, plan_dir, tmp_path / "outputs")
    assert len(ctx["runs"]) == 9
    assert {r["policy"] for r in ctx["runs"]} == {"S1", "S2", "S3"}
    assert {r["split_seed"] for r in ctx["runs"]} == {0}
    assert ctx["config"]["train"]["batch"] == 1
    assert config["train"]["batch"] is None
    plan["runs"] = [r for r in plan["runs"] if r["identity"]["detector_seed"] != 44]
    plan["plan_id"] = stable_id({k: v for k, v in plan.items() if k != "plan_id"})
    write_json(plan_dir / "plan.json", plan)
    hashes = signatures(plan_dir)
    hashes.pop("receipt.json")
    write_json(plan_dir / "receipt.json", {"artifacts": hashes})
    approved_payload["plan"] = binding(plan_dir / "plan.json")
    with pytest.raises(ValueError, match="Incomplete"):
        executor.context(document, tmp_path, plan_dir, tmp_path / "outputs")


@pytest.fixture
def rows(tmp_path):
    records = []
    for i, partition in enumerate(("train", "val", "test")):
        image = tmp_path / "original/images" / f"f{i}.png"
        label = tmp_path / "original/labels" / f"f{i}.txt"
        image.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (20, 20), (i * 50, 0, 0)).save(image)
        label.write_text(f"{i} 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        records.append(
            {
                "frame_id": f"f{i}",
                "content_id": f"c{i}",
                "image_sha256": file_sha256(image),
                "label_sha256": file_sha256(label),
                "new_split": partition,
                "image_path": str(image),
                "label_path": str(label),
            }
        )
    return pd.DataFrame(records)


def test_views_never_mix_test_and_development(tmp_path, rows):
    with pytest.raises(ValueError, match="separate"):
        executor.write_view(rows, tmp_path / "bad")
    view = executor.write_view(rows.iloc[:2], tmp_path / "view")
    assert "test" not in yaml.safe_load(view.read_text())
    assert set(p.stem for p in (view.parent / "images").iterdir()) == {"f0", "f1"}


def test_variant_missing_geometry_and_duplicate_mapping(tmp_path, rows):
    target = tmp_path / "transformed.png"
    Image.new("RGB", (10, 20)).save(target)
    mapping = tmp_path / "mapping.json"
    record = {
        "frame_id": "f0",
        "original_sha256": rows.iloc[0].image_sha256,
        **binding(target),
    }
    write_json(
        mapping,
        {"variant_id": "no-hud", "geometry": "identity_1to1", "records": [record]},
    )
    variant = {
        "kind": "geometry_preserving",
        "id": "no-hud",
        "mapping": binding(mapping),
    }
    with pytest.raises(ValueError, match="Geometry"):
        executor.variant_rows(rows.iloc[:1], variant)
    Image.new("RGB", (20, 20)).save(target)
    record.update(binding(target))
    write_json(
        mapping,
        {"variant_id": "no-hud", "geometry": "identity_1to1", "records": [record]},
    )
    variant["mapping"] = binding(mapping)
    result = executor.variant_rows(rows.iloc[:1], variant)
    assert result.iloc[0].label_sha256 == rows.iloc[0].label_sha256
    assert result.iloc[0].content_id == rows.iloc[0].content_id
    with pytest.raises(ValueError, match="correspondence"):
        executor.variant_rows(rows, variant)
    write_json(
        mapping,
        {
            "variant_id": "no-hud",
            "geometry": "identity_1to1",
            "records": [record, record],
        },
    )
    variant["mapping"] = binding(mapping)
    with pytest.raises(ValueError, match="one-to-one"):
        executor.variant_rows(rows, variant)


def test_receipts_reject_modified_checkpoints(tmp_path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"synthetic checkpoint; not pickle")
    executor.seal(tmp_path, {"state": "completed"})
    assert executor.verify_execution(tmp_path)["state"] == "completed"
    checkpoint.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        executor.verify_execution(tmp_path)


@pytest.fixture
def mocked_runtime(tmp_path, rows, monkeypatch):
    config = provisional.load_executor_config(
        Path("configs/detection/common_test_yolo11m_v2.yaml")
    )
    config["train"]["batch"] = 1
    prepared = tmp_path / "prepared/views/members"
    prepared.mkdir(parents=True)
    rows.iloc[:2].drop(columns="label_path").to_parquet(
        prepared / "records.parquet", index=False
    )
    weights = tmp_path / "yolo11m.pt"
    weights.write_bytes(b"synthetic official placeholder, never deserialized")
    ctx = {
        "prepared": prepared.parent.parent,
        "config": config,
        "admission_sha256": "synthetic",
        "admission": {
            "variant": {"kind": "original", "id": "A", "version": "synthetic"},
            "weights": binding(weights),
        },
    }
    run = {"run_id": "synthetic-run", "records_id": "members", "detector_seed": 42}
    monkeypatch.setattr(runtime, "frozen_environment", lambda ctx: {"synthetic": True})
    monkeypatch.setattr(runtime, "provenance", lambda: {"synthetic": True})
    calls = []

    class YOLO:
        def __init__(self, path):
            calls.append(path)
            self.callbacks = {}
            self.task = "detect"
            self.model = types.SimpleNamespace(yaml={"scale": "m"})

        def add_callback(self, event, fn):
            self.callbacks[event] = fn

        def train(self, **kwargs):
            assert "test" not in yaml.safe_load(Path(kwargs["data"]).read_text())
            trainer = kwargs["trainer"]()
            trainer.args = types.SimpleNamespace(
                **{k: v for k, v in kwargs.items() if k != "trainer"}
            )
            trainer.check_resume(kwargs)
            trainer.batch_size = kwargs["batch"]
            trainer.data = yaml.safe_load(Path(kwargs["data"]).read_text())
            records = pd.read_parquet(Path(kwargs["data"]).parent / "records.parquet")
            for split, name in (("train", "train_loader"), ("val", "test_loader")):
                labels = [
                    {"im_file": row.image_path, "cls": [0] * row.num_objects}
                    for row in records.loc[records.new_split == split].itertuples()
                ]
                setattr(
                    trainer,
                    name,
                    types.SimpleNamespace(dataset=types.SimpleNamespace(labels=labels)),
                )
            self.callbacks["on_train_start"](trainer)
            self.callbacks["on_train_batch_end"](trainer)
            path = Path(kwargs["project"]) / kwargs["name"]
            (path / "weights").mkdir(parents=True, exist_ok=True)
            for name in ("last.pt", "best.pt"):
                (path / "weights" / name).write_bytes(b"synthetic checkpoint")
            pd.DataFrame(
                {"epoch": range(1, 51), "metrics/mAP50-95(B)": [0.1] * 50}
            ).to_csv(path / "results.csv", index=False)
            arguments = vars(trainer.args)
            (path / "args.yaml").write_text(yaml.safe_dump(arguments), encoding="utf-8")
            trainer.epoch = 49
            self.callbacks["on_model_save"](trainer)

    class BaseTrainer:
        def check_resume(self, overrides):
            if overrides.get("resume"):
                prior = Path(overrides["resume"]).parent.parent / "args.yaml"
                self.args = types.SimpleNamespace(**yaml.safe_load(prior.read_text()))

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=YOLO))
    monkeypatch.setitem(
        sys.modules,
        "ultralytics.models.yolo.detect",
        types.SimpleNamespace(DetectionTrainer=BaseTrainer),
    )
    return ctx, run, calls, YOLO


def test_training_receipt_no_test_and_no_source_mutation(tmp_path, mocked_runtime):
    ctx, run, calls, _ = mocked_runtime
    source = ctx["prepared"] / "views/members/records.parquet"
    before = file_sha256(source)
    result = runtime.run_training(ctx, run, tmp_path / "attempt", None)
    assert result["state"] == "completed" and result["test_used"] is False
    assert len(calls) == 1 and file_sha256(source) == before
    assert executor.verify_execution(tmp_path / "attempt")["run"] == run


def test_interruption_preserved_and_no_budget_extension(
    tmp_path, mocked_runtime, monkeypatch
):
    ctx, run, _, model = mocked_runtime
    monkeypatch.setattr(
        model, "train", lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt())
    )
    with pytest.raises(KeyboardInterrupt):
        runtime.run_training(ctx, run, tmp_path / "attempt", None)
    assert read_json(tmp_path / "attempt/execution.json")["state"] == "interrupted"
    with pytest.raises(ValueError, match="exists"):
        runtime.run_training(ctx, run, tmp_path / "attempt", None)


def test_runtime_rejects_changed_config(tmp_path):
    config = provisional.load_executor_config(
        Path("configs/detection/common_test_yolo11m_v2.yaml")
    )
    args = {**config["train"], "seed": 42, "split": "test", "data": str(tmp_path)}
    with pytest.raises(ValueError, match="configuration"):
        runtime.check_arguments(args, config, 42, tmp_path)


def test_network_blocked():
    import socket

    with runtime.offline_runtime(), pytest.raises(RuntimeError, match="Network"):
        socket.create_connection(("example.invalid", 443))


def test_resume_new_attempt_preserves_old_bytes(tmp_path, mocked_runtime):
    ctx, run, calls, _ = mocked_runtime
    previous = tmp_path / "attempt-0000"
    runtime.run_training(ctx, run, previous, None)
    history = previous / "train/results.csv"
    pd.read_csv(history).iloc[:10].to_csv(history, index=False)
    old_hash = file_sha256(previous / "train/weights/last.pt")
    result = runtime.run_training(ctx, run, tmp_path / "attempt-0001", previous)
    assert result["state"] == "completed"
    assert calls[-1] == str((previous / "train/weights/last.pt").resolve())
    assert file_sha256(previous / "train/weights/last.pt") == old_hash
    assert len(pd.read_csv(history)) == 10
    assert result["effective_arguments"]["save_dir"] == str(
        (tmp_path / "attempt-0001/train").resolve()
    )


def test_resume_cannot_add_epochs_to_completed_run(tmp_path, mocked_runtime):
    ctx, run, _, _ = mocked_runtime
    previous = tmp_path / "complete"
    runtime.run_training(ctx, run, previous, None)
    with pytest.raises(ValueError, match="budget"):
        runtime.run_training(ctx, run, tmp_path / "extra", previous)


def test_resume_epoch_and_optimizer_must_match_history(tmp_path):
    history = tmp_path / "results.csv"
    pd.DataFrame({"epoch": [1, 2]}).to_csv(history, index=False)
    runtime.check_resume_state({"epoch": 1, "optimizer": {}}, history, 50)
    for checkpoint in (
        {"epoch": 50, "optimizer": {}},
        {"epoch": 1, "optimizer": None},
        {"epoch": 0, "optimizer": {}},
    ):
        with pytest.raises(ValueError, match="epoch/optimizer"):
            runtime.check_resume_state(checkpoint, history, 50)


def test_resume_wrong_run_rejected_before_runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(
        executor,
        "authorized",
        lambda *a: {"admission_sha256": "different", "run_id": "x"},
    )
    with pytest.raises(ValueError, match="another run"):
        executor._execute_locked(
            {"admission_sha256": "expected"},
            {"run_id": "x"},
            tmp_path,
            tmp_path,
            tmp_path,
        )


def test_evaluation_metrics_are_separate_and_class_names_preserved(
    tmp_path, mocked_runtime, rows, monkeypatch
):
    import numpy as np

    from flir_pipeline.detection.metrics import ImageStats

    ctx, run, _, _ = mocked_runtime
    train = tmp_path / "training"
    runtime.run_training(ctx, run, train, None)
    test = tmp_path / "test"
    executor.write_view(rows.iloc[2:], test / "view")
    executor.seal(test, {"state": "sealed_test"})
    sample = ImageStats(
        "f2",
        np.array([]),
        np.array([], dtype=int),
        np.zeros((0, 10), dtype=bool),
        np.array([2]),
        np.zeros(5, dtype=int),
        np.zeros(5, dtype=int),
    )

    class Validator:
        names = provisional.NAMES
        image_statistics = [sample]

        def __init__(self, args, save_dir):
            assert args["split"] == "test"
            assert args["model"].endswith("best.pt")

        def __call__(self):
            pass

    monkeypatch.setattr(runtime, "capture_validator", lambda *a: Validator)
    output = tmp_path / "evaluation"
    result = runtime.run_evaluation(ctx, run, train, test, output)
    assert result["state"] == "completed" and result["test_tuning"] is False
    assert read_json(output / "metrics.json")["overall"]["map50"] is None
    assert (
        pd.read_parquet(output / "metrics_per_class.parquet").iloc[4].class_name
        == "SDZI"
    )


def test_evaluation_requires_whole_matrix(tmp_path, monkeypatch):
    document = tmp_path / "training.json"
    document.write_text("synthetic", encoding="utf-8")
    admission = {
        "plan": {"path": str(tmp_path / "plan.json")},
        "output_root": str(tmp_path),
    }
    monkeypatch.setattr(
        executor,
        "authorized",
        lambda *a: {"admission_sha256": file_sha256(document), "executions": {}},
    )
    monkeypatch.setattr(executor, "training_admission", lambda *a: admission)
    monkeypatch.setattr(
        executor, "context", lambda *a: {"runs": [{"run_id": "required"}]}
    )
    with pytest.raises(ValueError, match="Entire"):
        executor.evaluation_context(tmp_path, tmp_path, document, tmp_path)


def test_test_view_cannot_self_authorize_changed_bytes(tmp_path, rows):
    expected = rows.iloc[2:].drop(columns=["new_split", "image_path", "label_path"])
    source = tmp_path / "test-records.parquet"
    expected.to_parquet(source, index=False)
    root = tmp_path / "sealed"
    executor.write_view(rows.iloc[2:], root / "view")
    permission = {"test_sources": {"records": binding(source)}}
    ctx = {
        "admission": {
            "test": {"records_id": provisional.records_identity(expected)},
            "variant": {"kind": "original"},
        }
    }
    executor.verify_test_view(root, permission, ctx)
    (root / "view/labels/f2.txt").write_text("", encoding="utf-8")
    executor.seal(root, {"state": "sealed_test"})
    with pytest.raises(ValueError, match="checksum"):
        executor.verify_test_view(root, permission, ctx)
