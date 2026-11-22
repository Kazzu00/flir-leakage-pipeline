"""Offline communicated authorization with real sealed freeze/phase integrity."""

import copy
import shutil
from pathlib import Path

import pandas as pd
import pytest
import yaml
from PIL import Image

from flir_pipeline.closeout import oe4, oe4_runtime
from flir_pipeline.closeout.io import read_json, seal, write_json


@pytest.fixture
def operation(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config = config_dir / "oe4.yaml"
    cfg = oe4.load_config(Path("configs/closeout/oe4_20261115.yaml"))
    shutil.copyfile(
        "configs/closeout/oe4_authorization_20261122.yaml",
        config_dir / cfg["authorization_record"],
    )
    for key in (
        "freeze",
        "common_test",
        "aug_off_output",
        "authorization_output",
        "evaluation_output",
        "aggregate_output",
        "initial_weights",
    ):
        cfg[key] = str(tmp_path / key)
    config.write_text(yaml.safe_dump(cfg, sort_keys=False))
    Path(cfg["initial_weights"]).write_bytes(b"synthetic initialization")
    common = Path(cfg["common_test"])
    common.mkdir()
    pd.DataFrame({"frame_id": [f"test-{i}" for i in range(372)]}).to_csv(
        common / "records.csv", index=False
    )
    seal(common, {"state": "MATERIALIZED_NOT_RELEASED"})
    freeze = Path(cfg["freeze"])
    freeze.mkdir()
    authorization = oe4.communicated_authorization(config, cfg)
    write_json(freeze / "authorization.json", authorization)
    runs = {}
    for policy in oe4.POLICIES:
        baseline = tmp_path / f"original-{policy}"
        baseline.mkdir()
        (baseline / "dataset.yaml").write_text("synthetic original development")
        image, label = baseline / "image.png", baseline / "label.txt"
        Image.new("RGB", (32, 32), (10, 20, 30)).save(image)
        label.write_text("0 .5 .5 .1 .1\n")
        validation_image, validation_label = baseline / "val.png", baseline / "val.txt"
        Image.new("RGB", (32, 32), (30, 20, 10)).save(validation_image)
        validation_label.write_text("1 .5 .5 .1 .1\n")
        seal(baseline, {"state": "completed"})
        runs[policy] = {
            "root": str(baseline),
            "execution_sha256": oe4.file_sha256(baseline / "receipt.json"),
            "dataset_yaml": str(baseline / "dataset.yaml"),
            "dataset_yaml_sha256": oe4.file_sha256(baseline / "dataset.yaml"),
        }
        from test_oe4 import baseline as example_baseline

        controls = example_baseline()
        try:
            from ultralytics.cfg import DEFAULT_CFG_DICT

            controls = {**DEFAULT_CFG_DICT, **controls}
        except ImportError:
            pass
        runs[policy]["baseline_args"] = controls
        runs[policy]["aug_off_args"] = oe4.aug_off(controls)
        # Native receipt uses execution.json; fixture keeps the same immutable binding.
        shutil.copyfile(baseline / "receipt.json", baseline / "execution.json")
        pd.DataFrame(
            [
                {
                    "frame_id": "train_frame",
                    "new_split": "train",
                    "image_path": str(image),
                    "label_path": str(label),
                    "image_sha256": oe4.file_sha256(image),
                    "label_sha256": oe4.file_sha256(label),
                },
                {
                    "frame_id": "val_frame",
                    "new_split": "val",
                    "image_path": str(validation_image),
                    "label_path": str(validation_label),
                    "image_sha256": oe4.file_sha256(validation_image),
                    "label_sha256": oe4.file_sha256(validation_label),
                },
            ]
        ).to_parquet(freeze / f"development_{policy}.parquet")
    report = {
        "protocol": oe4.PROTOCOL,
        "state": "READY",
        "blockers": [],
        "configuration_sha256": oe4.file_sha256(config),
        "code": oe4.code_hashes(),
        "scientific_authorization": authorization,
        "runs": runs,
        "checks": {
            "initial_weights_sha256": oe4.file_sha256(Path(cfg["initial_weights"])),
            "common_test_receipt_sha256": oe4.file_sha256(common / "receipt.json"),
        },
    }
    write_json(freeze / "preflight.json", report)
    seal(freeze, {"state": "READY"})
    return config, cfg, report


def complete_runs(cfg, report):
    for policy in oe4.POLICIES:
        root = Path(cfg["aug_off_output"]) / policy
        (root / "train/weights").mkdir(parents=True)
        for name in ("args.yaml", "results.csv", "weights/last.pt", "weights/best.pt"):
            (root / "train" / name).write_bytes(b"synthetic complete output")
        seal(
            root,
            {
                "state": "COMPLETED_AUG_OFF",
                "protocol": oe4.PROTOCOL,
                "policy": policy,
                "augmentation": "AUG_OFF",
                "test_used": False,
                "initial_weights_sha256": report["checks"]["initial_weights_sha256"],
                "authorization_sha256": report["scientific_authorization"]["sha256"],
                "freeze_sha256": oe4.file_sha256(Path(cfg["freeze"]) / "receipt.json"),
            },
        )


def test_train_records_communication_without_signature_infrastructure(
    operation, monkeypatch
):
    config, cfg, report = operation
    from flir_pipeline.detection import admission

    def forbidden(*args, **kwargs):
        pytest.fail("OE4 must not invoke the historical signature verifier")

    monkeypatch.setattr(admission, "authorized", forbidden)
    oe4.authorize(config, "train")
    root = Path(cfg["authorization_output"]) / "train"
    first = oe4.file_sha256(root / "receipt.json")
    oe4.authorize(config, "train")
    assert oe4.file_sha256(root / "receipt.json") == first
    record = read_json(root / "authorization.json")
    assert record["authorization"] == report["scientific_authorization"]
    assert record["authorization"]["record"]["cryptographically_verified"] is False
    assert record["authorization"]["record"]["documentary_approval_verified"] is False
    assert record["freeze_sha256"] == oe4.file_sha256(
        Path(cfg["freeze"]) / "receipt.json"
    )


def test_evaluation_requires_all_three_bound_receipts(operation):
    config, cfg, report = operation
    with pytest.raises(FileNotFoundError):
        oe4.authorize(config, "evaluate")
    complete_runs(cfg, report)
    oe4.authorize(config, "evaluate")
    release = read_json(
        Path(cfg["authorization_output"]) / "evaluate/authorization.json"
    )
    assert set(release["aug_off_receipts"]) == set(oe4.POLICIES)
    (Path(cfg["aug_off_output"]) / "c10/train/weights/last.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        oe4.authorize(config, "evaluate")


@pytest.mark.parametrize(
    "field,value",
    [
        ("protocol", "oe4_closeout_v1"),
        ("state", "RUNNING_AUG_OFF"),
        ("policy", "wrong"),
        ("freeze_sha256", "wrong"),
        ("authorization_sha256", "wrong"),
        ("test_used", True),
        ("initial_weights_sha256", "wrong"),
    ],
)
def test_incompatible_training_receipts_rejected(operation, field, value):
    config, cfg, report = operation
    complete_runs(cfg, report)
    root = Path(cfg["aug_off_output"]) / "c10"
    receipt = read_json(root / "receipt.json")
    receipt[field] = value
    write_json(root / "receipt.json", receipt)
    with pytest.raises(ValueError, match="All three"):
        oe4.authorize(config, "evaluate")


@pytest.mark.parametrize(
    "target", ["initial_weights", "development", "authorization", "freeze", "test"]
)
def test_integrity_still_blocks_training(operation, target):
    config, cfg, _ = operation
    if target == "initial_weights":
        Path(cfg[target]).write_bytes(b"changed")
    elif target == "development":
        (config.parent.parent / "original-c10/image.png").write_bytes(b"changed")
    elif target == "authorization":
        with (config.parent / cfg["authorization_record"]).open("a") as stream:
            stream.write("\n# changed statement bytes\n")
    elif target == "freeze":
        (Path(cfg["freeze"]) / "authorization.json").write_text("{}")
    else:
        (Path(cfg["common_test"]) / "records.csv").write_text("frame_id\nchanged\n")
    with pytest.raises(ValueError):
        oe4.authorize(config, "train")


@pytest.mark.parametrize(
    "phase,worker",
    [("train", oe4_runtime.train), ("evaluate", oe4_runtime.evaluate_policy)],
)
def test_workers_recheck_authorization_before_gpu(
    operation, monkeypatch, phase, worker
):
    config, cfg, report = operation
    if phase == "evaluate":
        complete_runs(cfg, report)

    class StopBeforeGPU(Exception):
        pass

    def stop():
        assert (Path(cfg["authorization_output"]) / phase / "receipt.json").is_file()
        raise StopBeforeGPU

    monkeypatch.setattr(oe4_runtime, "gpu_preflight", stop)
    with pytest.raises(StopBeforeGPU):
        worker(config, 0)
    # Re-entry rejects tampering before the GPU callback can run.
    (Path(cfg["freeze"]) / "authorization.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        worker(config, 0)


def test_report_cannot_claim_verified_signature_or_reuse_v1_config(tmp_path):
    config = Path("configs/closeout/oe4_20261115.yaml")
    cfg = oe4.load_config(config)
    statement = yaml.safe_load(
        (config.parent / cfg["authorization_record"]).read_text()
    )
    statement["cryptographically_verified"] = True
    (tmp_path / "statement.yaml").write_text(yaml.safe_dump(statement))
    with pytest.raises(ValueError, match="communicated"):
        oe4.communicated_authorization(
            tmp_path / "config", {"authorization_record": "statement.yaml"}
        )
    cfg = copy.deepcopy(cfg)
    cfg["protocol"] = "oe4_closeout_v1"
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    with pytest.raises(ValueError, match="protocol"):
        oe4.load_config(tmp_path / "config.yaml")


def test_real_submission_without_gpu_or_ssh_signatures(operation, tmp_path):
    """Execute real Bash -> bundle verifier -> CLI -> authorize -> fake sbatch."""
    pytest.importorskip("ultralytics")
    import hashlib
    import json
    import os
    import subprocess
    import sys

    from test_hypatia_commands import bash_path

    bash = bash_path()
    if not bash:
        pytest.skip("Bash unavailable")
    config, cfg, report = operation
    bundle = tmp_path / "bundle with spaces"
    original = Path.cwd()
    for source in (original / "src").rglob("*.py"):
        target = bundle / source.relative_to(original)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    for name in (
        "pyproject.toml",
        "uv.lock",
        "scripts/closeout/verify_bundle.py",
        "scripts/hypatia/submit_oe4.sh",
        "scripts/hypatia/oe4.sbatch",
    ):
        target = bundle / name
        target.parent.mkdir(parents=True, exist_ok=True)
        # Bash requires LF even in a Windows checkout.
        raw = (original / name).read_bytes()
        target.write_bytes(
            raw.replace(b"\r\n", b"\n") if name.startswith("scripts/") else raw
        )
    files = {
        p.relative_to(bundle).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in bundle.rglob("*")
        if p.is_file()
    }
    (bundle / "BUNDLE_MANIFEST.json").write_text(json.dumps({"files": files}))
    binary = tmp_path / "bin"
    binary.mkdir()
    interpreter = binary / "python-wrapper"
    interpreter.write_text(
        f'#!/usr/bin/env bash\nexec "{Path(sys.executable).as_posix()}" "$@"\n',
        newline="\n",
    )
    fake_sbatch = binary / "sbatch"
    fake_sbatch.write_text(
        '#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$SUBMIT_LOG"\nprintf "synthetic-job-id\\n"\n',
        newline="\n",
    )
    interpreter.chmod(0o755)
    fake_sbatch.chmod(0o755)
    log = tmp_path / "submitted.txt"
    posix_bin = binary.as_posix()
    if os.name == "nt":
        posix_bin = "/" + posix_bin[0].lower() + posix_bin[2:]
    env = {
        **os.environ,
        # The simulated deployment must select its own source tree. Inherited
        # Windows PYTHONPATH values break Git Bash's POSIX path-list conversion.
        "PYTHONPATH": "",
        "FAKE_BIN": posix_bin,
        "FLIR_OE4_PYTHON": interpreter.as_posix(),
        "FLIR_OE4_CONFIG": config.as_posix(),
        "SUBMIT_SCRIPT": (bundle / "scripts/hypatia/submit_oe4.sh").as_posix(),
        "SUBMIT_LOG": log.as_posix(),
        "HOME": (tmp_path / "home").as_posix(),
    }

    def submit(phase):
        return subprocess.run(
            [
                bash,
                "-c",
                'export PATH="$FAKE_BIN:$PATH"; exec bash "$SUBMIT_SCRIPT" "$1"',
                "test",
                phase,
            ],
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

    result = submit("train")
    assert result.returncode == 0, result.stderr
    assert "AUTHORIZED_BY_RECORDED_COMMUNICATION" in result.stdout
    assert "VERIFIED" in result.stdout and "synthetic-job-id" in result.stdout
    calls = log.read_text().splitlines()
    assert "--array=0-2%1" in calls and calls[-4] == "train"
    assert calls[-1].endswith("bundle with spaces")
    log.unlink()
    result = submit("evaluate")
    assert result.returncode != 0 and not log.exists()
    complete_runs(cfg, report)
    result = submit("evaluate")
    assert result.returncode == 0, result.stderr
    log.unlink()
    (Path(cfg["freeze"]) / "authorization.json").write_text("{}")
    result = submit("train")
    assert result.returncode != 0 and not log.exists()
