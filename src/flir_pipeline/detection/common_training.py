"""Real, development-only YOLO11m executor behind signed admission.

This module is intentionally separate from the synthetic GPU probes and the
historical YOLO11n protocol. No test image is materialized or scored here.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import yaml

from flir_pipeline.detection.common_admission import verify_admission
from flir_pipeline.detection.common_executor import external_output
from flir_pipeline.detection.runtime import (
    configure_runtime,
    environment,
    parse_training_history,
)
from flir_pipeline.similarity.storage import file_sha256, read_json, write_json


def _runtime_matches(freeze: dict, env: dict) -> None:
    if not env["cuda_available"]:
        raise ValueError("Real training requires an allocated CUDA GPU")
    packages = env["packages"]
    for package, expected in (
        ("torch", freeze["torch"]),
        ("torchvision", freeze["torchvision"]),
        ("ultralytics", freeze["ultralytics"]),
    ):
        if packages[package].split("+")[0] != expected:
            raise ValueError(f"Runtime changed: {package}")
    if env["ultralytics_defaults_sha256"] != freeze["ultralytics_defaults_sha256"]:
        raise ValueError("Ultralytics defaults changed after freeze")
    import torch

    if freeze["tf32"] is not False or torch.backends.cuda.matmul.allow_tf32:
        raise ValueError("TF32 must be disabled for this experiment")
    if freeze["amp"] is not False:
        raise ValueError("Mixed-precision training not approved")


def _view_for_run(approval: dict, entry: dict, preparation: Path) -> Path:
    view_id = entry["identity"]["records_id"]
    prepared = approval["prepared"]
    if view_id not in prepared["views"]:
        raise ValueError("Run has no prepared development-only view")
    view = preparation / "views" / view_id
    config = yaml.safe_load((view / "dataset.yaml").read_text(encoding="utf-8"))
    if set(config) != {"path", "nc", "names", "train", "val"}:
        raise ValueError("Unexpected dataset configuration, test exposure possible")
    if config["names"] != {0: "vehicle", 1: "building", 2: "road", 3: "river", 4: "SDZI"}:
        raise ValueError("Wrong dataset class mapping")
    for split in ("train", "val"):
        if Path(config[split]).resolve() != (view / f"{split}.txt").resolve():
            raise ValueError("Dataset split points outside the approved view")
    return view


def verify_training_run(directory: Path, approval: dict, entry: dict) -> dict:
    """Check a complete run using its input identity and exact output hashes."""
    receipt = read_json(directory / "training_receipt.json")
    if receipt["state"] != "TRAINED_NO_TEST" or receipt["run_id"] != entry["run_id"]:
        raise ValueError("Not a completed scientific training")
    if receipt["approval_id"] != approval["approval_id"]:
        raise ValueError("Training admission differs")
    if receipt["source_records_id"] != entry["identity"]["records_id"]:
        raise ValueError("Run assigned to a different train/val cohort")
    if receipt["initial_checkpoint_sha256"] != approval["checkpoint_sha256"]:
        raise ValueError("Run initialized from an unapproved checkpoint")
    if receipt["test_evaluated"] is not False:
        raise ValueError("Training phase must not evaluate protected test")
    for name, expected in receipt["files"].items():
        if file_sha256(directory / name) != expected:
            raise ValueError("Training artifact checksum mismatch")
    history = parse_training_history(
        directory / "train/results.csv",
        entry["identity"]["configuration"]["train"]["epochs"],
    )
    if history["epochs_completed"] != receipt["epochs_completed"]:
        raise ValueError("Incomplete training epoch history")
    return receipt


def train_one(
    plan: Path, preparation: Path, output: Path, index: int,
    checkpoint: Path, authorization: Path, signature: Path, signers: Path,
    reviewer: str, resume: bool = False,
) -> dict:
    """Execute one deduplicated cell after independently signed approval.

    A SLURM retry never starts a different cell or reuses a checkpoint from
    another split/seed. Resume is fail-closed until checkpoint identity can
    be attested without loading an untrusted pickle.
    """
    approval = verify_admission(
        plan, preparation, checkpoint, authorization, signature, signers, reviewer
    )
    runs = approval["plan"]["runs"]
    if not 0 <= index < len(runs):
        raise ValueError("Run index outside the approved plan")
    entry = runs[index]
    if resume:
        raise ValueError(
            "Resume not yet admitted: require signed per-checkpoint provenance "
            "and validated Ultralytics resume invariants"
        )
    view = _view_for_run(approval, entry, preparation)
    run_id = entry["run_id"]
    external_output(output, plan, preparation, checkpoint, authorization, signers)
    directory = output / run_id
    if directory.exists():
        receipt = directory / "training_receipt.json"
        if receipt.exists():
            return verify_training_run(directory, approval, entry)
        raise ValueError("Partial training preserved; do not overwrite or auto-resume")

    configure_runtime()
    env = environment()
    freeze = approval["freeze"]
    _runtime_matches(freeze, env)

    import torch
    from ultralytics import YOLO

    config = entry["identity"]["configuration"]
    training = {**config["train"], "batch": freeze["batch"]}
    if (
        training["epochs"] != freeze["epochs"]
        or training["imgsz"] != freeze["imgsz"]
        or training["optimizer"] != freeze["optimizer"]
    ):
        raise ValueError("Training configuration differs from signed freeze")
    expected = pd.read_parquet(view / "records.parquet")
    if set(expected.new_split) != {"train", "val"}:
        raise ValueError("Prepared development set is missing train or validation")
    expected_frames = expected.set_index("frame_id")
    if expected_frames.index.has_duplicates:
        raise ValueError("Repeated frame identity in prepared dataset")
    expected_counts = expected_frames.num_objects.to_dict() if "num_objects" in expected else None
    batch_seen: list[int] = []
    directory.mkdir(parents=True)
    identity = {
        "run_id": run_id,
        "approval_id": approval["approval_id"],
        "plan_id": approval["plan_id"],
        "source_records_id": entry["identity"]["records_id"],
        "seed": entry["identity"]["detector_seed"],
        "checkpoint_sha256": approval["checkpoint_sha256"],
        "training": training,
        "test_evaluated": False,
    }
    write_json(directory / "run_identity.json", identity)
    write_json(directory / "state.json", {"state": "RUNNING", **identity})
    started = time.monotonic()
    try:
        model = YOLO(str(checkpoint.resolve()))

        def check_batch(trainer) -> None:
            observed = int(trainer.batch_size)
            batch_seen.append(observed)
            if observed != freeze["batch"] or trainer.args.split != "val":
                raise ValueError("Effective batch changed or non-validation split used")

        def check_annotations(trainer) -> None:
            if expected_counts is None:
                return
            from flir_pipeline.detection.runtime import check_loader_annotations
            check_loader_annotations(trainer, expected_counts)

        model.add_callback("on_train_batch_end", check_batch)
        model.add_callback("on_train_start", check_annotations)
        torch.cuda.reset_peak_memory_stats()
        model.train(
            **training,
            data=str((view / "dataset.yaml").resolve()),
            device="0",
            seed=entry["identity"]["detector_seed"],
            split="val",
            project=str(directory.resolve()),
            name="train",
            exist_ok=False,
            save=True,
        )
        if not batch_seen or set(batch_seen) != {freeze["batch"]}:
            raise ValueError("No controlled training batches recorded")
        history = parse_training_history(
            directory / "train/results.csv", training["epochs"]
        )
        effective = yaml.safe_load(
            (directory / "train/args.yaml").read_text(encoding="utf-8")
        )
        if any(effective.get(k) != value for k, value in training.items()):
            raise ValueError("Ultralytics changed approved training arguments")
        if effective.get("seed") != identity["seed"] or effective.get("split") != "val":
            raise ValueError("Unexpected seed or validation split")
        required = (
            "run_identity.json", "train/args.yaml", "train/results.csv",
            "train/weights/best.pt", "train/weights/last.pt",
        )
        receipt = {
            "state": "TRAINED_NO_TEST",
            "run_id": run_id,
            "approval_id": approval["approval_id"],
            "source_records_id": identity["source_records_id"],
            "initial_checkpoint_sha256": approval["checkpoint_sha256"],
            "seed": identity["seed"],
            "epochs_completed": history["epochs_completed"],
            "validation_summary": history,
            "observed_batches": sorted(set(batch_seen)),
            "seconds": time.monotonic() - started,
            "peak_gpu_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "environment": env,
            "test_evaluated": False,
            "files": {name: file_sha256(directory / name) for name in required},
        }
        write_json(directory / "training_receipt.json", receipt)
        verify_training_run(directory, approval, entry)
        write_json(directory / "state.json", {"state": "TRAINED_NO_TEST", **identity})
        return receipt
    except BaseException as error:
        write_json(
            directory / "state.json",
            {
                "state": "INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAILED",
                **identity,
                "error_type": type(error).__name__,
                "error": str(error),
                "seconds": time.monotonic() - started,
                "partial_preserved": True,
            },
        )
        raise
