"""Isolated synthetic operations for the common-test executor, never FLIR pilots."""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import yaml

from flir_pipeline.detection.common_executor import (
    NAMES,
    VERSION,
    external_output,
    load_executor_config,
)
from flir_pipeline.detection.common_test_storage import signatures
from flir_pipeline.detection.runtime import (
    configure_runtime,
    environment,
    parse_training_history,
    provenance,
)
from flir_pipeline.similarity.storage import (
    file_sha256,
    read_json,
    stable_id,
    write_json,
)


def preflight(config_path: Path, output: Path, cpu: bool = False) -> dict:
    external_output(output)
    if output.exists():
        raise ValueError("Preflight output exists")
    config = load_executor_config(config_path)
    configure_runtime()
    env = environment()
    blockers = []
    if sys.version_info[:2] != (3, 11):
        blockers.append("python_3_11_required")
    if env["packages"]["torchvision"].split("+")[0] != "0.22.1":
        blockers.append("incompatible_torchvision")
    for name in ("torch", "ultralytics"):
        if env["packages"][name].split("+")[0] != config[f"{name}_version"]:
            blockers.append(f"incompatible_{name}")
    if not cpu and not env["cuda_available"]:
        blockers.append("cuda_required_no_cpu_fallback")
    report = {
        "schema_version": VERSION,
        "environment": env,
        "blockers": blockers,
        "config_id": stable_id(config),
        "provenance": provenance(),
        "state": "failed" if blockers else "completed",
        "device": "cpu" if cpu else "0",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, report)
    if blockers:
        raise ValueError("Preflight failed: " + "; ".join(blockers))
    return report


def synthetic_smoke(
    config_path: Path, output: Path, batch: int = 1, cpu: bool = False
) -> dict:
    """One synthetic epoch at protocol resolution, random YOLO11m initialization.

    No external checkpoint, downloader, publication, or private data is consumed.
    A successful synthetic probe does not establish FLIR memory sufficiency.
    """
    external_output(output)
    if output.exists() or batch not in (1, 2):
        raise ValueError("Fresh smoke output and explicit batch 1 or 2 required")
    config = load_executor_config(config_path)
    output.mkdir(parents=True)
    state = {
        "schema_version": VERSION,
        "mode": "synthetic-smoke",
        "state": "running",
        "identity": {
            "config": config,
            "batch": batch,
            "seed": 42,
            "initialization": "random_yolo11m.yaml",
            "device": "cpu" if cpu else "0",
        },
        "scientific_training_executed": False,
        "limitation": "Synthetic memory measurement is not a FLIR capacity guarantee",
    }
    write_json(output / "state.json", state)
    try:
        checked = preflight(config_path, output / "environment.json", cpu)
        state["environment"] = checked
        import torch
        from PIL import Image, ImageDraw
        from ultralytics import YOLO

        for split in ("train", "val"):
            for i in range(10):
                image = output / "synthetic/images" / split / f"{i}.png"
                label = output / "synthetic/labels" / split / f"{i}.txt"
                image.parent.mkdir(parents=True, exist_ok=True)
                label.parent.mkdir(parents=True, exist_ok=True)
                raster = Image.new("RGB", (640, 640), (i * 11, 30, 40))
                ImageDraw.Draw(raster).rectangle(
                    (160, 160, 480, 480), fill=(180, 150, 80)
                )
                raster.save(image)
                label.write_text(f"{i % 5} 0.5 0.5 0.5 0.5\n", encoding="utf-8")
        dataset = output / "synthetic.yaml"
        dataset.write_text(
            yaml.safe_dump(
                {
                    "path": str((output / "synthetic").resolve()),
                    "train": "images/train",
                    "val": "images/val",
                    "names": NAMES,
                    "nc": 5,
                }
            ),
            encoding="utf-8",
        )
        if not cpu:
            torch.cuda.reset_peak_memory_stats()
        model = YOLO("yolo11m.yaml")
        observed = []
        model.add_callback(
            "on_train_batch_end", lambda t: observed.append(int(t.batch_size))
        )
        train = {
            **config["train"],
            "batch": batch,
            "epochs": 1,
            "pretrained": False,
            "close_mosaic": 0,
        }
        model.train(
            **train,
            data=str(dataset.resolve()),
            seed=42,
            device="cpu" if cpu else "0",
            project=str(output.resolve()),
            name="training",
            exist_ok=False,
            save=True,
        )
        if not observed or set(observed) != {batch}:
            raise ValueError("Unexpected runtime batch modification")
        history = parse_training_history(output / "training/results.csv", 1)
        checkpoints = {
            name: file_sha256(output / "training/weights" / name)
            for name in ("best.pt", "last.pt")
        }
        state.update(
            state="completed",
            actual_epochs=1,
            observed_batches=observed,
            history=history,
            checkpoint_sha256=checkpoints,
            peak_memory_bytes=None if cpu else int(torch.cuda.max_memory_allocated()),
        )
    except BaseException as exc:
        state.update(
            state="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        write_json(output / "state.json", state)
        write_json(output / "receipt.json", {"artifacts": signatures(output)})
        raise
    write_json(output / "state.json", state)
    write_json(output / "receipt.json", {"artifacts": signatures(output)})
    return state


def verify_operation(output: Path) -> dict:
    expected = read_json(output / "receipt.json")["artifacts"]
    actual = signatures(output)
    actual.pop("receipt.json", None)
    if actual != expected:
        raise ValueError("Operation/checkpoint checksum mismatch")
    state = read_json(output / "state.json")
    if state["schema_version"] != VERSION or state["mode"] != "synthetic-smoke":
        raise ValueError("Not a synthetic executor operation")
    if state["state"] == "completed":
        parse_training_history(output / "training/results.csv", 1)
        for name in ("best.pt", "last.pt"):
            if (
                file_sha256(output / "training/weights" / name)
                != state["checkpoint_sha256"][name]
            ):
                raise ValueError("Output checkpoint provenance mismatch")
    return state


def probe(config_path: Path, output: Path) -> dict:
    external_output(output)
    if output.exists():
        raise ValueError("Probe destination exists")
    output.mkdir(parents=True)
    checked = preflight(config_path, output / "environment.json")
    import torch

    attempts = []
    for batch in (1, 2):
        try:
            result = synthetic_smoke(config_path, output / f"batch-{batch}", batch)
        except torch.cuda.OutOfMemoryError:
            attempts.append({"batch": batch, "state": "failed", "reason": "cuda_oom"})
            gc.collect()
            torch.cuda.empty_cache()
            break
        else:
            attempts.append(
                {
                    "batch": batch,
                    "state": "completed",
                    "peak_memory_bytes": result["peak_memory_bytes"],
                }
            )
    passed = [a["batch"] for a in attempts if a["state"] == "completed"]
    result = {
        "environment": checked,
        "attempts": attempts,
        "candidate_batch": max(passed) if passed else None,
        "batch_frozen": False,
        "ready_for_training": False,
        "limitation": "Synthetic only; common batch requires approved non-test pilot and freeze",
    }
    write_json(output / "probe.json", result)
    if not passed:
        raise ValueError("No tested batch fits; no CPU fallback")
    return result
