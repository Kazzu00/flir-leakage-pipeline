"""Optional admitted YOLO11m runtime. Test release is a separate operation.

Ultralytics 8.3.203 is pinned; synthetic doubles exercise this boundary offline.
No checkpoints are downloaded here, including during resume or evaluation.
"""

from __future__ import annotations

import platform
import shutil
import signal
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import yaml

from flir_pipeline.detection.admission import bound_file
from flir_pipeline.detection.common_executor import NAMES
from flir_pipeline.detection.metrics import evaluate, save_image_stats
from flir_pipeline.detection.runtime import (
    capture_validator,
    check_loader_annotations,
    configure_runtime,
    environment,
    environment_signature,
    parse_training_history,
    provenance,
)
from flir_pipeline.detection.scientific_executor import seal, variant_rows, write_view
from flir_pipeline.similarity.storage import file_sha256, write_json


@contextmanager
def offline_runtime():
    def blocked(*args, **kwargs):
        raise RuntimeError("Network/download forbidden in scientific worker")

    # Single-device execution has no distributed socket requirements.
    with (
        patch("socket.socket.connect", blocked),
        patch("socket.create_connection", blocked),
    ):
        yield


@contextmanager
def termination_receipt():
    def terminate(signum, frame):
        raise KeyboardInterrupt(f"scheduler signal {signum}")

    previous = signal.signal(signal.SIGTERM, terminate)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def frozen_environment(ctx: dict) -> dict:
    configure_runtime()
    env = environment()
    expected = ctx["admission"]["environment"]
    if not env["cuda_available"] or platform.python_version() != expected["python"]:
        raise ValueError("Approved Python/CUDA runtime required; no CPU fallback")
    if environment_signature(env) != expected["signature"]:
        raise ValueError("Runtime differs from signed environment freeze")
    for package in ("torch", "ultralytics"):
        if (
            env["packages"][package].split("+")[0]
            != ctx["config"][f"{package}_version"]
        ):
            raise ValueError("Runtime version differs from protocol")
    return env


def check_arguments(args: dict, config: dict, seed: int, dataset: Path) -> None:
    required = {**config["train"], "seed": seed, "split": "val"}
    if any(args.get(key) != value for key, value in required.items()):
        raise ValueError("Effective training configuration changed")
    if Path(args["data"]).resolve() != dataset.resolve():
        raise ValueError("Training dataset locator changed")


def check_resume_state(checkpoint: dict, csv: Path, epochs: int) -> None:
    history = pd.read_csv(csv)
    if (
        not 0 < len(history) < epochs
        or history.epoch.tolist() != list(range(1, len(history) + 1))
        or checkpoint.get("epoch") != len(history) - 1
        or checkpoint.get("optimizer") is None
    ):
        raise ValueError("Checkpoint epoch/optimizer differs from interrupted history")


def run_training(ctx: dict, run: dict, directory: Path, previous: Path | None) -> dict:
    env = frozen_environment(ctx)
    if directory.exists():
        raise ValueError("Attempt exists; never overwrite")
    directory.mkdir(parents=True)
    state = {
        "state": "running",
        "run": run,
        "variant": ctx["admission"]["variant"],
        "admission_sha256": ctx["admission_sha256"],
        "environment": env,
        "provenance": provenance(),
        "test_used": False,
        "previous_execution_sha256": file_sha256(previous / "execution.json")
        if previous
        else None,
        "resume_limitation": "Optimizer/EMA restored; bitwise RNG continuity is not claimed",
    }
    config = ctx["config"]
    train_dir = directory / "train"
    try:
        rows = pd.read_parquet(
            ctx["prepared"] / "views" / run["records_id"] / "records.parquet"
        )
        rows["label_path"] = rows.image_path.map(
            lambda p: str(Path(p).parent.parent / "labels" / (Path(p).stem + ".txt"))
        )
        rows = variant_rows(rows, ctx["admission"]["variant"])
        dataset = write_view(rows, directory / "development")
        input_hashes = {
            p: file_sha256(p)
            for p in (directory / "development").rglob("*")
            if p.is_file()
        }
        expected_counts = (
            pd.read_parquet(directory / "development/records.parquet")
            .set_index("frame_id")
            .num_objects.to_dict()
        )
        initial = bound_file(ctx["admission"]["weights"])
        checkpoint = previous / "train/weights/last.pt" if previous else initial
        if not checkpoint.is_file():
            raise ValueError("Interrupted run lacks last.pt; preserve and review")
        # Import after authorization/environment gate and CUDA initialization.
        from ultralytics import YOLO
        from ultralytics.models.yolo.detect import DetectionTrainer

        class FrozenTrainer(DetectionTrainer):
            def resume_training(self, checkpoint):
                if previous:
                    check_resume_state(
                        checkpoint,
                        previous / "train/results.csv",
                        config["train"]["epochs"],
                    )
                super().resume_training(checkpoint)
                if self.epochs != config["train"]["epochs"]:
                    raise ValueError("Runtime extended the frozen epoch budget")

            def check_resume(self, overrides):
                super().check_resume(overrides)
                if previous:
                    old_data = previous / "development/dataset.yaml"
                    check_arguments(
                        vars(self.args), config, run["detector_seed"], old_data
                    )
                    # Resume into a new attempt, preserving all prior evidence.
                    self.args.data = str(dataset.resolve())
                    self.args.project = str(directory.resolve())
                    self.args.name = "train"
                    self.args.save_dir = str(train_dir.resolve())
                    self.args.exist_ok = True

        if previous:
            train_dir.mkdir()
            (train_dir / "weights").mkdir()
            for name in ("results.csv", "weights/best.pt"):
                source = previous / "train" / name
                if source.is_file():
                    shutil.copyfile(source, train_dir / name)
            if (train_dir / "results.csv").is_file() and len(
                pd.read_csv(train_dir / "results.csv")
            ) >= config["train"]["epochs"]:
                raise ValueError("Full epoch budget already reached; no extra training")

        with offline_runtime(), termination_receipt():
            model = YOLO(str(checkpoint.resolve()))
            if model.task != "detect" or model.model.yaml.get("scale") != "m":
                raise ValueError("Checkpoint is not the admitted YOLO11m detector")

            def guard(trainer):
                check_arguments(
                    vars(trainer.args), config, run["detector_seed"], dataset
                )
                if int(trainer.batch_size) != config["train"]["batch"]:
                    raise ValueError("Runtime changed common batch")
                if trainer.data.get("test") is not None:
                    raise ValueError("Test dataset exposed to trainer")

            def start(trainer):
                guard(trainer)
                check_loader_annotations(trainer, expected_counts)
                observed = [
                    Path(label["im_file"]).stem
                    for loader in (trainer.train_loader, trainer.test_loader)
                    for label in loader.dataset.labels
                ]
                if sorted(observed) != sorted(expected_counts):
                    raise ValueError("Loader dropped or duplicated development members")
                for split, loader in (
                    ("train", trainer.train_loader),
                    ("val", trainer.test_loader),
                ):
                    if sorted(
                        Path(label["im_file"]).stem for label in loader.dataset.labels
                    ) != sorted(rows.loc[rows.new_split == split, "frame_id"]):
                        raise ValueError(
                            "Loader changed frozen train/validation assignment"
                        )

            model.add_callback("on_train_start", start)
            model.add_callback("on_train_batch_end", guard)

            def saved(trainer):
                guard(trainer)
                seal(
                    directory,
                    {
                        **state,
                        "state": "interrupted",
                        "epoch": int(trainer.epoch) + 1,
                        "recovery": "last complete checkpoint; check scheduler exit before resuming",
                    },
                    input_hashes,
                )

            model.add_callback("on_model_save", saved)
            model.train(
                trainer=FrozenTrainer,
                **config["train"],
                data=str(dataset.resolve()),
                seed=run["detector_seed"],
                split="val",
                device="0",
                project=str(directory.resolve()),
                name="train",
                exist_ok=bool(previous),
                save=True,
                resume=str(checkpoint.resolve()) if previous else False,
            )
        history = parse_training_history(
            train_dir / "results.csv", config["train"]["epochs"]
        )
        actual = yaml.safe_load((train_dir / "args.yaml").read_text(encoding="utf-8"))
        check_arguments(actual, config, run["detector_seed"], dataset)
        if any(file_sha256(path) != sha for path, sha in input_hashes.items()):
            raise ValueError("Runtime changed frozen inputs")
        state.update(
            state="completed",
            history=history,
            checkpoint_sha256={
                name: file_sha256(train_dir / "weights" / name)
                for name in ("best.pt", "last.pt")
            },
            effective_arguments=actual,
            checkpoint_selection="validation_only",
        )
    except BaseException as exc:
        state.update(
            state="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        seal(directory, state)
        raise
    return seal(directory, state)


def run_evaluation(
    ctx: dict, run: dict, directory: Path, test_root: Path, output: Path
) -> dict:
    env = frozen_environment(ctx)
    output.mkdir(parents=True)
    state = {
        "state": "evaluating",
        "run": run,
        "variant": ctx["admission"]["variant"],
        "admission_sha256": ctx["admission_sha256"],
        "environment": env,
        "training_execution_sha256": file_sha256(directory / "execution.json"),
        "sealed_test_sha256": file_sha256(test_root / "execution.json"),
        "checkpoint_selection": "validation_only",
        "test_tuning": False,
    }
    try:
        evaluation = ctx["config"]["evaluation"]
        validator_type = capture_validator(evaluation["precision_recall_confidence"])
        with offline_runtime(), termination_receipt():
            validator = validator_type(
                args={
                    "model": str((directory / "train/weights/best.pt").resolve()),
                    "data": str((test_root / "view/dataset.yaml").resolve()),
                    "split": "test",
                    "mode": "val",
                    "task": "detect",
                    "device": "0",
                    "imgsz": ctx["config"]["train"]["imgsz"],
                    "batch": ctx["config"]["train"]["batch"],
                    "workers": 0,
                    "conf": evaluation["confidence_floor"],
                    "iou": evaluation["nms_iou"],
                    "max_det": evaluation["max_det"],
                    "half": False,
                    "rect": False,
                    "augment": False,
                    "plots": False,
                    "save_json": False,
                    "verbose": False,
                },
                save_dir=output / "validation",
            )
            validator()
        if validator.names != NAMES:
            raise ValueError("Approved original class IDs/names changed")
        images = sorted(validator.image_statistics, key=lambda item: item.frame_id)
        expected = pd.read_parquet(test_root / "view/records.parquet")
        counts = expected.set_index("frame_id").num_objects.to_dict()
        if [item.frame_id for item in images] != sorted(counts) or any(
            len(item.target_class) != counts[item.frame_id] for item in images
        ):
            raise ValueError("Test membership/annotation coverage changed")
        overall, classes = evaluate(images)
        # Existing metrics use canonical IDs; preserve the approved original name.
        classes["class_name"] = classes.class_id.map(NAMES)
        if (
            overall["map50_95"] is not None
            and abs(overall["map50_95"] - float(validator.metrics.box.map)) > 1e-3
        ):
            raise ValueError("Metric parity with pinned validator failed")
        save_image_stats(images, output / "test_image_stats.npz")
        classes.to_parquet(output / "metrics_per_class.parquet", index=False)
        write_json(
            output / "metrics.json",
            {
                "overall": overall,
                "definition": evaluation,
                "bootstrap_executed": False,
                "test_tuning": False,
            },
        )
        state["state"] = "completed"
    except BaseException as exc:
        state.update(state="failed", error_type=type(exc).__name__, error=str(exc))
        seal(output, state)
        raise
    return seal(output, state)
