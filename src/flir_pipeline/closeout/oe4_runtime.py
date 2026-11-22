"""Slurm-only GPU workers. Imports are delayed until immutable gates pass."""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from flir_pipeline.closeout.common_test import verify_test, write_test_view
from flir_pipeline.closeout.io import read_json, seal, verify, write_csv, write_json
from flir_pipeline.closeout.oe4 import (
    AUG_ZERO,
    EVAL,
    OPERATIONAL,
    POLICIES,
    PROTOCOL,
    authorize,
    check_last_checkpoint_metadata,
    path,
    verify_freeze,
)
from flir_pipeline.detection.metrics import evaluate, load_image_stats, save_image_stats
from flir_pipeline.similarity.storage import file_sha256


def gpu_preflight():
    if not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("GPU execution requires an allocated Slurm job; never login")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    import ultralytics

    if (
        ultralytics.__version__ != "8.3.203"
        or torch.__version__.split("+")[0] != "2.7.1"
        or not torch.cuda.is_available()
    ):
        raise ValueError("BLOCKED: pinned Torch/Ultralytics/CUDA runtime unavailable")
    probe = torch.ones((16, 16), device="cuda")
    if float((probe @ probe).sum().item()) != 4096:
        raise ValueError("CUDA compute probe failed")
    return {
        "ultralytics": ultralytics.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
    }


def filter_shared(images):
    """Filter GT and post-class-aware-NMS detections together; no new inference.

    The common five-class max_det=300 budget is retained. Class-specific matching is
    unchanged by removing other classes. This is a restricted metric, not a two-class
    model or a second NMS policy.
    """
    result = []
    for item in images:
        keep = np.isin(item.predicted_class, [0, 1])
        fixed_pred, fixed_tp = (
            item.fixed_predictions.copy(),
            item.fixed_true_positives.copy(),
        )
        fixed_pred[2:], fixed_tp[2:] = 0, 0
        filtered = replace(
            item,
            confidence=item.confidence[keep],
            predicted_class=item.predicted_class[keep],
            true_positive=item.true_positive[keep],
            target_class=item.target_class[np.isin(item.target_class, [0, 1])],
            fixed_predictions=fixed_pred,
            fixed_true_positives=fixed_tp,
        )
        filtered.validate()
        result.append(filtered)
    return result


def metric_exports(images):
    from flir_pipeline.detection.common_executor import NAMES

    overall, table = evaluate(images)
    _, shared = evaluate(filter_shared(images))
    # The generic metric module uses a historical internal class catalogue.
    # OE4 retains original SDZI naming; it does not adjudicate machinery semantics.
    table["class_name"] = table.class_id.map(NAMES)
    shared["class_name"] = shared.class_id.map(NAMES)
    shared = shared.loc[shared.class_id.isin([0, 1])].copy()
    keys = ("precision", "recall", "map50", "map50_95")
    restricted = {
        k: float(shared[k].mean()) if shared[k].notna().all() else None for k in keys
    }
    return (
        {
            "five_class_macro": overall,
            "shared_01_macro": restricted,
            "shared_rule": "GT and predictions restricted to IDs 0/1 after common class-aware NMS and five-class max_det; absent support stays null",
            "precision_recall_confidence": 0.25,
        },
        table,
        shared,
    )


def train(config_path: Path, index: int):
    if index not in range(3):
        raise ValueError("Array index must be 0..2")
    cfg, freeze = authorize(config_path, "train")
    policy = POLICIES[index]
    env = gpu_preflight()
    from ultralytics import YOLO
    from ultralytics.cfg import DEFAULT_CFG_DICT

    from flir_pipeline.detection.runtime import (
        check_loader_annotations,
        configure_runtime,
        parse_training_history,
    )
    from flir_pipeline.detection.scientific_executor import write_view
    from flir_pipeline.detection.scientific_runtime import offline_runtime

    configure_runtime()
    baseline = freeze["runs"][policy]
    recorded = baseline["baseline_args"]
    # Full recorded configuration is frozen. New/default keys cannot silently enter.
    if set(DEFAULT_CFG_DICT) - set(recorded) or set(recorded) - set(
        DEFAULT_CFG_DICT
    ) - {"save_dir"}:
        raise ValueError("Recorded args do not cover pinned runtime keys exactly")
    output = path(cfg["aug_off_output"]) / policy
    output.mkdir(parents=True, exist_ok=False)
    state = {
        "state": "RUNNING_AUG_OFF",
        "protocol": PROTOCOL,
        "authorization_sha256": freeze["scientific_authorization"]["sha256"],
        "phase_authorization_receipt_sha256": file_sha256(
            path(cfg["authorization_output"]) / "train/receipt.json"
        ),
        "policy": policy,
        "freeze_sha256": file_sha256(path(cfg["freeze"]) / "receipt.json"),
        "augmentation": "AUG_OFF",
        "environment": env,
        "test_used": False,
        "initial_weights_sha256": freeze["checks"]["initial_weights_sha256"],
    }
    try:
        rows = pd.read_parquet(path(cfg["freeze"]) / f"development_{policy}.parquet")
        dataset = write_view(rows, output / "development", strict_geometry=False)
        settings = {
            **baseline["aug_off_args"],
            "data": str(dataset),
            "model": str(path(cfg["initial_weights"])),
            "device": "0",
            "project": str(output),
            "name": "train",
            "exist_ok": False,
            "resume": False,
        }
        write_json(output / "planned_arguments.json", settings)
        counts = rows.set_index("frame_id").num_objects.to_dict()

        def guard(trainer):
            actual = vars(trainer.args)
            for k, value in settings.items():
                if k not in OPERATIONAL and actual.get(k) != value:
                    raise ValueError(f"Effective AUG_OFF control changed: {k}")
            if trainer.data.get("test") or int(trainer.batch_size) != 1:
                raise ValueError("Test exposed or batch changed")
            check_loader_annotations(trainer, counts)
            for split, loader in (
                ("train", trainer.train_loader),
                ("val", trainer.test_loader),
            ):
                observed = sorted(
                    Path(label["im_file"]).stem for label in loader.dataset.labels
                )
                if observed != sorted(rows.loc[rows.new_split == split, "frame_id"]):
                    raise ValueError("Loader changed frozen development membership")

        with offline_runtime():
            model = YOLO(str(path(cfg["initial_weights"])))
            if model.task != "detect" or model.model.yaml.get("scale") != "m":
                raise ValueError("Expected YOLO11m detect initialization")
            model.add_callback("on_train_start", guard)
            model.train(**settings)
        state["history"] = parse_training_history(output / "train/results.csv", 50)
        effective = yaml.safe_load(
            (output / "train/args.yaml").read_text(encoding="utf-8")
        )
        if (
            any(effective.get(k) != 0 for k in AUG_ZERO)
            or effective.get("auto_augment") is not None
        ):
            raise ValueError("Effective AUG_OFF augmentation changed")
        for name in ("best.pt", "last.pt"):
            file_sha256(output / "train/weights" / name)
        verify_freeze(config_path)
        state.update(
            state="COMPLETED_AUG_OFF",
            checkpoint_selection="last_epoch_50_for_common_test; best preserved but unused",
        )
    except BaseException as exc:
        seal(output, {**state, "state": "FAILED_AUG_OFF", "error": str(exc)})
        raise
    return seal(output, state)


def evaluate_policy(config_path: Path, index: int):
    if index not in range(3):
        raise ValueError("Array index must be 0..2")
    cfg, freeze = authorize(config_path, "evaluate")
    env = gpu_preflight()
    from flir_pipeline.detection.common_evaluation import detailed_validator
    from flir_pipeline.detection.common_executor import NAMES
    from flir_pipeline.detection.runtime import capture_validator, configure_runtime
    from flir_pipeline.detection.scientific_runtime import offline_runtime

    configure_runtime()
    policy, common = POLICIES[index], path(cfg["common_test"])
    test_rows = verify_test(common)
    for treatment in ("AUG_ON", "AUG_OFF"):
        training = (
            Path(freeze["runs"][policy]["root"])
            if treatment == "AUG_ON"
            else path(cfg["aug_off_output"]) / policy
        )
        weights = training / "train/weights/last.pt"
        output = path(cfg["evaluation_output"]) / treatment / policy
        output.mkdir(parents=True, exist_ok=False)
        state = {
            "state": "EVALUATING",
            "protocol": PROTOCOL,
            "authorization_sha256": freeze["scientific_authorization"]["sha256"],
            "phase_authorization_receipt_sha256": file_sha256(
                path(cfg["authorization_output"]) / "evaluate/receipt.json"
            ),
            "policy": policy,
            "treatment": treatment,
            "freeze_sha256": file_sha256(path(cfg["freeze"]) / "receipt.json"),
            "training_receipt_sha256": file_sha256(
                training
                / ("execution.json" if treatment == "AUG_ON" else "receipt.json")
            ),
            "weights_sha256": file_sha256(weights),
            "test_receipt_sha256": file_sha256(common / "receipt.json"),
            "test_records_sha256": file_sha256(common / "records.csv"),
            "environment": env,
            "evaluation": EVAL,
            "split": "test",
            "checkpoint_selection": "last_epoch_50",
            "test_tuning": False,
        }
        try:
            view = output / "test-view"
            import torch

            checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
            check_last_checkpoint_metadata(
                checkpoint, pd.read_csv(training / "train/results.csv")
            )
            del checkpoint
            rows = write_test_view(test_rows.to_dict("records"), view, common)
            klass = detailed_validator(capture_validator(0.25))
            with offline_runtime():
                validator = klass(
                    args={
                        **EVAL,
                        "model": str(weights),
                        "data": str(view / "dataset.yaml"),
                        "split": "test",
                        "mode": "val",
                        "task": "detect",
                        "device": "0",
                        "plots": False,
                        "save_json": False,
                        "verbose": False,
                    },
                    save_dir=output / "validation",
                )
                validator()
            if validator.names != NAMES:
                raise ValueError("Original class ontology changed")
            stats = sorted(validator.image_statistics, key=lambda s: s.frame_id)
            support = {r["frame_id"]: r["num_objects"] for r in rows}
            if [s.frame_id for s in stats] != sorted(support) or any(
                len(s.target_class) != support[s.frame_id] for s in stats
            ):
                raise ValueError("Evaluation dropped/changed frozen occurrences or GT")
            metrics, table, shared = metric_exports(stats)
            own = metrics["five_class_macro"]["map50_95"]
            if own is not None and abs(own - float(validator.metrics.box.map)) > 0.001:
                raise ValueError("Pinned validator AP parity failed")
            save_image_stats(stats, output / "image_stats.npz")
            table.to_csv(output / "five_classes.csv", index=False)
            shared.to_csv(output / "shared_01.csv", index=False)
            pd.DataFrame(validator.prediction_rows).to_parquet(
                output / "predictions.parquet", index=False
            )
            write_json(output / "metrics.json", metrics)
            verify_freeze(config_path)
            state["state"] = "COMPLETED_EVALUATION"
        except BaseException as exc:
            seal(output, {**state, "state": "FAILED_EVALUATION", "error": str(exc)})
            raise
        seal(output, state)
    return {"policy": policy, "state": "COMPLETED_EVALUATION"}


def aggregate(config_path: Path):
    cfg, freeze = authorize(config_path, "evaluate")
    records, expected_test = [], None
    for treatment in ("AUG_ON", "AUG_OFF"):
        for policy in POLICIES:
            directory = path(cfg["evaluation_output"]) / treatment / policy
            receipt = verify(directory)
            if (
                receipt["state"] != "COMPLETED_EVALUATION"
                or receipt.get("protocol") != PROTOCOL
                or receipt.get("authorization_sha256")
                != freeze["scientific_authorization"]["sha256"]
                or receipt.get("phase_authorization_receipt_sha256")
                != file_sha256(
                    path(cfg["authorization_output"]) / "evaluate/receipt.json"
                )
                or receipt["evaluation"] != EVAL
                or receipt["policy"] != policy
                or receipt["treatment"] != treatment
                or receipt["freeze_sha256"]
                != file_sha256(path(cfg["freeze"]) / "receipt.json")
            ):
                raise ValueError("Incompatible evaluation receipt")
            test = (receipt["test_receipt_sha256"], receipt["test_records_sha256"])
            if expected_test is not None and test != expected_test:
                raise ValueError("Evaluations do not share exact frozen test")
            expected_test = test
            stats = load_image_stats(directory / "image_stats.npz")
            metrics, _, _ = metric_exports(stats)
            if metrics != read_json(directory / "metrics.json"):
                raise ValueError("Metrics differ from captured statistics")
            for cohort in ("five_class_macro", "shared_01_macro"):
                records.append(
                    {
                        "treatment": treatment,
                        "policy": policy,
                        "cohort": cohort,
                        **{
                            k: metrics[cohort][k]
                            for k in ("precision", "recall", "map50", "map50_95")
                        },
                    }
                )
    output = path(cfg["aggregate_output"])
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "comparison.csv", records)
    return seal(
        output,
        {
            "state": "AGGREGATED",
            "protocol": PROTOCOL,
            "test": expected_test,
            "interpretation": "Descriptive one-seed internal comparison; no causal or independence claim",
        },
    )
