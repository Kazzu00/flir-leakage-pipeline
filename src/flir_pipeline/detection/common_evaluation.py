"""Common-test release after the whole matrix, paired descriptive comparisons."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from flir_pipeline.detection.admission import bound_file
from flir_pipeline.detection.common_admission import (
    POLICY,
    binding,
    emit,
    receipt,
    verify_admission,
)
from flir_pipeline.detection.common_executor import NAMES, external_output
from flir_pipeline.detection.common_training import (
    completed_matrix,
    frozen_context,
    worker_lock,
)
from flir_pipeline.detection.metrics import evaluate, load_image_stats
from flir_pipeline.detection.protocol import METRICS
from flir_pipeline.detection.scientific_executor import (
    materialize_original_test,
    seal,
    verify_execution,
    verify_test_view,
)
from flir_pipeline.similarity.storage import (
    file_sha256,
    read_json,
    stable_id,
    write_json,
)


def release(freeze: Path, training: Path, output: Path) -> dict:
    ctx = frozen_context(freeze)
    executions = completed_matrix(ctx, training)
    gate = verify_admission(bound_file(ctx["freeze"]["gate"]))
    publication = bound_file(gate["publication"]).parent
    external_output(output, freeze, training, publication)
    return emit(
        output,
        {
            "policy": POLICY,
            "kind": "evaluation_release",
            "freeze": binding(freeze),
            "training_root": str(training.resolve()),
            "executions": {
                k: binding(v / "execution.json") for k, v in executions.items()
            },
            "test_sources": {
                "records": binding(
                    publication / "sources/proposal/test_records.parquet"
                ),
                "annotations": binding(
                    publication / "sources/audit/annotation_audit.json"
                ),
                "source_root": gate["source_root"],
            },
        },
    )


def evaluation_context(path: Path) -> tuple[dict, dict]:
    permission = receipt(path, "evaluation_release")
    ctx = frozen_context(bound_file(permission["freeze"]))
    completed = completed_matrix(ctx, Path(permission["training_root"]))
    if permission["executions"] != {
        k: binding(v / "execution.json") for k, v in completed.items()
    }:
        raise ValueError("Released matrix differs from completed executions")
    # Shared materializer binds this technical release rather than a legacy signature.
    permission["release_id"] = file_sha256(path)
    return permission, ctx


def prepare_test(release_path: Path, output: Path) -> dict:
    permission, ctx = evaluation_context(release_path)
    return materialize_original_test(permission, ctx, output, release_path)


def verify_result(path: Path, ctx: dict, run: dict, training: Path, test: Path) -> dict:
    state = verify_execution(
        path,
        {
            "state": "completed",
            "run": run,
            "admission_sha256": ctx["admission_sha256"],
            "training_execution_sha256": file_sha256(training / "execution.json"),
            "sealed_test_sha256": file_sha256(test / "execution.json"),
            "checkpoint_selection": "validation_only",
            "test_tuning": False,
        },
    )
    required = {
        "test_image_stats.npz",
        "metrics.json",
        "metrics_per_class.parquet",
        "metrics_per_image_class.parquet",
        "predictions.parquet",
        "aliases.json",
    }
    if not required <= set(state["files"]):
        raise ValueError("Evaluation exports incomplete")
    images = load_image_stats(path / "test_image_stats.npz")
    expected = pd.read_parquet(test / "view/records.parquet").set_index("frame_id")
    if [s.frame_id for s in images] != sorted(expected.index) or any(
        len(s.target_class) != expected.loc[s.frame_id, "num_objects"] for s in images
    ):
        raise ValueError("Evaluation did not cover the exact common test annotations")
    overall, classes = evaluate(images)
    if read_json(path / "metrics.json")["overall"] != overall:
        raise ValueError("Exported metrics differ from captured predictions")
    classes["class_name"] = classes.class_id.map(NAMES)
    actual = pd.read_parquet(path / "metrics_per_class.parquet")
    pd.testing.assert_frame_equal(actual[classes.columns], classes)
    per_image = pd.read_parquet(path / "metrics_per_image_class.parquet")
    pd.testing.assert_frame_equal(
        per_image, pd.DataFrame(image_metric_rows(run, images))
    )
    predictions = pd.read_parquet(path / "predictions.parquet")
    if (
        not set(predictions.frame_id) <= set(expected.index)
        or not predictions.run_id.eq(run["run_id"]).all()
        or not predictions.detector_seed.eq(run["detector_seed"]).all()
    ):
        raise ValueError("Prediction identity differs from evaluation")
    for sample in images:
        detections = predictions.loc[predictions.frame_id == sample.frame_id]
        if (
            len(detections) != len(sample.confidence)
            or not np.array_equal(
                detections.class_id.to_numpy(), sample.predicted_class
            )
            or not np.allclose(
                detections.confidence.to_numpy(dtype=float),
                sample.confidence,
                rtol=0,
                atol=1e-7,
            )
        ):
            raise ValueError(
                "Prediction export differs from captured matching statistics"
            )
        boxes = detections[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)
        if not np.isfinite(boxes).all() or np.any(boxes[:, 2:] < boxes[:, :2]):
            raise ValueError("Invalid exported prediction boxes")
    if read_json(path / "aliases.json") != run:
        raise ValueError("Evaluation aliases changed")
    return state


def evaluate_run(release_path: Path, test: Path, output: Path, index: int = 0) -> dict:
    permission, ctx = evaluation_context(release_path)
    verify_execution(
        test,
        {
            "state": "sealed_test",
            "admission_sha256": ctx["admission_sha256"],
            "evaluation_sha256": file_sha256(release_path),
        },
    )
    verify_test_view(test, permission, ctx)
    if not 0 <= index < len(ctx["runs"]):
        raise ValueError("Evaluation index outside frozen matrix")
    run = ctx["runs"][index]
    training = bound_file(permission["executions"][run["run_id"]]).parent
    external_output(output, test, training, release_path, ctx["prepared"])
    root = output / run["run_id"]
    with worker_lock(root):
        destination = root / "evaluation"
        if destination.exists():
            return verify_result(destination, ctx, run, training, test)
        import torch  # noqa: F401
        from ultralytics import settings  # noqa: F401

        from flir_pipeline.detection.scientific_runtime import run_evaluation

        run_evaluation(ctx, run, training, test, destination)
        return verify_result(destination, ctx, run, training, test)


def detailed_validator(base):
    """Capture the exact post-NMS boxes passed to the existing pinned matcher."""

    class DetailedValidator(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.prediction_rows = []

        def update_metrics(self, preds, batch):
            for index, prediction in enumerate(preds):
                pred = self._prepare_pred(prediction)
                for detection, (box, confidence, cls) in enumerate(
                    zip(
                        pred["bboxes"].cpu().tolist(),
                        pred["conf"].cpu().tolist(),
                        pred["cls"].cpu().tolist(),
                        strict=True,
                    )
                ):
                    self.prediction_rows.append(
                        {
                            "frame_id": Path(batch["im_file"][index]).stem,
                            "detection": detection,
                            "class_id": int(cls),
                            "confidence": confidence,
                            "x1": box[0],
                            "y1": box[1],
                            "x2": box[2],
                            "y2": box[3],
                            "coordinates": "validator_input_pixels_xyxy",
                            "input_height": int(batch["img"].shape[2]),
                            "input_width": int(batch["img"].shape[3]),
                        }
                    )
            super().update_metrics(preds, batch)

    return DetailedValidator


def image_metric_rows(run: dict, images: list) -> list:
    rows = []
    for sample in images:
        _, table = evaluate([sample])
        for row in table.to_dict("records"):
            tp = row["true_positives_at_fixed_confidence"]
            rows.append(
                {
                    **row,
                    "class_name": NAMES[row["class_id"]],
                    "frame_id": sample.frame_id,
                    "tp": tp,
                    "fp": row["predictions_at_fixed_confidence"] - tp,
                    "fn": row["support"] - tp,
                    "run_id": run["run_id"],
                    "detector_seed": run["detector_seed"],
                }
            )
    return rows


def export_details(output: Path, run: dict, images: list, predictions: list) -> None:
    pd.DataFrame(image_metric_rows(run, images)).to_parquet(
        output / "metrics_per_image_class.parquet", index=False
    )
    classes = pd.read_parquet(output / "metrics_per_class.parquet")
    classes["tp"] = classes.true_positives_at_fixed_confidence
    classes["fp"] = classes.predictions_at_fixed_confidence - classes.tp
    classes["fn"] = classes.support - classes.tp
    classes.to_parquet(output / "metrics_per_class.parquet", index=False)
    columns = [
        "frame_id",
        "detection",
        "class_id",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "coordinates",
        "input_height",
        "input_width",
    ]
    table = pd.DataFrame(predictions, columns=columns)
    table["run_id"], table["detector_seed"] = run["run_id"], run["detector_seed"]
    table.to_parquet(output / "predictions.parquet", index=False)
    write_json(output / "aliases.json", run)


def paired_difference(
    left: list,
    right: list,
    units: list[str],
    resamples: int,
    seed: int,
    confidence: float,
) -> dict:
    if (
        [s.frame_id for s in left] != [s.frame_id for s in right]
        or len(units) != len(left)
        or resamples < 1
        or not 0 < confidence < 1
    ):
        raise ValueError(
            "Paired comparison requires identical images and valid resampling"
        )
    for a, b in zip(left, right, strict=True):
        if not np.array_equal(a.target_class, b.target_class):
            raise ValueError("Paired annotations differ")
    groups = [np.flatnonzero(np.asarray(units) == unit) for unit in sorted(set(units))]
    rng = np.random.default_rng(seed)
    samples = {metric: [] for metric in METRICS}
    a, _ = evaluate(left)
    b, _ = evaluate(right)
    for _ in range(resamples):
        indices = np.concatenate(
            [groups[i] for i in rng.integers(0, len(groups), len(groups))]
        )
        x, _ = evaluate(left, indices)
        y, _ = evaluate(right, indices)
        for metric in METRICS:
            if x[metric] is not None and y[metric] is not None:
                samples[metric].append(y[metric] - x[metric])
    alpha = (1 - confidence) / 2
    return {
        m: {
            "difference_right_minus_left": b[m] - a[m]
            if a[m] is not None and b[m] is not None
            else None,
            "lower": float(np.quantile(v, alpha)) if v else None,
            "upper": float(np.quantile(v, 1 - alpha)) if v else None,
            "valid_resamples": len(v),
        }
        for m, v in samples.items()
    }


def compare(release_path: Path, test: Path, evaluations: Path, output: Path) -> dict:
    permission, ctx = evaluation_context(release_path)
    verify_execution(
        test,
        {
            "state": "sealed_test",
            "admission_sha256": ctx["admission_sha256"],
            "evaluation_sha256": file_sha256(release_path),
        },
    )
    verify_test_view(test, permission, ctx)
    external_output(output, release_path, test, evaluations)
    if output.exists():
        raise ValueError("Comparison exists")
    cohorts, cells, bindings = {}, [], {}
    for run in ctx["runs"]:
        path = evaluations / run["run_id"] / "evaluation"
        verify_result(
            path,
            ctx,
            run,
            bound_file(permission["executions"][run["run_id"]]).parent,
            test,
        )
        cohorts[run["run_id"]] = load_image_stats(path / "test_image_stats.npz")
        bindings[run["run_id"]] = binding(path / "execution.json")
        overall = read_json(path / "metrics.json")["overall"]
        cells.extend(
            {
                **alias,
                "run_id": run["run_id"],
                "detector_seed": run["detector_seed"],
                **{m: overall[m] for m in METRICS},
            }
            for alias in run["aliases"]
        )
    expected = pd.read_parquet(test / "view/records.parquet").sort_values("frame_id")
    units = expected.content_id.tolist()
    evaluation = ctx["config"]["evaluation"]
    pairs, cached = [], {}
    for left, right in combinations(cells, 2):
        same_seed = all(left[k] == right[k] for k in ("split_seed", "detector_seed"))
        within = (
            left["population"] == right["population"]
            and left["policy"] != right["policy"]
        )
        composition = (
            left["population"] != right["population"]
            and left["policy"] == right["policy"]
        )
        if not same_seed or not (within or composition):
            continue
        key = (left["run_id"], right["run_id"])
        if key not in cached:
            cached[key] = paired_difference(
                cohorts[key[0]],
                cohorts[key[1]],
                units,
                evaluation["bootstrap_resamples"],
                evaluation["bootstrap_seed"],
                evaluation["confidence_level"],
            )
        pairs.append(
            {
                "left": left,
                "right": right,
                "same_execution": key[0] == key[1],
                "contrast": "partition_strategy"
                if within
                else "population_composition",
                "metrics": cached[key],
            }
        )
    output.mkdir(parents=True)
    table = pd.DataFrame(cells)
    table.to_parquet(output / "experimental_cells.parquet", index=False)
    for keys, name in [
        (["population", "policy", "split_seed"], "detector_seed_variation"),
        (["population", "policy", "detector_seed"], "split_seed_variation"),
    ]:
        table.groupby(keys)[list(METRICS)].agg(["count", "mean", "std"]).to_parquet(
            output / f"{name}.parquet"
        )
    write_json(
        output / "paired_comparisons.json",
        {
            "pairs": pairs,
            "resampling_unit": "known exact content_id",
            "units": len(set(units)),
            "seed": evaluation["bootstrap_seed"],
            "resamples": evaluation["bootstrap_resamples"],
            "limitation": "Conditional descriptive intervals: unknown temporal/visual dependence remains. Aliases are not independent repetitions; no causal leakage effect identified.",
        },
    )
    return seal(
        output,
        {
            "state": "completed",
            "mode": "paired_descriptive_comparison",
            "release_sha256": file_sha256(release_path),
            "evaluations": bindings,
            "comparison_id": stable_id(
                {"release": file_sha256(release_path), "evaluations": bindings}
            ),
        },
    )
