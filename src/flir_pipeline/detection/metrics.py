"""Fixed-threshold P/R, AP and image bootstrap from independently matched images.

AP is recomputed over pooled detections for every resample, never averaged over
per-image AP. Matching is performed by the pinned Ultralytics validator. P/R
uses a separate match after the fixed confidence filter to avoid low-confidence
boxes stealing matches. No threshold is selected from test outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from flir_pipeline.data.classes import class_name
from flir_pipeline.detection.protocol import METRICS


@dataclass
class ImageStats:
    frame_id: str
    confidence: np.ndarray
    predicted_class: np.ndarray
    true_positive: np.ndarray
    target_class: np.ndarray
    fixed_predictions: np.ndarray  # counts per class at the prespecified confidence
    fixed_true_positives: np.ndarray

    def validate(self) -> None:
        n = len(self.confidence)
        if self.predicted_class.shape != (n,) or self.true_positive.shape != (n, 10):
            raise ValueError("Detection statistics shape mismatch")
        if not np.isfinite(self.confidence).all() or np.any((self.confidence < 0) | (self.confidence > 1)):
            raise ValueError("Invalid detection confidence")
        for values in (self.predicted_class, self.target_class):
            if values.ndim != 1 or not np.isin(values, np.arange(5)).all():
                raise ValueError("Noncanonical class in metrics")
        targets = np.bincount(self.target_class.astype(int), minlength=5)
        if self.fixed_predictions.shape != (5,) or self.fixed_true_positives.shape != (5,):
            raise ValueError("Fixed-threshold counts must cover five classes")
        if np.any(self.fixed_true_positives > np.minimum(self.fixed_predictions, targets)) or np.any(self.fixed_predictions < 0) or np.any(self.fixed_true_positives < 0):
            raise ValueError("Invalid fixed-threshold matching counts")
        if not np.isin(self.true_positive, (0, 1)).all():
            raise ValueError("Invalid TP matrix")
        for c in range(5):
            if np.any(self.true_positive[self.predicted_class == c].sum(axis=0) > targets[c]):
                raise ValueError("More matched detections than ground-truth boxes")


def average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    """101-point interpolated PR integral, including boundary sentinels (Ultralytics)."""
    r = np.concatenate(([0.0], recall, [1.0]))
    p = np.concatenate(([1.0], precision, [0.0]))
    envelope = np.maximum.accumulate(p[::-1])[::-1]
    x = np.linspace(0, 1, 101)
    integral = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(integral(np.interp(x, r, envelope), x))


def evaluate(images: list[ImageStats], indices: np.ndarray | None = None) -> tuple[dict, pd.DataFrame]:
    if not images:
        raise ValueError("Cannot evaluate an empty image cohort")
    selected = images if indices is None else [images[int(i)] for i in indices]
    confidence = np.concatenate([s.confidence for s in selected])
    predicted = np.concatenate([s.predicted_class for s in selected]).astype(int)
    tp = np.concatenate([s.true_positive for s in selected]).astype(bool)
    targets = np.bincount(np.concatenate([s.target_class for s in selected]).astype(int), minlength=5)
    fixed_pred = np.sum([s.fixed_predictions for s in selected], axis=0)
    fixed_tp = np.sum([s.fixed_true_positives for s in selected], axis=0)
    order = np.argsort(-confidence, kind="stable")
    predicted, tp = predicted[order], tp[order]
    rows = []
    for c in range(5):
        support = int(targets[c])
        row = {"class_id": c, "class_name": class_name(c), "support": support,
               "predictions_at_fixed_confidence": int(fixed_pred[c]), "true_positives_at_fixed_confidence": int(fixed_tp[c])}
        if support == 0:
            row.update(dict.fromkeys(METRICS, None))
        else:
            class_tp = tp[predicted == c]
            if len(class_tp):
                tpc = class_tp.cumsum(axis=0)
                fpc = (~class_tp).cumsum(axis=0)
                precision = tpc/(tpc+fpc)
                recall = tpc/support
                ap = [average_precision(recall[:, j], precision[:, j]) for j in range(10)]
            else:
                ap = [0.0]*10
            row.update(precision=float(fixed_tp[c]/fixed_pred[c]) if fixed_pred[c] else 0.0,
                       recall=float(fixed_tp[c]/support), map50=float(ap[0]), map50_95=float(np.mean(ap)))
        rows.append(row)
    table = pd.DataFrame(rows)
    # Fixed five-class macro: undefined if a class is absent, not an easier adaptive macro.
    overall = {m: float(table[m].mean()) if (table.support > 0).all() else None for m in METRICS}
    overall.update(image_count=len(selected), instance_count=int(targets.sum()), supported_classes=int((targets > 0).sum()),
                   averaging="fixed_five_class_macro", no_prediction_precision_convention=0.0)
    return overall, table


def save_image_stats(images: list[ImageStats], path: Path) -> None:
    if len({s.frame_id for s in images}) != len(images):
        raise ValueError("Duplicate image identity in test evaluation")
    data = {"frame_ids": np.asarray([s.frame_id for s in images], dtype=str)}
    for i, sample in enumerate(images):
        sample.validate()
        for key in ("confidence", "predicted_class", "true_positive", "target_class", "fixed_predictions", "fixed_true_positives"):
            data[f"{i}_{key}"] = getattr(sample, key)
    np.savez_compressed(path, **data)


def load_image_stats(path: Path) -> list[ImageStats]:
    with np.load(path, allow_pickle=False) as data:
        result = [ImageStats(str(name), **{key: data[f"{i}_{key}"] for key in ("confidence", "predicted_class", "true_positive", "target_class", "fixed_predictions", "fixed_true_positives")}) for i, name in enumerate(data["frame_ids"])]
    for sample in result:
        sample.validate()
    return result


def bootstrap(images: list[ImageStats], resamples: int, seed: int, confidence_level: float = .95) -> tuple[dict, pd.DataFrame]:
    if resamples < 1 or not 0 < confidence_level < 1:
        raise ValueError("Invalid bootstrap settings")
    rng = np.random.default_rng(seed)
    rows = []
    for replicate in range(resamples):
        overall, classes = evaluate(images, rng.integers(0, len(images), len(images)))
        rows.append({"replicate": replicate, "class_id": -1, **{m: overall[m] for m in METRICS}})
        rows.extend({"replicate": replicate, "class_id": int(r.class_id), **{m: getattr(r, m) for m in METRICS}} for r in classes.itertuples())
    samples = pd.DataFrame(rows)
    alpha = (1-confidence_level)/2
    intervals = []
    for cid, group in samples.groupby("class_id", sort=True):
        for metric in METRICS:
            values = group[metric].dropna()
            intervals.append({"class_id": int(cid), "metric": metric, "valid_resamples": len(values),
                              "lower": float(values.quantile(alpha)) if len(values) else None,
                              "upper": float(values.quantile(1-alpha)) if len(values) else None})
    return {"resamples": resamples, "seed": seed, "confidence_level": confidence_level, "method": "image_percentile_bootstrap",
            "limitation": "Images are resampled independently; residual inter-frame dependence is not corrected. Unsupported classes produce undefined values, counted explicitly.",
            "intervals": intervals}, samples


def aggregate_runs(runs: pd.DataFrame) -> dict[str, pd.DataFrame]:
    keys = ["strategy", "split_seed", "detector_seed", "class_id"]
    if runs.duplicated(keys).any():
        raise ValueError("Duplicate experimental cell; pilot cannot count twice")
    long = runs.melt(id_vars=keys, value_vars=list(METRICS), var_name="metric", value_name="value")
    training = long.groupby(["strategy", "split_seed", "class_id", "metric"]).value.agg(["count", "mean", "std", "median", "min", "max"]).reset_index()
    split = training.groupby(["strategy", "class_id", "metric"])["mean"].agg(["count", "mean", "std", "median", "min", "max"]).reset_index()
    effects = split.merge(split.loc[split.strategy == "historical", ["class_id", "metric", "mean"]].rename(columns={"mean": "historical_mean"}), on=["class_id", "metric"], how="left")
    effects["difference_from_historical"] = effects["mean"]-effects.historical_mean
    effects["interpretation"] = "descriptive difference; different test composition; not a causal leakage estimate"
    return {"training_seed_summary": training, "split_seed_summary": split, "effect_sizes": effects, "hierarchical_metrics": long}
