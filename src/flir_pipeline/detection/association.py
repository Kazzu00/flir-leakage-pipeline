"""Read-only, evidence-gated joins between detector runs and frozen split context.

The registry is independent of the frozen training plan. No similarity, threshold,
candidate or detector metric is estimated here. A partial matrix exposes progress
and residual context only; scientific rows require the entire controlled matrix.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from flir_pipeline.data.classes import class_name
from flir_pipeline.detection.metrics import aggregate_runs
from flir_pipeline.detection.protocol import (
    METRICS,
    detector_run_id,
    experiment_matrix,
    verify_plan,
)
from flir_pipeline.detection.runtime import fair_comparison, verify_run
from flir_pipeline.similarity.storage import file_sha256, read_json, stable_id

PENDING = "PENDING"
PARTIAL = "PARTIAL"
COMPLETE = "COMPLETE_CONTROLLED_COMPARISON"
STRATEGIES = ("historical", "random_content", "C10", "C12")
STRATEGY_LABELS = {
    "historical": "Historical",
    "random_content": "Random content-level",
    "C10": "C10 — DINOv2 / PaCMAP / DBSCAN",
    "C12": "C12 — DINOv2 / t-SNE / HDBSCAN",
}
COLORS = ("#7d8597", "#e0a526", "#137c8b", "#6943a5")
MARKERS = ("o", "s", "^", "D")
POPULATIONS = {-1: "Overall", **{i: class_name(i) for i in range(5)}}
RESIDUALS = (
    "exact_duplicate_cross_split_count", "dinov2_nn_mean", "clip_nn_mean",
    "dinov2_top001_pairs", "clip_top001_pairs", "temporal_at5", "class_deviation_pp",
)
FRACTIONS = ("dinov2_top001_fraction", "clip_top001_fraction")
RESIDUAL_LABELS = {
    "exact_duplicate_cross_split_count": "Exact duplicate cross-split count",
    "dinov2_nn_mean": "DINOv2 NN mean", "clip_nn_mean": "CLIP NN mean",
    "dinov2_top001_pairs": "DINOv2 top-0.1% (pairs)",
    "clip_top001_pairs": "CLIP top-0.1% (pairs)",
    "temporal_at5": "Temporal Δ≤5 (fraction)", "class_deviation_pp": "Class deviation (pp)",
    "dinov2_top001_fraction": "DINOv2 top-0.1% (fraction)",
    "clip_top001_fraction": "CLIP top-0.1% (fraction)",
}
METRIC_LABELS = {"map50_95": "mAP@50–95", "map50": "mAP@50", "precision": "Precision", "recall": "Recall"}
IDENTIFIERS = ("detector_run_id", "strategy", "split_seed", "detector_seed", "split_space_id")
ASSOCIATION_COLUMNS = (*IDENTIFIERS, "class_id", "class_name", "support", *METRICS, *RESIDUALS, *FRACTIONS)
NOTICE = (
    "Asociación ≠ causalidad. Las estrategias contienen distintos ejemplos en test; "
    "su composición y dificultad también cambian. La similitud residual no es una "
    "variable experimental aislada. Cada punto representa un run, no una observación "
    "independiente: las detector seeds de un mismo split comparten su contexto residual."
)
DEFAULT_CONFIG = Path("configs/detection/associations.yaml")


def load_registry(path: Path = DEFAULT_CONFIG) -> pd.DataFrame:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config["protocol"] != "detector_residual_associations_v1" or config["policy"] != "descriptive_association_not_causality":
        raise ValueError("Unsupported association policy")
    table = pd.DataFrame(config["associations"])
    required = {"association_id", "category", "detector_metric", "residual_metric", "population", "class_id"}
    if not required <= set(table) or table.empty or table.association_id.duplicated().any():
        raise ValueError("Invalid or duplicate association specification")
    if table.duplicated(["class_id", "detector_metric", "residual_metric"]).any():
        raise ValueError("Duplicate prespecified view")
    for row in table.itertuples():
        if row.detector_metric not in METRICS or row.residual_metric not in RESIDUAL_LABELS or row.class_id not in POPULATIONS:
            raise ValueError("Unsupported metric or population in registry")
        if row.population != ("overall" if row.class_id == -1 else "class"):
            raise ValueError("Association population/class mismatch")
    table["class"] = table.class_id.map(POPULATIONS)
    return table


def view_label(registry: pd.DataFrame, detector_metric: str, residual_metric: str, class_id: int) -> str:
    found = registry.detector_metric.eq(detector_metric) & registry.residual_metric.eq(residual_metric) & registry.class_id.eq(class_id)
    return "PRESPECIFIED ASSOCIATION" if found.any() else "EXPLORATORY VIEW"


def is_small_pilot(meta: dict) -> bool:
    return (meta.get("scientific_result") is False or "SMALL_PILOT" in str(meta.get("state", ""))
            or meta.get("identity", {}).get("model_configuration", {}).get("purpose") == "small_infrastructure_pilot")


def validate_binding(meta: dict, split: dict, plan: dict, freeze: dict) -> None:
    """Check duplicated metadata fields as well as the hashed mathematical identity."""
    if is_small_pilot(meta):
        raise ValueError("Small pilots cannot enter scientific association tables")
    identity = meta["identity"]
    if (meta["plan_id"] != plan["plan_id"] or identity["split_space_id"] != split["split_space_id"]
            or meta.get("split_space_id", split["split_space_id"]) != split["split_space_id"]):
        raise ValueError("Run/plan/split_space_id mismatch")
    if (meta["strategy"] != split["strategy"] or meta["split_seed"] != split["split_seed"]
            or meta["dataset_id"] != split["dataset_id"] or meta["context"] != split["context"]):
        raise ValueError("Run labels or residual context differ from the frozen split")
    if (identity["model_configuration"] != freeze["model_config"] or meta["detector_run_id"] != stable_id(identity)
            or meta["detector_seed"] != identity["training_seed"]
            or meta["detector_seed"] not in plan["identity"]["config"]["training_seeds"]):
        raise ValueError("Run identity, runtime or detector seed mismatch")
    if meta["state"] != "COMPLETE" or meta["test_tuning"] is not False:
        raise ValueError("Run is incomplete or test-tuned")


def frozen_context(plan: dict, split_root: Path | None = None) -> pd.DataFrame:
    """Optional fractions come from checksum-bound stored cohorts, never thresholds.

The frozen plan's seven residual fields remain authoritative. Missing optional
source artifacts leave fractions undefined; a present but changed source fails.
"""
    rows = []
    for split in plan["identity"]["splits"]:
        if not set(RESIDUALS) <= split["context"].keys():
            raise ValueError("Frozen residual context schema is incomplete")
        row = {k: split[k] for k in ("strategy", "split_seed", "split_space_id")}
        row.update({k: split["context"].get(k) for k in (*RESIDUALS, *FRACTIONS)})
        directory = split_root/split["split_space_id"] if split_root is not None else None
        if directory is not None and directory.exists():
            meta_path = directory/"metadata.json"
            if file_sha256(meta_path) != split["split_metadata_sha256"]:
                raise ValueError("Optional fraction source differs from frozen split metadata")
            meta = read_json(meta_path)
            if meta["split_space_id"] != split["split_space_id"]:
                raise ValueError("Fraction source split identity mismatch")
            for encoder in ("dinov2", "clip"):
                name = f"quantile_pairs_{encoder}.parquet"
                if name not in meta["output_sha256"]:
                    continue
                if file_sha256(directory/name) != meta["output_sha256"][name]:
                    raise ValueError("Stored quantile cohort checksum mismatch")
                cohort = pd.read_parquet(directory/name)
                cohort = cohort.loc[cohort["quantile"].eq(.999)]
                if len(cohort) != 1 or cohort.cross_split_count.iloc[0] != row[f"{encoder}_top001_pairs"]:
                    raise ValueError("Stored top-0.1% cohort differs from frozen count")
                fraction = cohort.cross_split_fraction.iloc[0]
                if pd.notna(fraction) and (not np.isfinite(fraction) or not 0 <= fraction <= 1):
                    raise ValueError("Invalid stored cohort fraction")
                existing = row[f"{encoder}_top001_fraction"]
                if existing is not None and existing != fraction:
                    raise ValueError("Stored fraction differs from frozen context")
                row[f"{encoder}_top001_fraction"] = fraction
        rows.append(row)
    context = pd.DataFrame(rows)
    if context.empty or context.split_space_id.duplicated().any() or context.duplicated(["strategy", "split_seed"]).any():
        raise ValueError("Duplicate or empty frozen split context")
    return context


def join_run(meta: dict, overall: dict, classes: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    """One stored metric row per run × overall/class; undefined values stay missing."""
    if is_small_pilot(meta):
        raise ValueError("Small pilots cannot enter scientific association tables")
    split_id = meta["identity"]["split_space_id"]
    selected = context.loc[context.split_space_id.eq(split_id)]
    if len(selected) != 1:
        raise ValueError("Run must join exactly one frozen split_space_id")
    source = selected.iloc[0]
    if meta["strategy"] != source.strategy or meta["split_seed"] != source.split_seed:
        raise ValueError("Run/split labels mismatch")
    if classes.class_id.tolist() != list(range(5)) or classes.class_name.tolist() != [POPULATIONS[i] for i in range(5)]:
        raise ValueError("Expected canonical class rows exactly once")
    values = pd.concat([pd.DataFrame([{"class_id": -1, "class_name": "Overall", "support": overall["instance_count"],
                                      **{metric: overall.get(metric) for metric in METRICS}}]), classes], ignore_index=True)
    for metric in METRICS:
        if metric not in values:
            values[metric] = np.nan
        values[metric] = pd.to_numeric(values[metric], errors="raise")
        present = values[metric].dropna()
        if not np.isfinite(present).all() or not present.between(0, 1).all():
            raise ValueError("Invalid stored detector metric")
    unsupported = values.support.eq(0)
    if values.loc[unsupported, list(METRICS)].notna().any().any():
        raise ValueError("Unsupported class must have undefined metrics, not zero")
    if overall.get("supported_classes", 5) != 5 and values.loc[values.class_id.eq(-1), list(METRICS)].notna().any().any():
        raise ValueError("Fixed five-class macro must remain undefined without full support")
    for name in ("detector_run_id", "strategy", "split_seed", "detector_seed"):
        values[name] = meta[name]
    values["split_space_id"] = split_id
    for name in (*RESIDUALS, *FRACTIONS):
        values[name] = source[name]
    return values


@dataclass
class AssociationEvidence:
    state: str
    matrix: pd.DataFrame
    context: pd.DataFrame
    associations: pd.DataFrame
    fairness: dict
    issues: list[str]
    excluded_pilots: int = 0

    @property
    def complete(self) -> bool:
        return self.state == COMPLETE and self.fairness["controlled"]


def load_evidence(plan_directory: Path, artifacts: Path, *, split_root: Path | None = None) -> AssociationEvidence:
    """Discover local run states, verify complete runs and close the scientific gate.

RUNNING is inferred from stored progress, not proof that a process is alive.
Invalid and duplicate artifacts block the complete gate, even with 48 valid cells.
No partial metric table is returned or made available to plotting consumers.
"""
    plan = verify_plan(plan_directory)
    splits = plan["identity"]["splits"]
    context = frozen_context(plan, split_root)
    expected = experiment_matrix(splits, plan["identity"]["config"]["training_seeds"])
    matrix = expected[["strategy", "split_seed", "split_space_id", "detector_seed"]].copy()
    matrix["detector_run_id"] = None
    matrix["status"] = PENDING
    matrix["detail"] = "No controlled run artifact"
    issues, metas, frames = [], [], []
    excluded = len(list((artifacts/"small_pilot").glob("*/pilot.json")))
    freeze_path = plan_directory/"runtime_freeze.json"
    if not freeze_path.exists():
        issues.append("runtime_not_frozen")
        return AssociationEvidence(PENDING, matrix, context, pd.DataFrame(columns=ASSOCIATION_COLUMNS),
                                   {"controlled": False, "completed_runs": 0, "expected_runs": len(matrix), "errors": issues}, issues, excluded)
    freeze = read_json(freeze_path)
    if freeze["plan_id"] != plan["plan_id"] or stable_id(freeze["model_config"]) != freeze["model_config_id"]:
        raise ValueError("Runtime freeze identity mismatch")
    expected_config = copy.deepcopy(plan["identity"]["config"])
    frozen_config = freeze["model_config"]["protocol_config"]
    expected_config["train"]["batch"] = frozen_config["train"]["batch"]
    if frozen_config != expected_config:
        raise ValueError("Frozen training protocol differs from plan")
    matrix["detector_run_id"] = [detector_run_id(r.split_space_id, freeze["model_config"], r.detector_seed) for r in matrix.itertuples()]
    cells = {(r.split_space_id, r.detector_seed): r.Index for r in matrix.itertuples()}
    lookup = {s["split_space_id"]: s for s in splits}
    found: dict[tuple, list[tuple[Path, dict, bool]]] = {}
    for directory in sorted((artifacts/"runs").glob("*")):
        if not directory.is_dir():
            continue
        meta_path, state_path = directory/"metadata.json", directory/"state.json"
        try:
            if not meta_path.exists() and not state_path.exists():
                raise ValueError("Run directory lacks metadata and progress state")
            finished = meta_path.exists()
            record = read_json(meta_path if finished else state_path)
            if is_small_pilot(record) or (directory/"pilot.json").exists():
                excluded += 1
                continue
            if finished and record.get("plan_id") != plan["plan_id"]:
                raise ValueError("Run belongs to a different or missing plan")
            identity = record["identity"]
            key = (identity["split_space_id"], identity["training_seed"])
            if key not in cells:
                raise ValueError("Unexpected run split_space_id or detector seed")
            found.setdefault(key, []).append((directory, record, finished))
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            issues.append(f"{directory.name}: {exc}")
            mask = matrix.detector_run_id.eq(directory.name)
            matrix.loc[mask, ["status", "detail"]] = ["INVALID", str(exc)]
    for key, records in found.items():
        idx = cells[key]
        try:
            if len(records) != 1:
                raise ValueError("Duplicate detector run / experimental cell")
            directory, meta, finished = records[0]
            if directory.name != matrix.loc[idx, "detector_run_id"] or meta["identity"]["model_configuration"] != freeze["model_config"]:
                raise ValueError("Run directory or model identity mismatch")
            if not finished:
                if meta["state"] not in {"INITIALIZED", "TRAINING", "TRAINED"}:
                    raise ValueError("Invalid progress state or missing completion metadata")
                matrix.loc[idx, ["status", "detail"]] = ["RUNNING", f"Stored {meta['state']}; process liveness is unknown"]
                continue
            split = lookup[key[0]]
            validate_binding(meta, split, plan, freeze)
            verified = verify_run(directory, split, freeze)
            validate_binding(verified, split, plan, freeze)
            metrics = read_json(directory/"metrics.json")
            bootstrap = read_json(directory/"bootstrap.json")
            evaluation = frozen_config["evaluation"]
            if metrics.get("definition") != evaluation:
                raise ValueError("Stored detector metric definition differs from freeze")
            if (bootstrap.get("method") != "image_percentile_bootstrap"
                    or any(bootstrap.get(k) != evaluation[v] for k, v in
                           (("resamples", "bootstrap_resamples"), ("seed", "bootstrap_seed"), ("confidence_level", "confidence_level")))):
                raise ValueError("Stored bootstrap protocol differs from freeze")
            rows = join_run(verified, metrics["overall"],
                            pd.read_parquet(directory/"metrics_per_class.parquet"), context)
            intervals = pd.DataFrame(bootstrap["intervals"])
            if not intervals.empty:
                for metric in METRICS:
                    part = intervals.loc[intervals.metric.eq(metric)].set_index("class_id")
                    for name in ("lower", "upper", "valid_resamples"):
                        rows[f"{metric}_ci_{name}"] = rows.class_id.map(part[name])
            frames.append(rows)
            metas.append(verified)
            matrix.loc[idx, ["status", "detail"]] = ["COMPLETE", "Source-bound run verification passed"]
        except (OSError, ValueError, KeyError, TypeError, AssertionError, AttributeError) as exc:
            matrix.loc[idx, ["status", "detail"]] = ["INVALID", str(exc)]
            issues.append(f"{matrix.loc[idx, 'detector_run_id']}: {exc}")
    fairness = fair_comparison(metas, expected)
    fairness["errors"] = [*fairness["errors"], *issues]
    fairness["controlled"] = fairness["controlled"] and not issues
    complete = fairness["controlled"] and matrix.status.eq("COMPLETE").all()
    state = COMPLETE if complete else (PARTIAL if matrix.status.ne(PENDING).any() or issues else PENDING)
    scientific = pd.concat(frames, ignore_index=True) if complete else pd.DataFrame(columns=ASSOCIATION_COLUMNS)
    if not scientific.empty and scientific.duplicated(["detector_run_id", "class_id"]).any():
        raise ValueError("Duplicate scientific run/class row")
    return AssociationEvidence(state, matrix, context, scientific, fairness, issues, excluded)


def select_view(evidence: AssociationEvidence, detector_metric: str, residual_metric: str, class_id: int,
                strategy: str | None = None) -> pd.DataFrame:
    if detector_metric not in METRICS or residual_metric not in RESIDUAL_LABELS or class_id not in POPULATIONS:
        raise ValueError("Unknown association view")
    if strategy is not None and strategy not in STRATEGIES:
        raise ValueError("Unknown strategy")
    if not evidence.complete:
        return pd.DataFrame(columns=[*IDENTIFIERS, "class_id", "class_name", "x", "y"])
    selected = evidence.associations.loc[evidence.associations.class_id.eq(class_id)].copy()
    if strategy is not None:
        selected = selected.loc[selected.strategy.eq(strategy)]
    return selected.assign(x=selected[residual_metric], y=selected[detector_metric])


def prespecified_table(evidence: AssociationEvidence, registry: pd.DataFrame) -> pd.DataFrame:
    table = registry.copy()
    table["status"] = evidence.state
    table["n_runs"] = [len(select_view(evidence, r.detector_metric, r.residual_metric, r.class_id).dropna(subset=["x", "y"]))
                       for r in registry.itertuples()]
    return table


def strategy_summaries(evidence: AssociationEvidence) -> dict[str, pd.DataFrame]:
    if not evidence.complete:
        return {}
    overall = evidence.associations.loc[evidence.associations.class_id.eq(-1)]
    summary = overall.groupby("strategy", sort=False).map50_95.agg(n_metric_values="count", mean="mean", median="median", std="std")
    summary["n_detector_runs"] = overall.groupby("strategy").detector_run_id.nunique()
    residual = evidence.context.groupby("strategy")[list(RESIDUALS)].mean().add_prefix("mean_")
    summary = summary.join(residual).reindex(STRATEGIES).reset_index()
    return {"strategy_summary": summary, **aggregate_runs(evidence.associations)}
