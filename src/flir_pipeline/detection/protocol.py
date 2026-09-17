"""Freeze existing splits before detection; never select candidates by YOLO outcomes."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from flir_pipeline.similarity.storage import (
    file_sha256,
    read_json,
    stable_id,
    write_json,
)
from flir_pipeline.splitting.experiments import verify_comparison

AUDIT_CANDIDATES = ("C01", "C05", "C07", "C09", "C10", "C12")
METRICS = ("precision", "recall", "map50", "map50_95")


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config["protocol"] != "controlled_detector_v1" or config["model"] != "yolo11n.pt":
        raise ValueError("Unsupported detector protocol/model")
    train = config["train"]
    if (train["batch"] is not None and train["batch"] < 1) or train["epochs"] < 1 or train["imgsz"] % 32:
        raise ValueError("Invalid batch, epochs or image size")
    if train["optimizer"] == "auto" or not train["val"] or train["patience"] != 0:
        raise ValueError("Protocol fixes optimizer, validation and the full epoch budget")
    if not train["pretrained"] or train["fraction"] != 1 or train["single_cls"]:
        raise ValueError("Protocol requires the full five-class data and pretrained initialization")
    if not train["deterministic"] or train["amp"]:
        raise ValueError("Protocol requires deterministic FP32 training")
    seeds = config["training_seeds"]
    if len(seeds) < 2 or len(set(seeds)) != len(seeds) or 42 not in seeds:
        raise ValueError("At least two distinct training seeds including pilot seed 42 are required")
    if config["split_seeds"] != list(range(5)):
        raise ValueError("This protocol freezes all five existing split seeds")
    evaluation = config["evaluation"]
    if not 0 < evaluation["confidence_floor"] <= evaluation["precision_recall_confidence"] < 1:
        raise ValueError("Invalid fixed confidence thresholds")
    if evaluation["precision_recall_iou"] != .5 or evaluation["metric_version"] != "image_stats_ap101_v1":
        raise ValueError("Unsupported metric definition")
    if evaluation["bootstrap_resamples"] < 1 or not 0 < evaluation["confidence_level"] < 1:
        raise ValueError("Invalid bootstrap configuration")
    return config


def detector_run_id(split_space_id: str, model_config: dict, seed: int) -> str:
    """Only mathematical inputs enter identity; paths, stage and timestamps do not."""
    return stable_id({"split_space_id": split_space_id, "model_configuration": model_config, "training_seed": seed})


def experiment_matrix(splits: list[dict], seeds: list[int]) -> pd.DataFrame:
    if len(set(seeds)) != len(seeds) or not seeds:
        raise ValueError("Duplicate or empty detector seed list")
    if len({s["split_space_id"] for s in splits}) != len(splits):
        raise ValueError("Duplicate split in matrix")
    return pd.DataFrame([{**s, "detector_seed": seed, "pilot": s["split_seed"] == 0 and seed == 42}
                         for s in splits for seed in seeds])


def audit_and_plan(comparison: Path, config_path: Path, output: Path) -> dict:
    verification = verify_comparison(comparison)
    if not verification["quality_valid"]:
        raise ValueError(f"Splitting comparison failed verification: {verification}")
    config = load_config(config_path)
    runs = pd.read_csv(comparison/"runs.csv")
    candidates = pd.read_csv(comparison/"clustering_candidates.csv")
    stability = pd.read_csv(comparison/"seed_stability.csv").groupby("candidate_label").same_named_split_fraction.agg(["mean", "min"])
    audit = runs.loc[runs.candidate_label.isin(AUDIT_CANDIDATES)].merge(candidates, on=["candidate_label", "clustering_space_id"], validate="many_to_one")
    audit = audit.copy()
    root = comparison.parent.parent/"runs"
    for idx, row in audit.iterrows():
        temporal = pd.read_parquet(root/row.split_space_id/"temporal_cross_split.parquet").set_index("frame_delta_max")
        for delta in (1, 5, 10):
            audit.loc[idx, f"temporal_at{delta}"] = temporal.loc[delta, "fraction"]
        for key in ("mean", "min"):
            audit.loc[idx, f"split_stability_{key}"] = stability.loc[row.candidate_label, key]
    historical = runs.loc[runs.strategy == "historical"].iloc[0]
    c10 = audit.loc[audit.candidate_label == "C10"]
    decisions = []
    for label, group in audit.groupby("candidate_label", sort=True):
        checks = {
            "coverage": bool(group.all_classes_covered.all()),
            "atomic": bool((group.exact_duplicate_cross_split_count == 0).all() and (group.cluster_fracture_count == 0).all()),
            "size_at_most_2_percent": bool(group.max_relative_record_deviation.max() <= .02),
            "class_deviation_at_most_2_pp": bool(group.class_deviation_pp.max() <= 2),
            "noise_below_80_percent": bool(group.noise_fraction.max() < .8),
            "split_retention_min_at_least_60_percent": bool(group.split_stability_min.min() >= .6),
            "temporal_worst_no_higher_than_C10": bool(group.temporal_at5.max() <= c10.temporal_at5.max()),
            "both_nn_below_historical": bool(all(group[f"{e}_nn_mean"].max() < historical[f"{e}_nn_mean"] for e in ("dinov2", "clip"))),
            "both_pair_counts_below_historical": bool(all(group[f"{e}_top001_pairs"].max() < historical[f"{e}_top001_pairs"] for e in ("dinov2", "clip"))),
        }
        decisions.append({"candidate_label": label, **checks, "eligible_secondary": label in ("C05", "C07", "C09", "C12") and all(checks.values())})
    eligible = [d["candidate_label"] for d in decisions if d["eligible_secondary"]]
    if len(eligible) > 1:
        raise ValueError("Multiple eligible trade-offs require an explicit pre-training Pareto decision")
    selected = ["historical", "random_content", "C10", *eligible]
    selected_runs = runs.loc[runs.candidate_label.isin(selected)].sort_values(["candidate_label", "seed"])
    splits = []
    for row in selected_runs.itertuples():
        directory = root/row.split_space_id
        meta = read_json(directory/"metadata.json")
        summary = read_json(directory/"split_summary.json")
        identity = meta["identity_payload"]
        splits.append({"strategy": row.candidate_label, "split_seed": int(row.seed), "split_space_id": row.split_space_id,
                       "dataset_id": identity["dataset_id"], "clustering_space_id": identity["clustering_space_id"],
                       "noise_policy": identity["configuration"]["noise_policy"], "target_ratios": identity["configuration"]["target_ratios"],
                       "record_counts": {s: int(getattr(row, f"{s}_records")) for s in ("train", "val", "test")},
                       "class_counts": pd.read_parquet(directory/"class_balance.parquet").to_dict("records"),
                       "balance": summary["balance"], "split_metadata_sha256": file_sha256(directory/"metadata.json"),
                       "context": {k: float(getattr(row, k)) for k in ("exact_duplicate_cross_split_count", "dinov2_top001_pairs", "clip_top001_pairs", "dinov2_nn_mean", "clip_nn_mean", "temporal_at5", "class_deviation_pp")}})
    identity = {"config": config, "splits": splits, "selection_constraints": decisions, "comparison_metadata_sha256": file_sha256(comparison/"metadata.json")}
    plan = {"plan_id": stable_id(identity), "identity": identity, "state": "AWAITING_HARDWARE_PROBE_AND_RUNTIME_FREEZE",
            "stage_a": "split seed 0, detector seed 42, exact candidate configuration; operational validation only",
            "stage_b": "all planned cells; valid identical pilot cells are reused once, only after every pilot passes",
            "primary_metric": "map50_95", "cpu_stage_b_automatic": False}
    output.mkdir(parents=True, exist_ok=True)
    if (output/"plan.json").exists() and read_json(output/"plan.json") != plan:
        raise ValueError("Existing plan differs; use a separate experiment directory, never overwrite a freeze")
    write_json(output/"plan.json", plan)
    audit.to_csv(output/"candidate_audit.csv", index=False)
    pd.DataFrame(decisions).to_csv(output/"selection_constraints.csv", index=False)
    experiment_matrix(splits, config["training_seeds"]).to_json(output/"matrix.json", orient="records", indent=2)
    write_json(output/"split_verification.json", verification)
    return plan


def verify_plan(directory: Path) -> dict:
    plan = read_json(directory/"plan.json")
    if stable_id(plan["identity"]) != plan["plan_id"]:
        raise ValueError("Plan identity mismatch")
    return plan
