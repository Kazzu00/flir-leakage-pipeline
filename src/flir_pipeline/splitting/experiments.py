"""Multi-seed baseline/cluster comparison, stability and robust Pareto selection."""

from __future__ import annotations

from dataclasses import replace
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from flir_pipeline.similarity.storage import (
    file_sha256,
    read_json,
    stable_id,
    write_json,
)
from flir_pipeline.splitting.base import SplitConfig
from flir_pipeline.splitting.metrics import cluster_fracture, membership_masks
from flir_pipeline.splitting.selection import (
    CANDIDATE_POLICY,
    FINAL_POLICY,
    select_final_candidates,
)
from flir_pipeline.splitting.storage import (
    SplitInputs,
    build_run,
    cluster_labels,
    splitting_provenance,
    verify_split,
)


def run_row(directory: Path, inputs: SplitInputs) -> dict:
    metadata, summary = read_json(directory/"metadata.json"), read_json(directory/"split_summary.json")
    identity = metadata["identity_payload"]
    strategy = identity["configuration"]["strategy"]
    cid = identity["clustering_space_id"]
    label = inputs.candidates.set_index("clustering_space_id").candidate_label[cid] if cid else strategy
    balance = summary["balance"]
    row = {"split_space_id": metadata["split_space_id"], "strategy": strategy, "clustering_space_id": cid,
           "candidate_label": label, "seed": identity["seed"], **summary["quality"],
           "max_relative_record_deviation": balance["max_relative_record_deviation"],
           "class_deviation_pp": balance["mean_absolute_class_distribution_deviation_pp"],
           "all_classes_covered": balance["all_classes_covered"],
           "solver_optimal": metadata["solver"].get("optimal_within_tolerance"),
           "solver_gap": metadata["solver"].get("mip_gap")}
    for size in balance["sizes"]:
        s = size["split"]
        row.update({f"{s}_records": size["records"], f"{s}_contents": size["contents"], f"{s}_empty": size["empty_annotations"],
                    f"{s}_heavy_machinery": balance["heavy_machinery_counts"][s]})
    for encoder in ("dinov2", "clip"):
        visual = summary["visual"][encoder]
        row.update({f"{encoder}_nn_{k}": v for k, v in visual["cross_split_nn"].items()})
        for k, metrics in visual["topk"].items():
            row[f"{encoder}_top{k}_cross_fraction"] = metrics["fraction"]
        quantiles = pd.read_parquet(directory/f"quantile_pairs_{encoder}.parquet")
        top = quantiles.loc[np.isclose(quantiles["quantile"], .999)].iloc[0]
        row[f"{encoder}_top001_fraction"] = top.cross_split_fraction
        row[f"{encoder}_top001_pairs"] = int(top.cross_split_count)
    temporal = pd.read_parquet(directory/"temporal_cross_split.parquet")
    row["temporal_at5"] = float(temporal.loc[temporal.frame_delta_max == 5, "fraction"].iloc[0])
    return row


def comparison_tables(directories: list[Path], inputs: SplitInputs) -> tuple[dict[str, pd.DataFrame], dict]:
    runs = pd.DataFrame([run_row(p, inputs) for p in directories]).sort_values(["strategy", "candidate_label", "seed"]).reset_index(drop=True)
    assignments = {p.name: pd.read_parquet(p/"record_split_assignments.parquet") for p in directories}
    ids = sorted(inputs.manifest.content_id.unique())
    masks = {i: membership_masks(a, ids) for i, a in assignments.items()}
    fractures = []
    for cid in inputs.candidates.clustering_space_id:
        labels = cluster_labels(inputs, cid).reindex(ids).to_numpy()
        candidate_label = inputs.candidates.set_index("clustering_space_id").candidate_label[cid]
        for row in runs.itertuples():
            if row.strategy == "cluster_aware" and row.clustering_space_id != cid:
                continue
            fractures.append({"clustering_space_id": cid, "candidate_label": candidate_label,
                              "split_space_id": row.split_space_id, "strategy": row.strategy, "seed": row.seed,
                              **cluster_fracture(labels, masks[row.split_space_id])})
    stability = []
    for (strategy, label), group in runs.loc[runs.strategy != "historical"].groupby(["strategy", "candidate_label"], sort=True):
        for a, b in combinations(group.itertuples(), 2):
            stability.append({"strategy": strategy, "candidate_label": label, "seed_a": a.seed, "seed_b": b.seed,
                              "same_named_split_fraction": float(np.mean(masks[a.split_space_id] == masks[b.split_space_id]))})
    pareto, final = select_final_candidates(runs)
    metric_columns = ["max_relative_record_deviation", "class_deviation_pp", "dinov2_nn_mean", "clip_nn_mean",
                      "dinov2_top001_pairs", "clip_top001_pairs", "temporal_at5"]
    variation = []
    for (strategy, label), group in runs.groupby(["strategy", "candidate_label"], sort=True):
        for column in metric_columns:
            v = group[column]
            variation.append({"strategy": strategy, "candidate_label": label, "metric": column, "n": len(v),
                              "mean": float(v.mean()), "std_population": float(v.std(ddof=0)),
                              "min": float(v.min()), "max": float(v.max())})
    summary = {"run_count": len(runs), "clustering_candidate_count": len(inputs.candidates),
               "random_seeds": runs.loc[runs.strategy == "random_content", "seed"].tolist(),
               "cluster_seeds": sorted(runs.loc[runs.strategy == "cluster_aware", "seed"].unique().tolist()),
               "noise_policy": "singleton", "valid_runs": int(runs.quality_valid.sum()),
               "pareto_count": int(pareto.pareto.sum()) if len(pareto) else 0,
               "eligible_candidates": int(pareto.eligible.sum()) if len(pareto) else 0,
               "final_candidates": final[["candidate_label", "clustering_space_id", "representative_split_space_id", "final_reason"]].to_dict("records") if len(final) else [],
               "new_exact_overlap_max": int(runs.loc[runs.strategy != "historical", "exact_duplicate_cross_split_count"].max()),
               "cluster_fracture_max": int(runs.loc[runs.strategy == "cluster_aware", "cluster_fracture_count"].max()),
               "annotation_audit": inputs.annotation_summary}
    return {"runs.csv": runs, "cluster_fracture.csv": pd.DataFrame(fractures), "seed_stability.csv": pd.DataFrame(stability),
            "metric_variation.csv": pd.DataFrame(variation), "pareto.csv": pareto, "final_candidates.csv": final}, summary


def compare_to_store(inputs: SplitInputs, cluster_config: SplitConfig, random_config: SplitConfig,
                     output_root: Path = Path("artifacts/splitting")) -> Path:
    if cluster_config.strategy != "cluster_aware" or random_config.strategy != "random_content":
        raise ValueError("Comparison needs cluster-aware and content-random configurations")
    if cluster_config.seeds != (0, 1, 2, 3, 4) or random_config.seeds != (0, 1, 2, 3, 4):
        raise ValueError("The principal comparison requires seeds 0 through 4")
    if cluster_config.target_ratios != random_config.target_ratios:
        raise ValueError("Baselines and clustered runs must use the same targets")
    historical = replace(random_config, strategy="historical")
    runs = [build_run(inputs, historical, 0, output_root)]
    for seed in random_config.seeds:
        runs.append(build_run(inputs, random_config, seed, output_root))
        print(f"RANDOM seed={seed} verified", flush=True)
    for candidate in inputs.candidates.itertuples():
        for seed in cluster_config.seeds:
            path = build_run(inputs, cluster_config, seed, output_root, candidate.clustering_space_id)
            runs.append(path)
            solver = read_json(path/"metadata.json")["solver"]
            print(f"{candidate.candidate_label} {candidate.encoder}/{candidate.representation}/{candidate.algorithm} seed={seed} status={solver['status']} gap={solver['mip_gap']:.5g}", flush=True)
    identity = {"runs": sorted(p.name for p in runs), "candidate_policy": CANDIDATE_POLICY, "final_policy": FINAL_POLICY,
                "input_signatures": inputs.signatures}
    output = output_root/"comparisons"/stable_id(identity)
    if output.exists():
        if not verify_comparison(output, inputs)["quality_valid"]:
            raise ValueError("Incomplete/incompatible split comparison preserved")
        return output
    tables, summary = comparison_tables(runs, inputs)
    output.mkdir(parents=True, exist_ok=False)
    for name, table in tables.items():
        table.to_csv(output/name, index=False)
    inputs.candidates.to_csv(output/"clustering_candidates.csv", index=False)
    write_json(output/"summary.json", summary)
    write_json(output/"metadata.json", {**splitting_provenance(), "artifact_kind": "split_comparison",
               "comparison_id": output.name, "identity_payload": identity,
               "runs": [{"path": p.relative_to(output_root).as_posix(), "metadata_sha256": file_sha256(p/"metadata.json")} for p in runs],
               "output_sha256": {p.name: file_sha256(p) for p in output.iterdir() if p.is_file()}})
    if not verify_comparison(output, inputs)["quality_valid"]:
        raise ValueError("Split comparison failed verification; outputs preserved")
    return output


def verify_comparison(directory: Path, inputs: SplitInputs | None = None) -> dict:
    checks = {}
    try:
        meta = read_json(directory/"metadata.json")
        checks["kind_valid"] = meta["artifact_kind"] == "split_comparison"
        checks["identity_valid"] = meta["comparison_id"] == stable_id(meta["identity_payload"])
        checks["policies_match"] = meta["identity_payload"]["candidate_policy"] == CANDIDATE_POLICY and meta["identity_payload"]["final_policy"] == FINAL_POLICY
        required = {"runs.csv", "cluster_fracture.csv", "seed_stability.csv", "metric_variation.csv", "pareto.csv", "final_candidates.csv", "clustering_candidates.csv", "summary.json"}
        checks["checksums_valid"] = set(meta["output_sha256"]) == required and all(file_sha256(directory/n) == h for n, h in meta["output_sha256"].items())
        root = directory.parents[1]
        directories = [root/r["path"] for r in meta["runs"]]
        checks["run_coverage"] = sorted(p.name for p in directories) == meta["identity_payload"]["runs"] and len(set(directories)) == len(directories)
        checks["runs_bound"] = all(file_sha256(root/r["path"]/"metadata.json") == r["metadata_sha256"] for r in meta["runs"])
        checks["all_runs_valid"] = all(verify_split(p, inputs)["quality_valid"] for p in directories)
        if inputs is not None:
            checks["sources_bound"] = meta["identity_payload"]["input_signatures"] == inputs.signatures
            tables, summary = comparison_tables(directories, inputs)
            checks["summary_recomputed"] = summary == read_json(directory/"summary.json")
            for name, table in {**tables, "clustering_candidates.csv": inputs.candidates}.items():
                expected = table.reset_index(drop=True).replace({None: np.nan, "": np.nan})
                actual = pd.read_csv(directory/name)
                pd.testing.assert_frame_equal(expected, actual, check_dtype=False, check_exact=False, atol=1e-12, rtol=1e-12)
            checks["tables_recomputed"] = True
        checks["quality_valid"] = all(checks.values())
    except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError, AssertionError) as error:
        checks.update(quality_valid=False, error_type=type(error).__name__)
    return checks


def export_image_lists(directory: Path, manifest_path: Path, image_root: Path, output: Path) -> None:
    """Later handoff only: reference already materialized images, never copy/move them."""
    if not verify_split(directory)["quality_valid"]:
        raise ValueError("Cannot export an invalid split")
    meta = read_json(directory/"metadata.json")
    if meta["identity_payload"]["configuration"]["strategy"] == "historical":
        raise ValueError("Historical overlap is not a new atomic split export")
    manifest = pd.read_parquet(manifest_path)
    from flir_pipeline.data.identity import dataset_id_from_manifest
    if dataset_id_from_manifest(manifest) != meta["identity_payload"]["dataset_id"]:
        raise ValueError("Export manifest differs from the assigned dataset")
    assignments = pd.read_parquet(directory/"record_split_assignments.parquet")
    verify_assignments_for_export = assignments.merge(manifest[["frame_id", "content_id", "relative_image_path"]], on=["frame_id", "content_id"], validate="one_to_one")
    if len(verify_assignments_for_export) != len(manifest):
        raise ValueError("Export occurrence coverage mismatch")
    paths = []
    root = image_root.resolve()
    for value in verify_assignments_for_export.relative_image_path:
        path = (root/value).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("Export requires safe paths to already materialized images")
        paths.append(str(path))
    verify_assignments_for_export["path"] = paths
    if len(set(paths)) != len(paths):
        raise ValueError("Record-level image paths must preserve separate annotation occurrences")
    output.mkdir(parents=True, exist_ok=False)
    for split in ("train", "val", "test"):
        values = verify_assignments_for_export.loc[verify_assignments_for_export.new_split == split].sort_values("frame_id").path
        (output/f"{split}.txt").write_text("\n".join(values)+"\n", encoding="utf-8")
