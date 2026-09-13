"""Small completed-run grids, seed stability and predeclared candidate selection."""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from flir_pipeline.reduction.base import ReductionConfig, reduction_space_id
from flir_pipeline.reduction.metrics import summarize_stability
from flir_pipeline.reduction.reducers import implementation_versions
from flir_pipeline.reduction.storage import (
    ReductionInputs,
    reduction_provenance,
    run_to_store,
    verify_reduction,
)
from flir_pipeline.similarity.storage import (
    file_sha256,
    read_json,
    stable_id,
    write_json,
)

SELECTION_POLICY = {
    "version": "pareto_then_equal_mean_rank_v1",
    "criteria": ["trustworthiness_mean", "continuity_mean", "preservation_mean", "stability_mean", "spearman_mean"],
    "required_seeds": [0, 1, 2], "representative_seed": 0,
    "tie_break": ["stability_mean", "preservation_mean", "trustworthiness_mean", "spearman_mean", "configuration_id"],
}
BENCHMARK_FILES = ("runs.csv", "seed_stability.parquet", "configuration_stability.csv",
                   "configuration_summary.csv", "candidates.csv", "summary.json")


def select_candidates(runs: pd.DataFrame, stability: pd.DataFrame,
                      ks: tuple[int, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select exploratory references using full metric evidence, never plot geometry."""
    rows = []
    for (method, config_id), group in runs.groupby(["method", "configuration_id"], sort=True):
        if sorted(group.seed.tolist()) != SELECTION_POLICY["required_seeds"]:
            raise ValueError("Candidate selection requires exactly seeds 0, 1 and 2 for each configuration")
        s = stability.loc[(stability.configuration_id == config_id) & (stability.method == method)]
        if sorted(s.k.tolist()) != list(ks):
            raise ValueError("Incomplete seed stability neighborhoods")
        fields = {"method": method, "configuration_id": config_id,
                  "label": group.label.iloc[0], "seed_count": len(group),
                  "eligible": bool(group.quality_valid.all() and not group.rank_deficient_warning.any()
                                   and np.isfinite(group.spearman_distance).all()),
                  "trustworthiness_mean": float(group[[f"trustworthiness@{k}" for k in ks]].to_numpy().mean()),
                  "continuity_mean": float(group[[f"continuity@{k}" for k in ks]].to_numpy().mean()),
                  "preservation_mean": float(group[[f"jaccard@{k}_mean_jaccard" for k in ks]].to_numpy().mean()),
                  "stability_mean": float(s["mean"].mean()), "spearman_mean": float(group.spearman_distance.mean()),
                  "fit_seconds_total": float(group.fit_seconds.sum()),
                  "reference_reduction_space_id": group.loc[group.seed == 0, "reduction_space_id"].item()}
        for k in ks:
            for metric in (f"trustworthiness@{k}", f"continuity@{k}", f"jaccard@{k}_mean_jaccard"):
                fields[f"{metric}_seed_mean"] = float(group[metric].mean())
                fields[f"{metric}_seed_std"] = float(group[metric].std(ddof=0))
        rows.append(fields)
    summaries = pd.DataFrame(rows)
    selected = []
    criteria = SELECTION_POLICY["criteria"]
    summaries["pareto_nondominated"] = False
    summaries["mean_criterion_rank"] = np.nan
    for _method, group in summaries.groupby("method", sort=True):
        valid = group.loc[group.eligible & np.isfinite(group[criteria]).all(axis=1)]
        if valid.empty:
            raise ValueError("No nonpathological configuration with complete selection metrics")
        values = valid[criteria].to_numpy()
        pareto = [not np.any(np.all(values >= row, axis=1) & np.any(values > row, axis=1)) for row in values]
        mean_rank = valid[criteria].rank(ascending=False, method="average").mean(axis=1)
        summaries.loc[valid.index, "pareto_nondominated"] = pareto
        summaries.loc[valid.index, "mean_criterion_rank"] = mean_rank
        candidates = summaries.loc[valid.index].loc[lambda x: x.pareto_nondominated]
        winner = candidates.sort_values(["mean_criterion_rank", *SELECTION_POLICY["tie_break"]],
                                         ascending=[True, False, False, False, False, True]).iloc[0]
        selected.append({**winner.to_dict(), "seed": 0,
                         "reduction_space_id": winner.reference_reduction_space_id,
                         "status": "exploratory_candidate_for_future_clustering_evaluation"})
    return summaries, pd.DataFrame(selected)


def _label(config: ReductionConfig) -> str:
    p = config.hyperparameters
    return f"perplexity={p['perplexity']:g}" if config.method == "tsne" else f"MN_ratio={p['MN_ratio']:g}"


def benchmark_to_store(inputs: ReductionInputs, configs: list[ReductionConfig],
                       output_root: Path = Path("artifacts/reduction")) -> Path:
    """Reuse only verified finished runs; preserve partial directories and record progress."""
    if not configs:
        raise ValueError("A benchmark needs explicit configurations")
    signatures = {(c.evaluation_ks, c.distance_sample_size, c.distance_sample_seed, c.output_dimension) for c in configs}
    if len(signatures) != 1:
        raise ValueError("A benchmark requires a shared evaluation protocol and output dimension")
    feature = inputs.feature
    versions = {c.method: implementation_versions(c.method) for c in configs}
    run_ids = [reduction_space_id(feature["dataset_id"], feature["feature_space_id"], c, versions[c.method]) for c in configs]
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("Duplicate runs in benchmark")
    for method in versions:
        if sum(c.method == method for c in configs) > 9:
            raise ValueError("The principal grid is limited to nine runs per encoder/method")
    benchmark_id = stable_id({"run_ids": sorted(run_ids), "selection_policy": SELECTION_POLICY})
    root = output_root/feature["extractor"]/feature["dataset_id"]/feature["feature_space_id"]
    output = root/"benchmarks"/benchmark_id
    if output.exists():
        if not (output/"metadata.json").is_file() or not verify_benchmark(output)["quality_valid"]:
            raise ValueError("Incomplete or invalid benchmark preserved; select another root")
        if read_json(output/"metadata.json")["input_signatures"] != inputs.signatures:
            raise ValueError("Benchmark source fingerprints changed")
        return output
    started = time.perf_counter()
    reference = inputs.reference(configs[0])
    rows, run_directories = [], []
    for index, config in enumerate(configs, 1):
        print(f"START {index}/{len(configs)} {config.method} {_label(config)} seed={config.seed}", flush=True)
        directory = run_to_store(inputs, config, output_root, reference)
        run_directories.append(directory)
        meta, metrics, quality = (read_json(directory/name) for name in ("metadata.json", "metrics.json", "quality.json"))
        rows.append({"reduction_space_id": meta["reduction_space_id"], "configuration_id": meta["configuration_id"],
                     "method": config.method, "label": _label(config), "seed": config.seed,
                     "fit_seconds": meta["fit_seconds"], "backend_total_seconds": meta["backend_total_seconds"],
                     "evaluation_seconds": meta["evaluation_seconds"], "quality_valid": quality["quality_valid"],
                     "rank_deficient_warning": quality["rank_deficient_warning"], **metrics})
        print(f"DONE {index}/{len(configs)} id={directory.name} fit={meta['fit_seconds']:.2f}s", flush=True)
    runs = pd.DataFrame(rows)
    stability_contents, stability_summaries = [], []
    for (method, config_id), group in runs.groupby(["method", "configuration_id"], sort=True):
        group = group.sort_values("seed")
        neighbors = [pd.read_parquet(root/method/run_id/"nearest_neighbors.parquet") for run_id in group.reduction_space_id]
        contents, summary = summarize_stability(neighbors, group.seed.tolist(), configs[0].evaluation_ks)
        stability_contents.append(contents.assign(method=method, configuration_id=config_id))
        stability_summaries.append(summary.assign(method=method, configuration_id=config_id))
    stability = pd.concat(stability_summaries, ignore_index=True)
    summaries, candidates = select_candidates(runs, stability, configs[0].evaluation_ks)
    output.mkdir(parents=True, exist_ok=False)
    runs.to_csv(output/"runs.csv", index=False)
    pd.concat(stability_contents, ignore_index=True).to_parquet(output/"seed_stability.parquet", index=False)
    stability.to_csv(output/"configuration_stability.csv", index=False)
    summaries.to_csv(output/"configuration_summary.csv", index=False)
    candidates.to_csv(output/"candidates.csv", index=False)
    summary = {"runs": len(runs), "runs_by_method": runs.method.value_counts().to_dict(),
               "contents": len(inputs.content_index), "seeds": sorted(runs.seed.unique().tolist()),
               "fit_seconds_total": float(runs.fit_seconds.sum()), "fit_seconds_min": float(runs.fit_seconds.min()),
               "fit_seconds_max": float(runs.fit_seconds.max()), "benchmark_seconds": time.perf_counter()-started,
               "candidate_count": len(candidates), "selection_policy": SELECTION_POLICY}
    write_json(output/"summary.json", summary)
    metadata = {**reduction_provenance(), "artifact_kind": "reduction_benchmark", "benchmark_id": benchmark_id,
                "dataset_id": feature["dataset_id"], "feature_space_id": feature["feature_space_id"], "extractor": feature["extractor"],
                "reference_similarity_space_id": inputs.similarity["similarity_space_id"],
                "configs": [asdict(c) for c in configs], "selection_policy": SELECTION_POLICY,
                "input_signatures": inputs.signatures,
                "runs": [{"method": c.method, "reduction_space_id": p.name, "metadata_sha256": file_sha256(p/"metadata.json")} for c, p in zip(configs, run_directories, strict=True)],
                "output_sha256": {name: file_sha256(output/name) for name in BENCHMARK_FILES}}
    write_json(output/"metadata.json", metadata)
    if not verify_benchmark(output)["quality_valid"]:
        raise ValueError("Written benchmark failed verification; preserved for inspection")
    return output


def verify_benchmark(directory: Path, inputs: ReductionInputs | None = None) -> dict:
    checks = {"metadata_exists": (directory/"metadata.json").is_file()}
    try:
        meta = read_json(directory/"metadata.json")
        checks["kind_and_policy_valid"] = meta["artifact_kind"] == "reduction_benchmark" and meta["selection_policy"] == SELECTION_POLICY
        configs = [ReductionConfig(**c) for c in meta["configs"]]
        ids = [r["reduction_space_id"] for r in meta["runs"]]
        checks["identity_valid"] = len(ids) == len(set(ids)) and meta["benchmark_id"] == stable_id({"run_ids": sorted(ids), "selection_policy": SELECTION_POLICY})
        checks["output_checksums_valid"] = set(meta["output_sha256"]) == set(BENCHMARK_FILES) and all(file_sha256(directory/name) == meta["output_sha256"][name] for name in BENCHMARK_FILES)
        reference = inputs.reference(configs[0]) if inputs is not None else None
        run_checks = []
        for config, run in zip(configs, meta["runs"], strict=True):
            path = directory.parents[1]/run["method"]/run["reduction_space_id"]
            current = read_json(path/"metadata.json")
            run_checks.append(file_sha256(path/"metadata.json") == run["metadata_sha256"]
                              and ReductionConfig(**current["config"]) == config
                              and all(current[key] == meta[key] for key in ("dataset_id", "feature_space_id", "extractor"))
                              and verify_reduction(path, inputs, reference)["quality_valid"])
        checks["all_runs_valid"] = bool(run_checks) and all(run_checks)
        runs, stability = pd.read_csv(directory/"runs.csv"), pd.read_csv(directory/"configuration_stability.csv")
        summaries, candidates = select_candidates(runs, stability, configs[0].evaluation_ks)
        pd.testing.assert_frame_equal(summaries, pd.read_csv(directory/"configuration_summary.csv"), check_exact=False, atol=1e-12, rtol=1e-12)
        pd.testing.assert_frame_equal(candidates, pd.read_csv(directory/"candidates.csv"), check_exact=False, atol=1e-12, rtol=1e-12)
        checks["selection_recomputed"] = True
        checks["runs_table_coverage_valid"] = len(runs) == len(ids) and set(runs.reduction_space_id) == set(ids)
        if inputs is not None:
            checks["source_inputs_match"] = meta["input_signatures"] == inputs.signatures
        checks["quality_valid"] = all(checks.values())
    except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError, AssertionError) as error:
        checks.update(quality_valid=False, error_type=type(error).__name__)
    return checks
