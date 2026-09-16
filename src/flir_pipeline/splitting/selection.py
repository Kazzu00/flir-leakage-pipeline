"""Reproducible diverse clustering sample and constraint/Pareto split review."""

from __future__ import annotations

import numpy as np
import pandas as pd

from flir_pipeline.clustering.selection import pareto_mask

DIVERSITY_COLUMNS = ("noise_fraction", "n_clusters_excluding_noise",
                     "parameters_common_clustered_ari_min", "parameters_common_clustered_ami_min",
                     "seeds_common_clustered_ari_min", "seeds_common_clustered_ami_min",
                     "visual_neighbor_coherence@10", "temporal_recall@5")
CANDIDATE_POLICY = {
    "version": "six_strata_two_anchors_v1", "source": "existing_stage_b_pareto",
    "strata": ["encoder", "representation"], "per_stratum": 2,
    "first": "minimum_noise_then_clustering_id",
    "second": "prefer_globally_missing_algorithm_then_maximin_rank_distance",
    "distance": "mean_absolute_percentile_rank_difference_on_jointly_available_metrics",
    "diversity_columns": list(DIVERSITY_COLUMNS),
    "tie_break": "clustering_space_id_ascending", "silhouette_not_used": True,
    "stratum_order": "encoder_representation_ascending",
}
SPLIT_CRITERIA = ("max_relative_record_deviation", "class_deviation_pp",
                  "dinov2_nn_mean", "clip_nn_mean", "dinov2_top001_fraction", "clip_top001_fraction",
                  "temporal_at5", "residual_seed_std_max")
FINAL_POLICY = {"version": "robust_constraints_pareto_v1", "required_seeds": [0, 1, 2, 3, 4],
                "constraints": "all seeds valid; zero exact overlap/fracture; all five classes in every split; max relative record deviation <=0.10",
                "aggregation": "worst (maximum) over seeds for each loss; residual std is population std maximum across two encoders",
                "criteria": list(SPLIT_CRITERIA), "direction": "minimize_all; no weighted score",
                "final_anchors": ["dinov2_top001_fraction", "clip_top001_fraction", "class_deviation_pp"],
                "representative_seed": 0, "max_final_candidates": 3,
                "tie_break": "clustering_space_id_ascending"}


def select_clustering_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    pool = candidates.loc[candidates.pareto_stage_b & (candidates.n_clusters_excluding_noise > 1)].copy()
    if pool.empty or not pool.clustering_space_id.is_unique:
        raise ValueError("Need distinct nondegenerate Pareto clustering candidates")
    pool = pool.sort_values("clustering_space_id").reset_index(drop=True)
    ranks = pool[list(DIVERSITY_COLUMNS)].rank(pct=True)
    strata = list(pool.groupby(["encoder", "representation"], sort=True))
    chosen, reasons = [], {}
    for _, group in strata:
        i = group.sort_values(["noise_fraction", "clustering_space_id"]).index[0]
        chosen.append(i)
        reasons[i] = "minimum noise within encoder/representation Pareto stratum"
    for _, group in strata:
        remaining = group.loc[~group.index.isin(chosen)]
        if remaining.empty:
            continue
        missing_algorithms = set(pool.algorithm) - set(pool.loc[chosen, "algorithm"])
        prioritized = remaining.loc[remaining.algorithm.isin(missing_algorithms)]
        if not prioritized.empty:
            remaining = prioritized
        distances = {}
        for i in remaining.index:
            difference = abs(ranks.loc[chosen] - ranks.loc[i]).mean(axis=1, skipna=True)
            distances[i] = float(difference.min())
        best = sorted(distances, key=lambda i: (-distances[i], pool.loc[i, "clustering_space_id"]))[0]
        chosen.append(best)
        reasons[best] = "algorithm coverage then maximin percentile-rank diversity"
    out = pool.loc[chosen].copy().sort_values(["encoder", "representation", "clustering_space_id"]).reset_index(drop=True)
    out["candidate_label"] = [f"C{i+1:02d}" for i in range(len(out))]
    out["splitting_selection_reason"] = out.clustering_space_id.map({pool.loc[i, "clustering_space_id"]: reasons[i] for i in chosen})
    return out


def select_final_candidates(runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate worst seed, then retain the entire Pareto front and <=3 explicit anchors."""
    rows = []
    clustered = runs.loc[runs.strategy == "cluster_aware"]
    for clustering_id, group in clustered.groupby("clustering_space_id", sort=True):
        row = {"clustering_space_id": clustering_id, "candidate_label": group.candidate_label.iloc[0],
               "seed_count": len(group), "all_classes_covered": bool(group.all_classes_covered.all()),
               "all_valid": bool(group.quality_valid.all()),
               "exact_duplicate_cross_split_count": int(group.exact_duplicate_cross_split_count.max()),
               "cluster_fracture_count": int(group.cluster_fracture_count.max())}
        row.update({k: float(group[k].max()) for k in SPLIT_CRITERIA if k != "residual_seed_std_max"})
        row["residual_seed_std_max"] = max(float(group[f"{e}_nn_mean"].std(ddof=0)) for e in ("dinov2", "clip"))
        row["eligible"] = bool(sorted(group.seed) == FINAL_POLICY["required_seeds"] and row["all_valid"] and row["all_classes_covered"]
                               and row["exact_duplicate_cross_split_count"] == 0 and row["cluster_fracture_count"] == 0
                               and row["max_relative_record_deviation"] <= .10)
        row["representative_split_space_id"] = group.loc[group.seed == 0, "split_space_id"].iloc[0] if (group.seed == 0).any() else None
        rows.append(row)
    table = pd.DataFrame(rows)
    if table.empty:
        return table, table.copy()
    table["pareto"] = False
    eligible = table.loc[table.eligible]
    if not eligible.empty:
        if not np.isfinite(eligible[list(SPLIT_CRITERIA)].to_numpy(float)).all():
            raise ValueError("Final selection requires all declared metrics")
        mask, omitted = pareto_mask(eligible, dict.fromkeys(SPLIT_CRITERIA, "min"))
        if omitted:
            raise ValueError("Final selection must not omit criteria")
        table.loc[eligible.loc[mask].index, "pareto"] = True
    table["final_reason"] = ""
    front = table.loc[table.pareto]
    selected = []
    for criterion in FINAL_POLICY["final_anchors"]:
        if front.empty:
            break
        winner = front.sort_values([criterion, "clustering_space_id"]).index[0]
        if winner not in selected:
            selected.append(winner)
        table.loc[winner, "final_reason"] += f"minimum worst-seed {criterion}; "
    return table, table.loc[selected].copy()
