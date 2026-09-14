"""Pareto sets and explicit review anchors; no weighted or class-derived score."""

from __future__ import annotations

import numpy as np
import pandas as pd

from flir_pipeline.clustering.base import ClusteringConfig

CRITERIA_A = {"silhouette_original_space": "max", "weighted_mean_intra_cluster_similarity": "max",
              "visual_neighbor_coherence@10": "max", "temporal_recall@5": "max",
              "noise_fraction": "min", "largest_cluster_fraction": "min"}
ANCHORS = ("silhouette_original_space", "visual_neighbor_coherence@10", "temporal_recall@5")
SELECTION_POLICY = {"version": "pareto_extreme_review_v1", "stage_a_criteria": CRITERIA_A,
                    "anchors": list(ANCHORS), "max_per_encoder_representation_algorithm": 3,
                    "tie_break": "configuration_id_ascending", "exclude": "n_clusters<=1 or undefined primary silhouette",
                    "stage_b": "Pareto with minimum common-clustered local robustness and, for reductions, seed stability",
                    "final_reference": "maximum original silhouette within final Pareto set; clustering ID tie break"}


def pareto_mask(frame: pd.DataFrame, criteria: dict[str, str]) -> tuple[pd.Series, list[str]]:
    """Omit a criterion only if undefined throughout the group; partial missing is ineligible."""
    active = [name for name in criteria if name in frame and pd.to_numeric(frame[name], errors="coerce").notna().any()]
    omitted = sorted(set(criteria)-set(active))
    if not active or frame.empty:
        return pd.Series(False, index=frame.index), omitted
    values = frame[active].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float, copy=True)
    values *= np.array([1 if criteria[name] == "max" else -1 for name in active])
    finite = np.isfinite(values).all(axis=1)
    result = np.zeros(len(frame), dtype=bool)
    for i in np.flatnonzero(finite):
        result[i] = not np.any(np.all(values[finite] >= values[i], axis=1) & np.any(values[finite] > values[i], axis=1))
    return pd.Series(result, index=frame.index), omitted


def shortlist_screening(runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    table = runs.copy()
    table["eligible_stage_a"] = (table.n_clusters_excluding_noise > 1) & table.silhouette_original_space.notna()
    table["pareto_stage_a"] = False
    table["shortlist_stage_a"] = False
    table["shortlist_reason"] = ""
    omitted_rows = []
    for keys, group in table.groupby(["encoder", "representation", "algorithm"], sort=True):
        eligible = group.loc[group.eligible_stage_a]
        mask, omitted = pareto_mask(eligible, CRITERIA_A)
        omitted_rows.append({"encoder": keys[0], "representation": keys[1], "algorithm": keys[2], "omitted_criteria": omitted})
        front = eligible.loc[mask]
        table.loc[front.index, "pareto_stage_a"] = True
        chosen = []
        for anchor in ANCHORS:
            if front.empty or anchor in omitted:
                continue
            winner = front.sort_values([anchor, "configuration_id"], ascending=[False, True]).index[0]
            if winner not in chosen:
                chosen.append(winner)
                table.loc[winner, "shortlist_reason"] = f"Pareto extreme: {anchor}"
        for index in front.sort_values("configuration_id").index:
            if len(chosen) >= 3:
                break
            if index not in chosen:
                chosen.append(index)
                table.loc[index, "shortlist_reason"] = "Pareto remainder: deterministic configuration ID order"
        table.loc[chosen, "shortlist_stage_a"] = True
    return table, table.loc[table.shortlist_stage_a].copy(), omitted_rows


def parameter_neighbors(candidate: ClusteringConfig, pool: dict[str, ClusteringConfig]) -> list[tuple[str, str]]:
    """Immediate observed grid neighbors along one prescribed coordinate at a time."""
    axes = {"dbscan": ("eps_quantile",), "optics": ("xi",), "hdbscan": ("min_cluster_size", "min_samples")}[candidate.algorithm]
    found = []
    p = candidate.hyperparameters
    for axis in axes:
        comparable = [(identity, config) for identity, config in pool.items() if config.algorithm == candidate.algorithm
                      and all(config.hyperparameters[k] == v for k, v in p.items() if k != axis)]
        values = sorted({c.hyperparameters[axis] for _, c in comparable})
        if p[axis] not in values:
            raise ValueError("Candidate is absent from its screening grid")
        position = values.index(p[axis])
        adjacent = {values[i] for i in (position-1, position+1) if 0 <= i < len(values)}
        found.extend((identity, axis) for identity, config in comparable if config.hyperparameters[axis] in adjacent)
    return sorted(found)


def final_candidates(evaluated: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    table = evaluated.copy()
    table["eligible_stage_b"] = table.eligible_stage_a & table["parameters_common_clustered_all_nontrivial"].fillna(False).astype(bool)
    reduced = table.representation.ne("original_l2")
    table.loc[reduced, "eligible_stage_b"] &= (table.loc[reduced, "seeds_all_nondegenerate"].fillna(False).astype(bool)
                                               & table.loc[reduced, "seeds_common_clustered_all_nontrivial"].fillna(False).astype(bool))
    table["pareto_stage_b"] = False
    table["reference_for_review"] = False
    omissions = []
    for (encoder, representation), group in table.groupby(["encoder", "representation"], sort=True):
        criteria = {**CRITERIA_A, "parameters_common_clustered_ari_min": "max", "parameters_common_clustered_ami_min": "max"}
        if representation != "original_l2":
            criteria.update(seeds_common_clustered_ari_min="max", seeds_common_clustered_ami_min="max")
        eligible = group.loc[group.eligible_stage_b]
        front, omitted = pareto_mask(eligible, criteria)
        # Stability is required at this stage; absent metrics cannot disappear
        # silently from the final candidate rule.
        missing_required = [k for k in omitted if k.startswith(("seeds_", "parameters_"))]
        indices = eligible.loc[front].index if not missing_required else []
        table.loc[indices, "pareto_stage_b"] = True
        if len(indices):
            winner = table.loc[indices].sort_values(["silhouette_original_space", "clustering_space_id"], ascending=[False, True]).index[0]
            table.loc[winner, "reference_for_review"] = True
        omissions.append({"encoder": encoder, "representation": representation, "omitted_criteria": omitted, "missing_required_stability": missing_required})
    return table, table.loc[table.pareto_stage_b].copy(), omissions
