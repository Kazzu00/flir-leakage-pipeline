"""Record-weighted atomic units, random baseline and a bounded deterministic MILP."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from flir_pipeline.splitting.base import (
    BALANCE_COLUMNS,
    CLASS_COLUMNS,
    SPLITS,
    SplitConfig,
)


def record_statistics(manifest: pd.DataFrame, instances: pd.DataFrame) -> pd.DataFrame:
    """Preserve conflicting annotations per occurrence; never choose a representative label."""
    if manifest.frame_id.isna().any() or not manifest.frame_id.is_unique or manifest.content_id.isna().any():
        raise ValueError("Manifest needs unique nonnull frame IDs and nonnull content IDs")
    if not set(instances.frame_id) <= set(manifest.frame_id) or not instances.class_id.isin(range(5)).all():
        raise ValueError("Instances must belong to canonical records and classes")
    out = manifest[["frame_id", "content_id", "original_split"]].copy()
    counts = pd.crosstab(instances.frame_id, instances.class_id).reindex(index=out.frame_id, columns=range(5), fill_value=0)
    out[list(CLASS_COLUMNS)] = counts.to_numpy(dtype=np.int64)
    if not np.array_equal(out[list(CLASS_COLUMNS)].sum(axis=1), manifest.num_objects.to_numpy()):
        raise ValueError("Parsed instance counts do not equal manifest counts")
    out["record_count"] = 1
    out["empty_annotation_count"] = (out[list(CLASS_COLUMNS)].sum(axis=1) == 0).astype(int)
    if not np.array_equal(out.empty_annotation_count.astype(bool), manifest.label_empty.astype(bool)):
        raise ValueError("Empty annotations disagree with manifest")
    return out.sort_values("frame_id").reset_index(drop=True)


def make_groups(content_ids: list[str], labels: pd.Series | None = None,
                noise_policy: str = "singleton") -> pd.DataFrame:
    """Noise units preserve cluster_id=-1; group identity is separate from cluster identity."""
    ids = sorted(content_ids)
    if len(set(ids)) != len(ids) or any(not isinstance(i, str) or not i for i in ids):
        raise ValueError("Content identities must be unique nonempty strings")
    if noise_policy != "singleton":
        raise ValueError("Similarity-component ablation is not enabled")
    if labels is not None:
        if not labels.index.is_unique or set(labels.index) != set(ids):
            raise ValueError("Clustering must cover exactly the manifest contents")
        values = labels.reindex(ids).to_numpy()
        if not np.issubdtype(values.dtype, np.integer) or (values < -1).any():
            raise ValueError("Cluster IDs must be integer labels >= -1")
    else:
        values = np.full(len(ids), -1)
    return pd.DataFrame({"content_id": ids, "cluster_id": pd.array(values if labels is not None else [None]*len(ids), dtype="Int64"),
                         "group_id": [f"cluster:{v}" if v >= 0 else f"content:{i}" for i, v in zip(ids, values, strict=True)],
                         "group_type": ["cluster" if v >= 0 else "noise_singleton" if labels is not None else "content_singleton" for v in values]})


def aggregate_groups(records: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    merged = records.merge(groups[["content_id", "group_id"]], on="content_id", validate="many_to_one", how="left")
    if merged.group_id.isna().any() or set(groups.content_id) != set(records.content_id):
        raise ValueError("Groups must cover all and only canonical contents")
    totals = merged.groupby("group_id", sort=True)[list(BALANCE_COLUMNS)].sum()
    totals["content_count"] = groups.groupby("group_id").size()
    return totals.reset_index()


def random_assignment(units: pd.DataFrame, ratios: np.ndarray, seed: int) -> tuple[dict, dict]:
    """Permute contents and cut near cumulative record targets; classes are unused."""
    if len(units) < 3 or not units.content_count.eq(1).all():
        raise ValueError("Random baseline needs at least three singleton contents")
    rng = np.random.default_rng(seed)
    ordered = units.sort_values("group_id").iloc[rng.permutation(len(units))]
    split_order = rng.permutation(3)
    cumulative = ordered.record_count.cumsum().to_numpy()
    boundaries, start = [], 0
    for j in range(2):
        options = np.arange(start + 1, len(units) - (2-j) + 1)
        target = cumulative[-1] * ratios[split_order[:j+1]].sum()
        boundary = int(options[np.argmin(abs(cumulative[options-1] - target))])
        boundaries.append(boundary)
        start = boundary
    assignments = {}
    for k, indices in enumerate(np.split(np.arange(len(units)), boundaries)):
        assignments.update({g: SPLITS[split_order[k]] for g in ordered.iloc[indices].group_id})
    return assignments, {"method": "seeded_content_permutation_record_cuts", "seed": seed, "class_balancing": False}


def milp_assignment(units: pd.DataFrame, ratios: np.ndarray, seed: int, config: SplitConfig) -> tuple[dict, dict]:
    """Integer counts of exchangeable groups per split, with normalized L1 slack.

    Aggregating identical balance vectors is lossless: their coefficients are equal
    in every constraint. Seeded expansion restores individual atomic units. Similarity,
    sequence and historical membership never enter the solver. The node budget is
    deterministic, unlike wall-time stopping. Incumbents retain gap/status explicitly.
    """
    if len(units) < 3:
        raise ValueError("At least three atomic groups are needed for nonempty splits")
    ordered = units.sort_values("group_id").reset_index(drop=True)
    vectors = ordered[list(BALANCE_COLUMNS)].to_numpy(dtype=float)
    profiles, inverse, counts = np.unique(vectors, axis=0, return_inverse=True, return_counts=True)
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(profiles))
    profiles, counts = profiles[permutation], counts[permutation]
    inverse = np.argsort(permutation)[inverse]
    p, d = profiles.shape
    nx, nd = p*3, 3*d
    nvars = nx + 2*nd
    targets = ratios[:, None] * vectors.sum(axis=0)[None, :]
    family_weights = np.array([config.record_weight, *([config.class_weight/5]*5), config.empty_weight])
    weights = (family_weights[None, :] / (3*np.maximum(targets, 1))).ravel()
    objective = np.r_[np.zeros(nx), weights, weights]
    a = lil_matrix((p+nd+3, nvars), dtype=float)
    lower = np.zeros(p+nd+3)
    upper = np.zeros(p+nd+3)
    for i in range(p):
        a[i, i*3:i*3+3] = 1
        lower[i] = upper[i] = counts[i]
    for s in range(3):
        for k in range(d):
            row, slack = p+s*d+k, s*d+k
            a[row, np.arange(p)*3+s] = profiles[:, k]
            a[row, nx+slack] = -1
            a[row, nx+nd+slack] = 1
            lower[row] = upper[row] = targets[s, k]
        a[p+nd+s, np.arange(p)*3+s] = 1
        lower[p+nd+s], upper[p+nd+s] = 1, np.inf
    started = time.perf_counter()
    result = milp(objective, integrality=np.r_[np.ones(nx), np.zeros(2*nd)],
                  bounds=Bounds(np.zeros(nvars), np.r_[np.repeat(counts, 3), np.full(2*nd, np.inf)]),
                  constraints=LinearConstraint(a.tocsc(), lower, upper),
                  options={"node_limit": config.node_limit, "mip_rel_gap": config.mip_rel_gap, "presolve": True})
    elapsed = time.perf_counter()-started
    if result.x is None:
        raise ValueError(f"MILP has no incumbent (status {result.status}: {result.message}); no heuristic substituted")
    # SciPy may map newer HiGHS stopping codes (e.g. Solution limit reached)
    # to status=4 despite returning a feasible incumbent. Validate the actual
    # primal solution rather than interpreting a status label as feasibility.
    primal = np.asarray(result.x)
    activity = a.tocsc() @ primal
    upper_bounds = np.r_[np.repeat(counts, 3), np.full(2*nd, np.inf)]
    if (not np.isfinite(primal).all() or (primal < -1e-6).any() or (primal > upper_bounds+1e-6).any()
            or (activity < lower-1e-6).any() or (activity > upper+1e-6).any()):
        raise ValueError("MILP incumbent fails direct primal feasibility checks")
    if not np.allclose(result.x[:nx], np.rint(result.x[:nx]), atol=1e-5, rtol=0):
        raise ValueError("MILP incumbent violates integrality")
    allocation = np.rint(result.x[:nx]).astype(int).reshape(p, 3)
    if (allocation < 0).any() or not np.array_equal(allocation.sum(axis=1), counts) or (allocation.sum(axis=0) == 0).any():
        raise ValueError("MILP incumbent violates group conservation")
    assignments = {}
    for i, row in enumerate(allocation):
        members = ordered.loc[inverse == i, "group_id"].to_numpy()
        members = members[rng.permutation(len(members))]
        for s, subset in zip(SPLITS, np.split(members, np.cumsum(row)[:-1]), strict=True):
            assignments.update({g: s for g in subset})
    return assignments, {"method": "scipy.optimize.milp/HiGHS", "seed": seed, "profile_count": p,
                         "atomic_group_count": len(units), "status": int(result.status), "message": result.message,
                         "optimal_within_tolerance": bool(result.success), "unique_optimum_proven": False,
                         "primal_feasibility_verified": True,
                         "objective": float(result.fun), "mip_gap": float(result.mip_gap),
                         "dual_bound": float(result.mip_dual_bound), "node_count": int(result.mip_node_count),
                         "solve_seconds": elapsed}


def propagate(records: pd.DataFrame, groups: pd.DataFrame, assignment: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    if set(assignment) != set(groups.group_id):
        raise ValueError("Each atomic group must be assigned exactly once")
    contents = groups.copy()
    contents["new_split"] = contents.group_id.map(assignment)
    if not contents.new_split.isin(SPLITS).all():
        raise ValueError("Only train/val/test assignments are allowed")
    expanded = records[["frame_id", "content_id", "original_split"]].merge(contents[["content_id", "new_split"]], on="content_id", validate="many_to_one")
    return contents.sort_values("content_id").reset_index(drop=True), expanded.sort_values("frame_id").reset_index(drop=True)
