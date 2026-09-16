"""Posterior split diagnostics with explicit occurrence/content denominators."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from flir_pipeline.data.classes import class_name
from flir_pipeline.splitting.base import CLASS_COLUMNS, SPLITS
from flir_pipeline.splitting.construction import make_groups

BITS = {s: 1 << i for i, s in enumerate(SPLITS)}


def membership_masks(records: pd.DataFrame, ids: list[str]) -> np.ndarray:
    if not records.new_split.isin(SPLITS).all() or set(records.content_id) != set(ids):
        raise ValueError("Memberships require full content coverage and known splits")
    masks = records.groupby("content_id").new_split.agg(lambda v: sum(BITS[s] for s in set(v)))
    return masks.reindex(ids).to_numpy(dtype=np.int8)


def crosses(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Existential unequal-occurrence split relation, including historical multisets."""
    union = a | b
    return (a > 0) & (b > 0) & ((union & (union-1)) != 0)


def verify_assignments(contents: pd.DataFrame, records: pd.DataFrame,
                       manifest: pd.DataFrame, strategy: str,
                       labels: pd.Series | None = None) -> dict:
    """Fail closed on missing/extra IDs, fractured identities or incorrect propagation.

    Historical is an explicitly audited exception: new_split is null for contents
    with multiple historical memberships. Its record table preserves all occurrences;
    a membership set never masquerades as a single train/val/test value.
    """
    if strategy not in ("historical", "random_content", "cluster_aware"):
        raise ValueError("Unknown strategy")
    if len(contents) != manifest.content_id.nunique() or not contents.content_id.is_unique or set(contents.content_id) != set(manifest.content_id):
        raise ValueError("Missing, extra or duplicated content assignments")
    if len(records) != len(manifest) or not records.frame_id.is_unique or set(records.frame_id) != set(manifest.frame_id):
        raise ValueError("Missing, extra or duplicated record assignments")
    joined = records.set_index("frame_id").sort_index()
    expected = manifest.set_index("frame_id").sort_index()
    for column in ("content_id", "original_split"):
        if not joined[column].equals(expected[column]):
            raise ValueError("Occurrence lineage changed")
    if not records.new_split.isin(SPLITS).all() or set(records.new_split) != set(SPLITS):
        raise ValueError("Splits must be allowed and nonempty")
    mask = membership_masks(records, contents.content_id.tolist())
    overlap = int(((mask & (mask-1)) != 0).sum())
    if strategy == "historical":
        if not records.new_split.equals(records.original_split):
            raise ValueError("Historical membership must be preserved exactly")
        memberships = records.groupby("content_id").new_split.agg(lambda v: sorted(set(v)))
        for row in contents.itertuples():
            values = memberships[row.content_id]
            if json.loads(row.split_membership_set) != values or (len(values) == 1 and row.new_split != values[0]) or (len(values) > 1 and not pd.isna(row.new_split)):
                raise ValueError("Historical content memberships were flattened")
        fractures = None
    else:
        if overlap or not contents.new_split.isin(SPLITS).all():
            raise ValueError("Exact duplicate cross-split invariant violated")
        mapping = contents.set_index("content_id").new_split
        if not np.array_equal(records.content_id.map(mapping), records.new_split):
            raise ValueError("Content to occurrence propagation failed")
        if strategy == "cluster_aware" and labels is None:
            raise ValueError("Cluster-aware verification needs source cluster labels")
        groups = make_groups(contents.content_id.tolist(), labels if strategy == "cluster_aware" else None)
        actual = contents.set_index("content_id").sort_index()
        for name in ("group_id", "group_type", "cluster_id"):
            pd.testing.assert_series_equal(actual[name], groups.set_index("content_id")[name], check_names=False, check_dtype=False)
        fractures = int((contents.groupby("group_id").new_split.nunique() > 1).sum())
        if fractures:
            raise ValueError("Indivisible cluster/group fracture invariant violated")
    return {"quality_valid": True, "content_count": len(contents), "record_count": len(records),
            "exact_duplicate_cross_split_count": overlap,
            "cluster_fracture_count": fractures if strategy == "cluster_aware" else None,
            "historical_exception": strategy == "historical"}


def balance_metrics(records: pd.DataFrame, statistics: pd.DataFrame,
                    ratios: np.ndarray) -> tuple[dict, pd.DataFrame]:
    data = records[["frame_id", "new_split"]].merge(statistics, on="frame_id", validate="one_to_one")
    global_instances = statistics[list(CLASS_COLUMNS)].sum().to_numpy(dtype=float)
    global_proportions = global_instances / max(global_instances.sum(), 1)
    rows, sizes = [], []
    for i, split in enumerate(SPLITS):
        group = data.loc[data.new_split == split]
        counts = group[list(CLASS_COLUMNS)].sum().to_numpy(dtype=int)
        total = int(counts.sum())
        target = float(len(data)*ratios[i])
        sizes.append({"split": split, "records": len(group), "contents": group.content_id.nunique(),
                      "target_records": target, "absolute_record_deviation": abs(len(group)-target),
                      "relative_record_deviation": abs(len(group)-target)/target,
                      "empty_annotations": int(group.empty_annotation_count.sum())})
        for k, count in enumerate(counts):
            share = count/total if total else 0.
            rows.append({"split": split, "class_id": k, "class_name": class_name(k),
                         "images": int((group[CLASS_COLUMNS[k]] > 0).sum()),
                         "contents": group.loc[group[CLASS_COLUMNS[k]] > 0, "content_id"].nunique(),
                         "instances": int(count), "instance_percentage": 100*share,
                         "global_instance_percentage": float(100*global_proportions[k]),
                         "absolute_percentage_point_deviation": float(100*abs(share-global_proportions[k])),
                         "target_instances": float(global_instances[k]*ratios[i]),
                         "empty_annotations": int(group.empty_annotation_count.sum())})
    table = pd.DataFrame(rows)
    summary = {"sizes": sizes, "max_relative_record_deviation": max(v["relative_record_deviation"] for v in sizes),
               "mean_absolute_class_distribution_deviation_pp": float(table.absolute_percentage_point_deviation.mean()),
               "all_classes_covered": bool((table.instances > 0).all()),
               "heavy_machinery_counts": {s: int(table.loc[(table.split == s) & (table.class_id == 4), "instances"].iloc[0]) for s in SPLITS},
               "background_is_class": False}
    return summary, table


def residual_similarity(matrix: np.ndarray, neighbors: pd.DataFrame, thresholds: pd.DataFrame,
                        masks: np.ndarray, content_ids: list[str]) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Full-matrix cross-split NN; top-20 truncation is never used as an NN substitute.

    All statistics give one vote per unique content, and unordered pairs i<j one
    vote per pair. Self-content is excluded even in historical multi-membership
    data; its exact leakage is counted separately by the identity verifier.
    """
    n = len(masks)
    if matrix.shape != (n, n) or len(content_ids) != n or not np.isfinite(matrix).all():
        raise ValueError("Invalid aligned cosine space")
    cross = crosses(masks[:, None], masks[None, :])
    np.fill_diagonal(cross, False)
    scores = np.where(cross, matrix, -np.inf)
    nn_rows = np.argmax(scores, axis=1)
    values = scores[np.arange(n), nn_rows]
    available = np.isfinite(values)
    sample = values[available]
    statistics = {"count": int(available.sum()), "unavailable_count": int((~available).sum()),
                  "mean": float(sample.mean()) if len(sample) else None}
    for name, q in (("median", .5), ("Q1", .25), ("Q3", .75), ("p90", .9), ("p95", .95), ("p99", .99), ("max", 1.)):
        statistics[name] = float(np.quantile(sample, q)) if len(sample) else None
    topk = {}
    for k in (1, 5, 10, 20):
        selected = neighbors.loc[neighbors.neighbor_rank <= k]
        a, b = selected.query_row.to_numpy(int), selected.neighbor_row.to_numpy(int)
        count = int(cross[a, b].sum())
        topk[str(k)] = {"cross_split_edges": count, "edge_count": len(selected),
                       "fraction": count/len(selected) if len(selected) else None,
                       "effective_k": min(k, n-1, int(neighbors.neighbor_rank.max())) if len(neighbors) else 0}
    a, b = np.triu_indices(n, 1)
    pair_values, pair_cross = matrix[a, b], cross[a, b]
    quantile_rows = []
    for row in thresholds.itertuples():
        # Tables serialize float32 thresholds as decimal doubles. Cast back to
        # the existing matrix dtype so an inclusive boundary retains equal ties.
        threshold = np.asarray(row.threshold_cosine, dtype=matrix.dtype).item()
        selected = pair_values >= threshold
        total, count = int(selected.sum()), int((selected & pair_cross).sum())
        if total != row.pair_count:
            raise ValueError("Existing quantile cohort differs from aligned matrix")
        quantile_rows.append({"quantile": float(row.quantile), "top_percentage": float(row.top_percentage),
                              "threshold_cosine": threshold, "pair_count": total, "cross_split_count": count,
                              "cross_split_fraction": count/total if total else None})
    nn = pd.DataFrame({"content_id": content_ids,
                       "neighbor_content_id": [content_ids[r] if ok else None for r, ok in zip(nn_rows, available, strict=True)],
                       "cross_split_nn_similarity": np.where(available, values, np.nan)})
    return {"cross_split_nn": statistics, "topk": topk}, pd.DataFrame(quantile_rows), nn


def temporal_metrics(provenance: pd.DataFrame, masks: np.ndarray) -> tuple[dict, pd.DataFrame]:
    known = provenance.sequence_provenance_valid.to_numpy(bool)
    indices = provenance.frame_index.to_numpy(dtype=float, na_value=np.nan)
    sequences = provenance.sequence_key.to_numpy()
    a, b = np.triu_indices(len(masks), 1)
    same = known[a] & known[b] & (sequences[a] == sequences[b]) & np.isfinite(indices[a]) & np.isfinite(indices[b])
    delta = abs(indices[a]-indices[b])
    cross = crosses(masks[a], masks[b])
    rows = []
    for window in (1, 5, 10, 25):
        eligible = same & (delta <= window)
        denominator, count = int(eligible.sum()), int((eligible & cross).sum())
        rows.append({"frame_delta_max": window, "pair_count": denominator, "cross_split_count": count,
                     "fraction": count/denominator if denominator else None})
    counts = {1: 0, 2: 0, 3: 0}
    for sequence in sorted(set(sequences[known])):
        union = int(np.bitwise_or.reduce(masks[known & (sequences == sequence)]))
        counts[union.bit_count()] += 1
    total = sum(counts.values())
    return {"provenance": "filename_heuristic; indices are not seconds", "sequence_count": total,
            "unknown_sequence_content_count": int((~known).sum()),
            "unknown_frame_index_content_count": int((~np.isfinite(indices)).sum()),
            "sequence_split_counts": {str(k): v for k, v in counts.items()},
            "sequence_split_fractions": {str(k): v/total if total else None for k, v in counts.items()}}, pd.DataFrame(rows)


def cluster_fracture(labels: np.ndarray, masks: np.ndarray) -> dict:
    """Noise is never one giant cluster; exact historical memberships remain sets."""
    clusters = sorted(set(labels)-{-1})
    fractured = sum(int(np.bitwise_or.reduce(masks[labels == c])).bit_count() > 1 for c in clusters)
    return {"cluster_count": len(clusters), "fractured_cluster_count": fractured,
            "fracture_rate": fractured/len(clusters) if clusters else None}
