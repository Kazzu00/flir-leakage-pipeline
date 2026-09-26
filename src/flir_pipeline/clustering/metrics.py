"""Exact original-space evaluation, posterior provenance and noise-aware agreement."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd


def validate_labels(labels: np.ndarray, n: int) -> None:
    if labels.shape != (n,) or not np.issubdtype(labels.dtype, np.integer) or np.any(labels < -1):
        raise ValueError("Cluster labels must be aligned integers >= -1; noise stays -1")


def silhouette_without_noise(distances: np.ndarray, labels: np.ndarray) -> float | None:
    from sklearn.metrics import silhouette_score

    validate_labels(labels, len(distances))
    mask = labels >= 0
    groups = np.unique(labels[mask])
    if not 2 <= len(groups) < mask.sum():
        return None
    return float(silhouette_score(distances[np.ix_(mask, mask)], labels[mask], metric="precomputed"))


@dataclass
class EvaluationContext:
    content_ids: list[str]
    original_distances: np.ndarray
    cosine: np.ndarray
    neighbors: pd.DataFrame
    provenance: pd.DataFrame
    temporal_pairs: dict[int, tuple[np.ndarray, np.ndarray]]

    @classmethod
    def create(cls, content_ids, original_distances, cosine, neighbors, provenance):
        n = len(content_ids)
        if len(set(content_ids)) != n or provenance.content_id.tolist() != content_ids:
            raise ValueError("Posterior provenance must align with unique contents")
        if original_distances.shape != (n, n) or cosine.shape != (n, n):
            raise ValueError("Original-space evaluation matrices are misaligned")
        if "temporal_source" in provenance and provenance.temporal_source.eq("sampled_video_grid").all():
            # Source membership is not scene membership. The historical temporal
            # recall has no valid denominator here; keep it unavailable explicitly.
            return cls(list(content_ids), original_distances, cosine, neighbors, provenance, {})
        a, b = np.triu_indices(n, 1)
        seq = provenance.sequence_key.to_numpy()
        valid = provenance.sequence_provenance_valid.to_numpy(dtype=bool)
        indices = provenance.frame_index.to_numpy(dtype=float, na_value=np.nan)
        delta = np.abs(indices[a]-indices[b])
        same = valid[a] & valid[b] & (seq[a] == seq[b]) & np.isfinite(delta)
        pairs = {k: (a[same & (delta <= k)], b[same & (delta <= k)]) for k in (1, 5, 10)}
        return cls(list(content_ids), original_distances, cosine, neighbors, provenance, pairs)


def cluster_summary(labels: np.ndarray, context: EvaluationContext) -> pd.DataFrame:
    """Medoids minimize original Euclidean distance; cosine summaries use i<j pairs."""
    validate_labels(labels, len(context.content_ids))
    ids = np.asarray(context.content_ids)
    rows = []
    for cluster_id in sorted(set(labels)-{-1}):
        members = np.flatnonzero(labels == cluster_id)
        original = context.original_distances[np.ix_(members, members)]
        costs = original.sum(axis=1)
        tied = members[costs == costs.min()]
        medoid = min(tied, key=lambda i: ids[i])
        a, b = np.triu_indices(len(members), 1)
        pairs = context.cosine[members[a], members[b]].astype(np.float64)
        provenance = context.provenance.iloc[members]
        video = "temporal_source" in provenance and provenance.temporal_source.eq("sampled_video_grid").all()
        known = pd.Series(False, index=provenance.index) if video else provenance.sequence_provenance_valid.astype(bool)
        counts = pd.Series(dtype=int) if video else provenance.loc[known, "sequence_key"].value_counts()
        fractions = counts.to_numpy()/counts.sum() if len(counts) else np.asarray([])
        split_mask = 0 if video else int(np.bitwise_or.reduce(provenance.split_mask.to_numpy(dtype=np.int32)))
        split_names = [name for name, bit in (("train", 1), ("val", 2), ("test", 4)) if split_mask & bit]
        rows.append({"cluster_id": int(cluster_id), "n_members": len(members), "medoid_content_id": ids[medoid],
                     "intra_pair_count": len(pairs), "mean_intra_cosine": float(pairs.mean()) if len(pairs) else None,
                     "median_intra_cosine": float(np.median(pairs)) if len(pairs) else None,
                     "Q1_intra_cosine": float(np.quantile(pairs, .25)) if len(pairs) else None,
                     "Q3_intra_cosine": float(np.quantile(pairs, .75)) if len(pairs) else None,
                     "known_sequence_members": int(known.sum()), "sequence_coverage": None if video else float(known.mean()), "sequence_count": None if video else len(counts),
                     "dominant_sequence_fraction": float(fractions.max()) if len(counts) else None,
                     "sequence_entropy_bits": float(-np.sum(fractions*np.log2(fractions))) if len(counts) else None,
                     "historical_split_memberships": None if video else json.dumps(split_names), "historical_split_count": None if video else len(split_names),
                     "historical_split_provenance_complete": False if video else bool(provenance.split_membership_complete.all())})
    columns = ["cluster_id", "n_members", "medoid_content_id", "intra_pair_count", "mean_intra_cosine", "median_intra_cosine",
               "Q1_intra_cosine", "Q3_intra_cosine", "known_sequence_members", "sequence_coverage", "sequence_count",
               "dominant_sequence_fraction", "sequence_entropy_bits", "historical_split_memberships", "historical_split_count",
               "historical_split_provenance_complete"]
    return pd.DataFrame(rows, columns=columns)


def evaluate_clustering(labels: np.ndarray, context: EvaluationContext,
                        clustering_distances: np.ndarray) -> tuple[dict, pd.DataFrame]:
    n = len(context.content_ids)
    validate_labels(labels, n)
    table = cluster_summary(labels, context)
    sizes = table.n_members.to_numpy(dtype=int)
    count = len(sizes)
    clustered = int(np.sum(labels >= 0))
    noise_fraction = (n-clustered)/n
    largest = int(sizes.max())/n if count else 0.
    result = {"total_points": n, "clustered_points": clustered, "noise_points": n-clustered,
              "noise_fraction": noise_fraction, "n_clusters_excluding_noise": count,
              "largest_cluster_fraction": largest, "singleton_cluster_count": int(np.sum(sizes == 1)),
              "all_noise": clustered == 0, "single_cluster": count <= 1,
              "nearly_all_noise": noise_fraction >= .95, "dominant_cluster": largest >= .90,
              "silhouette_original_space": silhouette_without_noise(context.original_distances, labels),
              "silhouette_clustering_space": silhouette_without_noise(clustering_distances, labels),
              "silhouette_population": clustered, "visual_query_coverage": clustered/n}
    for name, reducer in (("min", np.min), ("q1", lambda x: np.quantile(x, .25)), ("median", np.median),
                          ("mean", np.mean), ("q3", lambda x: np.quantile(x, .75)), ("max", np.max)):
        result[f"cluster_size_{name}"] = float(reducer(sizes)) if count else None
    visual = table.loc[table.intra_pair_count > 0]
    result["weighted_mean_intra_cluster_similarity"] = float(np.average(visual.mean_intra_cosine, weights=visual.intra_pair_count)) if len(visual) else None
    result["member_weighted_intra_cluster_similarity"] = float(np.average(visual.mean_intra_cosine, weights=visual.n_members)) if len(visual) else None
    result["median_cluster_intra_similarity"] = float(visual.mean_intra_cosine.median()) if len(visual) else None
    known = table.loc[table.known_sequence_members > 0]
    result["weighted_dominant_sequence_fraction"] = float(np.average(known.dominant_sequence_fraction, weights=known.known_sequence_members)) if len(known) else None
    result["weighted_sequence_entropy_bits"] = float(np.average(known.sequence_entropy_bits, weights=known.known_sequence_members)) if len(known) else None
    result["clustered_sequence_coverage"] = float(table.known_sequence_members.sum()/clustered) if clustered else None
    for k, (a, b) in context.temporal_pairs.items():
        together = int(np.sum((labels[a] >= 0) & (labels[a] == labels[b])))
        result.update({f"temporal_recall@{k}": together/len(a) if len(a) else None,
                       f"temporal_pairs@{k}": len(a), f"temporal_retained_pairs@{k}": together})
    for k in (5, 10, 20):
        neighbors = context.neighbors.loc[context.neighbors.neighbor_rank <= k]
        if len(neighbors) != n*k:
            raise ValueError("Original neighbor table must cover k=5/10/20 for all contents")
        a, b = neighbors.query_row.to_numpy(), neighbors.neighbor_row.to_numpy()
        numerator = int(np.sum((labels[a] >= 0) & (labels[a] == labels[b])))
        denominator = clustered*k
        result.update({f"visual_neighbor_coherence@{k}": numerator/denominator if denominator else None,
                       f"visual_neighbor_retention_all@{k}": numerator/(n*k),
                       f"visual_retained_edges@{k}": numerator, f"visual_eligible_edges@{k}": denominator})
    result["historical_multisplit_clusters"] = int((table.historical_split_count > 1).sum())
    result["historical_unknown_clusters"] = int((table.historical_split_count == 0).sum())
    for split in ("train", "val", "test"):
        result[f"historical_{split}_only_clusters"] = int(table.historical_split_memberships.eq(json.dumps([split])).sum())
    if "temporal_source" in context.provenance and context.provenance.temporal_source.eq("sampled_video_grid").all():
        result.update(temporal_provenance_mode="sampled_video_grid",
                      temporal_evaluation="sequence identity unknown; historical recall unavailable",
                      historical_split_evaluation="unavailable", clustered_sequence_coverage=None)
        for k in (1, 5, 10):
            result.update({f"temporal_recall@{k}": None, f"temporal_pairs@{k}": None,
                           f"temporal_retained_pairs@{k}": None})
        for key in list(result):
            if key.startswith("historical_") and key != "historical_split_evaluation":
                result[key] = None
    return result, table


def assignment_agreement(left: np.ndarray, right: np.ndarray) -> dict:
    """Keep the noise-as-one-label result distinct from common non-noise agreement."""
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

    validate_labels(left, len(left))
    validate_labels(right, len(left))
    result = {}
    for policy, mask in (("all_points", np.ones(len(left), dtype=bool)), ("common_clustered", (left >= 0) & (right >= 0))):
        n = int(mask.sum())
        a, b = left[mask], right[mask]
        groups_a, groups_b = len(np.unique(a)), len(np.unique(b))
        result.update({f"{policy}_n": n, f"{policy}_coverage": n/len(left),
                       f"{policy}_trivial": groups_a < 2 or groups_b < 2,
                       f"{policy}_ari": float(adjusted_rand_score(a, b)) if n >= 2 else None,
                       f"{policy}_ami": float(adjusted_mutual_info_score(a, b, average_method="arithmetic")) if n >= 2 else None})
    result["shared_noise_fraction"] = float(np.mean((left == -1) & (right == -1)))
    return result


def summarize_agreements(rows: list[dict]) -> dict:
    if not rows:
        return {"comparison_count": 0}
    frame = pd.DataFrame(rows)
    result = {"comparison_count": len(rows)}
    for policy in ("all_points", "common_clustered"):
        result[f"{policy}_all_nontrivial"] = bool((~frame[f"{policy}_trivial"]).all())
        result[f"{policy}_min_coverage"] = float(frame[f"{policy}_coverage"].min())
        for metric in ("ari", "ami"):
            values = pd.to_numeric(frame[f"{policy}_{metric}"], errors="coerce")
            valid = bool(values.notna().all())
            result[f"{policy}_{metric}_mean"] = float(values.mean()) if valid else None
            result[f"{policy}_{metric}_min"] = float(values.min()) if valid else None
    return result
