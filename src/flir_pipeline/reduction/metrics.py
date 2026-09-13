"""Exact rank-based preservation metrics with explicit distance and tie conventions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

from flir_pipeline.similarity.comparison import compare_neighbor_spaces
from flir_pipeline.similarity.cosine import distribution_summary


def neighbor_order(distances: np.ndarray, content_ids: list[str]) -> np.ndarray:
    """Return all non-self neighbors, ascending distance then ascending content ID."""
    n = len(content_ids)
    if distances.shape != (n, n) or not np.isfinite(distances).all() or len(set(content_ids)) != n:
        raise ValueError("Distance matrix must be finite and aligned to unique content IDs")
    columns = np.argsort(np.asarray(content_ids), kind="stable")
    scores = distances[:, columns].copy()
    scores[np.arange(n), np.argsort(columns)] = np.inf
    return columns[np.argsort(scores, axis=1, kind="stable")[:, :n-1]].astype(np.int32)


def inverse_ranks(order: np.ndarray) -> np.ndarray:
    """Rank 1 is the first other content; self has rank zero and is never evaluated."""
    n = len(order)
    if order.shape != (n, n-1) or not np.array_equal(np.sort(np.column_stack([order, np.arange(n)]), axis=1), np.broadcast_to(np.arange(n), (n, n))):
        raise ValueError("Each ranking must contain every non-self row exactly once")
    ranks = np.zeros((n, n), dtype=np.int32)
    ranks[np.arange(n)[:, None], order] = np.arange(1, n)
    return ranks


def trustworthiness_continuity(original_order: np.ndarray, reduced_order: np.ndarray, ks: tuple[int, ...]) -> dict:
    """Standard intrusion/omission penalties, not an overlap approximation.

    T penalizes reduced neighbors by their original ranks beyond k.
    C penalizes original neighbors by their reduced ranks beyond k.
    Both use 2/[N*k*(2*N-3*k-1)], valid here for 0 < k < N/2.
    Synthetic tests compare T to sklearn and C to the exact reversed-space T.
    """
    n = len(original_order)
    if reduced_order.shape != original_order.shape or not ks or any(type(k) is not int or not 0 < k < n/2 for k in ks):
        raise ValueError("Aligned rankings and 0 < k < N/2 are required")
    original_ranks, reduced_ranks = inverse_ranks(original_order), inverse_ranks(reduced_order)
    rows = np.arange(n)[:, None]
    metrics = {}
    for k in ks:
        factor = 2/(n*k*(2*n-3*k-1))
        intrusions = np.maximum(original_ranks[rows, reduced_order[:, :k]]-k, 0).sum(dtype=np.int64)
        omissions = np.maximum(reduced_ranks[rows, original_order[:, :k]]-k, 0).sum(dtype=np.int64)
        metrics[f"trustworthiness@{k}"] = float(1-factor*intrusions)
        metrics[f"continuity@{k}"] = float(1-factor*omissions)
    return metrics


def neighborhood_table(order: np.ndarray, ids: list[str], distances: np.ndarray, k: int) -> pd.DataFrame:
    n = len(ids)
    rows, neighbors = np.repeat(np.arange(n), k), order[:, :k].ravel()
    labels = np.asarray(ids)
    return pd.DataFrame({"query_row": rows.astype(np.int32), "neighbor_row": neighbors,
                         "query_content_id": labels[rows], "neighbor_content_id": labels[neighbors],
                         "neighbor_rank": np.tile(np.arange(1, k+1, dtype=np.int32), n),
                         "distance": distances[rows, neighbors]})


@dataclass
class EvaluationReference:
    """Reusable original-space ranks and deterministic pairs, separate from fitting."""

    content_ids: list[str]
    original_order: np.ndarray
    original_neighbors: pd.DataFrame
    sample_a: np.ndarray
    sample_b: np.ndarray
    original_sample_distances: np.ndarray
    sample_seed: int

    @classmethod
    def from_similarity(cls, cosine: np.ndarray, ids: list[str], neighbors: pd.DataFrame,
                        sample_size: int = 100000, seed: int = 0) -> EvaluationReference:
        distance = 1-cosine.astype(np.float64)
        order = neighbor_order(distance, ids)
        k = int(neighbors.neighbor_rank.max())
        expected = neighborhood_table(order, ids, distance, k)
        columns = ["query_content_id", "neighbor_rank", "neighbor_content_id"]
        actual = neighbors[columns].sort_values(columns[:2]).reset_index(drop=True)
        expected_sorted = expected[columns].sort_values(columns[:2]).reset_index(drop=True)
        if not actual.equals(expected_sorted):
            raise ValueError("Existing cosine neighbors do not match the original distance ranks")
        # Sample pairs in canonical content-ID order, shared across row permutations.
        index = np.argsort(np.asarray(ids), kind="stable")
        a, b = np.triu_indices(len(ids), 1)
        chosen = np.sort(np.random.default_rng(seed).choice(len(a), min(sample_size, len(a)), replace=False))
        a, b = index[a[chosen]], index[b[chosen]]
        return cls(ids, order, actual, a, b, distance[a, b], seed)


def evaluate_coordinates(coordinates: np.ndarray, reference: EvaluationReference,
                         ks: tuple[int, ...]) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Evaluate against full original L2 cosine ranks and saved top-k neighbors."""
    if coordinates.ndim != 2 or len(coordinates) != len(reference.content_ids) or not np.isfinite(coordinates).all():
        raise ValueError("Coordinates must be finite and aligned to the evaluation reference")
    if max(ks) > reference.original_neighbors.neighbor_rank.max():
        raise ValueError("Original neighbor table does not cover requested evaluation k")
    distances = squareform(pdist(coordinates.astype(np.float64), metric="euclidean"))
    order = neighbor_order(distances, reference.content_ids)
    metrics = trustworthiness_continuity(reference.original_order, order, ks)
    reduced = neighborhood_table(order, reference.content_ids, distances, max(ks))
    content_metrics, summaries = compare_neighbor_spaces(reference.original_neighbors, reduced, ks)
    for row in summaries.itertuples():
        for key in ("mean_jaccard", "median_jaccard", "Q1", "Q3"):
            metrics[f"jaccard@{row.k}_{key}"] = float(getattr(row, key))
    sampled = distances[reference.sample_a, reference.sample_b]
    rho = spearmanr(reference.original_sample_distances, sampled).statistic if np.ptp(sampled) and np.ptp(reference.original_sample_distances) else None
    metrics.update({"spearman_distance": float(rho) if rho is not None and np.isfinite(rho) else None,
                    "distance_pair_count": len(sampled), "distance_sample_seed": reference.sample_seed,
                    "original_distance": "1 - stored cosine of original L2 vectors",
                    "reduced_distance": "euclidean", "tie_break": "content_id_ascending"})
    return metrics, reduced, content_metrics


def summarize_stability(frames: list[pd.DataFrame], seeds: list[int], ks: tuple[int, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare all seed pairs using neighborhood sets, invariant to rigid geometry."""
    if len(frames) != len(seeds) or len(set(seeds)) != len(seeds) or len(seeds) < 2:
        raise ValueError("Seed stability requires at least two distinct aligned runs")
    from itertools import combinations

    pieces = []
    for a, b in combinations(range(len(seeds)), 2):
        contents, _ = compare_neighbor_spaces(frames[a], frames[b], ks)
        pieces.append(contents.assign(seed_left=seeds[a], seed_right=seeds[b]))
    contents = pd.concat(pieces, ignore_index=True)
    rows = [{"k": k, "seed_pairs": len(pieces), **distribution_summary(group.jaccard.to_numpy())} for k, group in contents.groupby("k", sort=True)]
    return contents, pd.DataFrame(rows)
