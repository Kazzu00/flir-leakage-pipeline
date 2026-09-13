"""Cosine and deterministic neighbors using only unique-content vectors and IDs."""

from __future__ import annotations

import numpy as np
import pandas as pd


def matrix_quality(matrix: np.ndarray, n: int, atol: float = 1e-5) -> dict:
    """Inspect an unmodified dot-product matrix; allow floating-point roundoff."""
    shape = matrix.ndim == 2 and matrix.shape == (n, n) and n > 1
    finite = bool(np.isfinite(matrix).all())
    return {
        "matrix_shape_valid": shape,
        "finite": finite,
        "symmetric": bool(shape and np.allclose(matrix, matrix.T, atol=atol, rtol=0)),
        "diagonal_near_one": bool(shape and np.allclose(np.diag(matrix), 1, atol=atol, rtol=0)),
        "cosine_range_valid": bool(finite and matrix.size and matrix.min() >= -1-atol and matrix.max() <= 1+atol),
    }


def compute_cosine_similarity(embeddings_l2: np.ndarray, atol: float = 1e-5) -> np.ndarray:
    """Return float32 dot products, without renormalization, clipping or diagonal edits.

    Inputs contain one row per content and no labels, splits or temporal metadata.
    Reject invalid norms instead of silently changing the supplied feature space.
    """
    x = np.asarray(embeddings_l2)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1 or x.dtype != np.float32:
        raise ValueError("Expected at least two float32 embedding rows")
    if not np.isfinite(x).all() or not np.allclose(np.linalg.norm(x, axis=1), 1, atol=atol, rtol=0):
        raise ValueError("Embeddings must already be finite and L2-normalized")
    matrix = x @ x.T
    if not all(matrix_quality(matrix, len(x), atol).values()):
        raise ValueError("Cosine matrix failed numerical verification")
    return matrix


def compute_topk_neighbors(matrix: np.ndarray, content_ids: list[str], top_k: int = 20) -> pd.DataFrame:
    """Rank by decreasing cosine, breaking exact ties by lexicographic content_id.

    Self exclusion uses row identity, not a cosine threshold: distinct contents
    with equal embeddings remain eligible. Full stable sorting is adequate here.
    """
    n = len(content_ids)
    if matrix.shape != (n, n) or not np.isfinite(matrix).all():
        raise ValueError("Matrix and content IDs must be aligned and finite")
    if len(set(content_ids)) != n or any(not isinstance(v, str) or not v for v in content_ids):
        raise ValueError("content_id must be unique nonempty strings")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k < n:
        raise ValueError("top_k must be an integer from 1 to N-1")
    ids = np.asarray(content_ids)
    id_order = np.argsort(ids, kind="stable")
    scores = matrix[:, id_order].copy()
    scores[np.arange(n), np.argsort(id_order)] = -np.inf
    order = np.argsort(-scores, axis=1, kind="stable")[:, :top_k]
    neighbors = id_order[order].ravel()
    queries = np.repeat(np.arange(n), top_k)
    return pd.DataFrame({
        "query_row": queries.astype(np.int32),
        "neighbor_row": neighbors.astype(np.int32),
        "query_content_id": ids[queries],
        "neighbor_rank": np.tile(np.arange(1, top_k+1, dtype=np.int32), n),
        "neighbor_content_id": ids[neighbors],
        "cosine_similarity": matrix[queries, neighbors],
    })


def distribution_summary(values: np.ndarray) -> dict:
    """Describe a finite enumerated population: std ddof=0, linear quantiles.

    Empty strata retain count=0 with null statistics. Enumerated correlated pairs
    are not independent samples; no confidence intervals or significance claims.
    """
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError("Distribution values must be a finite one-dimensional array")
    levels = {"Q1": .25, "median": .5, "Q3": .75, "p90": .9, "p95": .95,
              "p97_5": .975, "p99": .99, "p99_5": .995, "p99_9": .999}
    if not len(x):
        return {"count": 0, **{key: None for key in ["mean", "std", "min", "max", *levels]}}
    return {"count": len(x), "mean": float(x.mean()), "std": float(x.std(ddof=0)),
            "min": float(x.min()), "max": float(x.max()),
            **{key: float(np.quantile(x, q, method="linear")) for key, q in levels.items()}}


def summarize_topk(neighbors: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep per-content and global rank-1/top-5/10/20 descriptions separate."""
    sizes = [k for k in (1, 5, 10, 20) if k <= neighbors.neighbor_rank.max()]
    per_content = neighbors[["query_content_id"]].drop_duplicates().set_index("query_content_id")
    global_rows = []
    for k in sizes:
        metric = "rank1_similarity" if k == 1 else f"top{k}_mean_similarity"
        per_content[metric] = neighbors.loc[neighbors.neighbor_rank <= k].groupby("query_content_id").cosine_similarity.mean()
        global_rows.append({"metric": metric, **distribution_summary(per_content[metric].to_numpy())})
    return per_content.reset_index(), pd.DataFrame(global_rows)
