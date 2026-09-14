"""Vector-only CPU adapters; posterior provenance cannot enter a fit."""

from __future__ import annotations

import importlib.metadata
import time
import warnings
from dataclasses import dataclass

import numpy as np

from flir_pipeline.clustering.base import ClusteringConfig


def implementation_versions() -> dict:
    return {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn")}


def k_distances(distances: np.ndarray, min_samples: int) -> np.ndarray:
    """Distance to the (min_samples-1)-th OTHER point: sklearn includes self.

    Excluding self explicitly also handles zero-distance distinct contents without
    assuming that a nearest-neighbor backend places self first in a tied row.
    """
    n = len(distances)
    if distances.shape != (n, n) or not 2 <= min_samples <= n or not np.isfinite(distances).all() or (distances < 0).any():
        raise ValueError("Invalid distances or min_samples")
    values = distances.copy()
    np.fill_diagonal(values, np.inf)
    return np.partition(values, min_samples-2, axis=1)[:, min_samples-2]


def effective_parameters(config: ClusteringConfig, distances: np.ndarray) -> dict:
    p = dict(config.hyperparameters)
    n = len(distances)
    if p["min_samples"] > n or p.get("min_cluster_size", 2) > n:
        raise ValueError("Clustering parameters exceed the population")
    if config.algorithm == "dbscan":
        q = p.pop("eps_quantile")
        eps = float(np.quantile(k_distances(distances, p["min_samples"]), q, method="linear"))
        if not np.isfinite(eps) or eps <= 0:
            raise ValueError("Nonpositive k-distance epsilon; record/review rather than silently replacing it")
        p["eps"] = eps
    return {**p, "metric": config.metric}


@dataclass
class ClusteringResult:
    labels: np.ndarray
    probabilities: np.ndarray | None
    metadata: dict
    diagnostics: dict[str, np.ndarray]


def fit_clustering(values: np.ndarray, content_ids: list[str], config: ClusteringConfig,
                   parameters: dict) -> ClusteringResult:
    """Sort by content identity for stable ordering; map output back to embedding rows."""
    from sklearn.cluster import DBSCAN, HDBSCAN, OPTICS
    from threadpoolctl import threadpool_limits

    x = np.asarray(values)
    if x.ndim != 2 or not np.isfinite(x).all() or len(x) != len(content_ids) or len(set(content_ids)) != len(x):
        raise ValueError("Clustering requires finite vectors aligned to unique content IDs")
    order = np.argsort(np.asarray(content_ids), kind="stable")
    inverse = np.argsort(order)
    p = dict(parameters)
    if config.algorithm == "dbscan":
        model = DBSCAN(**p, algorithm="brute", n_jobs=1)
    elif config.algorithm == "optics":
        p["max_eps"] = np.inf
        model = OPTICS(**p, algorithm="brute", n_jobs=1, memory=None)
    else:
        model = HDBSCAN(**p, algorithm="brute", n_jobs=1, copy=True, store_centers=None)
    started = time.perf_counter()
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(x[order].astype(np.float64, copy=True))
    labels = np.asarray(model.labels_, dtype=np.int32)[inverse]
    if labels.shape != (len(x),) or np.any(labels < -1):
        raise ValueError("Unexpected missing/non-finite labels from a finite input")
    probabilities = getattr(model, "probabilities_", None)
    if probabilities is not None:
        probabilities = np.asarray(probabilities, dtype=np.float64)[inverse]
    diagnostics = {}
    if config.algorithm == "optics":
        # Reachability/core arrays are mapped to content_index; infinity is a valid
        # disconnected/start marker, not an invalid feature or probability.
        diagnostics = {"reachability": model.reachability_[inverse], "core_distances": model.core_distances_[inverse],
                       "ordering_rows": order[model.ordering_]}
    return ClusteringResult(labels, probabilities, {
        "fit_seconds": time.perf_counter()-started, "effective_parameters": parameters,
        "runtime_parameters": {"search_algorithm": "brute", "n_jobs": 1, "fit_dtype": "float64", "input_order": "content_id_ascending"},
        "warnings": [str(w.message) for w in caught],
        "membership_probabilities_available": probabilities is not None,
        "outlier_scores_available": hasattr(model, "outlier_scores_"),
        "cluster_persistence_available": hasattr(model, "cluster_persistence_"),
    }, diagnostics)
