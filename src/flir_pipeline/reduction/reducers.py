"""CPU adapters with explicit native preprocessing and controlled random state."""

from __future__ import annotations

import hashlib
import importlib.metadata
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from flir_pipeline.reduction.base import ReductionConfig


def implementation_versions(method: str) -> dict:
    names = ["numpy", "scipy", "scikit-learn"]
    if method == "pacmap":
        names += ["pacmap", "numba", "llvmlite", "faiss-cpu"]
    try:
        return {name: importlib.metadata.version(name) for name in names}
    except importlib.metadata.PackageNotFoundError as error:
        raise ImportError("Install the reduction extra: uv sync --extra reduction") from error


@dataclass
class ReductionResult:
    coordinates: np.ndarray
    metadata: dict


class DimensionalityReducer(ABC):
    """Accept only content-level visual vectors, never posterior provenance."""

    def __init__(self, config: ReductionConfig):
        self.config = config

    def fit_transform(self, embeddings_l2: np.ndarray) -> ReductionResult:
        x = np.asarray(embeddings_l2)
        if x.ndim != 2 or x.dtype != np.float32 or not np.isfinite(x).all() or not np.allclose(np.linalg.norm(x, axis=1), 1, atol=1e-5, rtol=0):
            raise ValueError("Reducer input must be the finite original float32 L2 embeddings")
        self.config.validate_population(*x.shape)
        # Protect the caller even when a third-party implementation centers in place.
        return self._fit(x.copy())

    @abstractmethod
    def _fit(self, values: np.ndarray) -> ReductionResult:
        """Return coordinates and observed effective backend parameters."""


class TSNEReducer(DimensionalityReducer):
    def _fit(self, values: np.ndarray) -> ReductionResult:
        from sklearn.manifold import TSNE
        from threadpoolctl import threadpool_limits

        model = TSNE(n_components=self.config.output_dimension, random_state=self.config.seed,
                     n_jobs=1, verbose=1, **self.config.hyperparameters)
        started = time.perf_counter()
        with threadpool_limits(limits=1):
            coordinates = model.fit_transform(values)
        return ReductionResult(coordinates, {
            "fit_seconds": time.perf_counter()-started,
            "effective_hyperparameters": model.get_params(deep=False),
            "learning_rate_effective": float(model.learning_rate_),
            "kl_divergence": float(model.kl_divergence_),
            "n_iter_reported": int(model.n_iter_),
            "preprocessing_effective": {"pre_reduction": "none", "input_dimension_used": values.shape[1],
                                        "initialization": self.config.hyperparameters["init"],
                                        "initialization_is_not_input_pca": True},
            "thread_policy": "native libraries and neighbor search limited to one thread",
        })


class PaCMAPReducer(DimensionalityReducer):
    def _fit(self, values: np.ndarray) -> ReductionResult:
        import faiss
        import numba
        import pacmap
        from threadpoolctl import threadpool_limits

        params = dict(self.config.hyperparameters)
        initialization = params.pop("init")
        params["num_iters"] = tuple(params["num_iters"])
        model = pacmap.PaCMAP(n_components=self.config.output_dimension, random_state=self.config.seed,
                              verbose=True, intermediate=False, save_tree=False, **params)
        effective = model.get_params(deep=False)
        prior_numba, prior_faiss = numba.get_num_threads(), faiss.omp_get_max_threads()
        started = time.perf_counter()
        try:
            numba.set_num_threads(1)
            faiss.omp_set_num_threads(1)
            with threadpool_limits(limits=1):
                coordinates = model.fit_transform(values, init=initialization, save_pairs=True)
        finally:
            numba.set_num_threads(prior_numba)
            faiss.omp_set_num_threads(prior_faiss)
        elapsed = time.perf_counter()-started
        expected = (params["n_neighbors"], round(params["n_neighbors"]*params["MN_ratio"]), round(params["n_neighbors"]*params["FP_ratio"]))
        if (model.n_neighbors, model.n_MN, model.n_FP) != expected or model.pca_solution:
            raise ValueError("PaCMAP silently changed pair counts or applied preliminary PCA")
        pair_metadata = {}
        for name in ("pair_neighbors", "pair_MN", "pair_FP"):
            pairs = np.asarray(getattr(model, name))
            if pairs.ndim != 2 or pairs.shape[1] != 2 or not np.issubdtype(pairs.dtype, np.integer) or (pairs < 0).any() or (pairs >= len(values)).any() or np.any(pairs[:, 0] == pairs[:, 1]):
                raise ValueError("PaCMAP returned invalid sampled pairs")
            pair_metadata[name] = {"count": len(pairs), "sha256": hashlib.sha256(pairs.tobytes()).hexdigest()}
        return ReductionResult(coordinates, {
            "fit_seconds": elapsed, "effective_hyperparameters": effective,
            "effective_pair_counts_per_content": {"near": model.n_neighbors, "mid_near": model.n_MN, "further": model.n_FP},
            "sampled_pairs": pair_metadata,
            "iterations_executed": sum(params["num_iters"]),
            "preprocessing_effective": {
                "pre_reduction": "none", "input_dimension_used": values.shape[1], "apply_pca": False,
                "native_affine": "subtract global scalar minimum; divide global scalar range; subtract column means",
                "native_affine_preserves": "Euclidean pair distances up to one positive scale, before float32 rounding",
                "scalar_minimum": float(model.xmin), "scalar_range": float(model.xmax),
                "initialization": initialization, "initialization_is_not_input_pca": True,
                "initialization_pca_solver": model.tsvd_transformer._fit_svd_solver,
                "initialization_pca_variance_ratio": model.tsvd_transformer.explained_variance_ratio_.tolist(),
            },
            "thread_policy": "native libraries, numba and FAISS limited to one thread; sequential fits",
            "neighbor_backend": "FAISS HNSWFlat, M=32, native search defaults; package source/version recorded",
        })


def make_reducer(config: ReductionConfig) -> DimensionalityReducer:
    implementation_versions(config.method)
    return TSNEReducer(config) if config.method == "tsne" else PaCMAPReducer(config)
