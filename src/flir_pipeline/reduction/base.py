"""Bounded reduction protocol, explicit preprocessing and mathematical identities."""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import yaml

from flir_pipeline.similarity.storage import stable_id

DEFAULTS = {
    "tsne": {
        "perplexity": 30.0, "learning_rate": "auto", "max_iter": 1000,
        "early_exaggeration": 12.0, "init": "pca", "metric": "euclidean",
        "method": "barnes_hut", "angle": 0.5, "n_iter_without_progress": 300,
        "min_grad_norm": 1e-7,
    },
    "pacmap": {
        "n_neighbors": 10, "MN_ratio": 0.5, "FP_ratio": 2.0,
        "distance": "euclidean", "lr": 1.0, "num_iters": [100, 100, 250],
        "apply_pca": False, "init": "pca", "knn_backend": "faiss",
    },
}


@dataclass(frozen=True)
class ReductionConfig:
    """Initialization PCA is distinct from pre-reducing the high-dimensional input.

    This first protocol deliberately uses no preliminary PCA. Future support
    requires an explicit protocol change and a direct-versus-preprocessed control.
    """

    method: str
    seed: int = 0
    output_dimension: int = 2
    input_representation: str = "embeddings_l2"
    preprocessing: str = "none"
    hyperparameters: dict = field(default_factory=dict)
    evaluation_ks: tuple[int, ...] = (5, 10, 20)
    distance_sample_size: int = 100000
    distance_sample_seed: int = 0
    protocol_version: str = "direct_l2_reduction_v1"

    def __post_init__(self):
        if self.method not in DEFAULTS:
            raise ValueError("Only tsne and pacmap belong to this protocol")
        if self.preprocessing != "none" or self.input_representation != "embeddings_l2":
            raise ValueError("This protocol requires original L2 input without preliminary PCA")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be a nonnegative 32-bit integer")
        if type(self.output_dimension) is not int or self.output_dimension not in (2, 3):
            raise ValueError("Only 2D and 3D coordinates are supported")
        if self.protocol_version != "direct_l2_reduction_v1":
            raise ValueError("Unsupported reduction protocol")
        ks = tuple(self.evaluation_ks)
        if not ks or tuple(sorted(set(ks))) != ks or any(type(k) is not int or k < 1 for k in ks):
            raise ValueError("Evaluation neighborhoods must be increasing positive integers")
        object.__setattr__(self, "evaluation_ks", ks)
        if type(self.distance_sample_size) is not int or self.distance_sample_size < 2 or type(self.distance_sample_seed) is not int or self.distance_sample_seed < 0:
            raise ValueError("Invalid deterministic pair sample")
        defaults = DEFAULTS[self.method]
        if not isinstance(self.hyperparameters, dict) or set(self.hyperparameters)-set(defaults):
            raise ValueError("Unknown or inappropriate reducer hyperparameter")
        params = {**defaults, **self.hyperparameters}
        if params["init"] not in ("pca", "random"):
            raise ValueError("Initialization must be explicitly pca or random")
        if self.method == "tsne":
            if not np.isfinite(params["perplexity"]) or params["perplexity"] <= 0:
                raise ValueError("perplexity must be positive and finite")
            if type(params["max_iter"]) is not int or params["max_iter"] < 300:
                raise ValueError("Use at least 300 iterations, including post-exaggeration optimization")
            if params["metric"] != "euclidean" or params["method"] != "barnes_hut" or not 0 <= params["angle"] <= 1:
                raise ValueError("This protocol uses Barnes-Hut t-SNE with Euclidean L2 input")
            if params["learning_rate"] != "auto" and (not isinstance(params["learning_rate"], (int, float)) or not np.isfinite(params["learning_rate"]) or params["learning_rate"] <= 0):
                raise ValueError("Invalid learning_rate")
            if params["early_exaggeration"] < 1 or params["n_iter_without_progress"] < 1 or params["min_grad_norm"] < 0:
                raise ValueError("Invalid t-SNE optimization parameters")
        else:
            if params["apply_pca"] is not False or params["distance"] != "euclidean" or params["knn_backend"] != "faiss":
                raise ValueError("PaCMAP requires apply_pca=false, Euclidean distance and explicit FAISS backend")
            if type(params["n_neighbors"]) is not int or params["n_neighbors"] < 1:
                raise ValueError("Invalid PaCMAP n_neighbors")
            for key in ("MN_ratio", "FP_ratio", "lr"):
                if not np.isfinite(params[key]) or params[key] <= 0:
                    raise ValueError(f"Invalid PaCMAP {key}")
            steps = list(params["num_iters"])
            if len(steps) != 3 or any(type(i) is not int or i < 1 for i in steps):
                raise ValueError("PaCMAP num_iters must specify all three positive phase lengths")
            params["num_iters"] = steps
        object.__setattr__(self, "hyperparameters", params)

    def validate_population(self, n: int, dimension: int) -> None:
        if n < 3 or dimension < self.output_dimension or max(self.evaluation_ks) >= n/2:
            raise ValueError("Population/dimension must support output and k < N/2")
        p = self.hyperparameters
        if self.method == "tsne" and p["perplexity"] >= n:
            raise ValueError("t-SNE perplexity must be smaller than N")
        if self.method == "pacmap":
            counts = (p["n_neighbors"], round(p["n_neighbors"]*p["MN_ratio"]), round(p["n_neighbors"]*p["FP_ratio"]))
            if min(counts) < 1 or sum(counts) >= n:
                raise ValueError("Population would make PaCMAP silently adjust pair counts")


def reduction_space_id(dataset_id: str, feature_space_id: str, config: ReductionConfig, versions: dict) -> str:
    """Dataset + mathematical settings + implementation versions; no runtime paths."""
    return stable_id({"dataset_id": dataset_id, "feature_space_id": feature_space_id,
                      "config": asdict(config), "implementation_versions": versions})


def configuration_id(config: ReductionConfig) -> str:
    """Group the same scientific configuration across seeds, within each encoder."""
    payload = asdict(config)
    payload.pop("seed")
    return stable_id(payload)


def load_grid(path: Path) -> list[ReductionConfig]:
    """Expand a small YAML grid with strict keys, rejecting duplicate configurations."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Reduction config must be a YAML mapping")
    payload = dict(payload)
    seeds = payload.pop("seeds", [0, 1, 2])
    grid = payload.pop("grid", {})
    if "seed" in payload or not seeds or len(set(seeds)) != len(seeds) or not isinstance(grid, dict):
        raise ValueError("Use distinct seeds and a named hyperparameter grid")
    if any(not isinstance(v, list) or not v for v in grid.values()):
        raise ValueError("Grid values must be nonempty lists")
    configurations = []
    for values in itertools.product(*grid.values()):
        params = {**payload.get("hyperparameters", {}), **dict(zip(grid, values, strict=True))}
        base = ReductionConfig(**{**payload, "hyperparameters": params})
        configurations.extend(replace(base, seed=seed) for seed in seeds)
    keys = [stable_id(asdict(c)) for c in configurations]
    if len(keys) > 9 or len(set(keys)) != len(keys):
        raise ValueError("The principal grid must contain at most nine distinct runs per method")
    return configurations
