"""Scientific clustering settings and bounded grids; no paths or posterior labels."""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import yaml

from flir_pipeline.similarity.storage import stable_id

DEFAULTS = {
    "dbscan": {"min_samples": 5, "eps_quantile": .90},
    "optics": {"min_samples": 5, "xi": .05, "min_cluster_size": 20,
               "cluster_method": "xi", "max_eps": "infinity", "predecessor_correction": True},
    "hdbscan": {"min_samples": 5, "min_cluster_size": 20, "cluster_selection_method": "eom",
                "cluster_selection_epsilon": 0., "alpha": 1., "allow_single_cluster": False},
}


@dataclass(frozen=True)
class ClusteringConfig:
    algorithm: str
    hyperparameters: dict = field(default_factory=dict)
    metric: str = "euclidean"
    protocol_version: str = "content_density_v1"

    def __post_init__(self):
        if self.algorithm not in DEFAULTS or self.metric != "euclidean" or self.protocol_version != "content_density_v1":
            raise ValueError("Only DBSCAN/OPTICS/HDBSCAN with Euclidean distance are supported")
        if not isinstance(self.hyperparameters, dict) or set(self.hyperparameters)-set(DEFAULTS[self.algorithm]):
            raise ValueError("Unknown clustering hyperparameter")
        p = {**DEFAULTS[self.algorithm], **self.hyperparameters}
        for key in ("min_samples", "min_cluster_size"):
            if key in p and (type(p[key]) is not int or p[key] < 2):
                raise ValueError(f"{key} must be an integer >= 2, including self for min_samples")
        if self.algorithm == "dbscan" and (not np.isfinite(p["eps_quantile"]) or not 0 < p["eps_quantile"] < 1):
            raise ValueError("DBSCAN epsilon must use a quantile in (0,1)")
        if self.algorithm == "optics" and (not 0 < p["xi"] < 1 or p["cluster_method"] != "xi" or p["max_eps"] != "infinity" or p["predecessor_correction"] is not True):
            raise ValueError("OPTICS protocol requires xi extraction, infinite max_eps and predecessor correction")
        if self.algorithm == "hdbscan" and (p["cluster_selection_method"] not in ("eom", "leaf") or p["cluster_selection_epsilon"] != 0 or p["alpha"] != 1 or p["allow_single_cluster"] is not False):
            raise ValueError("HDBSCAN protocol requires epsilon=0, alpha=1 and allow_single_cluster=false")
        object.__setattr__(self, "hyperparameters", p)

    @property
    def configuration_id(self) -> str:
        """Conceptual settings across reduction seeds; epsilon is re-derived per space."""
        return stable_id(asdict(self))


def clustering_space_id(dataset_id: str, feature_space_id: str, representation: str,
                        reduction_space_id: str | None, config: ClusteringConfig,
                        effective_parameters: dict, versions: dict) -> str:
    return stable_id({"dataset_id": dataset_id, "feature_space_id": feature_space_id,
                      "representation": representation, "reduction_space_id": reduction_space_id,
                      "config": asdict(config), "effective_parameters": effective_parameters,
                      "implementation_versions": versions})


def load_grid(path: Path) -> list[ClusteringConfig]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Clustering YAML must be a mapping")
    value = dict(value)
    grid = value.pop("grid", {})
    if not isinstance(grid, dict) or any(not isinstance(v, list) or not v for v in grid.values()):
        raise ValueError("Grid values must be nonempty lists")
    rows = [ClusteringConfig(**{**value, "hyperparameters": {**value.get("hyperparameters", {}), **dict(zip(grid, values, strict=True))}})
            for values in itertools.product(*grid.values())]
    if len(rows) > 36 or len({r.configuration_id for r in rows}) != len(rows):
        raise ValueError("A principal algorithm grid must contain <=36 distinct configurations")
    return rows
