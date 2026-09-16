"""Portable partition identity and explicit construction settings."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import yaml

from flir_pipeline.similarity.storage import stable_id

SPLITS = ("train", "val", "test")
CLASS_COLUMNS = tuple(f"class_{i}" for i in range(5))
BALANCE_COLUMNS = ("record_count", *CLASS_COLUMNS, "empty_annotation_count")


@dataclass(frozen=True)
class SplitConfig:
    strategy: str = "cluster_aware"
    noise_policy: str = "singleton"
    target_ratios: tuple[float, ...] | None = None
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    node_limit: int = 32
    mip_rel_gap: float = 0.001
    record_weight: float = 1.0
    class_weight: float = 1.0
    empty_weight: float = 1.0
    protocol_version: str = "atomic_partition_v1"

    def __post_init__(self):
        object.__setattr__(self, "seeds", tuple(self.seeds))
        if self.strategy not in ("historical", "random_content", "cluster_aware"):
            raise ValueError("Unsupported splitting strategy")
        if self.noise_policy != "singleton":
            raise ValueError("Only singleton is enabled; similarity-components needs an explicit ablation protocol")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds) or any(type(s) is not int or s < 0 for s in self.seeds):
            raise ValueError("Seeds must be distinct nonnegative integers")
        if self.target_ratios is not None:
            ratios = tuple(float(x) for x in self.target_ratios)
            if len(ratios) != 3 or not np.isfinite(ratios).all() or min(ratios) <= 0 or not np.isclose(sum(ratios), 1, atol=1e-12, rtol=0):
                raise ValueError("Targets must be three positive ratios summing to one")
            object.__setattr__(self, "target_ratios", ratios)
        if type(self.node_limit) is not int or self.node_limit < 1 or not 0 <= self.mip_rel_gap < 1:
            raise ValueError("Invalid deterministic MILP limits")
        if any(not np.isfinite(w) or w <= 0 for w in (self.record_weight, self.class_weight, self.empty_weight)):
            raise ValueError("Balance weights must be positive and finite")
        if self.protocol_version != "atomic_partition_v1":
            raise ValueError("Unsupported splitting protocol")

    @classmethod
    def load(cls, path: Path) -> SplitConfig:
        return cls(**yaml.safe_load(path.read_text(encoding="utf-8")))


def targets_from_manifest(manifest: pd.DataFrame, config: SplitConfig) -> np.ndarray:
    """Historical membership contributes only aggregate target record ratios."""
    if manifest.empty or not manifest.original_split.isin(SPLITS).all():
        raise ValueError("Complete historical train/val/test metadata is required")
    ratios = np.asarray(config.target_ratios if config.target_ratios is not None else
                        [int((manifest.original_split == s).sum()) / len(manifest) for s in SPLITS])
    if (ratios <= 0).any():
        raise ValueError("All three target splits must be represented")
    return ratios


def identity_payload(dataset_id: str, config: SplitConfig, ratios: np.ndarray,
                     seed: int, clustering_id: str | None = None,
                     historical_membership_id: str | None = None) -> dict:
    settings = asdict(config)
    settings.pop("seeds")
    settings["target_ratios"] = list(map(float, ratios))
    return {"dataset_id": dataset_id, "clustering_space_id": clustering_id,
            "historical_membership_id": historical_membership_id if config.strategy == "historical" else None,
            "configuration": settings, "seed": seed,
            "implementation": {"numpy": np.__version__, "scipy": scipy.__version__,
                               "assignment": "profile_milp_seeded_expansion_v1" if config.strategy == "cluster_aware" else config.strategy}}


def split_space_id(payload: dict) -> str:
    """Paths, dates and device never enter the mathematical partition identity."""
    return stable_id(payload)
