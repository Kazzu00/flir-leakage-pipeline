"""Strict, bounded experiment grids, separate from the committed v1 protocol."""

import itertools
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from flir_pipeline.clustering.base import ClusteringConfig
from flir_pipeline.reduction.base import ReductionConfig
from flir_pipeline.sequences.base import SequenceConfig
from flir_pipeline.similarity.storage import stable_id

Encoder = Literal["clip", "dinov2"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class BoundaryConfig(StrictModel):
    current_v1: SequenceConfig = SequenceConfig()
    windows: tuple[int, ...] = (1, 3, 5, 10, 20)
    baseline_radius: int = Field(30, ge=2, strict=True)
    baseline_min_count: int = Field(5, ge=2, strict=True)
    rank_threshold: float = Field(0.95, gt=0, le=1)
    minimum_change: float = Field(0.02, gt=0, le=2)
    local_excess: float = Field(0.01, ge=0, le=2)
    merge_gap: int = Field(3, ge=0, strict=True)
    max_index_gap: int = Field(1, ge=1, strict=True)

    @model_validator(mode="after")
    def valid_windows(self):
        if (
            not self.windows
            or self.windows[0] != 1
            or tuple(sorted(set(self.windows))) != self.windows
            or any(type(w) is not int or w < 1 for w in self.windows)
        ):
            raise ValueError("windows must be increasing positive integers including 1")
        return self


class RecurrenceConfig(StrictModel):
    thresholds: dict[Encoder, tuple[float, ...]] = Field(
        default_factory=lambda: {
            "clip": (0.85, 0.90, 0.95),
            "dinov2": (0.80, 0.90, 0.95),
        }
    )
    selection_threshold: dict[Encoder, float] = Field(
        default_factory=lambda: {"clip": 0.90, "dinov2": 0.90}
    )
    broad_support: float = Field(0.05, ge=0, le=1)
    refined_support: float = Field(0.30, ge=0, le=1)
    rank_fraction: float = Field(0.25, gt=0, le=1)
    warning_candidate_rate: float = Field(0.50, gt=0, le=1)
    block_rows: int = Field(256, ge=1, strict=True)
    cluster_min_members: int = Field(2, ge=1, strict=True)
    cluster_support: float = Field(0.20, gt=0, le=1)

    @model_validator(mode="after")
    def valid_thresholds(self):
        for mapping in (self.thresholds, self.selection_threshold):
            if set(mapping) != {"clip", "dinov2"}:
                raise ValueError("Both encoder threshold scales must be explicit")
        for encoder, thresholds in self.thresholds.items():
            if (
                not thresholds
                or tuple(sorted(set(thresholds))) != thresholds
                or any(not -1 <= t <= 1 for t in thresholds)
                or self.selection_threshold[encoder] not in thresholds
            ):
                raise ValueError(
                    "Selection threshold must belong to sorted cosine thresholds"
                )
        if self.refined_support < self.broad_support:
            raise ValueError("Refined support cannot be weaker than broad support")
        return self


class TransitionConfig(StrictModel):
    persistence: int = Field(3, ge=1, strict=True)
    tolerance: int = Field(0, ge=0, strict=True)
    max_index_gap: int = Field(1, ge=1, strict=True)


class ReductionGrid(StrictModel):
    method: Literal["tsne", "pacmap"]
    seeds: tuple[int, ...] = Field(default=(0, 1, 2), min_length=1)
    output_dimension: Literal[2, 3] = 2
    parameters: dict = Field(default_factory=dict)
    grid: dict[str, list] = Field(default_factory=dict)

    def expand(self) -> list[ReductionConfig]:
        return [
            ReductionConfig(
                self.method,
                seed=seed,
                output_dimension=self.output_dimension,
                hyperparameters=p,
                evaluation_ks=(1,),
            )
            for p in expand_parameters(self.parameters, self.grid)
            for seed in self.seeds
        ]


class ClusterGrid(StrictModel):
    algorithm: Literal["dbscan", "optics", "hdbscan", "agglomerative"]
    parameters: dict = Field(default_factory=dict)
    grid: dict[str, list] = Field(default_factory=dict)

    def expand(self) -> list[dict]:
        result = []
        for p in expand_parameters(self.parameters, self.grid):
            if self.algorithm == "agglomerative":
                p = {"n_clusters": 8, "linkage": "average", **p}
                if (
                    set(p) != {"n_clusters", "linkage"}
                    or type(p["n_clusters"]) is not int
                    or p["n_clusters"] < 2
                    or p["linkage"] not in {"average", "complete", "ward"}
                ):
                    raise ValueError("Invalid experimental Agglomerative parameters")
            else:
                p = ClusteringConfig(self.algorithm, p).hyperparameters
            result.append(p)
        return result


def expand_parameters(parameters: dict, grid: dict) -> list[dict]:
    if any(not values for values in grid.values()):
        raise ValueError("Grid axes cannot be empty")
    keys = sorted(grid)
    if (
        len(list(itertools.islice(itertools.product(*(grid[k] for k in keys)), 257)))
        > 256
    ):
        raise ValueError("Grid exceeds 256 cells")
    rows = [
        {**parameters, **dict(zip(keys, values, strict=True))}
        for values in itertools.product(*(grid[k] for k in keys))
    ]
    if len({stable_id(row) for row in rows}) != len(rows):
        raise ValueError("Duplicate grid configurations")
    return rows


class SuiteConfig(StrictModel):
    protocol: Literal["sequence_experiments_v1"] = "sequence_experiments_v1"
    encoders: tuple[Encoder, ...] = ("clip", "dinov2")
    original_l2: bool = True
    reductions: tuple[ReductionGrid, ...] = (
        ReductionGrid(method="pacmap"),
        ReductionGrid(method="tsne"),
    )
    clustering: tuple[ClusterGrid, ...] = (
        ClusterGrid(algorithm="dbscan"),
        ClusterGrid(algorithm="optics"),
        ClusterGrid(algorithm="hdbscan"),
        ClusterGrid(algorithm="agglomerative"),
    )
    boundary: BoundaryConfig = BoundaryConfig()
    boundary_grid: dict[str, list] = Field(default_factory=dict)
    recurrence: RecurrenceConfig = RecurrenceConfig()
    transitions: TransitionConfig = TransitionConfig()
    max_runs: int = Field(256, ge=1, le=2048, strict=True)

    @model_validator(mode="after")
    def validate_grids(self):
        if not self.encoders or len(set(self.encoders)) != len(self.encoders):
            raise ValueError("Encoders must be distinct and nonempty")
        spaces = [r for grid in self.reductions for r in grid.expand()]
        from dataclasses import asdict

        if len({stable_id(asdict(r)) for r in spaces}) != len(spaces):
            raise ValueError("Duplicate reduction cells/seeds")
        cluster_cells = [(g.algorithm, p) for g in self.clustering for p in g.expand()]
        if not cluster_cells or len(
            {stable_id(dict(algorithm=a, parameters=p)) for a, p in cluster_cells}
        ) != len(cluster_cells):
            raise ValueError("Empty or duplicate clustering cells")
        count = (
            len(self.encoders)
            * (len(spaces) + int(self.original_l2))
            * len(cluster_cells)
        )
        if not 0 < count <= self.max_runs:
            raise ValueError("Experiment grid empty or exceeds max_runs")
        self.boundaries()
        return self

    def boundaries(self) -> list[BoundaryConfig]:
        return [
            BoundaryConfig.model_validate(p)
            for p in expand_parameters(
                self.boundary.model_dump(mode="json"), self.boundary_grid
            )
        ]

    @classmethod
    def load(cls, path: Path):
        from flir_pipeline.sequences.experiments.inputs import load_profile

        return load_profile(path)[0]


SEMANTICS = {
    "ground_truth": False,
    "split_created": False,
    "automatic_confirmation": False,
    "sequence_instances_created": False,
    "visual_dependency_groups_created": False,
    "clusters": "algorithmic visual evidence, never sequence membership",
    "boundary_zone": "inclusive uncertainty interval, never an exact cut",
    "candidate_components": "diagnostic only; never VDGs",
    "manual_review": "external evidence, not ground truth",
    "fitting_inputs": "unique-content visual features only",
}
