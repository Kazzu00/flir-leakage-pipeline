"""Lightweight presentation contract; scientific calculations belong to the producer.

The JSON Schema describes a logical bundle keyed by filename stem. The producer
validates this bundle before serializing the individual JSON files. No raw input
paths, image identities, weights, bootstrap samples or training logs are allowed.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "detection-export-v1"
StrategyName = Literal["historical", "random_content", "C10", "C12"]
MetricName = Literal["map50_95", "map50", "precision", "recall"]
Metric = Annotated[float, Field(ge=0, le=1)]
Correlation = Annotated[float, Field(ge=-1, le=1)]
Count = Annotated[int, Field(ge=0)]
ClassId = Literal[0, 1, 2, 3, 4]
State = Literal["COMPLETE_CONTROLLED_COMPARISON"]


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Semantics(ContractModel):
    causal: Literal[False]
    unit_of_analysis: Literal["split_mean_across_detector_seeds"]
    comparison: str
    temporal_at5: str
    clustering: str
    missing_values: str


class Protocol(ContractModel):
    plan_id: str
    model_config_id: str
    model: str
    epochs: Annotated[int, Field(ge=1)]
    batch: Annotated[int, Field(ge=1)]
    detector_seeds: list[int]
    bootstrap_method: Literal["image_percentile_bootstrap"]
    bootstrap_resamples: Annotated[int, Field(ge=1)]
    bootstrap_seed: int
    confidence_level: Annotated[float, Field(gt=0, lt=1)]


class Manifest(ContractModel):
    schema_version: Literal["detection-export-v1"]
    state: State
    scientific_result: Literal[True]
    generated_from_verified_artifacts: Literal[True]
    plan_id: str
    model_config_id: str
    expected_runs: Annotated[int, Field(ge=1)]
    completed_runs: Annotated[int, Field(ge=1)]
    strategy_count: Annotated[int, Field(ge=1)]
    split_count: Annotated[int, Field(ge=1)]
    detector_seed_count: Annotated[int, Field(ge=1)]
    excluded_small_pilots: Count
    generated_at: str
    source_commit: str
    source_worktree_dirty: bool | None
    source_receipt_sha256: dict[str, str]
    file_sha256: dict[str, str]
    limitations: list[str]


class SplitKey(ContractModel):
    strategy: StrategyName
    split_seed: int
    split_space_id: str


class Run(SplitKey):
    detector_run_id: str
    detector_seed: int
    state: Literal["COMPLETE"]
    map50_95: Metric | None
    map50: Metric | None
    precision: Metric | None
    recall: Metric | None
    best_epoch: Annotated[int, Field(ge=1)] | None
    training_seconds: Annotated[float, Field(ge=0)] | None
    peak_memory_bytes: Count | None


class MeanMetrics(ContractModel):
    mean_map50_95: Metric | None
    mean_map50: Metric | None
    mean_precision: Metric | None
    mean_recall: Metric | None


class Split(SplitKey, MeanMetrics):
    n_detector_runs: Annotated[int, Field(ge=1)]
    std_map50_95: Metric | None
    temporal_at5: Metric | None
    dinov2_nn_mean: Correlation | None
    clip_nn_mean: Correlation | None
    dinov2_top001_pairs: Count
    clip_top001_pairs: Count
    exact_duplicate_cross_split_count: Count
    class_deviation_pp: Annotated[float, Field(ge=0, le=100)]


def single_split_schema(field: str) -> dict:
    return {
        "allOf": [
            {
                "if": {"properties": {"n_splits": {"const": 1}}},
                "then": {"properties": {field: {"type": "null"}}},
            }
        ]
    }


class Strategy(MeanMetrics):
    model_config = ConfigDict(
        json_schema_extra=single_split_schema("std_between_splits")
    )
    strategy: StrategyName
    display_name: str
    n_splits: Annotated[int, Field(ge=1)]
    n_detector_runs: Annotated[int, Field(ge=1)]
    std_between_splits: Metric | None
    median_map50_95: Metric | None
    min_map50_95: Metric | None
    max_map50_95: Metric | None
    mean_temporal_at5: Metric | None
    mean_dinov2_nn_mean: Correlation | None
    mean_clip_nn_mean: Correlation | None

    @model_validator(mode="after")
    def undefined_single_split(self):
        if self.n_splits == 1 and self.std_between_splits is not None:
            raise ValueError("Single-split standard deviation must be null")
        return self


class ClassSummary(MeanMetrics):
    model_config = ConfigDict(json_schema_extra=single_split_schema("std_map50_95"))
    strategy: StrategyName
    class_id: ClassId
    class_name: str
    n_splits: Annotated[int, Field(ge=1)]
    std_map50_95: Metric | None

    @model_validator(mode="after")
    def undefined_single_split(self):
        if self.n_splits == 1 and self.std_map50_95 is not None:
            raise ValueError("Single-split class standard deviation must be null")
        return self


class Support(SplitKey):
    class_id: ClassId
    class_name: str
    support: Count


class Variance(ContractModel):
    strategy: StrategyName
    n_splits: Annotated[int, Field(ge=1)]
    mean_within_split_detector_seed_std: Metric | None
    between_split_std: Metric | None
    between_vs_within_ratio: Annotated[float, Field(ge=0)] | None
    model_config = ConfigDict(
        json_schema_extra={
            "allOf": [
                {
                    "if": {"properties": {"n_splits": {"const": 1}}},
                    "then": {
                        "properties": {
                            "between_split_std": {"type": "null"},
                            "between_vs_within_ratio": {"type": "null"},
                        }
                    },
                }
            ]
        }
    )

    @model_validator(mode="after")
    def undefined_single_split(self):
        if self.n_splits == 1 and (
            self.between_split_std is not None
            or self.between_vs_within_ratio is not None
        ):
            raise ValueError("Single-split variance and ratio must be null")
        return self


class Point(SplitKey):
    x: float | None
    y: Metric | None


class Correlations(ContractModel):
    n_splits: Count
    n_valid_splits: Count
    global_pearson: Correlation | None
    global_spearman: Correlation | None
    within_strategy_centered_pearson: Correlation | None


class StrategyCorrelation(ContractModel):
    strategy: StrategyName
    n_valid_splits: Count
    pearson: Correlation | None
    spearman: Correlation | None


class AssociationInterpretation(ContractModel):
    causal: Literal[False]
    unit_of_analysis: Literal["split_mean_across_detector_seeds"]
    global_vs_within: str
    centering_reduces_absolute_r: bool | None
    absolute_r_reduction: float | None
    undefined_correlation: str


class Association(Correlations):
    association_id: str
    category: str
    prespecified: Literal[True]
    population: Literal["overall", "class"]
    class_id: Literal[-1, 0, 1, 2, 3, 4]
    detector_metric: MetricName
    residual_metric: str
    per_strategy: list[StrategyCorrelation]
    points: list[Point]
    interpretation: AssociationInterpretation


class Interval(ContractModel):
    detector_run_id: str
    class_id: Literal[-1, 0, 1, 2, 3, 4]
    metric: MetricName
    lower: Metric | None
    upper: Metric | None
    valid_resamples: Count

    @model_validator(mode="after")
    def ordered_bounds(self):
        if (self.lower is None) != (self.upper is None):
            raise ValueError("Bootstrap bounds must be both present or both null")
        if self.lower is not None and self.lower > self.upper:
            raise ValueError("Bootstrap bounds are reversed")
        if (self.valid_resamples == 0) != (self.lower is None):
            raise ValueError("Bootstrap support and bounds disagree")
        return self


class Summary(ContractModel):
    state: State
    scientific_result: Literal[True]
    protocol: Protocol
    headline_metrics: dict[StrategyName, Metric | None]
    strategies: list[Strategy]
    variance: list[Variance]
    association_summary: dict[str, Correlations]
    interpretation: Semantics
    limitations: list[str]


class DetectionExport(ContractModel):
    manifest: Manifest
    summary: Summary
    strategies: list[Strategy]
    splits: list[Split]
    runs: list[Run]
    classes: list[ClassSummary]
    support: list[Support]
    variance: list[Variance]
    associations: list[Association]
    bootstrap: list[Interval]

    @model_validator(mode="after")
    def consistent_bundle(self):
        m = self.manifest
        if not (m.expected_runs == m.completed_runs == len(self.runs)):
            raise ValueError("Export run counts disagree")
        if (
            m.split_count != len(self.splits)
            or m.strategy_count != len(self.strategies)
            or m.detector_seed_count != len(self.summary.protocol.detector_seeds)
        ):
            raise ValueError("Export matrix counts disagree")
        if (
            self.summary.strategies != self.strategies
            or self.summary.variance != self.variance
            or self.summary.headline_metrics
            != {s.strategy: s.mean_map50_95 for s in self.strategies}
        ):
            raise ValueError("Overview disagrees with detailed export")
        if (
            m.plan_id != self.summary.protocol.plan_id
            or m.model_config_id != self.summary.protocol.model_config_id
        ):
            raise ValueError("Export identities disagree")
        if len({r.detector_run_id for r in self.runs}) != len(self.runs):
            raise ValueError("Duplicate exported run")
        keys = {(s.strategy, s.split_seed, s.split_space_id) for s in self.splits}
        if len(keys) != len(self.splits):
            raise ValueError("Duplicate exported split")
        seeds = set(self.summary.protocol.detector_seeds)
        expected_cells = {(*key, seed) for key in keys for seed in seeds}
        actual_cells = {
            (r.strategy, r.split_seed, r.split_space_id, r.detector_seed)
            for r in self.runs
        }
        if actual_cells != expected_cells or len(actual_cells) != len(self.runs):
            raise ValueError("Export cells do not cover the frozen matrix exactly once")
        if any(s.n_detector_runs != len(seeds) for s in self.splits):
            raise ValueError("Split detector counts disagree with matrix")
        if len({a.association_id for a in self.associations}) != len(self.associations):
            raise ValueError("Duplicate exported association")
        expected_intervals = {
            (r.detector_run_id, cid, metric)
            for r in self.runs
            for cid in range(-1, 5)
            for metric in ("map50_95", "map50", "precision", "recall")
        }
        if (
            len(self.bootstrap) != len(expected_intervals)
            or {(i.detector_run_id, i.class_id, i.metric) for i in self.bootstrap}
            != expected_intervals
        ):
            raise ValueError("Export bootstrap intervals are incomplete or duplicated")
        for a in self.associations:
            if (
                len(a.points) != len(keys)
                or {(p.strategy, p.split_seed, p.split_space_id) for p in a.points}
                != keys
            ):
                raise ValueError(
                    "Association points must cover every split exactly once"
                )
            if a.n_splits != len(keys) or a.n_valid_splits != sum(
                p.x is not None and p.y is not None for p in a.points
            ):
                raise ValueError("Association counts must count splits")
            exported_summary = self.summary.association_summary.get(a.association_id)
            if exported_summary != Correlations.model_validate(
                {k: getattr(a, k) for k in Correlations.model_fields}
            ):
                raise ValueError("Overview correlations disagree with detailed export")
        return self


def export_schema() -> dict:
    schema = DetectionExport.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = "urn:flir:detection-export-v1"
    schema["description"] = (
        "Validate the bundle {filename stem: parsed JSON} for all ten payload files."
    )
    return schema
