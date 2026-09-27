"""Versioned scientific policy; runtime paths never define sequence identity."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class SequenceConfig(BaseModel):
    """The v1 protocol exposes thresholds/radii, never alternative fitting inputs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    algorithm_version: Literal["occurrence_centroid_sequences_v1"] = (
        "occurrence_centroid_sequences_v1"
    )
    sample_fps: Literal[1.0] = 1.0
    windows: tuple[Literal[3], Literal[5], Literal[10], Literal[20]] = (3, 5, 10, 20)
    rounding_decimals: Literal[12] = 12
    percentile_policy: Literal[
        "per_encoder_per_video_average_rank_over_valid_count"
    ] = "per_encoder_per_video_average_rank_over_valid_count"
    centroid_metric: Literal["one_minus_cosine_of_normalized_centroids"] = (
        "one_minus_cosine_of_normalized_centroids"
    )
    persistent_rule: Literal["min_P5_P10_P20"] = "min_P5_P10_P20"
    candidate_threshold: float = Field(default=0.95, gt=0, le=1, strict=True)
    high_confidence_threshold: float = Field(default=0.975, gt=0, le=1, strict=True)
    merge_gap: int = Field(default=3, ge=0, strict=True)
    localization_radius: int = Field(default=20, ge=0, strict=True)
    coarse_tie_policy: Literal[
        "score_median_P1_stable_percentile_consensus_lower_index"
    ] = "score_median_P1_stable_percentile_consensus_lower_index"
    localizer_policy: Literal[
        "F3_nearest_coarse_lower_index_midpoint_floor_to_left"
    ] = "F3_nearest_coarse_lower_index_midpoint_floor_to_left"
    boundary_policy: Literal["confirmed_accept_only_cut_starts_next_sequence"] = (
        "confirmed_accept_only_cut_starts_next_sequence"
    )
    dependency_policy: Literal["exact_shared_content_star_v1"] = (
        "exact_shared_content_star_v1"
    )

    @model_validator(mode="after")
    def thresholds(self):
        if self.high_confidence_threshold < self.candidate_threshold:
            raise ValueError("high_confidence threshold must be >= candidate threshold")
        return self

    @classmethod
    def load(cls, path: Path) -> "SequenceConfig":
        return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


SEMANTICS = {
    "video_id": "source video; not sequence_id",
    "sample_index": "contiguous 1 FPS sampling-grid position; not decoder frame",
    "timestamp_seconds": "relative sampling-grid time; not capture timestamp",
    "occurrences": "all preserved, including exact copies",
    "boundary": "cut t ends preceding sequence at t-1 and starts next at t",
    "ground_truth": False,
    "split_created": False,
    "clustering_or_reduction_inputs": False,
    "historical_split_inputs": False,
    "high_confidence": "diagnostic annotation only; never manual acceptance",
    "dependency_group": "exact-copy connected component; not visual_group or split",
}
