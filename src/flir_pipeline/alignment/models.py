"""Separate correspondence evidence from temporal/region human judgments."""

from fractions import Fraction
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Config(Model):
    seed: int = 20261103
    per_split: int = Field(default=12, ge=1, le=50)
    offset_hypothesis: float = 1978
    local_radius: float = Field(default=8, gt=0, le=120)
    coarse_step: float = Field(default=0.5, gt=0)
    expanded_radius: float = Field(default=60, gt=0, le=300)
    expanded_step: float = Field(default=3, gt=0)
    refine_radius: float = Field(default=0.3, gt=0, le=2)
    top_k: int = Field(default=5, ge=2, le=10)
    max_decoded_frames: int = Field(default=90000, ge=1)
    max_seconds: float = Field(default=1800, gt=0)
    max_images: int = Field(default=60, ge=1, le=200)
    selection_archive: str = "Imagenes.zip"
    # Retrieval support gates, not calibrated probabilities or truth labels.
    min_inliers: int = Field(default=12, ge=4)
    min_inlier_ratio: float = Field(default=0.35, ge=0, le=1)
    min_coverage: float = Field(default=0.03, ge=0, le=1)


class ImageRecord(Model):
    image_id: str
    source_archive_or_path: str
    member: str | None
    member_ordinal: int | None
    original_split: Literal["train", "val", "test"] | None
    dataset_index: int
    image_sha256: str | None
    image_dimensions: list[int] | None
    source_video_candidate: str
    read_status: Literal["readable", "error"]
    discrepancies: list[str]


class Match(Model):
    pts: int
    time_base: str
    second: float
    duration_seconds: float | None
    decoded_rgb_sha256: str
    requested_seek_second: float
    first_decoded_second: float
    correlation: float
    dhash_distance: int
    matches: int
    inliers: int
    inlier_ratio: float
    coverage: float
    reprojection_median_px: float | None
    supported: bool
    search_stage: str

    @model_validator(mode="after")
    def consistent_time_and_support(self):
        try:
            base = Fraction(self.time_base)
        except (ValueError, ZeroDivisionError) as exc:
            raise ValueError("Invalid media time base") from exc
        if base <= 0 or abs(self.second - float(self.pts * base)) > 1e-9:
            raise ValueError("PTS/time-base inconsistency")
        if (
            not 0 <= self.inliers <= self.matches
            or not 0 <= self.inlier_ratio <= 1
            or not 0 <= self.coverage <= 1
        ):
            raise ValueError("Invalid geometric support")
        return self


class Case(Model):
    alignment_id: str
    image_id: str
    status: Literal["candidate", "ambiguous", "not_found"]
    reason: str
    top_k: list[Match]
    controls: list[Match]
    searched_intervals: list[list[float]]
    candidate_interval: list[float] | None
    evidence_locator: str | None
    evidence_sha256: str | None
    control_evidence_locator: str | None = None
    control_evidence_sha256: str | None = None
    audit_event_links: list[dict[str, str]]
    errors: list[str]


class Evidence(Model):
    schema_name: Literal["alignment-evidence-v1"] = Field(alias="schema")
    run_id: str
    config: Config
    manifest_sha256: str
    selection_sha256: str
    source_video: str
    source_video_sha256: str
    video_metadata: dict
    execution: dict
    cases: list[Case]


class HumanVerification(Model):
    """Explicit later human action; not emitted by the matching pipeline."""

    schema_name: Literal["alignment-human-review-v1"] = Field(alias="schema")
    alignment_id: str = Field(min_length=1)
    status: Literal["human_verified", "ambiguous", "not_found"]
    reviewer: str = Field(min_length=1)
    justification: str = Field(min_length=1)
    evidence_consulted: list[str] = Field(min_length=1)
    selected_pts: int | None
    source_locator: str = Field(min_length=1)

    @model_validator(mode="after")
    def explicit(self):
        if not all(
            x.strip()
            for x in [
                self.reviewer,
                self.justification,
                self.source_locator,
                *self.evidence_consulted,
            ]
        ):
            raise ValueError(
                "Human review requires nonblank evidence, locator and reviewer"
            )
        if self.status == "human_verified" and self.selected_pts is None:
            raise ValueError("Verification requires an explicitly selected media PTS")
        return self
