"""Versioned audit contract. Import integrity is never scientific verification."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

SCHEMA = "audit-evidence-v1"


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class SourceAsset(Model):
    asset_id: str
    kind: Literal["json", "csv", "zip"]
    filename: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    provenance: str = "user_supplied_local_artifact"
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class TemporalObservation(Model):
    observation_id: str
    asset_id: str
    source_video: str | None = None
    dataset_index: float | None = Field(default=None, ge=0)
    nominal_second: float | None = Field(default=None, ge=0)
    alignment_method: str = "reported_nominal_not_verified"
    precision_seconds: float | None = Field(default=None, gt=0)
    verification: Literal["unverified"] = "unverified"
    original_split: str | None = None


class CandidateEvent(Model):
    event_id: str
    asset_id: str
    record_locator: str
    source_video: str | None = None
    start_second: float | None = Field(default=None, ge=0)
    end_second: float | None = Field(default=None, ge=0)
    start_index: float | None = Field(default=None, ge=0)
    end_index: float | None = Field(default=None, ge=0)
    method: str
    method_version: str = "unreported"
    importer_version: Literal["legacy-audit-import-v1"] = "legacy-audit-import-v1"
    alert_type: str
    metrics: dict[str, float | None] = Field(default_factory=dict)
    evidence: list[str] = Field(default_factory=list)
    observations: list[str] = Field(default_factory=list)
    verification: Literal["candidate_only"] = "candidate_only"

    @model_validator(mode="after")
    def ordered(self):
        for start, end in [
            (self.start_second, self.end_second),
            (self.start_index, self.end_index),
        ]:
            if start is not None and end is not None and start > end:
                raise ValueError("Reversed interval")
        return self


class DecisionInput(Model):
    event_id: str
    expected_version: int = Field(ge=0)
    status: Literal["pending", "reviewed"] = "pending"
    continuity: Literal["continuous", "shot_change", "uncertain"] = "uncertain"
    dependence: Literal["same_region", "different_region", "uncertain"] = "uncertain"
    confidence: Literal["low", "medium", "high", "undetermined"] = "undetermined"
    justification: str = Field(min_length=1, max_length=10000)
    reviewer: str = Field(min_length=1, max_length=200)
    phenomena: list[
        Literal[
            "zoom",
            "camera_motion",
            "gain_change",
            "polarity_change",
            "signal_loss",
            "unknown",
        ]
    ] = Field(default_factory=list)
    evidence_consulted: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def nonblank(self):
        if not self.reviewer.strip() or not self.justification.strip():
            raise ValueError("Reviewer and justification must not be blank")
        return self


class ReviewDecision(DecisionInput):
    decision_id: str
    version: int = Field(ge=1)
    created_at: datetime
    evidence_sha256: str

    @model_validator(mode="after")
    def version_order(self):
        if self.version != self.expected_version + 1 or self.created_at.tzinfo is None:
            raise ValueError("Invalid version or timestamp without timezone")
        return self
