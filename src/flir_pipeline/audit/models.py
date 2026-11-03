"""Versioned audit contract. Import integrity is never scientific verification."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

SCHEMA = "audit-evidence-v1"
REVISION = "1.1"

# JSON Schema and runtime use the same non-binary definition of a conclusion.
CONCLUSIVE_RULE = {
    "if": {
        "required": ["status"],
        "properties": {"status": {"const": "reviewed"}},
        "anyOf": [
            {"required": [field], "properties": {field: {"not": {"const": uncertain}}}}
            for field, uncertain in [
                ("continuity", "uncertain"),
                ("dependence", "uncertain"),
                ("confidence", "undetermined"),
            ]
        ],
    },
    "then": {
        "anyOf": [
            {"required": [field], "properties": {field: {"minItems": 1}}}
            for field in ["evidence_consulted", "external_evidence_consulted"]
        ]
    },
}


def require_evidence(value):
    if (
        value.status == "reviewed"
        and (
            value.continuity != "uncertain"
            or value.dependence != "uncertain"
            or value.confidence != "undetermined"
        )
        and not (value.evidence_consulted or value.external_evidence_consulted)
    ):
        raise ValueError(
            "Una revisión conclusiva requiere evidencia consultada y trazable; mantenga el caso pendiente o indeterminado."
        )


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
    importer_version: Literal["legacy-audit-import-v1", "legacy-audit-import-v1.1"] = (
        "legacy-audit-import-v1.1"
    )
    event_start_sample: int | None = Field(default=None, ge=0)
    event_end_sample: int | None = Field(default=None, ge=0)
    representative_second: float | None = Field(default=None, ge=0)
    sample_rate_fps: float | None = Field(default=None, gt=0)
    temporal_basis: str = "reported_nominal"
    precision_seconds: float | None = Field(default=None, gt=0)
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


class DecisionFields(Model):
    event_id: str = Field(min_length=1)
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
    external_evidence_consulted: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def nonblank(self):
        if not self.reviewer.strip() or not self.justification.strip():
            raise ValueError("Reviewer and justification must not be blank")
        return self


class DecisionInput(DecisionFields):
    model_config = ConfigDict(json_schema_extra=CONCLUSIVE_RULE)

    @model_validator(mode="after")
    def grounded(self):
        require_evidence(self)
        return self


class ReviewDecision(DecisionFields):
    model_config = ConfigDict(
        json_schema_extra={
            "if": {
                "required": ["policy_version"],
                "properties": {"policy_version": {"const": "1.1"}},
            },
            "then": CONCLUSIVE_RULE,
        }
    )
    policy_version: Literal["1.0", "1.1"] = "1.0"
    decision_id: str
    version: int = Field(ge=1)
    created_at: datetime
    evidence_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def version_order(self):
        if self.version != self.expected_version + 1 or self.created_at.tzinfo is None:
            raise ValueError("Invalid version or timestamp without timezone")
        if self.policy_version == "1.1":
            require_evidence(self)
        return self


class ExternalEvidence(Model):
    evidence_id: str
    event_id: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    event_asset_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    filename: str
    locator: str = Field(min_length=1)
    basis: Literal["explicit_record_reference", "manual_correspondence"]
    rationale: str = Field(min_length=1)
    reviewer: str = Field(min_length=1)
    created_at: datetime
    mime: Literal["image/png", "image/jpeg"]


class EventLink(Model):
    canonical_event_id: str = Field(min_length=1)
    companion_event_id: str = Field(min_length=1)


class Companion(Model):
    asset_id: str = Field(min_length=1)
    canonical_asset_id: str = Field(min_length=1)
    decisions_migrated: Literal[False]
    event_links: list[EventLink]
    receipt: dict[str, JsonValue] | None


class HistoricalAnomaly(Model):
    kind: Literal["historical_index_overlap"]
    asset_id: str = Field(min_length=1)
    dataset_index: float = Field(ge=0)
    splits: list[Literal["train", "val", "test"]] = Field(min_length=2)
    observation_ids: list[str] = Field(min_length=2)
    semantic_leakage_confirmed: Literal[False]


class Snapshot(Model):
    schema_name: Literal["audit-evidence-v1"] = Field(alias="schema")
    contract_revision: Literal["1.1"]
    assets: list[SourceAsset]
    events: list[CandidateEvent]
    decisions: list[ReviewDecision]
    observation_count: int = Field(ge=0)
    last_import: dict[str, JsonValue] | None
    external_evidence: list[ExternalEvidence]
    companions: list[Companion]
    anomalies: list[HistoricalAnomaly]
    legacy_review_warnings: list[str]


class DecisionExport(Model):
    schema_name: Literal["audit-evidence-v1"] = Field(alias="schema")
    contract_revision: Literal["1.1"]
    decisions: list[ReviewDecision]
