"""Consume the frozen manual CSV contract without promoting candidates to truth."""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    model_validator,
)

from flir_pipeline.similarity.storage import file_sha256, read_json

VALIDATION_FILES = (
    "boundary_validation.csv",
    "accepted_boundary_candidates.csv",
    "metadata.json",
)
REVIEW_COLUMNS = (
    "event_id",
    "video_id",
    "coarse_sample_index",
    "search_start",
    "search_end",
    "localized_sample_index",
    "localization_shift",
    "f3_score",
    "persistent_stable_min",
    "persistent_stable_median",
    "high_confidence",
    "repeat_partner_sample_index",
    "decision",
    "boundary_type",
    "notes",
    "review_status",
    "sequence_boundary_committed",
    "manual_confirmation",
)


class ReviewMetadata(BaseModel):
    """Only the explicit confirmed-review contract can authorize a sequence cut."""

    model_config = ConfigDict(extra="forbid", strict=True)
    artifact_kind: Literal["confirmed_manual_boundary_validation"]
    ground_truth: StrictBool
    diagnostic_source: str = Field(min_length=1)
    review_status: Literal["confirmed_manual_review"]
    manual_confirmation_complete: StrictBool
    manual_confirmation_required_before_sequence_commit: StrictBool
    sequence_boundaries_committed: StrictBool
    confirmation_timestamp_utc: str
    event_count: int = Field(ge=0)
    accepted_count: int = Field(ge=0)
    rejected_count: int = Field(ge=0)
    boundary_type_counts: dict[str, int]
    source_checksums: dict[str, str]

    @model_validator(mode="after")
    def confirmed(self):
        if (
            self.ground_truth
            or not self.manual_confirmation_complete
            or self.manual_confirmation_required_before_sequence_commit
        ):
            raise ValueError("Requires confirmed manual review with ground_truth=false")
        stamp = datetime.fromisoformat(self.confirmation_timestamp_utc)
        if stamp.utcoffset() is None or stamp.utcoffset().total_seconds() != 0:
            raise ValueError("Confirmation timestamp must be UTC")
        if not self.source_checksums or any(
            len(v) != 64 or any(c not in "0123456789abcdef" for c in v)
            for v in self.source_checksums.values()
        ):
            raise ValueError("Review provenance must declare SHA256 checksums")
        return self


class ReviewRow(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    event_id: int = Field(ge=0)
    video_id: str = Field(min_length=1)
    coarse_sample_index: int = Field(gt=0)
    search_start: int = Field(gt=0)
    search_end: int = Field(gt=0)
    localized_sample_index: int = Field(gt=0)
    localization_shift: StrictInt
    f3_score: float = Field(gt=0, le=1, allow_inf_nan=False)
    persistent_stable_min: float = Field(gt=0, le=1, allow_inf_nan=False)
    persistent_stable_median: float = Field(gt=0, le=1, allow_inf_nan=False)
    high_confidence: StrictBool
    repeat_partner_sample_index: int | None = Field(default=None, ge=0)
    decision: Literal["accept", "reject"]
    boundary_type: Literal[
        "scene_change", "degradation_transition", "transition_interval", "reject"
    ]
    notes: str
    review_status: Literal["confirmed_manual_review"]
    sequence_boundary_committed: StrictBool
    manual_confirmation: StrictBool

    @model_validator(mode="after")
    def consistent(self):
        if not self.manual_confirmation:
            raise ValueError("Every review row requires manual confirmation")
        if (self.decision == "reject") != (self.boundary_type == "reject"):
            raise ValueError("Decision and boundary_type disagree")
        if (
            not self.search_start <= self.localized_sample_index <= self.search_end
            or self.localization_shift
            != self.localized_sample_index - self.coarse_sample_index
        ):
            raise ValueError("Invalid localized review position/shift")
        return self


def _review_table(path: Path) -> pd.DataFrame:
    # Read strings first: pandas' bool('False') and float-to-int casts must never
    # turn an unconfirmed/invalid CSV row into a valid review.
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if set(frame) != set(REVIEW_COLUMNS):
        raise ValueError("Unexpected confirmed validation CSV schema")
    integers = (
        "event_id",
        "coarse_sample_index",
        "search_start",
        "search_end",
        "localized_sample_index",
        "localization_shift",
        "repeat_partner_sample_index",
    )
    numbers = ("f3_score", "persistent_stable_min", "persistent_stable_median")
    booleans = ("high_confidence", "sequence_boundary_committed", "manual_confirmation")
    rows = []
    for row in frame.to_dict("records"):
        for name in integers:
            value = row[name]
            if name == "repeat_partner_sample_index" and value.lower() in {"", "nan"}:
                row[name] = None
            else:
                # Optional pandas nullable partner columns often serialize 12.0.
                number = float(value)
                if not np.isfinite(number) or number != int(number):
                    raise ValueError(f"Review {name} must be an integer")
                row[name] = int(number)
        for name in numbers:
            row[name] = float(row[name])
        for name in booleans:
            if row[name].lower() not in {"true", "false"}:
                raise ValueError(f"Review {name} must be explicitly true or false")
            row[name] = row[name].lower() == "true"
        rows.append(ReviewRow.model_validate(row).model_dump())
    dtypes = {
        name: "Int64"
        if name == "repeat_partner_sample_index"
        else "int64"
        if name in integers
        else "float64"
        if name in numbers
        else "bool"
        if name in booleans
        else "string"
        for name in REVIEW_COLUMNS
    }
    result = pd.DataFrame(rows, columns=REVIEW_COLUMNS).astype(dtypes)
    if (
        not result.event_id.is_unique
        or result.duplicated(["video_id", "coarse_sample_index"]).any()
    ):
        raise ValueError("Review event identities must be unique")
    return result.sort_values("event_id").reset_index(drop=True)


def load_confirmed_validation(
    directory: Path, events: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    """Match frozen review geometry to recomputed candidates, not historical labels.

    The original review predates this stage and has no manifest/feature binding.
    Its provisional-source checksums are retained as declarations, not claimed
    reverified. This run binds the actual three review files plus recomputed
    geometry/scores to the supplied manifest and both complete feature stores.
    """
    initial_files = {name: file_sha256(directory / name) for name in VALIDATION_FILES}
    metadata = ReviewMetadata.model_validate(read_json(directory / "metadata.json"))
    review = _review_table(directory / "boundary_validation.csv")
    accepted_file = _review_table(directory / "accepted_boundary_candidates.csv")
    accepted = review.loc[review.decision.eq("accept")].reset_index(drop=True)
    pd.testing.assert_frame_equal(accepted_file, accepted, check_exact=True)
    counts = {str(k): int(v) for k, v in review.boundary_type.value_counts().items()}
    if (
        metadata.event_count != len(review)
        or metadata.accepted_count != len(accepted)
        or metadata.rejected_count != len(review) - len(accepted)
        or metadata.boundary_type_counts != counts
    ):
        raise ValueError(
            "Confirmed validation metadata counts do not match authoritative CSV"
        )
    # Event ordinals are review provenance, not portable sequence IDs. Match by
    # video/coarse position so an external review's numbering need not be reused.
    if len(review) != len(events):
        raise ValueError("Confirmed review must cover every recomputed candidate event")
    columns = {
        "search_start": "search_start_sample_index",
        "search_end": "search_end_sample_index",
        "localized_sample_index": "localized_sample_index",
        "high_confidence": "high_confidence",
        "f3_score": "localized_F3",
        "persistent_stable_min": "coarse_score",
        "persistent_stable_median": "coarse_median",
    }
    event_map = events.set_index(["video_id", "coarse_sample_index"])
    for row in review.to_dict("records"):
        key = (row["video_id"], row["coarse_sample_index"])
        if key not in event_map.index:
            raise ValueError(
                "Review event has no recomputed same-video coarse candidate"
            )
        event = event_map.loc[key]
        for source, target in columns.items():
            match = (
                np.isclose(row[source], event[target], atol=1e-12, rtol=0)
                if source
                in {"f3_score", "persistent_stable_min", "persistent_stable_median"}
                else row[source] == event[target]
            )
            if not match:
                raise ValueError(
                    f"Confirmed review disagrees with recomputed candidate: {source}"
                )
    files = {name: file_sha256(directory / name) for name in VALIDATION_FILES}
    if files != initial_files:
        raise ValueError("Confirmed validation changed while being read")
    checksum = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return review, {
        "files": files,
        "validation_sha256": checksum,
        "metadata": metadata.model_dump(),
        "source_binding": "recomputed_candidate_geometry_and_scores",
        "provisional_source_checksums_verified": False,
    }
