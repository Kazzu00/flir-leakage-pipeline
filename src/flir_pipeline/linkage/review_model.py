"""Manual group-level calibration: decisions, identity and explicit denominators."""

import json
from datetime import UTC, datetime
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from flir_pipeline.similarity.storage import stable_id

KIND = "labeled_visual_dependency_manual_calibration"
DECISIONS = ("supported", "ambiguous", "unsupported", "")
KEYS = ["labeled_content_id", "proposed_visual_dependency_group_id"]
MANUAL_COLUMNS = [
    "manual_decision",
    "manual_notes",
    "reviewer",
    "decision_source",
    "reviewed_at_utc",
]
SEMANTICS = {
    "ground_truth": False,
    "confirmed_matches_created": False,
    "split_created": False,
    "automatic_confirmation": False,
    "leakage_safe_split_exists": False,
    "review_unit": "one labeled query and its proposed visual dependency group",
    "target": "evidence for group-level linkage; not exact frame or sequence-instance identification",
    "visual_dependency_group": "existing must-link split constraint; does not merge sequence instances",
    "sequence_instance": "existing temporally continuous segment",
    "rank_agreement": "evidence only; no automatic truth or decision threshold",
    "encoder_cosines": "separate CLIP and DINOv2 values; never averaged",
    "statistics": "stratified manual calibration counts/rates; not representative accuracy estimates",
    "unresolved": "blank or ambiguous decision",
    "decision_rates_denominator": "all sampled queries in the corresponding stratum/group, including blanks",
}


class ReviewConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    context_seconds: int = Field(default=3, ge=0, le=30, strict=True)
    protocol: Literal["manual_group_calibration_v1"] = "manual_group_calibration_v1"
    temporal_policy: Literal["all_candidate_occurrences_same_video_context"] = (
        "all_candidate_occurrences_same_video_context"
    )
    rendering_policy: Literal["rgb_png_240x180_default_font_v1"] = (
        "rgb_png_240x180_default_font_v1"
    )


def canonical_json(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def frame_records(frame):
    """JSON-safe values without silently dropping nullable occurrence metadata."""
    return json.loads(frame.to_json(orient="records", double_precision=15))


def read_csv(path):
    # Identity strings such as 'NA' and empty manual decisions are not pandas nulls.
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def require_strings(frame, columns):
    if not set(columns) <= set(frame):
        raise ValueError(f"Missing columns: {sorted(set(columns) - set(frame))}")
    for column in columns:
        if (
            not frame[column]
            .map(lambda x: isinstance(x, str) and bool(x.strip()))
            .all()
        ):
            raise ValueError(f"Empty or invalid identity: {column}")


def validate_sample(sample):
    require_strings(sample, [*KEYS, "review_stratum"])
    if sample.empty or not sample.labeled_content_id.is_unique:
        raise ValueError(
            "Calibration requires exactly one proposed group per labeled query; duplicate queries are not decisions per occurrence"
        )
    return sample.sort_values(KEYS).reset_index(drop=True)


def validate_decisions(review):
    require_strings(review, [*KEYS, "review_stratum", "review_query_id"])
    if not set(MANUAL_COLUMNS) <= set(review):
        raise ValueError("Missing manual decision/provenance columns")
    if (
        not review.labeled_content_id.is_unique
        or not review.review_query_id.is_unique
        or review.duplicated(KEYS).any()
    ):
        raise ValueError(
            "Review must contain one decision per labeled query/group pair"
        )
    if not review.manual_decision.isin(DECISIONS).all():
        raise ValueError(
            "Allowed manual decisions: supported, ambiguous, unsupported, or blank"
        )
    if not review[MANUAL_COLUMNS].map(lambda x: isinstance(x, str)).all().all():
        raise ValueError("Manual fields must be strings, with blanks preserved")
    for row in review.itertuples():
        has_history = any((row.reviewer, row.decision_source, row.reviewed_at_utc))
        if has_history:
            if not row.reviewer.strip() or not row.decision_source.strip():
                raise ValueError("Recorded decisions require reviewer and source")
            validate_timestamp(row.reviewed_at_utc)
        elif row.manual_decision or row.manual_notes:
            raise ValueError(
                "Manual decisions/notes require explicit reviewer provenance"
            )


def validate_timestamp(value):
    moment = datetime.fromisoformat(value)
    if moment.tzinfo is None or moment.utcoffset().total_seconds() != 0:
        raise ValueError("Review timestamps must be timezone-aware UTC")
    return moment


def decision_summary(review):
    validate_decisions(review)

    def counts(frame):
        total = len(frame)
        decisions = {
            decision or "blank": int(frame.manual_decision.eq(decision).sum())
            for decision in DECISIONS
        }
        return {
            "query_count": total,
            "distinct_group_count": int(frame[KEYS[1]].nunique()),
            "decision_counts": decisions,
            "decision_rates": {
                key: value / total if total else None
                for key, value in decisions.items()
            },
            "unresolved_count": decisions["blank"] + decisions["ambiguous"],
        }

    return {
        "semantics": SEMANTICS,
        "overall": counts(review),
        "by_stratum": [
            {"review_stratum": key, **counts(group)}
            for key, group in review.groupby("review_stratum", sort=True)
        ],
        "by_visual_dependency_group": [
            {"visual_dependency_group_id": key, **counts(group)}
            for key, group in review.groupby(KEYS[1], sort=True)
        ],
        "by_stratum_and_group": [
            {
                "review_stratum": key[0],
                "visual_dependency_group_id": key[1],
                **counts(group),
            }
            for key, group in review.groupby(["review_stratum", KEYS[1]], sort=True)
        ],
    }


def apply_decisions(review, decisions, *, reviewer, source, timestamp=None):
    """Apply an explicit import; preserve readonly evidence and record before/after.

    Blank rows in a full CSV are valid. Changing a previous decision to blank is
    an auditable clearing action, not an omitted update. Absent pairs are untouched.
    """
    validate_decisions(review)
    require_strings(decisions, KEYS)
    if (
        decisions.empty
        or decisions.duplicated(KEYS).any()
        or not {"manual_decision", "manual_notes"} <= set(decisions)
    ):
        raise ValueError(
            "Decision import requires unique query/group pairs and manual_decision/manual_notes"
        )
    if not decisions.manual_decision.isin(DECISIONS).all():
        raise ValueError("Invalid manual decision vocabulary")
    if not reviewer.strip() or not source.strip():
        raise ValueError("Provide a nonempty reviewer and decision source")
    timestamp = timestamp or datetime.now(UTC).isoformat()
    when = validate_timestamp(timestamp)
    current = review.set_index(KEYS, drop=False)
    changes = []
    unknown = set(decisions) - set(review)
    if unknown:
        raise ValueError(f"Unknown import columns: {sorted(unknown)}")
    for row in decisions.sort_values(KEYS).to_dict("records"):
        key = tuple(row[name] for name in KEYS)
        if key not in current.index:
            raise ValueError(f"Unknown labeled query/group pair: {key}")
        old = current.loc[key]
        for name in set(row) - {"manual_decision", "manual_notes"}:
            if row[name] != old[name]:
                raise ValueError(
                    f"Import changed readonly provenance/evidence column: {name}"
                )
        if (
            row["manual_decision"] == old.manual_decision
            and row["manual_notes"] == old.manual_notes
        ):
            continue
        if old.reviewed_at_utc and when < validate_timestamp(old.reviewed_at_utc):
            raise ValueError("Review timestamp predates the previous decision")
        after = {
            "manual_decision": row["manual_decision"],
            "manual_notes": row["manual_notes"],
            "reviewer": reviewer,
            "decision_source": source,
            "reviewed_at_utc": timestamp,
        }
        changes.append(
            {
                **dict(zip(KEYS, key, strict=True)),
                "before": {name: old[name] for name in MANUAL_COLUMNS},
                "after": after,
            }
        )
        for name, value in after.items():
            current.loc[key, name] = value
    result = current.reset_index(drop=True)
    validate_decisions(result)
    return result, changes


def event_identity(parent_id, import_sha256, reviewer, source, timestamp, changes):
    payload = {
        "parent_revision_id": parent_id,
        "import_sha256": import_sha256,
        "reviewer": reviewer,
        "source": source,
        "recorded_at_utc": timestamp,
        "changes": changes,
    }
    return {"event_id": stable_id(payload), **payload}
