"""Strict external evidence import and conservative cores; no exact-cut invention."""

from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import Field, field_validator, model_validator

from flir_pipeline.data.local_images import declared_file
from flir_pipeline.sequences.experiments.artifacts import binding, inspect, publish
from flir_pipeline.sequences.experiments.config import StrictModel
from flir_pipeline.similarity.storage import file_sha256, read_json, stable_id


class Interval(StrictModel):
    element_id: str = Field(min_length=1)
    timeline_id: str = Field(min_length=1)
    start: int = Field(ge=0, strict=True)
    end: int = Field(ge=0, strict=True)
    kind: Literal[
        "boundary_zone",
        "sequence_core",
        "sequence_core_candidate",
        "sequence_instance",
        "known_source_video",
    ]
    decision: Literal["supported", "ambiguous", "unsupported", "candidate"]
    notes: str

    @model_validator(mode="after")
    def ordered(self):
        if self.end < self.start:
            raise ValueError("Interval end precedes start")
        if (self.kind == "sequence_core_candidate") != (self.decision == "candidate"):
            raise ValueError("Candidate core evidence must remain a candidate")
        return self


class Observation(StrictModel):
    observation_id: str = Field(min_length=1)
    category: Literal["provenance", "recurrence"]
    subject_id: str = Field(min_length=1)
    object_id: str = Field(min_length=1)
    status: Literal["candidate", "lineage_supported", "ambiguous", "unsupported"]
    measurements: dict[str, float | str | bool | None]
    notes: str


class EvidenceEnvelope(StrictModel):
    """Explicit normalized adapter; legacy filenames never establish identity.

    The sidecar lists producer files by SHA256; import re-reads each one. Schema
    translation is explicit external work when an unknown legacy schema exists.
    Neither lineage_supported nor a manual decision is byte identity.
    """

    schema_version: Literal["sequence_evidence_import_v1"] = (
        "sequence_evidence_import_v1"
    )
    artifact_kind: Literal[
        "provenance_decision_v1",
        "video11_manual_transition_review_v2",
        "video11_sequence_structure_v1",
        "video11_core_recurrence_refined_v1",
        "sequence_structure_external_v1",
    ]
    dataset_id: str
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    family: str
    producer_files: dict[str, str]
    reviewer: str = Field(min_length=1)
    reviewed_at: str = Field(min_length=1)
    ground_truth: Literal[False]
    split_created: Literal[False]
    automatic_confirmation: Literal[False]
    exhaustive_boundary_review: bool = False
    intervals: tuple[Interval, ...] = ()
    observations: tuple[Observation, ...] = ()

    @field_validator(
        "ground_truth", "split_created", "automatic_confirmation", mode="before"
    )
    @classmethod
    def false_boolean(cls, value):
        if value is not False:
            raise ValueError("Evidence semantics require the literal boolean false")
        return value

    @model_validator(mode="after")
    def valid(self):
        from datetime import datetime

        if (
            datetime.fromisoformat(self.reviewed_at.replace("Z", "+00:00")).tzinfo
            is None
        ):
            raise ValueError("reviewed_at requires a timezone")
        if not self.producer_files:
            raise ValueError("Import must bind actual producer files")
        import re

        if any(
            re.fullmatch(r"[0-9a-f]{64}", v) is None
            for v in self.producer_files.values()
        ):
            raise ValueError("Invalid producer checksum")
        for items, attr in (
            (self.intervals, "element_id"),
            (self.observations, "observation_id"),
        ):
            if len({getattr(i, attr) for i in items}) != len(items):
                raise ValueError("Duplicate/conflicting external evidence IDs")
        return self


def expected_binding(source):
    return {
        "dataset_id": source.signature["dataset_id"],
        "manifest_sha256": source.signature["checksums"]["manifest_sha256"],
        "family": source.family,
    }


def import_evidence(path: Path, source, output: Path):
    source.safe_output(output, path)
    envelope = EvidenceEnvelope.model_validate(read_json(path))
    if {k: getattr(envelope, k) for k in expected_binding(source)} != expected_binding(
        source
    ):
        raise ValueError(
            "External evidence belongs to different dataset/manifest/family"
        )
    for name, digest in envelope.producer_files.items():
        if file_sha256(declared_file(path.parent, name)) != digest:
            raise ValueError(f"External producer file changed: {name}")
    return publish_normalized(
        envelope,
        source,
        output,
        {
            "external_sha256": file_sha256(path),
            "producer_files": envelope.producer_files,
        },
    )


def publish_normalized(
    envelope, source, output, provenance, *, adapter="explicit_envelope_v1"
):
    """Common strict publication path for inspected producer adapters and envelopes."""
    source.safe_output(output)
    if {k: getattr(envelope, k) for k in expected_binding(source)} != expected_binding(
        source
    ):
        raise ValueError(
            "External evidence belongs to different dataset/manifest/family"
        )
    intervals = pd.DataFrame(
        [i.model_dump() for i in envelope.intervals],
        columns=list(Interval.model_fields),
    )
    validate_intervals(intervals, source)
    if len(intervals):
        intervals = intervals.sort_values(
            ["timeline_id", "kind", "start", "element_id"]
        ).reset_index(drop=True)
    observations = pd.DataFrame(
        [
            {
                **o.model_dump(exclude={"measurements"}),
                "measurements_json": __import__("json").dumps(
                    o.measurements, sort_keys=True, allow_nan=False
                ),
            }
            for o in envelope.observations
        ],
        columns=[*list(Observation.model_fields)[:-2], "notes", "measurements_json"],
    )
    if len(observations):
        observations = observations.sort_values("observation_id").reset_index(drop=True)
    membership, cores = structure_membership(source, intervals)
    source.unchanged()
    return publish(
        output,
        "sequence_structure_review_v1",
        {
            "adapter": adapter,
            "exhaustive_boundary_review": envelope.exhaustive_boundary_review,
        },
        {
            "input": source.signature,
            **provenance,
        },
        {
            "intervals": intervals,
            "observations": observations,
            "membership": membership,
            "cores": cores,
            "occurrences": source.records.copy(),
        },
        {
            "interval_count": len(intervals),
            "core_count": len(cores),
            "reviewer": envelope.reviewer,
            "reviewed_at": envelope.reviewed_at,
            "producer_kind": envelope.artifact_kind,
        },
        extras={"external_evidence": envelope.model_dump(mode="json")},
    )


def validate_intervals(intervals, source):
    if list(intervals) != list(Interval.model_fields):
        raise ValueError("Invalid exact interval schema")
    timelines = source.timelines()
    if not intervals.element_id.is_unique:
        raise ValueError("Conflicting interval IDs")
    for item in intervals.to_dict("records"):
        row = Interval.model_validate(item)
        timeline = timelines.loc[timelines.timeline_id.eq(row.timeline_id)]
        if (
            timeline.empty
            or row.start < timeline.position.min()
            or row.end > timeline.position.max()
        ):
            raise ValueError("Interval outside the source timeline")
    supported = intervals.loc[intervals.decision.isin(["supported", "candidate"])]
    for _, group in supported.groupby(["timeline_id", "kind"]):
        group = group.sort_values("start")
        if (group.start.to_numpy()[1:] <= group.end.to_numpy()[:-1]).any():
            raise ValueError("Overlapping/conflicting supported intervals")
    # An exact membership claim must never assign an uncertain boundary frame.
    for zone in intervals.loc[
        intervals.kind.eq("boundary_zone") & intervals.decision.ne("unsupported")
    ].itertuples():
        exact = supported.loc[
            supported.timeline_id.eq(zone.timeline_id)
            & supported.kind.isin(
                ["sequence_core", "sequence_core_candidate", "sequence_instance"]
            )
        ]
        if (exact.start.le(zone.end) & exact.end.ge(zone.start)).any():
            raise ValueError("Exact membership overlaps an unresolved boundary zone")


def structure_membership(source, intervals):
    """Complement of supported zones yields core CANDIDATES, never instances.

    Labels require temporal continuity in the available grid. Missing positions
    split conservative cores. Unreviewed timelines get no core labels.
    """
    supported = intervals.loc[intervals.decision.eq("supported")]
    cores = []
    for timeline, group in source.timelines().groupby("timeline_id", sort=True):
        legacy = intervals.loc[
            intervals.timeline_id.eq(timeline)
            & intervals.kind.eq("sequence_core_candidate")
            & intervals.decision.eq("candidate")
        ]
        explicit = supported.loc[
            supported.timeline_id.eq(timeline) & supported.kind.eq("sequence_core")
        ]
        if len(legacy):
            if len(explicit):
                raise ValueError(
                    "Cannot silently reconcile legacy candidate and reviewed cores"
                )
            cores.extend(
                {
                    **row,
                    "kind": "sequence_core",
                    "status": "candidate",
                    "origin": "imported_legacy_core_candidate",
                }
                for row in legacy.to_dict("records")
            )
            continue
        if len(explicit):
            cores.extend(
                {**row, "status": "candidate", "origin": "explicit_manual_core"}
                for row in explicit.to_dict("records")
            )
            continue
        zones = supported.loc[
            supported.timeline_id.eq(timeline) & supported.kind.eq("boundary_zone")
        ]
        if zones.empty:
            continue
        outside = group.loc[
            ~group.position.apply(
                lambda p, zones=zones: any(
                    z.start <= p <= z.end for z in zones.itertuples()
                )
            )
        ]
        for _, core in outside.groupby(outside.position.diff().ne(1).cumsum()):
            start, end = int(core.position.min()), int(core.position.max())
            cores.append(
                {
                    "element_id": stable_id(
                        {
                            "kind": "conservative_core",
                            "dataset_id": source.signature["dataset_id"],
                            "timeline": timeline,
                            "start": start,
                            "end": end,
                        }
                    ),
                    "timeline_id": timeline,
                    "start": start,
                    "end": end,
                    "kind": "sequence_core",
                    "decision": "",
                    "status": "candidate",
                    "origin": "conservative_complement_of_reviewed_zones",
                    "notes": "conservative complement candidate",
                }
            )
    core_table = pd.DataFrame(
        cores, columns=[*Interval.model_fields, "status", "origin"]
    )
    rows = []
    for target, exact in (
        ("sequence_core", core_table),
        ("sequence_instance", supported.loc[supported.kind.eq("sequence_instance")]),
        ("known_source_video", supported.loc[supported.kind.eq("known_source_video")]),
    ):
        for record in source.records.itertuples():
            zone = intervals.loc[
                intervals.timeline_id.eq(record.timeline_id)
                & intervals.kind.eq("boundary_zone")
                & intervals.decision.ne("unsupported")
            ]
            uncertain = pd.isna(record.position) or any(
                z.start <= record.position <= z.end for z in zone.itertuples()
            )
            options = (
                exact.loc[
                    exact.timeline_id.eq(record.timeline_id)
                    & exact.start.le(record.position)
                    & exact.end.ge(record.position)
                ]
                if not pd.isna(record.position)
                else exact.iloc[:0]
            )
            valid = not uncertain and len(options) == 1
            rows.append(
                {
                    "frame_id": record.frame_id,
                    "content_id": record.content_id,
                    "timeline_id": record.timeline_id,
                    "position": record.position,
                    "target": target,
                    "target_label": str(options.element_id.iloc[0]) if valid else "",
                    "evaluation_mask": valid,
                    "exclusion_reason": ""
                    if valid
                    else (
                        "boundary_zone_or_unknown_position"
                        if uncertain
                        else "unreviewed_membership"
                    ),
                }
            )
    return pd.DataFrame(rows), core_table


def load_structure(directory, source):
    meta = inspect(directory)
    if (
        meta["artifact_kind"] != "sequence_structure_review_v1"
        or meta["identity"]["sources"]["input"] != source.signature
    ):
        raise ValueError("Structure review is not bound to these sources")
    if "native_producers" in meta["identity"]["sources"]:
        from flir_pipeline.sequences.experiments.native_evidence import verify_native

        verify_native(directory, source)
    intervals = pd.read_parquet(directory / "intervals.parquet")
    validate_intervals(intervals, source)
    membership, cores = structure_membership(source, intervals)
    pd.testing.assert_frame_equal(
        membership, pd.read_parquet(directory / "membership.parquet")
    )
    pd.testing.assert_frame_equal(cores, pd.read_parquet(directory / "cores.parquet"))
    return membership, cores, intervals, binding(directory)


def content_targets(membership, contents, target):
    rows = []
    selected = membership.loc[membership.target.eq(target)]
    for content in contents:
        occurrences = selected.loc[selected.content_id.eq(content)]
        labels = set(occurrences.loc[occurrences.evaluation_mask, "target_label"])
        valid = bool(
            len(occurrences) and occurrences.evaluation_mask.all() and len(labels) == 1
        )
        rows.append(
            {
                "content_id": content,
                "target": target,
                "target_label": next(iter(labels)) if valid else "",
                "evaluation_mask": valid,
                "occurrence_count": len(occurrences),
                "exclusion_reason": ""
                if valid
                else "unresolved_or_conflicting_occurrence_membership",
            }
        )
    return pd.DataFrame(rows)
