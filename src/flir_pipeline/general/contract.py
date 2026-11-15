"""Versioned portable evidence, occurrence and class contracts.

An accepted dependency is a decision about co-assignment, not a similarity
threshold. A rejected edge is a cannot-link constraint on graph components,
not a request to force its endpoints into different dataset partitions.
"""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SHA = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
ID = Annotated[str, Field(min_length=1)]


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode()
    ).hexdigest()


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Evidence(Contract):
    path: ID
    sha256: SHA
    method: ID
    version: ID
    provenance: ID
    reviewer: str | None = None
    reviewed_at: str | None = None


class ClassEntry(Contract):
    # Sparse original IDs are preserved; no positional renumbering.
    class_id: Annotated[int, Field(strict=True, ge=0)]
    name: ID


class Video(Contract):
    video_id: ID
    path: ID
    sha256: SHA


class Position(Contract):
    video_id: ID
    status: Literal["confirmed", "candidate", "ambiguous", "rejected"]
    kind: Literal["decoded_pts", "nominal_sample", "external_timestamp"]
    seconds: float
    decoded_ordinal: Annotated[int, Field(strict=True, ge=0)] | None = None
    pts: Annotated[int, Field(strict=True)] | None = None
    time_base: str | None = None
    decoded_rgb_sha256: SHA | None = None
    evidence: Evidence

    @model_validator(mode="after")
    def observed_position(self):
        if self.kind == "decoded_pts":
            if None in (
                self.decoded_ordinal,
                self.pts,
                self.time_base,
                self.decoded_rgb_sha256,
            ):
                raise ValueError(
                    "Decoded positions require ordinal, PTS, time base and raster SHA256"
                )
            base = Fraction(self.time_base)
            if base <= 0 or abs(float(self.pts * base) - self.seconds) > 1e-8:
                raise ValueError("Seconds must equal observed PTS times time_base")
        elif any(
            x is not None
            for x in (
                self.decoded_ordinal,
                self.pts,
                self.time_base,
                self.decoded_rgb_sha256,
            )
        ):
            raise ValueError("Nominal/external timestamps cannot claim decoded PTS")
        if self.kind == "nominal_sample" and self.status == "confirmed":
            raise ValueError("Nominal sampling is not confirmed image localization")
        return self


class Occurrence(Contract):
    frame_id: ID
    content_id: SHA
    image_sha256: SHA
    source: ID
    original_name: ID
    original_split: str | None = None
    annotation_status: Literal["missing", "empty", "valid", "invalid"] = "missing"
    label_sha256: SHA | None = None
    label_source: str | None = None
    boxes: list[list[float | int]] = Field(default_factory=list)
    annotation_errors: list[str] = Field(default_factory=list)
    positions: list[Position] = Field(default_factory=list)
    exclusion_reasons: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def identity_and_labels(self):
        if self.content_id != self.image_sha256:
            raise ValueError("Exact content identity must equal image byte SHA256")
        if self.annotation_status == "missing":
            if (
                self.label_sha256 is not None
                or self.label_source is not None
                or self.boxes
            ):
                raise ValueError("Missing annotation cannot contain labels")
        elif not self.label_sha256 or not self.label_source:
            raise ValueError(
                "Present annotations require their original byte checksum and locator"
            )
        if self.annotation_status == "empty" and self.boxes:
            raise ValueError("Empty annotation has no objects")
        if self.annotation_status == "valid" and not self.boxes:
            raise ValueError("Use empty for a valid annotation with no objects")
        if self.annotation_status == "invalid" and not self.annotation_errors:
            raise ValueError("Invalid labels require explicit QA errors")
        return self


class Relation(Contract):
    relation_id: ID
    left: SHA
    right: SHA
    kind: Literal["exact", "validated_variant", "visual", "temporal", "external"]
    status: Literal["confirmed", "candidate", "ambiguous", "rejected"]
    accepted: bool = False
    score: float | None = None
    evidence: Evidence

    @model_validator(mode="after")
    def explicit_acceptance(self):
        if self.kind == "exact" and self.left != self.right:
            raise ValueError("Different byte identities cannot be an exact relation")
        if self.accepted and self.status != "confirmed":
            raise ValueError("Only expressly confirmed dependencies can be accepted")
        if (
            self.accepted
            and self.kind != "exact"
            and not (self.evidence.reviewer and self.evidence.reviewed_at)
        ):
            raise ValueError("Nonexact acceptance requires an attributed dated review")
        return self


class Manifest(Contract):
    schema_version: Literal["general_manifest_v1"] = "general_manifest_v1"
    classes: list[ClassEntry] = Field(default_factory=list)
    videos: list[Video] = Field(default_factory=list)
    records: list[Occurrence]
    relations: list[Relation] = Field(default_factory=list)
    sources: list[Evidence] = Field(default_factory=list)

    @model_validator(mode="after")
    def referential_integrity(self):
        if not self.records:
            raise ValueError("Empty datasets cannot be partitioned")
        for values in (
            [r.frame_id for r in self.records],
            [c.class_id for c in self.classes],
            [v.video_id for v in self.videos],
            [r.relation_id for r in self.relations],
        ):
            if len(values) != len(set(values)):
                raise ValueError("Duplicate contract identifiers")
        classes = {c.class_id for c in self.classes}
        videos = {v.video_id for v in self.videos}
        contents = {r.content_id for r in self.records}
        for r in self.records:
            for b in r.boxes:
                if len(b) != 5 or type(b[0]) is not int or b[0] not in classes:
                    raise ValueError(
                        "Boxes must retain original catalog class IDs and four coordinates"
                    )
                if r.annotation_status != "invalid":
                    _, x, y, w, h = b
                    if not (
                        0 < w <= 1
                        and 0 < h <= 1
                        and 0 <= x - w / 2 <= 1
                        and 0 <= x + w / 2 <= 1
                        and 0 <= y - h / 2 <= 1
                        and 0 <= y + h / 2 <= 1
                    ):
                        raise ValueError(
                            "Valid boxes must have normalized in-image geometry"
                        )
            if any(p.video_id not in videos for p in r.positions):
                raise ValueError("Temporal positions must reference checksummed videos")
        if any(
            r.left not in contents or r.right not in contents for r in self.relations
        ):
            raise ValueError("Relation endpoint outside manifest")
        return self

    def canonical(self) -> dict:
        data = self.model_dump(mode="json")
        for name, key in (
            ("records", "frame_id"),
            ("relations", "relation_id"),
            ("classes", "class_id"),
            ("videos", "video_id"),
        ):
            data[name].sort(key=lambda r: r[key])
        for row in data["records"]:
            row["positions"].sort(key=digest)
            row["exclusion_reasons"].sort()
        data["sources"].sort(key=digest)
        return data

    def dataset_id(self) -> str:
        # Reuse the historical identity recipe with a distinct manifest version;
        # absence of labels has an explicit sentinel, unlike SHA256(empty bytes).
        import pandas as pd

        from flir_pipeline.data.identity import dataset_id_from_manifest

        return dataset_id_from_manifest(
            pd.DataFrame(
                [
                    {
                        "frame_id": r.frame_id,
                        "image_sha256": r.image_sha256,
                        "label_sha256": r.label_sha256 or "missing",
                        "manifest_version": self.schema_version,
                    }
                    for r in self.records
                ]
            )
        )
