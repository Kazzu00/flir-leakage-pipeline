"""Explicit namespaces and candidate-only provenance for external image variants."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)

from flir_pipeline.data.local_images import relative_posix_path

KIND = "video_variant_ingestion_v1"
MANIFEST_VERSION = "flir_video_variant_occurrences_v1"
Key = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")]
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Nonnegative = Annotated[int, Field(ge=0, le=2**63 - 1)]


class Model(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, allow_inf_nan=False, strict=True
    )


class InputFile(Model):
    path: str
    expected_sha256: Digest | None = None

    @field_validator("path")
    @classmethod
    def portable_path(cls, value):
        return relative_posix_path(value)


class Archive(InputFile):
    archive_key: Key


class Video(InputFile):
    video_key: Key
    declared_metadata: dict[str, JsonValue] = Field(default_factory=dict)


class Limits(Model):
    max_member_bytes: int = Field(default=64 * 1024 * 1024, gt=0, le=2**63 - 1)
    max_archive_uncompressed_bytes: int = Field(
        default=16 * 1024**3, gt=0, le=2**63 - 1
    )
    max_archive_entries: int = Field(default=250_000, gt=0, le=2**63 - 1)
    max_compression_ratio: float = Field(default=1000.0, ge=1)
    max_width: int = Field(default=8192, gt=0)
    max_height: int = Field(default=8192, gt=0)
    max_pixels: int = Field(default=40_000_000, gt=0)


class CandidateTime(Model):
    """A declared uniform-grid hypothesis, never native indices or verified PTS."""

    fps: float = Field(gt=0)
    index_origin: Nonnegative
    offset_seconds: float = 0.0


class Series(Model):
    series_id: Key
    archive_keys: list[Key] = Field(min_length=1)
    member_pattern: str = ".*"
    index_pattern: str
    expected_index_start: Nonnegative | None = None
    expected_index_stop: Nonnegative | None = None
    expected_index_step: int = Field(default=1, gt=0, le=2**63 - 1)
    candidate_video_key: Key | None = None
    candidate_time: CandidateTime | None = None

    @model_validator(mode="after")
    def valid_rules(self):
        try:
            re.compile(self.member_pattern)
            index_regex = re.compile(self.index_pattern)
        except re.error as error:
            raise ValueError(f"Invalid regular expression: {error}") from error
        if "index" not in index_regex.groupindex:
            raise ValueError("index_pattern requires a named (?P<index>...) group")
        if len(set(self.archive_keys)) != len(self.archive_keys):
            raise ValueError("Repeated archive_keys within a series")
        start, stop = self.expected_index_start, self.expected_index_stop
        if (start is None) != (stop is None):
            raise ValueError(
                "Declare both expected_index_start and expected_index_stop"
            )
        if start is not None and (
            stop < start or (stop - start) % self.expected_index_step
        ):
            raise ValueError("Expected range must be ordered and divisible by its step")
        if self.candidate_time is not None and self.candidate_video_key is None:
            raise ValueError("Candidate timing requires a candidate_video_key")
        return self


class Observation(Model):
    """External reports support candidates; phase one accepts no verified alignment."""

    observation_id: Key
    series_id: Key
    video_key: Key
    description: str = Field(min_length=1)
    observed_index: Nonnegative | None = None
    reported_timestamp_seconds: float | None = Field(default=None, ge=0)
    evidence_path: str | None = None
    reviewer: str | None = None
    reviewed_at: str | None = None
    evidence_status: Literal["externally_reported"] = "externally_reported"
    verification_scope: Literal["none"] = "none"

    @field_validator("evidence_path")
    @classmethod
    def portable_evidence(cls, value):
        return relative_posix_path(value) if value is not None else None


class IngestionConfig(Model):
    schema_version: Literal["video_variant_ingestion_config_v1"] = (
        "video_variant_ingestion_config_v1"
    )
    collection_id: Key
    variant_name: Key
    transformation: dict[str, JsonValue]
    archives: list[Archive] = Field(min_length=1)
    series: list[Series] = Field(min_length=1)
    videos: list[Video] = Field(default_factory=list)
    observations: list[Observation] = Field(default_factory=list)
    auxiliary_inventory: InputFile | None = None
    image_extensions: list[str] = Field(default_factory=lambda: [".png"])
    expected_image_format: str | None = "PNG"
    limits: Limits = Field(default_factory=Limits)

    @model_validator(mode="after")
    def references(self):
        for items, key in (
            (self.archives, "archive_key"),
            (self.series, "series_id"),
            (self.videos, "video_key"),
            (self.observations, "observation_id"),
        ):
            values = [getattr(item, key) for item in items]
            if len(values) != len(set(values)):
                raise ValueError(f"Duplicate {key}")
        if len({a.path for a in self.archives}) != len(self.archives):
            raise ValueError("Do not register the same archive path twice")
        archives = {a.archive_key for a in self.archives}
        videos = {v.video_key for v in self.videos}
        series = {s.series_id for s in self.series}
        for item in self.series:
            if not set(item.archive_keys) <= archives:
                raise ValueError("Unknown series archive key")
            if (
                item.candidate_video_key is not None
                and item.candidate_video_key not in videos
            ):
                raise ValueError("Unknown candidate video key")
        for observation in self.observations:
            if (
                observation.series_id not in series
                or observation.video_key not in videos
            ):
                raise ValueError("Unknown observation series/video")
        if (
            not self.image_extensions
            or any(
                not re.fullmatch(r"\.[a-z0-9]+", extension)
                for extension in self.image_extensions
            )
            or len(set(self.image_extensions)) != len(self.image_extensions)
        ):
            raise ValueError("image_extensions must be unique lowercase suffixes")
        return self


def digest_document(value) -> str:
    """Hash the explicit recipe, with paths/times included only by its caller."""
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def scientific_frame_id(
    collection: str, series: str, index: int, image_sha256: str
) -> str:
    """Storage never enters this ID; callers must first establish a unique index.

    The index denotes the externally named observation, not a native video frame.
    Repeated indices have no uniquely attributable scientific occurrence in v1.
    Their physical entries survive with null frame_id until explicit adjudication.
    """
    return digest_document([MANIFEST_VERSION, collection, series, index, image_sha256])


def load_config(path: Path) -> IngestionConfig:
    try:
        return IngestionConfig.model_validate(
            yaml.safe_load(path.read_text(encoding="utf-8"))
        )
    except re.error as error:
        raise ValueError(f"Invalid regular expression: {error}") from error
