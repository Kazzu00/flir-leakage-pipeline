"""Versioned structural contracts; runtime adds content and reviewer invariants."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Provenance(BaseModel):
    model_config = ConfigDict(extra="allow")
    dataset_index: int = Field(ge=0)
    original_split: Literal["train", "val", "test"]


class Source(BaseModel):
    model_config = ConfigDict(extra="allow")
    image_id: str = Field(min_length=1)
    original: Provenance
    alignment_id: str = Field(min_length=1)
    alignment_metrics: dict[str, Any]
    time_uncertainty: Any
    media: list[str] = Field(min_length=1)


class Pair(BaseModel):
    model_config = ConfigDict(extra="allow")
    pair_id: str = Field(min_length=1)
    category: Literal[
        "diagnostic",
        "nearby",
        "nonconsecutive",
        "change_candidate",
        "negative_candidate",
    ]
    a: Source
    b: Source
    metrics: dict[str, Any]
    media: list[str] = Field(min_length=2)
    secondary: dict[str, Any]
    status: Literal["pending"]


class Media(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str = Field(min_length=1)
    sha256: str = Field(pattern="^[0-9a-f]{64}$")


class Bundle(BaseModel):
    model_config = ConfigDict(extra="allow")
    schema_version: Literal["dependency-pilot-v1"] = Field(alias="schema")
    pairs: list[Pair]
    media: dict[str, Media]
    second_reviewer_pairs: list[str]
    method: dict[str, Any]
    source_video: str = Field(min_length=1)
