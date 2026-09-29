"""Generic image-variant provenance, independent of any particular transformation.

Variant identity scopes data/artifacts, never changes the mathematical encoder
space. Manifest byte checksums bind provenance separately from semantic identity.
Legacy stores remain explicitly unspecified; naming a variant cannot relabel them.
"""

import hashlib
import json
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from flir_pipeline.data.identity import dataset_id_from_manifest


class ParentDataset(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    dataset_id: str = Field(min_length=1)
    dataset_variant_id: str | None = None


class Variant(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    schema_version: Literal["dataset_variant_v1"] = "dataset_variant_v1"
    dataset_id: str = Field(min_length=1)
    dataset_variant_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    variant_name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    parent_dataset_identity: ParentDataset | None = None
    definition: dict = Field(default_factory=dict)
    source_checksums: dict[str, str]


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def variant_id(document):
    fields = {
        k: document[k]
        for k in (
            "schema_version",
            "dataset_id",
            "variant_name",
            "parent_dataset_identity",
            "definition",
        )
    }
    return hashlib.sha256(
        json.dumps(
            fields, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def make_variant(
    dataset_id, manifest_sha256, name="unspecified", *, parent=None, definition=None
):
    document = dict(
        schema_version="dataset_variant_v1",
        dataset_id=dataset_id,
        variant_name=name,
        parent_dataset_identity=parent,
        definition=definition or {},
        source_checksums={"manifest_sha256": manifest_sha256},
    )
    document["dataset_variant_id"] = variant_id(document)
    return validate_variant(document)


def validate_variant(document, dataset_id=None, manifest_sha256=None):
    value = Variant.model_validate(document).model_dump(mode="json")
    if value["dataset_variant_id"] != variant_id(value):
        raise ValueError("Dataset variant identity differs from its declaration")
    if (
        set(value["source_checksums"]) != {"manifest_sha256"}
        or len(value["source_checksums"]["manifest_sha256"]) != 64
    ):
        raise ValueError("Variant requires the manifest SHA256")
    try:
        int(value["source_checksums"]["manifest_sha256"], 16)
    except ValueError as error:
        raise ValueError("Invalid manifest SHA256") from error
    if dataset_id is not None and value["dataset_id"] != dataset_id:
        raise ValueError("Dataset variant belongs to a different dataset")
    if (
        manifest_sha256 is not None
        and value["source_checksums"]["manifest_sha256"] != manifest_sha256
    ):
        raise ValueError("Dataset variant manifest checksum changed")
    return value


def read_variant(path, manifest=None):
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_variant(
        document,
        dataset_id_from_manifest(pd.read_parquet(manifest)) if manifest else None,
        checksum(manifest) if manifest else None,
    )


def register_variant(manifest, name, output, *, parent=None, definition=None):
    manifest, output = Path(manifest), Path(output)
    repository = Path(__file__).resolve().parents[3]
    if output.resolve().is_relative_to(repository) and not any(
        output.resolve().is_relative_to(repository / root)
        for root in ("artifacts", "reports")
    ):
        raise ValueError(
            "Variant declarations in the checkout must remain in artifacts/ or reports/"
        )
    if any(
        (p / "metadata.json").exists()
        for p in (output.resolve(), *output.resolve().parents)
    ):
        raise ValueError("Do not write inside an immutable artifact")
    parent_identity = None
    if parent:
        previous = read_variant(parent)
        parent_identity = {k: previous[k] for k in ("dataset_id", "dataset_variant_id")}
    document = make_variant(
        dataset_id_from_manifest(pd.read_parquet(manifest)),
        checksum(manifest),
        name,
        parent=parent_identity,
        definition=definition,
    )
    # A reordered manifest can keep semantic variant identity, but is a distinct
    # immutable declaration snapshot, never an overwrite of its earlier receipt.
    path = (
        output
        / document["dataset_variant_id"]
        / document["source_checksums"]["manifest_sha256"][:16]
        / "variant.json"
    )
    if path.exists():
        if read_variant(path, manifest) != document:
            raise ValueError("Existing variant declaration differs")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return path


def store_variant(metadata, dataset, manifest_sha256, requested=None):
    if "dataset_variant" in metadata:
        variant = validate_variant(
            metadata["dataset_variant"], dataset, manifest_sha256
        )
    else:
        variant = make_variant(dataset, manifest_sha256)
    if requested is not None and requested not in {
        variant["variant_name"],
        variant["dataset_variant_id"],
    }:
        raise ValueError(
            "Feature store dataset variant mismatch; do not relabel or mix immutable stores"
        )
    return variant


def variant_fields(variant):
    return {
        k: variant[k]
        for k in ("dataset_variant_id", "variant_name", "parent_dataset_identity")
    }
