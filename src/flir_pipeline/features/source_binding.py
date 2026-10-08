"""Versioned source provenance for features; never mathematical encoder identity.

ZIP repackaging may preserve dataset/variant/feature identities while changing
this binding. Resume and reuse require an identical verified publication; neither
operation silently rebinds an immutable store. Absolute roots and filesystem
inode/mtime details are deliberately excluded so unchanged inputs can relocate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from flir_pipeline.data.local_images import declared_file, relative_posix_path
from flir_pipeline.data.variants import read_variant, validate_variant
from flir_pipeline.data.video_variant_contract import (
    MANIFEST_VERSION,
    Digest,
    Key,
    Nonnegative,
)
from flir_pipeline.data.video_variant_storage import inspect_ingestion
from flir_pipeline.utils.hashing import sha256_file


class SourceFingerprint(BaseModel):
    """The source snapshot contract already verified by ingestion inspection."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    path: str
    sha256: Digest
    size_bytes: Nonnegative

    @field_validator("path")
    @classmethod
    def portable_path(cls, value: str) -> str:
        return relative_posix_path(value)


class VideoVariantSourceBinding(BaseModel):
    """An immutable-publication binding, not a claim of temporal alignment.

    Sources include all declared fingerprints. ZIPs are verified by the shared
    reader once per session; other declared files are checked once at preparation.
    Image QA is replayed for the selected representatives, not all occurrences.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    schema_version: Literal["video_variant_feature_source_v1"] = (
        "video_variant_feature_source_v1"
    )
    artifact_kind: Literal["video_variant_ingestion_v1"] = "video_variant_ingestion_v1"
    manifest_version: Literal[MANIFEST_VERSION] = MANIFEST_VERSION
    artifact_id: Digest
    dataset_id: Digest
    dataset_variant_id: Digest
    metadata_sha256: Digest
    receipt_sha256: Digest
    manifest_sha256: Digest
    archive_keys: list[Key]
    sources: dict[str, SourceFingerprint]
    output_checksums: dict[str, Digest]

    @model_validator(mode="after")
    def coverage(self):
        if not self.archive_keys or self.archive_keys != sorted(set(self.archive_keys)):
            raise ValueError("Source binding requires unique sorted archive keys")
        if {k for k in self.sources if k.startswith("archive:")} != {
            f"archive:{k}" for k in self.archive_keys
        }:
            raise ValueError("Source binding archive coverage differs")
        if self.output_checksums.get("manifest.parquet") != self.manifest_sha256:
            raise ValueError("Source binding manifest checksum differs")
        for name in self.output_checksums:
            relative_posix_path(name)
        return self


def prepare_video_variant_binding(
    publication: Path, manifest_path: Path, input_root: Path
) -> tuple[VideoVariantSourceBinding, dict]:
    """Reuse publication verification and bind its original manifest/variant.

    No ZIP image replay occurs here. The source factory independently owns the
    ZIP session and its verified central directory. The two snapshots are compared
    before storage so an incompatible publication cannot be paired with a reader.
    """
    publication = Path(publication).expanduser().resolve()
    if Path(manifest_path).expanduser().resolve() != declared_file(
        publication, "manifest.parquet"
    ):
        raise ValueError(
            "Use the original manifest.parquet of the ingestion publication"
        )
    checked = inspect_ingestion(publication)
    metadata, summary = checked["metadata"], checked["summary"]
    if not summary["integrity_valid"] or not summary["scientific_manifest_available"]:
        raise ValueError(
            "Video variant features require a valid scientific publication"
        )
    binding = VideoVariantSourceBinding(
        artifact_id=metadata["artifact_id"],
        dataset_id=summary["dataset_id"],
        dataset_variant_id=summary["dataset_variant_id"],
        metadata_sha256=sha256_file(declared_file(publication, "metadata.json")),
        receipt_sha256=sha256_file(declared_file(publication, "receipt.json")),
        manifest_sha256=metadata["identity"]["output_checksums"]["manifest.parquet"],
        archive_keys=sorted(
            a["archive_key"] for a in metadata["identity"]["config"]["archives"]
        ),
        sources=metadata["identity"]["sources"],
        output_checksums=metadata["identity"]["output_checksums"],
    )
    for key, fingerprint in binding.sources.items():
        if not key.startswith("archive:"):
            path = declared_file(
                Path(input_root).expanduser().resolve(), fingerprint.path
            )
            if (
                path.stat().st_size != fingerprint.size_bytes
                or sha256_file(path) != fingerprint.sha256
            ):
                raise ValueError(
                    "Non-ZIP ingestion source differs from its fingerprint"
                )
    variant = read_variant(declared_file(publication, "variant.json"), manifest_path)
    if variant["dataset_variant_id"] != binding.dataset_variant_id:
        raise ValueError("Source binding dataset variant differs from publication")
    return binding, variant


def validate_store_binding(metadata: dict) -> VideoVariantSourceBinding:
    """Validate a store snapshot offline; original-source checking is session-bound."""
    binding = VideoVariantSourceBinding.model_validate(metadata["source_binding"])
    variant = validate_variant(
        metadata["dataset_variant"], binding.dataset_id, binding.manifest_sha256
    )
    if (
        metadata.get("image_source_type") != "zip_collection"
        or metadata.get("dataset_id") != binding.dataset_id
        or variant["dataset_variant_id"] != binding.dataset_variant_id
        or metadata.get("cache_signature", {}).get("source_binding")
        != binding.model_dump(mode="json")
    ):
        raise ValueError(
            "Feature store source binding differs from scientific/cache identity"
        )
    return binding
