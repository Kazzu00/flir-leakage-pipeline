"""Operational input discovery; paths and scheduler resources are not science."""

import os
from pathlib import Path

import pandas as pd
import yaml
from pydantic import Field

from flir_pipeline.data.identity import dataset_id_from_manifest
from flir_pipeline.data.variants import checksum, store_variant
from flir_pipeline.linkage.sources import _feature
from flir_pipeline.sequences.experiments.config import StrictModel, SuiteConfig
from flir_pipeline.sequences.experiments.sources import load_sources
from flir_pipeline.similarity.storage import read_json


class Inputs(StrictModel):
    dataset_variant: str | None = None
    manifest: str | None = None
    clip_features: str | None = None
    dinov2_features: str | None = None
    images_root: str | None = None
    images_archive: str | None = None
    search_roots: tuple[str, ...] = ("artifacts",)


class Slurm(StrictModel):
    cpus_per_task: int = Field(1, ge=1, le=64)
    memory: str = Field("8G", pattern=r"^[1-9][0-9]*[KMGT]?$")
    time: str = Field("12:00:00", pattern=r"^(?:[0-9]+-)?[0-9]+:[0-9]{2}:[0-9]{2}$")
    partition: str | None = Field(None, pattern=r"^[\w.-]+$")
    account: str | None = Field(None, pattern=r"^[\w.-]+$")
    run_root: str = "reports/sequence_experiments"


def load_profile(path):
    document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("Profile must be a YAML mapping")
    document = dict(document)
    inputs = Inputs.model_validate(document.pop("inputs", {}))
    slurm = Slurm.model_validate(document.pop("slurm", {}))
    return SuiteConfig.model_validate(document), inputs, slurm


def local_path(value):
    expanded = os.path.expandvars(str(value))
    if "$" in expanded or "%" in expanded:
        raise ValueError(f"Unresolved environment variable in path: {value}")
    return Path(expanded).expanduser().resolve()


def one(candidates, role):
    paths = sorted(set(Path(p).resolve() for p in candidates))
    if len(paths) != 1:
        raise ValueError(
            f"Expected one compatible {role}; found {len(paths)}: "
            + ", ".join(map(str, paths))
            + ". Select explicitly via config/CLI/env."
        )
    return paths[0]


def resolve_inputs(
    inputs,
    family,
    *,
    manifest=None,
    clip_features=None,
    dinov2_features=None,
    require_images=False,
    dataset_variant=None,
):
    """Validate complete stores against the manifest; never choose newest/smoke.

    Priority is explicit CLI, environment, profile, then metadata discovery.
    Discovery only reads metadata/indexes/arrays; it never fits or downloads.
    A corrupt store claiming the requested dataset is an error, not a fallback.
    """
    selected = {}
    requested_variant = (
        dataset_variant
        or os.environ.get("FLIR_DATASET_VARIANT")
        or inputs.dataset_variant
    )
    for role, explicit in (
        ("manifest", manifest),
        ("clip_features", clip_features),
        ("dinov2_features", dinov2_features),
        ("images_root", None),
        ("images_archive", None),
    ):
        value = (
            explicit or os.environ.get("FLIR_" + role.upper()) or getattr(inputs, role)
        )
        if value:
            selected[role] = local_path(value)
    roots = [local_path(p) for p in inputs.search_roots]
    if "manifest" not in selected:
        candidates = []
        for root in roots:
            for path in sorted(root.rglob("*.parquet")):
                if "manifest" not in path.name.lower():
                    continue
                frame = pd.read_parquet(path)
                if not {"manifest_version", "frame_id", "content_id"} <= set(frame):
                    continue
                try:
                    dataset_id_from_manifest(frame)
                except (ValueError, KeyError):
                    continue
                # Family selection is finally validated by load_sources.
                candidates.append(path)
        selected["manifest"] = one(candidates, "manifest")
    manifest_frame = pd.read_parquet(selected["manifest"])
    dataset = dataset_id_from_manifest(manifest_frame)
    for encoder in ("clip", "dinov2"):
        role = encoder + "_features"
        if role in selected:
            store_variant(
                read_json(selected[role] / "metadata.json"),
                dataset,
                checksum(selected["manifest"]),
                requested_variant,
            )
            _feature(selected[role], manifest_frame, encoder)
            continue
        candidates = []
        for root in roots:
            for path in sorted(root.rglob("metadata.json")):
                meta = read_json(path)
                if not isinstance(meta, dict):
                    continue
                if (
                    meta.get("extractor") != encoder
                    or meta.get("dataset_id") != dataset
                ):
                    continue
                candidate_variant = store_variant(
                    meta, dataset, checksum(selected["manifest"])
                )
                if requested_variant and requested_variant not in {
                    candidate_variant["variant_name"],
                    candidate_variant["dataset_variant_id"],
                }:
                    continue
                if (
                    meta.get("selected_content_ids")
                    != manifest_frame.content_id.nunique()
                ):
                    continue  # Declared smoke/subset is not a full compatible store.
                _feature(path.parent, manifest_frame, encoder)
                candidates.append(path.parent)
        selected[role] = one(candidates, role)
    source = load_sources(
        selected["manifest"],
        selected["clip_features"],
        selected["dinov2_features"],
        family,
        requested_variant,
    )
    if require_images and ("images_root" in selected) == ("images_archive" in selected):
        raise ValueError(
            "Set exactly one FLIR_IMAGES_ROOT / FLIR_IMAGES_ARCHIVE (or inputs config) for visual review"
        )
    for role in ("images_root", "images_archive"):
        if role in selected and not selected[role].exists():
            raise ValueError(f"Missing {role}: {selected[role]}")
    return selected, source
