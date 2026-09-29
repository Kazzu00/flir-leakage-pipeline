"""Original features plus explicit source-grid or filename-inferred timelines."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from flir_pipeline.data.identity import dataset_id_from_manifest
from flir_pipeline.data.temporal import audit_temporal_lineage
from flir_pipeline.data.variants import (
    make_variant,
    store_variant,
    validate_variant,
    variant_fields,
)
from flir_pipeline.data.video_temporal import audit_video_temporal_lineage
from flir_pipeline.linkage.sources import _feature, _labeled_manifest
from flir_pipeline.sequences.storage import source_checksums
from flir_pipeline.similarity.storage import read_json, stable_id


@dataclass
class ExperimentSources:
    records: pd.DataFrame
    contents: list[str]
    embeddings: dict[str, np.ndarray]
    signature: dict
    family: str
    paths: tuple[Path, Path, Path] | None = None

    def __post_init__(self):
        variant = self.signature.get("dataset_variant") or make_variant(
            self.signature["dataset_id"], self.signature["checksums"]["manifest_sha256"]
        )
        variant = validate_variant(
            variant,
            self.signature["dataset_id"],
            self.signature["checksums"]["manifest_sha256"],
        )
        self.signature = {
            **self.signature,
            "dataset_variant": variant,
            **variant_fields(variant),
        }

    def unchanged(self):
        if self.paths and source_checksums(*self.paths) != self.signature["checksums"]:
            raise ValueError("Experiment sources changed during execution")

    def timelines(self):
        """Collapse historical copies at the SAME position, never across time.

        Multiple different contents at one nominal position are ambiguous. They
        stay in occurrence tables but cannot silently choose a temporal vector.
        """
        known = self.records.dropna(subset=["position"])
        if known.groupby(["timeline_id", "position"]).content_id.nunique().gt(1).any():
            raise ValueError(
                "Conflicting contents at one temporal position; resolve lineage explicitly"
            )
        return (
            known.sort_values(["timeline_id", "position", "frame_id"])
            .drop_duplicates(["timeline_id", "position"])
            .reset_index(drop=True)
        )

    def safe_output(self, output: Path, *dependencies: Path):
        root = output.resolve()
        repo = Path(__file__).resolve().parents[4]
        if root.is_relative_to(repo) and not any(
            root.is_relative_to(repo / name) for name in ("artifacts", "reports")
        ):
            raise ValueError(
                "Experimental outputs inside Git must stay in ignored artifacts/ or reports/"
            )
        protected = [*(self.paths or ()), *dependencies]
        if any(root.is_relative_to(p.resolve()) for p in protected) or any(
            p.resolve().is_relative_to(root) for p in (self.paths or ())
        ):
            raise ValueError("Outputs must be separate from immutable sources")
        if any((p / "metadata.json").exists() for p in (root, *root.parents)):
            raise ValueError("Output root cannot be nested inside a publication")


def load_sources(
    manifest: Path, clip: Path, dinov2: Path, family: str = "all", dataset_variant=None
):
    initial = source_checksums(manifest, clip, dinov2)
    raw = pd.read_parquet(manifest)
    dataset = dataset_id_from_manifest(raw)
    variants = [
        store_variant(
            read_json(directory / "metadata.json"),
            dataset,
            initial["manifest_sha256"],
            dataset_variant,
        )
        for directory in (clip, dinov2)
    ]
    if variants[0] != variants[1]:
        raise ValueError(
            "CLIP and DINOv2 must belong to the same declared dataset variant"
        )
    if raw.manifest_version.eq("flir_video_samples_v1").all():
        records = audit_video_temporal_lineage(raw)
        records["timeline_id"] = records.video_id
        records["family"] = records.video_id
        records["position"] = records.sample_index.astype("Int64")
        records["temporal_source"] = "sampled_video_grid"
    else:
        raw = _labeled_manifest(manifest)
        lineage = audit_temporal_lineage(raw, max_frame_gap=0).lineage
        records = raw.copy()
        records["family"] = lineage.possible_sequence
        # Historical copies may reside in several archives. Family + nominal
        # position is an explicitly inferred timeline, never source provenance.
        records["timeline_id"] = lineage.possible_sequence
        records["position"] = lineage.possible_frame_index.where(
            lineage.order_reconstructable_from_name
        )
        records["temporal_source"] = "filename_heuristic"
    if family != "all":
        records = records.loc[records.family.eq(family)].copy()
    if records.empty:
        raise ValueError(f"No occurrences in requested family: {family}")
    records = records.sort_values("frame_id").reset_index(drop=True)
    ids = sorted(records.content_id.unique())
    embeddings, spaces = {}, {}
    for encoder, directory in (("clip", clip), ("dinov2", dinov2)):
        loaded = _feature(directory, raw, encoder)
        lookup = {c: i for i, c in enumerate(loaded.ids)}
        embeddings[encoder] = loaded.vectors[[lookup[c] for c in ids]]
        records[f"{encoder}_embedding_row"] = records.frame_id.map(
            loaded.records.set_index("frame_id").embedding_row
        ).astype("int64")
        spaces[encoder] = loaded.space
    records["content_row"] = records.content_id.map({c: i for i, c in enumerate(ids)})
    signature = {
        "dataset_id": dataset,
        "dataset_variant": variants[0],
        "checksums": initial,
        "feature_spaces": spaces,
        "family": family,
    }
    result = ExperimentSources(
        records, ids, embeddings, signature, family, (manifest, clip, dinov2)
    )
    result.unchanged()
    return result


def scientific_input_id(source: ExperimentSources) -> str:
    """Stable numerical identity; source byte checksums bind the publication separately.

    Row shuffles and historical split edits change source bytes, but cannot change
    numerical fits, candidate IDs or this scientific input identity.
    """
    return stable_id(
        {
            "dataset_id": source.signature["dataset_id"],
            "dataset_variant_id": source.signature["dataset_variant_id"],
            "family": source.family,
            "feature_spaces": source.signature["feature_spaces"],
            "contents": source.contents,
        }
    )
