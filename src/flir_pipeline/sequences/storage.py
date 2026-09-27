"""Immutable source-bound publications and full reconstruction, without image I/O."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow

from flir_pipeline.data.identity import dataset_id_from_manifest
from flir_pipeline.data.video_temporal import audit_video_temporal_lineage
from flir_pipeline.features.storage import (
    feature_space_id,
    verify_features_against_manifest,
)
from flir_pipeline.sequences.base import SEMANTICS, SequenceConfig
from flir_pipeline.sequences.construction import (
    check_partition_invariants,
    construct_tables,
)
from flir_pipeline.sequences.detection import detect_tables
from flir_pipeline.sequences.validation import load_confirmed_validation
from flir_pipeline.similarity.storage import (
    execution_provenance,
    file_sha256,
    read_json,
    stable_id,
    write_json,
)

FEATURE_FILES = (
    "metadata.json",
    "content_index.parquet",
    "record_index.parquet",
    "embeddings_raw.npy",
    "embeddings_l2.npy",
)
CANDIDATE_KIND = "sequence_boundary_candidates"
SEQUENCE_KIND = "sequence_instance_set"


@dataclass
class SequenceSources:
    records: pd.DataFrame
    embeddings: dict[str, np.ndarray]
    signature: dict
    paths: tuple[Path, Path, Path]


def source_checksums(manifest: Path, clip: Path, dinov2: Path) -> dict:
    return {
        "manifest_sha256": file_sha256(manifest),
        "feature_files": {
            encoder: {name: file_sha256(directory / name) for name in FEATURE_FILES}
            for encoder, directory in (("clip", clip), ("dinov2", dinov2))
        },
    }


def load_sources(manifest_path: Path, clip: Path, dinov2: Path) -> SequenceSources:
    """Require complete original L2 spaces; join occurrences via EACH record_index."""
    initial = source_checksums(manifest_path, clip, dinov2)
    manifest = pd.read_parquet(manifest_path)
    records = (
        audit_video_temporal_lineage(manifest)
        .sort_values(["video_id", "sample_index"])
        .reset_index(drop=True)
    )
    if not records.sample_fps.eq(1.0).all():
        raise ValueError("Sequences v1 requires 1 FPS sampling grids")
    for _, video in records.groupby("video_id", sort=False):
        if not np.array_equal(
            video.sample_index.to_numpy(dtype=np.int64), np.arange(len(video))
        ):
            raise ValueError(
                "Require contiguous sample_index starting at zero in every video"
            )
    embeddings, spaces = {}, {}
    for encoder, directory in (("clip", clip), ("dinov2", dinov2)):
        quality = verify_features_against_manifest(directory, manifest)
        if not quality["reproducible_full_dataset_valid"]:
            raise ValueError(
                f"{encoder} requires verified full features and resolved model revision"
            )
        metadata = read_json(directory / "metadata.json")
        pooling = "projected_pooler_output" if encoder == "clip" else "cls_token"
        if (
            metadata.get("extractor") != encoder
            or metadata.get("pooling_strategy") != pooling
        ):
            raise ValueError(f"Expected original {encoder} representation ({pooling})")
        config = {
            name: metadata[name]
            for name in (
                "extractor",
                "model_id",
                "model_revision",
                "pooling_strategy",
                "preprocessing",
            )
        }
        config["normalization_policy"] = "raw_and_l2_float32"
        if metadata["feature_space_id"] != feature_space_id(config):
            raise ValueError(
                "Feature identity does not match its mathematical configuration"
            )
        values = np.load(
            directory / "embeddings_l2.npy", mmap_mode="r", allow_pickle=False
        )
        if values.dtype != np.float32:
            raise ValueError("Sequences v1 requires original float32 L2 arrays")
        index = pd.read_parquet(directory / "record_index.parquet").set_index(
            "frame_id"
        )
        if not pd.api.types.is_integer_dtype(index.embedding_row.dtype):
            raise ValueError("embedding_row must contain integer indices")
        records[f"{encoder}_embedding_row"] = records.frame_id.map(
            index.embedding_row
        ).astype("int64")
        embeddings[encoder] = values
        spaces[encoder] = {
            "feature_space_id": metadata["feature_space_id"],
            "feature_config": config,
            "embedding_dimension": values.shape[1],
        }
    if source_checksums(manifest_path, clip, dinov2) != initial:
        raise ValueError(
            "Sources changed while loading; use immutable inputs and a single writer"
        )
    signature = {
        "dataset_id": dataset_id_from_manifest(manifest),
        **initial,
        "feature_spaces": spaces,
    }
    return SequenceSources(
        records, embeddings, signature, (manifest_path, clip, dinov2)
    )


def detection_identity(source: SequenceSources, config: SequenceConfig) -> dict:
    return {
        "artifact_kind": CANDIDATE_KIND,
        "sources": source.signature,
        "config": config.model_dump(mode="json"),
    }


def _candidate_tables(source, config, detection_id):
    scores, events = detect_tables(
        source.records, source.embeddings, config, detection_id
    )
    return {"temporal_scores": scores, "candidate_events": events}


def _summary(kind, tables, source):
    summary = {
        "artifact_kind": kind,
        "ground_truth": False,
        "split_created": False,
        "video_count": int(source.records.video_id.nunique()),
        "occurrence_count": len(source.records),
        "unique_content_count": int(source.records.content_id.nunique()),
        "candidate_event_count": len(tables["candidate_events"]),
        "high_confidence_event_count": int(
            tables["candidate_events"].high_confidence.sum()
        ),
        "valid_cuts_by_video": {
            str(video): {
                column: int(group[column].notna().sum())
                for column in ("S", "F3", "clip_change_w3", "dinov2_change_w3")
            }
            for video, group in tables["temporal_scores"].groupby("video_id", sort=True)
        },
        "sequence_boundaries_committed": kind == SEQUENCE_KIND,
    }
    if kind == SEQUENCE_KIND:
        summary.update(
            {
                "sequence_count": len(tables["sequence_instances"]),
                "accepted_boundary_count": len(tables["boundaries"]),
                "rejected_event_count": int(
                    tables["manual_review"].decision.eq("reject").sum()
                ),
                "dependency_edge_count": len(tables["dependency_edges"]),
                "exact_duplicate_dependency_group_count": int(
                    tables[
                        "sequence_instances"
                    ].exact_duplicate_dependency_group_id.nunique()
                ),
            }
        )
    return summary


def _checksums(directory: Path, names) -> dict:
    return {name: file_sha256(directory / name) for name in sorted(names)}


def _assert_unchanged(source):
    latest = source_checksums(*source.paths)
    if any(source.signature[key] != value for key, value in latest.items()):
        raise ValueError("Source checksums changed during computation")


def _publish(directory, identity, tables, source, extra=None):
    # mkdir(exist_ok=False) also prevents two writers from publishing the same run.
    # Failed directories remain untouched for inspection, with no completion marker.
    if identity["artifact_kind"] == SEQUENCE_KIND:
        check_partition_invariants(source.records, tables, stable_id(identity))
    directory.mkdir(parents=True, exist_ok=False)
    for name, table in tables.items():
        table.to_parquet(directory / f"{name}.parquet", index=False)
        pd.testing.assert_frame_equal(
            pd.read_parquet(directory / f"{name}.parquet"), table, check_exact=True
        )
    summary = _summary(identity["artifact_kind"], tables, source)
    write_json(directory / "summary.json", summary)
    _assert_unchanged(source)
    provenance = execution_provenance()
    provenance["source_sha256"] = {
        p.name: file_sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))
    }
    provenance["pyarrow_version"] = pyarrow.__version__
    metadata = {
        "artifact_kind": identity["artifact_kind"],
        "artifact_id": stable_id(identity),
        "identity": identity,
        "semantics": SEMANTICS,
        "output_checksums": _checksums(
            directory, [*(f"{n}.parquet" for n in tables), "summary.json"]
        ),
        "execution": provenance,
        **(extra or {}),
    }
    # Metadata is the completion marker. A failed/full check is never recorded as
    # quality_valid in a lightweight summary; formal verification reconstructs it.
    write_json(directory / "metadata.json", metadata)
    return directory


def _require_match(directory, metadata, identity, tables, source, extra):
    if set(metadata) != {
        "artifact_kind",
        "artifact_id",
        "identity",
        "semantics",
        "output_checksums",
        "execution",
        *extra,
    }:
        raise ValueError("Unexpected artifact metadata schema")
    if (
        metadata["identity"] != identity
        or metadata["artifact_id"] != stable_id(identity)
        or metadata["artifact_kind"] != identity["artifact_kind"]
        or metadata["semantics"] != SEMANTICS
        or any(metadata[key] != value for key, value in extra.items())
    ):
        raise ValueError("Source/config/identity/policy binding mismatch")
    required = {*(f"{name}.parquet" for name in tables), "summary.json"}
    if (
        set(metadata["output_checksums"]) != required
        or _checksums(directory, required) != metadata["output_checksums"]
    ):
        raise ValueError("Output checksums/schema mismatch")
    if {p.name for p in directory.iterdir()} != required | {"metadata.json"}:
        raise ValueError("Unexpected files in immutable sequence publication")
    stored = {}
    for name, expected in tables.items():
        stored[name] = pd.read_parquet(directory / f"{name}.parquet")
        pd.testing.assert_frame_equal(stored[name], expected, check_exact=True)
    if identity["artifact_kind"] == SEQUENCE_KIND:
        check_partition_invariants(source.records, stored, stable_id(identity))
    if read_json(directory / "summary.json") != _summary(
        identity["artifact_kind"], tables, source
    ):
        raise ValueError("Summary does not match reconstructed outputs")
    _assert_unchanged(source)


def detect_to_store(
    manifest: Path,
    clip: Path,
    dinov2: Path,
    config: SequenceConfig,
    output_root: Path = Path("artifacts/sequences"),
) -> Path:
    source = load_sources(manifest, clip, dinov2)
    identity = detection_identity(source, config)
    directory = output_root / "candidates" / stable_id(identity)
    tables = _candidate_tables(source, config, stable_id(identity))
    if directory.exists():
        _require_match(
            directory,
            read_json(directory / "metadata.json"),
            identity,
            tables,
            source,
            {},
        )
        return directory
    return _publish(directory, identity, tables, source)


def _sequence_plan(source, config, tables, validation):
    detect_id = stable_id(detection_identity(source, config))
    review, signature = load_confirmed_validation(
        validation, tables["candidate_events"]
    )
    identity = {
        "artifact_kind": SEQUENCE_KIND,
        "detection": detection_identity(source, config),
        "detection_id": detect_id,
        "validation": signature,
    }
    tables = {
        **tables,
        **construct_tables(
            source.records,
            tables["candidate_events"],
            review,
            signature,
            stable_id(identity),
        ),
    }
    return identity, tables


def build_to_store(
    detection: Path,
    manifest: Path,
    clip: Path,
    dinov2: Path,
    validation: Path,
    output_root: Path = Path("artifacts/sequences"),
) -> Path:
    """Explicit commit step. Manual flags that construction was pending are valid."""
    source = load_sources(manifest, clip, dinov2)
    metadata = read_json(detection / "metadata.json")
    if metadata["artifact_kind"] != CANDIDATE_KIND:
        raise ValueError("build requires a candidate detection artifact")
    config = SequenceConfig.model_validate(metadata["identity"]["config"])
    detect_identity = detection_identity(source, config)
    tables = _candidate_tables(source, config, stable_id(detect_identity))
    _require_match(detection, metadata, detect_identity, tables, source, {})
    identity, tables = _sequence_plan(source, config, tables, validation)
    directory = output_root / "sets" / stable_id(identity)
    if directory.exists():
        _require_match(
            directory,
            read_json(directory / "metadata.json"),
            identity,
            tables,
            source,
            {},
        )
        return directory
    # Bind the same frozen review bytes across computation and publication.
    if (
        load_confirmed_validation(validation, tables["candidate_events"])[1]
        != identity["validation"]
    ):
        raise ValueError("Confirmed review changed during sequence construction")
    return _publish(directory, identity, tables, source)


def verify_directory(
    directory: Path,
    manifest: Path,
    clip: Path,
    dinov2: Path,
    validation: Path | None = None,
) -> dict:
    """Formal verification ALWAYS needs sources; checksums alone cannot pass QA."""
    result = {"quality_valid": False, "source_bound": False}
    try:
        metadata = read_json(directory / "metadata.json")
        kind = metadata["artifact_kind"]
        if kind not in (CANDIDATE_KIND, SEQUENCE_KIND):
            raise ValueError("Unknown sequences artifact kind")
        if kind == SEQUENCE_KIND and validation is None:
            raise ValueError("Committed sequence verification requires --validation")
        source = load_sources(manifest, clip, dinov2)
        config = SequenceConfig.model_validate(
            metadata["identity"]["config"]
            if kind == CANDIDATE_KIND
            else metadata["identity"]["detection"]["config"]
        )
        identity = detection_identity(source, config)
        tables = _candidate_tables(source, config, stable_id(identity))
        if kind == SEQUENCE_KIND:
            identity, tables = _sequence_plan(source, config, tables, validation)
        _require_match(directory, metadata, identity, tables, source, {})
        result.update(
            {
                "source_bound": True,
                "output_checksums_valid": True,
                "deterministic_identity_valid": True,
                "config_policy_valid": True,
                "candidate_reconstruction_valid": True,
                "no_split_or_clustering_dependency": True,
            }
        )
        if kind == SEQUENCE_KIND:
            result.update(
                {
                    "confirmed_validation_matches": True,
                    "rejected_events_absent": True,
                    "sorted_nonoverlapping_boundaries": True,
                    "occurrence_coverage_exactly_once": True,
                    "no_cross_video_sequence": True,
                    "sequence_ids_unique": True,
                    "dependency_edges_justified": True,
                    "dependency_components_recomputed": True,
                    "deterministic_dependency_group_ids": True,
                }
            )
        result["quality_valid"] = all(result[k] for k in result if k != "quality_valid")
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        AssertionError,
        IndexError,
        OverflowError,
    ) as error:
        result["error"] = f"{type(error).__name__}: {error}"
    return result


def summarize_directory(directory: Path) -> dict:
    """Metadata-only inspection; explicitly not a checksum/quality verification."""
    metadata = read_json(directory / "metadata.json")
    return {
        "artifact_id": metadata["artifact_id"],
        **read_json(directory / "summary.json"),
        "verification_performed": False,
        "quality_valid": None,
    }
