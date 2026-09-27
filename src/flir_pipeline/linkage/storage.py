"""Immutable candidate publications, lightweight counts and source reconstruction."""

from pathlib import Path

import pandas as pd
import pyarrow

from flir_pipeline.linkage.base import ARTIFACT_KIND, SEMANTICS, LinkageConfig
from flir_pipeline.linkage.candidates import generate_candidates
from flir_pipeline.linkage.sources import (
    InputPaths,
    assert_sources_unchanged,
    load_sources,
)
from flir_pipeline.similarity.storage import (
    execution_provenance,
    file_sha256,
    read_json,
    stable_id,
    write_json,
)

TABLE_NAMES = ("content_candidates", "candidate_occurrences", "labeled_occurrences")
OUTPUT_FILES = ("summary.json", *(f"{name}.parquet" for name in TABLE_NAMES))
OCCURRENCE_COLUMNS = (
    "content_id",
    "frame_id",
    "video_id",
    "sample_index",
    "timestamp_seconds",
    "sequence_id",
    "source_video",
    "source_video_sha256",
    "sample_fps",
    "source_frame_index_estimate",
    "clip_embedding_row",
    "dinov2_embedding_row",
)


def identity_payload(source, config):
    return {
        "artifact_kind": ARTIFACT_KIND,
        "artifact_version": 1,
        "configuration": config.model_dump(mode="json"),
        "sources": source.signature,
    }


def candidate_tables(source, config, linkage_id):
    features = source.features
    labeled_ids, video_ids = features["labeled_clip"].ids, features["video_clip"].ids
    candidates = generate_candidates(
        labeled_ids,
        video_ids,
        {
            encoder: features[f"labeled_{encoder}"].vectors
            for encoder in ("clip", "dinov2")
        },
        {
            encoder: features[f"video_{encoder}"].vectors
            for encoder in ("clip", "dinov2")
        },
        config,
        linkage_id,
    )
    counts = source.video_records.groupby("content_id", sort=False).agg(
        video_occurrence_count=("frame_id", "size"),
        video_sequence_count=("sequence_id", "nunique"),
    )
    for name in counts:
        candidates[name] = candidates.video_content_id.map(counts[name]).astype("int64")
    # Normalized many-to-many relation: join by video_content_id to expand EACH
    # candidate to ALL occurrences. No per-candidate representative sequence.
    occurrences = source.video_records.loc[
        source.video_records.content_id.isin(candidates.video_content_id),
        list(OCCURRENCE_COLUMNS),
    ].rename(columns={"content_id": "video_content_id", "frame_id": "video_frame_id"})
    occurrences = occurrences.sort_values(
        ["video_content_id", "video_id", "sample_index", "video_frame_id"]
    ).reset_index(drop=True)
    return {
        "content_candidates": candidates,
        "candidate_occurrences": occurrences,
        "labeled_occurrences": source.labeled_records.copy(),
    }


def summary_counts(source, tables):
    candidates = tables["content_candidates"]
    counts = candidates.groupby("labeled_content_id").size()
    return {
        "labeled_occurrences": len(source.labeled_records),
        "labeled_unique_contents": len(source.features["labeled_clip"].ids),
        "video_occurrences": len(source.video_records),
        "video_unique_contents": len(source.features["video_clip"].ids),
        "candidate_pairs": len(candidates),
        "both_topk_pairs": int(candidates.both_topk.sum()),
        "per_query_candidate_count_min": int(counts.min()),
        "per_query_candidate_count_max": int(counts.max()),
        "candidate_video_unique_contents": int(candidates.video_content_id.nunique()),
        "candidate_video_occurrences": len(tables["candidate_occurrences"]),
        "candidate_pairs_with_multiple_video_occurrences": int(
            candidates.video_occurrence_count.gt(1).sum()
        ),
        "candidate_pairs_with_multiple_sequence_instances": int(
            candidates.video_sequence_count.gt(1).sum()
        ),
    }


def _header(source, config):
    identity = identity_payload(source, config)
    return {
        "artifact_kind": ARTIFACT_KIND,
        "artifact_version": 1,
        "artifact_id": stable_id(identity),
        "ground_truth": False,
        "top_k": config.top_k,
        "effective_top_k": min(config.top_k, len(source.features["video_clip"].ids)),
        "labeled_dataset_id": source.signature["labeled_dataset_id"],
        "video_dataset_id": source.signature["video_dataset_id"],
        "clip_feature_space_id": source.features["video_clip"].space[
            "feature_space_id"
        ],
        "dinov2_feature_space_id": source.features["video_dinov2"].space[
            "feature_space_id"
        ],
        "sequence_set_id": source.signature["sequence_set_id"],
        "identity": identity,
        "semantics": SEMANTICS,
    }


def _checksums(directory):
    return {name: file_sha256(directory / name) for name in OUTPUT_FILES}


def _verify_expected(directory, source, config, tables):
    meta = read_json(directory / "metadata.json")
    header = _header(source, config)
    if (
        set(meta)
        != {*header, "output_checksums", "git_commit", "created_at", "execution"}
        or any(meta[key] != value for key, value in header.items())
        or meta["git_commit"] != meta["execution"]["git_commit"]
        or meta["created_at"] != meta["execution"]["created_at"]
    ):
        raise ValueError(
            "Linkage identity, source or candidate-only policy binding mismatch"
        )
    if set(p.name for p in directory.iterdir()) != {
        *OUTPUT_FILES,
        "metadata.json",
    } or meta["output_checksums"] != _checksums(directory):
        raise ValueError("Incomplete linkage artifact or invalid output checksums")
    for name, expected in tables.items():
        actual = pd.read_parquet(directory / f"{name}.parquet")
        # Reconstructing ranks and the full normalized occurrence table detects
        # collapsed ambiguity even if an editor rewrites output checksums.
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    if read_json(directory / "summary.json") != summary_counts(source, tables):
        raise ValueError("Linkage summary counts do not match reconstructed candidates")
    assert_sources_unchanged(source)


def build_to_store(
    paths: InputPaths,
    config: LinkageConfig = LinkageConfig(),
    output_root: Path = Path("artifacts/linkage"),
) -> Path:
    source = load_sources(paths)
    header = _header(source, config)
    directory = output_root / header["artifact_id"]
    tables = candidate_tables(source, config, header["artifact_id"])
    if directory.exists():
        _verify_expected(directory, source, config, tables)
        return directory
    # Exclusive directory creation; preserve interrupted publications for audit.
    # Inputs are immutable and should not be replaced by another process.
    directory.mkdir(parents=True, exist_ok=False)
    for name, table in tables.items():
        table.to_parquet(directory / f"{name}.parquet", index=False)
        pd.testing.assert_frame_equal(
            pd.read_parquet(directory / f"{name}.parquet"), table, check_exact=True
        )
    write_json(directory / "summary.json", summary_counts(source, tables))
    assert_sources_unchanged(source)
    execution = execution_provenance()
    execution["source_sha256"] = {
        p.name: file_sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))
    }
    execution["pyarrow_version"] = pyarrow.__version__
    # metadata.json is the completion marker. Counts never claim quality_valid;
    # formal verification must reconstruct candidates against all supplied inputs.
    write_json(
        directory / "metadata.json",
        {
            **header,
            "output_checksums": _checksums(directory),
            "git_commit": execution["git_commit"],
            "created_at": execution["created_at"],
            "execution": execution,
        },
    )
    return directory


def verify_directory(directory: Path, paths: InputPaths) -> dict:
    result = {"quality_valid": False, "source_bound": False}
    try:
        meta = read_json(directory / "metadata.json")
        config = LinkageConfig.model_validate(meta["identity"]["configuration"])
        source = load_sources(paths)
        tables = candidate_tables(
            source, config, stable_id(identity_payload(source, config))
        )
        _verify_expected(directory, source, config, tables)
        result.update(
            {
                "quality_valid": True,
                "source_bound": True,
                "feature_pairing_valid": True,
                "content_coverage_valid": True,
                "deterministic_ids_valid": True,
                "topk_union_ranks_scores_flags_valid": True,
                "all_video_occurrences_preserved": True,
                "labeled_lineage_preserved": True,
                "candidate_only_semantics_valid": True,
                "output_checksums_valid": True,
            }
        )
    except (
        ValueError,
        OSError,
        KeyError,
        TypeError,
        AssertionError,
        IndexError,
        OverflowError,
    ) as error:
        result["error"] = f"{type(error).__name__}: {error}"
    return result


def summarize_directory(directory: Path) -> dict:
    """Counts only; no array, Parquet, checksum, source or verification work."""
    meta = read_json(directory / "metadata.json")
    if meta.get("artifact_kind") != ARTIFACT_KIND or meta.get("artifact_version") != 1:
        raise ValueError("Not a completed linkage candidate artifact")
    return read_json(directory / "summary.json")
