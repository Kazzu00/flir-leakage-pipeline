"""Offline cross-dataset linkage invariants; all identities and vectors are synthetic."""

import hashlib
import shutil
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_sequences import synthetic_review, synthetic_sources
from typer.testing import CliRunner

from flir_pipeline.cli import app
from flir_pipeline.data.identity import dataset_id_from_manifest
from flir_pipeline.features.storage import feature_space_id
from flir_pipeline.linkage import candidates as candidate_module
from flir_pipeline.linkage.base import LinkageConfig
from flir_pipeline.linkage.candidates import generate_candidates
from flir_pipeline.linkage.sources import InputPaths, load_sources
from flir_pipeline.linkage.storage import (
    build_to_store,
    summarize_directory,
    verify_directory,
)
from flir_pipeline.sequences.base import SequenceConfig
from flir_pipeline.sequences.storage import build_to_store as build_sequences
from flir_pipeline.sequences.storage import detect_to_store
from flir_pipeline.similarity.storage import file_sha256, read_json, write_json


def labeled_inputs(root, video_features):
    root.mkdir()
    rows = []
    for occurrence, content in enumerate([0, 1, 2, 0]):
        digest = hashlib.sha256(f"labeled-content-{content}".encode()).hexdigest()
        rows.append(
            {
                "manifest_version": "flir_canonical_candidate_v1",
                "frame_id": f"labeled-frame-{occurrence}",
                "content_id": digest,
                "image_sha256": digest,
                "label_sha256": hashlib.sha256(
                    f"annotation-{occurrence}".encode()
                ).hexdigest(),
                "original_split": ["train", "val", "train", "test"][occurrence],
                "image_member_path": f"images/{occurrence}.jpg",
                "label_member_path": f"labels/{occurrence}.txt",
                "class_counts": f'{{"{occurrence}":1}}',
            }
        )
    manifest = pd.DataFrame(rows)
    path = root / "manifest.parquet"
    manifest.to_parquet(path, index=False)
    directories = []
    for encoder, video_directory in zip(
        ("clip", "dinov2"), video_features, strict=True
    ):
        directory = root / encoder
        directory.mkdir()
        directories.append(directory)
        # Neither encoder nor dataset shares a positional content-index order.
        index = manifest.drop_duplicates("content_id").sort_values(
            "content_id", ascending=encoder == "dinov2"
        )
        index = (
            index[["content_id", "frame_id"]]
            .rename(columns={"frame_id": "representative_frame_id"})
            .reset_index(drop=True)
        )
        index["embedding_row"] = np.arange(len(index))
        x = []
        for frame_id in index.representative_frame_id:
            content = int(frame_id.rsplit("-", 1)[1])
            axis = (0, 2, 1)[content] if encoder == "clip" else (1, 2, 1)[content]
            x.append(np.eye(8, dtype=np.float32)[axis])
        x = np.stack(x)
        index.to_parquet(directory / "content_index.parquet", index=False)
        records = manifest[["frame_id", "content_id"]].copy()
        records["embedding_row"] = records.content_id.map(
            index.set_index("content_id").embedding_row
        )
        records.iloc[::-1].to_parquet(directory / "record_index.parquet", index=False)
        for name in ("embeddings_raw", "embeddings_l2"):
            np.save(directory / f"{name}.npy", x)
        meta = read_json(video_directory / "metadata.json")
        meta.update(
            dataset_id=dataset_id_from_manifest(manifest),
            total_records=len(manifest),
            selected_content_ids=len(index),
            unique_content_ids=len(index),
        )
        write_json(directory / "metadata.json", meta)
    return path, *directories


@pytest.fixture(scope="module")
def template(tmp_path_factory):
    root = tmp_path_factory.mktemp("linkage-synthetic-template")
    video, clip, dino = synthetic_sources(root / "video", sizes=(360,))
    detected = detect_to_store(video, clip, dino, SequenceConfig(), root / "sequences")
    review = synthetic_review(
        root / "review",
        pd.read_parquet(detected / "candidate_events.parquet"),
        reject=(),
    )
    sequences = build_sequences(detected, video, clip, dino, review, root / "sequences")
    labeled, labeled_clip, labeled_dino = labeled_inputs(root / "labeled", (clip, dino))
    paths = InputPaths(
        labeled, video, labeled_clip, labeled_dino, clip, dino, sequences
    )
    artifact = build_to_store(paths, LinkageConfig(top_k=2), root / "linkage")
    return root, paths, artifact


@pytest.fixture
def publication(template, tmp_path):
    root, paths, artifact = template
    copied = tmp_path / "synthetic"
    shutil.copytree(root, copied)
    paths = InputPaths(
        **{
            field.name: copied / getattr(paths, field.name).relative_to(root)
            for field in fields(paths)
        }
    )
    return paths, copied / artifact.relative_to(root)


def rewrite_table(directory, name, mutate):
    """Simulate semantic corruption even with a newly valid checksum."""
    path = directory / f"{name}.parquet"
    frame = pd.read_parquet(path)
    changed = mutate(frame)
    (frame if changed is None else changed).to_parquet(path, index=False)
    meta = read_json(directory / "metadata.json")
    meta["output_checksums"][path.name] = file_sha256(path)
    write_json(directory / "metadata.json", meta)


def test_union_separate_scores_ranks_ties_and_bounded_blocks(monkeypatch):
    labeled_ids = [f"q{i}" for i in range(7)]
    video_ids = ["a", "b", "c", "d"]
    queries = {
        e: np.tile(np.array([[1, 0]], dtype=np.float32), (7, 1))
        for e in ("clip", "dinov2")
    }
    targets = {
        "clip": np.array([[1, 0], [1, 0], [0, 1], [-1, 0]], dtype=np.float32),
        "dinov2": np.array([[0, 1], [1, 0], [1, 0], [-1, 0]], dtype=np.float32),
    }
    original = candidate_module.cosine_block
    shapes = []

    def bounded(queries, targets):
        shapes.append((len(queries), len(targets)))
        return original(queries, targets)

    monkeypatch.setattr(candidate_module, "cosine_block", bounded)
    result = generate_candidates(
        labeled_ids,
        video_ids,
        queries,
        targets,
        LinkageConfig(top_k=2),
        "synthetic",
        block_rows=2,
    )
    assert max(q for q, _ in shapes) == 2
    assert max(v for _, v in shapes) == 4
    first = result[result.labeled_content_id.eq("q0")].set_index("video_content_id")
    assert first.index.tolist() == ["a", "b", "c"]
    assert first.clip_rank.tolist()[:2] == [1, 2]
    assert pd.isna(first.loc["c", "clip_rank"])
    assert pd.isna(first.loc["a", "dinov2_rank"])
    assert first.loc["b", "dinov2_rank"] == 1
    assert first.loc["c", "dinov2_rank"] == 2
    assert first.both_topk.tolist() == [False, True, False]
    assert first.clip_topk.tolist() == [True, True, False]
    assert first.dinov2_topk.tolist() == [False, True, True]
    assert first.clip_cosine.tolist() == [1, 1, 0]
    assert first.dinov2_cosine.tolist() == [0, 1, 1]
    assert first.mean_reciprocal_rank.tolist() == [0.5, 0.75, 0.25]
    for block_rows in (1, 3, 32):
        other = generate_candidates(
            labeled_ids,
            video_ids,
            queries,
            targets,
            LinkageConfig(top_k=2),
            "synthetic",
            block_rows=block_rows,
        )
        pd.testing.assert_frame_equal(result, other, check_exact=True)


def test_nontrivial_cross_cosines_against_independent_dense_oracle():
    rng = np.random.default_rng(812)
    labeled_ids = [f"q{i:02}" for i in range(5)]
    video_ids = [f"v{i:02}" for i in range(13)]
    labeled, video = {}, {}
    for encoder in ("clip", "dinov2"):
        for arrays, count in ((labeled, 5), (video, 13)):
            x = rng.normal(size=(count, 7)).astype(np.float32)
            arrays[encoder] = x / np.linalg.norm(x, axis=1, keepdims=True)
    result = generate_candidates(
        labeled_ids,
        video_ids,
        labeled,
        video,
        LinkageConfig(top_k=4),
        "synthetic-oracle",
        block_rows=2,
    )
    # A tiny dense reference is appropriate only in this offline test. Dot uses
    # a different implementation from the production fixed-accumulation kernel.
    reference = {
        encoder: labeled[encoder].astype(np.float64)
        @ video[encoder].astype(np.float64).T
        for encoder in ("clip", "dinov2")
    }
    for q, labeled_id in enumerate(labeled_ids):
        group = result[result.labeled_content_id.eq(labeled_id)]
        expected_union = set()
        for encoder in ("clip", "dinov2"):
            indices = sorted(
                range(len(video_ids)),
                key=lambda i: (-reference[encoder][q, i], video_ids[i]),
            )[:4]
            expected_union.update(video_ids[i] for i in indices)
            selected = group[group[f"{encoder}_topk"]].sort_values(f"{encoder}_rank")
            assert selected.video_content_id.tolist() == [video_ids[i] for i in indices]
            for row in group.itertuples():
                assert getattr(row, f"{encoder}_cosine") == pytest.approx(
                    reference[encoder][q, video_ids.index(row.video_content_id)],
                    abs=1e-14,
                )
        assert set(group.video_content_id) == expected_union
    other = generate_candidates(
        labeled_ids,
        video_ids,
        labeled,
        video,
        LinkageConfig(top_k=4),
        "synthetic-oracle",
        block_rows=1,
    )
    pd.testing.assert_frame_equal(result, other, check_exact=True)


def test_complete_source_binding_and_all_ambiguous_lineage(publication):
    paths, artifact = publication
    result = verify_directory(artifact, paths)
    assert result["quality_valid"], result
    metadata = read_json(artifact / "metadata.json")
    assert metadata["labeled_dataset_id"] != metadata["video_dataset_id"]
    assert metadata["artifact_kind"] == "labeled_video_link_candidates"
    assert metadata["artifact_version"] == 1
    assert metadata["ground_truth"] is False
    assert metadata["semantics"]["links_auto_confirmed"] is False
    candidates = pd.read_parquet(artifact / "content_candidates.parquet")
    occurrences = pd.read_parquet(artifact / "candidate_occurrences.parquet")
    labeled = pd.read_parquet(artifact / "labeled_occurrences.parquet")
    pd.testing.assert_frame_equal(
        labeled,
        pd.read_parquet(paths.labeled_manifest)
        .sort_values("frame_id")
        .reset_index(drop=True),
    )
    repeated = labeled[labeled.content_id.duplicated(keep=False)]
    assert len(repeated) == 2
    assert repeated.original_split.tolist() == ["train", "test"]
    assert repeated.label_sha256.nunique() == 2
    assert repeated.class_counts.nunique() == 2
    assert candidates.labeled_content_id.nunique() == 3
    assert not candidates.duplicated(["labeled_content_id", "video_content_id"]).any()
    assert candidates.both_topk.any() and (~candidates.both_topk).any()
    assert candidates.video_sequence_count.gt(1).all()
    assigned = pd.read_parquet(paths.sequence_set / "occurrence_assignments.parquet")
    for candidate in candidates.itertuples():
        actual = occurrences[
            occurrences.video_content_id.eq(candidate.video_content_id)
        ]
        expected = assigned[assigned.content_id.eq(candidate.video_content_id)]
        assert set(actual.video_frame_id) == set(expected.frame_id)
        assert set(actual.sequence_id) == set(expected.sequence_id)
        assert candidate.video_occurrence_count == len(expected)
        assert candidate.video_sequence_count == expected.sequence_id.nunique()
    assert not {
        "sequence_id",
        "matched",
        "confirmed",
        "visual_dependency_group",
        "split_id",
    } & set(candidates)
    assert summarize_directory(artifact) == {
        "labeled_occurrences": 4,
        "labeled_unique_contents": 3,
        "video_occurrences": 360,
        "video_unique_contents": 180,
        "candidate_pairs": 8,
        "both_topk_pairs": 4,
        "per_query_candidate_count_min": 2,
        "per_query_candidate_count_max": 4,
        "candidate_video_unique_contents": 6,
        "candidate_video_occurrences": len(occurrences),
        "candidate_pairs_with_multiple_video_occurrences": 8,
        "candidate_pairs_with_multiple_sequence_instances": 8,
    }


def test_content_mapping_matches_independent_cosine_oracle(publication):
    paths, artifact = publication
    candidates = pd.read_parquet(artifact / "content_candidates.parquet")
    for encoder in ("clip", "dinov2"):
        maps = {}
        for role in ("labeled", "video"):
            directory = getattr(paths, f"{role}_{encoder}")
            index = pd.read_parquet(directory / "content_index.parquet")
            vectors = np.load(directory / "embeddings_l2.npy").astype(np.float64)
            maps[role] = {
                row.content_id: vectors[row.embedding_row] for row in index.itertuples()
            }
        for labeled_id, group in candidates.groupby("labeled_content_id"):
            scores = {
                cid: float(np.dot(maps["labeled"][labeled_id], vector))
                for cid, vector in maps["video"].items()
            }
            best = sorted(scores, key=lambda cid: (-scores[cid], cid))[:2]
            selected = group.loc[group[f"{encoder}_topk"]].sort_values(
                f"{encoder}_rank"
            )
            assert selected.video_content_id.tolist() == best
            for row in group.itertuples():
                assert getattr(row, f"{encoder}_cosine") == pytest.approx(
                    scores[row.video_content_id], abs=1e-14
                )


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "extra",
        "duplicate",
        "missing_column",
        "negative_row",
        "float_row",
        "outside_row",
        "missing_record",
    ],
)
def test_invalid_feature_content_and_record_mapping_refused(publication, fault):
    paths, artifact = publication
    name = "record_index" if fault == "missing_record" else "content_index"
    target = paths.labeled_clip / f"{name}.parquet"
    index = pd.read_parquet(target)
    if fault in ("missing", "missing_record"):
        index = index.iloc[1:]
    elif fault == "extra":
        index.loc[len(index)] = index.iloc[-1]
        index.loc[len(index) - 1, "content_id"] = "foreign"
    elif fault == "duplicate":
        index.loc[0, "content_id"] = index.loc[1, "content_id"]
    elif fault == "missing_column":
        index = index.drop(columns="content_id")
    elif fault == "negative_row":
        index.loc[0, "embedding_row"] = -1
    elif fault == "float_row":
        index.embedding_row = index.embedding_row.astype(float) + 0.5
    else:
        index.loc[0, "embedding_row"] = 500
    index.to_parquet(target, index=False)
    assert not verify_directory(artifact, paths)["quality_valid"]
    with pytest.raises((ValueError, KeyError)):
        load_sources(paths)


@pytest.mark.parametrize(
    "fault", ["shape", "dimension", "nan", "inf", "zero", "not_l2", "dtype"]
)
def test_invalid_embeddings_refused(publication, fault):
    paths, artifact = publication
    target = paths.labeled_dinov2 / "embeddings_l2.npy"
    x = np.load(target)
    if fault == "shape":
        x = x.ravel()
    elif fault == "dimension":
        x = x[:, :0]
    elif fault == "dtype":
        x = x.astype(np.float64)
    elif fault == "not_l2":
        x *= 2
    else:
        x[0] = {"nan": np.nan, "inf": np.inf, "zero": 0}[fault]
    np.save(target, x)
    assert not verify_directory(artifact, paths)["quality_valid"]


@pytest.mark.parametrize("fault", ["id", "config", "dataset", "revision"])
def test_wrong_feature_space_or_dataset_binding_refused(publication, fault):
    paths, artifact = publication
    target = paths.labeled_clip / "metadata.json"
    meta = read_json(target)
    if fault == "id":
        meta["feature_space_id"] = "wrong-space"
    elif fault == "config":
        meta["preprocessing"] = {"synthetic": "different resize"}
        config = {
            k: meta[k]
            for k in (
                "extractor",
                "model_id",
                "model_revision",
                "pooling_strategy",
                "preprocessing",
                "normalization_policy",
            )
        }
        meta["feature_space_id"] = feature_space_id(config)
    elif fault == "dataset":
        meta["dataset_id"] = read_json(paths.video_clip / "metadata.json")["dataset_id"]
    else:
        meta["resolved_model_revision"] = "main"
    write_json(target, meta)
    assert not verify_directory(artifact, paths)["quality_valid"]
    with pytest.raises(ValueError):
        load_sources(paths)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("labeled_content_id", "outside-labeled-manifest"),
        ("video_content_id", "outside-video-manifest"),
        ("candidate_id", "arbitrary-id"),
        ("clip_rank", 0),
        ("dinov2_rank", 3),
        ("clip_cosine", -0.7),
        ("dinov2_cosine", np.nan),
        ("mean_reciprocal_rank", 0.123),
        ("video_occurrence_count", 1),
        ("video_sequence_count", 1),
    ],
)
def test_candidate_corruption_fails_despite_updated_checksums(
    publication, column, value
):
    paths, artifact = publication
    rewrite_table(
        artifact,
        "content_candidates",
        lambda df: df.__setitem__(column, df[column].mask(df.index == 0, value)),
    )
    assert not verify_directory(artifact, paths)["quality_valid"]


@pytest.mark.parametrize("flag", ["clip_topk", "dinov2_topk", "both_topk"])
def test_wrong_flags_refused(publication, flag):
    paths, artifact = publication
    rewrite_table(
        artifact, "content_candidates", lambda df: df.__setitem__(flag, ~df[flag])
    )
    assert not verify_directory(artifact, paths)["quality_valid"]


@pytest.mark.parametrize(
    "fault",
    [
        "collapse_occurrences",
        "collapse_sequences",
        "wrong_sample",
        "lost_labeled_duplicate",
        "changed_annotation",
        "missing_pair",
        "duplicate_pair",
    ],
)
def test_lineage_or_pair_corruption_refused(publication, fault):
    paths, artifact = publication
    if fault == "collapse_occurrences":
        rewrite_table(
            artifact,
            "candidate_occurrences",
            lambda df: df.drop_duplicates("video_content_id"),
        )
    elif fault == "collapse_sequences":
        rewrite_table(
            artifact,
            "candidate_occurrences",
            lambda df: df.assign(
                sequence_id=df.groupby("video_content_id").sequence_id.transform(
                    "first"
                )
            ),
        )
    elif fault == "wrong_sample":
        rewrite_table(
            artifact,
            "candidate_occurrences",
            lambda df: df.assign(sample_index=df.sample_index + 1),
        )
    elif fault == "lost_labeled_duplicate":
        rewrite_table(
            artifact, "labeled_occurrences", lambda df: df.drop_duplicates("content_id")
        )
    elif fault == "changed_annotation":
        rewrite_table(
            artifact,
            "labeled_occurrences",
            lambda df: df.assign(label_sha256="silently-resolved-conflict"),
        )
    elif fault == "missing_pair":
        rewrite_table(
            artifact,
            "content_candidates",
            lambda df: df.iloc[1:].reset_index(drop=True),
        )
    else:
        rewrite_table(
            artifact,
            "content_candidates",
            lambda df: pd.concat([df, df.iloc[:1]], ignore_index=True),
        )
    assert not verify_directory(artifact, paths)["quality_valid"]


def test_sequence_source_collapse_refused_even_with_rehashed_output(publication):
    paths, artifact = publication
    rewrite_table(
        paths.sequence_set,
        "occurrence_assignments",
        lambda df: df.assign(
            sequence_id=df.groupby("content_id").sequence_id.transform("first")
        ),
    )
    assert not verify_directory(artifact, paths)["quality_valid"]
    with pytest.raises((ValueError, AssertionError)):
        load_sources(paths)


def test_deterministic_rebuild_and_immutable_reuse(publication, tmp_path):
    paths, artifact = publication
    before = {p.name: file_sha256(p) for p in artifact.iterdir()}
    assert build_to_store(paths, LinkageConfig(top_k=2), artifact.parent) == artifact
    assert before == {p.name: file_sha256(p) for p in artifact.iterdir()}
    other = build_to_store(paths, LinkageConfig(top_k=2), tmp_path / "rebuilt")
    assert other.name == artifact.name
    assert (
        read_json(other / "metadata.json")["output_checksums"]
        == read_json(artifact / "metadata.json")["output_checksums"]
    )
    incomplete = tmp_path / "incomplete" / artifact.name
    incomplete.mkdir(parents=True)
    sentinel = incomplete / "partial.bin"
    sentinel.write_bytes(b"preserve interrupted work")
    with pytest.raises(OSError):
        build_to_store(paths, LinkageConfig(top_k=2), incomplete.parent)
    assert sentinel.read_bytes() == b"preserve interrupted work"


def test_top_k_larger_than_population_and_invalid_config(publication, tmp_path):
    paths, _ = publication
    artifact = build_to_store(paths, LinkageConfig(top_k=999), tmp_path / "large-k")
    counts = summarize_directory(artifact)
    assert counts["candidate_pairs"] == 3 * 180
    assert counts["both_topk_pairs"] == 3 * 180
    assert read_json(artifact / "metadata.json")["effective_top_k"] == 180
    assert verify_directory(artifact, paths)["quality_valid"]
    for invalid in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            LinkageConfig(top_k=invalid)


def test_summary_never_loads_arrays_tables_or_sources(publication, monkeypatch):
    _, artifact = publication

    def forbidden(*args, **kwargs):
        pytest.fail("Summary must only read JSON counts")

    monkeypatch.setattr(pd, "read_parquet", forbidden)
    monkeypatch.setattr(np, "load", forbidden)
    monkeypatch.setattr("flir_pipeline.linkage.storage.load_sources", forbidden)
    monkeypatch.setattr("flir_pipeline.linkage.storage.file_sha256", forbidden)
    summary = summarize_directory(artifact)
    assert all(type(value) is int for value in summary.values())
    assert summary["candidate_pairs"] == 8


def test_cli_build_verify_summary_and_failures(publication, tmp_path):
    paths, artifact = publication
    runner = CliRunner()
    flags = [
        part
        for field in fields(paths)
        for part in (
            "--" + field.name.replace("_", "-"),
            str(getattr(paths, field.name)),
        )
    ]
    result = runner.invoke(
        app,
        ["linkage", "build", *flags, "--top-k", "2", "--output", str(tmp_path / "cli")],
    )
    assert result.exit_code == 0, result.output
    assert Path(result.output.strip()).name == artifact.name
    result = runner.invoke(app, ["linkage", "verify", str(artifact), *flags])
    assert result.exit_code == 0, result.output
    assert '"quality_valid": true' in result.output
    result = runner.invoke(app, ["linkage", "summary", str(artifact)])
    assert result.exit_code == 0 and '"candidate_pairs": 8' in result.output
    assert (
        runner.invoke(app, ["linkage", "build", *flags, "--top-k", "0"]).exit_code != 0
    )
    rewrite_table(
        artifact, "candidate_occurrences", lambda df: df.iloc[1:].reset_index(drop=True)
    )
    result = runner.invoke(app, ["linkage", "verify", str(artifact), *flags])
    assert result.exit_code == 1 and '"quality_valid": false' in result.output
