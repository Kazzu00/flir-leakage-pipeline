"""Synthetic sampling receipts and JPEGs; no videos, FFmpeg, models or network."""

import hashlib
import json
import shutil
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from typer.testing import CliRunner

from flir_pipeline.cli import app
from flir_pipeline.data.identity import dataset_id_from_manifest
from flir_pipeline.data.video_frames import (
    FRAME_DTYPES,
    PRODUCER,
    SCHEMA_VERSION,
    video_id_for,
)
from flir_pipeline.data.video_manifest import MANIFEST_VERSION, build_video_manifest
from flir_pipeline.features.base import DeterministicFakeExtractor
from flir_pipeline.features.storage import (
    extract_to_store,
    summarize_feature_directory,
    verify_feature_directory,
    verify_features_against_manifest,
)
from flir_pipeline.utils.hashing import sha256_file


def write_receipt(root, frames, summary):
    frames.to_parquet(root / "frames.parquet", index=False)
    summary["frames_parquet_sha256"] = sha256_file(root / "frames.parquet")
    (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


@pytest.fixture
def samples(tmp_path):
    root = tmp_path / "samples"
    root.mkdir()
    rows, videos = [], []
    for name, colors in [("one.mp4", ["red", "red", "blue"]), ("sub/two.mov", ["red"])]:
        identity = video_id_for(name)
        (root / identity).mkdir()
        facts = {
            "source_fps": 25.0,
            "source_width": 8,
            "source_height": 6,
            "source_duration_seconds": 3.0,
            "source_nb_frames": 75,
            "codec_name": "synthetic",
        }
        for index, color in enumerate(colors):
            relative = f"{identity}/frame_{index:06d}.jpg"
            Image.new("RGB", (8, 6), color).save(root / relative, format="JPEG")
            rows.append(
                {
                    "video_id": identity,
                    "source_video": name,
                    "sample_index": index,
                    "timestamp_seconds": float(index),
                    "source_frame_index_estimate": index * 25,
                    "sample_fps": 1.0,
                    "image_path": relative,
                    **facts,
                }
            )
        videos.append(
            {
                "video_id": identity,
                "source_video": name,
                "extracted_frames": len(colors),
                "source_sha256": hashlib.sha256(name.encode()).hexdigest(),
                **facts,
            }
        )
    frames = pd.DataFrame(rows).astype(FRAME_DTYPES)
    summary = {
        "producer": PRODUCER,
        "schema_version": SCHEMA_VERSION,
        "sample_fps": 1.0,
        "processed_videos": 2,
        "total_frames": len(rows),
        "videos": videos,
    }
    write_receipt(root, frames, summary)
    return root, frames, summary


def build(samples, tmp_path):
    path = tmp_path / "manifest.parquet"
    report = build_video_manifest(samples[0], path, tmp_path / "reports")
    return path, pd.read_parquet(path), report


def test_occurrences_content_identity_portability_and_read_only(samples, tmp_path):
    root, frames, summary = samples
    before = {
        p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()
    }
    _, manifest, report = build(samples, tmp_path)
    assert report["total_records"] == 4 and report["unique_content_ids"] == 2
    assert report["exact_duplicate_content_ids"] == 1
    assert report["duplicate_records"] == 3 and report["redundant_records"] == 2
    assert sorted(report["records_per_video"].values()) == [1, 3]
    assert manifest.frame_id.is_unique and manifest.image_decode_valid.all()
    repeated = manifest[manifest.exact_duplicate]
    assert len(repeated) == 3 and repeated.content_id.nunique() == 1
    assert repeated.duplicate_occurrence_count.eq(3).all()
    assert repeated.duplicate_group_id.eq("duplicate-" + repeated.content_id).all()
    assert manifest.content_id.eq(manifest.image_sha256).all()
    assert manifest.label_sha256.eq("").all() and not manifest.label_exists.any()
    assert manifest.original_split.eq("").all() and manifest.source_archive.eq("").all()
    assert manifest.manifest_version.eq(MANIFEST_VERSION).all()
    assert not {
        "sequence_id",
        "possible_sequence",
        "possible_frame_index",
        "split_id",
    } & set(manifest)
    assert manifest.image_path.eq(manifest.source_member_path).all()
    assert manifest.source_type.eq("directory").all()
    assert report["read_only_source"] and report["decode_failures"] == 0
    assert report["frames_parquet_sha256"] == sha256_file(root / "frames.parquet")
    assert report["sampling_summary_sha256"] == sha256_file(root / "summary.json")
    assert {
        p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()
    } == before
    moved = tmp_path / "relocated"
    shutil.copytree(root, moved)
    write_receipt(moved, frames.iloc[::-1], summary)
    _, second, second_report = build((moved, frames, summary), tmp_path)
    pd.testing.assert_frame_equal(manifest, second)
    assert (
        report["dataset_id"]
        == second_report["dataset_id"]
        == dataset_id_from_manifest(manifest.iloc[::-1])
    )
    relative = frames.iloc[0].image_path
    Image.new("RGB", (8, 6), "green").save(moved / relative)
    _, changed, changed_report = build((moved, frames, summary), tmp_path)
    assert changed_report["dataset_id"] != report["dataset_id"]
    assert (
        changed.set_index("image_path").loc[relative, "frame_id"]
        != manifest.set_index("image_path").loc[relative, "frame_id"]
    )


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("producer", "wrong", "producer/schema"),
        ("schema_version", 2, "producer/schema"),
        ("processed_videos", 3, "counts"),
        ("total_frames", 90, "counts"),
        ("sample_fps", 2.0, "provenance"),
        ("sample_fps", float("nan"), "sample_fps"),
    ],
)
def test_bad_summary(samples, tmp_path, field, value, match):
    root, frames, summary = samples
    summary[field] = value
    write_receipt(root, frames, summary)
    with pytest.raises(ValueError, match=match):
        build(samples, tmp_path)
    assert not (tmp_path / "manifest.parquet").exists()


@pytest.mark.parametrize(
    "column,value,match",
    [
        ("image_path", "../outside.jpg", "Unsafe"),
        ("image_path", "/absolute.jpg", "Unsafe"),
        ("image_path", "C:/private.jpg", "Unsafe"),
        ("image_path", "folder\\image.jpg", "Unsafe"),
        ("image_path", "folder/./image.jpg", "Unsafe"),
        ("image_path", "folder//image.jpg", "Unsafe"),
        ("image_path", "different/frame_000000.jpg", "image_path"),
        ("timestamp_seconds", 99.0, "sampling grid"),
        ("source_frame_index_estimate", 99, "estimate"),
        ("source_width", 20, "source_width"),
        ("source_video", "other.mp4", "provenance"),
        ("sample_index", 1, "Duplicate temporal"),
    ],
)
def test_bad_frame_rows(samples, tmp_path, column, value, match):
    root, frames, summary = samples
    frames.loc[0, column] = value
    write_receipt(root, frames, summary)
    with pytest.raises(ValueError, match=match):
        build(samples, tmp_path)


@pytest.mark.parametrize("missing", ["frames.parquet", "summary.json", "image"])
def test_missing_input(samples, tmp_path, missing):
    root, frames, _ = samples
    (root / (frames.iloc[0].image_path if missing == "image" else missing)).unlink()
    with pytest.raises(ValueError, match="missing"):
        build(samples, tmp_path)


def test_checksum_and_publication_marker(samples, tmp_path):
    root, frames, summary = samples
    (root / "frames.parquet").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        build(samples, tmp_path)
    write_receipt(root, frames, summary)
    (root / ".video-frames-publishing").touch()
    with pytest.raises(ValueError, match="publication"):
        build(samples, tmp_path)


@pytest.mark.parametrize("change", ["counts", "identity", "duplicate", "schema"])
def test_video_receipt_consistency(samples, tmp_path, change):
    root, frames, summary = samples
    if change == "counts":
        summary["videos"][0]["extracted_frames"] = 2
        summary["videos"][1]["extracted_frames"] = 2
    elif change == "identity":
        summary["videos"][0]["video_id"] = video_id_for("unknown.mp4")
    elif change == "duplicate":
        summary["videos"][1] = summary["videos"][0].copy()
    else:
        frames = frames.drop(columns="source_fps")
    write_receipt(root, frames, summary)
    with pytest.raises(ValueError):
        build(samples, tmp_path)


def test_occurrence_binds_source_video_bytes_and_sampling_grid(samples, tmp_path):
    root, frames, summary = samples
    _, first, report = build(samples, tmp_path)
    summary["videos"][0]["source_sha256"] = "a" * 64
    write_receipt(root, frames, summary)
    _, second, source_report = build(samples, tmp_path)
    assert report["dataset_id"] != source_report["dataset_id"]
    assert first.content_id.equals(second.content_id)
    assert (first.frame_id != second.frame_id).sum() == 3
    summary["sample_fps"] = 2.0
    frames["sample_fps"] = 2.0
    frames["timestamp_seconds"] = frames.sample_index / 2.0
    frames["source_frame_index_estimate"] = np.floor(
        frames.timestamp_seconds * 25 + 0.5
    ).astype("Int64")
    write_receipt(root, frames, summary)
    _, third, grid_report = build(samples, tmp_path)
    assert source_report["dataset_id"] != grid_report["dataset_id"]
    assert second.content_id.equals(third.content_id)
    assert not second.frame_id.eq(third.frame_id).any()


def test_non_jpeg_bytes_are_explicit_decode_failure(samples, tmp_path):
    root, frames, _ = samples
    Image.new("RGB", (8, 6), "green").save(
        root / frames.iloc[0].image_path, format="PNG"
    )
    _, _, report = build(samples, tmp_path)
    assert report["decode_failures"] == 1
    assert "Expected JPEG" in report["decode_failure_records"][0]["image_error"]


def test_real_manifest_schema_fake_features_to_video_similarity(samples, tmp_path):
    """Exercise the actual JPEG/manifest/feature adapters with synthetic pixels."""
    from flir_pipeline.similarity.storage import (
        SimilarityConfig,
        compute_to_store,
        verify_similarity_directory,
    )

    class PinnedSyntheticExtractor(DeterministicFakeExtractor):
        def metadata(self):
            return {**super().metadata(), "model_revision": "a"*40,
                    "resolved_model_revision": "a"*40}

    path, manifest, _ = build(samples, tmp_path)
    features = extract_to_store(path, extractor=PinnedSyntheticExtractor(),
                                images_root=samples[0], output_root=tmp_path/"features")
    config = SimilarityConfig(top_k=1, algorithm_version="content_cosine_v2",
                              provenance_mode="sampled_video_grid", pair_storage="summary_only_v1",
                              temporal_rule="all_occurrences_min_same_source_video_gaps_v1",
                              cross_split_rule="unavailable")
    output = compute_to_store(features, path, config, tmp_path/"similarity")
    assert verify_similarity_directory(output, features, path)["quality_valid"]
    assert np.load(output/"cosine_similarity.npy").shape == (2, 2)
    assert len(pd.read_parquet(output/"record_provenance.parquet")) == len(manifest) == 4
    assert not (output/"pair_analysis.parquet").exists()


def test_corrupt_jpeg_is_retained_and_features_refuse(samples, tmp_path):
    root, frames, _ = samples
    (root / frames.iloc[0].image_path).write_bytes(b"not a JPEG")
    path, manifest, report = build(samples, tmp_path)
    assert len(manifest) == 4 and report["decode_failures"] == 1
    assert len(report["decode_failure_records"]) == 1
    assert (
        report["decode_failure_records"][0]["image_path"] == frames.iloc[0].image_path
    )
    assert manifest.image_decode_valid.sum() == 3
    with pytest.raises(ValueError, match="invalid image decodes"):
        extract_to_store(
            path,
            extractor=DeterministicFakeExtractor(),
            images_root=root,
            output_root=tmp_path / "features",
        )
    assert not (tmp_path / "features").exists()


@pytest.mark.parametrize("target", ["image", "parent"])
def test_symlink_escape(samples, tmp_path, target):
    root, frames, _ = samples
    path, _, _ = build(samples, tmp_path)
    local = root / frames.iloc[0].image_path
    if target == "parent":
        local = local.parent
    outside = tmp_path / "outside"
    local.rename(outside)
    try:
        local.symlink_to(outside, target_is_directory=target == "parent")
    except OSError:
        pytest.skip("Symlink creation unavailable on this platform/account")
    with pytest.raises(ValueError, match="escapes root"):
        build(samples, tmp_path)
    with pytest.raises(ValueError, match="escapes root"):
        extract_to_store(
            path,
            extractor=DeterministicFakeExtractor(),
            images_root=root,
            output_root=tmp_path / "features",
        )


def test_optional_video_facts_remain_null(samples, tmp_path):
    root, frames, summary = samples
    for key in [
        "source_fps",
        "source_nb_frames",
        "source_duration_seconds",
        "source_frame_index_estimate",
    ]:
        frames[key] = pd.Series(pd.NA, index=frames.index, dtype=FRAME_DTYPES[key])
        for video in summary["videos"]:
            video[key] = None
    write_receipt(root, frames, summary)
    _, manifest, _ = build(samples, tmp_path)
    assert (
        manifest.source_fps.isna().all()
        and manifest.source_frame_index_estimate.isna().all()
    )


def test_local_features_match_zip_and_preserve_all_occurrences(samples, tmp_path):
    root, frames, _ = samples
    path, manifest, _ = build(samples, tmp_path)
    extractor = DeterministicFakeExtractor(6)
    local = extract_to_store(
        path, extractor=extractor, images_root=root, output_root=tmp_path / "local"
    )
    archive = tmp_path / "images.zip"
    with ZipFile(archive, "w") as stream:
        for relative in frames.image_path:
            stream.write(root / relative, relative)
    zipped = extract_to_store(path, archive, extractor, tmp_path / "zipped")
    assert local.name == zipped.name
    np.testing.assert_array_equal(
        np.load(local / "embeddings_raw.npy"), np.load(zipped / "embeddings_raw.npy")
    )
    np.testing.assert_array_equal(
        np.load(local / "embeddings_l2.npy"), np.load(zipped / "embeddings_l2.npy")
    )
    assert np.load(local / "embeddings_raw.npy").shape == (2, 6)
    records = pd.read_parquet(local / "record_index.parquet")
    assert len(records) == 4 and records.frame_id.is_unique
    assert records.groupby("content_id").embedding_row.nunique().eq(1).all()
    assert verify_feature_directory(local)["quality_valid"]
    assert verify_features_against_manifest(local, manifest)["full_dataset_valid"]
    metadata = summarize_feature_directory(local)["metadata"]
    assert metadata["image_source_type"] == "directory" and metadata["read_only_source"]
    assert (
        metadata["cache_signature"]
        == summarize_feature_directory(zipped)["metadata"]["cache_signature"]
    )
    assert (
        extract_to_store(
            path,
            extractor=extractor,
            images_root=root,
            output_root=tmp_path / "local",
            batch_size=1,
        )
        == local
    )
    smoke = extract_to_store(
        path,
        extractor=extractor,
        images_root=root,
        output_root=tmp_path / "smoke",
        limit_content=1,
    )
    assert smoke.name == local.name and verify_feature_directory(smoke)["quality_valid"]
    assert pd.read_parquet(smoke / "record_index.parquet").embedding_row.eq(-1).any()


def test_extra_input_columns_do_not_invent_sequences_or_labels(samples, tmp_path):
    root, frames, summary = samples
    frames["possible_sequence"] = "unvalidated"
    frames["original_split"] = "train"
    frames["num_objects"] = 42
    write_receipt(root, frames, summary)
    _, manifest, _ = build(samples, tmp_path)
    assert "possible_sequence" not in manifest and "num_objects" not in manifest
    assert manifest.original_split.eq("").all()


def test_historical_cli_fallback_and_missing_source(samples, tmp_path, monkeypatch):
    root, frames, _ = samples
    path, _, _ = build(samples, tmp_path)
    with ZipFile(tmp_path / "Imagenes.zip", "w") as archive:
        for relative in frames.image_path:
            archive.write(root / relative, relative)
    runner = CliRunner()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FLIR_DATA_ROOT", str(tmp_path))
    args = [
        "features",
        "extract",
        "--extractor",
        "fake",
        "--manifest",
        str(path),
        "--output-root",
        str(tmp_path / "features"),
    ]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    monkeypatch.delenv("FLIR_DATA_ROOT")
    result = runner.invoke(app, args)
    assert result.exit_code != 0 and "Provide" in result.output
    with pytest.raises(ValueError, match="exactly one"):
        extract_to_store(path, extractor=DeterministicFakeExtractor())
    with pytest.raises(ValueError, match="exactly one"):
        extract_to_store(
            path,
            tmp_path / "Imagenes.zip",
            DeterministicFakeExtractor(),
            images_root=root,
        )


def test_missing_duplicate_is_rejected_even_in_smoke(samples, tmp_path):
    root, frames, _ = samples
    path, _, _ = build(samples, tmp_path)
    (root / frames.iloc[1].image_path).unlink()
    with pytest.raises(ValueError, match="missing"):
        extract_to_store(
            path,
            extractor=DeterministicFakeExtractor(),
            images_root=root,
            output_root=tmp_path / "features",
            limit_content=1,
        )


@pytest.mark.parametrize("cached", [False, True])
def test_changed_duplicate_bytes_rejected_before_extraction_or_reuse(
    samples, tmp_path, cached
):
    root, frames, _ = samples
    path, _, _ = build(samples, tmp_path)
    options = dict(
        extractor=DeterministicFakeExtractor(),
        images_root=root,
        output_root=tmp_path / "features",
    )
    if cached:
        extract_to_store(path, **options)
    Image.new("RGB", (8, 6), "green").save(root / frames.iloc[1].image_path)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        extract_to_store(path, **options)


def test_local_resume_checks_bytes_and_keeps_checkpoint(samples, tmp_path):
    root, frames, _ = samples
    path, _, _ = build(samples, tmp_path)

    class Interruptible(DeterministicFakeExtractor):
        calls = 0
        fail = True

        def encode_batch(self, batch):
            self.calls += 1
            if self.calls == 2 and self.fail:
                raise RuntimeError("synthetic interruption")
            return super().encode_batch(batch)

    extractor = Interruptible(6)
    options = dict(
        extractor=extractor,
        images_root=root,
        output_root=tmp_path / "features",
        batch_size=1,
    )
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        extract_to_store(path, **options)
    checkpoint = next((tmp_path / "features").rglob("checkpoint.json"))
    before = checkpoint.read_bytes()
    image = root / frames.iloc[0].image_path
    original = image.read_bytes()
    image.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        extract_to_store(path, **options)
    assert checkpoint.read_bytes() == before
    image.write_bytes(original)
    extractor.fail, extractor.calls = False, 0
    result = extract_to_store(path, **options)
    assert extractor.calls == 1 and not checkpoint.exists()
    assert verify_feature_directory(result)["quality_valid"]


@pytest.mark.parametrize(
    "relative", ["../outside.jpg", "C:/outside.jpg", "a\\b.jpg", "/absolute.jpg"]
)
def test_feature_local_traversal(samples, tmp_path, relative):
    path, manifest, _ = build(samples, tmp_path)
    manifest.loc[0, "image_path"] = relative
    manifest.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="Unsafe"):
        extract_to_store(
            path,
            extractor=DeterministicFakeExtractor(),
            images_root=samples[0],
            output_root=tmp_path / "features",
        )


def test_read_only_output_boundaries(samples, tmp_path):
    root = samples[0]
    with pytest.raises(ValueError, match="outside"):
        build_video_manifest(root, root / "frames.parquet", tmp_path / "reports")
    with pytest.raises(ValueError, match="outside"):
        build_video_manifest(root, tmp_path / "manifest.parquet", root)
    path, _, _ = build(samples, tmp_path)
    with pytest.raises(ValueError, match="outside"):
        extract_to_store(
            path,
            extractor=DeterministicFakeExtractor(),
            images_root=root,
            output_root=root / "features",
        )


def test_cli_contract_and_local_workflow(samples, tmp_path, monkeypatch):
    runner = CliRunner()
    assert runner.invoke(app, ["data", "build-video-manifest", "--help"]).exit_code == 0
    for missing in ["--frames-root", "--output", "--report-output"]:
        values = {
            "--frames-root": samples[0],
            "--output": tmp_path / "manifest.parquet",
            "--report-output": tmp_path / "reports",
        }
        args = ["data", "build-video-manifest"]
        for key, value in values.items():
            if key != missing:
                args.extend([key, str(value)])
        assert runner.invoke(app, args).exit_code != 0
    args = [
        "data",
        "build-video-manifest",
        "--frames-root",
        str(samples[0]),
        "--output",
        str(tmp_path / "manifest.parquet"),
        "--report-output",
        str(tmp_path / "reports"),
    ]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    monkeypatch.setenv("FLIR_DATA_ROOT", str(tmp_path / "nonexistent-historical-root"))
    extract = [
        "features",
        "extract",
        "--extractor",
        "fake",
        "--manifest",
        str(tmp_path / "manifest.parquet"),
        "--images-root",
        str(samples[0]),
        "--output-root",
        str(tmp_path / "features"),
    ]
    result = runner.invoke(app, extract + ["--images-archive", "images.zip"])
    assert result.exit_code != 0 and "exactly one" in result.output
    result = runner.invoke(app, extract)
    assert result.exit_code == 0, result.output
    feature_dir = next((tmp_path / "features").rglob("metadata.json")).parent
    for command in ["verify", "summary"]:
        result = runner.invoke(app, ["features", command, str(feature_dir)])
        assert result.exit_code == 0, result.output
    (samples[0] / ".video-frames-publishing").touch()
    result = runner.invoke(app, args)
    assert result.exit_code == 1 and "publication" in result.output
