"""Offline transport integration; no encoders, real data or temporal inference."""

import hashlib
import inspect
import zipfile
from dataclasses import fields

import pandas as pd
import pytest
from test_video_variant_images import (
    assert_closed,
    frames,
    png,
    reseal,
    snapshot,
)
from test_video_variant_images import handles as handles
from test_video_variant_images import publish as publish

import flir_pipeline.data.video_variant_images as reader_module
import flir_pipeline.features.image_source as source_module
from flir_pipeline.data.video_variant_contract import Limits
from flir_pipeline.data.video_variant_images import ImageLocation, VideoVariantImages
from flir_pipeline.features.image_source import ImageSource


def test_interfaces_read_identical_bytes_pixels_and_ledger_provenance(publish):
    root, artifact = publish()
    before = snapshot(root), snapshot(artifact)
    manifest = pd.read_parquet(artifact / "manifest.parquet")
    ids = frames(artifact)
    with (
        VideoVariantImages(artifact, root, max_open_archives=1) as direct,
        ImageSource.from_video_variant(artifact, root, max_open_archives=1) as source,
    ):
        assert isinstance(source, ImageSource)
        assert source.kind == "zip_collection"
        assert source.root == root.resolve()
        assert source.archive is None and source.archive_path is None
        assert source.frame_ids == direct.frame_ids == tuple(sorted(ids.values()))
        assert source.path_column(manifest) == "frame_id"
        # Physical and logical order differ, and switching shards exercises eviction.
        for index in (2, 3, 1, 4, 2):
            content, location = direct.read(ids[index])
            assert source.read(ids[index], location.image_sha256) == content
            assert source.locate(ids[index]) == location
            with source.decode(ids[index], location.image_sha256) as image:
                assert image.mode == location.image_mode
                assert image.size == (location.width, location.height)
                assert (
                    image.getpixel((0, 0))
                    == {
                        1: (255, 0, 0),
                        2: (255, 0, 0),
                        3: (0, 0, 255),
                        4: (0, 128, 0),
                    }[index]
                )
        source.validate(manifest)
    assert (snapshot(root), snapshot(artifact)) == before


def test_copies_keep_separate_occurrences_without_temporal_or_label_inputs(
    publish, monkeypatch
):
    root, artifact = publish()
    manifest = pd.read_parquet(artifact / "manifest.parquet")
    ids = frames(artifact)
    scientific = manifest[["frame_id", "content_id", "image_sha256"]].copy()
    scientific["timestamp_seconds"] = "unverified"
    scientific["sequence_id"] = "unverified"
    scientific["yolo_annotations"] = "not-an-input"
    scientific["original_split"] = "not-an-input"
    scientific["image_path"] = "not-a-physical-locator"
    scientific["source_member_path"] = "not-a-physical-locator"
    calls = []
    original = reader_module.read_entry

    def record(*args, **kwargs):
        calls.append(args[1])
        return original(*args, **kwargs)

    monkeypatch.setattr(reader_module, "read_entry", record)
    with ImageSource.from_video_variant(artifact, root) as source:
        left, right = source.locate(ids[1]), source.locate(ids[2])
        assert left.frame_id != right.frame_id and left.entry_id != right.entry_id
        assert left.content_id == right.content_id == left.image_sha256
        assert left.archive_key != right.archive_key
        source.validate(scientific)
        assert len(calls) == len(scientific) == 4
    assert not {
        "timestamp_seconds",
        "sequence_id",
        "yolo_annotations",
        "original_split",
    } & {field.name for field in fields(ImageLocation)}


@pytest.mark.parametrize("transport", ["directory", "zip"])
def test_historical_constructor_paths_validation_and_pixels(tmp_path, transport):
    content = png("red")
    digest = hashlib.sha256(content).hexdigest()
    root = tmp_path / "images"
    root.mkdir()
    (root / "frame.png").write_bytes(content)
    archive = tmp_path / "images.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("frame.png", content)
    manifest = pd.DataFrame(
        dict(
            frame_id=["historical-copy-a", "historical-copy-b"],
            content_id=[digest, digest],
            image_sha256=[digest, digest],
            image_path=["frame.png", "frame.png"],
            relative_image_path=["ignored.png", "ignored.png"],
            source_member_path=["frame.png", "frame.png"],
            image_decode_valid=[True, True],
        )
    )
    arguments = (None, root) if transport == "directory" else (archive, None)
    with ImageSource(*arguments) as source:
        assert source.kind == transport
        assert source.path_column(manifest) == (
            "image_path" if transport == "directory" else "source_member_path"
        )
        source.validate(manifest)
        assert source.read("frame.png", digest) == content
        with source.decode("frame.png", digest) as image:
            assert image.mode == "RGB" and image.size == (6, 4)
            assert image.getpixel((0, 0)) == (255, 0, 0)
        with pytest.raises(ValueError, match="SHA256 mismatch"):
            source.read("frame.png", "0" * 64)
    if transport == "zip":
        assert source.archive.fp is None


def test_historical_public_signatures_remain_compatible():
    assert list(inspect.signature(ImageSource).parameters) == [
        "images_archive",
        "images_root",
    ]
    for method in ("read", "decode"):
        assert list(inspect.signature(getattr(ImageSource, method)).parameters) == [
            "self",
            "relative",
            "expected_sha256",
        ]
    for method in ("validate", "path_column"):
        assert list(inspect.signature(getattr(ImageSource, method)).parameters) == [
            "self",
            "manifest",
        ]


@pytest.mark.parametrize("failure", ["unknown_frame", "wrong_hash"])
def test_read_reference_errors_close_the_session(publish, handles, failure):
    root, artifact = publish()
    source = ImageSource.from_video_variant(artifact, root)
    with source:
        location = source.locate(source.frame_ids[0])
        frame_id = "absent" if failure == "unknown_frame" else location.frame_id
        digest = "0" * 64 if failure == "wrong_hash" else location.image_sha256
        with pytest.raises((KeyError, ValueError)):
            source.read(frame_id, digest)
        assert_closed(handles)
        with pytest.raises(RuntimeError, match="active reader context"):
            source.read(location.frame_id, location.image_sha256)
    source.close()
    assert_closed(handles)


@pytest.mark.parametrize(
    "failure",
    [
        "missing_frame_id",
        "missing_content_id",
        "missing_sha256",
        "null_frame_id",
        "duplicate_frame_id",
        "unknown_frame_id",
        "changed_content_id",
        "changed_hash",
        "null_content_id",
        "invalid_decode",
    ],
)
def test_validate_rejects_invalid_or_ambiguous_scientific_references(
    publish, handles, failure
):
    root, artifact = publish()
    manifest = pd.read_parquet(artifact / "manifest.parquet")
    if failure.startswith("missing_"):
        column = {"missing_sha256": "image_sha256"}.get(
            failure, failure.removeprefix("missing_")
        )
        manifest = manifest.drop(columns=column)
    elif failure == "duplicate_frame_id":
        manifest.loc[1, "frame_id"] = manifest.loc[0, "frame_id"]
    else:
        column, value = {
            "null_frame_id": ("frame_id", None),
            "unknown_frame_id": ("frame_id", "absent"),
            "changed_content_id": ("content_id", "0" * 64),
            "changed_hash": ("image_sha256", "0" * 64),
            "null_content_id": ("content_id", None),
            "invalid_decode": ("image_decode_valid", False),
        }[failure]
        manifest.loc[0, column] = value
    with ImageSource.from_video_variant(artifact, root) as source:
        with pytest.raises((KeyError, ValueError)):
            source.validate(manifest)
        assert_closed(handles)


@pytest.mark.parametrize("failure", ["ambiguous_members", "bad_publication_checksum"])
def test_factory_reuses_publication_and_ingestion_integrity(publish, handles, failure):
    if failure == "ambiguous_members":
        root, artifact = publish(
            {
                "ambiguous.zip": [
                    ("frame_1.png", png("red")),
                    ("frame_1.png", png("blue")),
                ]
            }
        )
    else:
        root, artifact = publish()
        artifact = reseal(artifact, wrong_checksum=True)
    with pytest.raises(ValueError):
        ImageSource.from_video_variant(artifact, root)
    assert_closed(handles)


@pytest.mark.parametrize("failure", ["decode", "downstream"])
def test_handles_close_when_decoding_or_a_later_consumer_fails(
    publish, handles, monkeypatch, failure
):
    root, artifact = publish()
    source = ImageSource.from_video_variant(artifact, root)

    def fail(_):
        raise RuntimeError("synthetic consumer failure")

    with pytest.raises(RuntimeError, match="synthetic consumer failure"), source:
        location = source.locate(source.frame_ids[0])
        if failure == "decode":
            monkeypatch.setattr(source_module, "decode_image_bytes", fail)
            with pytest.raises(RuntimeError, match="synthetic consumer failure"):
                source.decode(location.frame_id, location.image_sha256)
            assert_closed(handles)  # Even if a consumer catches the exception.
            raise RuntimeError("synthetic consumer failure")
        source.read(location.frame_id, location.image_sha256)
        fail(None)  # Failure outside source.read, before the context exits.
    assert_closed(handles)


def test_verification_is_once_per_zip_session_and_never_extracts(
    publish, monkeypatch, handles
):
    root, artifact = publish()
    ids = frames(artifact)
    calls = []
    actual = reader_module.sha256_stream

    def hashed(stream):
        calls.append(stream)
        return actual(stream)

    def forbidden(*args, **kwargs):
        pytest.fail("ZIP extraction must never be used")

    monkeypatch.setattr(reader_module, "sha256_stream", hashed)
    monkeypatch.setattr(zipfile.ZipFile, "extract", forbidden)
    monkeypatch.setattr(zipfile.ZipFile, "extractall", forbidden)
    with ImageSource.from_video_variant(artifact, root, max_open_archives=1) as source:
        for index in (1, 2, 3, 4, 1, 2):
            location = source.locate(ids[index])
            source.read(location.frame_id, location.image_sha256)
        assert len(calls) == 2
        assert sum(z.fp is not None for z in handles[0]) <= 1
    assert_closed(handles)


def test_tighter_limits_forwarded_to_the_shared_reader(publish, handles):
    root, artifact = publish()
    with ImageSource.from_video_variant(
        artifact, root, limits=Limits(max_pixels=1)
    ) as source:
        location = source.locate(source.frame_ids[0])
        with pytest.raises(ValueError, match="decode QA failed"):
            source.read(location.frame_id, location.image_sha256)
        assert_closed(handles)


def test_same_member_name_in_distinct_shards_uses_occurrence_locator(publish):
    root, artifact = publish(
        {
            "first.zip": [("frame_1.png", png("red"))],
            "second.zip": [("frame_1.png", png("blue"))],
        },
        series=[
            dict(
                series_id=key,
                archive_keys=[f"shard-{i}"],
                index_pattern=r"frame_(?P<index>\d+)\.png",
            )
            for i, key in enumerate(("left", "right"))
        ],
    )
    manifest = pd.read_parquet(artifact / "manifest.parquet")
    with ImageSource.from_video_variant(artifact, root) as source:
        for row in manifest.itertuples(index=False):
            location = source.locate(row.frame_id)
            assert location.source_member_path == "frame_1.png"
            assert source.read(row.frame_id, row.image_sha256) == png(
                "red" if row.series_id == "left" else "blue"
            )


def test_inconsistent_ledger_cannot_be_consumed_even_with_rebound_checksums(publish):
    root, artifact = publish()
    occurrences = pd.read_parquet(artifact / "occurrences.parquet")
    occurrences.loc[0, "entry_id"] = "0" * 64
    occurrences.to_parquet(artifact / "occurrences.parquet", index=False)
    artifact = reseal(artifact)
    with pytest.raises(ValueError):
        ImageSource.from_video_variant(artifact, root)


def test_enter_failure_closes_every_archive_already_opened(publish, handles):
    root, artifact = publish()
    source = ImageSource.from_video_variant(artifact, root)
    with (root / "second.zip").open("ab") as stream:
        stream.write(b"synthetic mutation")
    with pytest.raises(ValueError, match="ZIP source size"), source:
        pytest.fail("Unverified sources must never enter the context")
    assert_closed(handles)
