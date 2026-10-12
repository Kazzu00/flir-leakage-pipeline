"""Synthetic multishard reader checks; no private data, encoders or downloads."""

import io
import json
import os
import subprocess
import sys
import zipfile
from dataclasses import FrozenInstanceError, fields
from pathlib import Path

import pandas as pd
import pytest
import yaml
from PIL import Image

import flir_pipeline.data.video_variant_images as reader_module
from flir_pipeline.data.variants import make_variant
from flir_pipeline.data.video_variant_contract import (
    IngestionConfig,
    Limits,
    digest_document,
)
from flir_pipeline.data.video_variant_images import ImageLocation, VideoVariantImages
from flir_pipeline.data.video_variant_ingestion import derive_evidence
from flir_pipeline.data.video_variant_storage import (
    build_ingestion,
    inspect_ingestion,
    verify_ingestion,
)
from flir_pipeline.data.zip_image_collection import EntryLimitError
from flir_pipeline.utils.hashing import sha256_file


def png(color, size=(6, 4)):
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def publish(tmp_path):
    count = 0

    def build(shards=None, *, series=None, compression=zipfile.ZIP_DEFLATED):
        nonlocal count
        case = tmp_path / str(count)
        count += 1
        root = case / "inputs"
        root.mkdir(parents=True)
        shards = shards or {
            "first.zip": [
                ("note.txt", b"metadata"),
                ("frame_3.png", png("blue")),
                ("frame_1.png", png("red")),
            ],
            "second.zip": [("frame_4.png", png("green")), ("frame_2.png", png("red"))],
        }
        for name, pairs in shards.items():
            (root / name).parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(root / name, "w", compression) as archive:
                for member, content in pairs:
                    if member in archive.namelist():
                        with pytest.warns(UserWarning, match="Duplicate name"):
                            archive.writestr(member, content)
                    else:
                        archive.writestr(member, content)
        keys = [f"shard-{i}" for i in range(len(shards))]
        declaration = dict(
            collection_id="synthetic-collection",
            variant_name="external_processed",
            transformation={"description": "synthetic external image processing"},
            archives=[
                dict(archive_key=k, path=p) for k, p in zip(keys, shards, strict=True)
            ],
            series=series
            or [
                dict(
                    series_id="example-series",
                    archive_keys=keys,
                    index_pattern=r"frame_(?P<index>\d+)\.png",
                    expected_index_start=1,
                    expected_index_stop=4,
                )
            ],
        )
        config = case / "config.yaml"
        config.write_text(yaml.safe_dump(declaration), encoding="utf-8")
        return root, build_ingestion(config, root, case / "outputs")

    return build


def frames(artifact):
    return (
        pd.read_parquet(artifact / "manifest.parquet")
        .set_index("observed_index")
        .frame_id.to_dict()
    )


def snapshot(directory):
    return {
        p.relative_to(directory).as_posix(): sha256_file(p)
        for p in directory.rglob("*")
        if p.is_file()
    }


def reseal(artifact, *, wrong_checksum=False):
    """Rebind synthetic bytes to test semantic verification beyond checksums."""
    metadata = json.loads((artifact / "metadata.json").read_text())
    metadata["identity"]["output_checksums"] = {
        p.name: sha256_file(p)
        for p in artifact.iterdir()
        if p.name not in {"metadata.json", "receipt.json"}
    }
    if wrong_checksum:
        metadata["identity"]["output_checksums"]["manifest.parquet"] = "0" * 64
    metadata["artifact_id"] = digest_document(metadata["identity"])
    (artifact / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (artifact / "receipt.json").write_text(
        json.dumps({"metadata_sha256": sha256_file(artifact / "metadata.json")}),
        encoding="utf-8",
    )
    destination = artifact.parent / metadata["artifact_id"]
    artifact.rename(destination)
    return destination


@pytest.fixture
def handles(monkeypatch):
    actual = zipfile.ZipFile
    opened = []
    streams = []

    class TrackedZip(actual):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            opened.append(self)
            streams.append(self.fp)

    monkeypatch.setattr(reader_module.zipfile, "ZipFile", TrackedZip)
    return opened, streams


def assert_closed(handles):
    archives, streams = handles
    assert all(z.fp is None for z in archives)
    assert all(s.closed for s in streams)


def test_interleaved_shards_physical_order_and_occurrence_mapping(publish, monkeypatch):
    root, artifact = publish()
    before = snapshot(root), snapshot(artifact)
    ids = frames(artifact)
    original_open = zipfile.ZipFile.open
    calls = []

    def by_zipinfo(archive, member, *args, **kwargs):
        assert isinstance(member, zipfile.ZipInfo)
        calls.append(member)
        return original_open(archive, member, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "open", by_zipinfo)
    expected = {1: png("red"), 2: png("red"), 3: png("blue"), 4: png("green")}
    images = VideoVariantImages(artifact, root, max_open_archives=1)
    assert images.frame_ids == tuple(sorted(ids.values()))
    assert images.locate(ids[1]).member_ordinal == 2  # Auxiliaries count as entries.
    assert images.locate(ids[2]).archive_key == "shard-1"
    assert images.locate(ids[1]).content_id == images.locate(ids[2]).content_id
    assert images.locate(ids[1]).entry_id != images.locate(ids[2]).entry_id
    with images:
        for index in (2, 3, 1, 4, 2):
            content, location = images.read(ids[index])
            assert content == expected[index]
            assert location == images.locate(ids[index])
            assert location.artifact_id == artifact.name
            assert location.content_id == location.image_sha256
            assert not Path(location.source_archive).is_absolute()
            with pytest.raises(FrozenInstanceError):
                location.member_ordinal = 99
    assert (
        len(calls) == 5
    )  # Repeated reads do not cache images or deduplicate occurrences.
    assert (snapshot(root), snapshot(artifact)) == before
    checked = inspect_ingestion(artifact)
    assert images.dataset_id == checked["summary"]["dataset_id"]
    assert images.dataset_variant_id == checked["summary"]["dataset_variant_id"]
    assert not {"timestamp_seconds", "video_id", "sequence_id", "label", "split_id"} & {
        f.name for f in fields(ImageLocation)
    }
    assert verify_ingestion(artifact, root)["source_bound"]


def test_repackaging_preserves_scientific_identity(publish):
    root, first = publish()
    other, second = publish(
        {
            "repacked.zip": [
                ("nested/frame_4.png", png("green")),
                ("nested/frame_2.png", png("red")),
                ("frame_1.png", png("red")),
                ("frame_3.png", png("blue")),
            ]
        },
        compression=zipfile.ZIP_STORED,
    )
    pd.testing.assert_frame_equal(
        pd.read_parquet(first / "manifest.parquet"),
        pd.read_parquet(second / "manifest.parquet"),
    )
    assert sha256_file(first / "manifest.parquet") == sha256_file(
        second / "manifest.parquet"
    )
    with (
        VideoVariantImages(first, root) as left,
        VideoVariantImages(second, other) as right,
    ):
        assert left.frame_ids == right.frame_ids
        assert left.dataset_id == right.dataset_id
        assert left.dataset_variant_id == right.dataset_variant_id
        assert left.artifact_id != right.artifact_id
        for frame_id in left.frame_ids:
            a, old = left.read(frame_id)
            b, new = right.read(frame_id)
            assert a == b and old.content_id == new.content_id
            assert old.entry_id != new.entry_id


def test_equal_member_names_in_different_shards_resolve_exactly(publish):
    series = [
        dict(
            series_id=k,
            archive_keys=[f"shard-{i}"],
            index_pattern=r"frame_(?P<index>\d+)\.png",
        )
        for i, k in enumerate(("left", "right"))
    ]
    root, artifact = publish(
        {
            "first.zip": [("frame_1.png", png("red"))],
            "second.zip": [("frame_1.png", png("blue"))],
        },
        series=series,
    )
    manifest = pd.read_parquet(artifact / "manifest.parquet")
    with VideoVariantImages(artifact, root, max_open_archives=1) as images:
        for row in manifest.itertuples(index=False):
            content, location = images.read(row.frame_id)
            assert location.source_member_path == "frame_1.png"
            assert location.member_ordinal == 0
            assert content == png("red" if row.series_id == "left" else "blue")


def test_repeated_auxiliary_names_preserve_complete_entry_ordinals(publish):
    root, artifact = publish(
        {
            "all.zip": [
                ("folder/", b""),
                ("note.txt", b"first"),
                ("note.txt", b"second"),
                *[(f"frame_{i}.png", png("red")) for i in range(1, 5)],
            ]
        }
    )
    with VideoVariantImages(artifact, root) as images:
        content, location = images.read(frames(artifact)[1])
        assert location.member_ordinal == 3 and content == png("red")


@pytest.mark.parametrize("same_bytes", [False, True])
def test_repeated_image_names_stay_ambiguous_and_are_rejected(publish, same_bytes):
    root, artifact = publish(
        {
            "all.zip": [
                ("frame_1.png", png("red")),
                ("frame_1.png", png("red" if same_bytes else "blue")),
                *[(f"frame_{i}.png", png("green")) for i in range(2, 5)],
            ]
        }
    )
    assert not inspect_ingestion(artifact)["summary"]["scientific_manifest_available"]
    entries = pd.read_parquet(artifact / "entries.parquet")
    assert entries.iloc[:2].member_ordinal.tolist() == [0, 1]
    assert entries.iloc[:2].entry_id.nunique() == 2
    with pytest.raises(ValueError, match="available scientific manifest"):
        VideoVariantImages(artifact, root)


@pytest.mark.parametrize(
    "failure", ["missing_index", "unparsed_index", "corrupt_image"]
)
def test_incomplete_or_failed_ingestion_is_not_consumable(publish, failure):
    pairs = [(f"frame_{i}.png", png("red")) for i in range(1, 5)]
    if failure == "missing_index":
        pairs.pop()
    elif failure == "unparsed_index":
        pairs[0] = ("unknown.png", png("red"))
    else:
        pairs[0] = ("frame_1.png", b"not-a-png")
    root, artifact = publish({"all.zip": pairs})
    with pytest.raises(ValueError, match="valid integrity"):
        VideoVariantImages(artifact, root)


@pytest.mark.parametrize(
    "failure", ["missing_file", "bad_receipt", "bad_checksum", "staging"]
)
def test_invalid_or_partial_publication_rejected(publish, failure):
    root, artifact = publish()
    if failure == "missing_file":
        (artifact / "manifest.parquet").unlink()
    elif failure == "bad_receipt":
        (artifact / "receipt.json").write_text('{"metadata_sha256":"invalid"}')
    elif failure == "bad_checksum":
        artifact = reseal(artifact, wrong_checksum=True)
    else:
        destination = artifact.with_name("ingestion.partial")
        artifact.rename(destination)
        artifact = destination
    with pytest.raises(ValueError):
        VideoVariantImages(artifact, root)


@pytest.mark.parametrize(
    "failure",
    [
        "missing_occurrence",
        "duplicate_occurrence",
        "duplicate_frame",
        "wrong_content",
        "unknown_entry",
        "unknown_archive",
    ],
)
def test_rebound_checksums_cannot_hide_inconsistent_ledger(publish, failure):
    root, artifact = publish()
    name = (
        "manifest"
        if failure == "duplicate_frame"
        else "entries"
        if failure == "unknown_archive"
        else "occurrences"
    )
    frame = pd.read_parquet(artifact / f"{name}.parquet")
    if failure == "missing_occurrence":
        frame = frame.iloc[1:]
    elif failure == "duplicate_occurrence":
        frame = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
    elif failure == "duplicate_frame":
        frame.loc[0, "frame_id"] = frame.loc[1, "frame_id"]
    elif failure == "wrong_content":
        frame.loc[0, "content_id"] = "0" * 64
    elif failure == "unknown_entry":
        frame.loc[0, "entry_id"] = "0" * 64
    else:
        frame.loc[0, "archive_key"] = "absent"
    frame.to_parquet(artifact / f"{name}.parquet", index=False)
    artifact = reseal(artifact)
    with pytest.raises(ValueError):
        VideoVariantImages(artifact, root)


@pytest.mark.parametrize(
    "failure", ["missing_zip", "wrong_size", "same_size_changed_bytes"]
)
def test_original_zip_modifications_are_rejected(publish, failure):
    root, artifact = publish()
    path = root / "first.zip"
    if failure == "missing_zip":
        path.unlink()
    elif failure == "wrong_size":
        with path.open("ab") as stream:
            stream.write(b"changed")
    else:
        data = bytearray(path.read_bytes())
        data[10] ^= 1
        path.write_bytes(data)
    with pytest.raises(ValueError):
        with VideoVariantImages(artifact, root):
            pytest.fail("Changed source was accepted")


def test_sources_changed_during_session_close_handles(publish, monkeypatch):
    root, artifact = publish()
    images = VideoVariantImages(artifact, root)
    opened = []
    original = reader_module.zipfile.ZipFile

    def tracked(*args, **kwargs):
        archive = original(*args, **kwargs)
        opened.append((archive, archive.fp))
        return archive

    monkeypatch.setattr(reader_module.zipfile, "ZipFile", tracked)
    with images:
        path = root / "first.zip"
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10_000_000))
        with pytest.raises(ValueError, match="changed during"):
            images.read(frames(artifact)[1])
        assert all(z.fp is None and s.closed for z, s in opened)
        with pytest.raises(RuntimeError, match="active reader context"):
            images.read(frames(artifact)[2])


@pytest.mark.parametrize(
    "limit",
    [
        "max_member_bytes",
        "max_width",
        "max_height",
        "max_pixels",
        "max_archive_entries",
        "max_archive_uncompressed_bytes",
    ],
)
def test_stricter_consumer_limits_close_resources(publish, limit, handles):
    root, artifact = publish()
    limits = Limits.model_validate({limit: 1})
    with pytest.raises((EntryLimitError, ValueError), match="exceeds|QA failed"):
        with VideoVariantImages(artifact, root, limits=limits) as images:
            images.read(frames(artifact)[1])
    assert_closed(handles)


def test_compression_ratio_limit_uses_shared_reader(publish, handles):
    pairs = [(f"frame_{i}.png", png("red", (32, 32))) for i in range(1, 5)]
    root, artifact = publish({"all.zip": pairs})
    with pytest.raises(EntryLimitError, match="max_compression_ratio"):
        with VideoVariantImages(
            artifact, root, limits=Limits(max_compression_ratio=1.0)
        ) as images:
            images.read(frames(artifact)[1])
    assert_closed(handles)


def test_actual_decompression_is_bounded_even_when_declared_size_lies(
    publish, monkeypatch, handles
):
    root, artifact = publish()
    images = VideoVariantImages(artifact, root, limits=Limits(max_member_bytes=100))
    with images:
        monkeypatch.setattr(
            zipfile.ZipFile, "open", lambda *_args, **_kwargs: io.BytesIO(b"x" * 101)
        )
        with pytest.raises(EntryLimitError, match="Decompressed entry exceeds"):
            images.read(frames(artifact)[1])
        assert_closed(handles)


def test_image_sha_is_checked_on_every_read(publish, monkeypatch, handles):
    root, artifact = publish()
    original = reader_module.read_entry

    def changed_digest(*args, **kwargs):
        content, _ = original(*args, **kwargs)
        return content, "0" * 64

    with VideoVariantImages(artifact, root) as images:
        monkeypatch.setattr(reader_module, "read_entry", changed_digest)
        with pytest.raises(ValueError, match="Image SHA256"):
            images.read(frames(artifact)[1])
        assert_closed(handles)


@pytest.mark.parametrize("claimed_valid_corrupt_bytes", [False, True])
def test_stored_qa_is_rechecked_against_actual_image(
    publish, claimed_valid_corrupt_bytes
):
    pairs = [
        (
            f"frame_{i}.png",
            b"corrupt" if i == 1 and claimed_valid_corrupt_bytes else png("red"),
        )
        for i in range(1, 5)
    ]
    root, artifact = publish({"all.zip": pairs})
    metadata = json.loads((artifact / "metadata.json").read_text())
    observed = {
        name: pd.read_parquet(artifact / f"{name}.parquet")
        for name in ("archives", "entries", "video_sources")
    }
    entries = observed["entries"]
    # A self-consistent synthetic publication can lie about pixel QA. Stored
    # replay alone does not decode originals; the consumer must do so per read.
    entries.loc[
        0,
        [
            "width",
            "height",
            "channels",
            "image_mode",
            "image_format",
            "image_decode_valid",
            "image_error",
        ],
    ] = [7, 4, 3, "RGB", "PNG", True, None]
    data, manifest, summary = derive_evidence(
        IngestionConfig.model_validate(metadata["identity"]["config"]),
        observed,
        metadata["identity"]["sources"],
    )
    for name, table in data.items():
        table.to_parquet(artifact / f"{name}.parquet", index=False)
    manifest.to_parquet(artifact / "manifest.parquet", index=False)
    variant = make_variant(
        summary["dataset_id"],
        sha256_file(artifact / "manifest.parquet"),
        "external_processed",
        definition=metadata["identity"]["config"]["transformation"],
    )
    (artifact / "variant.json").write_text(json.dumps(variant))
    summary["dataset_variant_id"] = variant["dataset_variant_id"]
    (artifact / "integrity.json").write_text(json.dumps(summary))
    artifact = reseal(artifact)
    assert inspect_ingestion(artifact)["summary"]["integrity_valid"]
    with (
        VideoVariantImages(artifact, root) as images,
        pytest.raises(ValueError, match="QA failed|properties differ"),
    ):
        images.read(frames(artifact)[1])


@pytest.mark.parametrize(
    "bad_path",
    [
        "../outside.zip",
        "/absolute.zip",
        "C:/outside.zip",
        "nested\\outside.zip",
        "first.zip:stream",
    ],
)
def test_unsafe_physical_declarations_are_rejected(publish, bad_path):
    root, artifact = publish()
    metadata = json.loads((artifact / "metadata.json").read_text())
    metadata["identity"]["config"]["archives"][0]["path"] = bad_path
    (artifact / "config.json").write_text(json.dumps(metadata["identity"]["config"]))
    (artifact / "metadata.json").write_text(json.dumps(metadata))
    artifact = reseal(artifact)
    with pytest.raises(ValueError, match="Unsafe relative POSIX path"):
        VideoVariantImages(artifact, root)


def test_resolved_archive_cannot_escape_root(publish, tmp_path, monkeypatch):
    root, artifact = publish()
    outside = tmp_path / "outside.zip"
    outside.write_bytes((root / "first.zip").read_bytes())
    actual_resolve = Path.resolve

    def resolve(path, *args, **kwargs):
        # Model a symlink/junction target without needing Windows link privileges.
        return (
            outside
            if path == root / "first.zip"
            else actual_resolve(path, *args, **kwargs)
        )

    images = VideoVariantImages(artifact, root)
    monkeypatch.setattr(Path, "resolve", resolve)
    with pytest.raises(ValueError, match="escapes root"):
        with images:
            pytest.fail("Escaping physical path was accepted")


@pytest.mark.parametrize("link_kind", ["file", "directory"])
def test_real_symlink_source_escape_is_rejected(publish, tmp_path, link_kind):
    root, artifact = publish(
        {"nested/all.zip": [(f"frame_{i}.png", png("red")) for i in range(1, 5)]}
    )
    relocated = tmp_path / "relocated"
    relocated.mkdir()
    try:
        if link_kind == "directory":
            (relocated / "nested").symlink_to(root / "nested", target_is_directory=True)
        else:
            (relocated / "nested").mkdir()
            (relocated / "nested/all.zip").symlink_to(root / "nested/all.zip")
    except OSError:
        pytest.skip("This host does not grant symlink creation")
    with pytest.raises(ValueError, match="escapes root"):
        with VideoVariantImages(artifact, relocated):
            pytest.fail("Escaping symlink was accepted")


def test_reopening_evicted_modified_source_fails_closed(publish, handles):
    root, artifact = publish()
    ids = frames(artifact)
    with VideoVariantImages(artifact, root, max_open_archives=1) as images:
        images.read(ids[1])
        # The second shard is evicted: a stale verified fingerprint must not
        # authorize its replacement when the pool reopens it.
        with (root / "second.zip").open("ab") as stream:
            stream.write(b"changed")
        with pytest.raises(ValueError, match="ZIP source"):
            images.read(ids[2])
        assert_closed(handles)


@pytest.mark.parametrize("limit", [1, 2])
def test_lru_bound_and_closure_on_success(publish, limit, handles):
    root, artifact = publish()
    ids = frames(artifact)
    images = VideoVariantImages(artifact, root, max_open_archives=limit)
    with images:
        for i in (1, 2, 3, 4, 1, 2):
            images.read(ids[i])
            assert sum(z.fp is not None for z in handles[0]) <= limit
            assert sum(not s.closed for s in handles[1]) <= limit
    assert_closed(handles)
    images.close()
    with pytest.raises(RuntimeError, match="closed"):
        with images:
            pytest.fail("Closed reader reopened")


@pytest.mark.parametrize("failure", ["enter", "read", "body"])
def test_handles_close_on_all_failure_paths(publish, failure, handles, monkeypatch):
    root, artifact = publish()
    images = VideoVariantImages(artifact, root)
    raw_streams = []
    original_open = Path.open

    def tracked_file(path, *args, **kwargs):
        stream = original_open(path, *args, **kwargs)
        if path.suffix == ".zip":
            raw_streams.append(stream)
        return stream

    if failure == "enter":
        path = root / "second.zip"
        data = bytearray(path.read_bytes())
        data[10] ^= 1
        path.write_bytes(data)
    monkeypatch.setattr(Path, "open", tracked_file)
    with pytest.raises((ValueError, KeyError), match="SHA256|unknown|body"):
        with images:
            if failure == "read":
                images.read("unknown")
            else:
                raise ValueError("body")
    assert_closed(handles)
    assert raw_streams and all(s.closed for s in raw_streams)


def test_reader_never_extracts_and_requires_context(publish, monkeypatch):
    root, artifact = publish()
    before = snapshot(root.parent)

    def forbidden(*args, **kwargs):
        raise AssertionError("Reader must not extract ZIP entries")

    monkeypatch.setattr(zipfile.ZipFile, "extract", forbidden)
    monkeypatch.setattr(zipfile.ZipFile, "extractall", forbidden)
    images = VideoVariantImages(artifact, root)
    with pytest.raises(RuntimeError, match="active reader context"):
        images.read(images.frame_ids[0])
    with images:
        for frame_id in images.frame_ids:
            images.read(frame_id)
    assert snapshot(root.parent) == before


def test_reentering_context_closes_existing_handles(publish, handles):
    root, artifact = publish()
    images = VideoVariantImages(artifact, root)
    with images:
        with pytest.raises(RuntimeError, match="already active"):
            images.__enter__()
        assert_closed(handles)


def test_only_requested_images_are_decoded(publish, monkeypatch):
    root, artifact = publish()
    original = reader_module.image_properties
    decoded = []

    def tracked(*args, **kwargs):
        decoded.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(reader_module, "image_properties", tracked)
    with VideoVariantImages(artifact, root) as images:
        assert not decoded
        frame_id = frames(artifact)[3]
        assert images.locate(frame_id).frame_id == frame_id
        assert not decoded
        images.read(frame_id)
        assert decoded == [png("blue")]


def test_close_attempts_all_resources_even_when_one_callback_fails(publish, handles):
    root, artifact = publish()
    images = VideoVariantImages(artifact, root)
    images.__enter__()

    def failed_close():
        raise OSError("synthetic close failure")

    # ExitStack must still close each ZIP and its stream when a callback raises.
    next(iter(images._open.values())).resources.callback(failed_close)
    with pytest.raises(OSError, match="synthetic close failure"):
        images.close()
    assert_closed(handles)


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_invalid_handle_budget_is_rejected(publish, limit):
    root, artifact = publish()
    with pytest.raises(ValueError, match="positive integer"):
        VideoVariantImages(artifact, root, max_open_archives=limit)


def test_import_is_independent_of_encoders():
    script = """
import builtins
original = builtins.__import__
def checked(name, *args, **kwargs):
    if name.startswith(('torch', 'transformers', 'flir_pipeline.features')):
        raise AssertionError('Reader must stay independent of encoders')
    return original(name, *args, **kwargs)
builtins.__import__ = checked
from flir_pipeline.data.video_variant_images import VideoVariantImages
"""
    subprocess.run(
        [sys.executable, "-B", "-c", script], check=True, capture_output=True, text=True
    )
