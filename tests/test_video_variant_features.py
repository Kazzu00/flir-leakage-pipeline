"""Offline storage integration with tiny synthetic ZIPs and a CPU fake encoder."""

import json
import shutil
import zipfile

import numpy as np
import pandas as pd
import pytest
import yaml
from test_video_variant_images import assert_closed, frames, png, snapshot
from test_video_variant_images import handles as handles
from test_video_variant_images import publish as publish

import flir_pipeline.data.video_variant_images as reader_module
import flir_pipeline.features.storage as storage
from flir_pipeline.data.identity import dataset_id_from_manifest
from flir_pipeline.data.variants import make_variant, read_variant
from flir_pipeline.data.video_variant_images import VideoVariantImages
from flir_pipeline.data.video_variant_storage import build_ingestion
from flir_pipeline.features.base import DeterministicFakeExtractor
from flir_pipeline.features.storage import (
    extract_to_store,
    feature_space_id,
    verify_feature_directory,
    verify_features_against_manifest,
)
from flir_pipeline.utils.hashing import sha256_file


class RecordingFake(DeterministicFakeExtractor):
    def __init__(self):
        super().__init__(6)
        self.calls = []
        self.interrupt = False

    def encode_batch(self, batch):
        self.calls.append(len(batch))
        if self.interrupt and len(self.calls) == 2:
            raise RuntimeError("synthetic interruption")
        return super().encode_batch(batch)


def run(root, artifact, output, extractor=None, **kwargs):
    return extract_to_store(
        artifact / "manifest.parquet",
        extractor=extractor or RecordingFake(),
        output_root=output,
        video_variant_ingestion=artifact,
        input_root=root,
        max_open_archives=1,
        **kwargs,
    )


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_interleaved_collection_preserves_science_and_every_physical_occurrence(
    publish, tmp_path, handles, monkeypatch
):
    root, artifact = publish()
    original = snapshot(root), snapshot(artifact)
    digest_calls = []
    actual_hash = reader_module.sha256_stream

    def tracked_hash(stream):
        digest_calls.append(1)
        return actual_hash(stream)

    def forbidden(*args, **kwargs):
        pytest.fail("ZIPs must never be extracted")

    monkeypatch.setattr(reader_module, "sha256_stream", tracked_hash)
    monkeypatch.setattr(zipfile.ZipFile, "extract", forbidden)
    monkeypatch.setattr(zipfile.ZipFile, "extractall", forbidden)
    encoder = RecordingFake()
    output = run(root, artifact, tmp_path / "features", encoder, batch_size=1)
    assert encoder.calls == [1, 1, 1]
    assert len(digest_calls) == 2  # LRU reopening/batches never rehash the ZIPs.
    assert_closed(handles)
    assert (snapshot(root), snapshot(artifact)) == original
    manifest = pd.read_parquet(artifact / "manifest.parquet")
    content = pd.read_parquet(output / "content_index.parquet")
    records = pd.read_parquet(output / "record_index.parquet")
    metadata = read_json(output / "metadata.json")
    assert metadata["dataset_id"] == dataset_id_from_manifest(manifest)
    assert metadata["dataset_variant"] == read_variant(artifact / "variant.json")
    assert metadata["feature_space_id"] == feature_space_id(
        encoder.feature_space_config()
    )
    binding = metadata["source_binding"]
    assert binding["schema_version"] == "video_variant_feature_source_v1"
    assert binding["artifact_id"] == artifact.name
    assert binding["metadata_sha256"] == sha256_file(artifact / "metadata.json")
    assert binding["receipt_sha256"] == sha256_file(artifact / "receipt.json")
    assert binding == metadata["cache_signature"]["source_binding"]
    assert str(root) not in json.dumps(metadata)
    assert content.content_id.tolist() == sorted(manifest.content_id.unique())
    assert len(content) == 3 and len(records) == 4
    assert records.groupby("content_id").embedding_row.nunique().eq(1).all()
    pd.testing.assert_frame_equal(records[list(manifest)], manifest)
    assert not records.label_exists.any()
    assert records.label_sha256.eq("").all() and records.original_split.eq("").all()
    assert not {"timestamp_seconds", "sequence_id", "cluster_id", "split_id"} & set(
        records
    )
    direct = VideoVariantImages(artifact, root)
    for row in records.itertuples(index=False):
        location = direct.locate(row.frame_id)
        for column in storage._PHYSICAL_COLUMNS:
            assert getattr(row, column) == getattr(location, column)
    representative = manifest.groupby("content_id").frame_id.min()
    assert (
        content.set_index("content_id").representative_frame_id.to_dict()
        == representative.to_dict()
    )
    red = records.loc[records.frame_id.isin([frames(artifact)[1], frames(artifact)[2]])]
    assert red.content_id.nunique() == 1 and red.archive_key.nunique() == 2
    assert red.entry_id.is_unique and red.embedding_row.nunique() == 1
    assert verify_feature_directory(output)["source_binding_valid"]
    assert verify_features_against_manifest(output, manifest)["full_dataset_valid"]
    assert not verify_features_against_manifest(output, manifest)[
        "reproducible_full_dataset_valid"
    ]


def test_equal_bytes_equal_raw_l2_spaces_and_scientific_ids_across_transports(
    publish, tmp_path
):
    root, artifact = publish()
    manifest = pd.read_parquet(artifact / "manifest.parquet")
    materialized = tmp_path / "reference-images"
    materialized.mkdir()
    archive = tmp_path / "reference.zip"
    expected = {1: png("red"), 2: png("red"), 3: png("blue"), 4: png("green")}
    with zipfile.ZipFile(archive, "w") as zipped:
        for index, frame_id in frames(artifact).items():
            (materialized / f"{frame_id}.png").write_bytes(expected[index])
            zipped.writestr(f"{frame_id}.png", expected[index])
    reference = manifest.assign(
        image_path=manifest.frame_id + ".png",
        source_member_path=manifest.frame_id + ".png",
        source_archive="reference.zip",
    )
    path = tmp_path / "reference.parquet"
    reference.to_parquet(path, index=False)
    encoder = RecordingFake()
    stores = [
        run(root, artifact, tmp_path / "multishard", encoder),
        extract_to_store(
            path,
            extractor=encoder,
            output_root=tmp_path / "local",
            images_root=materialized,
        ),
        extract_to_store(path, archive, encoder, tmp_path / "zip"),
    ]
    for other in stores[1:]:
        for name in ("embeddings_raw.npy", "embeddings_l2.npy"):
            a, b = np.load(stores[0] / name), np.load(other / name)
            np.testing.assert_array_equal(a, b)
            assert a.dtype == b.dtype == np.float32
        first, legacy = (
            read_json(stores[0] / "metadata.json"),
            read_json(other / "metadata.json"),
        )
        assert first["dataset_id"] == legacy["dataset_id"]
        assert first["feature_space_id"] == legacy["feature_space_id"]
        assert "source_binding" not in legacy and "dataset_variant" not in legacy
        pd.testing.assert_frame_equal(
            pd.read_parquet(stores[0] / "record_index.parquet")[
                ["frame_id", "content_id", "embedding_row"]
            ],
            pd.read_parquet(other / "record_index.parquet"),
        )
        before = snapshot(other)
        kwargs = (
            {"images_root": materialized}
            if other == stores[1]
            else {"images_archive": archive}
        )
        assert (
            extract_to_store(
                path, extractor=encoder, output_root=other.parents[2], **kwargs
            )
            == other
        )
        assert snapshot(other) == before


def test_compatible_resume_keeps_checkpoint_binding_and_closes_resources(
    publish, tmp_path, handles
):
    root, artifact = publish()
    encoder = RecordingFake()
    encoder.interrupt = True
    output = tmp_path / "resume"
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        run(root, artifact, output, encoder, batch_size=1)
    assert_closed(handles)
    checkpoint = next(output.rglob("checkpoint.json"))
    saved = read_json(checkpoint)
    assert saved["completed"] == 1
    assert saved["signature"]["source_binding"]["artifact_id"] == artifact.name
    assert not (checkpoint.parent.parent / "metadata.json").exists()
    encoder.calls = []
    encoder.interrupt = False
    resumed = run(root, artifact, output, encoder, batch_size=1)
    assert encoder.calls == [1, 1]
    assert_closed(handles)
    reference = run(root, artifact, tmp_path / "fresh", batch_size=3)
    for name in ("embeddings_raw.npy", "embeddings_l2.npy"):
        np.testing.assert_array_equal(
            np.load(resumed / name), np.load(reference / name)
        )
    assert not checkpoint.exists()
    assert verify_feature_directory(resumed)["quality_valid"]


@pytest.mark.parametrize("state", ["partial", "final"])
def test_repackaging_preserves_science_but_cannot_rebind_existing_store(
    publish, tmp_path, state
):
    root, first = publish()
    other, repackaged = publish(
        {
            "repacked.zip": [
                ("frame_4.png", png("green")),
                ("frame_2.png", png("red")),
                ("frame_1.png", png("red")),
                ("frame_3.png", png("blue")),
            ]
        },
        compression=zipfile.ZIP_STORED,
    )
    encoder = RecordingFake()
    encoder.interrupt = state == "partial"
    output = tmp_path / "bound"
    if state == "partial":
        with pytest.raises(RuntimeError, match="interruption"):
            run(root, first, output, encoder, batch_size=1)
    else:
        run(root, first, output, encoder)
    before = snapshot(output)
    encoder.interrupt = False
    with pytest.raises(RuntimeError, match="does not match"):
        run(other, repackaged, output, encoder)
    assert snapshot(output) == before
    a, b = (
        pd.read_parquet(first / "manifest.parquet"),
        pd.read_parquet(repackaged / "manifest.parquet"),
    )
    pd.testing.assert_frame_equal(a, b)
    original_space = feature_space_id(encoder.feature_space_config())
    independent = run(other, repackaged, tmp_path / "independent", encoder)
    assert independent.name == original_space
    assert (
        read_json(independent / "metadata.json")["dataset_variant"][
            "dataset_variant_id"
        ]
        == read_variant(first / "variant.json")["dataset_variant_id"]
    )


def test_relocation_is_compatible_and_final_reuse_is_read_only(
    publish, tmp_path, handles
):
    root, artifact = publish()
    encoder = RecordingFake()
    output = tmp_path / "features"
    store = run(root, artifact, output, encoder)
    before = snapshot(store)
    alternate = tmp_path / "relocated"
    shutil.copytree(root, alternate)
    encoder.calls = []
    assert run(alternate, artifact, output, encoder, batch_size=1) == store
    assert encoder.calls == []
    assert snapshot(store) == before
    assert_closed(handles)


def test_smoke_keeps_every_occurrence_with_explicit_unsampled_rows(publish, tmp_path):
    root, artifact = publish()
    first = run(root, artifact, tmp_path / "one", limit_content=1, seed=19)
    second = run(root, artifact, tmp_path / "two", limit_content=1, seed=19)
    records = pd.read_parquet(first / "record_index.parquet")
    assert len(records) == 4 and records.embedding_row.eq(-1).any()
    assert records.groupby("content_id").embedding_row.nunique().eq(1).all()
    assert len(pd.read_parquet(first / "content_index.parquet")) == 1
    assert verify_feature_directory(first)["quality_valid"]
    assert not verify_features_against_manifest(
        first, pd.read_parquet(artifact / "manifest.parquet")
    )["full_dataset_valid"]
    np.testing.assert_array_equal(
        np.load(first / "embeddings_raw.npy"), np.load(second / "embeddings_raw.npy")
    )
    assert pd.read_parquet(first / "content_index.parquet").equals(
        pd.read_parquet(second / "content_index.parquet")
    )
    before = snapshot(first)
    with pytest.raises(RuntimeError, match="does not match"):
        run(root, artifact, tmp_path / "one")
    assert snapshot(first) == before


@pytest.mark.parametrize(
    "failure",
    [
        "manifest",
        "receipt",
        "artifact",
        "zip",
        "wrong_manifest",
        "dataset_override",
        "variant",
    ],
)
def test_incompatible_publication_inputs_fail_without_encoding(
    publish, tmp_path, failure, handles
):
    root, artifact = publish()
    encoder = RecordingFake()
    output = tmp_path / "features"
    kwargs = {}
    if failure in {"manifest", "receipt", "artifact"}:
        name = {
            "manifest": "manifest.parquet",
            "receipt": "receipt.json",
            "artifact": "metadata.json",
        }[failure]
        with (artifact / name).open("ab") as stream:
            stream.write(b"modified")
    elif failure == "zip":
        content = bytearray((root / "second.zip").read_bytes())
        content[10] ^= 1
        (root / "second.zip").write_bytes(content)
    elif failure == "dataset_override":
        kwargs["dataset_id"] = "0" * 64
    elif failure == "variant":
        declaration = make_variant(
            dataset_id_from_manifest(pd.read_parquet(artifact / "manifest.parquet")),
            sha256_file(artifact / "manifest.parquet"),
            "incompatible",
        )
        path = tmp_path / "different-variant.json"
        path.write_text(json.dumps(declaration), encoding="utf-8")
        kwargs["variant_spec"] = path
    if failure == "wrong_manifest":
        wrong = tmp_path / "copy.parquet"
        shutil.copyfile(artifact / "manifest.parquet", wrong)
        with pytest.raises(ValueError, match="original manifest"):
            extract_to_store(
                wrong,
                extractor=encoder,
                output_root=output,
                video_variant_ingestion=artifact,
                input_root=root,
            )
    else:
        with pytest.raises((ValueError, OSError)):
            run(root, artifact, output, encoder, **kwargs)
    assert encoder.calls == []
    assert_closed(handles)
    assert not output.exists()


def test_modified_source_rejects_resume_and_preserves_partial(
    publish, tmp_path, handles
):
    root, artifact = publish()
    encoder = RecordingFake()
    encoder.interrupt = True
    output = tmp_path / "features"
    with pytest.raises(RuntimeError, match="interruption"):
        run(root, artifact, output, encoder, batch_size=1)
    before = snapshot(output)
    with (root / "second.zip").open("ab") as stream:
        stream.write(b"modified")
    encoder.interrupt = False
    encoder.calls = []
    with pytest.raises(ValueError, match="ZIP source size"):
        run(root, artifact, output, encoder)
    assert encoder.calls == [] and snapshot(output) == before
    assert_closed(handles)


@pytest.mark.parametrize(
    "failure", ["checkpoint_write", "index_write", "metadata_write"]
)
def test_write_failures_close_resources_and_do_not_complete(
    publish, tmp_path, monkeypatch, handles, failure
):
    root, artifact = publish()
    output = tmp_path / "features"
    original = storage._atomic_json
    original_parquet = pd.DataFrame.to_parquet
    original_memmap = np.lib.format.open_memmap
    memory_handles = []

    def tracked_memmap(*args, **kwargs):
        values = original_memmap(*args, **kwargs)
        memory_handles.append(values._mmap)
        return values

    def json_write(path, value):
        if (
            failure == "checkpoint_write"
            and path.name == "checkpoint.json"
            and value["completed"] > 0
        ):
            raise OSError("synthetic write failure")
        if failure == "metadata_write" and path.name == "metadata.json":
            raise OSError("synthetic write failure")
        return original(path, value)

    def parquet_write(frame, path, *args, **kwargs):
        if failure == "index_write":
            raise OSError("synthetic write failure")
        return original_parquet(frame, path, *args, **kwargs)

    monkeypatch.setattr(storage, "_atomic_json", json_write)
    monkeypatch.setattr(pd.DataFrame, "to_parquet", parquet_write)
    monkeypatch.setattr(np.lib.format, "open_memmap", tracked_memmap)
    with pytest.raises(OSError, match="synthetic write failure"):
        run(root, artifact, output, batch_size=1)
    assert_closed(handles)
    assert memory_handles and all(handle.closed for handle in memory_handles)
    assert not list(output.rglob("metadata.json"))
    assert list(output.rglob("checkpoint.json"))


@pytest.mark.parametrize(
    "failure",
    [
        "binding",
        "binding_missing",
        "binding_version",
        "binding_marker",
        "record_locator",
        "content_locator",
        "mapping",
        "dtype",
        "dimension",
        "nan",
        "inf",
        "zero",
        "l2",
        "coverage",
    ],
)
def test_completed_store_verification_rejects_inconsistent_science_or_provenance(
    publish, tmp_path, failure
):
    root, artifact = publish()
    store = run(root, artifact, tmp_path / "features")
    metadata_path = store / "metadata.json"
    metadata = read_json(metadata_path)
    if failure == "binding":
        metadata["source_binding"]["dataset_id"] = "0" * 64
    elif failure in {"binding_missing", "binding_marker"}:
        del metadata["source_binding"]
        if failure == "binding_marker":
            metadata["image_source_type"] = "zip"
            del metadata["source_index_checksums"]
    elif failure == "binding_version":
        metadata["source_binding"]["schema_version"] = "unsupported_v2"
    elif failure in {"record_locator", "content_locator", "mapping", "coverage"}:
        name = (
            "content_index.parquet"
            if failure == "content_locator"
            else "record_index.parquet"
        )
        index = pd.read_parquet(store / name)
        if failure in {"record_locator", "content_locator"}:
            index.loc[0, "member_ordinal"] += 1
        elif failure == "mapping":
            index.loc[0, "embedding_row"] = 99
        else:
            index = index.iloc[1:]
        index.to_parquet(store / name, index=False)
    elif failure == "dimension":
        metadata["embedding_dimension"] += 1
    else:
        name = "embeddings_l2.npy" if failure == "l2" else "embeddings_raw.npy"
        values = np.load(store / name)
        if failure == "dtype":
            values = values.astype(np.float64)
        elif failure == "nan":
            values[0, 0] = np.nan
        elif failure == "inf":
            values[0, 0] = np.inf
        elif failure == "zero":
            values[0] = 0
        else:
            values[0] *= 0.5
        np.save(store / name, values)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    assert not verify_feature_directory(store)["quality_valid"]
    before = snapshot(store)
    with pytest.raises(RuntimeError, match="cache"):
        run(root, artifact, tmp_path / "features")
    assert snapshot(store) == before


@pytest.mark.parametrize(
    "failure", ["binding", "shape", "dtype", "completed", "dimension", "unpaired"]
)
def test_invalid_checkpoint_is_preserved_without_restarting(
    publish, tmp_path, handles, failure
):
    root, artifact = publish()
    encoder = RecordingFake()
    encoder.interrupt = True
    output = tmp_path / "features"
    with pytest.raises(RuntimeError, match="interruption"):
        run(root, artifact, output, encoder, batch_size=1)
    checkpoint = next(output.rglob("checkpoint.json"))
    saved = read_json(checkpoint)
    if failure == "binding":
        saved["signature"]["source_binding"]["receipt_sha256"] = "0" * 64
    elif failure == "completed":
        saved["completed"] = 99
    elif failure == "dimension":
        saved["embedding_dimension"] += 1
    elif failure == "unpaired":
        saved["completed"] = 0
        (checkpoint.parent / "embeddings_l2.npy").unlink()
    else:
        path = checkpoint.parent / "embeddings_raw.npy"
        values = np.load(path)
        values = values[:-1] if failure == "shape" else values.astype(np.float64)
        np.save(path, values)
    checkpoint.write_text(json.dumps(saved), encoding="utf-8")
    before = snapshot(output)
    encoder.calls = []
    encoder.interrupt = False
    with pytest.raises(RuntimeError, match="Partial"):
        run(root, artifact, output, encoder, batch_size=1)
    assert encoder.calls == [] and snapshot(output) == before
    assert_closed(handles)


@pytest.mark.parametrize("failure", ["shape", "nan", "inf", "zero"])
def test_invalid_encoder_output_cannot_complete_a_store(
    publish, tmp_path, handles, failure
):
    root, artifact = publish()

    class InvalidFake(RecordingFake):
        def encode_batch(self, batch):
            values = super().encode_batch(batch)
            if failure == "shape":
                return values[:, :-1]
            values[0] = {"nan": np.nan, "inf": np.inf, "zero": 0.0}[failure]
            return values

    output = tmp_path / "features"
    with np.errstate(invalid="ignore"), pytest.raises((ValueError, RuntimeError)):
        run(root, artifact, output, InvalidFake())
    assert not list(output.rglob("metadata.json"))
    assert list(output.rglob("checkpoint.json"))
    assert_closed(handles)


def test_same_variant_legacy_store_cannot_be_retroactively_bound(publish, tmp_path):
    root, artifact = publish()
    manifest = pd.read_parquet(artifact / "manifest.parquet")
    images = tmp_path / "legacy-images"
    images.mkdir()
    pixels = {1: png("red"), 2: png("red"), 3: png("blue"), 4: png("green")}
    for index, frame_id in frames(artifact).items():
        (images / f"{frame_id}.png").write_bytes(pixels[index])
    path = tmp_path / "legacy.parquet"
    manifest.assign(image_path=manifest.frame_id + ".png").to_parquet(path, index=False)
    variant = read_variant(artifact / "variant.json")
    variant["source_checksums"]["manifest_sha256"] = sha256_file(path)
    declaration = tmp_path / "legacy-variant.json"
    declaration.write_text(json.dumps(variant), encoding="utf-8")
    output = tmp_path / "features"
    store = extract_to_store(
        path,
        extractor=RecordingFake(),
        output_root=output,
        images_root=images,
        variant_spec=declaration,
    )
    assert "source_binding" not in read_json(store / "metadata.json")
    assert (
        variant["dataset_variant_id"]
        == read_variant(artifact / "variant.json")["dataset_variant_id"]
    )
    before = snapshot(store)
    with pytest.raises(RuntimeError, match="does not match"):
        run(root, artifact, output)
    assert snapshot(store) == before
    assert verify_feature_directory(store)["quality_valid"]


def test_store_verification_releases_array_handles_even_on_error(
    publish, tmp_path, monkeypatch
):
    root, artifact = publish()
    store = run(root, artifact, tmp_path / "features")
    memory_handles = []
    actual = np.lib.format.open_memmap

    def tracked_memmap(*args, **kwargs):
        values = actual(*args, **kwargs)
        memory_handles.append(values._mmap)
        return values

    monkeypatch.setattr(np.lib.format, "open_memmap", tracked_memmap)
    assert verify_feature_directory(store)["quality_valid"]
    assert memory_handles and all(handle.closed for handle in memory_handles)
    # Only this synthetic store is damaged; the failed verifier must still close.
    (store / "record_index.parquet").unlink()
    with pytest.raises(FileNotFoundError):
        verify_feature_directory(store)
    assert all(handle.closed for handle in memory_handles)


def test_unbound_orphan_files_are_preserved(publish, tmp_path):
    root, artifact = publish()
    store = run(root, artifact, tmp_path / "features")
    (store / "metadata.json").unlink()  # Synthetic failed-finalization scenario.
    before = snapshot(store)
    with pytest.raises(RuntimeError, match="Unbound partial"):
        run(root, artifact, tmp_path / "features")
    assert snapshot(store) == before


def test_non_zip_original_fingerprint_is_verified_once_and_cannot_change(
    publish, tmp_path
):
    root, initial = publish()
    declaration_path = root.parent / "config.yaml"
    declaration = yaml.safe_load(declaration_path.read_text(encoding="utf-8"))
    declaration["auxiliary_inventory"] = {"path": "inventory.csv"}
    declaration_path.write_text(yaml.safe_dump(declaration), encoding="utf-8")
    (root / "inventory.csv").write_text("synthetic inventory", encoding="utf-8")
    artifact = build_ingestion(
        declaration_path, root, initial.parent.parent / "with-aux"
    )
    store = run(root, artifact, tmp_path / "features")
    before = snapshot(store)
    (root / "inventory.csv").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="Non-ZIP ingestion source"):
        run(root, artifact, tmp_path / "features")
    assert snapshot(store) == before


@pytest.mark.parametrize(
    "failure",
    [
        "missing_root",
        "images_root",
        "images_archive",
        "root_without_publication",
        "output_in_publication",
        "output_in_inputs",
    ],
)
def test_explicit_source_selection_and_read_only_output_boundaries(
    publish, tmp_path, failure
):
    root, artifact = publish()
    kwargs = dict(
        extractor=RecordingFake(),
        output_root=tmp_path / "features",
        video_variant_ingestion=artifact,
        input_root=root,
    )
    if failure == "missing_root":
        del kwargs["input_root"]
    elif failure in {"images_root", "images_archive"}:
        kwargs[failure] = root if failure == "images_root" else root / "first.zip"
    elif failure == "root_without_publication":
        del kwargs["video_variant_ingestion"]
    else:
        kwargs["output_root"] = (
            artifact / "features"
            if failure == "output_in_publication"
            else root / "features"
        )
    before = snapshot(root), snapshot(artifact)
    with pytest.raises(ValueError):
        extract_to_store(artifact / "manifest.parquet", **kwargs)
    assert (snapshot(root), snapshot(artifact)) == before
