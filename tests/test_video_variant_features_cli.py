"""Offline CLI bridge: verified synthetic shards and injected model constructors."""

import builtins
import json
import socket
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from test_video_variant_images import assert_closed, frames, png, snapshot
from test_video_variant_images import handles as handles
from test_video_variant_images import publish as publish
from typer.testing import CliRunner

import flir_pipeline.cli as cli
import flir_pipeline.data.video_variant_images as reader_module
from flir_pipeline.data.identity import dataset_id_from_manifest
from flir_pipeline.data.variants import make_variant, read_variant
from flir_pipeline.features.base import DeterministicFakeExtractor
from flir_pipeline.features.storage import extract_to_store, verify_feature_directory
from flir_pipeline.utils.hashing import sha256_file

REPOSITORY = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def forbid_models_and_network(monkeypatch):
    actual_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "transformers", "huggingface_hub"}:
            pytest.fail(f"Real model dependency imported: {name}")
        return actual_import(name, *args, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("Network and ZIP extraction are forbidden")

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(zipfile.ZipFile, "extract", forbidden)
    monkeypatch.setattr(zipfile.ZipFile, "extractall", forbidden)


class SimulatedExtractor(DeterministicFakeExtractor):
    """Keep simulation explicit in metadata, including unresolved fake revision."""

    def __init__(self, name, state=None):
        super().__init__({"dinov2": 384, "clip": 512}[name])
        self.name = name
        self.model_id = f"test-only-simulated-{name}"
        self.calls = []
        self.state = state or {}

    def encode_batch(self, batch):
        self.calls.append(len(batch))
        if self.state.get("interrupt") and len(self.calls) == 2:
            raise RuntimeError("synthetic CLI interruption")
        return super().encode_batch(batch)


@pytest.fixture
def simulated_models(monkeypatch):
    from flir_pipeline.features import clip, dinov2

    state = {"constructed": [], "instances": [], "interrupt": False}

    def constructor(name):
        # Match the existing constructor parameters: unknown config keys still
        # fail before its body, instead of being swallowed by a permissive mock.
        def build(
            model_id="simulation",
            device="auto",
            batch_size=8,
            mixed_precision=False,
            local_files_only=False,
            model_revision=None,
            require_resolved_revision=False,
        ):
            state["constructed"].append(
                {
                    "name": name,
                    "model_id": model_id,
                    "device": device,
                    "batch_size": batch_size,
                    "local_files_only": local_files_only,
                    "model_revision": model_revision,
                    "require_resolved_revision": require_resolved_revision,
                    "mixed_precision": mixed_precision,
                }
            )
            if state.get("constructor_failure"):
                raise RuntimeError("synthetic model initialization failure")
            result = SimulatedExtractor(name, state)
            state["instances"].append(result)
            return result

        return build

    monkeypatch.setattr(dinov2, "DinoV2Extractor", constructor("dinov2"))
    monkeypatch.setattr(clip, "CLIPExtractor", constructor("clip"))
    return state


def arguments(root, publication, output):
    return [
        "features",
        "extract",
        "--manifest",
        str(publication / "manifest.parquet"),
        "--video-variant-ingestion",
        str(publication),
        "--input-root",
        str(root),
        "--max-open-archives",
        "1",
        "--output-root",
        str(output),
        "--local-files-only",
    ]


def replace_option(args, name, value):
    position = args.index(name)
    del args[position : position + 2]
    if value is not None:
        args.extend([name, str(value)])


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def store_path(output):
    return next(output.rglob("metadata.json")).parent


@pytest.mark.parametrize("name", ["dinov2", "clip"])
def test_configured_model_cli_matches_programmatic_engine_and_preserves_inputs(
    publish, tmp_path, monkeypatch, handles, simulated_models, name
):
    root, publication = publish()
    original = snapshot(root), snapshot(publication)
    output = tmp_path / "cli-features"
    hashed = []
    actual_hash = reader_module.sha256_stream

    def tracked_hash(stream):
        assert not simulated_models["constructed"]
        assert not output.exists()  # Source verification precedes models/outputs.
        hashed.append(stream.name)
        return actual_hash(stream)

    def no_fallback():
        pytest.fail("Explicit ingestion must not consult the historical default")

    monkeypatch.setattr(cli, "_default_root", no_fallback)
    monkeypatch.setattr(reader_module, "sha256_stream", tracked_hash)
    config = REPOSITORY / "configs" / "embeddings" / f"{name}_full.yaml"
    args = arguments(root, publication, output) + [
        "--config",
        str(config),
        "--batch-size",
        "1",
    ]
    result = CliRunner().invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    assert len(hashed) == 2  # One reader session, despite pre-model validation.
    assert len(simulated_models["constructed"]) == 1
    call = simulated_models["constructed"][0]
    assert call["name"] == name and call["local_files_only"] is True
    assert call["batch_size"] == 1 and call["device"] == "cpu"
    assert (
        call["model_revision"] == yaml.safe_load(config.read_text())["model_revision"]
    )
    assert simulated_models["instances"][0].calls == [1, 1, 1]
    assert_closed(handles)
    monkeypatch.setattr(reader_module, "sha256_stream", actual_hash)
    actual = store_path(output)
    reference = extract_to_store(
        publication / "manifest.parquet",
        extractor=SimulatedExtractor(name),
        output_root=tmp_path / "reference",
        video_variant_ingestion=publication,
        input_root=root,
        max_open_archives=1,
        batch_size=1,
    )
    for filename in ("embeddings_raw.npy", "embeddings_l2.npy"):
        np.testing.assert_array_equal(
            np.load(actual / filename), np.load(reference / filename)
        )
    for filename in ("content_index.parquet", "record_index.parquet"):
        pd.testing.assert_frame_equal(
            pd.read_parquet(actual / filename), pd.read_parquet(reference / filename)
        )
    metadata = read_json(actual / "metadata.json")
    expected = read_json(reference / "metadata.json")
    for key in (
        "dataset_id",
        "dataset_variant",
        "feature_space_id",
        "source_binding",
        "cache_signature",
    ):
        assert metadata[key] == expected[key]
    manifest = pd.read_parquet(publication / "manifest.parquet")
    records = pd.read_parquet(actual / "record_index.parquet")
    pd.testing.assert_frame_equal(records[list(manifest)], manifest)
    assert (
        len(records) == 4
        and records.groupby("content_id").embedding_row.nunique().eq(1).all()
    )
    assert records.entry_id.is_unique and records.archive_key.nunique() == 2
    assert not records.label_exists.any() and records.original_split.eq("").all()
    assert not {"timestamp_seconds", "sequence_id", "cluster_id", "split_id"} & set(
        records
    )
    assert metadata["dataset_id"] == dataset_id_from_manifest(manifest)
    assert metadata["dataset_variant"] == read_variant(publication / "variant.json")
    assert metadata["source_binding"]["artifact_id"] == publication.name
    assert verify_feature_directory(actual)["source_binding_valid"]
    assert (snapshot(root), snapshot(publication)) == original


def test_cli_help_documents_new_options_without_loading_models(simulated_models):
    result = CliRunner().invoke(cli.app, ["features", "extract", "--help"], color=False)
    assert result.exit_code == 0
    for name in ("video-variant-ingestion", "input-root", "max-open-archives"):
        assert name in result.output
    assert not simulated_models["constructed"]


@pytest.mark.parametrize(
    "failure",
    [
        "images_archive",
        "images_root",
        "both_historical",
        "missing_input",
        "input_without_publication",
        "limit_without_publication",
        "zero_limit",
        "negative_limit",
        "missing_publication",
        "publication_file",
        "missing_root",
        "missing_manifest",
        "output_in_publication",
        "output_in_sources",
    ],
)
def test_invalid_source_arguments_fail_before_model_and_outputs(
    publish, tmp_path, simulated_models, handles, failure
):
    root, publication = publish()
    output = tmp_path / "features"
    args = arguments(root, publication, output)
    if failure in {"images_archive", "both_historical"}:
        args += ["--images-archive", str(root / "first.zip")]
    if failure in {"images_root", "both_historical"}:
        args += ["--images-root", str(root)]
    if failure == "missing_input":
        replace_option(args, "--input-root", None)
    elif failure in {"input_without_publication", "limit_without_publication"}:
        replace_option(args, "--video-variant-ingestion", None)
        replace_option(
            args,
            "--max-open-archives"
            if failure == "input_without_publication"
            else "--input-root",
            None,
        )
    elif failure in {"zero_limit", "negative_limit"}:
        replace_option(
            args, "--max-open-archives", 0 if failure == "zero_limit" else -1
        )
    elif failure in {"missing_publication", "publication_file"}:
        replace_option(
            args,
            "--video-variant-ingestion",
            tmp_path / "missing"
            if failure == "missing_publication"
            else publication / "metadata.json",
        )
    elif failure == "missing_root":
        replace_option(args, "--input-root", tmp_path / "missing")
    elif failure == "missing_manifest":
        replace_option(args, "--manifest", tmp_path / "missing.parquet")
    elif failure in {"output_in_publication", "output_in_sources"}:
        replace_option(
            args,
            "--output-root",
            (publication if failure == "output_in_publication" else root) / "features",
        )
    before = snapshot(root), snapshot(publication)
    result = CliRunner().invoke(cli.app, args)
    assert result.exit_code in {1, 2}, result.output
    assert "Error" in result.output or "failed" in result.output
    assert not simulated_models["constructed"] and not output.exists()
    assert (snapshot(root), snapshot(publication)) == before
    assert_closed(handles)


@pytest.mark.parametrize(
    "failure",
    [
        "wrong_manifest",
        "manifest_checksum",
        "variant",
        "receipt",
        "metadata",
        "missing_zip",
        "changed_zip",
        "corrupt_zip",
        "incompatible_kind",
    ],
)
def test_incompatible_scientific_or_physical_sources_fail_before_models(
    publish, tmp_path, simulated_models, handles, failure
):
    root, publication = publish()
    output = tmp_path / "features"
    args = arguments(root, publication, output)
    if failure == "wrong_manifest":
        path = tmp_path / "copy.parquet"
        path.write_bytes((publication / "manifest.parquet").read_bytes())
        replace_option(args, "--manifest", path)
    elif failure == "variant":
        manifest = pd.read_parquet(publication / "manifest.parquet")
        variant = make_variant(
            dataset_id_from_manifest(manifest),
            sha256_file(publication / "manifest.parquet"),
            "different-variant",
        )
        path = tmp_path / "variant.json"
        path.write_text(json.dumps(variant), encoding="utf-8")
        args += ["--variant-spec", str(path)]
    elif failure in {"receipt", "metadata", "manifest_checksum"}:
        path = (
            publication
            / {
                "receipt": "receipt.json",
                "metadata": "metadata.json",
                "manifest_checksum": "manifest.parquet",
            }[failure]
        )
        with path.open("ab") as stream:
            stream.write(b"synthetic modification")
    elif failure == "missing_zip":
        (root / "second.zip").unlink()
    elif failure == "changed_zip":
        with (root / "second.zip").open("ab") as stream:
            stream.write(b"synthetic modification")
    elif failure == "corrupt_zip":
        data = bytearray((root / "second.zip").read_bytes())
        data[10] ^= 1
        (root / "second.zip").write_bytes(data)
    else:
        metadata = read_json(publication / "metadata.json")
        metadata["artifact_kind"] = "unsupported_ingestion_v2"
        (publication / "metadata.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
        (publication / "receipt.json").write_text(
            json.dumps({"metadata_sha256": sha256_file(publication / "metadata.json")}),
            encoding="utf-8",
        )
    before = snapshot(root), snapshot(publication)
    result = CliRunner().invoke(cli.app, args)
    assert result.exit_code == 1 and "Feature extraction failed:" in result.output
    assert not simulated_models["constructed"] and not output.exists()
    assert (snapshot(root), snapshot(publication)) == before
    assert_closed(handles)


@pytest.mark.parametrize(
    "settings",
    [
        {"batch_size": 0},
        {"batch_size": -1},
        {"batch_size": True},
        {"batch_size": 1.5},
        {"batch_size": "eight"},
        {"dtype": "float16"},
        {"store_raw": False},
        {"store_l2_normalized": False},
        {"pooling_strategy": "mean"},
        {"feature_type": "text_embedding"},
        {"device": "unknown"},
        {"extractor": "unknown"},
        {"unknown_model_option": True},
        [],
        False,
        0,
        "not-a-mapping",
        "bad-yaml",
        "missing-config",
    ],
)
def test_incompatible_config_fails_before_constructor_and_outputs(
    publish, tmp_path, simulated_models, handles, settings
):
    root, publication = publish()
    output = tmp_path / "features"
    config = tmp_path / "encoder.yaml"
    if settings != "missing-config":
        content = (
            "unterminated: [" if settings == "bad-yaml" else yaml.safe_dump(settings)
        )
        config.write_text(content, encoding="utf-8")
    result = CliRunner().invoke(
        cli.app, arguments(root, publication, output) + ["--config", str(config)]
    )
    assert result.exit_code in {1, 2}, result.output
    assert "Error" in result.output or "failed" in result.output
    assert not simulated_models["constructed"] and not output.exists()
    assert_closed(handles)


def test_final_reuse_and_changed_source_rejection_preserve_store(
    publish, tmp_path, simulated_models, handles
):
    root, publication = publish()
    output = tmp_path / "features"
    args = arguments(root, publication, output)
    runner = CliRunner()
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    store = store_path(output)
    before = snapshot(store)
    result = runner.invoke(cli.app, args + ["--batch-size", "1"])
    assert result.exit_code == 0, result.output
    assert simulated_models["instances"][-1].calls == [] and snapshot(store) == before
    calls = len(simulated_models["constructed"])
    with (root / "second.zip").open("ab") as stream:
        stream.write(b"synthetic mutation")
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 1 and "ZIP source" in result.output
    assert len(simulated_models["constructed"]) == calls
    assert snapshot(store) == before
    assert_closed(handles)


def test_interrupted_cli_resumes_same_checkpoint_and_keeps_occurrences(
    publish, tmp_path, simulated_models, handles
):
    root, publication = publish()
    output = tmp_path / "features"
    args = arguments(root, publication, output) + ["--batch-size", "1"]
    runner = CliRunner()
    simulated_models["interrupt"] = True
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 1 and "synthetic CLI interruption" in result.output
    checkpoint = next(output.rglob("checkpoint.json"))
    saved = read_json(checkpoint)
    assert saved["completed"] == 1
    assert saved["signature"]["source_binding"]["artifact_id"] == publication.name
    assert_closed(handles)
    simulated_models["interrupt"] = False
    before = snapshot(output)
    original_zip = (root / "second.zip").read_bytes()
    (root / "second.zip").write_bytes(original_zip + b"synthetic mutation")
    calls = len(simulated_models["constructed"])
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 1 and "ZIP source" in result.output
    assert len(simulated_models["constructed"]) == calls and snapshot(output) == before
    assert_closed(handles)
    (root / "second.zip").write_bytes(original_zip)
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    assert simulated_models["instances"][-1].calls == [1, 1]
    assert not checkpoint.exists()
    records = pd.read_parquet(store_path(output) / "record_index.parquet")
    assert len(records) == 4 and records.embedding_row.ge(0).all()
    assert records.groupby("content_id").embedding_row.nunique().eq(1).all()
    assert_closed(handles)


def test_matching_explicit_variant_and_default_archive_limit_are_supported(
    publish, tmp_path, simulated_models
):
    root, publication = publish()
    output = tmp_path / "features"
    args = arguments(root, publication, output)
    replace_option(args, "--max-open-archives", None)
    args += ["--variant-spec", str(publication / "variant.json")]
    result = CliRunner().invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    assert read_json(store_path(output) / "metadata.json")[
        "dataset_variant"
    ] == read_variant(publication / "variant.json")


def test_corrupt_final_store_is_rejected_without_rebinding_or_overwrite(
    publish, tmp_path, simulated_models, handles
):
    root, publication = publish()
    output = tmp_path / "features"
    args = arguments(root, publication, output)
    runner = CliRunner()
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    store = store_path(output)
    path = store / "metadata.json"
    metadata = read_json(path)
    metadata["source_binding"]["receipt_sha256"] = "0" * 64
    path.write_text(json.dumps(metadata), encoding="utf-8")
    before = snapshot(store)
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 1 and "cache failed verification" in result.output
    assert simulated_models["instances"][-1].calls == [] and snapshot(store) == before
    result = runner.invoke(cli.app, ["features", "verify", str(store)])
    assert result.exit_code == 1
    assert not json.loads(result.output)["source_binding_valid"]
    assert_closed(handles)


def test_failed_model_initialization_closes_verified_zip_session(
    publish, tmp_path, simulated_models, handles
):
    root, publication = publish()
    simulated_models["constructor_failure"] = True
    output = tmp_path / "features"
    result = CliRunner().invoke(cli.app, arguments(root, publication, output))
    assert result.exit_code == 1 and "model initialization failure" in result.output
    assert len(simulated_models["constructed"]) == 1 and not output.exists()
    assert_closed(handles)


def test_smoke_and_verification_do_not_claim_full_dataset_or_real_revision(
    publish, tmp_path, simulated_models
):
    root, publication = publish()
    output = tmp_path / "smoke"
    runner = CliRunner()
    result = runner.invoke(
        cli.app,
        arguments(root, publication, output) + ["--limit-content", "1", "--seed", "19"],
    )
    assert result.exit_code == 0, result.output
    store = store_path(output)
    records = pd.read_parquet(store / "record_index.parquet")
    assert len(records) == 4 and records.embedding_row.eq(-1).any()
    result = runner.invoke(cli.app, ["features", "verify", str(store)])
    assert result.exit_code == 0
    assert json.loads(result.output)["source_binding_valid"]
    result = runner.invoke(
        cli.app,
        [
            "features",
            "verify",
            str(store),
            "--manifest",
            str(publication / "manifest.parquet"),
        ],
    )
    assert result.exit_code == 1
    checked = json.loads(result.output)
    assert (
        not checked["full_dataset_valid"]
        and not checked["reproducible_full_dataset_valid"]
    )


@pytest.mark.parametrize("source", ["directory", "zip", "historical_fallback"])
def test_historical_cli_transports_preserve_legacy_metadata_and_mapping(
    publish, tmp_path, monkeypatch, source
):
    root, publication = publish()
    manifest = pd.read_parquet(publication / "manifest.parquet")
    images = tmp_path / "historical-images"
    images.mkdir()
    archive = tmp_path / "Imagenes.zip"
    pixels = {1: png("red"), 2: png("red"), 3: png("blue"), 4: png("green")}
    with zipfile.ZipFile(archive, "w") as zipped:
        for index, frame_id in frames(publication).items():
            path = f"{frame_id}.png"
            (images / path).write_bytes(pixels[index])
            zipped.writestr(path, pixels[index])
    path = tmp_path / "historical.parquet"
    manifest.assign(
        image_path=manifest.frame_id + ".png",
        source_member_path=manifest.frame_id + ".png",
        source_archive=archive.name,
    ).to_parquet(path, index=False)
    output = tmp_path / "features"
    args = [
        "features",
        "extract",
        "--extractor",
        "fake",
        "--manifest",
        str(path),
        "--output-root",
        str(output),
    ]
    if source == "historical_fallback":
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("FLIR_DATA_ROOT", str(tmp_path))
    else:
        args += [
            "--images-root" if source == "directory" else "--images-archive",
            str(images if source == "directory" else archive),
        ]
    result = CliRunner().invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    store = store_path(output)
    metadata = read_json(store / "metadata.json")
    assert "source_binding" not in metadata and "dataset_variant" not in metadata
    assert metadata["dataset_id"] == dataset_id_from_manifest(manifest)
    records = pd.read_parquet(store / "record_index.parquet")
    assert list(records) == ["frame_id", "content_id", "embedding_row"]
    assert (
        len(records) == 4
        and records.groupby("content_id").embedding_row.nunique().eq(1).all()
    )
    assert verify_feature_directory(store)["quality_valid"]


def test_factory_api_rejects_ambiguous_or_invalid_factories(publish, tmp_path, handles):
    root, publication = publish()
    kwargs = dict(
        video_variant_ingestion=publication,
        input_root=root,
        output_root=tmp_path / "features",
    )
    with pytest.raises(ValueError, match="exactly one"):
        extract_to_store(
            publication / "manifest.parquet",
            extractor=DeterministicFakeExtractor(),
            extractor_factory=DeterministicFakeExtractor,
            **kwargs,
        )
    with pytest.raises(TypeError, match="must return"):
        extract_to_store(
            publication / "manifest.parquet", extractor_factory=lambda: None, **kwargs
        )
    assert not kwargs["output_root"].exists()
    assert_closed(handles)
