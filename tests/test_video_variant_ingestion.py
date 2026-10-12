"""Synthetic ingestion only: no FLIR images, video decode, models, GPU or network."""

import io
import json
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd
import pytest
import yaml
from PIL import Image
from typer.testing import CliRunner

from flir_pipeline.cli import app
from flir_pipeline.data.identity import dataset_id_from_manifest
from flir_pipeline.data.variants import read_variant
from flir_pipeline.data.video_variant_contract import IngestionConfig, digest_document
from flir_pipeline.data.video_variant_storage import (
    build_ingestion,
    inspect_ingestion,
    summarize_ingestion,
    verify_ingestion,
)
from flir_pipeline.utils.hashing import sha256_file


def png(color):
    stream = io.BytesIO()
    Image.new("RGB", (6, 4), color).save(stream, format="PNG")
    return stream.getvalue()


def write_zip(path, pairs, compression=ZIP_DEFLATED):
    with ZipFile(path, "w", compression) as archive:
        for name, content in pairs:
            if name in archive.namelist():
                with pytest.warns(UserWarning, match="Duplicate name"):
                    archive.writestr(name, content)
            else:
                archive.writestr(name, content)


@pytest.fixture
def sample(tmp_path):
    root = tmp_path / "inputs"
    root.mkdir()
    write_zip(
        root / "first.zip", [("frame_3.png", png("blue")), ("frame_1.png", png("red"))]
    )
    write_zip(
        root / "second.zip", [("frame_2.png", png("red")), ("note.txt", b"metadata")]
    )
    (root / "video.mp4").write_bytes(b"synthetic-video-source-not-decoded")
    (root / "inventory.csv").write_text("reported_count\n9999\n", encoding="utf-8")
    declaration = dict(
        collection_id="example",
        variant_name="hud_reduced",
        transformation={
            "description": "external processing",
            "tool_version": None,
            "limitations": ["HUD reduction is not exhaustive elimination"],
        },
        archives=[
            {"archive_key": "a", "path": "first.zip"},
            {"archive_key": "b", "path": "second.zip"},
        ],
        series=[
            {
                "series_id": "series-a",
                "archive_keys": ["a", "b"],
                "index_pattern": r"frame_(?P<index>\d+)\.png",
                "expected_index_start": 1,
                "expected_index_stop": 3,
                "candidate_video_key": "video-a",
                "candidate_time": {
                    "fps": 7.5,
                    "index_origin": 1,
                    "offset_seconds": 0.25,
                },
            }
        ],
        videos=[
            {
                "video_key": "video-a",
                "path": "video.mp4",
                "declared_metadata": {"fps": 8.0},
            }
        ],
        auxiliary_inventory={
            "path": "inventory.csv",
            "expected_sha256": sha256_file(root / "inventory.csv"),
        },
        observations=[
            {
                "observation_id": "point-a",
                "series_id": "series-a",
                "video_key": "video-a",
                "description": "reported comparison; exact processed index unavailable",
                "reported_timestamp_seconds": 1.0,
            }
        ],
    )
    config = tmp_path / "ingestion.yaml"

    def save():
        config.write_text(yaml.safe_dump(declaration), encoding="utf-8")

    save()
    return root, config, declaration, save


def build(sample, tmp_path, name="outputs"):
    root, config, _, _ = sample
    return build_ingestion(config, root, tmp_path / name)


def read(artifact, name):
    return pd.read_parquet(artifact / f"{name}.parquet")


def test_complete_audit_preserves_content_copies_and_never_verifies_time(
    sample, tmp_path
):
    root, _, _, _ = sample
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    artifact = build(sample, tmp_path)
    summary = summarize_ingestion(artifact)
    assert summary["integrity_valid"] and summary["scientific_manifest_available"]
    assert (
        summary["image_occurrence_count"] == 3 and summary["unique_content_count"] == 2
    )
    assert summary["duplicate_records"] == 2 and summary["redundant_records"] == 1
    manifest = read(artifact, "manifest")
    assert (
        manifest.frame_id.is_unique
        and manifest.content_id.eq(manifest.image_sha256).all()
    )
    assert manifest.original_split.eq("").all() and manifest.label_sha256.eq("").all()
    assert not manifest.label_exists.any()
    assert not {
        "label_empty",
        "num_objects",
        "video_id",
        "timestamp_seconds",
        "sample_index",
        "cluster_id",
    } & set(manifest)
    variant = read_variant(artifact / "variant.json", artifact / "manifest.parquet")
    assert (
        variant["dataset_id"]
        == dataset_id_from_manifest(manifest)
        == summary["dataset_id"]
    )
    candidates = read(artifact, "alignment_candidates").sort_values("observed_index")
    assert candidates.candidate_timestamp_seconds.tolist() == [
        0.25,
        0.25 + 1 / 7.5,
        0.25 + 2 / 7.5,
    ]
    assert candidates.alignment_status.eq("candidate").all()
    assert candidates.verification_scope.eq("none").all()
    assert candidates.verified_native_frame_index.isna().all()
    assert candidates.verified_timestamp_seconds.isna().all()
    assert not summary["alignment_verified"] and not summary["detector_ready"]
    observation = read(artifact, "observations").iloc[0]
    assert pd.isna(observation.observed_index) and pd.isna(observation.reviewer)
    assert observation.evidence_status == "externally_reported"
    assert {p.name: p.read_bytes() for p in root.iterdir()} == before
    assert verify_ingestion(artifact, root)["source_bound"]
    assert not verify_ingestion(artifact)["source_bound"]


def test_repackaging_names_order_shards_and_root_do_not_change_scientific_identity(
    sample, tmp_path
):
    first = build(sample, tmp_path, "initial")
    root, config, declaration, save = sample
    other = tmp_path / "repacked"
    other.mkdir()
    for name in ("video.mp4", "inventory.csv"):
        (other / name).write_bytes((root / name).read_bytes())
    write_zip(
        other / "new.zip",
        [
            ("nested/frame_2.png", png("red")),
            ("nested/frame_3.png", png("blue")),
            ("frame_1.png", png("red")),
        ],
        compression=0,
    )
    declaration["archives"] = [{"archive_key": "new-shard", "path": "new.zip"}]
    declaration["series"][0]["archive_keys"] = ["new-shard"]
    save()
    second = build_ingestion(config, other, tmp_path / "new-outputs")
    pd.testing.assert_frame_equal(read(first, "manifest"), read(second, "manifest"))
    assert sha256_file(first / "manifest.parquet") == sha256_file(
        second / "manifest.parquet"
    )
    for key in ("dataset_id", "dataset_variant_id"):
        assert summarize_ingestion(first)[key] == summarize_ingestion(second)[key]
    assert first.name != second.name
    assert set(read(first, "entries").entry_id).isdisjoint(
        read(second, "entries").entry_id
    )


@pytest.mark.parametrize("same_bytes", [True, False])
def test_repeated_indices_preserve_all_entries_and_withhold_manifest(
    sample, tmp_path, same_bytes
):
    root, _, _, _ = sample
    write_zip(
        root / "second.zip",
        [
            ("frame_2.png", png("red")),
            ("frame_1.png", png("red" if same_bytes else "green")),
        ],
    )
    artifact = build(sample, tmp_path)
    occurrences = read(artifact, "occurrences")
    duplicates = occurrences[occurrences.observed_index.eq(1)]
    assert len(occurrences) == 4 and len(duplicates) == 2
    assert duplicates.entry_id.is_unique and duplicates.frame_id.isna().all()
    assert duplicates.observation_id.nunique() == 1
    assert duplicates.identity_status.eq("ambiguous_index").all()
    summary = summarize_ingestion(artifact)
    assert (
        not summary["scientific_manifest_available"] and summary["dataset_id"] is None
    )
    assert summary["dataset_variant_id"] is None and not summary["integrity_valid"]
    assert (
        not (artifact / "manifest.parquet").exists()
        and not (artifact / "variant.json").exists()
    )
    assert "duplicate_index" in set(read(artifact, "index_issues").issue_type)


def test_duplicate_names_inside_one_zip_are_not_lost(sample, tmp_path):
    root, _, _, _ = sample
    write_zip(
        root / "second.zip",
        [("frame_2.png", png("red")), ("frame_2.png", png("green"))],
    )
    artifact = build(sample, tmp_path)
    occurrences = read(artifact, "occurrences")
    duplicates = occurrences[occurrences.observed_index.eq(2)]
    assert len(duplicates) == 2 and duplicates.content_id.nunique() == 2
    assert duplicates.member_ordinal.nunique() == 2 and duplicates.entry_id.is_unique


def test_multiple_series_and_overlapping_selectors(sample, tmp_path):
    root, _, declaration, save = sample
    write_zip(
        root / "first.zip",
        [("left/frame_1.png", png("red")), ("right/frame_1.png", png("blue"))],
    )
    write_zip(root / "second.zip", [])
    declaration["series"] = [
        {
            "series_id": side,
            "archive_keys": ["a", "b"],
            "member_pattern": f"{side}/.*",
            "index_pattern": r"frame_(?P<index>\d+)\.png",
        }
        for side in ("left", "right")
    ]
    declaration["observations"] = []
    save()
    artifact = build(sample, tmp_path)
    assert read(artifact, "manifest").series_id.nunique() == 2
    assert summarize_ingestion(artifact)["scientific_manifest_available"]
    declaration["series"][1]["member_pattern"] = ".*"
    save()
    ambiguous = build(sample, tmp_path, "overlap")
    assert "ambiguous_series" in set(read(ambiguous, "index_issues").issue_type)
    assert not summarize_ingestion(ambiguous)["scientific_manifest_available"]


def test_missing_endpoints_off_grid_and_unparsed_indices(sample, tmp_path):
    root, _, declaration, save = sample
    write_zip(
        root / "first.zip", [("frame_2.png", png("red")), ("unparsed.png", png("blue"))]
    )
    write_zip(root / "second.zip", [("frame_9.png", png("green"))])
    declaration["series"][0]["expected_index_stop"] = 4
    save()
    artifact = build(sample, tmp_path)
    issues = read(artifact, "index_issues")
    assert set(issues.issue_type) == {
        "missing_index_range",
        "off_grid_index",
        "unparsed_index",
    }
    assert summarize_ingestion(artifact)["missing_index_count"] == 3
    assert len(read(artifact, "occurrences")) == 3


def test_large_expected_range_is_compact_not_expanded(sample, tmp_path):
    _, _, declaration, save = sample
    declaration["series"][0]["expected_index_stop"] = 10**12
    save()
    artifact = build(sample, tmp_path)
    assert summarize_ingestion(artifact)["missing_index_count"] == 10**12 - 3
    assert len(read(artifact, "index_issues")) == 1


def test_corrupt_png_retains_exact_identity_but_fails_integrity(sample, tmp_path):
    root, _, _, _ = sample
    write_zip(root / "second.zip", [("frame_2.png", b"corrupt PNG bytes")])
    artifact = build(sample, tmp_path)
    summary = summarize_ingestion(artifact)
    assert summary["scientific_manifest_available"] and not summary["integrity_valid"]
    assert summary["decode_errors"] == 1 and len(read(artifact, "manifest")) == 3
    corrupt = read(artifact, "occurrences").query("observed_index == 2").iloc[0]
    assert corrupt.content_id and not corrupt.image_decode_valid


@pytest.mark.parametrize(
    "kind",
    [
        "corrupt_zip",
        "unsafe_member",
        "member_limit",
        "archive_limit",
        "dimension_limit",
    ],
)
def test_errors_and_limits_publish_explicit_audits(sample, tmp_path, kind):
    root, _, declaration, save = sample
    if kind == "corrupt_zip":
        (root / "second.zip").write_bytes(b"corrupt ZIP bytes")
    elif kind == "unsafe_member":
        write_zip(root / "second.zip", [("../frame_2.png", png("red"))])
    elif kind == "member_limit":
        declaration["limits"] = {"max_member_bytes": 5}
    elif kind == "archive_limit":
        declaration["limits"] = {"max_archive_entries": 1}
    else:
        declaration["limits"] = {"max_pixels": 3}
    save()
    artifact = build(sample, tmp_path)
    summary = summarize_ingestion(artifact)
    assert not summary["integrity_valid"]
    assert (
        summary["archive_errors"] or summary["read_errors"] or summary["decode_errors"]
    )
    assert verify_ingestion(artifact, root)["publication_valid"]


@pytest.mark.parametrize("source", ["first.zip", "video.mp4", "inventory.csv"])
def test_altered_original_sources_fail_verification(sample, tmp_path, source):
    artifact = build(sample, tmp_path)
    root, _, _, _ = sample
    (root / source).write_bytes(b"changed source")
    with pytest.raises(ValueError, match="checksum"):
        verify_ingestion(artifact, root)


def test_sources_changed_during_ingestion_refuse_publication(
    sample, tmp_path, monkeypatch
):
    from flir_pipeline.data import video_variant_ingestion as ingestion

    original = ingestion.snapshot_sources
    calls = 0

    def changing(*args):
        nonlocal calls
        snapshot = original(*args)
        calls += 1
        if calls == 2:
            snapshot["archive:a"]["sha256"] = "0" * 64
        return snapshot

    monkeypatch.setattr(ingestion, "snapshot_sources", changing)
    with pytest.raises(ValueError, match="changed during"):
        build(sample, tmp_path)
    assert not [
        p for p in (tmp_path / "outputs").iterdir() if not p.name.endswith(".partial")
    ]


def test_publication_is_immutable_and_reuses_verified_equivalent(sample, tmp_path):
    first = build(sample, tmp_path)
    before = {p.name: p.read_bytes() for p in first.iterdir()}
    assert build(sample, tmp_path) == first
    assert {p.name: p.read_bytes() for p in first.iterdir()} == before
    assert not list(first.parent.glob("*.partial"))
    (first / "integrity.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Modified ingestion output"):
        build(sample, tmp_path)
    assert (first / "integrity.json").read_text() == "{}"


def test_interrupted_publication_keeps_staging_and_previous_outputs(
    sample, tmp_path, monkeypatch
):
    from flir_pipeline.data import video_variant_storage as storage

    previous = build(sample, tmp_path)
    before = {p.name: p.read_bytes() for p in previous.iterdir()}
    original = storage._write_json

    def fail(path, value):
        if path.name == "metadata.json":
            raise OSError("synthetic interruption")
        original(path, value)

    monkeypatch.setattr(storage, "_write_json", fail)
    with pytest.raises(OSError, match="interruption"):
        build(sample, tmp_path)
    partials = list(previous.parent.glob("*.partial"))
    assert len(partials) == 1
    assert {p.name: p.read_bytes() for p in previous.iterdir()} == before
    assert not (previous.parent / ".video-variant-writer.lock").exists()
    with pytest.raises(ValueError, match="Incomplete"):
        inspect_ingestion(partials[0])


def test_single_writer_lock_is_never_broken(sample, tmp_path):
    output = tmp_path / "outputs"
    output.mkdir()
    lock = output / ".video-variant-writer.lock"
    lock.write_text("other writer", encoding="utf-8")
    with pytest.raises(FileExistsError):
        build(sample, tmp_path)
    assert lock.read_text() == "other writer"


def test_rebound_checksums_cannot_promote_candidate_alignment(sample, tmp_path):
    artifact = build(sample, tmp_path)
    candidates = read(artifact, "alignment_candidates")
    candidates["alignment_status"] = "verified"
    candidates.to_parquet(artifact / "alignment_candidates.parquet", index=False)
    meta = json.loads((artifact / "metadata.json").read_text())
    meta["identity"]["output_checksums"]["alignment_candidates.parquet"] = sha256_file(
        artifact / "alignment_candidates.parquet"
    )
    meta["artifact_id"] = digest_document(meta["identity"])
    (artifact / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    (artifact / "receipt.json").write_text(
        json.dumps({"metadata_sha256": sha256_file(artifact / "metadata.json")}),
        encoding="utf-8",
    )
    renamed = artifact.with_name(meta["artifact_id"])
    artifact.rename(renamed)
    with pytest.raises(ValueError):
        inspect_ingestion(renamed)


@pytest.mark.parametrize(
    "path", ["../first.zip", "/first.zip", "C:/first.zip", "first.zip:ads"]
)
def test_config_paths_are_portable_and_safe(sample, path):
    _, _, declaration, _ = sample
    declaration["archives"][0]["path"] = path
    with pytest.raises(ValueError, match="Unsafe"):
        IngestionConfig.model_validate(declaration)


def test_output_cannot_write_into_sources_or_public_checkout(sample, tmp_path):
    from pathlib import Path

    root, config, _, _ = sample
    for destination in (
        root / "outputs",
        root.parent,
        Path(__file__).resolve().parents[1] / "exports" / "new-variant",
    ):
        with pytest.raises(ValueError):
            build_ingestion(config, root, destination)


def test_symlink_source_escape_is_rejected(sample, tmp_path):
    root, _, declaration, save = sample
    outside = tmp_path / "outside.zip"
    write_zip(outside, [("frame_1.png", png("red"))])
    link = root / "escaped.zip"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("This host does not grant symlink creation")
    declaration["archives"][0]["path"] = "escaped.zip"
    save()
    with pytest.raises(ValueError, match="escapes"):
        build(sample, tmp_path)


def test_explicit_observations_cannot_claim_verified_status(sample):
    _, _, declaration, _ = sample
    declaration["observations"][0]["verification_scope"] = "collection"
    with pytest.raises(ValueError):
        IngestionConfig.model_validate(declaration)


@pytest.mark.parametrize(
    "change", ["collection", "pixels", "candidate_time", "transformation"]
)
def test_identity_changes_only_at_its_declared_boundary(sample, tmp_path, change):
    first = build(sample, tmp_path, "before")
    root, _, declaration, save = sample
    if change == "collection":
        declaration["collection_id"] = "another-collection"
    elif change == "pixels":
        write_zip(root / "second.zip", [("frame_2.png", png("green"))])
    elif change == "candidate_time":
        declaration["series"][0]["candidate_time"]["fps"] = 5.0
    else:
        declaration["transformation"]["tool_version"] = "declared-version"
    save()
    second = build(sample, tmp_path, "after")
    before, after = summarize_ingestion(first), summarize_ingestion(second)
    assert (before["dataset_id"] != after["dataset_id"]) == (
        change in {"collection", "pixels"}
    )
    assert (before["dataset_variant_id"] != after["dataset_variant_id"]) == (
        change != "candidate_time"
    )
    assert first.name != second.name


def test_no_candidate_rule_infers_time_from_metadata_or_image_count(sample, tmp_path):
    _, _, declaration, save = sample
    declaration["series"][0].pop("candidate_time")
    save()
    artifact = build(sample, tmp_path)
    candidates = read(artifact, "alignment_candidates")
    assert candidates.candidate_timestamp_seconds.isna().all()
    assert candidates.timestamp_rule.isna().all()
    assert candidates.alignment_status.eq("candidate").all()


def test_negative_candidate_times_remain_invalid_and_unverified(sample, tmp_path):
    _, _, declaration, save = sample
    declaration["series"][0]["candidate_time"]["index_origin"] = 10
    save()
    artifact = build(sample, tmp_path)
    assert (
        read(artifact, "alignment_candidates").candidate_timestamp_seconds.isna().all()
    )
    assert "invalid_candidate_time" in set(read(artifact, "index_issues").issue_type)
    assert not summarize_ingestion(artifact)["alignment_verified"]


def test_same_observed_index_with_different_padding_is_ambiguous(sample, tmp_path):
    root, _, _, _ = sample
    write_zip(
        root / "second.zip",
        [("frame_2.png", png("red")), ("frame_0001.png", png("red"))],
    )
    artifact = build(sample, tmp_path)
    duplicates = read(artifact, "occurrences").query("observed_index == 1")
    assert len(duplicates) == 2 and duplicates.frame_id.isna().all()


def test_missing_range_without_declared_endpoints_has_limited_scope(sample, tmp_path):
    root, _, declaration, save = sample
    write_zip(root / "second.zip", [])
    declaration["series"][0].pop("expected_index_start")
    declaration["series"][0].pop("expected_index_stop")
    save()
    artifact = build(sample, tmp_path)
    summary = summarize_ingestion(artifact)
    assert summary["missing_index_count"] == 1
    assert summary["index_completeness_scope"]["series-a"] == "observed_interior_only"
    assert summary["scientific_manifest_available"] and not summary["integrity_valid"]


def test_generic_config_example_is_valid_and_contains_no_actual_identity():
    from pathlib import Path

    from flir_pipeline.data.video_variant_contract import load_config

    example = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "data"
        / "video_variant_ingestion.example.yaml"
    )
    config = load_config(example)
    assert config.collection_id == "example-collection"
    assert config.series[0].candidate_time.fps == 7.5


@pytest.mark.parametrize(
    "change",
    ["regex", "bool_index", "nonfinite_fps", "unknown_archive", "verified_extra"],
)
def test_invalid_configuration_is_rejected(sample, change):
    _, _, declaration, _ = sample
    series = declaration["series"][0]
    if change == "regex":
        series["index_pattern"] = "("
    elif change == "bool_index":
        series["expected_index_start"] = True
    elif change == "nonfinite_fps":
        series["candidate_time"]["fps"] = float("inf")
    elif change == "unknown_archive":
        series["archive_keys"] = ["missing"]
    else:
        series["verified_native_frame_index"] = 1
    with pytest.raises(ValueError):
        IngestionConfig.model_validate(declaration)


def test_change_during_staging_cancels_promotion(sample, tmp_path, monkeypatch):
    from flir_pipeline.data import video_variant_storage as storage

    original = storage._write_json
    root, _, _, _ = sample

    def change(path, value):
        original(path, value)
        if path.name == "receipt.json":
            (root / "video.mp4").write_bytes(b"source changed after scanning")

    monkeypatch.setattr(storage, "_write_json", change)
    with pytest.raises(ValueError, match="changed before publication"):
        build(sample, tmp_path)
    assert all(
        path.name.endswith(".partial") for path in (tmp_path / "outputs").iterdir()
    )


def test_optional_ffprobe_metadata_does_not_verify_alignment(
    sample, tmp_path, monkeypatch
):
    from flir_pipeline.data import video_frames

    calls = []

    def probe(path, binary, relative):
        calls.append(relative)
        return video_frames.VideoInfo(
            8.0,
            12,
            8,
            1.0,
            8,
            "synthetic",
            "8/1",
            "8/1",
            "avg_frame_rate",
            "stream.duration",
        )

    monkeypatch.setattr(video_frames, "probe_video", probe)
    root, config, _, _ = sample
    artifact = build_ingestion(
        config, root, tmp_path / "outputs", ffprobe_bin="missing-synthetic-ffprobe"
    )
    assert calls == ["video.mp4"]
    assert read(artifact, "video_sources").probe_status.eq("observed").all()
    assert not summarize_ingestion(artifact)["alignment_verified"]
    # Source replay binds bytes/images, without repeating the optional tool probe.
    assert verify_ingestion(artifact, root)["source_bound"] and len(calls) == 1


def test_extreme_zip64_declaration_remains_auditable_without_int64_overflow(
    sample, tmp_path, monkeypatch
):
    original = ZipFile.infolist

    def extreme(archive):
        infos = original(archive)
        if infos:
            infos[0].file_size = 2**64 - 1
        return infos

    monkeypatch.setattr(ZipFile, "infolist", extreme)
    artifact = build(sample, tmp_path)
    summary = summarize_ingestion(artifact)
    assert (
        summary["archive_errors"] == 2 and not summary["scientific_manifest_available"]
    )
    assert read(artifact, "archives").uncompressed_bytes.isna().all()
    assert read(artifact, "archives").archive_error.str.contains("declared bytes").all()


def test_cli_build_verify_summary_and_clear_failures(sample, tmp_path, monkeypatch):
    root, config, _, _ = sample
    runner = CliRunner()
    monkeypatch.setenv("FLIR_DATA_ROOT", str(root))
    result = runner.invoke(
        app,
        [
            "data",
            "video-variant-ingestion",
            "build",
            "--config",
            str(config),
            "--output-root",
            str(tmp_path / "cli-outputs"),
        ],
    )
    assert result.exit_code == 0, result.output
    artifact = json.loads(result.output)["artifact"]
    for command in ("summary", "verify"):
        result = runner.invoke(
            app, ["data", "video-variant-ingestion", command, artifact]
        )
        assert result.exit_code == 0, result.output
        assert not json.loads(result.output)["alignment_verified"]
    result = runner.invoke(
        app,
        [
            "data",
            "video-variant-ingestion",
            "verify",
            artifact,
            "--input-root",
            str(root),
        ],
    )
    assert result.exit_code == 0 and json.loads(result.output)["source_bound"]
    missing = runner.invoke(
        app, ["data", "video-variant-ingestion", "summary", str(tmp_path / "missing")]
    )
    assert missing.exit_code == 1
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("FLIR_DATA_ROOT")
    assert (
        runner.invoke(
            app, ["data", "video-variant-ingestion", "build", "--config", str(config)]
        ).exit_code
        != 0
    )
    for command in ("build", "verify", "summary"):
        assert (
            runner.invoke(
                app, ["data", "video-variant-ingestion", command, "--help"]
            ).exit_code
            == 0
        )
