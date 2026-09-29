"""Synthetic variant isolation and paired evidence; no real models or HUD claims."""

import copy
import hashlib

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from test_features import _image_bytes, _manifest, _write_zip
from test_sequence_experiments import synthetic_memory, zones
from test_sequences import synthetic_sources
from typer.testing import CliRunner

from flir_pipeline.cli import app
from flir_pipeline.data.variants import make_variant, read_variant, register_variant
from flir_pipeline.features.base import DeterministicFakeExtractor
from flir_pipeline.features.storage import extract_to_store, verify_feature_directory
from flir_pipeline.sequences.experiments.artifacts import inspect, publish, tables
from flir_pipeline.sequences.experiments.config import (
    BoundaryConfig,
    ClusterGrid,
    SuiteConfig,
)
from flir_pipeline.sequences.experiments.inputs import Inputs, resolve_inputs
from flir_pipeline.sequences.experiments.runner import run_boundary, suite
from flir_pipeline.sequences.experiments.sources import (
    ExperimentSources,
    load_sources,
    scientific_input_id,
)
from flir_pipeline.sequences.experiments.structure import (
    EvidenceEnvelope,
    expected_binding,
    publish_normalized,
)
from flir_pipeline.sequences.experiments.variant_comparison import (
    PAIR_COLUMNS,
    compare_assignments,
    compare_variants,
    content_pairing,
    import_correspondence,
    normalize_pairs,
    read_suite,
)
from flir_pipeline.similarity.storage import file_sha256, read_json, write_json


def test_variant_feature_stores_are_immutable_and_separate(tmp_path):
    image = _image_bytes()
    manifest, archive = tmp_path / "manifest.parquet", tmp_path / "images.zip"
    _manifest(manifest, hashlib.sha256(image).hexdigest())
    _write_zip(archive, {"Imagenes/train/a.png": image, "Imagenes/val/b.png": image})
    declarations = [
        register_variant(manifest, name, tmp_path / "variants")
        for name in ("first", "second")
    ]
    stores = [
        extract_to_store(
            manifest,
            archive,
            DeterministicFakeExtractor(),
            tmp_path / "features",
            variant_spec=p,
        )
        for p in declarations
    ]
    assert stores[0] != stores[1]
    a, b = [read_json(p / "metadata.json") for p in stores]
    assert a["feature_space_id"] == b["feature_space_id"]
    assert (
        a["dataset_variant"]["dataset_variant_id"]
        != b["dataset_variant"]["dataset_variant_id"]
    )
    assert (
        extract_to_store(
            manifest,
            archive,
            DeterministicFakeExtractor(),
            tmp_path / "features",
            variant_spec=declarations[0],
        )
        == stores[0]
    )
    # A top-level declaration cannot be relabeled independently of its checkpoint identity.
    a["dataset_variant"] = b["dataset_variant"]
    write_json(stores[0] / "metadata.json", a)
    assert not verify_feature_directory(stores[0])["quality_valid"]
    with pytest.raises(RuntimeError, match="verification"):
        extract_to_store(
            manifest,
            archive,
            DeterministicFakeExtractor(),
            tmp_path / "features",
            variant_spec=declarations[0],
        )


def test_variant_registration_parent_and_source_checksum(tmp_path):
    manifest = synthetic_sources(tmp_path / "sources", (24,))[0]
    first = register_variant(manifest, "arbitrary_rendering", tmp_path / "variants")
    second = register_variant(
        manifest,
        "crop",
        tmp_path / "variants",
        parent=first,
        definition={"region": [0, 0, 8, 8]},
    )
    declared = read_variant(second, manifest)
    assert (
        declared["parent_dataset_identity"]["dataset_variant_id"]
        == read_variant(first)["dataset_variant_id"]
    )
    assert declared["dataset_variant_id"] != read_variant(first)["dataset_variant_id"]
    changed = pd.read_parquet(manifest).iloc[::-1]
    changed.to_parquet(manifest, index=False)
    with pytest.raises(ValueError, match="checksum"):
        read_variant(second, manifest)


def test_feature_resolution_refuses_relabeling_and_mixed_variants(
    tmp_path, monkeypatch
):
    for key in (
        "FLIR_MANIFEST",
        "FLIR_CLIP_FEATURES",
        "FLIR_DINOV2_FEATURES",
        "FLIR_DATASET_VARIANT",
    ):
        monkeypatch.delenv(key, raising=False)
    manifest, clip, dino = synthetic_sources(tmp_path / "sources", (24,))
    with pytest.raises(ValueError, match="variant mismatch"):
        load_sources(manifest, clip, dino, dataset_variant="first")
    for directory, name in ((clip, "first"), (dino, "second")):
        meta = read_json(directory / "metadata.json")
        meta["dataset_variant"] = make_variant(
            meta["dataset_id"], file_sha256(manifest), name
        )
        write_json(directory / "metadata.json", meta)
    with pytest.raises(ValueError, match="same declared"):
        load_sources(manifest, clip, dino)
    meta = read_json(dino / "metadata.json")
    meta["dataset_variant"] = read_json(clip / "metadata.json")["dataset_variant"]
    write_json(dino / "metadata.json", meta)
    inputs = Inputs(manifest=str(manifest), search_roots=[str(tmp_path / "sources")])
    _, source = resolve_inputs(inputs, "all", dataset_variant="first")
    assert source.signature["variant_name"] == "first"
    with pytest.raises(ValueError, match="variant mismatch"):
        load_sources(manifest, clip, dino, dataset_variant="second")


def test_variant_alone_changes_scientific_and_artifact_identity(tmp_path):
    original = synthetic_memory()
    other = ExperimentSources(
        original.records.copy(),
        original.contents,
        original.embeddings,
        {
            **original.signature,
            "dataset_variant": make_variant(
                original.signature["dataset_id"], "a" * 64, "alternate"
            ),
        },
        original.family,
    )
    assert scientific_input_id(original) != scientific_input_id(other)
    first = run_boundary(original, BoundaryConfig(), tmp_path / "results")
    second = run_boundary(other, BoundaryConfig(), tmp_path / "results")
    assert first != second
    for path in (first, second):
        source = inspect(path)["identity"]["sources"]["input"]
        assert {
            "dataset_id",
            "dataset_variant_id",
            "variant_name",
            "parent_dataset_identity",
            "feature_spaces",
            "checksums",
        } <= set(source)


def pixel_source(overlay):
    """A digest-based fake encoder responds to pixels, without any model claim."""
    source = synthetic_memory()
    extractor = DeterministicFakeExtractor()
    content, vectors = [], []
    for i in range(len(source.records)):
        pixels = np.full((16, 16, 3), 40 if i // 24 != 1 else 160, dtype=np.uint8)
        pixels[-1, -1] = i  # Unique content, identical across the paired variants.
        if overlay:
            pixels[:2, :8] = 240 - i
        image = Image.fromarray(pixels)
        content.append(hashlib.sha256(pixels.tobytes()).hexdigest())
        vectors.append(extractor.encode_batch([extractor.preprocess(image)])[0])
    vectors = np.asarray(vectors)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    records = source.records.copy()
    name = "overlay" if overlay else "plain"
    records["frame_id"] = name + "-" + records.frame_id
    records["content_id"] = content
    ids = sorted(content)
    lookup = {v: i for i, v in enumerate(content)}
    records["content_row"] = records.content_id.map({v: i for i, v in enumerate(ids)})
    vectors = vectors[[lookup[c] for c in ids]]
    signature = copy.deepcopy(source.signature)
    signature["dataset_id"] = "synthetic-" + name
    signature["dataset_variant"] = make_variant(signature["dataset_id"], "a" * 64, name)
    return ExperimentSources(
        records,
        ids,
        {"clip": vectors, "dinov2": vectors.copy()},
        signature,
        source.family,
    )


@pytest.fixture(scope="module")
def variant_suites(tmp_path_factory):
    root = tmp_path_factory.mktemp("variants")
    config = SuiteConfig(
        reductions=(),
        clustering=(
            ClusterGrid(algorithm="agglomerative", grid={"n_clusters": [2, 3]}),
        ),
        boundary=BoundaryConfig(
            windows=(1, 3, 5), baseline_radius=10, baseline_min_count=3
        ),
    )
    sources = [pixel_source(True), pixel_source(False)]
    paths = []
    for source in sources:
        envelope = EvidenceEnvelope(
            **expected_binding(source),
            artifact_kind="sequence_structure_external_v1",
            ground_truth=False,
            split_created=False,
            automatic_confirmation=False,
            reviewer="synthetic reviewer",
            reviewed_at="2026-01-01T00:00:00Z",
            producer_files={"review.csv": "b" * 64},
            intervals=zones(source).to_dict("records"),
        )
        structure = publish_normalized(
            envelope, source, root / "artifacts", {"synthetic": True}
        )
        paths.append(suite(source, config, root / "artifacts", structure))
    return sources, paths


def pairing_csv(tmp_path, sources, count=72):
    frame = pd.DataFrame(
        {
            "left_frame_id": sources[0].records.frame_id.iloc[:count].tolist(),
            "right_frame_id": sources[1].records.frame_id.iloc[:count].tolist(),
            "mapping_method": "synthetic construction",
            "confidence": 1.0,
            "evidence": "same source occurrence; changed top strip only",
            "ground_truth": False,
        },
        columns=PAIR_COLUMNS,
    )
    path = tmp_path / "pairs.csv"
    frame.to_csv(path, index=False)
    return path


def test_hud_like_pixels_change_features_and_results_not_temporal_identity(
    variant_suites,
):
    sources, paths = variant_suites
    pd.testing.assert_frame_equal(
        sources[0].records[["timeline_id", "position", "temporal_source"]],
        sources[1].records[["timeline_id", "position", "temporal_source"]],
    )
    assert set(sources[0].contents).isdisjoint(sources[1].contents)
    ordered = [s.embeddings["clip"][s.records.content_row] for s in sources]
    assert not np.array_equal(*ordered)
    scores = [
        read_suite(p).children["boundary_0"]["temporal_scores"].clip_adjacent_cosine
        for p in paths
    ]
    assert not np.allclose(*scores, equal_nan=True)


def test_compare_without_pairing_preserves_populations_and_creates_no_groups(
    variant_suites, tmp_path, monkeypatch
):
    _, paths = variant_suites

    def forbidden(*args, **kwargs):
        raise AssertionError("comparison must not fit models")

    monkeypatch.setattr(
        "flir_pipeline.sequences.experiments.fitting.fit_visual", forbidden
    )
    result = compare_variants(*paths, tmp_path / "comparison")
    summary = read_json(result / "summary.json")
    assert (
        summary["left_unpaired_occurrences"]
        == summary["right_unpaired_occurrences"]
        == 72
    )
    assert summary["left_occurrence_coverage"] == 0
    for flag in (
        "automatic_winner_selected",
        "split_created",
        "visual_dependency_groups_created",
        "sequence_instances_created",
        "ground_truth",
    ):
        assert summary[flag] is False
    data = tables(result)
    assert len(data["assignments"]) == 2 * 4 * 72
    assert data["assignment_agreement"].all_points_ari.isna().all()
    assert len(data["adjacent_distributions"]) == 4
    assert {
        "boundary_zones",
        "recurrence_pairs",
        "candidate_shortlists",
        "stability",
        "evaluation_diagnostics",
    } <= set(data)
    assert len(data["left_occurrences"]) == 72


@pytest.mark.parametrize("count", [30, 72])
def test_pairing_reports_coverage_and_unique_content_agreement(
    variant_suites, tmp_path, count
):
    sources, paths = variant_suites
    csv = pairing_csv(tmp_path, sources, count)
    paired = import_correspondence(*paths, csv, tmp_path / "pairing")
    result = compare_variants(*paths, tmp_path / "comparison", paired)
    summary, data = read_json(result / "summary.json"), tables(result)
    assert summary["left_paired_occurrences"] == count
    assert summary["right_unpaired_occurrences"] == 72 - count
    assert summary["left_occurrence_coverage"] == count / 72
    assert data["assignment_agreement"].all_points_n.eq(count).all()
    assert data["assignment_agreement"].all_points_ari.notna().all()
    assert (
        data["paired_occurrences"].left_position.tolist()
        == data["paired_occurrences"].right_position.tolist()
    )
    assert not data["paired_occurrences"].byte_identity_asserted.any()
    assert summary["paired_cores"] == (3 if count == 72 else 1)
    assert summary["compared_recurrence_pairs"] == (3 if count == 72 else 0)
    with pytest.raises(ValueError, match="orientation"):
        compare_variants(paths[1], paths[0], tmp_path / "bad", paired)


@pytest.mark.parametrize(
    "column,value",
    [
        ("left_frame_id", "unknown"),
        ("confidence", -1),
        ("confidence", "nan"),
        ("ground_truth", "maybe"),
        ("mapping_method", ""),
        ("evidence", ""),
    ],
)
def test_invalid_correspondence_rejected(variant_suites, tmp_path, column, value):
    sources, paths = variant_suites
    frame = pd.read_csv(pairing_csv(tmp_path, sources), dtype=object)
    frame.loc[0, column] = value
    with pytest.raises(ValueError):
        normalize_pairs(frame, *[read_suite(p) for p in paths])


def test_duplicate_occurrences_rejected_and_nonbijective_contents_reported(
    variant_suites, tmp_path
):
    sources, paths = variant_suites
    frame = pd.read_csv(pairing_csv(tmp_path, sources))
    frame.loc[1, "right_frame_id"] = frame.loc[0, "right_frame_id"]
    with pytest.raises(ValueError, match="one-to-one"):
        normalize_pairs(frame, *[read_suite(p) for p in paths])
    relation = pd.DataFrame(
        {
            "left_content_id": ["a", "a", "a", "b"],
            "right_content_id": ["x", "x", "y", "z"],
        }
    )
    edges = content_pairing(relation)
    assert len(edges) == 3
    assert edges.eligible.tolist() == [False, False, True]
    assert edges.exclusion_reason.iloc[0] == "non_bijective_content_relation"


def test_variant_comparison_cli_and_incomplete_rejection(variant_suites, tmp_path):
    _, paths = variant_suites
    runner = CliRunner()
    response = runner.invoke(
        app,
        [
            "sequences",
            "experiment",
            "compare-variants",
            str(paths[0]),
            str(paths[1]),
            "--output",
            str(tmp_path / "cli"),
        ],
    )
    assert response.exit_code == 0, response.output + repr(response.exception)
    with pytest.raises(ValueError, match="distinct"):
        compare_variants(paths[0], paths[0], tmp_path / "same")
    meta = inspect(paths[0])
    incomplete_summary = read_json(paths[0] / "summary.json")
    incomplete_summary["all_requested_cells_succeeded"] = False
    incomplete = publish(
        tmp_path / "incomplete",
        meta["artifact_kind"],
        meta["identity"]["config"],
        meta["identity"]["sources"],
        tables(paths[0]),
        incomplete_summary,
    )
    with pytest.raises(ValueError, match="incomplete"):
        read_suite(incomplete)
    # A corrupted completion claim is never accepted as a completed experiment.
    import shutil

    copied = tmp_path / "corrupt" / paths[0].name
    shutil.copytree(paths[0], copied)
    summary = read_json(copied / "summary.json")
    summary["all_requested_cells_succeeded"] = False
    write_json(copied / "summary.json", summary)
    with pytest.raises(ValueError, match="Modified"):
        read_suite(copied)


def test_agreement_reports_feature_space_changes_and_unmatched_fit_configs(
    variant_suites, tmp_path
):
    sources, paths = variant_suites
    a, b = [read_suite(p) for p in paths]
    pairs = normalize_pairs(pd.read_csv(pairing_csv(tmp_path, sources)), a, b)
    b.source = copy.deepcopy(b.source)
    b.source["feature_spaces"]["clip"]["preprocessing"] = (
        "alternate synthetic rendering"
    )
    result = compare_assignments(a, b, pairs)
    assert result["assignment_agreement"].all_points_ari.notna().all()
    assert result["assignment_agreement"].feature_space_equal.tolist().count(False) == 2
    b.children["clustering"]["runs"].loc[0, "parameters_json"] = '{"n_clusters":99}'
    result = compare_assignments(a, b, pairs)
    assert len(result["assignment_agreement"]) == 3
    assert result["run_matching"].configuration_match.tolist().count("left_only") == 1
    assert result["run_matching"].configuration_match.tolist().count("right_only") == 1


def test_compare_suites_without_review_reports_unavailable_components(tmp_path):
    config = SuiteConfig(
        reductions=(),
        clustering=(
            ClusterGrid(algorithm="agglomerative", parameters={"n_clusters": 2}),
        ),
    )
    paths = [
        suite(pixel_source(overlay), config, tmp_path / "suites")
        for overlay in (True, False)
    ]
    compared = compare_variants(*paths, tmp_path / "comparison")
    summary = read_json(compared / "summary.json")
    for side in ("left", "right"):
        availability = summary["component_availability"][side]
        assert availability["temporal_recall_visual_coherence"].startswith(
            "unavailable"
        )
        assert availability["recurrence_and_shortlist"].startswith("unavailable")
    assert summary["compared_recurrence_pairs"] == 0
    assert tables(compared)["candidate_shortlists"].empty
