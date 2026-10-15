"""Synthetic ground-truth evidence: counts, raw bytes, blocked states and replay."""

import copy
import io

import pandas as pd
import pytest
from PIL import Image
from test_annotation_review import audited as original_audited
from test_annotation_review import inputs as inputs
from test_common_test import bundle as bundle
from typer.testing import CliRunner

from flir_pipeline.cli import app
from flir_pipeline.data.adendo import audit_adendo
from flir_pipeline.detection.annotation_review import build_review
from flir_pipeline.detection.common_test_storage import signatures
from flir_pipeline.detection.ground_truth_gallery import render_case
from flir_pipeline.detection.ground_truth_quality import (
    annotation_facts,
    annotation_sensitivity,
    descriptive_support,
    perceptual_relations,
    review_template,
    validate_human_review,
)
from flir_pipeline.detection.ground_truth_quality_storage import (
    build_quality,
    verify_quality,
)
from flir_pipeline.similarity.storage import read_json, write_json
from flir_pipeline.splitting.fixed_test import FixedTestConfig
from flir_pipeline.splitting.fixed_test_storage import build_fixed_test


@pytest.fixture
def audited(tmp_path):
    _, root, _ = original_audited.__wrapped__(tmp_path)
    evidence = root / "prior_evidence.json"
    write_json(evidence, {"schema_version": "adendo_evidence_v1", "relations": []})
    output = root / "audit-with-evidence"
    audit_adendo(
        root / "images.zip",
        root / "labels.zip",
        root / "adendo.zip",
        output,
        evidence=evidence,
        phash_threshold=0,
    )
    return output, root, evidence


def test_annotation_precision_geometry_and_empty():
    raw = b"0 0.5 0.5 0.123456789012 0.2\n4 0.5 0.5 1 1\n"
    facts = annotation_facts(raw, 100, 200)
    assert facts["box_count"] == 2 and facts["classes"] == [0, 4]
    assert facts["boxes"][0]["w"] == 0.123456789012
    assert facts["boxes"][1]["area_pixels"] == 20000
    assert facts["boxes"][1]["touches_boundary"]
    empty = annotation_facts(b"\n", 10, 20)
    assert empty["empty"] and empty["boxes"] == [] and empty["area_max"] is None
    assert not facts["automatic_missing_object_detection"]


@pytest.mark.parametrize(
    "raw",
    [
        b"0 .1 .1 1 1",
        b"0 nan .5 .2 .2",
        b"5 .5 .5 .2 .2",
        b"-1 .5 .5 .2 .2",
        b"0 .5 .5 0 .2",
    ],
)
def test_invalid_labels_fail_without_repair(raw):
    with pytest.raises(ValueError):
        annotation_facts(raw, 100, 100)


def test_support_and_sensitivity_are_not_effective_sample_sizes():
    records = pd.DataFrame({"frame_id": ["a", "b", "c"], "content_id": ["x", "x", "y"]})
    stats = records.assign(
        class_0=[1, 2, 3], class_1=0, class_2=0, class_3=0, class_4=[1, 0, 0]
    )
    edge = {
        "left_frame_id": "a",
        "right_frame_id": "c",
        "left_content_id": "x",
        "right_content_id": "y",
        "state": "visual_candidate",
        "relation_id": "e",
    }
    rows, _, scenarios, mapping, _ = descriptive_support(
        records, stats, [edge], "synthetic"
    )
    assert (rows[0]["instances"], rows[0]["images"], rows[0]["contents"]) == (6, 3, 2)
    assert rows[0]["candidate_positive_groups"] == 1
    assert rows[0]["candidate_max_instance_share"] == 1
    assert (
        rows[4]["single_positive_image"]
        and not rows[4]["checkpoint_selection_eligible"]
    )
    assert rows[1]["instances"] == 0 and not rows[1]["coverage_positive"]
    assert any(
        s["level"] == "candidate"
        and s["class_id"] == 0
        and s["remaining_instances"] == 0
        for s in scenarios
    )
    assert mapping.candidate_group.nunique() == 1
    # Outside occurrences must not introduce a path joining included contents.
    bridges = [
        {**edge, "right_frame_id": "outside", "right_content_id": "z"},
        {
            **edge,
            "left_frame_id": "outside",
            "left_content_id": "z",
            "relation_id": "e2",
        },
    ]
    rows, *_ = descriptive_support(records, stats, bridges, "synthetic")
    assert rows[0]["candidate_positive_groups"] == 2


def test_pending_and_incomplete_human_observations():
    form = review_template("audit", ["a"])
    assert validate_human_review(form, "audit", ["a"])["pending"] == 1
    assert not validate_human_review(form, "audit", ["a"])["adjudicated_labels_applied"]
    for state in ("observaciones_registradas", "propuesta_de_correccion", "APPROVED"):
        changed = copy.deepcopy(form)
        changed["cases"][0]["state"] = state
        with pytest.raises(ValueError):
            validate_human_review(changed, "audit", ["a"])


def test_within_population_screening_is_separate_and_never_provenance():
    records = pd.DataFrame(
        {
            "frame_id": ["a", "b", "c"],
            "content_id": ["x", "y", "z"],
            "phash": ["0000000000000000", "0000000000000001", "ffffffffffffffff"],
        }
    )
    stats = records[["frame_id", "content_id"]].assign(
        class_0=1, class_1=0, class_2=0, class_3=0, class_4=0
    )
    before = records.copy(deep=True)
    edges = perceptual_relations(records, 8)
    assert len(edges) == 1 and edges[0]["hamming_distance"] == 1
    assert (
        not edges[0]["confirmed_provenance"]
        and not edges[0]["used_to_construct_splits"]
    )
    rows, _, _, _, _ = descriptive_support(
        records, stats, [], "test", phash_threshold=8
    )
    assert rows[0]["candidate_positive_groups"] == 3
    assert rows[0]["perceptual_positive_groups"] == 2
    pd.testing.assert_frame_equal(records, before)
    assert perceptual_relations(records.iloc[[0, 2]], 8) == []
    with pytest.raises(ValueError, match="pHash"):
        perceptual_relations(records.assign(phash=None), 8)


def test_alternative_annotation_counts_never_apply_labels():
    support = [{"class_id": c, "instances": 12 if c == 0 else 0} for c in range(5)]
    cases = [
        {
            "conflict_id": "example",
            "A": {"boxes": [[0, 0.5, 0.5, 0.5, 0.5]]},
            "B": {"boxes": [[0, 0.3, 0.3, 0.2, 0.2], [0, 0.7, 0.7, 0.2, 0.2]]},
        }
    ]
    before = copy.deepcopy(cases)
    rows = annotation_sensitivity(support, cases)
    assert (
        rows[0]["current_test_instances"] == 12
        and rows[0]["hypothetical_instances"] == 13
    )
    assert all(not r["applied"] for r in rows) and cases == before


def test_gallery_preserves_raw_bytes_and_uses_same_native_orientation(tmp_path):
    image = Image.new("RGB", (40, 20), "black")
    exif = Image.Exif()
    exif[274] = 6
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", exif=exif)
    raw, label = buffer.getvalue(), b"0 .5 .5 .5 .5\n"
    record = {
        "frame_id": "synthetic",
        "basename": "example",
        "width": 40,
        "height": 20,
        "source_archive": "source.zip",
        "zip_member_name": "example.jpg",
        "image_sha256": "synthetic-image",
        "label_sha256": "synthetic-label",
    }
    render_case(tmp_path, raw, label, record, annotation_facts(label, 40, 20), None)
    folder = tmp_path / "cases/synthetic"
    assert (folder / "original.jpg").read_bytes() == raw
    assert (folder / "original.txt").read_bytes() == label
    with (
        Image.open(folder / "unannotated.png") as plain,
        Image.open(folder / "annotated.png") as marked,
    ):
        assert plain.size == marked.size == (40, 20)
        assert not plain.getexif() and not marked.getexif()
        assert plain.getpixel((0, 0)) == (0, 0, 0)


def test_package_reproduction_and_integrity(inputs, bundle, tmp_path):
    audit, proposal, acceptance, root = inputs
    evidence = root / "prior_evidence.json"
    review = tmp_path.parent / (tmp_path.name + "-review")
    build_review(audit, proposal, acceptance, root, review, expected_cases=1)
    partitions = tmp_path.parent / (tmp_path.name + "-partitions")
    build_fixed_test(
        audit, bundle[1], proposal, review, partitions, FixedTestConfig(node_limit=2)
    )
    output = tmp_path.parent / (tmp_path.name + "-quality")
    with pytest.raises(ValueError, match="Original evidence checksum"):
        build_quality(
            audit,
            proposal,
            review,
            partitions,
            root,
            output,
            expected_test=10,
            expected_arms=6,
        )
    before = [signatures(p) for p in (audit, proposal, review, partitions)]
    result = build_quality(
        audit,
        proposal,
        review,
        partitions,
        root,
        output,
        expected_test=10,
        expected_arms=6,
        evidence=evidence,
    )
    assert result["test_images"] == 10 and result["pending_conflicts"] == 1
    assert result["historically_exposed_test_contents"] == 1
    assert result["test_state"] == "PROPOSED" and not result["ready_for_training"]
    assert len(list((output / "cases").glob("*/index.html"))) == 10
    assert signatures(review) == signatures(output / "prior_review")
    assert len(pd.read_csv(output / "support.csv")) == 65
    assert verify_quality(
        output, audit, proposal, review, partitions, root, evidence=evidence
    )["reproduced"]
    assert before == [signatures(p) for p in (audit, proposal, review, partitions)]
    receipt = read_json(output / "receipt.json")
    manifest = read_json(output / "manifest.json")
    write_json(output / "manifest.json", {**manifest, "ready_for_training": True})
    with pytest.raises(ValueError, match="checksum"):
        verify_quality(
            output, audit, proposal, review, partitions, root, evidence=evidence
        )
    hashes = signatures(output)
    hashes.pop("receipt.json")
    write_json(output / "receipt.json", {**receipt, "artifacts": hashes})
    with pytest.raises(ValueError, match="reproduction"):
        verify_quality(
            output, audit, proposal, review, partitions, root, evidence=evidence
        )
    with pytest.raises(ValueError, match="external output"):
        build_quality(audit, proposal, review, partitions, root, output)
    # Full archive verification must reject changed original ZIPs before publication.
    archive = next(root.glob("*.zip"))
    original = archive.read_bytes()
    try:
        archive.write_bytes(original + b"tampering")
        with pytest.raises(ValueError, match="checksum"):
            build_quality(
                audit,
                proposal,
                review,
                partitions,
                root,
                tmp_path.parent / (tmp_path.name + "-bad-source"),
                expected_test=10,
                expected_arms=6,
            )
    finally:
        archive.write_bytes(original)


@pytest.mark.parametrize(
    "command",
    [
        "ground-truth-quality",
        "verify-ground-truth-quality",
        "validate-ground-truth-review",
    ],
)
def test_cli_surface(command):
    assert CliRunner().invoke(app, ["detection", command, "--help"]).exit_code == 0
