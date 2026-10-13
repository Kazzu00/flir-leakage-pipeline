"""Synthetic offline tests of scientific identity and conservative evidence."""

import base64
import hashlib
import json
from io import BytesIO
from zipfile import BadZipFile, ZipFile

import pandas as pd
import pytest
from PIL import Image
from typer.testing import CliRunner

from flir_pipeline.cli import app
from flir_pipeline.data.adendo import audit_adendo
from flir_pipeline.data.identity import dataset_id_from_manifest
from flir_pipeline.data.manifest import build_manifest


def image_bytes(offset=0):
    image = Image.new("L", (32, 32))
    image.putdata(
        [min(255, (x * 7 + y * 3) % 200 + offset) for y in range(32) for x in range(32)]
    )
    stream = BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def write_zip(path, records):
    with ZipFile(path, "w") as archive:
        for name, payload in records.items():
            archive.writestr(name, payload)
    return path


@pytest.fixture
def sources(tmp_path):
    image = image_bytes()
    return (
        write_zip(
            tmp_path / "images.zip",
            {"images/train/a.png": image, "images/val/a.png": image},
        ),
        write_zip(
            tmp_path / "labels.zip",
            {
                "labels/training/a.txt": b"0 0.5 0.5 0.2 0.2\n",
                "different/validation/a.txt": b"",
                "labels/train/orphan.txt": b"1 0.5 0.5 0.1 0.1",
            },
        ),
        write_zip(
            tmp_path / "supplement.zip",
            {"imgs/b.png": image_bytes(1), "annotations/b.txt": b"1 0.5 0.5 0.3 0.3"},
        ),
    )


def records(output, name):
    return json.loads((output / name).read_text())["records"]


def test_consolidation_reproducible_read_only_and_historical_compatible(
    tmp_path, sources
):
    before = [path.read_bytes() for path in sources]
    output = tmp_path / "run1"
    first = audit_adendo(*sources, output)
    second = audit_adendo(*sources, tmp_path / "run2")
    assert first == second
    assert [path.read_bytes() for path in sources] == before
    assert first["total_occurrences"] == 3
    assert first["unique_contents"] == 2
    assert first["counts"]["historical"]["orphan_labels"] == 1
    assert first["counts"]["historical"]["empty_matched_labels"] == 1
    assert first["exact_annotation_conflict_pairs"] == 1
    assert first["exact_duplicate_groups"] == 1
    assert first["exact_duplicate_occurrences"] == 2
    assert first["exact_duplicate_additional_occurrences"] == 1
    assert first["exact_duplicate_relation_pairs"] == 1
    assert first["candidate_unique_pairs"] == 2
    assert first["candidate_evidence_records"] == 2
    assert first["phash_candidate_evidence_records"] == 2
    assert first["imported_candidate_evidence_records"] == 0
    assert first["relation_states"] == {"exact_bytes": 1, "visual_candidate": 2}
    manifest = pd.read_parquet(output / "candidate_manifest.parquet")
    assert dataset_id_from_manifest(manifest) == first["dataset_id"]
    assert manifest.loc[manifest.cohort == "adendo", "original_split"].isna().all()
    relations = records(output, "relations.json")
    assert (
        len(
            {
                row["right_frame_id"]
                for row in relations
                if row["state"] == "visual_candidate"
            }
        )
        == 2
    )
    assert all(not row["split_constraint_applied"] for row in relations)
    labels = records(output, "annotation_audit.json")
    for label in labels:
        assert (
            hashlib.sha256(base64.b64decode(label["original_bytes_base64"])).hexdigest()
            == label["label_sha256"]
        )
    build_manifest(
        sources[0],
        sources[1],
        tmp_path / "old.parquet",
        tmp_path / "old_reports",
        tmp_path,
    )
    old = pd.read_parquet(tmp_path / "old.parquet")
    assert set(old.frame_id) == set(
        manifest.loc[manifest.cohort == "historical", "frame_id"]
    )
    for name, digest in first["artifacts"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest


def test_invalid_missing_and_ambiguous_annotations_retained(tmp_path, sources):
    write_zip(
        sources[2],
        {
            "images/b.png": image_bytes(),
            "labels/b.txt": b"0 0.1 0.5 0.9 0.2",
            "other/b.txt": b"1 0.5 0.5 0.2 0.2",
            "images/c.png": image_bytes(2),
        },
    )
    summary = audit_adendo(*sources, tmp_path / "out")
    assert summary["counts"]["adendo"]["invalid_labels"] == 1
    assert summary["counts"]["adendo"]["pairing"] == {"ambiguous": 1, "missing": 1}
    assert not summary["annotation_qa_passed"]
    manifest = records(tmp_path / "out", "candidate_manifest.json")
    assert (
        len(next(row for row in manifest if row["basename"] == "b")["label_ids"]) == 2
    )
    assert len(records(tmp_path / "out", "annotation_audit.json")) == 5


def test_duplicate_groups_are_distinct_from_additional_occurrences(tmp_path, sources):
    duplicate = image_bytes()
    write_zip(
        sources[0],
        {
            "images/train/a.png": duplicate,
            "images/val/a.png": duplicate,
            "images/test/a.png": duplicate,
        },
    )
    write_zip(
        sources[1],
        {
            "labels/train/a.txt": b"",
            "labels/val/a.txt": b"",
            "labels/test/a.txt": b"",
        },
    )
    summary = audit_adendo(*sources, tmp_path / "out", phash_threshold=0)
    assert summary["exact_duplicate_groups"] == 1
    assert summary["exact_duplicate_occurrences"] == 3
    assert summary["exact_duplicate_additional_occurrences"] == 2
    assert summary["exact_duplicate_relation_pairs"] == 3


@pytest.mark.parametrize(
    "name",
    [
        "../escape.png",
        "/absolute.png",
        "C:/drive.png",
        "a/../escape.png",
        "a\\..\\escape.png",
    ],
)
def test_unsafe_paths_rejected_before_publication(tmp_path, sources, name):
    write_zip(sources[2], {name: image_bytes()})
    with pytest.raises(ValueError, match="Unsafe"):
        audit_adendo(*sources, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_duplicate_entries_and_corrupt_zip(tmp_path, sources):
    with ZipFile(sources[2], "w") as archive:
        archive.writestr("b.png", image_bytes())
        with pytest.warns(UserWarning):
            archive.writestr("b.png", image_bytes(1))
    with pytest.raises(ValueError, match="Ambiguous duplicate"):
        audit_adendo(*sources, tmp_path / "out")
    sources[2].write_bytes(b"not a ZIP")
    with pytest.raises(BadZipFile):
        audit_adendo(*sources, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_normalized_alias_duplicate_rejected(tmp_path, sources):
    write_zip(sources[2], {"a/b.png": image_bytes(), "a/./b.png": image_bytes(1)})
    with pytest.raises(ValueError, match="Ambiguous duplicate"):
        audit_adendo(*sources, tmp_path / "out")


def evidence_file(tmp_path, state="visual_candidate", **extra):
    path = tmp_path / "evidence.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "adendo_evidence_v1",
                "relations": [
                    {
                        "adendo_basename": "b",
                        "historical_basename": "a",
                        "historical_split": "train",
                        "state": state,
                        "source": "synthetic reviewer",
                        "method": "visual inspection",
                        "score": None,
                        **extra,
                    }
                ],
            }
        )
    )
    return path


@pytest.mark.parametrize(
    "state", ["visual_candidate", "unresolved", "confirmed_provenance", "rejected"]
)
def test_imported_evidence_states_and_no_automatic_constraints(
    tmp_path, sources, state
):
    evidence = evidence_file(
        tmp_path,
        state,
        reviewer="synthetic",
        reviewed_at="2026-01-01",
        additional_evidence="synthetic source alignment",
    )
    audit_adendo(*sources, tmp_path / "out", evidence=evidence, phash_threshold=0)
    imported = [
        row
        for row in records(tmp_path / "out", "relations.json")
        if row["evidence"].get("origin") == "imported_review"
    ]
    assert len(imported) == 1 and imported[0]["state"] == state
    assert imported[0]["annotation_difference"]["annotation_conflict"]
    assert not imported[0]["split_constraint_applied"]


@pytest.mark.parametrize(
    "state,extra",
    [
        ("confirmed_provenance", {}),
        ("exact_bytes", {}),
        ("visual_candidate", {"historical_split": "test"}),
        ("visual_candidate", {"score": float("nan")}),
    ],
)
def test_bad_review_rejected(tmp_path, sources, state, extra):
    with pytest.raises(ValueError):
        audit_adendo(
            *sources, tmp_path / "out", evidence=evidence_file(tmp_path, state, **extra)
        )
    assert not (tmp_path / "out").exists()


def test_undecodable_image_retained_and_cross_cohort_exact(tmp_path, sources):
    write_zip(
        sources[2],
        {"b.png": image_bytes(), "b.txt": b"", "bad.png": b"broken", "bad.txt": b""},
    )
    summary = audit_adendo(*sources, tmp_path / "out")
    assert summary["cross_cohort_exact_contents"] == 1
    assert not summary["annotation_qa_passed"]
    assert len(records(tmp_path / "out", "candidate_manifest.json")) == 4


def test_existing_output_refused_and_cli(tmp_path, sources):
    output = tmp_path / "out"
    args = [
        "data",
        "audit-adendo",
        "--images-zip",
        str(sources[0]),
        "--labels-zip",
        str(sources[1]),
        "--adendo-zip",
        str(sources[2]),
        "--output",
        str(output),
    ]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["total_occurrences"] == 3
    receipt = (output / "summary.json").read_bytes()
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 1
    assert (output / "summary.json").read_bytes() == receipt
