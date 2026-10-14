"""Synthetic, offline protected review: original bytes and no scientific promotion."""

import csv
import hashlib
import io
import zipfile

import numpy as np
import pytest
from PIL import Image
from test_common_test import bundle as bundle
from typer.testing import CliRunner

from flir_pipeline.cli import app
from flir_pipeline.data.adendo import audit_adendo
from flir_pipeline.detection.annotation_review import (
    build_review,
    differences,
    project_box,
    source_bytes,
    validate_decisions,
    verify_review,
)
from flir_pipeline.detection.common_test import CommonTestConfig
from flir_pipeline.detection.common_test_storage import build_common_test, signatures
from flir_pipeline.similarity.storage import file_sha256, read_json, write_json


@pytest.fixture
def audited(tmp_path):
    def png(seed):
        stream = io.BytesIO()
        Image.fromarray(
            np.random.default_rng(seed).integers(0, 256, (32, 32), dtype=np.uint8)
        ).save(stream, format="PNG")
        return stream.getvalue()

    with (
        zipfile.ZipFile(tmp_path / "images.zip", "w") as images,
        zipfile.ZipFile(tmp_path / "labels.zip", "w") as labels,
    ):
        for i in range(30):
            split = ("train", "val", "test")[i % 3]
            images.writestr(f"{split}/h{i}.png", png(i))
            labels.writestr(f"{split}/h{i}.txt", f"{i % 5} 0.5 0.5 0.2 0.2\n")
        images.writestr("train/h2.png", png(2))
        labels.writestr("train/h2.txt", "")
    with zipfile.ZipFile(tmp_path / "adendo.zip", "w") as archive:
        archive.writestr("a.png", png(100))
        archive.writestr("a.txt", "0 0.5 0.5 0.2 0.2\n")
    audit = tmp_path / "audit"
    audit_adendo(
        tmp_path / "images.zip",
        tmp_path / "labels.zip",
        tmp_path / "adendo.zip",
        audit,
        phash_threshold=0,
    )
    return audit, tmp_path, None


@pytest.fixture
def inputs(bundle, tmp_path):
    proposal = tmp_path / "proposal"
    build_common_test(*bundle, proposal, CommonTestConfig())
    acceptance = tmp_path / "acceptance"
    acceptance.mkdir()
    audit = bundle[0]
    write_json(
        acceptance / "independent_acceptance_audit.json",
        {
            "source": {
                "audit_summary_sha256": file_sha256(audit / "summary.json"),
                "internal_summary_sha256": file_sha256(proposal / "summary.json"),
            }
        },
    )
    records = read_json(audit / "candidate_manifest.json")["records"]
    pair = [r for r in records if r["basename"] == "h2"]
    a = next(r for r in pair if r["original_split"] == "test")
    b = next(r for r in pair if r["original_split"] == "train")
    row = {
        "content_id": a["content_id"],
        "test_frame_id": a["frame_id"],
        "quarantine_frame_id": b["frame_id"],
        "test_label_sha256": a["label_sha256"],
        "quarantine_label_sha256": b["label_sha256"],
    }
    with (acceptance / "protected_annotation_conflicts.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    return audit, proposal, acceptance, tmp_path


def test_source_recovery_and_bad_checksum(tmp_path):
    with zipfile.ZipFile(tmp_path / "raw.zip", "w") as archive:
        archive.writestr("labels/raw.txt", b"0 .5 .5 .2 .2\r\n")
    locator = {
        "source_archive": "raw.zip",
        "zip_member_name": "labels/raw.txt",
        "zip_header_offset": 0,
    }
    raw = b"0 .5 .5 .2 .2\r\n"
    assert source_bytes(tmp_path, locator, hashlib.sha256(raw).hexdigest()) == raw
    with pytest.raises(ValueError, match="checksum"):
        source_bytes(tmp_path, locator, "0" * 64)
    with pytest.raises(ValueError, match="locator"):
        source_bytes(
            tmp_path,
            {**locator, "zip_header_offset": 1},
            hashlib.sha256(raw).hexdigest(),
        )
    with pytest.raises(ValueError, match="Unsafe"):
        source_bytes(tmp_path, {**locator, "source_archive": "../raw.zip"}, "")


def test_projection_preserves_fractional_geometry():
    assert project_box((2, 0.5, 0.4, 0.2, 0.1), 200, 100) == pytest.approx(
        (80, 35, 120, 45)
    )


def test_differences_do_not_invent_correspondences():
    box = (0, 0.5, 0.5, 0.2, 0.2)
    assert differences([box, box], [box])["only_A_indices"] == [1]
    absent = differences([], [box])
    assert (
        absent["annotation_presence_difference"]
        and absent["real_object_absence"] == "not_established"
    )
    changed = differences([box], [(1, *box[1:])])
    assert changed["unambiguous_comparisons"][0]["kind"] == "class_change_same_geometry"
    geom = differences([box], [(0, 0.51, 0.5, 0.2, 0.2)])
    assert geom["unambiguous_comparisons"][0]["delta_xywh_B_minus_A"][
        0
    ] == pytest.approx(0.01)
    ambiguous = differences([box], [(0, 0.4, 0.5, 0.1, 0.2), (0, 0.6, 0.5, 0.1, 0.2)])
    assert (
        ambiguous["ambiguous_correspondence"]
        and not ambiguous["unambiguous_comparisons"]
    )
    assert differences([box, box], [(1, *box[1:])])["ambiguous_correspondence"]


def test_package_replays_and_sources_unchanged(inputs, tmp_path):
    before = [signatures(p) for p in inputs[:3]]
    first, second = (
        tmp_path.parent / (tmp_path.name + "-out1"),
        tmp_path.parent / (tmp_path.name + "-out2"),
    )
    manifest = build_review(*inputs, first, expected_cases=1)
    build_review(*inputs, second, expected_cases=1)
    assert signatures(first) == signatures(second)
    assert before == [signatures(p) for p in inputs[:3]]
    result = verify_review(first, *inputs)
    assert (
        result["reproduced"]
        and result["pending"] == 1
        and not result["ready_for_training"]
    )
    case = manifest["cases"][0]
    assert (first / f"cases/{case['conflict_id']}/B.txt").read_bytes() == b""
    with Image.open(first / case["comparison"]) as image:
        assert image.width == 1920
    decision = read_json(first / "decisions.template.json")
    row = decision["cases"][0]
    assert (
        row["reviewer"]
        == row["review_date"]
        == row["justification"]
        == row["additional_evidence"]
        == ""
    )
    row["decision"] = "conservar_anotacion_A"
    with pytest.raises(ValueError, match="Incomplete"):
        validate_decisions(decision, manifest)
    row.update(
        reviewer="Synthetic reviewer",
        review_date="2026-10-13",
        justification="Synthetic reason",
        additional_evidence="No additional evidence",
    )
    assert not validate_decisions(decision, manifest)["approval_granted"]
    decision["review_id"] = "foreign"
    with pytest.raises(ValueError, match="another review"):
        validate_decisions(decision, manifest)
    # Even repaired file receipts cannot turn an evidence package into approval.
    manifest["ready_for_training"] = True
    write_json(first / "manifest.json", manifest)
    receipt = read_json(first / "receipt.json")
    receipt["artifacts"]["manifest.json"] = file_sha256(first / "manifest.json")
    write_json(first / "receipt.json", receipt)
    with pytest.raises(ValueError, match="reproduction"):
        verify_review(first, *inputs)


@pytest.mark.parametrize("corruption", ["archive", "acceptance", "count", "output"])
def test_fail_closed(inputs, tmp_path, corruption):
    output = tmp_path.parent / (tmp_path.name + "-bad")
    if corruption == "archive":
        with (inputs[3] / "labels.zip").open("ab") as stream:
            stream.write(b"changed")
    elif corruption == "acceptance":
        report = read_json(inputs[2] / "independent_acceptance_audit.json")
        report["source"]["internal_summary_sha256"] = "wrong"
        write_json(inputs[2] / "independent_acceptance_audit.json", report)
    elif corruption == "output":
        output = inputs[0] / "forbidden"
    with pytest.raises(ValueError):
        build_review(*inputs, output, expected_cases=8 if corruption == "count" else 1)
    assert not (output / "receipt.json").exists()


@pytest.mark.parametrize(
    "command",
    [
        "protected-annotation-review",
        "verify-annotation-review",
        "validate-annotation-decisions",
    ],
)
def test_cli_help(command):
    assert CliRunner().invoke(app, ["detection", command, "--help"]).exit_code == 0


def test_missing_or_duplicate_decisions():
    manifest = {"review_id": "review", "cases": [{"conflict_id": "one"}]}
    with pytest.raises(ValueError, match="Missing"):
        validate_decisions({"review_id": "review", "cases": []}, manifest)
    row = {
        "conflict_id": "one",
        "reviewer": "",
        "review_date": "",
        "justification": "",
        "additional_evidence": "",
        "decision": "APPROVED",
    }
    with pytest.raises(ValueError, match="Invalid"):
        validate_decisions({"review_id": "review", "cases": [row]}, manifest)
