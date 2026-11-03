"""Offline synthetic invariants for CPU provenance alignment, no FAC or models."""

import io
import json
import sqlite3
from fractions import Fraction
from zipfile import ZipFile

import numpy as np
import pytest
from PIL import Image
from typer.testing import CliRunner

from flir_pipeline.alignment.cli import app
from flir_pipeline.alignment.inventory import encoded, file_sha, inventory, select
from flir_pipeline.alignment.matching import Decoder, classify, features, support
from flir_pipeline.alignment.models import Config, Evidence, HumanVerification, Match
from flir_pipeline.alignment.pipeline import record_review, run, verify
from flir_pipeline.audit.demo import create_demo
from flir_pipeline.audit.store import Store

av = pytest.importorskip("av")
cv2 = pytest.importorskip("cv2")


def texture(seed=1):
    rng = np.random.default_rng(seed)
    image = np.zeros((216, 384, 3), dtype=np.uint8)
    for _ in range(100):
        x, y = rng.integers([0, 0], [384, 216])
        color = tuple(int(v) for v in rng.integers(30, 255, 3))
        cv2.circle(image, (int(x), int(y)), int(rng.integers(2, 12)), color, -1)
    return image


def png(image):
    stream = io.BytesIO()
    Image.fromarray(image).save(stream, format="PNG")
    return stream.getvalue()


def frame(rgb, second=1):
    return dict(
        rgb=rgb,
        pts=int(second * 1000),
        time_base="1/1000",
        second=second,
        duration_seconds=0.04,
        decoded_rgb_sha256="a" * 64,
        requested_seek_second=second,
        first_decoded_second=second,
    )


@pytest.mark.parametrize("change", ["scale", "noise", "tone"])
def test_support_known_transformations_and_negatives(change):
    original = texture()
    if change == "scale":
        transform = cv2.getRotationMatrix2D((192, 108), 2, 0.90)
        changed = cv2.warpAffine(original, transform, (384, 216))
    elif change == "noise":
        changed = np.clip(
            original.astype(float)
            + np.random.default_rng(2).normal(0, 3, original.shape),
            0,
            255,
        ).astype(np.uint8)
    else:
        changed = np.clip(original.astype(float) * 0.7 + 25, 0, 255).astype(np.uint8)
    query = features(original)
    assert support(query, frame(changed), Config(), "synthetic")["supported"]
    negative = support(
        query, frame(np.full_like(original, 127)), Config(), "negative_constant"
    )
    assert not negative["supported"]
    assert classify([negative], [])[0] == "not_found"


def test_competing_temporal_modes_are_ambiguous():
    image = texture()
    query = features(image)
    first = support(query, frame(image, 1), Config(), "synthetic")
    other = support(query, frame(image, 10), Config(), "synthetic_repeat")
    assert classify([first, other], [])[0] == "ambiguous"
    assert classify([first], [other])[0] == "ambiguous"


def write_video(path):
    with av.open(str(path), "w") as container:
        stream = container.add_stream("ffv1", rate=25)
        stream.width, stream.height, stream.pix_fmt = 384, 216, "yuv444p"
        for i in range(50):
            item = av.VideoFrame.from_ndarray(texture(i), format="rgb24")
            item.pts, item.time_base = i, Fraction(1, 25)
            for packet in stream.encode(item):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


@pytest.fixture
def inputs(tmp_path):
    root = tmp_path / "inputs"
    root.mkdir()
    write_video(root / "source.mkv")
    with ZipFile(root / "originals.zip", "w") as archive:
        for split in ("train", "val", "test"):
            archive.writestr(f"{split}/video_13min_001.png", png(texture(25)))
    return root


def test_pts_sequential_decode_seek_and_tolerance(inputs):
    decoder = Decoder(inputs / "source.mkv")
    try:
        frames = decoder.frames(0.11, 0.4, 0)
        assert abs(frames[0]["second"] - 0.12) < 1e-9
        assert frames[0]["second"] - 0.11 <= 0.04
        assert all(
            abs(f["second"] - f["pts"] * float(Fraction(f["time_base"]))) < 1e-12
            for f in frames
        )
        assert all(
            abs(b["second"] - a["second"] - 0.04) < 1e-9
            for a, b in zip(frames, frames[1:], strict=False)
        )
        again = decoder.frames(0.11, 0.4, 0)
        assert [(f["pts"], f["decoded_rgb_sha256"]) for f in again] == [
            (f["pts"], f["decoded_rgb_sha256"]) for f in frames
        ]
    finally:
        decoder.close()


def test_inventory_duplicate_archives_names_corrupt_and_sampling(inputs):
    with ZipFile(inputs / "copy.zip", "w") as archive:
        archive.writestr("train/video_13min_001.png", png(texture(25)))
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr("train/video_13min_001.png", png(texture(99)))
        archive.writestr("train/video_13min_002.png", b"not an image")
        archive.writestr("train/aug_video_13min_001.png", png(texture()))
        archive.writestr("labels/video_13min_001.txt", b"must not parse labels")
    (inputs / "bad.zip").write_bytes(b"broken")
    manifest = inventory(inputs, "source.mkv")
    rows = manifest["images"]
    assert len(rows) == 6
    assert len({r["image_id"] for r in rows}) == 6
    readable = [r for r in rows if r["read_status"] == "readable"]
    assert len({r["image_sha256"] for r in readable}) == 2
    assert all("same_index_different_bytes" in r["discrepancies"] for r in readable)
    assert {r["original_split"] for r in readable} == {"train", "val", "test"}
    assert all(r["image_dimensions"] == [384, 216] for r in readable)
    assert manifest["source_errors"]
    config = Config(selection_archive="originals.zip", per_split=1)
    assert select(manifest, config) == select(manifest, config)
    assert len(select(manifest, config)["selected"]) == 3
    assert select(manifest, config)["missing_diagnostic_indices"] == [
        199,
        200,
        349,
        350,
        399,
        400,
        499,
        500,
        501,
    ]


def test_real_pipeline_synthetic_integration_idempotence_and_ledger(inputs, tmp_path):
    source = tmp_path / "audit-input"
    create_demo(source)
    db = tmp_path / "audit.sqlite3"
    Store(db).import_directory(source)
    before = file_sha(db)
    with sqlite3.connect(db) as connection:
        old = connection.execute("SELECT payload FROM observations").fetchall()
    output = tmp_path / "result"
    config = Config(
        selection_archive="originals.zip",
        per_split=1,
        offset_hypothesis=0,
        local_radius=0.12,
        coarse_step=0.04,
        refine_radius=0.05,
        expanded_radius=0.5,
        expanded_step=0.1,
    )
    result = run(inputs, "source.mkv", output, config, audit_db=db)
    assert result["selected"] == 3
    assert result["human_verified"] == 0
    evidence = verify(output)
    assert all(c["top_k"] and c["status"] == "candidate" for c in evidence["cases"])
    hashes = {p.name: file_sha(p) for p in output.iterdir() if p.is_file()}
    assert run(inputs, "source.mkv", output, config, audit_db=db) == result
    assert hashes == {p.name: file_sha(p) for p in output.iterdir() if p.is_file()}
    assert file_sha(db) == before
    with sqlite3.connect(db) as connection:
        assert connection.execute("SELECT payload FROM observations").fetchall() == old
    case = evidence["cases"][0]
    review = {
        "schema": "alignment-human-review-v1",
        "alignment_id": case["alignment_id"],
        "status": "human_verified",
        "reviewer": "Synthetic test reviewer",
        "justification": "Synthetic known source; not FAC annotation",
        "evidence_consulted": [case["evidence_locator"]],
        "selected_pts": case["top_k"][0]["pts"],
        "source_locator": "source.mkv",
    }
    target = tmp_path / "review.json"
    record_review(output, review, target)
    assert record_review(output, review, target)["recorded"]
    with pytest.raises(ValueError):
        record_review(
            output,
            {**review, "evidence_consulted": ["invented.jpg"]},
            tmp_path / "bad-review.json",
        )
    assert CliRunner().invoke(app, ["verify", str(output)]).exit_code == 0
    from jsonschema import Draft202012Validator

    Draft202012Validator(Evidence.model_json_schema()).validate(evidence)
    (output / "evidence.csv").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        verify(output)


@pytest.mark.parametrize(
    "override", [{"reviewer": " "}, {"evidence_consulted": []}, {"selected_pts": None}]
)
def test_reject_unsubstantiated_human_verification(override):
    with pytest.raises(ValueError):
        HumanVerification.model_validate(
            {
                "schema": "alignment-human-review-v1",
                "alignment_id": "synthetic",
                "status": "human_verified",
                "reviewer": "Reviewer",
                "justification": "References",
                "source_locator": "source.mkv",
                "selected_pts": 1,
                "evidence_consulted": ["sheet.jpg"],
                **override,
            }
        )


def test_budgets_missing_video_and_sample_only(inputs, tmp_path):
    config = Config(
        selection_archive="originals.zip",
        per_split=1,
        offset_hypothesis=0,
        max_decoded_frames=1,
    )
    result = run(inputs, "source.mkv", tmp_path / "bounded", config)
    assert result["budget_exhausted"]
    assert sum(r["not_found"] for r in result["by_split"].values()) == 3
    plan = run(inputs, "source.mkv", tmp_path / "plan", config, sample_only=True)
    assert plan["sample_only"]
    assert not (tmp_path / "plan" / "sheets").exists()
    with pytest.raises(OSError):
        run(inputs, "missing.mkv", tmp_path / "missing", config)
    with pytest.raises(ValueError):
        run(inputs, "source.mkv", inputs / "forbidden", config)
    (inputs / "bad.mkv").write_bytes(b"unsupported video")
    with pytest.raises(Exception, match="Invalid data"):
        Decoder(inputs / "bad.mkv")


def test_schema_committed_matches_model():
    from pathlib import Path

    schema = json.loads(
        Path("docs/contracts/alignment-evidence-v1.schema.json").read_bytes()
    )
    assert schema["properties"] == Evidence.model_json_schema()["properties"]
    assert encoded(schema)


def test_reject_inconsistent_pts():
    value = support(features(texture()), frame(texture()), Config(), "synthetic")
    with pytest.raises(ValueError, match="PTS/time-base"):
        Match.model_validate({**value, "second": 100})
