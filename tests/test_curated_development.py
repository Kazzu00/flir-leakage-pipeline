"""Offline synthetic invariants for an explicitly non-admitted preparation."""

import csv
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml
from PIL import Image
from typer.testing import CliRunner

from flir_pipeline.detection.cli import app
from flir_pipeline.detection.common_executor import verify_preparation
from flir_pipeline.detection.curated_development import prepare_curated
from flir_pipeline.similarity.storage import file_sha256


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def fixture(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    rows = []
    for i in range(2):
        image, label = source / f"{i}.png", source / f"{i}.txt"
        Image.new("RGB", (16, 16), (i * 100, 20, 30)).save(image)
        label.write_text("0 0.5 0.5 0.2 0.2\n")
        rows.append(dict(frame_id=f"{i:064x}", content_id=file_sha256(image),
                         image_sha256=file_sha256(image), label_sha256=file_sha256(label),
                         image_path=image.name, label_path=label.name, width="16", height="16",
                         num_objects="1", label_empty="False",
                         **{f"{p}_split": ("train" if i == 0 else "val") for p in ("S1", "S2", "S3")}))
    manifest, matrix, provenance = [source / n for n in ("manifest.csv", "matrix.json", "provenance.json")]
    args = dict(source_root=source, manifest=manifest, matrix=matrix, provenance=provenance,
                output=tmp_path / "out", linux_root="/work/curated")
    write_json(matrix, {"state": "PROPOSED_NOT_ADMITTED_NOT_SUBMITTED", "runs": [
        dict(policy=p, split_seed=0, detector_seed=s, data_yaml=f"{p}_seed0/data.yaml")
        for p in ("S1", "S2", "S3") for s in (42, 43, 44)]})

    def save():
        with manifest.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        write_json(provenance, {"source_manifest_sha256": file_sha256(manifest)})

    save()
    return args, rows, save


def test_replay_pairing_and_no_admission(fixture):
    args, rows, _ = fixture
    before = {p: file_sha256(p) for p in args["source_root"].iterdir()}
    result = prepare_curated(**args)
    snapshots = {p: p.stat().st_mtime_ns for p in args["output"].rglob("*")}
    assert prepare_curated(**args) == result
    assert {p: p.stat().st_mtime_ns for p in snapshots} == snapshots
    assert {p: file_sha256(p) for p in before} == before
    assert result["ready_for_training"] is False
    assert result["protected_exclusion"]["status"] == "PENDING_REFERENCE_MISSING"
    second = prepare_curated(**{**args, "output": args["output"].with_name("replay")})
    assert second == result
    for p in ("S1", "S2", "S3"):
        config = yaml.safe_load((args["output"] / f"{p}_seed0/data.yaml").read_text())
        assert "test" not in config
        for split in ("train", "val"):
            line = (args["output"] / f"{p}_seed0/{split}.txt").read_text().strip()
            relative = Path(line.removeprefix("/work/curated/"))
            row = next(r for r in rows if r[f"{p}_split"] == split)
            assert file_sha256(args["output"] / relative) == row["image_sha256"]
            label = args["output"] / "canonical/labels" / (relative.stem + ".txt")
            assert file_sha256(label) == row["label_sha256"]
    with pytest.raises((OSError, ValueError, KeyError)):
        verify_preparation(args["output"])


@pytest.mark.parametrize("kind", ["duplicate", "traversal", "windows", "split", "label", "hash", "dimensions", "pairing", "content"])
def test_bad_manifest_rejected_before_output(fixture, kind):
    args, rows, save = fixture
    if kind == "duplicate":
        rows[1]["content_id"] = rows[0]["content_id"]
    else:
        key, value = {"traversal": ("image_path", "../outside.png"),
                      "windows": ("image_path", "C:/outside.png"),
                      "split": ("S1_split", "test"), "label": ("num_objects", "2"),
                      "hash": ("label_sha256", "a" * 64), "dimensions": ("width", "17"),
                      "pairing": ("label_path", "1.txt"), "content": ("content_id", "b" * 64)}[kind]
        rows[0][key] = value
    save()
    with pytest.raises(ValueError):
        prepare_curated(**args)
    assert not args["output"].exists()


@pytest.mark.parametrize("kind", ["matrix", "provenance", "linux", "partial", "tamper", "extra", "nested"])
def test_bindings_and_existing_output_fail_closed(fixture, kind):
    args, _, _ = fixture
    if kind == "matrix":
        write_json(args["matrix"], {"state": "APPROVED", "runs": []})
    elif kind == "provenance":
        write_json(args["provenance"], {"source_manifest_sha256": "bad"})
    elif kind == "linux":
        args["linux_root"] = "relative/path"
    elif kind == "nested":
        args["output"] = args["source_root"] / "generated"
    elif kind == "partial":
        args["output"].mkdir()
    else:
        prepare_curated(**args)
        target = "extra.txt" if kind == "extra" else "S1_seed0/train.txt"
        (args["output"] / target).write_text("tampered")
    with pytest.raises(ValueError):
        prepare_curated(**args)


@pytest.mark.parametrize("kind", ["valid", "overlap", "bad_receipt", "empty", "missing_pair"])
def test_protected_metadata(fixture, tmp_path, kind):
    args, rows, _ = fixture
    records, receipt = tmp_path / "protected.parquet", tmp_path / "receipt.json"
    data = {k: "f" * 64 for k in ("frame_id", "content_id", "image_sha256")}
    if kind == "overlap":
        data["image_sha256"] = rows[0]["image_sha256"]
    pd.DataFrame([] if kind == "empty" else [data], columns=list(data)).to_parquet(records)
    write_json(receipt, {"artifacts": {records.name: file_sha256(records) if kind != "bad_receipt" else "bad"}})
    args.update(protected_records=records, protected_receipt=None if kind == "missing_pair" else receipt)
    if kind == "valid":
        result = prepare_curated(**args)
        assert result["protected_exclusion"]["test_status"] == "NOT_APPROVED"
        assert result["protected_exclusion"]["overlap"] == 0
    else:
        with pytest.raises(ValueError):
            prepare_curated(**args)
        assert not args["output"].exists()


def test_cli(fixture):
    args, _, _ = fixture
    result = CliRunner().invoke(app, ["common-executor", "prepare-curated-development",
        *(str(args[k]) for k in ("source_root", "manifest", "matrix", "provenance", "output")),
        "--linux-root", args["linux_root"]])
    assert result.exit_code == 0, result.output
    assert "DEVELOPMENT_NOT_ADMITTED" in result.output
