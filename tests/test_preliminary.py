"""Offline synthetic contracts for user-accepted BASE only; no model downloads."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml
from PIL import Image

from flir_pipeline.detection import preliminary as p
from flir_pipeline.similarity.storage import file_sha256, read_json, write_json


@pytest.fixture
def inputs(tmp_path):
    source = tmp_path / "curated"
    rows = []
    for i in range(4):
        frame = f"{i:064x}"
        image = source / f"canonical/images/{frame}.png"
        label = source / f"canonical/labels/{frame}.txt"
        image.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (24, 24), (i * 40, 10, 20)).save(image)
        label.write_text("0 0.5 0.5 0.2 0.2\n" if i < 2 else "", encoding="utf-8")
        rows.append(
            dict(
                frame_id=frame,
                content_id=file_sha256(image),
                image_sha256=file_sha256(image),
                label_sha256=file_sha256(label),
                image_path=image.relative_to(source).as_posix(),
                label_path=label.relative_to(source).as_posix(),
                width="24",
                height="24",
                num_objects="1" if i < 2 else "0",
                label_empty=str(i >= 2),
                **{
                    f"{x}_split": "train" if i % 2 == 0 else "val"
                    for x in ("S1", "S2", "S3")
                },
            )
        )
    p.write_rows(source / "manifest.csv", rows, list(rows[0]))
    receipt = dict(
        schema_version="curated_development_v1",
        artifacts=p.signatures(source),
        protected_exclusion=dict(
            status="EXACT_EXCLUSION_VERIFIED_AGAINST_REFERENCE", overlap=0
        ),
    )
    write_json(source / "development_receipt.json", receipt)
    weights = tmp_path / "synthetic.pt"
    weights.write_bytes(b"synthetic checkpoint; never deserialized")
    args = dict(
        source=source,
        receipt_sha256=file_sha256(source / "development_receipt.json"),
        config=Path("configs/detection/common_test_yolo11m_v2.yaml"),
        output=tmp_path / "pilot",
        weights_sha256=file_sha256(weights),
        accept_original_nonempty=True,
    )
    return args, weights


def prepared(inputs):
    args, weights = inputs
    p.prepare(**args)
    return args["output"], file_sha256(args["output"] / "receipt.json"), weights


def test_selection_preserves_splits_and_never_admits_final(inputs):
    args, _ = inputs
    before = p.signatures(args["source"])
    result = p.prepare(**args)
    assert result["counts"] == {"train": 1, "val": 1}
    assert not result["final_scientific_admission"]
    assert p.signatures(args["source"]) == before
    excluded = p.read_rows(args["output"] / "exclusions.csv")
    assert len(excluded) == 2
    assert all("not_assumed_wrong" in r["reason"] for r in excluded)


def test_acceptance_is_explicit(inputs):
    args, _ = inputs
    with pytest.raises(ValueError, match="explicit"):
        p.prepare(**{**args, "accept_original_nonempty": False})
    assert not args["output"].exists()


@pytest.mark.parametrize("change", ["label", "test", "receipt", "extra_receipt"])
def test_payload_tamper_rejected(inputs, change):
    package, sha, _ = prepared(inputs)
    if change == "label":
        next((package / "labels").rglob("*.txt")).write_text("")
    elif change == "test":
        (package / "data.yaml").write_text("test: /forbidden\n")
    elif change == "receipt":
        (package / "receipt.json").write_text("{}")
    else:
        (package / "extra").mkdir()
        (package / "extra/receipt.json").write_text("{}")
    with pytest.raises(ValueError):
        p.verify(package, sha)


def test_weights_mismatch_before_runtime(inputs, tmp_path):
    package, sha, weights = prepared(inputs)
    weights.write_bytes(b"different")
    with pytest.raises(ValueError, match="weights"):
        p.train(package, sha, weights, tmp_path / "attempt")
    assert not (tmp_path / "attempt").exists()


@pytest.fixture
def fake_runtime(monkeypatch):
    from flir_pipeline.detection import common_runtime, runtime

    monkeypatch.setattr(common_runtime, "preflight", lambda *args: {"synthetic": True})
    monkeypatch.setattr(runtime, "provenance", lambda: {"synthetic": True})

    class YOLO:
        drop = False
        fail = False

        def __init__(self, path):
            self.task = "detect"
            self.model = SimpleNamespace(yaml={"scale": "m"})
            self.callbacks = {}

        def add_callback(self, name, fn):
            self.callbacks[name] = fn

        def train(self, **kwargs):
            data = yaml.safe_load(Path(kwargs["data"]).read_text())
            assert "test" not in data and kwargs["split"] == "val"
            trainer = SimpleNamespace(
                args=SimpleNamespace(**kwargs), data=data, batch_size=kwargs["batch"]
            )
            for split, name in (("train", "train_loader"), ("val", "test_loader")):
                labels = [
                    dict(im_file=str(x), cls=[0])
                    for x in (Path(data["path"]) / "images" / split).glob("*.png")
                ]
                if self.drop:
                    labels = []
                setattr(
                    trainer,
                    name,
                    SimpleNamespace(dataset=SimpleNamespace(labels=labels)),
                )
            self.callbacks["on_train_start"](trainer)
            if self.fail:
                raise RuntimeError("synthetic interrupted")
            self.callbacks["on_train_batch_end"](trainer)
            out = Path(kwargs["project"]) / kwargs["name"]
            (out / "weights").mkdir(parents=True)
            for n in ("last.pt", "best.pt"):
                (out / "weights" / n).write_bytes(b"synthetic")
            pd.DataFrame(
                {"epoch": range(1, 51), "metrics/mAP50-95(B)": [0.1] * 50}
            ).to_csv(out / "results.csv", index=False)

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=YOLO))
    return YOLO


def test_runtime_development_only_and_immutable_inputs(inputs, tmp_path, fake_runtime):
    package, sha, weights = prepared(inputs)
    before = p.signatures(package)
    result = p.train(package, sha, weights, tmp_path / "attempt")
    assert result["state"] == "completed" and not result["test_used"]
    assert result["history"]["epochs_completed"] == 50
    assert p.signatures(package) == before
    with pytest.raises(ValueError, match="fresh"):
        p.train(package, sha, weights, tmp_path / "attempt")


@pytest.mark.parametrize("problem", ["drop", "fail"])
def test_runtime_failures_are_preserved(inputs, tmp_path, fake_runtime, problem):
    package, sha, weights = prepared(inputs)
    setattr(fake_runtime, problem, True)
    with pytest.raises((ValueError, RuntimeError)):
        p.train(package, sha, weights, tmp_path / "failed")
    assert (
        read_json(tmp_path / "failed/execution.json")["state"]
        == "failed_or_interrupted"
    )
