import json
from pathlib import Path

import pandas as pd

from flir_pipeline.data.identity import dataset_id_from_manifest


def test_identity_ignores_ambient_reports_and_historical_split(tmp_path: Path, monkeypatch) -> None:
    frame = pd.DataFrame([{
        "frame_id": "occurrence", "image_sha256": "image", "label_sha256": "label",
        "original_split": "train", "manifest_version": "synthetic-v1",
    }])
    expected = dataset_id_from_manifest(frame)
    ambient = tmp_path / "reports/data_manifest"
    ambient.mkdir(parents=True)
    (ambient / "manifest_metadata.json").write_text(json.dumps({"dataset_id": "unrelated"}))
    monkeypatch.chdir(tmp_path)
    frame["original_split"] = "test"
    assert dataset_id_from_manifest(frame) == expected
