import pandas as pd

from flir_pipeline.closeout import anchors
from flir_pipeline.closeout.io import read_json


def test_native_extensionless_names_are_candidates_only(tmp_path, monkeypatch):
    dataset, publication = tmp_path / "dataset", tmp_path / "publication"
    dataset.mkdir()
    (publication / "sources/audit").mkdir(parents=True)
    (publication / "sources/proposal").mkdir()
    for root in (dataset, publication):
        (root / "receipt.json").write_text("{}")
    pd.DataFrame(
        [
            {"basename": "one", "frame_id": "p0"},
            {"basename": "two", "frame_id": "a"},
            {"basename": "two", "frame_id": "b"},
        ]
    ).to_parquet(publication / "sources/audit/candidate_manifest.parquet")
    pd.DataFrame({"frame_id": [f"p{i}" for i in range(372)]}).to_parquet(
        publication / "sources/proposal/test_records.parquet"
    )
    frame = pd.DataFrame(
        [
            {
                "frame_id": f"d{i}",
                "nombre_original": name,
                "dataset_origen": "reported",
                "source_frame_key": f"v:{i}",
            }
            for i, name in enumerate(["one.jpg", "two.png", "three.jpg"])
        ]
    )
    monkeypatch.setattr(anchors, "load_manifest", lambda *args: frame)
    monkeypatch.setattr(anchors, "verify_fixed_test", lambda *args: {})
    result = anchors.audit(dataset, publication, tmp_path / "out")
    assert result["status_counts"] == {"NOMINAL_ONLY": 1, "AMBIGUOUS": 1, "UNMAPPED": 1}
    assert result["nominal_test_candidates"] == 1
    assert result["verified_anchors"] == 0
    assert (
        read_json(tmp_path / "out/receipt.json")["state"] == "BLOCKED_PROTECTED_FREEZE"
    )
