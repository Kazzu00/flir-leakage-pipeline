import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def test_independent_graph_rejects_omitted_visual_relation(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts/closeout/audit_outputs.py"
    spec = importlib.util.spec_from_file_location("independent_audit", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = pd.DataFrame(
        [
            {
                "frame_id": str(i),
                "content_id": f"c{i}",
                "source_frame_key": f"v:{i}",
                "archivo_video": "v",
                "segundo": i * 10,
                "nombre_original": f"f{i}.png",
                "dataset_origen": "fixture",
            }
            for i in range(3)
        ]
    )
    source.to_csv(tmp_path / "manifest.csv", index=False)
    source.assign(group_id=["g", "g", "h"]).to_csv(
        tmp_path / "preassignment_groups.csv", index=False
    )
    stores = {}
    for encoder, dimension in (("dinov2", 384), ("clip", 512)):
        directory = tmp_path / encoder
        directory.mkdir()
        vectors = np.zeros((3, dimension), dtype=np.float32)
        vectors[0:2, 0], vectors[2, 1] = 1, 1
        np.save(directory / "embeddings_raw.npy", vectors)
        np.save(directory / "embeddings_l2.npy", vectors)
        pd.DataFrame(
            {"content_id": source.content_id, "embedding_row": range(3)}
        ).to_parquet(directory / "content_index.parquet")
        stores[encoder] = {"path": str(directory)}
    stores_file = tmp_path / "stores.json"
    stores_file.write_text(json.dumps(stores))
    (tmp_path / "preassignment.json").write_text(
        json.dumps(
            {
                "configuration": {
                    "temporal_window_seconds": 2,
                    "cosine_thresholds": {"dinov2": 0.9, "clip": 0.9},
                },
                "features": {encoder: {"files": {}} for encoder in stores},
            }
        )
    )
    relations = pd.DataFrame(
        [
            {
                "left": "0",
                "right": "1",
                "kind": "visual_threshold",
                "evidence": f"{encoder} cosine >= 0.9",
            }
            for encoder in stores
        ]
    )
    relations.to_csv(tmp_path / "relations.csv", index=False)
    assert module.audit_graph(tmp_path, tmp_path, stores_file)["groups"] == 2
    relations.iloc[:1].to_csv(tmp_path / "relations.csv", index=False)
    with pytest.raises(ValueError, match="Dependency edges differ"):
        module.audit_graph(tmp_path, tmp_path, stores_file)
