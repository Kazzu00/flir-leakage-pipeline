import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from flir_pipeline.data.identity import dataset_id_from_manifest
from flir_pipeline.features.storage import verify_features_against_manifest
from flir_pipeline.features.visualization import discover_feature_directories


def test_full_coverage_checks_canonical_mapping_not_vector_uniqueness(tmp_path: Path) -> None:
    manifest = pd.DataFrame({"frame_id": ["a", "b", "c"], "content_id": ["x", "x", "y"], "image_sha256": ["x", "x", "y"], "label_sha256": ["l", "l", "l"]})
    # Different contents may legitimately share the same numerical embedding.
    raw = np.array([[1, 1], [1, 1]], dtype=np.float32)
    np.save(tmp_path / "embeddings_raw.npy", raw)
    np.save(tmp_path / "embeddings_l2.npy", raw / np.linalg.norm(raw, axis=1, keepdims=True))
    pd.DataFrame({"content_id": ["x", "y"], "embedding_row": [0, 1]}).to_parquet(tmp_path / "content_index.parquet")
    records = manifest[["frame_id", "content_id"]].assign(embedding_row=[0, 0, 1])
    records.to_parquet(tmp_path / "record_index.parquet")
    metadata = {"dataset_id": dataset_id_from_manifest(manifest), "total_records": 3, "unique_content_ids": 2, "selected_content_ids": 2, "embedding_dimension": 2, "model_revision": "a" * 40, "resolved_model_revision": "a" * 40}
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    result = verify_features_against_manifest(tmp_path, manifest)
    assert result["reproducible_full_dataset_valid"]
    metadata["extractor"] = "dinov2"
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    root = tmp_path / "selection"
    full = root / "dinov2" / "dataset" / "full"
    full.mkdir(parents=True)
    for name in ("metadata.json", "embeddings_raw.npy", "embeddings_l2.npy", "content_index.parquet", "record_index.parquet"):
        shutil.copyfile(tmp_path / name, full / name)
    sample = full.parent / "sample"
    sample.mkdir()
    (sample / "metadata.json").write_text(json.dumps({**metadata, "selected_content_ids": 1}))
    assert discover_feature_directories(root, full_manifest=manifest) == {"dinov2": full}
    with pytest.raises(ValueError, match="Multiple dinov2"):
        discover_feature_directories(root)
    shutil.copytree(full, full.parent / "another-full")
    with pytest.raises(ValueError, match="Multiple dinov2"):
        discover_feature_directories(root, full_manifest=manifest)
    metadata["resolved_model_revision"] = "unknown"
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    provenance = verify_features_against_manifest(tmp_path, manifest)
    assert provenance["full_dataset_valid"] and not provenance["reproducible_full_dataset_valid"]
    records.loc[0, ["content_id", "embedding_row"]] = ["y", 1]
    records.to_parquet(tmp_path / "record_index.parquet")
    result = verify_features_against_manifest(tmp_path, manifest)
    assert result["quality_valid"]  # Internally consistent, but wrong canonical occurrence.
    assert not result["full_record_content_coverage"]
    assert not result["full_dataset_valid"]
