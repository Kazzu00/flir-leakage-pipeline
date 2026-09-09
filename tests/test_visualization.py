from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from flir_pipeline.features.visualization import (
    generate_feature_engineering_report,
    visualize_embedding_health,
)


def _manifest_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "frame_id": "f1",
                "content_id": "c1",
                "original_split": "train",
                "image_sha256": "sha1",
                "label_sha256": "lab1",
                "label_exists": True,
                "label_empty": False,
                "num_objects": 2,
                "classes_present": "0|1",
                "bbox_area_mean": 0.2,
                "bbox_area_min": 0.1,
                "bbox_area_max": 0.3,
                "width": 640,
                "height": 512,
                "aspect_ratio": 1.25,
                "pixel_mean": 0.5,
                "pixel_std": 0.2,
                "entropy": 5.1,
                "laplacian_variance": 100.0,
            },
            {
                "frame_id": "f2",
                "content_id": "c1",
                "original_split": "train",
                "image_sha256": "sha1",
                "label_sha256": "lab1",
                "label_exists": True,
                "label_empty": False,
                "num_objects": 2,
                "classes_present": "0|1",
                "bbox_area_mean": 0.2,
                "bbox_area_min": 0.1,
                "bbox_area_max": 0.3,
                "width": 640,
                "height": 512,
                "aspect_ratio": 1.25,
                "pixel_mean": 0.5,
                "pixel_std": 0.2,
                "entropy": 5.1,
                "laplacian_variance": 100.0,
            },
            {
                "frame_id": "f3",
                "content_id": "c2",
                "original_split": "test",
                "image_sha256": "sha2",
                "label_sha256": "lab2",
                "label_exists": True,
                "label_empty": True,
                "num_objects": 0,
                "classes_present": "",
                "bbox_area_mean": float("nan"),
                "bbox_area_min": float("nan"),
                "bbox_area_max": float("nan"),
                "width": 1280,
                "height": 720,
                "aspect_ratio": 1.7777777777777777,
                "pixel_mean": 0.7,
                "pixel_std": 0.3,
                "entropy": 7.1,
                "laplacian_variance": 250.0,
            },
        ]
    )


def _diagnostics_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "content_id": "c1",
                "width": 640,
                "height": 512,
                "aspect_ratio": 1.25,
                "pixel_mean": 0.5,
                "pixel_std": 0.2,
                "entropy": 5.1,
                "laplacian_variance": 100.0,
            },
            {
                "content_id": "c2",
                "width": 1280,
                "height": 720,
                "aspect_ratio": 1.7777777777777777,
                "pixel_mean": 0.7,
                "pixel_std": 0.3,
                "entropy": 7.1,
                "laplacian_variance": 250.0,
            },
        ]
    )


def test_generate_feature_engineering_report_creates_outputs(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.parquet"
    diagnostics_path = tmp_path / "diagnostics.parquet"
    output_dir = tmp_path / "report"
    _manifest_df().to_parquet(manifest_path, index=False)
    _diagnostics_df().to_parquet(diagnostics_path, index=False)

    result = generate_feature_engineering_report(
        manifest_path=manifest_path,
        diagnostics_path=diagnostics_path,
        output_dir=output_dir,
        feature_dirs={},
    )

    assert (output_dir / "figures").is_dir()
    assert (output_dir / "tables").is_dir()
    assert (output_dir / "feature_engineering_report.md").is_file()
    assert (output_dir / "tables" / "dataset_summary.csv").is_file()
    assert result["metadata"]["dataset_id"]
    assert "absolute" not in json.dumps(result["metadata"]).lower()
    values = pd.read_csv(output_dir / "tables" / "dataset_summary.csv").to_numpy(dtype=float)
    assert np.isfinite(values).all()


def test_visualize_embedding_health_handles_arbitrary_dimensions(tmp_path: Path) -> None:
    feature_dir = tmp_path / "feature_space"
    feature_dir.mkdir()
    raw = np.random.default_rng(0).normal(size=(8, 11)).astype(np.float32)
    np.save(feature_dir / "embeddings_raw.npy", raw)
    np.save(feature_dir / "embeddings_l2.npy", raw / np.linalg.norm(raw, axis=1, keepdims=True))
    (feature_dir / "metadata.json").write_text(
        json.dumps({
            "feature_space_id": "abc123",
            "extractor": "dinov2",
            "embedding_dimension": 11,
            "selected_content_ids": 8,
            "dataset_id": "demo",
        }),
        encoding="utf-8",
    )

    result = visualize_embedding_health(feature_dir, tmp_path / "embeddings_report", label="smoke")

    assert result["label"] == "smoke"
    assert (tmp_path / "embeddings_report" / "figures" / "embedding_raw_norm_distribution_dinov2_smoke.png").exists()
    assert result["stats"]["dimension_std_min"] >= 0.0
    assert np.isfinite(result["stats"]["dimension_std_median"]).all()
    assert result["stats"]["l2_mean_near_one"]


def test_generate_feature_engineering_report_handles_empty_inputs(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.parquet"
    diagnostics_path = tmp_path / "diagnostics.parquet"
    output_dir = tmp_path / "report"
    pd.DataFrame(columns=["content_id", "frame_id", "original_split"]).to_parquet(manifest_path, index=False)
    pd.DataFrame(columns=["content_id", "width", "height", "aspect_ratio"]).to_parquet(diagnostics_path, index=False)

    result = generate_feature_engineering_report(
        manifest_path=manifest_path,
        diagnostics_path=diagnostics_path,
        output_dir=output_dir,
        feature_dirs={},
    )

    assert result["metadata"]["sample_sizes"]["manifest_records"] == 0
    assert (output_dir / "feature_engineering_report.md").exists()


def test_visualize_embedding_health_rejects_missing_arrays(tmp_path: Path) -> None:
    feature_dir = tmp_path / "missing"
    feature_dir.mkdir()
    (feature_dir / "metadata.json").write_text(json.dumps({"extractor": "clip"}), encoding="utf-8")

    with pytest.raises(ValueError, match="missing embedding arrays"):
        visualize_embedding_health(feature_dir, tmp_path / "out")
