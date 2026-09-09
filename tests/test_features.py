import hashlib
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from flir_pipeline.features.base import DeterministicFakeExtractor, l2_normalize
from flir_pipeline.features.clip import projected_image_features
from flir_pipeline.features.diagnostics import run_diagnostics
from flir_pipeline.features.storage import (
    _quality,
    extract_to_store,
    feature_space_id,
    verify_feature_directory,
)


def _image_bytes(mode: str = "RGB") -> bytes:
    stream = BytesIO()
    image = Image.new(mode, (8, 6), 120)
    image.save(stream, format="PNG")
    return stream.getvalue()


def _write_zip(path: Path, members: dict[str, bytes]) -> None:
    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        for name, content in members.items():
            archive.writestr(name, content)


def _manifest(path: Path, image_hash: str) -> None:
    pd.DataFrame(
        [
            {
                "frame_id": "frame-a",
                "content_id": image_hash,
                "image_sha256": image_hash,
                "label_sha256": "label-a",
                "source_archive": "Imagenes.zip",
                "source_member_path": "Imagenes/train/a.png",
            },
            {
                "frame_id": "frame-b",
                "content_id": image_hash,
                "image_sha256": image_hash,
                "label_sha256": "label-b",
                "source_archive": "Imagenes.zip",
                "source_member_path": "Imagenes/val/b.png",
            },
        ]
    ).to_parquet(path, index=False)


def test_fake_extractor_and_feature_space_identity() -> None:
    extractor = DeterministicFakeExtractor(4)
    image = Image.new("L", (4, 4), 100)
    first = extractor.encode_batch([extractor.preprocess(image)])
    second = extractor.encode_batch([extractor.preprocess(image)])
    normalized, norms = l2_normalize(first)

    assert first.shape == (1, 4)
    assert np.array_equal(first, second)
    assert np.allclose(np.linalg.norm(normalized, axis=1), 1)
    assert norms[0] > 0
    base = {"extractor": "fake", "model_id": "x", "batch_size": 1, "device": "cpu"}
    assert feature_space_id(base) == feature_space_id({**base, "batch_size": 32})
    assert feature_space_id(base) != feature_space_id({**base, "model_id": "y"})
    assert feature_space_id(base) != feature_space_id({**base, "pooling_strategy": "mean"})


def test_content_level_storage_and_resume(tmp_path: Path) -> None:
    content = _image_bytes()
    image_hash = hashlib.sha256(content).hexdigest()
    archive = tmp_path / "Imagenes.zip"
    _write_zip(archive, {"Imagenes/train/a.png": content, "Imagenes/val/b.png": content})
    manifest = tmp_path / "manifest.parquet"
    _manifest(manifest, image_hash)
    extractor = DeterministicFakeExtractor(6)

    feature_dir = extract_to_store(
        manifest, archive, extractor, tmp_path / "artifacts", batch_size=1
    )
    second_dir = extract_to_store(
        manifest, archive, extractor, tmp_path / "artifacts", batch_size=2
    )
    result = verify_feature_directory(feature_dir)
    content_index = pd.read_parquet(feature_dir / "content_index.parquet")
    record_index = pd.read_parquet(feature_dir / "record_index.parquet")
    raw = np.load(feature_dir / "embeddings_raw.npy")

    assert feature_dir == second_dir
    assert raw.shape == (1, 6)
    assert len(content_index) == 1
    assert len(record_index) == 2
    assert record_index["embedding_row"].tolist() == [0, 0]
    assert result["quality_valid"]
    with pytest.raises(ValueError, match="limit_content"):
        extract_to_store(manifest, archive, extractor, tmp_path / "artifacts", limit_content=0)


def test_quality_detects_nan_inf_and_zero_vectors() -> None:
    content_index = pd.DataFrame({"content_id": ["one"], "embedding_row": [0]})
    record_index = pd.DataFrame({"frame_id": ["frame"], "content_id": ["one"]})
    result = _quality(
        np.asarray([[np.nan, np.inf]], dtype=np.float32),
        np.asarray([[0.0, 0.0]], dtype=np.float32),
        content_index,
        record_index,
    )

    assert result["raw_has_nan"]
    assert result["raw_has_inf"]
    assert result["zero_norm_count"] == 1
    assert not result["quality_valid"]


def test_diagnostics_are_reproducible(tmp_path: Path) -> None:
    content = _image_bytes()
    image_hash = hashlib.sha256(content).hexdigest()
    archive = tmp_path / "Imagenes.zip"
    _write_zip(archive, {"Imagenes/train/a.png": content})
    manifest = tmp_path / "manifest.parquet"
    _manifest(manifest, image_hash)

    first_dir = run_diagnostics(manifest, archive, tmp_path / "diagnostics", limit_content=1)
    first = pd.read_parquet(first_dir / "diagnostics.parquet")
    second_dir = run_diagnostics(manifest, archive, tmp_path / "diagnostics", limit_content=1)
    second = pd.read_parquet(second_dir / "diagnostics.parquet")

    assert len(first) == 1
    assert first["phash"].tolist() == second["phash"].tolist()
    assert first["dhash"].tolist() == second["dhash"].tolist()
    assert first["original_mode"].tolist() == ["RGB"]


class _FakeTensor:
    def __init__(self, values: list[list[float]]) -> None:
        self.values = np.asarray(values, dtype=np.float32)

    def detach(self) -> "_FakeTensor":
        return self

    def float(self) -> "_FakeTensor":
        return self

    def cpu(self) -> "_FakeTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self.values


class _FakeTorch:
    Tensor = _FakeTensor


class _FakePoolingOutput:
    def __init__(self, pooler_output: _FakeTensor | None) -> None:
        self.pooler_output = pooler_output
        self.last_hidden_state = _FakeTensor([[99.0, 99.0]])


class _FakeClipModel:
    def __init__(self, output: object) -> None:
        self.output = output

    def get_image_features(self, **_: object) -> object:
        return self.output


def test_clip_projected_output_accepts_tensor_and_pooler_output() -> None:
    torch_module = _FakeTorch()
    direct_model = _FakeClipModel(_FakeTensor([[1.0, 2.0, 3.0]]))
    pooled_model = _FakeClipModel(
        _FakePoolingOutput(_FakeTensor([[4.0, 5.0, 6.0]]))
    )

    direct = projected_image_features(direct_model.get_image_features(), torch_module)
    pooled = projected_image_features(pooled_model.get_image_features(), torch_module)
    direct_array = direct.detach().float().cpu().numpy()
    pooled_array = pooled.detach().float().cpu().numpy()

    assert direct_array.dtype == np.float32
    assert pooled_array.dtype == np.float32
    assert direct_array.shape == (1, 3)
    assert pooled_array.shape == (1, 3)
    assert np.array_equal(pooled_array, [[4.0, 5.0, 6.0]])


def test_clip_projected_output_rejects_missing_pooler_output() -> None:
    torch_module = _FakeTorch()
    output = _FakePoolingOutput(None)

    with pytest.raises(TypeError, match="pooler_output"):
        projected_image_features(output, torch_module)
