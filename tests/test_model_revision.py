"""Exercise real adapter wiring with stand-ins: no torch, weights or network."""

import importlib
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from flir_pipeline.features.model_revision import resolve_model_revision
from flir_pipeline.features.storage import feature_space_id


def test_revision_resolution_and_fail_closed() -> None:
    sha = "a" * 40  # Synthetic commit identifier.
    revision = resolve_model_revision(SimpleNamespace(_commit_hash=sha), "main")
    assert revision.effective == sha
    assert revision.metadata()["requested_model_revision"] == "main"
    assert resolve_model_revision(SimpleNamespace(), sha, True).resolved == sha
    with pytest.warns(UserWarning, match="immutable"):
        unknown = resolve_model_revision(SimpleNamespace(), None)
    assert unknown.effective == "unknown" and unknown.resolved is None
    with pytest.raises(ValueError, match="immutable"):
        resolve_model_revision(SimpleNamespace(), "main", True)
    with pytest.raises(ValueError, match="differs"):
        resolve_model_revision(SimpleNamespace(_commit_hash=sha), "b" * 40)


class Tensor:
    def __init__(self, data):
        self.data = np.asarray(data, dtype=np.float32)

    def __getitem__(self, key):
        return Tensor(self.data[key])

    def to(self, *_):
        return self

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.data


@pytest.mark.parametrize("name,class_name,loader", [
    ("dinov2", "DinoV2Extractor", "AutoModel"),
    ("clip", "CLIPExtractor", "CLIPModel"),
])
def test_adapter_pins_processor_to_loaded_model_and_preserves_representation(
    name, class_name, loader, monkeypatch
) -> None:
    sha = "a" * 40
    calls = []

    class Model:
        config = SimpleNamespace(_commit_hash=sha, hidden_size=3, projection_dim=3)

        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls.append(("model", model_id, kwargs))
            return cls()

        def to(self, *_):
            return self

        def eval(self):
            pass

        def __call__(self, **kwargs):
            assert set(kwargs) == {"pixel_values"}
            return SimpleNamespace(last_hidden_state=Tensor([[[1, 2, 3], [9, 9, 9]]]))

        def get_image_features(self, **kwargs):
            assert set(kwargs) == {"pixel_values"}
            return SimpleNamespace(pooler_output=Tensor([[4, 5, 6]]))

    class Processor:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls.append(("processor", model_id, kwargs))
            return cls()

        @property
        def image_processor(self):
            return self

        def to_dict(self):
            return {"size": 224}

        def __call__(self, images, return_tensors):
            assert len(images) == 1 and images[0].mode == "RGB"
            return {"pixel_values": Tensor([[1]])}

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(Tensor=Tensor, inference_mode=nullcontext))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(**{loader: Model, "AutoProcessor": Processor, "AutoImageProcessor": Processor}))
    module = importlib.import_module(f"flir_pipeline.features.{name}")
    monkeypatch.setattr(module, "version", lambda _: "synthetic-version")
    extractor = getattr(module, class_name)(device="cpu", model_revision="main", local_files_only=True)
    metadata = extractor.metadata()
    assert calls[0][2] == {"revision": "main", "local_files_only": True}
    assert calls[1][2] == {"revision": sha, "local_files_only": True}
    assert metadata["resolved_model_revision"] == sha
    assert metadata["model_revision_source"] == "model.config._commit_hash"
    assert metadata["library_versions"] == {"torch": "synthetic-version", "transformers": "synthetic-version"}
    assert metadata["embedding_dimension"] == 3
    assert metadata["raw_dtype"] == metadata["normalized_dtype"] == "float32"
    assert metadata["preprocessing"]["processor_config"] == {"size": 224}
    result = extractor.encode_batch([extractor.preprocess(Image.new("L", (3, 3)))])
    np.testing.assert_array_equal(result, [[1, 2, 3]] if name == "dinov2" else [[4, 5, 6]])
    config = extractor.feature_space_config()
    assert feature_space_id(config) != feature_space_id({**config, "model_revision": "b" * 40})
