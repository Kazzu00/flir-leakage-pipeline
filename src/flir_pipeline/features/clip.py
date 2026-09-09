"""CLIP image encoder adapter using Transformers, loaded on demand."""

from __future__ import annotations

from contextlib import nullcontext
from importlib.metadata import version
from typing import Any

import numpy as np
from PIL import Image

from flir_pipeline.features.base import (
    FeatureExtractor,
    PreprocessedImage,
    resolve_device,
)
from flir_pipeline.features.model_revision import resolve_model_revision
from flir_pipeline.features.preprocessing import ensure_rgb


def projected_image_features(output: Any, torch_module: Any) -> Any:
    """Select CLIP's projected image embedding across Transformers API variants."""
    if isinstance(output, torch_module.Tensor):
        return output
    pooler_output = getattr(output, "pooler_output", None)
    if pooler_output is not None:
        return pooler_output
    raise TypeError(
        "CLIP get_image_features() returned neither a torch.Tensor nor "
        "an object with a non-null pooler_output; refusing implicit pooling."
    )


class CLIPExtractor(FeatureExtractor):
    """Encode image batches as raw float32 CLIP projected image embeddings.

    No text prompts, labels, bounding boxes or historical splits enter the image
    encoder. Storage deduplicates content and normalizes vectors separately.
    Weights load only at construction; model and processor revisions are paired.
    """

    name = "clip"

    def __init__(
        self,
        model_id: str = "openai/clip-vit-base-patch32",
        device: str = "auto",
        batch_size: int = 8,
        mixed_precision: bool = False,
        local_files_only: bool = False,
        model_revision: str | None = None,
        require_resolved_revision: bool = False,
    ) -> None:
        try:
            import torch
            from transformers import AutoProcessor, CLIPModel
        except ImportError as error:
            raise RuntimeError(
                "CLIP requires the optional 'vision' dependencies. "
                "Run: uv sync --extra vision"
            ) from error
        self.model_id = model_id
        self.device = resolve_device(device)
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        self.batch_size = batch_size
        self.mixed_precision = mixed_precision and self.device == "cuda"
        load_kwargs = {"local_files_only": local_files_only}
        if model_revision:
            load_kwargs["revision"] = model_revision
        self.model = CLIPModel.from_pretrained(model_id, **load_kwargs).to(self.device)
        self.revision = resolve_model_revision(
            self.model.config, model_revision, require_resolved_revision
        )
        self.model_revision = self.revision.effective
        if self.revision.resolved:
            load_kwargs["revision"] = self.revision.resolved
        self.processor = AutoProcessor.from_pretrained(model_id, **load_kwargs)
        self.model.eval()
        self.embedding_dimension = int(self.model.config.projection_dim)
        self._torch = torch

    def preprocess(self, image: Image.Image) -> PreprocessedImage:
        """Convert a decoded image to RGB in memory, preserving its source mode."""
        return ensure_rgb(image)

    def encode_batch(self, batch: list[PreprocessedImage]) -> np.ndarray:
        """Return (batch, projection_dim) vectors; refuse implicit alternate pooling."""
        inputs = self.processor(
            images=[item.image for item in batch], return_tensors="pt"
        )
        inputs = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
            if key == "pixel_values"
        }
        context = (
            self._torch.autocast(device_type="cuda", dtype=self._torch.float16)
            if self.mixed_precision
            else nullcontext()
        )
        with self._torch.inference_mode(), context:
            output = self.model.get_image_features(**inputs)
            embeddings = projected_image_features(output, self._torch)
        return embeddings.detach().float().cpu().numpy()

    def metadata(self) -> dict[str, Any]:
        """Describe the projected representation, weight provenance and runtime."""
        return {
            "extractor": self.name,
            "model_id": self.model_id,
            **self.revision.metadata(),
            "library_versions": {name: version(name) for name in ("torch", "transformers")},
            "embedding_dimension": self.embedding_dimension,
            "feature_type": "image_embedding",
            "pooling_strategy": "projected_pooler_output",
            "preprocessing": {
                "processor": "transformers.AutoProcessor",
                "model_input_mode": "RGB",
                "processor_config": self.processor.image_processor.to_dict(),
            },
            "device": self.device,
            "batch_size": self.batch_size,
            "mixed_precision": self.mixed_precision,
            "raw_dtype": "float32",
            "normalized_dtype": "float32",
        }
