"""DINOv2 image encoder adapter using Transformers, loaded on demand."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import numpy as np
from PIL import Image

from flir_pipeline.features.base import (
    FeatureExtractor,
    PreprocessedImage,
    resolve_device,
)


class DinoV2Extractor(FeatureExtractor):
    """DINOv2 CLS-token extractor; model weights load only at construction."""

    name = "dinov2"

    def __init__(
        self,
        model_id: str = "facebook/dinov2-small",
        device: str = "auto",
        batch_size: int = 8,
        mixed_precision: bool = False,
        local_files_only: bool = False,
        model_revision: str | None = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as error:
            raise RuntimeError(
                "DINOv2 requires the optional 'vision' dependencies. "
                "Run: uv sync --extra vision"
            ) from error
        self.model_id = model_id
        self.device = resolve_device(device)
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        self.batch_size = batch_size
        self.mixed_precision = mixed_precision and self.device == "cuda"
        self.local_files_only = local_files_only
        self.model_revision = model_revision or "unknown"
        load_kwargs = {"local_files_only": local_files_only}
        if model_revision:
            load_kwargs["revision"] = model_revision
        self.processor = AutoImageProcessor.from_pretrained(model_id, **load_kwargs)
        self.model = AutoModel.from_pretrained(model_id, **load_kwargs).to(self.device)
        self.model.eval()
        self.embedding_dimension = int(self.model.config.hidden_size)
        self._torch = torch

    def preprocess(self, image: Image.Image) -> PreprocessedImage:
        return PreprocessedImage(image=image.convert("RGB"), original_mode=image.mode)

    def encode_batch(self, batch: list[PreprocessedImage]) -> np.ndarray:
        inputs = self.processor(
            images=[item.image for item in batch], return_tensors="pt"
        )
        inputs = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        context = (
            self._torch.autocast(device_type="cuda", dtype=self._torch.float16)
            if self.mixed_precision
            else nullcontext()
        )
        with self._torch.inference_mode(), context:
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]
        return embeddings.detach().float().cpu().numpy()

    def metadata(self) -> dict[str, Any]:
        return {
            "extractor": self.name,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "embedding_dimension": self.embedding_dimension,
            "pooling_strategy": "cls_token",
            "preprocessing": {
                "processor": "transformers.AutoImageProcessor",
                "model_input_mode": "RGB",
                "processor_config": self.processor.to_dict(),
            },
            "device": self.device,
            "batch_size": self.batch_size,
            "mixed_precision": self.mixed_precision,
            "raw_dtype": "float32",
            "normalized_dtype": "float32",
        }
