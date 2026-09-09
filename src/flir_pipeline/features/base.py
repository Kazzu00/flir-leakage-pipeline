"""Common feature extractor contracts and deterministic test extractor."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class PreprocessedImage:
    """An in-memory model input plus source mode metadata."""

    image: Image.Image
    original_mode: str
    model_input_mode: str = "RGB"


class FeatureExtractor(ABC):
    """Image-only contract consumed by content-level embedding storage.

    Implementations preserve batch order and return a finite float array of shape
    (batch length, embedding_dimension). Storage owns content deduplication,
    record mapping and L2 normalization; adapters never consume annotations or
    original_split. DINOv2 and CLIP define separate mathematical spaces.
    """

    name: str
    model_id: str
    embedding_dimension: int

    @abstractmethod
    def preprocess(self, image: Image.Image) -> Any:
        """Convert one decoded image into the extractor's model input."""

    @abstractmethod
    def encode_batch(self, batch: list[Any]) -> np.ndarray:
        """Return raw (N, D) vectors in input order; do not normalize or deduplicate."""

    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        """Return serializable runtime and model metadata."""

    def feature_space_config(self) -> dict[str, Any]:
        """Return weight, pooling and preprocessing identity for deterministic hashing.

        Runtime device and batch size are excluded. Resolved model commits affect
        identity so distinct weights cannot silently share a cache.
        """
        metadata = self.metadata()
        return {
            "extractor": self.name,
            "model_id": self.model_id,
            "model_revision": metadata.get("model_revision", "unknown"),
            "pooling_strategy": metadata.get("pooling_strategy", "none"),
            "preprocessing": metadata.get("preprocessing", {}),
            "normalization_policy": "raw_and_l2_float32",
        }


def l2_normalize(embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized embeddings and their norms without hiding zero vectors."""
    values = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1)
    normalized = np.zeros_like(values)
    nonzero = norms > 0
    normalized[nonzero] = values[nonzero] / norms[nonzero, None]
    return normalized, norms


def resolve_device(device: str) -> str:
    """Resolve auto without importing torch until feature execution."""
    if device not in {"cpu", "cuda", "auto"}:
        raise ValueError("device must be cpu, cuda, or auto")
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class DeterministicFakeExtractor(FeatureExtractor):
    """Offline extractor for tests; never use it for scientific results."""

    name = "fake"

    def __init__(self, embedding_dimension: int = 8) -> None:
        if embedding_dimension < 1:
            raise ValueError("embedding_dimension must be positive")
        self.model_id = "deterministic-fake"
        self.embedding_dimension = embedding_dimension

    def preprocess(self, image: Image.Image) -> PreprocessedImage:
        return PreprocessedImage(image=image.convert("RGB"), original_mode=image.mode)

    def feature_space_config(self) -> dict[str, Any]:
        """Include the configurable fake dimension to avoid test-cache collisions."""
        return {**super().feature_space_config(), "embedding_dimension": self.embedding_dimension}

    def encode_batch(self, batch: list[PreprocessedImage]) -> np.ndarray:
        rows = []
        for item in batch:
            pixels = np.asarray(item.image, dtype=np.uint8).tobytes()
            digest = hashlib.sha256(pixels).digest()
            values = np.frombuffer(digest, dtype=np.uint8).astype(np.float32) / 255.0
            rows.append(np.resize(values, self.embedding_dimension))
        return np.asarray(rows, dtype=np.float32)

    def metadata(self) -> dict[str, Any]:
        return {
            "extractor": self.name,
            "model_id": self.model_id,
            "model_revision": "local-deterministic-v1",
            "embedding_dimension": self.embedding_dimension,
            "pooling_strategy": "deterministic_digest",
            "preprocessing": {"input_mode": "RGB", "source": "in_memory"},
            "device": "cpu",
            "batch_size": None,
            "raw_dtype": "float32",
            "normalized_dtype": "float32",
        }
