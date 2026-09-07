"""Image decoding and in-memory preprocessing helpers."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

from PIL import Image

from flir_pipeline.features.base import PreprocessedImage


def decode_zip_image(archive: zipfile.ZipFile, member_path: str) -> Image.Image:
    """Decode one ZIP image without extracting it to disk."""
    with archive.open(member_path, "r") as stream:
        content = stream.read()
    with Image.open(io.BytesIO(content)) as image:
        image.load()
        return image.copy()


def decode_image_from_zip(archive_path: Path, member_path: str) -> Image.Image:
    """Open one image temporarily in memory and close the ZIP afterwards."""
    with zipfile.ZipFile(archive_path) as archive:
        return decode_zip_image(archive, member_path)


def ensure_rgb(image: Image.Image) -> PreprocessedImage:
    """Convert non-RGB images only in memory for model input."""
    return PreprocessedImage(image=image.convert("RGB"), original_mode=image.mode)
