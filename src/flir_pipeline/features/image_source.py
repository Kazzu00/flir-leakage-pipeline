"""Small read-only image source shared by the existing embedding pipeline."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import pandas as pd
from PIL import Image

from flir_pipeline.data.local_images import declared_file
from flir_pipeline.features.preprocessing import decode_image_bytes


class ImageSource:
    """Read declared bytes and bind them to the hashes used in cache signatures.

    Directory validation covers every occurrence, including duplicates and smoke
    exclusions. ZIP validation keeps the historical representative-only policy.
    Neither a root path nor its transport belongs to mathematical feature identity.
    """

    def __init__(self, images_archive: Path | None, images_root: Path | None):
        if (images_archive is None) == (images_root is None):
            raise ValueError("Provide exactly one of images_archive / images_root")
        self.root = (
            images_root.expanduser().resolve() if images_root is not None else None
        )
        self.archive_path = images_archive
        self.archive: zipfile.ZipFile | None = None
        self.kind = "directory" if self.root is not None else "zip"

    def __enter__(self) -> ImageSource:
        if self.root is not None:
            if not self.root.is_dir():
                raise ValueError("images-root must be an existing directory")
        else:
            self.archive = zipfile.ZipFile(self.archive_path, "r")
        return self

    def __exit__(self, *_: object) -> None:
        if self.archive is not None:
            self.archive.close()

    def path_column(self, manifest: pd.DataFrame) -> str:
        candidates = (
            ("image_path", "relative_image_path", "source_member_path")
            if self.root is not None
            else ("source_member_path",)
        )
        for key in candidates:
            if key in manifest:
                return key
        raise ValueError(f"Manifest requires an image path column: {candidates}")

    def read(self, relative: str, expected_sha256: str) -> bytes:
        if self.root is not None:
            content = declared_file(self.root, relative).read_bytes()
        else:
            with self.archive.open(relative, "r") as stream:
                content = stream.read()
        if hashlib.sha256(content).hexdigest() != expected_sha256:
            raise ValueError(
                f"Image SHA256 mismatch against manifest/cache signature: {relative}"
            )
        return content

    def validate(self, manifest: pd.DataFrame) -> None:
        """Check actual source bytes before writing, resuming or reusing a cache."""
        if (
            "image_decode_valid" in manifest
            and not manifest.image_decode_valid.eq(True).fillna(False).all()
        ):
            raise ValueError(
                "Manifest contains invalid image decodes; inspect its QA report"
            )
        if manifest[["content_id", "image_sha256"]].isna().any().any():
            raise ValueError("Manifest content identity must not be null")
        if not manifest.content_id.eq(manifest.image_sha256).all():
            raise ValueError("content_id must equal exact image SHA256")
        column = self.path_column(manifest)
        for relative, expected in manifest[[column, "image_sha256"]].itertuples(
            index=False, name=None
        ):
            self.read(relative, expected)

    def decode(self, relative: str, expected_sha256: str) -> Image.Image:
        return decode_image_bytes(self.read(relative, expected_sha256))
