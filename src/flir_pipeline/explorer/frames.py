"""Read original ZIP members in memory and verify exact image bytes."""

from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import pandas as pd
from PIL import Image


class FrameReader:
    def __init__(self, data_root: Path, manifest: pd.DataFrame):
        self.root = data_root.resolve()
        if not manifest.frame_id.is_unique:
            raise ValueError("FrameReader requires unique frame occurrences")
        self.rows = manifest.set_index("frame_id")
        self.archives: dict[Path, zipfile.ZipFile] = {}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        for archive in self.archives.values():
            archive.close()

    def image(self, frame_id: str, *, width: int = 640) -> Image.Image:
        row = self.rows.loc[frame_id]
        relative = Path(row.source_archive)
        path = (self.root / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(self.root) or path.suffix.lower() != ".zip":
            raise ValueError("Source ZIP must be relative to FLIR_DATA_ROOT")
        if path not in self.archives:
            self.archives[path] = zipfile.ZipFile(path, mode="r")
        data = self.archives[path].read(row.source_member_path)
        if hashlib.sha256(data).hexdigest() != row.image_sha256:
            raise ValueError("Source image bytes differ from manifest")
        with Image.open(io.BytesIO(data)) as original:
            image = original.convert("RGB")
            image.thumbnail((width, width), Image.Resampling.LANCZOS)
        return image
