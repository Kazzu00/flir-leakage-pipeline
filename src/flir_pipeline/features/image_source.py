"""Small read-only image source shared by the existing embedding pipeline."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from PIL import Image

from flir_pipeline.data.local_images import declared_file
from flir_pipeline.features.preprocessing import decode_image_bytes

if TYPE_CHECKING:
    from flir_pipeline.data.video_variant_contract import Limits
    from flir_pipeline.data.video_variant_images import ImageLocation


class ImageSource:
    """Read declared bytes and bind them to the hashes used in cache signatures.

    Directory validation covers every occurrence, including duplicates and smoke
    exclusions. ZIP validation keeps the historical representative-only policy.
    Neither a root path nor its transport belongs to mathematical feature identity.
    from_video_variant() adds occurrence-based multishard reading without changing
    this ZIP/directory constructor. Extraction storage/CLI support is separate.
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

    @staticmethod
    def from_video_variant(
        ingestion_directory: Path,
        input_root: Path,
        *,
        max_open_archives: int = 8,
        limits: Limits | None = None,
    ) -> _VideoVariantImageSource:
        """Create a verified multishard source; read/decode references are frame_id.

        Use as a context manager. The reader verifies ZIP sources once per session
        and returns original bytes through the same read/decode signatures. locate()
        preserves the authoritative scientific/physical mapping. No manifest is
        rewritten and no encoder, embedding store or extraction CLI is invoked.
        """
        return _VideoVariantImageSource(
            ingestion_directory,
            input_root,
            max_open_archives=max_open_archives,
            limits=limits,
        )

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


class _VideoVariantImageSource(ImageSource):
    """Additive transport owned by the ImageSource.from_video_variant factory.

    root denotes the authorized ZIP-input root, not a materialized image folder.
    kind is explicitly zip_collection; archive/archive_path remain None. Existing
    consumers opt in through the factory rather than historical source inference.
    """

    def __init__(
        self,
        ingestion_directory: Path,
        input_root: Path,
        *,
        max_open_archives: int,
        limits: Limits | None,
    ) -> None:
        # Keep ingestion/ledger imports lazy for historical ZIP/directory clients.
        from flir_pipeline.data.video_variant_images import VideoVariantImages

        super().__init__(None, Path(input_root))
        self.kind = "zip_collection"
        self._reader = VideoVariantImages(
            ingestion_directory,
            input_root,
            max_open_archives=max_open_archives,
            limits=limits,
        )

    @property
    def frame_ids(self) -> tuple[str, ...]:
        """Scientific ID order for reproducibility; it is not temporal order."""
        return self._reader.frame_ids

    def locate(self, frame_id: str) -> ImageLocation:
        """Return the immutable ledger locator without substituting physical IDs."""
        return self._reader.locate(frame_id)

    def __enter__(self) -> _VideoVariantImageSource:
        self._reader.__enter__()
        return self

    def close(self) -> None:
        """Release all reader resources; safe after a failed read or context exit."""
        self._reader.close()

    def __exit__(self, *_: object) -> None:
        self.close()

    def path_column(self, manifest: pd.DataFrame) -> str:
        """Return the scientific reference column, never an inferred ZIP path."""
        if "frame_id" not in manifest:
            raise ValueError("Video variant source requires frame_id references")
        return "frame_id"

    def read(self, relative: str, expected_sha256: str) -> bytes:
        """Read a frame_id using the legacy argument signature and bound SHA256."""
        try:
            location = self.locate(relative)
            if expected_sha256 != location.image_sha256:
                raise ValueError(
                    "Image SHA256 mismatch against video variant occurrence"
                )
            content, _ = self._reader.read(relative)
            return content
        except BaseException:
            self.close()
            raise

    def validate(self, manifest: pd.DataFrame) -> None:
        """Validate the supplied occurrences, including copies; not full coverage.

        Extra temporal/annotation fields are not inputs. Identity and image QA are
        checked against the authoritative publication, without constructing another
        manifest or deduplicating occurrence references.
        """
        try:
            if "frame_id" not in manifest:
                raise ValueError("Video variant source requires frame_id references")
            if manifest.frame_id.isna().any() or not manifest.frame_id.is_unique:
                raise ValueError(
                    "Video variant frame_id references must be unique and non-null"
                )
            super().validate(manifest)
        except BaseException:
            self.close()
            raise

    def decode(self, relative: str, expected_sha256: str) -> Image.Image:
        """Keep the existing Pillow decoding contract and close on decode failure."""
        try:
            return super().decode(relative, expected_sha256)
        except BaseException:
            self.close()
            raise
