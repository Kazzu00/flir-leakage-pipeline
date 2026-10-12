"""Bounded, read-only ZIP entry access; no extraction or name-based deduplication."""

from __future__ import annotations

import io
import warnings
import zipfile

from PIL import Image

from flir_pipeline.data.local_images import relative_posix_path
from flir_pipeline.data.video_variant_contract import Limits
from flir_pipeline.utils.hashing import sha256_stream


class EntryLimitError(ValueError):
    """Declared or actual resource use exceeds the configured ingestion budget."""


def check_archive(infos: list[zipfile.ZipInfo], limits: Limits) -> None:
    if len(infos) > limits.max_archive_entries:
        raise EntryLimitError("Archive entry count exceeds max_archive_entries")
    total = sum(info.file_size for info in infos)
    if total > limits.max_archive_uncompressed_bytes:
        raise EntryLimitError(
            f"Archive exceeds max_archive_uncompressed_bytes ({total} declared bytes)"
        )


def check_entry(info: zipfile.ZipInfo, limits: Limits) -> None:
    # Never normalize away traversal or aliases before validating their identity.
    # ZipInfo.filename normalizes backslashes on Windows and truncates NULs.
    # orig_filename retains the central-directory spelling to audit those cases.
    relative_posix_path(info.orig_filename)
    if info.flag_bits & 1:
        raise ValueError("Encrypted ZIP entry is unsupported")
    if info.file_size > limits.max_member_bytes:
        raise EntryLimitError("Entry exceeds max_member_bytes")
    ratio = info.file_size / max(info.compress_size, 1)
    if ratio > limits.max_compression_ratio:
        raise EntryLimitError("Entry exceeds max_compression_ratio")


def read_entry(
    archive: zipfile.ZipFile, info: zipfile.ZipInfo, limits: Limits
) -> tuple[bytes, str]:
    """Read by ZipInfo so equal filenames still denote distinct physical entries.

    One bounded image is buffered for Pillow; the collection is never buffered or
    extracted. Reading through EOF also exercises the ZIP CRC check.
    """
    check_entry(info, limits)
    buffer = io.BytesIO()
    with archive.open(info, "r") as stream:
        while block := stream.read(
            min(1024 * 1024, limits.max_member_bytes + 1 - buffer.tell())
        ):
            buffer.write(block)
            if buffer.tell() > limits.max_member_bytes:
                raise EntryLimitError("Decompressed entry exceeds max_member_bytes")
    if buffer.tell() != info.file_size:
        raise ValueError("Decompressed size differs from the ZIP declaration")
    buffer.seek(0)
    digest = sha256_stream(buffer)
    return buffer.getvalue(), digest


def image_properties(
    content: bytes, limits: Limits, expected_format: str | None
) -> dict:
    """Inspect dimensions before decode; failures remain data QA, never negatives."""
    result = dict(
        width=None,
        height=None,
        channels=None,
        image_mode=None,
        image_format=None,
        image_decode_valid=False,
        image_error=None,
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(content)) as image:
                result.update(
                    width=image.width,
                    height=image.height,
                    channels=len(image.getbands()),
                    image_mode=image.mode,
                    image_format=image.format,
                )
                if (
                    image.width > limits.max_width
                    or image.height > limits.max_height
                    or image.width * image.height > limits.max_pixels
                ):
                    raise EntryLimitError("Image dimensions exceed configured limits")
                if expected_format is not None and image.format != expected_format:
                    raise ValueError("Image format differs from expected_image_format")
                image.load()
                result["image_decode_valid"] = True
    except (
        OSError,
        ValueError,
        SyntaxError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as error:
        result["image_error"] = str(error)
    return result
