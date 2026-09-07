"""Streaming SHA256 helpers for files and ZIP members."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path
from typing import BinaryIO


def sha256_stream(stream: BinaryIO, chunk_size: int = 1024 * 1024) -> str:
    """Hash a binary stream without loading it fully into memory."""
    digest = hashlib.sha256()
    while chunk := stream.read(chunk_size):
        digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file using chunked reads."""
    with path.open("rb") as stream:
        return sha256_stream(stream, chunk_size)


def sha256_zip_member(
    archive_path: Path,
    member_path: str,
    chunk_size: int = 1024 * 1024,
) -> str:
    """Hash one ZIP member by streaming it directly from the archive."""
    with zipfile.ZipFile(archive_path) as archive:
        with archive.open(member_path, "r") as stream:
            return sha256_stream(stream, chunk_size)
