"""Occurrence-based, bounded access to a verified external image-variant ledger.

Scientific IDs come exclusively from ingestion. ZIP packaging supplies physical
locators, never frame identity, logical order or verified temporal provenance.
"""

from __future__ import annotations

import os
import zipfile
from collections import OrderedDict
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import pandas as pd

from flir_pipeline.data.local_images import declared_file
from flir_pipeline.data.video_variant_contract import IngestionConfig, Limits
from flir_pipeline.data.video_variant_storage import inspect_ingestion
from flir_pipeline.data.zip_image_collection import (
    check_archive,
    image_properties,
    read_entry,
)
from flir_pipeline.utils.hashing import sha256_file, sha256_stream


@dataclass(frozen=True)
class ImageLocation:
    """Exact scientific occurrence and its source-bound physical ZIP locator.

    Paths are relative to input_root. member_ordinal indexes the complete ZIP
    central directory, including directories and non-image entries. entry_id and
    archive_sha256 may change on repackaging while frame_id/content_id stay fixed.
    No temporal, annotation, sequence or split interpretation is added.
    """

    artifact_id: str
    frame_id: str
    content_id: str
    image_sha256: str
    entry_id: str
    archive_key: str
    source_archive: str
    archive_sha256: str
    member_ordinal: int
    source_member_path: str
    size_bytes: int
    compressed_bytes: int
    crc32: int
    width: int
    height: int
    channels: int
    image_mode: str
    image_format: str


@dataclass
class _OpenArchive:
    resources: ExitStack
    stream: BinaryIO
    archive: zipfile.ZipFile


def _fingerprint(stat: os.stat_result) -> tuple[int, ...]:
    """Detect ordinary source replacement/mutation without rehashing each image."""
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


class VideoVariantImages:
    """Read bytes by frame_id without extracting or caching images.

    Construction verifies the final ingestion publication and ledger joins; it
    opens no ZIPs. Entering the mandatory context verifies every declared ZIP's
    SHA256 and complete central directory, without decoding images or requiring
    source videos/auxiliary evidence. read() verifies CRC, image SHA256 and decode
    QA for one occurrence and returns (bytes, immutable ImageLocation).

    At most max_open_archives ZIPs are open, with LRU eviction. ZIP checksums are
    reused only while filesystem fingerprints remain unchanged; selected bytes
    are hashed on every read. Inputs/publication must remain immutable throughout
    the session. This single-reader API is not a lock against hostile concurrent
    changes, a full image replay, or a temporal-alignment verifier.

    Metadata is O(ledger size); only the current image is buffered internally.
    frame_ids follows deterministic scientific frame_id order, never ZIP order.
    Optional consumer limits can tighten, but never relax, ingestion budgets.
    Every read/enter failure closes the session. close() is idempotent; a closed
    instance cannot be reopened. locate() and frame_ids need no active session.
    """

    def __init__(
        self,
        ingestion_directory: Path,
        input_root: Path,
        *,
        max_open_archives: int = 8,
        limits: Limits | None = None,
    ) -> None:
        if type(max_open_archives) is not int or max_open_archives < 1:
            raise ValueError("max_open_archives must be a positive integer")
        self._root = Path(input_root).expanduser().resolve()
        if not self._root.is_dir():
            raise ValueError("input_root must be an existing directory")
        publication = Path(ingestion_directory).expanduser().resolve()
        checked = inspect_ingestion(publication)
        summary, metadata = checked["summary"], checked["metadata"]
        if (
            not summary["scientific_manifest_available"]
            or not summary["integrity_valid"]
        ):
            raise ValueError(
                "Ingestion requires an available scientific manifest and valid integrity"
            )
        config = IngestionConfig.model_validate(metadata["identity"]["config"])
        budgets = config.limits.model_dump()
        if limits is not None:
            budgets = {k: min(v, getattr(limits, k)) for k, v in budgets.items()}
        self._limits = Limits.model_validate(budgets)
        self._expected_format = config.expected_image_format
        self._artifact_id = metadata["artifact_id"]
        self._dataset_id = summary["dataset_id"]
        self._dataset_variant_id = summary["dataset_variant_id"]

        def read_table(name: str) -> pd.DataFrame:
            path = declared_file(publication, f"{name}.parquet")
            frame = pd.read_parquet(path)
            if sha256_file(path) != metadata["identity"]["output_checksums"][path.name]:
                raise ValueError(f"Ingestion table changed after inspection: {name}")
            return frame

        manifest, occurrences, entries, archives = (
            read_table(name)
            for name in ("manifest", "occurrences", "entries", "archives")
        )
        self._locations = self._resolve(manifest, occurrences, entries, archives)
        self._frame_ids = tuple(sorted(self._locations))
        self._archives = archives.set_index("archive_key").to_dict("index")
        # Keep central-directory facts, not image bytes; ordinals include auxiliaries.
        self._entry_facts = {
            key: tuple(
                group.sort_values("member_ordinal")[
                    ["source_member_path", "size_bytes", "compressed_bytes", "crc32"]
                ].itertuples(index=False, name=None)
            )
            for key, group in entries.groupby("archive_key", sort=True)
        }
        self._max_open = max_open_archives
        self._open: OrderedDict[str, _OpenArchive] = OrderedDict()
        self._verified: dict[str, tuple[int, ...]] = {}
        self._active = False
        self._closed = False

    def _resolve(
        self, manifest, occurrences, entries, archives
    ) -> dict[str, ImageLocation]:
        """Check relational coverage without inventing or recomputing identities."""
        science = ["frame_id", "content_id", "image_sha256"]
        joined = manifest[science].merge(
            occurrences[[*science, "entry_id", "identity_status"]],
            on="frame_id",
            how="outer",
            validate="one_to_one",
            suffixes=("", "_occurrence"),
            indicator=True,
        )
        if (
            not joined._merge.eq("both").all()
            or not joined.identity_status.eq("identified").all()
            or joined[science + ["entry_id"]].isna().any().any()
            or not joined.content_id.eq(joined.content_id_occurrence).all()
            or not joined.image_sha256.eq(joined.image_sha256_occurrence).all()
        ):
            raise ValueError(
                "Scientific occurrence coverage/identity differs from manifest"
            )
        physical = (
            entries.loc[entries.is_image]
            .drop(columns=["is_image"])
            .rename(columns={"image_sha256": "entry_sha256"})
        )
        joined = joined.drop(columns="_merge").merge(
            physical,
            on="entry_id",
            how="outer",
            validate="one_to_one",
            indicator=True,
        )
        if (
            not joined._merge.eq("both").all()
            or not joined.image_sha256.eq(joined.entry_sha256).all()
            or not joined.content_id.eq(joined.image_sha256).all()
        ):
            raise ValueError(
                "Physical image entries differ from scientific occurrences"
            )
        joined = joined.drop(columns="_merge").merge(
            archives[["archive_key", "path", "sha256"]],
            on="archive_key",
            how="left",
            validate="many_to_one",
            indicator=True,
        )
        if (
            not joined._merge.eq("both").all()
            or not joined.source_archive.eq(joined.path).all()
        ):
            raise ValueError(
                "Physical image references an unknown or inconsistent archive"
            )
        locator_fields = [
            "frame_id",
            "content_id",
            "image_sha256",
            "entry_id",
            "archive_key",
            "source_archive",
            "source_member_path",
            "member_ordinal",
            "sha256",
            "size_bytes",
            "compressed_bytes",
            "crc32",
            "width",
            "height",
            "channels",
            "image_mode",
            "image_format",
        ]
        if joined[locator_fields].isna().any().any():
            raise ValueError("Scientific image locator/QA fields must not be null")
        result = {}
        for row in joined.itertuples(index=False):
            result[row.frame_id] = ImageLocation(
                artifact_id=self._artifact_id,
                **{
                    key: getattr(row, key)
                    for key in (
                        "frame_id",
                        "content_id",
                        "image_sha256",
                        "entry_id",
                        "archive_key",
                        "source_archive",
                        "source_member_path",
                        "image_mode",
                        "image_format",
                    )
                },
                archive_sha256=row.sha256,
                **{
                    key: int(getattr(row, key))
                    for key in (
                        "member_ordinal",
                        "size_bytes",
                        "compressed_bytes",
                        "crc32",
                        "width",
                        "height",
                        "channels",
                    )
                },
            )
        return result

    @property
    def artifact_id(self) -> str:
        return self._artifact_id

    @property
    def dataset_id(self) -> str:
        return self._dataset_id

    @property
    def dataset_variant_id(self) -> str:
        return self._dataset_variant_id

    @property
    def frame_ids(self) -> tuple[str, ...]:
        """All scientific occurrences in stable frame_id order, including copies."""
        return self._frame_ids

    def locate(self, frame_id: str) -> ImageLocation:
        """Resolve a declared occurrence; unknown IDs raise KeyError."""
        return self._locations[frame_id]

    def __enter__(self) -> VideoVariantImages:
        if self._active or self._closed:
            self.close()
            raise RuntimeError("Reader session is already active or closed")
        try:
            for key in sorted(self._archives):
                self._open_archive(key)
            self._active = True
            return self
        except BaseException:
            self.close()
            raise

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        """Close every ZIP and underlying stream, including when a close fails."""
        self._active = False
        self._closed = True
        opened = list(self._open.values())
        self._open.clear()
        with ExitStack() as cleanup:
            for item in opened:
                cleanup.callback(item.resources.close)

    def _unchanged(self, key: str, stream: BinaryIO) -> None:
        path = declared_file(self._root, self._archives[key]["path"])
        expected = self._verified[key]
        if (
            _fingerprint(path.stat()) != expected
            or _fingerprint(os.fstat(stream.fileno())) != expected
        ):
            raise ValueError("ZIP source changed during reader session")

    def _open_archive(self, key: str) -> _OpenArchive:
        if key in self._open:
            opened = self._open[key]
            self._unchanged(key, opened.stream)
            self._open.move_to_end(key)
            return opened
        if len(self._open) == self._max_open:
            _, evicted = self._open.popitem(last=False)
            evicted.resources.close()
        source = self._archives[key]
        path = declared_file(self._root, source["path"])
        with ExitStack() as resources:
            stream = resources.enter_context(path.open("rb"))
            signature = _fingerprint(os.fstat(stream.fileno()))
            if signature[2] != source["size_bytes"]:
                raise ValueError("ZIP source size differs from ingestion")
            previously_verified = key in self._verified
            if previously_verified:
                self._unchanged(key, stream)
            else:
                if sha256_stream(stream) != source["sha256"]:
                    raise ValueError("ZIP source SHA256 differs from ingestion")
                self._verified[key] = signature
                self._unchanged(key, stream)
                stream.seek(0)
            archive = resources.enter_context(zipfile.ZipFile(stream, "r"))
            infos = archive.infolist()
            check_archive(infos, self._limits)
            facts = self._entry_facts.get(key, ())
            if len(infos) != len(facts):
                raise ValueError(
                    "ZIP central-directory coverage differs from ingestion"
                )
            if not previously_verified:
                for info, expected in zip(infos, facts, strict=True):
                    if (
                        info.orig_filename,
                        info.file_size,
                        info.compress_size,
                        info.CRC,
                    ) != expected:
                        raise ValueError(
                            "ZIP central-directory entry differs from ingestion ordinal"
                        )
            self._unchanged(key, stream)
            opened = _OpenArchive(resources.pop_all(), stream, archive)
            self._open[key] = opened
            return opened

    def read(self, frame_id: str) -> tuple[bytes, ImageLocation]:
        """Read one exact occurrence, validate pixels, and return bytes/provenance.

        Every read uses the ledger ordinal's ZipInfo, then the shared bounded
        reader and image QA. No image cache or scientific deduplication is done.
        """
        if not self._active:
            raise RuntimeError("Read images inside an active reader context")
        try:
            location = self.locate(frame_id)
            opened = self._open_archive(location.archive_key)
            info = opened.archive.infolist()[location.member_ordinal]
            content, digest = read_entry(opened.archive, info, self._limits)
            if digest != location.image_sha256:
                raise ValueError("Image SHA256 differs from scientific occurrence")
            properties = image_properties(content, self._limits, self._expected_format)
            if not properties["image_decode_valid"]:
                raise ValueError(f"Image decode QA failed: {properties['image_error']}")
            if any(
                properties[k] != getattr(location, k)
                for k in ("width", "height", "channels", "image_mode", "image_format")
            ):
                raise ValueError("Decoded image properties differ from ingestion")
            self._unchanged(location.archive_key, opened.stream)
            return content, location
        except BaseException:
            self.close()
            raise
