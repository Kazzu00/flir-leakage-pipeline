"""Explicit image-list ingestion with optional labels/videos; read-only sources."""

from __future__ import annotations

import hashlib
import io
import tarfile
import zipfile
from collections import defaultdict
from pathlib import Path

import yaml
from PIL import Image
from pydantic import Field

from flir_pipeline.data.local_images import declared_file
from flir_pipeline.data.yolo_labels import validate_label_bytes
from flir_pipeline.general.contract import (
    ID,
    ClassEntry,
    Contract,
    Evidence,
    Manifest,
    Occurrence,
    Position,
    Relation,
    Video,
)
from flir_pipeline.utils.hashing import sha256_file


class ImageInput(Contract):
    frame_id: ID
    image: ID
    label: str | None = None
    original_split: str | None = None
    positions: list[Position] = Field(default_factory=list)
    exclusion_reasons: list[str] = Field(default_factory=list)


class InputConfig(Contract):
    classes: list[ClassEntry] = Field(default_factory=list)
    images: list[ImageInput]
    videos: list[Video] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)


def ingest(config_path: Path, root: Path) -> Manifest:
    cfg = InputConfig.model_validate(
        yaml.safe_load(config_path.read_text(encoding="utf-8"))
    )
    sources = {}

    def read(relative):
        path = declared_file(root, relative)
        data = path.read_bytes()
        checksum = hashlib.sha256(data).hexdigest()
        sources[relative] = Evidence(
            path=relative,
            sha256=checksum,
            method="read_original_bytes",
            version="1",
            provenance="explicit_input_list",
        )
        return data, checksum

    records = []
    for row in cfg.images:
        raw, checksum = read(row.image)
        with Image.open(io.BytesIO(raw)) as image:
            image.load()  # fail closed on corrupt data; no fabricated empty image
        kwargs = {}
        if row.label is not None:
            raw_label, label_checksum = read(row.label)
            label = validate_label_bytes(raw_label)
            kwargs = {
                "annotation_status": "invalid"
                if not label.label_valid
                else "empty"
                if label.label_empty
                else "valid",
                "label_sha256": label_checksum,
                "label_source": row.label,
                "boxes": [list(b) for b in label.boxes],
                "annotation_errors": label.errors + label.warnings,
            }
        records.append(
            Occurrence(
                frame_id=row.frame_id,
                content_id=checksum,
                image_sha256=checksum,
                source=row.image,
                original_name=Path(row.image).name,
                original_split=row.original_split,
                positions=row.positions,
                exclusion_reasons=row.exclusion_reasons,
                **kwargs,
            )
        )
    manifest = Manifest(
        classes=cfg.classes,
        videos=cfg.videos,
        records=records,
        relations=cfg.relations,
        sources=list(sources.values()),
    )
    verify_sources(manifest, root)
    return manifest


def verify_sources(manifest: Manifest, root: Path) -> dict:
    """Rehash source bytes, including videos and every relation/position evidence.

    An evidence checksum authenticates the consulted file, not its scientific
    conclusion. Producing decoded positions is a separate video-localization step.
    """
    sources = [
        *manifest.sources,
        *manifest.videos,
        *(r.evidence for r in manifest.relations),
        *(p.evidence for r in manifest.records for p in r.positions),
    ]
    expected = {}
    for source in sources:
        if source.path in expected and expected[source.path] != source.sha256:
            raise ValueError("Conflicting source checksums")
        expected[source.path] = source.sha256
    for relative, checksum in expected.items():
        if sha256_file(declared_file(root, relative)) != checksum:
            raise ValueError(f"Source checksum mismatch: {relative}")
    # Source receipts alone cannot prove that a supplied manifest maps the right
    # member to a content ID. Re-read every declared image/annotation without
    # extracting archives; compare original annotation values, never rewrite them.
    locators = defaultdict(list)
    for r in manifest.records:
        locators[r.source].append((r, False))
        if r.label_source is not None:
            locators[r.label_source].append((r, True))

    def check(locator, data):
        checksum = hashlib.sha256(data).hexdigest()
        for record, is_label in locators[locator]:
            if checksum != (record.label_sha256 if is_label else record.image_sha256):
                raise ValueError(
                    f"Occurrence byte identity mismatch: {record.frame_id}"
                )
            if is_label:
                qa = validate_label_bytes(data)
                status = (
                    "invalid"
                    if not qa.label_valid
                    else "empty"
                    if qa.label_empty
                    else "valid"
                )
                if (
                    status != record.annotation_status
                    or [list(b) for b in qa.boxes] != record.boxes
                ):
                    raise ValueError(f"Original annotations differ: {record.frame_id}")

    archives = defaultdict(dict)
    for locator in locators:
        if "::" in locator:
            archive, member = locator.split("::", 1)
            if archive not in expected:
                raise ValueError(
                    "Archive locator needs an independently checksummed source"
                )
            archives[archive][member] = locator
        else:
            data = declared_file(root, locator).read_bytes()
            check(locator, data)
            expected[locator] = hashlib.sha256(data).hexdigest()
    for archive, members in archives.items():
        path = declared_file(root, archive)
        found = set()
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as stream:
                for member in stream.infolist():
                    if member.filename in members:
                        if member.filename in found:
                            raise ValueError("Ambiguous archive member name")
                        check(members[member.filename], stream.read(member))
                        found.add(member.filename)
        else:
            with tarfile.open(path, "r:*") as stream:
                for member in stream:
                    if member.name in members:
                        if member.name in found or not member.isfile():
                            raise ValueError("Ambiguous or nonregular archive member")
                        with stream.extractfile(member) as source:
                            check(members[member.name], source.read())
                        found.add(member.name)
        if found != set(members):
            raise ValueError("Missing declared archive members")
    return dict(sorted(expected.items()))
