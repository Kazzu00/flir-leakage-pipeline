"""Read image entries only; split is provenance, never a matching feature."""

import hashlib
import io
import json
import random
import re
from collections import Counter
from contextlib import ExitStack, nullcontext
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from PIL import Image

from .models import ImageRecord

PATTERN = re.compile(r"^video_13min_(\d+)\.(jpg|jpeg|png)$", re.I)
BOUNDARIES = (199, 200, 349, 350, 399, 400, 499, 500, 501)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def encoded(value):
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def split_for(name):
    values = {part.lower() for part in Path(name.replace("\\", "/")).parts} & {
        "train",
        "val",
        "test",
    }
    return next(iter(values)) if len(values) == 1 else None


def safe_path(root, relative):
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError("Source escapes declared root")
    return path


def read_image(root, row, opened_archive=None):
    path = safe_path(root, row["source_archive_or_path"])
    if row["member"] is not None:
        with (
            nullcontext(opened_archive) if opened_archive is not None else ZipFile(path)
        ) as archive:
            item = archive.infolist()[row["member_ordinal"]]
            if item.filename != row["member"] or item.file_size > 32 * 1024 * 1024:
                raise ValueError("Changed ZIP entry or oversized image")
            raw = archive.read(item)
    else:
        if path.stat().st_size > 32 * 1024 * 1024:
            raise ValueError("Oversized image")
        raw = path.read_bytes()
    if row.get("image_sha256") and sha(raw) != row["image_sha256"]:
        raise ValueError("Source image changed")
    with Image.open(io.BytesIO(raw)) as image:
        if image.width * image.height > 40_000_000:
            raise ValueError("Oversized decoded image")
        image.load()
        return raw, image.convert("RGB")


def inventory(root, source_video):
    with ExitStack() as stack:
        return _inventory(root, source_video, stack)


def _inventory(root, source_video, stack):
    root = Path(root).resolve()
    rows, excluded, failures = [], Counter(), []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        archive = None
        if path.suffix.lower() == ".zip":
            try:
                archive = stack.enter_context(ZipFile(path))
                members = [
                    (i, item.filename) for i, item in enumerate(archive.infolist())
                ]
            except BadZipFile:
                failures.append({"source": relative, "error": "corrupt_zip"})
                continue
        else:
            members = [(None, path.name)]
        for ordinal, member in members:
            match = PATTERN.fullmatch(Path(member).name)
            if not match:
                if "video_13min_" in member and Path(member).suffix.lower() in {
                    ".jpg",
                    ".png",
                    ".jpeg",
                }:
                    excluded[relative] += 1
                continue
            locator = relative + (f"!{ordinal}:{member}" if ordinal is not None else "")
            row = dict(
                source_archive_or_path=relative,
                member=member if ordinal is not None else None,
                member_ordinal=ordinal,
                original_split=split_for(member if ordinal is not None else relative),
                dataset_index=int(match[1]),
                image_sha256=None,
                image_dimensions=None,
                source_video_candidate=source_video,
                read_status="readable",
                discrepancies=[],
            )
            try:
                raw, image = read_image(root, row, archive)
                row.update(image_sha256=sha(raw), image_dimensions=list(image.size))
            except (OSError, ValueError, RuntimeError, BadZipFile) as exc:
                row.update(read_status="error", discrepancies=[str(exc)])
            row["image_id"] = sha(encoded([locator, row["image_sha256"]]))
            if row["original_split"] is None:
                row["discrepancies"].append("unknown_or_conflicting_split")
            rows.append(ImageRecord(**row).model_dump())
    groups = {}
    for row in rows:
        groups.setdefault(row["dataset_index"], []).append(row)
    for items in groups.values():
        if len(items) > 1:
            for row in items:
                row["discrepancies"].append(
                    "index_repeated_across_physical_occurrences"
                )
        if len({r["image_sha256"] for r in items if r["image_sha256"]}) > 1:
            for row in items:
                row["discrepancies"].append("same_index_different_bytes")
    return {
        "schema": "alignment-image-manifest-v1",
        "images": rows,
        "excluded_non_original_family_names": dict(excluded),
        "source_errors": failures,
    }


def select(manifest, config):
    rng = random.Random(config.seed)
    eligible = [
        r
        for r in manifest["images"]
        if r["read_status"] == "readable"
        and r["source_archive_or_path"] == config.selection_archive
    ]
    chosen, deficits = {}, {}
    for split in ("train", "val", "test"):
        rows = sorted(
            [r for r in eligible if r["original_split"] == split],
            key=lambda r: (r["dataset_index"], r["image_id"]),
        )
        # Equal-count temporal strata: one random occurrence per stratum.
        count = min(len(rows), config.per_split)
        deficits[split] = config.per_split - count
        for i in range(count):
            group = rows[i * len(rows) // count : (i + 1) * len(rows) // count]
            row = rng.choice(group)
            chosen[row["image_id"]] = {
                "image_id": row["image_id"],
                "selection_reason": "split_temporal_stratum",
                "stratum": f"{split}:{i}/{count}",
            }
    missing = []
    for index in BOUNDARIES:
        matches = [r for r in eligible if r["dataset_index"] == index]
        if not matches:
            missing.append(index)
        for row in matches:
            chosen.setdefault(
                row["image_id"],
                {
                    "image_id": row["image_id"],
                    "selection_reason": "diagnostic_boundary_additional",
                    "stratum": "directed_nonrepresentative",
                },
            )
    if len(chosen) > config.max_images:
        raise ValueError("Sample exceeds image budget; no silent truncation")
    return {
        "seed": config.seed,
        "selection_archive": config.selection_archive,
        "selected": list(chosen.values()),
        "missing_diagnostic_indices": missing,
        "split_deficits": deficits,
        "exclusions": [
            {"image_id": r["image_id"], "reason": "other_provenance_or_read_failure"}
            for r in manifest["images"]
            if r not in eligible
        ],
    }
