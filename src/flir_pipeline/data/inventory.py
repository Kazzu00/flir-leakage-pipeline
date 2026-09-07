"""Read-only inventory, archive inspection and lineage analysis."""

from __future__ import annotations

import csv
import json
import posixpath
import re
import time
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from tqdm import tqdm

from flir_pipeline.utils.hashing import sha256_file, sha256_stream

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
LABEL_EXTENSIONS = {".txt"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}
MODEL_EXTENSIONS = {".pt", ".pth", ".ckpt", ".onnx"}
ARCHIVE_EXTENSIONS = {".zip", ".tar", ".gz", ".tgz", ".rar", ".7z"}
CONCEPTS = (
    "train",
    "training",
    "val",
    "validation",
    "test",
    "images",
    "imagenes",
    "labels",
    "etiquetas",
    "video",
    "videos",
)


@dataclass
class FileInventory:
    filename: str
    relative_path: str
    extension: str
    size_bytes: int
    file_type: str
    is_archive: bool
    archive_sha256: str | None = None
    modified_time: str | None = None
    archive_readable: bool | None = None
    archive_error: str | None = None


@dataclass
class ArchiveMember:
    archive_name: str
    member_path: str
    filename: str
    extension: str
    file_type: str
    uncompressed_size: int
    compressed_size: int
    crc: int
    directory_depth: int
    is_directory: bool
    sha256: str | None = None
    concepts: str = ""


@dataclass
class ArchiveInspection:
    archive_name: str
    total_members: int = 0
    total_files: int = 0
    total_directories: int = 0
    compressed_size: int = 0
    uncompressed_size: int = 0
    compression_ratio: float | None = None
    extensions: list[str] = field(default_factory=list)
    extension_counts: dict[str, int] = field(default_factory=dict)
    top_level_directories: list[str] = field(default_factory=list)
    second_level_directories: list[str] = field(default_factory=list)
    maximum_observed_depth: int = 0
    suspicious_members: list[str] = field(default_factory=list)
    readable: bool = True
    error: str | None = None


def _extension(name: str) -> str:
    return Path(name).suffix.lower()


def _normalise_member_path(name: str) -> str:
    return posixpath.normpath(name.replace("\\", "/")).lstrip("./")


def classify_member(name: str) -> str:
    extension = _extension(name)
    if extension in IMAGE_EXTENSIONS:
        return "image"
    if extension in LABEL_EXTENSIONS:
        return "label"
    if extension in VIDEO_EXTENSIONS:
        return "video"
    if extension in MODEL_EXTENSIONS:
        return "model"
    if extension in ARCHIVE_EXTENSIONS:
        return "archive"
    if extension in {".yaml", ".yml", ".json", ".toml", ".ini", ".cfg"}:
        return "config"
    if extension in {".csv", ".md", ".xml", ".log"}:
        return "metadata"
    return "other"


def _concepts(path: str) -> str:
    parts = re.split(r"[/_. -]+", path.lower())
    return ",".join(concept for concept in CONCEPTS if concept in parts)


def _depth(path: str, is_directory: bool) -> int:
    parts = [part for part in path.split("/") if part]
    return max(len(parts) - (0 if is_directory else 1), 0)


def _member_from_info(archive_name: str, info: zipfile.ZipInfo) -> ArchiveMember:
    member_path = _normalise_member_path(info.filename)
    is_directory = info.is_dir() or info.filename.endswith(("/", "\\"))
    return ArchiveMember(
        archive_name=archive_name,
        member_path=member_path,
        filename=posixpath.basename(member_path),
        extension=_extension(member_path),
        file_type=classify_member(member_path),
        uncompressed_size=info.file_size,
        compressed_size=info.compress_size,
        crc=info.CRC,
        directory_depth=_depth(member_path, is_directory),
        is_directory=is_directory,
        concepts=_concepts(member_path),
    )


def inspect_archive(
    archive_path: Path,
    hash_members: bool = False,
    hash_file_types: set[str] | None = None,
) -> tuple[ArchiveInspection, list[ArchiveMember]]:
    """Inspect a ZIP stream without extracting any members."""
    archive_name = archive_path.name
    inspection = ArchiveInspection(archive_name=archive_name)
    members: list[ArchiveMember] = []
    hash_file_types = hash_file_types or {"image", "label"}
    try:
        with zipfile.ZipFile(archive_path) as archive:
            infos = archive.infolist()
            inspection.total_members = len(infos)
            top_level: set[str] = set()
            second_level: set[str] = set()
            extension_counts: Counter[str] = Counter()
            for info in tqdm(infos, desc=f"Inspecting {archive_name}", unit="member"):
                member = _member_from_info(archive_name, info)
                members.append(member)
                inspection.compressed_size += member.compressed_size
                inspection.uncompressed_size += member.uncompressed_size
                inspection.maximum_observed_depth = max(
                    inspection.maximum_observed_depth, member.directory_depth
                )
                if member.is_directory:
                    inspection.total_directories += 1
                else:
                    inspection.total_files += 1
                    extension_counts[member.extension or "[none]"] += 1
                    parts = [part for part in member.member_path.split("/") if part]
                    if len(parts) > 1:
                        top_level.add(parts[0])
                    if len(parts) > 2:
                        second_level.add("/".join(parts[:2]))
                    if hash_members and member.file_type in hash_file_types:
                        try:
                            with archive.open(info, "r") as stream:
                                member.sha256 = sha256_stream(stream)
                        except (OSError, RuntimeError, zipfile.BadZipFile) as error:
                            inspection.suspicious_members.append(
                                f"{member.member_path}: {error}"
                            )
                if info.flag_bits & 0x1:
                    inspection.suspicious_members.append(
                        f"{member.member_path}: encrypted member"
                    )
            inspection.extension_counts = dict(sorted(extension_counts.items()))
            inspection.extensions = sorted(extension_counts)
            inspection.top_level_directories = sorted(top_level)
            inspection.second_level_directories = sorted(second_level)
            if inspection.uncompressed_size:
                inspection.compression_ratio = round(
                    inspection.compressed_size / inspection.uncompressed_size, 6
                )
    except (OSError, zipfile.BadZipFile, RuntimeError) as error:
        inspection.readable = False
        inspection.error = str(error)
    return inspection, members


def inventory_files(
    root: Path, inspect_archives: bool = True
) -> tuple[list[FileInventory], dict[str, ArchiveInspection], list[ArchiveMember]]:
    """Inventory direct files under a data root without touching their contents."""
    files: list[FileInventory] = []
    inspections: dict[str, ArchiveInspection] = {}
    members: list[ArchiveMember] = []
    for path in sorted(root.iterdir()):
        if not path.is_file():
            continue
        relative_path = path.relative_to(root).as_posix()
        extension = _extension(path.name)
        is_archive = extension == ".zip"
        item = FileInventory(
            filename=path.name,
            relative_path=relative_path,
            extension=extension,
            size_bytes=path.stat().st_size,
            file_type="archive" if is_archive else classify_member(path.name),
            is_archive=is_archive,
            archive_sha256=sha256_file(path) if is_archive else None,
            modified_time=time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(path.stat().st_mtime)
            ),
        )
        if is_archive and inspect_archives:
            inspection, archive_members = inspect_archive(path)
            item.archive_readable = inspection.readable
            item.archive_error = inspection.error
            inspections[path.name] = inspection
            members.extend(archive_members)
        files.append(item)
    return files, inspections, members


def _write_csv(path: Path, rows: Iterable[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _basename(member_path: str) -> str:
    return Path(member_path).stem.casefold()


def _split_for_path(member_path: str) -> str:
    parts = [part.casefold() for part in member_path.split("/")]
    for split in ("train", "training", "val", "validation", "test"):
        if split in parts:
            return (
                "val"
                if split == "validation"
                else ("train" if split == "training" else split)
            )
    return "unknown"


def match_images_labels(members: list[ArchiveMember]) -> list[dict]:
    """Match image and label members by split and basename, without edits."""
    images = [member for member in members if member.file_type == "image"]
    labels = [member for member in members if member.file_type == "label"]
    labels_by_key: dict[tuple[str, str], list[ArchiveMember]] = defaultdict(list)
    for label in labels:
        labels_by_key[
            (_split_for_path(label.member_path), _basename(label.member_path))
        ].append(label)
    rows: list[dict] = []
    matched_labels: set[tuple[str, str]] = set()
    for image in images:
        key = (_split_for_path(image.member_path), _basename(image.member_path))
        possible = labels_by_key.get(key, [])
        label = possible[0] if possible else None
        if label:
            matched_labels.add((label.archive_name, label.member_path))
        rows.append(
            {
                "split": key[0],
                "image_archive": image.archive_name,
                "image_member": image.member_path,
                "image_basename": _basename(image.member_path),
                "label_archive": label.archive_name if label else "",
                "label_member": label.member_path if label else "",
                "match_status": "matched" if label else "image_without_label",
            }
        )
    for label in labels:
        if (label.archive_name, label.member_path) not in matched_labels:
            rows.append(
                {
                    "split": _split_for_path(label.member_path),
                    "image_archive": "",
                    "image_member": "",
                    "image_basename": "",
                    "label_archive": label.archive_name,
                    "label_member": label.member_path,
                    "match_status": "label_without_image",
                }
            )
    return rows


def _filename_features(member: ArchiveMember) -> dict:
    stem = Path(member.filename).stem
    components = re.findall(r"\d+", stem)
    sequence_match = re.search(
        r"(?:^|[_-])(v\d+|video\d+|vid\d+)(?:[_-]|$)", stem, re.I
    )
    frame_match = re.search(r"(?:^|[_-])frame[_-]?(\d+)(?:[_-]|$)", stem, re.I)
    frame_match = frame_match or re.search(r"(?:^|[_-])(\d{3,})(?:[_-]|$)", stem)
    sequence = sequence_match.group(1).lower() if sequence_match else ""
    if not sequence and frame_match:
        sequence = stem[: frame_match.start(1)].rstrip("_-").lower()
    if sequence == "frame":
        sequence = ""
    return {
        "archive_name": member.archive_name,
        "member_path": member.member_path,
        "basename": stem,
        "split": _split_for_path(member.member_path),
        "prefix": stem.split("_")[0] if stem else "",
        "numeric_components": ",".join(components),
        "numeric_component_count": len(components),
        "possible_sequence": sequence,
        "possible_frame_index": int(frame_match.group(1)) if frame_match else "",
        "inference_confidence": "medium"
        if frame_match and sequence
        else ("low" if frame_match else "unknown"),
    }


def filename_patterns(members: list[ArchiveMember]) -> list[dict]:
    return [
        _filename_features(member) for member in members if member.file_type == "image"
    ]


def label_content_rows(root: Path, members: list[ArchiveMember]) -> list[dict]:
    """Collect lightweight TXT structure facts without changing label files."""
    rows: list[dict] = []
    grouped: dict[str, list[ArchiveMember]] = defaultdict(list)
    for member in members:
        if member.file_type == "label":
            grouped[member.archive_name].append(member)
    for archive_name, archive_members in grouped.items():
        try:
            with zipfile.ZipFile(root / archive_name) as archive:
                for member in archive_members:
                    line_count = 0
                    first_line_columns = 0
                    content_error = ""
                    try:
                        with archive.open(member.member_path, "r") as stream:
                            for raw_line in stream:
                                line_count += 1
                                if not first_line_columns and raw_line.strip():
                                    first_line_columns = len(
                                        raw_line.decode("utf-8", errors="replace")
                                        .strip()
                                        .split()
                                    )
                    except (OSError, UnicodeError, KeyError, RuntimeError) as error:
                        content_error = str(error)
                    rows.append(
                        {
                            "archive_name": archive_name,
                            "member_path": member.member_path,
                            "split": _split_for_path(member.member_path),
                            "is_empty": member.uncompressed_size == 0,
                            "line_count": line_count,
                            "first_line_columns": first_line_columns,
                            "content_error": content_error,
                        }
                    )
        except (OSError, zipfile.BadZipFile) as error:
            rows.extend(
                {
                    "archive_name": archive_name,
                    "member_path": member.member_path,
                    "split": _split_for_path(member.member_path),
                    "is_empty": member.uncompressed_size == 0,
                    "line_count": "",
                    "first_line_columns": "",
                    "content_error": str(error),
                }
                for member in archive_members
            )
    return rows


def temporal_neighbors(members: list[ArchiveMember]) -> list[dict]:
    rows = filename_patterns(members)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["possible_frame_index"] != "" and row["possible_sequence"]:
            grouped[row["possible_sequence"]].append(row)
    neighbors: list[dict] = []
    split_pairs = (("train", "val"), ("train", "test"), ("val", "test"))
    for sequence, group in grouped.items():
        for split_a, split_b in split_pairs:
            candidates = [
                (left, right)
                for left in group
                if left["split"] == split_a
                for right in group
                if right["split"] == split_b
            ]
            if not candidates:
                continue
            left, right = min(
                candidates,
                key=lambda pair: abs(
                    pair[0]["possible_frame_index"]
                    - pair[1]["possible_frame_index"]
                ),
            )
            neighbors.append(
                {
                    "frame_a": left["member_path"],
                    "split_a": split_a,
                    "frame_b": right["member_path"],
                    "split_b": split_b,
                    "delta_frame_index": abs(
                        left["possible_frame_index"] - right["possible_frame_index"]
                    ),
                    "possible_sequence": sequence,
                }
            )
    return sorted(neighbors, key=lambda row: row["delta_frame_index"])


def compare_archive_members(
    archive_members: dict[str, list[ArchiveMember]],
) -> tuple[list[dict], list[dict]]:
    """Compare image/label members using hashes and basenames."""
    overlap_rows: list[dict] = []
    relationship_rows: list[dict] = []
    names = sorted(archive_members)
    for index, archive_a in enumerate(names):
        for archive_b in names[index + 1 :]:
            for file_type in ("image", "label"):
                members_a = [
                    m for m in archive_members[archive_a] if m.file_type == file_type
                ]
                members_b = [
                    m for m in archive_members[archive_b] if m.file_type == file_type
                ]
                hashes_a = {m.sha256 for m in members_a if m.sha256}
                hashes_b = {m.sha256 for m in members_b if m.sha256}
                basenames_a = {_basename(m.member_path) for m in members_a}
                basenames_b = {_basename(m.member_path) for m in members_b}
                exact_overlap = len(hashes_a & hashes_b)
                union = len(hashes_a | hashes_b)
                jaccard = exact_overlap / union if union else 0.0
                relationship = (
                    "UNKNOWN"
                    if (members_a or members_b) and not (hashes_a or hashes_b)
                    else _relationship(len(hashes_a), len(hashes_b), exact_overlap)
                )
                row = {
                    "archive_a": archive_a,
                    "archive_b": archive_b,
                    "comparable_file_type": file_type,
                    "count_a": len(members_a),
                    "count_b": len(members_b),
                    "exact_hash_overlap": exact_overlap,
                    "basename_overlap": len(basenames_a & basenames_b),
                    "jaccard_hash": round(jaccard, 6),
                    "relationship": relationship,
                }
                overlap_rows.append(row)
                relationship_rows.append(row)
    return overlap_rows, relationship_rows


def _relationship(count_a: int, count_b: int, overlap: int) -> str:
    if not count_a and not count_b:
        return "UNKNOWN"
    if overlap == 0:
        return "NO_OVERLAP"
    if overlap == count_a == count_b:
        return "IDENTICAL_CONTENT_SET"
    if overlap == count_a:
        return "SUBSET"
    if overlap == count_b:
        return "SUPERSET"
    return "PARTIAL_OVERLAP"


def _safe_json(value):
    if isinstance(value, Path):
        return value.as_posix()
    return value


def run_inventory(
    root: Path,
    output: Path,
    inspect_archives: bool = True,
    hash_members: bool = False,
) -> dict:
    """Generate all available reports while keeping the data root read-only."""
    output.mkdir(parents=True, exist_ok=True)
    files, inspections, members = inventory_files(root, inspect_archives)
    if hash_members:
        members_by_archive: dict[str, list[ArchiveMember]] = {}
        for archive_path in sorted(root.glob("*.zip")):
            _, archive_members = inspect_archive(archive_path, hash_members=True)
            members_by_archive[archive_path.name] = archive_members
        members = [member for group in members_by_archive.values() for member in group]
    else:
        members_by_archive = defaultdict(list)
        for member in members:
            members_by_archive[member.archive_name].append(member)
    matching = match_images_labels(members)
    patterns = filename_patterns(members)
    label_rows = label_content_rows(root, members)
    image_split_members = [
        member
        for member in members
        if member.archive_name.casefold() == "imagenes.zip"
    ]
    neighbors = temporal_neighbors(image_split_members)
    overlap, relationships = compare_archive_members(dict(members_by_archive))
    archive_rows = [asdict(inspection) for inspection in inspections.values()]
    extension_rows = [
        {"archive_name": row["archive_name"], "extension": extension, "count": count}
        for row in archive_rows
        for extension, count in row["extension_counts"].items()
    ]
    archive_file_rows = []
    for item in files:
        row = asdict(item)
        inspection = inspections.get(item.filename)
        if inspection:
            row.update(asdict(inspection))
        archive_file_rows.append(row)
    _write_csv(
        output / "archives.csv",
        archive_file_rows,
        list(FileInventory.__annotations__) + list(ArchiveInspection.__annotations__),
    )
    _write_csv(
        output / "archive_members.csv",
        [asdict(member) for member in members],
        list(ArchiveMember.__annotations__),
    )
    _write_csv(
        output / "extension_counts.csv",
        extension_rows,
        ["archive_name", "extension", "count"],
    )
    _write_csv(
        output / "image_structure.csv",
        [asdict(member) for member in members if member.file_type == "image"],
        list(ArchiveMember.__annotations__),
    )
    _write_csv(
        output / "label_structure.csv",
        label_rows,
        [
            "archive_name",
            "member_path",
            "split",
            "is_empty",
            "line_count",
            "first_line_columns",
            "content_error",
        ],
    )
    _write_csv(
        output / "image_label_matching.csv",
        matching,
        list(matching[0].keys())
        if matching
        else [
            "split",
            "image_archive",
            "image_member",
            "image_basename",
            "label_archive",
            "label_member",
            "match_status",
        ],
    )
    _write_csv(
        output / "filename_patterns.csv",
        patterns,
        list(patterns[0].keys()) if patterns else ["archive_name", "member_path"],
    )
    adendo = [row for row in patterns if row["archive_name"].casefold() == "adendo.zip"]
    _write_csv(
        output / "adendo_analysis.csv",
        adendo,
        list(adendo[0].keys()) if adendo else ["archive_name", "member_path"],
    )
    _write_csv(
        output / "archive_overlap_matrix.csv",
        overlap,
        list(overlap[0].keys()) if overlap else list(_relationship_fields()),
    )
    _write_csv(
        output / "dataset_relationships.csv",
        relationships,
        list(relationships[0].keys())
        if relationships
        else list(_relationship_fields()),
    )
    _write_csv(
        output / "potential_temporal_cross_split_neighbors.csv",
        neighbors,
        [
            "frame_a",
            "split_a",
            "frame_b",
            "split_b",
            "delta_frame_index",
            "possible_sequence",
        ],
    )
    summary = {
        "root_description": "external FLIR_DATA_ROOT; absolute path intentionally omitted",
        "file_count": len(files),
        "archive_count": len(inspections),
        "archives_readable": sum(1 for item in inspections.values() if item.readable),
        "member_count": len(members),
        "image_count": sum(member.file_type == "image" for member in members),
        "label_count": sum(member.file_type == "label" for member in members),
        "matching_rows": len(matching),
        "potential_temporal_neighbor_count": len(neighbors),
        "hash_members": hash_members,
        "read_only": True,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    _write_inventory_markdown(output, files, inspections, members)
    _write_findings(
        output,
        files,
        inspections,
        members,
        matching,
        neighbors,
        relationships,
    )
    return summary


def _relationship_fields() -> tuple[str, ...]:
    return (
        "archive_a",
        "archive_b",
        "comparable_file_type",
        "count_a",
        "count_b",
        "exact_hash_overlap",
        "basename_overlap",
        "jaccard_hash",
        "relationship",
    )


def _write_inventory_markdown(output: Path, files, inspections, members) -> None:
    lines = [
        "# FLIR data inventory",
        "",
        "This report was generated read-only; absolute local paths are omitted.",
        "",
        "## Files",
    ]
    for item in files:
        lines.append(
            f"- `{item.relative_path}`: {item.size_bytes} bytes, {item.file_type}, archive_readable={item.archive_readable}"
        )
    lines.extend(["", "## Archives"])
    for inspection in inspections.values():
        lines.append(
            f"- `{inspection.archive_name}`: {inspection.total_files} files, {inspection.total_directories} directories, extensions={inspection.extensions}, depth={inspection.maximum_observed_depth}"
        )
        if inspection.error:
            lines.append(f"  - error: {inspection.error}")
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "Member contents were not extracted or semantically decoded. TXT files were only classified structurally; frame indexes are exploratory filename features.",
        ]
    )
    (output / "inventory.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_findings(
    output: Path, files, inspections, members, matching, neighbors, relationships
) -> None:
    image_members = [member for member in members if member.file_type == "image"]
    label_members = [member for member in members if member.file_type == "label"]
    image_zip = [
        member
        for member in image_members
        if member.archive_name.casefold() == "imagenes.zip"
    ]
    labels_zip = [
        member
        for member in label_members
        if member.archive_name.casefold() == "etiquetas.zip"
    ]
    primary_members = [
        member
        for member in members
        if member.archive_name.casefold() in {"imagenes.zip", "etiquetas.zip"}
    ]
    primary_matching = match_images_labels(primary_members)
    matched = sum(row["match_status"] == "matched" for row in primary_matching)
    image_without_label = sum(
        row["match_status"] == "image_without_label" for row in primary_matching
    )
    label_without_image = sum(
        row["match_status"] == "label_without_image" for row in primary_matching
    )
    validation_count = sum(
        "/validation/" in f"/{member.member_path.casefold()}/"
        for member in labels_zip
    )
    video_zip = [
        member
        for member in members
        if member.archive_name.casefold() == "video_13min_778.zip"
    ]
    def relation(archive_a: str, archive_b: str, file_type: str) -> dict:
        return next(
            (
                row
                for row in relationships
                if row["archive_a"] == archive_a
                and row["archive_b"] == archive_b
                and row["comparable_file_type"] == file_type
            ),
            {},
        )

    balance_images = relation("Dataset_Balanceado.zip", "Imagenes.zip", "image")
    split_images = relation(
        "Dataset_Balanceado.zip", "dataset_split_completo.zip", "image"
    )
    adendo_balance = relation("Adendo.zip", "Dataset_Balanceado.zip", "image")
    lines = [
        "# Findings",
        "",
        "## Hechos observados",
        "",
        f"- Imagenes.zip contiene {len(image_zip)} miembros clasificados como imágenes; esto responde empíricamente si son 1657, sin asumir validez de píxeles.",
        f"- Distribución de Imagenes.zip por split: {dict(Counter(_split_for_path(member.member_path) for member in image_zip))}.",
        f"- Etiquetas.zip contiene {len(labels_zip)} miembros clasificados como labels/textos.",
        f"- En Etiquetas.zip hay {validation_count} miembros bajo validation, además de los splits train/test; en este ZIP no son una sola anotación.",
        f"- Entre Imagenes.zip y Etiquetas.zip: {matched} basenames coinciden, {image_without_label} imágenes no tienen label homónimo y {label_without_image} labels no tienen imagen homónima.",
        f"- Adendo.zip contiene {sum(member.file_type == 'image' for member in members if member.archive_name.casefold() == 'adendo.zip')} imágenes y {sum(member.file_type == 'label' for member in members if member.archive_name.casefold() == 'adendo.zip')} textos.",
        f"- video_13min_778.zip contiene {sum(member.file_type == 'image' for member in video_zip)} JPG y {sum(member.file_type == 'video' for member in video_zip)} videos; no se decodificó ni extrajo.",
        f"- Se detectaron {len(neighbors)} posibles vecinos temporales mínimos entre splits de Imagenes.zip; no son leakage confirmado.",
        f"- Hashes: Dataset_Balanceado↔Imagenes comparte {balance_images.get('exact_hash_overlap', 'sin dato')} imágenes; Dataset_Balanceado↔dataset_split_completo tiene relación {split_images.get('relationship', 'sin dato')} con {split_images.get('exact_hash_overlap', 'sin dato')} hashes compartidos.",
        f"- Adendo↔Dataset_Balanceado comparte {adendo_balance.get('exact_hash_overlap', 'sin dato')} imágenes por SHA256; el nombre por sí solo no demuestra igualdad.",
        "",
        "## Inferencias",
        "",
        "- Los índices de frame y secuencias se extraen únicamente como features exploratorias de regex; la confianza se conserva por fila.",
        "- Los seis vecinos nominales son candidatos a proximidad temporal cross-split, porque sus nombres comparten secuencia e índice o índices cercanos; todavía no prueban procedencia temporal ni identidad visual.",
        "- Imagenes.zip + Etiquetas.zip parecen el candidato más cercano al conjunto reportado de 1657 frames por counts y splits, pero su correspondencia no es completa y no se declara fuente canónica.",
        "- Las relaciones de contenido solo son concluyentes cuando exact_hash_overlap fue calculado con --hash-members.",
        "- La igualdad de tamaños, nombres o estructuras no demuestra igualdad de contenido.",
        "",
        "## Hipótesis y limitaciones",
        "",
        "- La coincidencia observacional entre 778 en el nombre del ZIP y la duración contextual aproximada de 777 segundos no demuestra identidad del video.",
        "- Dataset_Balanceado y dataset_split_completo no son idénticos como conjuntos de archivos aunque comparten casi todas sus imágenes; sus labels y estructura deben investigarse antes de elegir uno.",
        "- No se puede decidir definitivamente el dataset canónico sin procedencia original, revisión de duplicados visuales, validación de imágenes y explicación de labels ausentes/adicionales.",
        "- La validación de contenido YOLO, decodificación de imágenes, metadatos de video y decisión canónica quedan para fases posteriores.",
        "- Los reportes generados desde datos reales deben revisarse antes de publicar hashes o conteos detallados.",
    ]
    (output / "findings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def archive_tree(archive_path: Path) -> list[dict]:
    """Return a normalized, non-extracting tree for one archive."""
    _, members = inspect_archive(archive_path)
    return [asdict(member) for member in members]
