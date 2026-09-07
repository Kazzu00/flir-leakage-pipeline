"""Candidate canonical manifest construction from the two source ZIPs."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import platform
import subprocess
import zipfile
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

from flir_pipeline import __version__
from flir_pipeline.data.inventory import (
    ArchiveMember,
    _basename,
    _filename_features,
    _split_for_path,
    inspect_archive,
)
from flir_pipeline.data.yolo_labels import LabelValidationResult, validate_label_bytes

MANIFEST_VERSION = "flir_canonical_candidate_v1"


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_member(archive: zipfile.ZipFile, member_name: str) -> tuple[bytes, str]:
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    with archive.open(member_name, "r") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            chunks.append(chunk)
    return b"".join(chunks), digest.hexdigest()


def _channels(image: Image.Image) -> int:
    return len(image.getbands())


def _stable_id(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _label_stats(result: LabelValidationResult) -> dict:
    return {
        "label_empty": result.label_empty,
        "label_valid": result.label_valid,
        "syntax_valid": result.syntax_valid,
        "normalized_values_valid": result.normalized_values_valid,
        "geometry_valid": result.geometry_valid,
        "num_objects": result.num_objects,
        "classes_present": "|".join(str(value) for value in result.classes_present),
        "bbox_area_mean": result.bbox_area_mean,
        "bbox_area_min": result.bbox_area_min,
        "bbox_area_max": result.bbox_area_max,
        "bbox_width_mean": result.bbox_width_mean,
        "bbox_height_mean": result.bbox_height_mean,
        "label_canonical": result.canonical,
        "label_errors": " | ".join(result.errors),
        "label_warnings": " | ".join(result.warnings),
    }


def _label_record(
    archive_name: str,
    member: ArchiveMember,
    content: bytes,
    label_sha256: str,
) -> dict:
    result = validate_label_bytes(content)
    stats = _label_stats(result)
    features = _filename_features(member)
    return {
        "source_archive": archive_name,
        "member_path": member.member_path,
        "filename": member.filename,
        "basename": _basename(member.member_path),
        "split": _split_for_path(member.member_path),
        "label_sha256": label_sha256,
        "line_count": len(content.decode("utf-8", errors="replace").splitlines()),
        "possible_sequence": features["possible_sequence"],
        "possible_frame_index": features["possible_frame_index"],
        **stats,
    }


def _inspect_labels(
    labels_archive: Path,
) -> tuple[list[dict], dict[tuple[str, str], dict]]:
    _, members = inspect_archive(labels_archive)
    label_members = [member for member in members if member.file_type == "label"]
    records: list[dict] = []
    by_key: dict[tuple[str, str], dict] = {}
    with zipfile.ZipFile(labels_archive) as archive:
        for member in tqdm(label_members, desc="Validating labels", unit="label"):
            content, label_sha256 = _read_member(archive, member.member_path)
            record = _label_record(labels_archive.name, member, content, label_sha256)
            records.append(record)
            by_key[(record["split"], record["basename"])] = record
    return records, by_key


def _image_properties(content: bytes) -> dict:
    try:
        with Image.open(io.BytesIO(content)) as image:
            image.load()
            return {
                "width": image.width,
                "height": image.height,
                "channels": _channels(image),
                "image_mode": image.mode,
                "image_format": image.format or "",
                "image_error": "",
            }
    except (OSError, ValueError, Image.DecompressionBombError) as error:
        return {
            "width": None,
            "height": None,
            "channels": None,
            "image_mode": "",
            "image_format": "",
            "image_error": str(error),
        }


def _comparison_basenames(root: Path, names: tuple[str, ...]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for name in names:
        path = root / name
        if not path.is_file():
            result[name] = set()
            continue
        try:
            _, members = inspect_archive(path)
            result[name] = {
                _basename(member.member_path)
                for member in members
                if member.file_type == "image"
            }
        except (OSError, zipfile.BadZipFile):
            result[name] = set()
    return result


def _orphan_rows(
    label_records: list[dict],
    image_keys: set[tuple[str, str]],
    comparison_basenames: dict[str, set[str]],
) -> list[dict]:
    rows = []
    for record in label_records:
        key = (record["split"], record["basename"])
        if key in image_keys:
            continue
        rows.append(
            {
                "source_archive": record["source_archive"],
                "member_path": record["member_path"],
                "filename": record["filename"],
                "basename": record["basename"],
                "split": record["split"],
                "label_sha256": record["label_sha256"],
                "line_count": record["line_count"],
                "syntax_valid": record["syntax_valid"],
                "possible_sequence": record["possible_sequence"],
                "possible_frame_index": record["possible_frame_index"],
                "matches_adendo": record["basename"] in comparison_basenames.get("Adendo.zip", set()),
                "matches_dataset_balanceado": record["basename"] in comparison_basenames.get("Dataset_Balanceado.zip", set()),
                "matches_dataset_split_completo": record["basename"] in comparison_basenames.get("dataset_split_completo.zip", set()),
            }
        )
    return rows


def _duplicate_reports(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["content_id"]].append(row)
    group_rows: list[dict] = []
    consistency_rows: list[dict] = []
    for content_id, group in sorted(groups.items()):
        if len(group) < 2:
            continue
        label_hashes = {row["label_sha256"] for row in group}
        canonicals = {row["label_canonical"] for row in group}
        all_valid = all(row["label_valid"] for row in group)
        semantic_equal = all_valid and len(canonicals) == 1
        group_id = group[0]["duplicate_group_id"]
        base = {
            "duplicate_group_id": group_id,
            "content_id": content_id,
            "occurrence_count": len(group),
            "splits": "|".join(sorted({row["original_split"] for row in group})),
            "frame_ids": "|".join(row["frame_id"] for row in group),
            "all_labels_identical_bytes": len(label_hashes) == 1,
            "all_labels_semantically_equivalent": semantic_equal,
            "annotation_conflict": not semantic_equal,
        }
        group_rows.append(base)
        consistency_rows.append(
            {
                **base,
                "label_sha256_values": "|".join(sorted(label_hashes)),
                "label_canonical_values": json.dumps(
                    [repr(value) for value in sorted(canonicals)], ensure_ascii=True
                ),
            }
        )
    return group_rows, consistency_rows


def _write_findings(
    path: Path,
    rows: list[dict],
    label_records: list[dict],
    duplicate_summary: dict,
    orphan_rows: list[dict],
) -> None:
    invalid = [row for row in label_records if not row["label_valid"]]
    empty = sum(row["label_empty"] for row in label_records)
    classes = sorted(
        {
            value
            for row in rows
            for value in row["classes_present"].split("|")
            if value
        }
    )
    lines = [
        "# Canonical candidate manifest findings",
        "",
        "## Observed facts",
        "",
        f"- The candidate manifest contains {len(rows)} image occurrences.",
        f"- {sum(bool(row['label_exists']) for row in rows)} occurrences have matched labels; {len(orphan_rows)} labels are orphaned and excluded.",
        f"- Valid labels: {sum(row['label_valid'] for row in rows)}; empty labels: {empty}; invalid labels: {len(invalid)}.",
        f"- Classes observed: {', '.join(classes) if classes else 'none'}.",
        f"- Unique content IDs: {duplicate_summary['unique_content_ids']}; cross-split duplicate records: {duplicate_summary['cross_split_duplicate_records']}.",
        f"- Cross-split duplicate content IDs: {duplicate_summary['cross_split_duplicate_content_ids']}.",
        "",
        "## Interpretation",
        "",
        "- original_split is preserved as historical metadata only; it is not an input to representation, clustering, or future split selection.",
        "- content_id identifies exact bytes; frame_id identifies one portable dataset occurrence. Duplicate records are retained.",
        "- Label consistency is assessed by both label SHA256 and sorted semantic canonicalization; no source file is rewritten.",
        "",
        "## Limitations",
        "",
        "- Frame and sequence fields remain filename-based exploratory inference, not source video or timestamp metadata.",
        "- Image validity is decode validity, not scientific or annotation correctness.",
        "- No embeddings, learned similarity, clustering, new splits, or detector training were run.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_manifest(
    images_archive: Path,
    labels_archive: Path,
    output: Path,
    report_output: Path,
    comparison_root: Path | None = None,
) -> dict:
    """Build the candidate manifest without extracting or modifying either ZIP."""
    output.parent.mkdir(parents=True, exist_ok=True)
    report_output.mkdir(parents=True, exist_ok=True)
    _, image_members = inspect_archive(images_archive)
    image_members = [member for member in image_members if member.file_type == "image"]
    label_records, labels_by_key = _inspect_labels(labels_archive)
    image_keys = {
        (_split_for_path(member.member_path), _basename(member.member_path))
        for member in image_members
    }
    rows: list[dict] = []
    with zipfile.ZipFile(images_archive) as archive:
        for member in tqdm(image_members, desc="Building manifest", unit="image"):
            content, image_sha256 = _read_member(archive, member.member_path)
            image_features = _image_properties(content)
            split = _split_for_path(member.member_path)
            label_record = labels_by_key.get((split, _basename(member.member_path)))
            pattern = _filename_features(member)
            frame_id = _stable_id(images_archive.name, member.member_path, image_sha256)
            row = {
                "frame_id": frame_id,
                "content_id": image_sha256,
                "duplicate_group_id": "",
                "source_archive": images_archive.name,
                "source_member_path": member.member_path,
                "relative_image_path": member.member_path,
                "relative_label_path": label_record["member_path"] if label_record else "",
                "image_filename": member.filename,
                "image_basename": _basename(member.member_path),
                "label_filename": label_record["filename"] if label_record else "",
                "original_split": split,
                "image_sha256": image_sha256,
                "label_sha256": label_record["label_sha256"] if label_record else "",
                "label_exists": label_record is not None,
                "manifest_version": MANIFEST_VERSION,
                "image_decode_valid": not image_features["image_error"],
                "possible_sequence": pattern["possible_sequence"],
                "possible_frame_index": pattern["possible_frame_index"],
                "temporal_inference_confidence": pattern["inference_confidence"],
                **image_features,
            }
            if label_record:
                row.update(
                    {
                        key: value
                        for key, value in label_record.items()
                        if key in {
                            "label_empty", "label_valid", "syntax_valid",
                            "normalized_values_valid", "geometry_valid", "num_objects",
                            "classes_present", "bbox_area_mean", "bbox_area_min",
                            "bbox_area_max", "bbox_width_mean", "bbox_height_mean",
                            "label_canonical", "label_errors", "label_warnings",
                        }
                    }
                )
            else:
                row.update(
                    {
                        "label_empty": False,
                        "label_valid": False,
                        "syntax_valid": False,
                        "normalized_values_valid": False,
                        "geometry_valid": False,
                        "num_objects": 0,
                        "classes_present": "",
                        "label_canonical": (),
                        "label_errors": "missing label",
                        "label_warnings": "",
                    }
                )
            rows.append(row)
    content_counts = Counter(row["content_id"] for row in rows)
    content_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        content_splits[row["content_id"]].add(row["original_split"])
    for row in rows:
        count = content_counts[row["content_id"]]
        splits = content_splits[row["content_id"]]
        row["duplicate_group_id"] = (
            f"duplicate-{row['content_id']}" if count > 1 else ""
        )
        row["exact_duplicate"] = count > 1
        row["cross_split_exact_duplicate"] = len(splits) > 1
        row["duplicate_occurrence_count"] = count
        row["duplicate_splits"] = "|".join(sorted(splits)) if count > 1 else ""
    duplicate_groups, consistency = _duplicate_reports(rows)
    comparison_root = comparison_root or labels_archive.parent
    comparison_basenames = _comparison_basenames(
        comparison_root,
        ("Adendo.zip", "Dataset_Balanceado.zip", "dataset_split_completo.zip"),
    )
    orphan_rows = _orphan_rows(label_records, image_keys, comparison_basenames)
    manifest_columns = [
        key for key in rows[0] if key not in {"label_canonical", "label_errors", "label_warnings", "image_error"}
    ]
    dataframe = pd.DataFrame([{key: row.get(key) for key in manifest_columns} for row in rows])
    dataframe.to_parquet(output, index=False)
    invalid_rows = [
        {
            "member_path": record["member_path"],
            "split": record["split"],
            "label_sha256": record["label_sha256"],
            "syntax_valid": record["syntax_valid"],
            "normalized_values_valid": record["normalized_values_valid"],
            "geometry_valid": record["geometry_valid"],
            "errors": record["label_errors"],
        }
        for record in label_records
        if not record["label_valid"]
    ]
    warning_rows = [
        {
            "member_path": record["member_path"],
            "split": record["split"],
            "warnings": record["label_warnings"],
        }
        for record in label_records
        if record["label_warnings"]
    ]
    image_error_rows = [
        {
            "member_path": row["source_member_path"],
            "split": row["original_split"],
            "image_error": row.get("image_error", ""),
        }
        for row in rows
        if not row["image_decode_valid"]
    ]
    _write_csv(report_output / "invalid_labels.csv", invalid_rows, list(invalid_rows[0]) if invalid_rows else ["member_path", "errors"])
    _write_csv(report_output / "geometry_warnings.csv", warning_rows, ["member_path", "split", "warnings"])
    _write_csv(report_output / "image_decode_errors.csv", image_error_rows, ["member_path", "split", "image_error"])
    class_counts = Counter()
    for row in rows:
        for class_id in row["classes_present"].split("|"):
            if class_id:
                class_counts[("all", class_id)] += 1
                class_counts[(row["original_split"], class_id)] += 1
    _write_csv(
        report_output / "class_distribution.csv",
        [
            {"split": split, "class_id": class_id, "image_count": value}
            for (split, class_id), value in sorted(class_counts.items())
        ],
        ["split", "class_id", "image_count"],
    )
    _write_csv(report_output / "orphan_labels.csv", orphan_rows, list(orphan_rows[0]) if orphan_rows else ["source_archive", "member_path"])
    _write_csv(report_output / "duplicate_groups.csv", duplicate_groups, list(duplicate_groups[0]) if duplicate_groups else ["duplicate_group_id", "content_id"])
    _write_csv(report_output / "duplicate_annotation_consistency.csv", consistency, list(consistency[0]) if consistency else ["duplicate_group_id", "content_id"])
    duplicate_content_ids = {content_id for content_id, count in content_counts.items() if count > 1}
    cross_content_ids = {row["content_id"] for row in rows if row["cross_split_exact_duplicate"]}
    split_counts = Counter(row["original_split"] for row in rows)
    train_content = {row["content_id"] for row in rows if row["original_split"] == "train"}
    val_content = {row["content_id"] for row in rows if row["original_split"] == "val"}
    test_content = {row["content_id"] for row in rows if row["original_split"] == "test"}
    cross_records = sum(row["cross_split_exact_duplicate"] for row in rows)
    test_count = split_counts["test"] or 1
    duplicate_summary = {
        "total_records": len(rows),
        "unique_content_ids": len(content_counts),
        "duplicate_content_ids": len(duplicate_content_ids),
        "duplicate_records": sum(count for count in content_counts.values() if count > 1),
        "cross_split_duplicate_content_ids": len(cross_content_ids),
        "cross_split_duplicate_records": cross_records,
        "train_val_duplicate_content_ids": len(train_content & val_content),
        "train_test_duplicate_content_ids": len(train_content & test_content),
        "val_test_duplicate_content_ids": len(val_content & test_content),
        "proportion_of_records_in_cross_split_duplicate_groups": cross_records / len(rows),
        "proportion_val_with_exact_train_duplicate": len(val_content & train_content) / (split_counts["val"] or 1),
        "proportion_test_with_exact_train_duplicate": len(test_content & train_content) / test_count,
    }
    dataset_id_items = sorted(
        (row["frame_id"], row["image_sha256"], row["label_sha256"]) for row in rows
    )
    dataset_id = _stable_id(
        MANIFEST_VERSION,
        json.dumps(dataset_id_items, separators=(",", ":"), ensure_ascii=True),
    )
    metadata = {
        "dataset_id": dataset_id,
        "manifest_version": MANIFEST_VERSION,
        "total_records": len(rows),
        "source_archives": [images_archive.name, labels_archive.name],
        "generated_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "python_version": platform.python_version(),
        "package_version": __version__,
        "hash_algorithm": "SHA256; dataset_id=SHA256(manifest_version + sorted(frame_id,image_sha256,label_sha256))",
        "original_split_counts": dict(sorted(split_counts.items())),
        "unique_content_count": len(content_counts),
        "duplicate_group_count": len(duplicate_groups),
    }
    (report_output / "manifest_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (report_output / "manifest_summary.json").write_text(
        json.dumps({**duplicate_summary, "dataset_id": dataset_id, "orphan_labels": len(orphan_rows)}, indent=2),
        encoding="utf-8",
    )
    label_summary = {
        "total_labels": len(label_records),
        "valid_labels": sum(record["label_valid"] for record in label_records),
        "invalid_labels": len(invalid_rows),
        "empty_labels": sum(record["label_empty"] for record in label_records),
        "geometry_warnings": len(warning_rows),
        "objects": sum(record["num_objects"] for record in label_records),
        "class_distribution": {
            f"{split}:{class_id}": value
            for (split, class_id), value in sorted(class_counts.items())
        },
    }
    (report_output / "label_validation_summary.json").write_text(json.dumps(label_summary, indent=2), encoding="utf-8")
    _write_findings(report_output / "manifest_findings.md", rows, label_records, duplicate_summary, orphan_rows)
    return {**duplicate_summary, "dataset_id": dataset_id, "orphan_labels": len(orphan_rows)}


def validate_labels_archive(labels_archive: Path, report_output: Path) -> dict:
    """Validate one label archive and write lightweight reports."""
    report_output.mkdir(parents=True, exist_ok=True)
    records, _ = _inspect_labels(labels_archive)
    invalid = [record for record in records if not record["label_valid"]]
    warnings = [record for record in records if record["label_warnings"]]
    _write_csv(report_output / "invalid_labels.csv", invalid, list(invalid[0]) if invalid else ["member_path", "label_errors"])
    _write_csv(report_output / "geometry_warnings.csv", warnings, list(warnings[0]) if warnings else ["member_path", "label_warnings"])
    summary = {
        "total_labels": len(records),
        "valid_labels": len(records) - len(invalid),
        "invalid_labels": len(invalid),
        "empty_labels": sum(record["label_empty"] for record in records),
        "geometry_warnings": len(warnings),
    }
    (report_output / "label_validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def manifest_summary(manifest_path: Path) -> dict:
    """Return a small summary for an existing Parquet manifest."""
    dataframe = pd.read_parquet(manifest_path, columns=["frame_id", "content_id", "original_split"])
    return {
        "total_records": len(dataframe),
        "unique_frame_ids": int(dataframe["frame_id"].nunique()),
        "unique_content_ids": int(dataframe["content_id"].nunique()),
        "original_split_counts": dataframe["original_split"].value_counts().sort_index().to_dict(),
    }
