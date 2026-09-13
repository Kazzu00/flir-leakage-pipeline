"""Recompute occurrence-level annotation counts directly from read-only label ZIPs."""

from __future__ import annotations

import hashlib
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from flir_pipeline.data.classes import class_name
from flir_pipeline.data.yolo_labels import validate_label_bytes


@dataclass
class AnnotationAudit:
    """Separate image presence, object instances and excluded-label quality facts."""

    classes: pd.DataFrame
    duplicate_groups: pd.DataFrame
    summary: dict[str, int]
    instances: pd.DataFrame
    geometry: pd.DataFrame


BBOX_METRICS = ("normalized_width", "normalized_height", "normalized_area", "aspect_ratio")


def summarize_bbox_geometry(instances: pd.DataFrame) -> pd.DataFrame:
    """Instance statistics with sample std (ddof=1) and linear quartiles.

    A singleton has undefined sample std (NaN), not an invented zero variance.
    YOLO width/height ratio is normalized-axis geometry, not pixel aspect ratio
    when image width and height differ. Empty labels add no observations.
    """
    rows = []
    for class_id, group in instances.groupby("class_id", sort=True):
        for metric in BBOX_METRICS:
            values = group[metric]
            rows.append({
                "class_id": class_id, "class_name": class_name(class_id), "metric": metric,
                "count": len(values), "mean": values.mean(), "std": values.std(ddof=1),
                "median": values.median(), "Q1": values.quantile(0.25), "Q3": values.quantile(0.75),
                "min": values.min(), "max": values.max(),
            })
    return pd.DataFrame(rows, columns=["class_id", "class_name", "metric", "count", "mean", "std", "median", "Q1", "Q3", "min", "max"])


def audit_annotations(manifest: pd.DataFrame, labels_archive: Path) -> AnnotationAudit:
    """Count boxes per historical occurrence, preserving annotation conflicts.

    Read each label member once, verify matched SHA256 against the manifest, and
    reuse the parsed result for matching occurrences. Counts are record-level,
    not deduplicated content counts. Orphan labels are audited separately and
    never included in the candidate's instance distribution. No source is edited.
    """
    required = {"frame_id", "content_id", "relative_label_path", "label_sha256", "num_objects"}
    if not required <= set(manifest) or not manifest["frame_id"].is_unique:
        raise ValueError("Annotation audit requires canonical occurrence and label lineage columns")
    image_counts, object_counts = Counter(), Counter()
    matched_results = {}
    instance_rows = []
    excluded_geometry_objects = 0
    with zipfile.ZipFile(labels_archive) as archive:
        names = [info.filename for info in archive.infolist() if not info.is_dir() and info.filename.lower().endswith(".txt")]
        if len(names) != len(set(names)):
            raise ValueError("Ambiguous repeated label member names in ZIP")
        parsed = {}
        hashes = {}
        for name in names:
            content = archive.read(name)
            parsed[name] = validate_label_bytes(content)
            hashes[name] = hashlib.sha256(content).hexdigest()
        matched_names = set()
        missing = 0
        for row in manifest.itertuples(index=False):
            name = row.relative_label_path
            if not name or name not in parsed:
                missing += 1
                continue
            if hashes[name] != row.label_sha256:
                raise ValueError("Label bytes differ from the canonical manifest; audit stopped")
            result = parsed[name]
            if result.num_objects != row.num_objects:
                raise ValueError("Manifest object count differs from parsed label content")
            if hasattr(row, "classes_present"):
                recorded_classes = set(str(row.classes_present).split("|")) - {"", "nan", "None"}
                if recorded_classes != {str(value) for value in result.classes_present}:
                    raise ValueError("Manifest class presence differs from parsed labels")
            if hasattr(row, "label_empty") and bool(row.label_empty) != result.label_empty:
                raise ValueError("Manifest empty-label status differs from parsed labels")
            matched_names.add(name)
            matched_results[row.frame_id] = result
            image_counts.update(result.classes_present)
            object_counts.update(box[0] for box in result.boxes)
            for box_index, (class_id, _, _, width, height) in enumerate(result.boxes):
                name_for_class = class_name(class_id)
                if not result.label_valid:
                    # Preserve invalid-label QA/counts, but exclude the entire
                    # invalid annotation from geometry rather than correcting it.
                    excluded_geometry_objects += 1
                    continue
                instance_rows.append({
                    "frame_id": row.frame_id, "content_id": row.content_id,
                    "relative_label_path": name, "box_index": box_index,
                    "class_id": class_id, "class_name": name_for_class,
                    "normalized_width": width, "normalized_height": height,
                    "normalized_area": width * height, "aspect_ratio": width / height,
                })
        orphan_results = [result for name, result in parsed.items() if name not in matched_names]
    duplicate_rows = []
    for content_id, group in manifest.groupby("content_id", sort=True):
        if len(group) < 2:
            continue
        results = [matched_results.get(frame_id) for frame_id in group["frame_id"]]
        consistent = all(r is not None and r.label_valid for r in results) and len({r.canonical for r in results if r is not None}) == 1
        duplicate_rows.append({"content_id": content_id, "occurrences": len(group), "annotation_consistent": consistent})
    duplicates = pd.DataFrame(duplicate_rows, columns=["content_id", "occurrences", "annotation_consistent"])
    candidate = list(matched_results.values())
    summary = {
        "candidate_records": len(manifest), "matched_labels": len(candidate), "missing_labels": missing,
        "candidate_valid_labels": sum(r.label_valid for r in candidate),
        "candidate_invalid_labels": sum(not r.label_valid for r in candidate),
        "candidate_invalid_geometry": sum(not r.geometry_valid for r in candidate),
        "candidate_empty_labels": sum(r.label_empty for r in candidate),
        "candidate_objects": sum(r.num_objects for r in candidate),
        "orphan_labels": len(orphan_results), "orphan_objects": sum(r.num_objects for r in orphan_results),
        "archive_labels": len(parsed), "archive_objects": sum(r.num_objects for r in parsed.values()),
        "duplicate_groups": len(duplicates),
        "consistent_duplicate_groups": int(duplicates["annotation_consistent"].sum()),
        "conflicting_duplicate_groups": int((~duplicates["annotation_consistent"].astype(bool)).sum()),
        "geometry_instances": len(instance_rows),
        "geometry_excluded_invalid_label_objects": excluded_geometry_objects,
    }
    classes = pd.DataFrame([
        {"class_id": class_id, "class_name": class_name(class_id), "images_containing_class": image_counts[class_id], "object_instances": object_counts[class_id],
         "instance_count": object_counts[class_id], "percentage_of_total_instances": 100 * object_counts[class_id] / (sum(object_counts.values()) or 1)}
        for class_id in sorted(image_counts.keys() | object_counts.keys())
    ], columns=["class_id", "class_name", "images_containing_class", "object_instances", "instance_count", "percentage_of_total_instances"])
    instances = pd.DataFrame(instance_rows, columns=["frame_id", "content_id", "relative_label_path", "box_index", "class_id", "class_name", *BBOX_METRICS])
    return AnnotationAudit(classes, duplicates, summary, instances, summarize_bbox_geometry(instances))
