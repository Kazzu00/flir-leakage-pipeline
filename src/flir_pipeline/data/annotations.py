"""Recompute occurrence-level annotation counts directly from read-only label ZIPs."""

from __future__ import annotations

import hashlib
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from flir_pipeline.data.yolo_labels import validate_label_bytes


@dataclass
class AnnotationAudit:
    """Separate image presence, object instances and excluded-label quality facts."""

    classes: pd.DataFrame
    duplicate_groups: pd.DataFrame
    summary: dict[str, int]


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
            matched_names.add(name)
            matched_results[row.frame_id] = result
            image_counts.update(result.classes_present)
            object_counts.update(box[0] for box in result.canonical)
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
    }
    classes = pd.DataFrame([
        {"class_id": class_id, "images_containing_class": image_counts[class_id], "object_instances": object_counts[class_id]}
        for class_id in sorted(image_counts.keys() | object_counts.keys())
    ], columns=["class_id", "images_containing_class", "object_instances"])
    return AnnotationAudit(classes, duplicates, summary)
