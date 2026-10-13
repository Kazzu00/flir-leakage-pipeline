"""Read-only labeled consolidation; perceptual evidence never assigns splits."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import platform
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict
from itertools import combinations
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd
import PIL
import scipy
from PIL import Image

from flir_pipeline.data.classes import DETECTION_CLASSES
from flir_pipeline.data.identity import dataset_id_from_manifest
from flir_pipeline.data.inventory import (
    _basename,
    _filename_features,
    _member_from_info,
    _normalise_member_path,
    _split_for_path,
)
from flir_pipeline.data.manifest import _git_commit, _image_properties, _stable_id
from flir_pipeline.data.yolo_labels import validate_label_bytes
from flir_pipeline.features.diagnostics import _phash
from flir_pipeline.utils.hashing import sha256_file

VERSION = "flir_labeled_consolidation_candidate_v1"
STATES = {"visual_candidate", "confirmed_provenance", "rejected", "unresolved"}


def _json(value: object) -> str:
    return (
        json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False, indent=2)
        + "\n"
    )


def _key(row: dict) -> tuple[str, str, str]:
    # The supplement has no historical split; folder layout cannot assign one.
    return (
        row["cohort"],
        row["original_split"] if row["cohort"] == "historical" else "",
        row["basename"],
    )


def _scan(path: Path, cohort: str, kinds: set[str]) -> tuple[list[dict], list[dict]]:
    """Read ZipInfo objects, retaining raw locators and refusing ambiguous paths.

    Validate before the inventory normalizer, which intentionally strips leading
    punctuation for legacy discovery. No normalized path is used to open a ZIP.
    """
    images, labels = [], []
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        seen: set[str] = set()
        for info in infos:
            raw = info.filename.replace("\\", "/")
            normalized = _normalise_member_path(raw)
            if (
                raw.startswith("/")
                or PureWindowsPath(raw).drive
                or ".." in raw.split("/")
                or "\x00" in raw
                or (info.external_attr >> 16) & 0o170000 == 0o120000
            ):
                raise ValueError(f"Unsafe ZIP member: {path.name}:{info.filename}")
            if normalized in seen:
                raise ValueError(
                    f"Ambiguous duplicate ZIP entry: {path.name}:{info.filename}"
                )
            seen.add(normalized)
        for info in sorted(infos, key=lambda item: item.filename):
            member = _member_from_info(path.name, info)
            if member.is_directory or member.file_type not in kinds:
                continue
            payload = archive.read(info)
            digest = hashlib.sha256(payload).hexdigest()
            base = {
                "cohort": cohort,
                "source_archive": path.name,
                "source_member_path": member.member_path,
                "zip_member_name": info.filename,
                "zip_header_offset": info.header_offset,
                "basename": _basename(member.member_path),
                "original_split": _split_for_path(member.member_path)
                if cohort == "historical"
                else None,
                "source_folder_split": _split_for_path(member.member_path),
            }
            if member.file_type == "label":
                result = validate_label_bytes(payload)
                labels.append(
                    {
                        **base,
                        "label_id": _stable_id(path.name, member.member_path, digest),
                        "label_sha256": digest,
                        "validation": asdict(result),
                        "unknown_class_ids": sorted(
                            set(result.classes_present) - DETECTION_CLASSES.keys()
                        ),
                        "original_bytes_base64": base64.b64encode(payload).decode(
                            "ascii"
                        ),
                        "class_counts": dict(
                            Counter(str(box[0]) for box in result.boxes)
                        ),
                    }
                )
            else:
                properties = _image_properties(payload)
                phash = None
                if not properties["image_error"]:
                    with Image.open(io.BytesIO(payload)) as image:
                        phash = _phash(image)
                pattern = _filename_features(member)
                images.append(
                    {
                        **base,
                        "frame_id": _stable_id(path.name, member.member_path, digest),
                        "content_id": digest,
                        "image_sha256": digest,
                        "manifest_version": VERSION,
                        "phash": phash,
                        "possible_sequence": pattern["possible_sequence"],
                        "possible_frame_index": pattern["possible_frame_index"],
                        "temporal_provenance": "filename_heuristic_not_ground_truth",
                        **properties,
                    }
                )
    return images, labels


def _pair(images: list[dict], labels: list[dict]) -> None:
    by_key: dict[tuple, list[dict]] = defaultdict(list)
    images_by_key: dict[tuple, list[dict]] = defaultdict(list)
    for label in labels:
        by_key[_key(label)].append(label)
    for image in images:
        images_by_key[_key(image)].append(image)
    for image in images:
        matches = by_key[_key(image)]
        unique = len(matches) == 1 and len(images_by_key[_key(image)]) == 1
        image.update(
            {
                "label_ids": [label["label_id"] for label in matches],
                "pairing_status": "matched"
                if unique
                else "ambiguous"
                if matches
                else "missing",
                "label_sha256": matches[0]["label_sha256"] if unique else "",
                "label_valid": unique and matches[0]["validation"]["label_valid"],
                "label_empty": unique and matches[0]["validation"]["label_empty"],
                "image_decode_valid": not image["image_error"],
            }
        )
    for label in labels:
        matches = images_by_key[_key(label)]
        label.update(
            {
                "frame_ids": [image["frame_id"] for image in matches],
                "pairing_status": "orphan"
                if not matches
                else "matched"
                if len(matches) == 1 and len(by_key[_key(label)]) == 1
                else "ambiguous",
            }
        )


def _annotation_difference(left: dict, right: dict, labels: dict[str, dict]) -> dict:
    def describe(image: dict) -> list[dict]:
        return [
            {
                "label_id": key,
                "label_sha256": labels[key]["label_sha256"],
                "class_counts": labels[key]["class_counts"],
                "num_objects": labels[key]["validation"]["num_objects"],
            }
            for key in image["label_ids"]
        ]

    comparable = left["label_valid"] and right["label_valid"]
    equivalent = (
        (
            labels[left["label_ids"][0]]["validation"]["canonical"]
            == labels[right["label_ids"][0]]["validation"]["canonical"]
        )
        if comparable
        else None
    )
    return {
        "left": describe(left),
        "right": describe(right),
        "semantically_equivalent": equivalent,
        "annotation_conflict": not equivalent if comparable else None,
    }


def _edge(left: dict, right: dict, state: str, evidence: dict, labels: dict) -> dict:
    return {
        "relation_id": _stable_id(
            left["frame_id"], right["frame_id"], state, _json(evidence)
        ),
        "left_frame_id": left["frame_id"],
        "right_frame_id": right["frame_id"],
        "left_content_id": left["content_id"],
        "right_content_id": right["content_id"],
        "state": state,
        "observed_phash_distance": (
            int(left["phash"], 16) ^ int(right["phash"], 16)
        ).bit_count()
        if left["phash"] is not None and right["phash"] is not None
        else None,
        "review_status": "confirmed_by_bytes"
        if state == "exact_bytes"
        else "pending"
        if state in {"visual_candidate", "unresolved"}
        else "reviewed",
        "evidence": evidence,
        "annotation_difference": _annotation_difference(left, right, labels),
        "split_constraint_applied": False,
    }


def _import_evidence(path: Path | None, images: list[dict], labels: dict) -> list[dict]:
    if path is None:
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != "adendo_evidence_v1"
        or not isinstance(payload.get("relations"), list)
    ):
        raise ValueError("Expected adendo_evidence_v1 with a relations list")
    edges = []
    for item in payload["relations"]:
        if not isinstance(item, dict) or item.get("state") not in STATES:
            raise ValueError(
                "Invalid imported evidence state; exact identity is computed from bytes only"
            )
        if any(
            not isinstance(item.get(key), str) or not item[key].strip()
            for key in (
                "source",
                "method",
                "adendo_basename",
                "historical_basename",
                "historical_split",
            )
        ):
            raise ValueError("Evidence requires explicit endpoints, source and method")
        if item["state"] in {"confirmed_provenance", "rejected"} and any(
            not isinstance(item.get(key), str) or not item[key].strip()
            for key in ("reviewer", "reviewed_at", "additional_evidence")
        ):
            raise ValueError(
                "Reviewed relationships require reviewer, reviewed_at and additional_evidence"
            )
        score = item.get("score")
        if score is not None and (
            type(score) not in (int, float) or not np.isfinite(score)
        ):
            raise ValueError("Evidence score must be finite or null")
        left = [
            row
            for row in images
            if row["cohort"] == "adendo"
            and row["basename"] == item["adendo_basename"].casefold()
        ]
        right = [
            row
            for row in images
            if row["cohort"] == "historical"
            and row["basename"] == item["historical_basename"].casefold()
            and row["original_split"] == item["historical_split"]
        ]
        if len(left) != 1 or len(right) != 1:
            raise ValueError("Imported evidence endpoint missing or ambiguous")
        edges.append(
            _edge(
                left[0],
                right[0],
                item["state"],
                {
                    **item,
                    "import_sha256": sha256_file(path),
                    "origin": "imported_review",
                },
                labels,
            )
        )
    return edges


def audit_adendo(
    images_zip: Path,
    labels_zip: Path,
    adendo_zip: Path,
    output: Path,
    *,
    evidence: Path | None = None,
    phash_threshold: int = 8,
) -> dict:
    """Publish a fresh candidate audit after all read/identity checks succeed.

    Invalid/missing/ambiguous annotations remain in the audit and block readiness.
    Dataset identity follows the existing recipe; audit identity additionally binds
    all ZIP bytes (including orphan/ambiguous labels), evidence and configuration.
    Output must not exist; summary.json is the completion receipt, written last.
    """
    if type(phash_threshold) is not int or not 0 <= phash_threshold <= 64:
        raise ValueError("pHash threshold must be an integer in [0, 64]")
    sources = {
        "historical_images": images_zip,
        "historical_labels": labels_zip,
        "adendo": adendo_zip,
    }
    if (
        len({path.resolve() for path in sources.values()}) != 3
        or len({path.name for path in sources.values()}) != 3
    ):
        raise ValueError("Three distinct archives with distinct basenames are required")
    if output.exists() or any(
        output.resolve() == path.resolve() or output.resolve() in path.resolve().parents
        for path in [*sources.values(), *([evidence] if evidence else [])]
    ):
        raise ValueError("Output must be a fresh directory separate from input files")
    hashes = {
        role: {"name": path.name, "sha256": sha256_file(path)}
        for role, path in sources.items()
    }
    evidence_hash = sha256_file(evidence) if evidence else None
    historical, _ = _scan(images_zip, "historical", {"image"})
    _, historical_labels = _scan(labels_zip, "historical", {"label"})
    supplement, supplement_labels = _scan(adendo_zip, "adendo", {"image", "label"})
    images, labels = historical + supplement, historical_labels + supplement_labels
    if not historical or not supplement:
        raise ValueError(
            "Both historical and Adendo image populations must be nonempty"
        )
    _pair(images, labels)
    label_index = {row["label_id"]: row for row in labels}
    relations = _import_evidence(evidence, images, label_index)
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in images:
        groups[row["content_id"]].append(row)
    for group in groups.values():
        for left, right in combinations(group, 2):
            relations.append(
                _edge(
                    left,
                    right,
                    "exact_bytes",
                    {"source": "input_zip_bytes", "method": "sha256", "score": 1},
                    label_index,
                )
            )
    # Compare unique contents, then expand to every historical occurrence. No top-1
    # choice can hide that a supplement image has candidates in several splits.
    distances = {}
    for left in supplement:
        for right in historical:
            if (
                left["content_id"] == right["content_id"]
                or left["phash"] is None
                or right["phash"] is None
            ):
                continue
            key = left["content_id"], right["content_id"]
            if key not in distances:
                distances[key] = (
                    int(left["phash"], 16) ^ int(right["phash"], 16)
                ).bit_count()
            if distances[key] <= phash_threshold:
                relations.append(
                    _edge(
                        left,
                        right,
                        "visual_candidate",
                        {
                            "source": "input_images",
                            "method": "phash64_hamming",
                            "score": distances[key],
                            "threshold": phash_threshold,
                        },
                        label_index,
                    )
                )
    images.sort(key=lambda row: row["frame_id"])
    labels.sort(key=lambda row: row["label_id"])
    relations.sort(key=lambda row: row["relation_id"])
    if len({row["relation_id"] for row in relations}) != len(relations):
        raise ValueError("Duplicate imported relation evidence")
    manifest = pd.DataFrame(images)
    dataset_id = dataset_id_from_manifest(manifest)
    config = {
        "schema_version": VERSION,
        "sources": hashes,
        "evidence_sha256": evidence_hash,
        "phash_threshold": phash_threshold,
        "phash_implementation": "features.diagnostics._phash:grayscale32,dct2_ortho,8x8,median[1:,1:]",
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pillow": PIL.__version__,
            "scipy": scipy.__version__,
        },
        "git_commit": _git_commit(),
        "implementation_sha256": {
            name: sha256_file(Path(__file__).with_name(name))
            for name in (
                "adendo.py",
                "inventory.py",
                "manifest.py",
                "identity.py",
                "yolo_labels.py",
                "classes.py",
            )
        }
        | {
            "diagnostics.py": sha256_file(
                Path(__file__).parents[1] / "features" / "diagnostics.py"
            )
        },
    }
    audit_id = hashlib.sha256(_json(config).encode()).hexdigest()
    counts = {}
    for cohort in ("historical", "adendo"):
        cohort_images = [row for row in images if row["cohort"] == cohort]
        cohort_labels = [row for row in labels if row["cohort"] == cohort]
        matched = [row for row in cohort_labels if row["pairing_status"] == "matched"]
        counts[cohort] = {
            "images": len(cohort_images),
            "unique_contents": len({row["content_id"] for row in cohort_images}),
            "labels": len(cohort_labels),
            "pairing": dict(Counter(row["pairing_status"] for row in cohort_images)),
            "orphan_labels": sum(
                row["pairing_status"] == "orphan" for row in cohort_labels
            ),
            "invalid_labels": sum(
                not row["validation"]["label_valid"] for row in cohort_labels
            ),
            "syntax_invalid_labels": sum(
                not row["validation"]["syntax_valid"] for row in cohort_labels
            ),
            "geometry_invalid_labels": sum(
                not row["validation"]["geometry_valid"] for row in cohort_labels
            ),
            "unknown_class_labels": sum(
                bool(row["unknown_class_ids"]) for row in cohort_labels
            ),
            "empty_matched_labels": sum(
                row["validation"]["label_empty"] for row in matched
            ),
            "matched_objects": sum(row["validation"]["num_objects"] for row in matched),
            "all_label_objects": sum(
                row["validation"]["num_objects"] for row in cohort_labels
            ),
            "matched_class_counts": dict(
                sum((Counter(row["class_counts"]) for row in matched), Counter())
            ),
            "original_splits": dict(
                Counter(
                    row["original_split"]
                    for row in cohort_images
                    if row["original_split"] is not None
                )
            ),
        }
    warnings = [
        "Candidate audit only: no splits, dependency groups or training produced.",
        "pHash candidates are not ground truth; no candidate does not establish independence.",
        "Imported reviews are attributed assertions, not independently validated provenance.",
        "Filename sequence/frame/scene information is heuristic; timestamps are not inferred.",
    ]
    issues = Counter(
        row["pairing_status"] for row in images if row["pairing_status"] != "matched"
    )
    if issues or any(
        not row["label_valid"] or not row["image_decode_valid"] for row in images
    ):
        warnings.append(
            "Missing, ambiguous, invalid labels or undecodable images require review."
        )
    if any(row["pairing_status"] == "orphan" for row in labels):
        warnings.append(
            "Orphan labels retained separately and excluded from the image population."
        )
    conflicts = sum(
        row["annotation_difference"]["annotation_conflict"] is True
        for row in relations
        if row["state"] == "exact_bytes"
    )
    if conflicts:
        warnings.append(
            "Exact duplicates have conflicting annotations; no labels were merged."
        )
    duplicate_groups = [group for group in groups.values() if len(group) > 1]
    candidate_relations = [
        row for row in relations if row["state"] in {"visual_candidate", "unresolved"}
    ]
    candidate_pairs = {
        (row["left_frame_id"], row["right_frame_id"]) for row in candidate_relations
    }
    summary = {
        "schema_version": VERSION,
        "dataset_id": dataset_id,
        "audit_id": audit_id,
        "configuration": config,
        "read_only": True,
        "counts": counts,
        "total_occurrences": len(images),
        "unique_contents": len(groups),
        "exact_duplicate_groups": len(duplicate_groups),
        "exact_duplicate_occurrences": sum(len(group) for group in duplicate_groups),
        "exact_duplicate_additional_occurrences": sum(
            len(group) - 1 for group in duplicate_groups
        ),
        "exact_duplicate_relation_pairs": sum(
            len(group) * (len(group) - 1) // 2 for group in duplicate_groups
        ),
        "exact_annotation_conflict_pairs": conflicts,
        "cross_cohort_exact_contents": len(
            {row["content_id"] for row in historical}
            & {row["content_id"] for row in supplement}
        ),
        "relation_states": dict(Counter(row["state"] for row in relations)),
        "candidate_unique_pairs": len(candidate_pairs),
        "candidate_evidence_records": len(candidate_relations),
        "phash_candidate_evidence_records": sum(
            row["evidence"].get("method") == "phash64_hamming"
            for row in candidate_relations
        ),
        "imported_candidate_evidence_records": sum(
            row["evidence"].get("origin") == "imported_review"
            for row in candidate_relations
        ),
        "annotation_qa_passed": all(
            row["label_valid"] and row["image_decode_valid"] for row in images
        ),
        "split_ready": False,
        "warnings": warnings,
    }
    for role, path in sources.items():
        if sha256_file(path) != hashes[role]["sha256"]:
            raise ValueError("Source changed during audit; publication refused")
    if evidence and sha256_file(evidence) != evidence_hash:
        raise ValueError("Evidence changed during audit; publication refused")
    output.mkdir(parents=True, exist_ok=False)
    manifest.to_parquet(output / "candidate_manifest.parquet", index=False)
    for name, rows in (
        ("candidate_manifest.json", images),
        ("annotation_audit.json", labels),
        ("relations.json", relations),
    ):
        (output / name).write_text(
            _json(
                {"schema_version": VERSION, "dataset_id": dataset_id, "records": rows}
            ),
            encoding="utf-8",
        )
    (output / "README.md").write_text(
        "# Auditoría candidata de datos etiquetados\n\n"
        "Manifest: una fila por ocurrencia; content_id identifica bytes exactos. "
        "label_ids enlaza todas las etiquetas posibles; no se resuelve ambigüedad automáticamente.\n\n"
        "annotation_audit conserva bytes originales (base64), cajas y validación, incluidos huérfanos. "
        "relations contiene evidencia y diferencias de anotación; ninguna arista asigna splits. "
        "Adendo no tiene original_split. confirmed_provenance requiere evidencia de revisión importada.\n\n"
        "No se demuestra independencia mediante pHash ni se valida ground truth temporal. "
        "summary.json es el recibo final con checksums; una carpeta sin él está incompleta.\n",
        encoding="utf-8",
    )
    summary["artifacts"] = {
        path.name: sha256_file(path) for path in sorted(output.iterdir())
    }
    (output / "summary.json").write_text(_json(summary), encoding="utf-8")
    return summary
