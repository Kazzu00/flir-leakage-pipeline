"""Descriptive annotation support: no independence claims or sufficiency cutoffs."""

from __future__ import annotations

from datetime import date
from itertools import combinations

import numpy as np

from flir_pipeline.data.classes import class_name
from flir_pipeline.data.yolo_labels import validate_label_bytes
from flir_pipeline.detection.common_test import class_support
from flir_pipeline.similarity.storage import stable_id
from flir_pipeline.splitting.labeled import LabeledConfig, sensitivity_groups

VERSION = "ground_truth_quality_v1"
REVIEW_FIELDS = (
    "frame_id",
    "state",
    "reviewer",
    "review_date",
    "annotation_criteria",
    "visual_observations",
    "possible_omissions",
    "possible_class_confusions",
    "possible_grouped_objects",
    "possible_partial_boxes",
    "video_evidence",
    "justification",
    "proposed_action",
)


def annotation_facts(raw, width, height):
    """Retain source precision and order; geometric facts are not label defects."""
    result = validate_label_bytes(raw)
    if not result.label_valid or width <= 0 or height <= 0:
        raise ValueError("Invalid YOLO geometry or image dimensions")
    boxes = []
    for i, (c, x, y, w, h) in enumerate(result.boxes):
        class_name(c)  # The generic YOLO parser deliberately has no ontology.
        boxes.append(
            {
                "box_index": i,
                "class_id": c,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "area_fraction": w * h,
                "area_pixels": w * width * h * height,
                "touches_boundary": min(x - w / 2, y - h / 2) == 0
                or max(x + w / 2, y + h / 2) == 1,
                "identical_previous_box": (c, x, y, w, h) in result.boxes[:i],
            }
        )
    return {
        "boxes": boxes,
        "empty": result.label_empty,
        "classes": result.classes_present,
        "box_count": result.num_objects,
        "area_min": result.bbox_area_min,
        "area_mean": result.bbox_area_mean,
        "area_max": result.bbox_area_max,
        "automatic_missing_object_detection": False,
    }


def perceptual_relations(records, threshold):
    """Fresh within-set screening; never promote these edges to split constraints.

    The consolidation's imported/cross-cohort pairs are not exhaustive within the
    historical test. Reuse its recorded pHash64 and screening threshold explicitly.
    """
    if type(threshold) is not int or not 0 <= threshold <= 64:
        raise ValueError("Invalid recorded pHash screening threshold")
    unique = records.sort_values("frame_id").drop_duplicates("content_id")
    hashes = {}
    for row in records.to_dict(orient="records"):
        value = row.get("phash")
        if (
            not isinstance(value, str)
            or len(value) != 16
            or any(c not in "0123456789abcdef" for c in value)
        ):
            raise ValueError("Missing or invalid recorded pHash64")
        if row["content_id"] in hashes and hashes[row["content_id"]] != value:
            raise ValueError("Exact copies have different pHash")
        hashes[row["content_id"]] = value
    pairs = []
    for a, b in combinations(unique.to_dict(orient="records"), 2):
        distance = (int(a["phash"], 16) ^ int(b["phash"], 16)).bit_count()
        if distance <= threshold:
            pairs.append(
                {
                    "left_frame_id": a["frame_id"],
                    "right_frame_id": b["frame_id"],
                    "left_content_id": a["content_id"],
                    "right_content_id": b["content_id"],
                    "relation_id": "diagnostic-"
                    + stable_id([a["content_id"], b["content_id"], threshold]),
                    "state": "visual_candidate",
                    "hamming_distance": distance,
                    "method": "recorded_phash64_within_subset",
                    "threshold": threshold,
                    "used_to_construct_splits": False,
                    "confirmed_provenance": False,
                }
            )
    return pairs


def descriptive_support(records, statistics, relations, scope, *, phash_threshold=None):
    """Induce candidates on this evaluated subset, never through excluded nodes.

    Candidate components are a sensitivity hypothesis, not inferred sequences or
    effective sample sizes. Removing a group is arithmetic only, not a new split.
    Conflicting per-occurrence annotations are retained in instance counts.
    """
    groups, _, selection = sensitivity_groups(records, relations, "S3", LabeledConfig())
    membership = groups[["content_id", "group_id"]].copy()
    # Stable identities independent of the helper's internal component numbering.
    ids = {
        g: "candidate-" + stable_id(sorted(rows.content_id))
        for g, rows in membership.groupby("group_id", sort=True)
    }
    membership["candidate_group"] = membership.group_id.map(ids)
    screening = []
    if phash_threshold is not None:
        screening = perceptual_relations(records, phash_threshold)
        rejected = [e for e in relations if e["state"] == "rejected"]
        screened, _, _ = sensitivity_groups(
            records, [*screening, *rejected], "S3", LabeledConfig()
        )
        screened_ids = {
            g: "screened-" + stable_id(sorted(rows.content_id))
            for g, rows in screened.groupby("group_id", sort=True)
        }
        screened["perceptual_group"] = screened.group_id.map(screened_ids)
        membership = membership.merge(
            screened[["content_id", "perceptual_group"]],
            on="content_id",
            validate="one_to_one",
        )
    data = (
        records[["frame_id", "content_id"]]
        .merge(
            statistics.drop(columns="content_id"), on="frame_id", validate="one_to_one"
        )
        .merge(
            membership.drop(columns="group_id"),
            on="content_id",
            validate="many_to_one",
        )
    )
    support, concentration, sensitivity = [], [], []
    for row in class_support(records, statistics):
        c = row["class_id"]
        positives = data.loc[data[f"class_{c}"] > 0]
        item = {
            "scope": scope,
            **row,
            "class_name": class_name(c),
            "coverage_positive": row["instances"] > 0,
            "single_positive_image": row["images"] == 1,
            "statistical_sufficiency": "NOT_ESTABLISHED",
            "temporal_independence": "NOT_DEMONSTRATED",
            "checkpoint_selection_eligible": False,
            "ready_for_training": False,
        }
        levels = [
            ("exact", "content_id"),
            ("candidate", "candidate_group"),
        ]
        if phash_threshold is not None:
            levels.append(("perceptual", "perceptual_group"))
        for level, column in levels:
            counts = positives.groupby(column, sort=True).agg(
                instances=(f"class_{c}", "sum"),
                images=("frame_id", "size"),
                contents=("content_id", "nunique"),
            )
            item[f"{level}_positive_groups"] = len(counts)
            item[f"{level}_max_instance_share"] = (
                float(counts.instances.max() / row["instances"])
                if len(counts)
                else None
            )
            for group, values in counts.iterrows():
                concentration.append(
                    {
                        "scope": scope,
                        "class_id": c,
                        "level": level,
                        "group": group,
                        **{k: int(v) for k, v in values.items()},
                    }
                )
                if c in (0, 4):
                    sensitivity.append(
                        {
                            "scope": scope,
                            "class_id": c,
                            "level": level,
                            "removed_group": group,
                            "remaining_instances": row["instances"]
                            - int(values.instances),
                            "remaining_images": row["images"] - int(values.images),
                            "remaining_contents": row["contents"]
                            - int(values.contents),
                            "removed_instance_fraction": float(
                                values.instances / row["instances"]
                            ),
                            "interpretation": "descriptive_support_only_not_precision_or_new_split",
                        }
                    )
        support.append(item)
    columns = ["frame_id", "content_id", "candidate_group"]
    if phash_threshold is not None:
        columns.append("perceptual_group")
        selection["new_within_subset_screening"] = {
            "threshold": phash_threshold,
            "relations": screening,
            "interpretation": "diagnostic_only_not_split_constraints_or_independent_sequences",
        }
    mapping = data[columns].copy()
    mapping.insert(0, "scope", scope)
    return support, concentration, sensitivity, mapping, selection


def area_summary(instances):
    rows = []
    for c in range(5):
        values = [r["area_fraction"] for r in instances if r["class_id"] == c]
        rows.append(
            {
                "class_id": c,
                "class_name": class_name(c),
                "instances": len(values),
                **{
                    key: float(np.quantile(values, q)) if values else None
                    for key, q in (
                        ("min", 0),
                        ("q25", 0.25),
                        ("median", 0.5),
                        ("q75", 0.75),
                        ("max", 1),
                    )
                },
            }
        )
    return rows


def annotation_sensitivity(test_support, cases):
    """Alternative stored labels, not preferred labels or corrected ground truth."""
    rows = []
    for c in range(5):
        baseline = next(r["instances"] for r in test_support if r["class_id"] == c)
        delta_sum = 0
        for case in cases:
            a = sum(b[0] == c for b in case["A"]["boxes"])
            b = sum(b[0] == c for b in case["B"]["boxes"])
            delta_sum += b - a
            rows.append(
                {
                    "class_id": c,
                    "scenario": case["conflict_id"] + ":B_only",
                    "current_test_instances": baseline,
                    "hypothetical_instances": baseline + b - a,
                    "delta_instances": b - a,
                    "applied": False,
                }
            )
        rows.append(
            {
                "class_id": c,
                "scenario": "all_stored_B_annotations",
                "current_test_instances": baseline,
                "hypothetical_instances": baseline + delta_sum,
                "delta_instances": delta_sum,
                "applied": False,
            }
        )
    return rows


def review_template(audit_id, frame_ids):
    return {
        "audit_id": audit_id,
        "cases": [
            {**dict.fromkeys(REVIEW_FIELDS, ""), "frame_id": f, "state": "pendiente"}
            for f in sorted(frame_ids)
        ],
    }


def validate_human_review(document, audit_id, frame_ids):
    """Accept pending cases without coercion; observations never adjudicate labels."""
    cases = document.get("cases", [])
    if (
        document.get("audit_id") != audit_id
        or len(cases) != len(frame_ids)
        or {r.get("frame_id") for r in cases} != set(frame_ids)
    ):
        raise ValueError("Human review scope/identity mismatch")
    for row in cases:
        if set(row) != set(REVIEW_FIELDS) or any(
            not isinstance(v, str) for v in row.values()
        ):
            raise ValueError("Invalid human review fields")
        if row["state"] not in {
            "pendiente",
            "observaciones_registradas",
            "propuesta_de_correccion",
        }:
            raise ValueError("Review cannot approve or adjudicate")
        if row["state"] != "pendiente":
            if any(
                not row[k].strip()
                for k in (
                    "reviewer",
                    "review_date",
                    "annotation_criteria",
                    "justification",
                    "visual_observations",
                )
            ):
                raise ValueError("Incomplete attributed human observation")
            date.fromisoformat(row["review_date"])
    return {
        "pending": sum(r["state"] == "pendiente" for r in cases),
        "ready_for_training": False,
        "adjudicated_labels_applied": False,
    }
