"""Adapt an audited paired delivery without changing canonical historical IDs.

Supplier native frame indices are not the sampled indices in Hypatia sequences.
Only exact reference-content joins carry historical evidence to the new delivery.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml

from flir_pipeline.explorer.timeline import ordered_contents
from flir_pipeline.utils.hashing import sha256_file


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def rows_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {
                k: json.dumps(v, ensure_ascii=False)
                if isinstance(v, (dict, list, tuple))
                else v
                for k, v in row.items()
            }
            for row in rows
        )


def safe_relative(root: Path, relative: str) -> Path:
    result = (root / relative).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError("Input escapes its declared root")
    return result


def load_inputs(root: Path, config_path: Path):
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if (
        cfg["protocol"] != "jp_paired_provisional_v1"
        or cfg["training_allowed"] is not False
        or cfg["publication_status"] != "provisional"
    ):
        raise ValueError(
            "This adapter only produces provisional non-training artifacts"
        )
    audit = safe_relative(root, cfg["inventory"])
    checksums = read_json(audit / "artifact_checksums.json")
    receipts = {str(config_path): sha256_file(config_path)}

    def bind(path):
        path = Path(path)
        receipts[str(path)] = sha256_file(path)
        return path

    for name in (
        "canonical_manifest.csv",
        "inventory_report.json",
        "member_inventory.json",
        "historical_correspondence.csv",
        "cross_archive_annotation_conflicts.json",
        "verification_report.json",
        "supplier_metadata.json",
    ):
        path = bind(audit / name)
        if receipts[str(path)] != checksums[name]:
            raise ValueError(f"Inventory receipt mismatch: {name}")
    # Rehash just the two inputs, not their contents/labels/inventory again.
    # This binds inherited label QA to unchanged TAR bytes as well as previews.
    for source in read_json(audit / "inventory_report.json")["sources"]:
        if source["source"].endswith(".tar"):
            path = bind(
                safe_relative(safe_relative(root, cfg["data"]), source["source"])
            )
            if receipts[str(path)] != source["sha256"]:
                raise ValueError("Source TAR differs from the validated inventory")
    manifest = rows_csv(audit / "canonical_manifest.csv")
    members = read_json(audit / "member_inventory.json")
    references = {
        r["filename"]: r for r in members if "_referencia_origen/images/" in r["member"]
    }
    historical = {
        r["imagen"]: r for r in rows_csv(audit / "historical_correspondence.csv")
    }
    conflicts = {
        r["content_sha256"]
        for r in read_json(audit / "cross_archive_annotation_conflicts.json")
    }
    hypatia = safe_relative(root, cfg["hypatia"])
    assessment = safe_relative(root, cfg["assessment"])
    decisions = []
    for name in (
        "singleton_rank1_dominant_group_manual_decisions_v1.csv",
        "singleton_rank1_group_diverse_manual_decisions_v1.csv",
    ):
        decisions.extend(rows_csv(bind(hypatia / name)))
    old_decisions = rows_csv(bind(assessment / "historical_decisions.csv"))
    if sorted(map(digest, decisions)) != sorted(map(digest, old_decisions)):
        raise ValueError(
            "Recovered decisions differ from historical assessment; reconcile explicitly"
        )
    historical_decisions = defaultdict(list)
    candidate_support = defaultdict(list)
    for row in rows_csv(bind(hypatia / "shared_group_rank_support.csv")):
        candidate_support[row["labeled_content_id"]].append(row)
    for row in decisions:
        historical_decisions[row["labeled_content_id"]].append(row)
    base = defaultdict(list)
    for row in rows_csv(bind(assessment / "base_link_assessment.csv")):
        base[row["content_id"]].append(row)
    # Validate the actual files used against the assessment's recorded checksums.
    upstream_checks = read_json(bind(assessment / "input_checksums.json"))
    for path, checksum in list(receipts.items()):
        if path in upstream_checks and checksum != upstream_checks[path]:
            raise ValueError(f"Historical evidence changed: {path}")
    membership = rows_csv(bind(hypatia / "visual_dependency_membership.csv"))
    protected = pd.read_parquet(bind(safe_relative(root, cfg["protected_records"])))
    test = pd.read_parquet(bind(safe_relative(root, cfg["test_records"])))
    protected_contents = set(protected.content_id)
    test_contents = set(test.content_id)
    protected_groups = {
        d["proposed_visual_dependency_group_id"]
        for d in decisions
        if d["labeled_content_id"] in protected_contents
        and d["manual_decision"] == "supported"
    }
    for key in ("october_decisions", "october_boundaries"):
        bind(safe_relative(root, cfg[key]))
    if cfg.get("relations"):
        bind(safe_relative(root, cfg["relations"]))
    producer_available = False
    for key in ("producer_script", "producer_receipt"):
        if cfg.get(key) and safe_relative(root, cfg[key]).is_file():
            bind(safe_relative(root, cfg[key]))
            producer_available |= key == "producer_script"
    primary = [r for r in manifest if r["role"] == "primary"]
    variants = defaultdict(dict)
    for r in primary:
        variant = (
            "original" if r["source"].endswith("/dataset_original.tar") else "no_hud"
        )
        if variant in variants[r["filename"]]:
            raise ValueError("Ambiguous primary filename")
        variants[r["filename"]][variant] = r
    pairs = []
    for name, vv in sorted(variants.items()):
        if set(vv) != {"original", "no_hud"}:
            raise ValueError("Incomplete pair must be re-audited before this adapter")
        a, b = vv["original"], vv["no_hud"]
        claims = json.loads(a["supplier_correspondence"])
        if len(claims) != 1:
            raise ValueError("Ambiguous supplier correspondence")
        claim = claims[0]
        label = json.loads(a["label_validation"])
        ref = references.get(Path(name).with_suffix(".jpg").name)
        ref_hash = ref["sha256"] if ref else ""
        hdec = historical_decisions[ref_hash]
        history = historical[name]
        reasons = []
        if not json.loads(history["exact_reference_matches"]):
            reasons.append("historical_origin_unresolved")
        if ref_hash in conflicts:
            reasons.append("historical_annotation_conflict")
        if not label["label_valid"]:
            reasons.append("strict_label_geometry")
        if "ambigua" in claim["verificado_por"]:
            reasons.append("ambiguous_temporal_correspondence")
        if ref_hash in protected_contents:
            reasons.append("protected_reference_do_not_reassign")
        candidate_groups = {
            d["proposed_visual_dependency_group_id"]
            for d in hdec
            if d["manual_decision"] == "supported"
        }
        if candidate_groups & protected_groups or any(
            r["conditional_protected_dependency"].lower() == "true"
            for r in base[ref_hash]
        ):
            reasons.append("conditional_protected_dependency")
        if (a["width"], a["height"], a["label_sha256"]) != (
            b["width"],
            b["height"],
            b["label_sha256"],
        ):
            reasons.append("pair_contract_mismatch")
        state = (
            "quarantine"
            if any(
                x in reasons
                for x in (
                    "strict_label_geometry",
                    "historical_annotation_conflict",
                    "protected_reference_do_not_reassign",
                    "conditional_protected_dependency",
                    "pair_contract_mismatch",
                )
            )
            else "unresolved"
            if reasons
            else "eligible"
        )
        pair = {
            "pair_id": digest(
                [a["inventory_occurrence_id"], b["inventory_occurrence_id"]]
            ),
            "filename": name,
            "original_name": claim["nombre_original"],
            "video": claim["video"],
            "source_video": claim["archivo_video"],
            "frame_index": int(claim["frame"]),
            "timestamp_seconds": float(claim["segundo"]),
            "temporal_status": "supplier_reported_not_video_verified",
            "sequence_id": "unresolved",
            "sequence_status": "no_current_authorized_membership",
            "historical_split": claim["split_original"],
            "reference_content_id": ref_hash,
            "protected_reference": ref_hash in protected_contents,
            "historical_test_reference": ref_hash in test_contents,
            "status": state,
            "reasons": reasons,
            "label_empty": label["label_empty"],
            "num_objects": label["num_objects"],
            "classes": label["classes_present"],
            "boxes": label["boxes"],
            "width": int(a["width"]),
            "height": int(a["height"]),
            "historical_decisions": hdec,
            "candidate_dependency_groups": sorted(
                {r["visual_dependency_group_id"] for r in candidate_support[ref_hash]}
                | candidate_groups
            ),
            "historical_candidate_support": candidate_support[ref_hash],
            "historical_base_assessment": base[ref_hash],
            "pair_status": "structural_pair_precaution_not_population_visual_acceptance",
            "original": a,
            "no_hud": b,
        }
        pairs.append(pair)
    # Reuse existing temporal ordering, never concatenate independent videos.
    order = pd.DataFrame(
        {
            "sequence_key": [p["source_video"] for p in pairs],
            "frame_index": [p["frame_index"] for p in pairs],
            "content_id": [p["pair_id"] for p in pairs],
        }
    )
    lookup = {p["pair_id"]: p for p in pairs}
    pairs = [lookup[k] for k in ordered_contents(order).content_id]
    reconciliation = {
        "decisions_recovered": len(decisions),
        "historical_decisions_equal": True,
        "current_accepted_relations": 0,
        "historical_sequence_rows": len(membership),
        "historical_sequence_producer_verified": False,
        "sequence_mapping": "not_joined_native_frame_index_is_not_sample_index",
        "october_policy": "D5_dependency_policy_and_D6_test_freeze_pending_in_20261021_package;_20261028_boundaries_unconfirmed",
        "protected_policy": "exclude_matching_reference_contents_and_conditional_supported_groups;_not_an_authorized_test_freeze",
        "base_candidates_promoted": 0,
        "historical_assessment": read_json(bind(assessment / "summary.json")),
        "producer_resolution_explanation": "script_records_intermediate_dimensions_before_compuesto_resize_and_mask_composite;_executed_revision_not_bound",
    }
    reconciliation["producer_evidence_optional"] = True
    # Recheck all newly bound historical sources, including protected records.
    for path, checksum in receipts.items():
        if path in upstream_checks and checksum != upstream_checks[path]:
            raise ValueError(f"Historical input checksum mismatch: {path}")
    if not producer_available:
        reconciliation["producer_resolution_explanation"] = (
            "intermediate_resize_and_composite_possible_but_not_verified;_producer_code_not_required"
        )
    return cfg, pairs, receipts, reconciliation
