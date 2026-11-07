"""Explicit FAC case-study adapter. No FAC naming rule enters the general engine.

Consume the audited inventory and video-grounded sidecar, preserve every inventory
occurrence, and keep non-primary roles out of the provisional policy population.
Structural original/processed pairs are candidates, not accepted dependencies.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml

from flir_pipeline.data.local_images import declared_file
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
from flir_pipeline.general.publication import read_json
from flir_pipeline.utils.hashing import sha256_file


class FACConfig(Contract):
    inventory: ID
    delivery: ID
    review: ID
    data_prefix: ID
    video_prefix: ID
    classes: list[ClassEntry]


def adapt(config_path: Path, root: Path) -> tuple[Manifest, dict]:
    cfg = FACConfig.model_validate(
        yaml.safe_load(config_path.read_text(encoding="utf-8"))
    )
    sources = {}

    def bind(relative, method="inherited_audit", checksum=None):
        actual = sha256_file(declared_file(root, relative))
        if checksum is not None and actual != checksum:
            raise ValueError(f"Upstream receipt mismatch: {relative}")
        evidence = Evidence(
            path=relative,
            sha256=actual,
            method=method,
            version="1",
            provenance="FAC_existing_publication",
        )
        sources[relative] = evidence
        return evidence

    inv_receipt_path = f"{cfg.inventory}/artifact_checksums.json"
    bind(inv_receipt_path)
    inv_receipt = read_json(root / inv_receipt_path)
    for name in ("canonical_manifest.csv", "inventory_report.json"):
        bind(f"{cfg.inventory}/{name}", checksum=inv_receipt[name])
    review_receipt_path = f"{cfg.review}/package_checksums.json"
    bind(review_receipt_path)
    review_receipt = read_json(root / review_receipt_path)["artifacts_sha256"]
    side_path = f"{cfg.review}/temporal_status_sidecar.json"
    side_evidence = bind(
        side_path,
        "inherited_complete_decoded_raster_scan",
        review_receipt["temporal_status_sidecar.json"],
    )
    for name in (
        "video_localization/sources.json",
        "video_localization/summary.json",
        "historical_candidate_edges.csv",
    ):
        bind(f"{cfg.review}/{name}", checksum=review_receipt[name])
    source_rows = read_json(root / f"{cfg.review}/video_localization/sources.json")
    videos = [
        Video(
            video_id=s["source_video"],
            path=f"{cfg.video_prefix}/{s['source_video']}",
            sha256=s["sha256"],
        )
        for s in source_rows
    ]
    if any(not s["complete"] for s in source_rows):
        raise ValueError("Incomplete inherited decoded scan")
    # These parent bytes authenticate all original members; generic verification
    # also rehashes declared members and reparses annotation bytes on every replay.
    for source in read_json(root / f"{cfg.inventory}/inventory_report.json")["sources"]:
        if source["source"].endswith(".tar"):
            relative = f"{cfg.data_prefix}/{source['source']}"
            sources[relative] = Evidence(
                path=relative,
                sha256=source["sha256"],
                method="audited_tar",
                version="1",
                provenance=inv_receipt_path,
            )
    side = read_json(root / side_path)
    lookup = {r["filename"]: r for r in side}
    if len(lookup) != len(side):
        raise ValueError("Ambiguous sidecar filename mapping")
    with (root / f"{cfg.inventory}/canonical_manifest.csv").open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        inventory = list(csv.DictReader(stream))
    records = []
    for r in inventory:
        state = lookup.get(r["filename"]) if r["role"] == "primary" else None
        primary_original = (
            state is not None and r["content_id"] == state["original_content_id"]
        )
        if state is not None and r["content_id"] not in (
            state["original_content_id"],
            state["no_hud_content_id"],
        ):
            raise ValueError("Inventory and reviewed pair identities disagree")
        qa = json.loads(r["label_validation"]) if r["label_validation"] else None
        labels = json.loads(r["label_members"])
        kwargs = {}
        if r["label_status"] != "missing":
            if len(labels) != 1 or qa is None:
                raise ValueError("Annotation location unresolved")
            kwargs = {
                "annotation_status": "invalid"
                if not qa["label_valid"]
                else "empty"
                if qa["label_empty"]
                else "valid",
                "label_sha256": r["label_sha256"],
                "label_source": f"{cfg.data_prefix}/{r['source']}::{labels[0]}",
                "boxes": qa["boxes"],
                "annotation_errors": qa["errors"] + qa["warnings"],
            }
        positions = []
        if primary_original:
            for p in state["exact_occurrences"]:
                video = next(v for v in videos if v.video_id == p["source_video"])
                if video.sha256 != p["video_sha256"]:
                    raise ValueError("Position is bound to different video bytes")
                positions.append(
                    Position(
                        video_id=p["source_video"],
                        status="confirmed",
                        kind="decoded_pts",
                        seconds=p["second"],
                        pts=p["pts"],
                        time_base=p["time_base"],
                        decoded_ordinal=p["decoded_ordinal_zero_based"],
                        decoded_rgb_sha256=p["decoded_rgb_sha256"],
                        evidence=side_evidence,
                    )
                )
        claims = (
            json.loads(r["supplier_correspondence"])
            if r["supplier_correspondence"]
            else []
        )
        original_split = claims[0].get("split_original") if len(claims) == 1 else None
        reasons = (
            list(state["reasons"])
            if state
            else [f"outside_primary_population:{r['role']}"]
        )
        records.append(
            Occurrence(
                frame_id=r["inventory_occurrence_id"],
                content_id=r["content_id"],
                image_sha256=r["sha256"],
                source=f"{cfg.data_prefix}/{r['source']}::{r['member']}",
                original_name=r["filename"],
                original_split=original_split,
                positions=positions,
                exclusion_reasons=reasons,
                metadata={
                    "role": r["role"],
                    "supplier_claims_auxiliary_only": claims,
                    "historical_candidate_groups": state["groups"] if state else [],
                    "pair_id": state["pair_id"] if state else None,
                },
                **kwargs,
            )
        )
    relations = [
        Relation(
            relation_id=f"structural-variant:{r['pair_id']}",
            left=r["original_content_id"],
            right=r["no_hud_content_id"],
            kind="visual",
            status="candidate",
            accepted=False,
            evidence=side_evidence,
        )
        for r in side
    ]
    contents = {r.content_id for r in records}
    edge_evidence = sources[f"{cfg.review}/historical_candidate_edges.csv"]
    external_edges = 0
    with (root / edge_evidence.path).open(encoding="utf-8-sig", newline="") as stream:
        raw_edges = list(csv.DictReader(stream))
    for i, r in enumerate(raw_edges):
        if r["labeled_content_id"] in contents and r["video_content_id"] in contents:
            relations.append(
                Relation(
                    relation_id=f"historical-candidate-row:{i}",
                    left=r["labeled_content_id"],
                    right=r["video_content_id"],
                    kind="visual",
                    status="candidate",
                    accepted=False,
                    evidence=edge_evidence,
                )
            )
        else:
            external_edges += 1
    manifest = Manifest(
        classes=cfg.classes,
        records=records,
        videos=videos,
        relations=relations,
        sources=list(sources.values()),
    )
    report = {
        "inventory_occurrences": len(inventory),
        "canonical_occurrences": len(records),
        "primary_occurrences": sum(r.metadata["role"] == "primary" for r in records),
        "historical_or_diagnostic_occurrences_preserved_excluded": sum(
            r.metadata["role"] != "primary" for r in records
        ),
        "structural_pair_candidates": len(side),
        "accepted_dependencies": 0,
        "historical_candidate_edges": len(raw_edges),
        "edges_outside_manifest_not_evaluable": external_edges,
        "decoded_scan_reused_frames": sum(r["decoded_frames"] for r in source_rows),
        "decoded_scan_executed_in_this_closure": False,
        "processed_variant_inherits_original_time": False,
        "training_allowed": False,
    }
    return manifest, report
