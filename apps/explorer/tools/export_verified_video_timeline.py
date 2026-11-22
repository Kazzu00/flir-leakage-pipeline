"""Metadata-only projection: admit exact, uniquely localized canonical frames only.

Consumes a sealed existing timeline and an operator's correspondence evidence.
Never discovers matches, reads pixels, adjudicates sequences or generates splits.
The original inference-based export remains untouched.
"""

import argparse
import copy
import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode()


def project(original, correspondence, videos, delivery, audit):
    data = copy.deepcopy(original)
    runs = [r for r in data["runs"] if r["population"] == "historical"]
    selected = next(r for r in runs if r["run_id"] == "historical-S2-3")
    if (
        selected["membership_id"] != audit["source_membership_id"]
        or selected["candidate_id"] != audit["source_candidate_id"]
    ):
        raise ValueError("Dataset delivery and timeline identities differ")
    if (
        delivery["package_id"] != audit["package_id"]
        or audit["clustering_memberships_nonnull_in_source_groups"] != 0
    ):
        raise ValueError("Unsupported delivery or compatible clustering requires explicit import")
    ids = {m["frame_id"] for m in selected["members"]}
    if any({m["frame_id"] for m in run["members"]} != ids for run in runs):
        raise ValueError("Historical populations differ")
    rows = {r["frame_id"]: r for r in correspondence}
    if len(rows) != len(correspondence) or set(rows) != ids:
        raise ValueError("Correspondence must cover every canonical occurrence exactly once")
    sources = {v["source_video_id"]: v for v in videos}
    if len(sources) != 3 or len(videos) != 3 or any(not v["unchanged"] for v in videos):
        raise ValueError("Exactly three hash-verified authorized videos required")
    data["records"] = [r for r in data["records"] if r["frame_id"] in ids]
    if {r["frame_id"] for r in data["records"]} != ids:
        raise ValueError("Missing canonical records")
    admitted = []
    for r in data["records"]:
        match = rows[r["frame_id"]]
        # The legacy projection exported the native `basename` without suffix.
        # Normalize display metadata only after exact occurrence/content binding.
        if r["content_id"] != match["content_id"] or r["filename"] not in {
            match["filename"],
            Path(match["filename"]).stem,
        }:
            raise ValueError("Occurrence/content identity changed")
        r["filename"] = match["filename"]
        accepted = match["match_status"] == "VIDEO_VERIFIED"
        r.update(
            axis_id=None,
            position=None,
            sequence_id=None,
            order_basis="reported_index",
            reason="ambiguous_association"
            if "AMBIGUOUS" in match["match_status"]
            else "unknown_video",
            temporal_provenance=match["match_status"],
            source_reference=match["evidence_reference"],
        )
        if not accepted:
            if match["source_video_id"] or match["appearance_order"]:
                raise ValueError("Unaccepted candidate must not supply a represented position")
            continue
        v = sources.get(match["source_video_id"])
        if (
            v is None
            or match["source_video_sha256"] != v["sha256"]
            or match["match_method"] != "complete_rgb_raster_equality_via_existing_completed_scan"
            or int(match["existing_exact_occurrence_count"]) != 1
            or not match["canonical_rgb_sha256"]
            or not match["evidence_reference"]
        ):
            raise ValueError("Insufficient exact unique video evidence")
        position = int(match["appearance_order"])
        if position < 0 or position >= v["decoded_frames_in_existing_complete_scan"]:
            raise ValueError("Invalid decoded ordinal")
        r.update(axis_id=v["source_video_id"], position=position, reason=None)
        admitted.append(
            {
                "frame_id": r["frame_id"],
                "content_id": r["content_id"],
                "source_video_id": v["source_video_id"],
                "appearance_order": position,
                "source_video_sha256": v["sha256"],
                "evidence_reference": match["evidence_reference"],
            }
        )
    positions = {}
    for m in admitted:
        key = (m["source_video_id"], m["appearance_order"])
        if key in positions and positions[key] != m["content_id"]:
            raise ValueError("Conflicting video position")
        positions[key] = m["content_id"]
    per_content = {}
    for m in admitted:
        position = (m["source_video_id"], m["appearance_order"])
        if m["content_id"] in per_content and per_content[m["content_id"]] != position:
            raise ValueError("Repeated content has ambiguous appearances")
        per_content[m["content_id"]] = position
    data.update(
        runs=runs,
        axes=[
            {
                "axis_id": v["source_video_id"],
                "source_video_id": v["source_video_id"],
                "label": v["source_video_id"],
                "association": "reported_video",
                "source_reference": "AUTHORIZED_VIDEO_SOURCES.json",
            }
            for v in videos
        ],
        intervals=[],
        clustering_runs=[],
        cluster_memberships=[],
    )
    data["limitations"] = [
        "Only exact unique canonical correspondences to three authorized videos are represented. Filename indices are excluded.",
        "No match in the existing query scope does not demonstrate absence from the videos.",
        "Delivery A is unchanged historical-S2-3, a partial mitigation candidate, not restricted to these videos or leakage-free.",
        "Delivery B remains blocked: canonical video correspondences and verified sequence units are insufficient.",
        "Compatible cluster_id and verified sequence memberships are unavailable; unavailable crossing metrics are null.",
        "Protected test status is the source publication's declaration, not a new approval.",
    ]
    admission = {
        "schema_version": "verified-video-admission-v1",
        "delivery_run_id": selected["run_id"],
        "membership_id": selected["membership_id"],
        "package_id": delivery["package_id"],
        "scope": "exact_unique_three_authorized_videos",
        "delivery_b_status": "BLOCKED_NO_VERIFIED_SEQUENCE_UNITS",
        "authorized_videos": [
            {"source_video_id": v["source_video_id"], "sha256": v["sha256"]} for v in videos
        ],
        "accepted_records": admitted,
        "absence_from_videos_claimed": False,
    }
    return data, admission


def export(source, scope, delivery_root, output):
    source, scope, delivery_root, output = [
        Path(p).resolve() for p in (source, scope, delivery_root, output)
    ]
    if output.exists() or any(output.is_relative_to(p) for p in [source, scope, delivery_root]):
        raise ValueError("New independent output required")
    raw = (source / "index.json").read_bytes()
    receipt_raw = (source / "receipt.json").read_bytes()
    receipt = json.loads(receipt_raw)
    original = json.loads(raw)
    if (
        receipt["files"]["index.json"] != digest(raw)
        or receipt["source_checksums"] != original["source_checksums"]
    ):
        raise ValueError("Source timeline checksum/provenance mismatch")
    receipt_delivery = json.loads((delivery_root / "DELIVERY_RECEIPT.json").read_bytes())
    audit_path = Path(receipt_delivery["package_path"]) / "split_audit.json"
    inputs = {
        "video_correspondence.csv": (scope / "video_correspondence.csv").read_bytes(),
        "AUTHORIZED_VIDEO_SOURCES.json": (scope / "AUTHORIZED_VIDEO_SOURCES.json").read_bytes(),
        "CANONICAL_RGB_PROBE.json": (scope / "CANONICAL_RGB_PROBE.json").read_bytes(),
        "DELIVERY_RECEIPT.json": (delivery_root / "DELIVERY_RECEIPT.json").read_bytes(),
        "split_audit.json": audit_path.read_bytes(),
    }
    correspondence = list(
        csv.DictReader(io.StringIO(inputs["video_correspondence.csv"].decode("utf-8-sig")))
    )
    data, admission = project(
        original,
        correspondence,
        json.loads(inputs["AUTHORIZED_VIDEO_SOURCES.json"]),
        receipt_delivery,
        json.loads(inputs["split_audit.json"]),
    )
    probe = json.loads(inputs["CANONICAL_RGB_PROBE.json"])
    expected_counts = Counter(m["source_video_id"] for m in admission["accepted_records"])
    if len(correspondence) != probe["canonical_occurrences"] or any(
        expected_counts[k] != n for k, n in probe["accepted_occurrences_by_video"].items()
    ):
        raise ValueError("Scope proof count mismatch")
    checks = {
        **data["source_checksums"],
        "previous_timeline/index.json": digest(raw),
        "previous_timeline/receipt.json": digest(receipt_raw),
        **{"video_scope/" + k: digest(v) for k, v in inputs.items()},
    }
    data["source_checksums"] = checks
    output.mkdir(parents=True, exist_ok=False)
    (output / "index.json").write_bytes(encoded(data))
    (output / "admission.json").write_bytes(encoded(admission))
    by_id = {r["frame_id"]: r for r in data["records"]}
    selected = next(r for r in data["runs"] if r["run_id"] == admission["delivery_run_id"])
    excluded = [
        {**by_id[m["frame_id"]], "new_split": m["new_split"]}
        for m in selected["members"]
        if by_id[m["frame_id"]]["reason"] is not None
    ]
    fields = [
        "frame_id",
        "content_id",
        "filename",
        "new_split",
        "reason",
        "temporal_provenance",
        "source_reference",
    ]
    with (output / "excluded_frames.csv").open("x", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(excluded)
    coverage = {
        "source_run": admission["delivery_run_id"],
        "total": len(selected["members"]),
        "accepted": len(admission["accepted_records"]),
        "excluded": len(excluded),
        "accepted_by_video": {
            v["source_video_id"]: expected_counts[v["source_video_id"]]
            for v in admission["authorized_videos"]
        },
        "absence_claimed": False,
        "verified_sequence_crossings": None,
        "verified_temporal_proximity": None,
    }
    (output / "coverage.json").write_bytes(encoded(coverage))
    files = {p.name: digest(p.read_bytes()) for p in output.iterdir()}
    (output / "receipt.json").write_bytes(
        encoded(
            {
                "schema_version": "scientific-timeline-export-receipt-v1",
                "files": files,
                "source_checksums": checks,
            }
        )
    )
    return coverage


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["source", "scope", "delivery", "output"]:
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    print(json.dumps(export(args.source, args.scope, args.delivery, args.output)))
