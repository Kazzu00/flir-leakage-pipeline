# /// script
# requires-python = ">=3.11"
# dependencies = ["pyarrow==25.0.1"]
# ///
"""Read-only projection of receipt-bound occurrences and organization-evidence-v2.

No backend imports, images, annotations, metrics, solver or temporal discovery.
Filename families remain explicitly inferred axes, never verified source videos.
"""

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return (
        json.dumps(
            value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
        )
        + "\n"
    ).encode("utf-8")


def unique(rows, key):
    result = {}
    for row in rows:
        value = row[key]
        if value in result:
            raise ValueError(f"Duplicate {key}")
        result[value] = row
    return result


def index_value(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value) if math.isfinite(value) and value >= 0 and int(value) == value else None


def build_projection(organization, native_runs, *, expected_historical=1657):
    """Join occurrences by frame_id, clustering by (run, content), never by group.

    A collision of different contents at an inferred index is excluded explicitly.
    Repeated occurrences of the SAME content at the same index are retained.
    """
    org_records = unique(organization["records"], "record_id")
    contents = unique(organization["contents"], "content_id")
    timelines = unique(organization["timelines"], "timeline_id")
    configurations = unique(organization["clustering_configurations"], "cluster_run_id")
    media = unique(organization["media"], "preview_key")
    cluster_summaries = {
        (c["cluster_run_id"], c["cluster_id"]): c for c in organization["clusters"]
    }
    if len(cluster_summaries) != len(organization["clusters"]):
        raise ValueError("Duplicate cluster summary")
    memberships = {}
    for m in organization["cluster_memberships"]:
        key = (m["cluster_run_id"], m["content_id"])
        config = configurations.get(m["cluster_run_id"])
        if (
            key in memberships
            or m["content_id"] not in contents
            or not config
            or not config["membership_consistency_verified"]
            or (m["cluster_run_id"], m["cluster_id"]) not in cluster_summaries
            or m["is_noise"] != (m["cluster_id"] == -1)
        ):
            raise ValueError("Unverified or inconsistent content-to-cluster membership")
        if (
            config["source_kind"] == "frozen_split_membership"
            and not config["source_membership_checksums"]
        ):
            raise ValueError("Frozen clustering membership has no bound source")
        memberships[key] = {k: m[k] for k in ("cluster_run_id", "content_id", "cluster_id")}

    axes = {}
    for t in timelines.values():
        if not t["source_video_id"] and not t["inferred_family"]:
            continue
        axes[t["timeline_id"]] = {
            "axis_id": t["timeline_id"],
            "source_video_id": t["source_video_id"],
            "label": t["source_video_id"] or t["inferred_family"],
            "association": "reported_video" if t["source_video_id"] else "filename_inference",
            "source_reference": "organization/timelines.json",
        }

    records = {}
    runs = []
    common_test = None
    run_ids = set()
    selectors = set()
    for native in native_runs:
        arm, rows, meta = native["arm"], native["rows"], native["metadata"]
        identity = meta["identity"]["arm"]
        selector = (identity["population"], identity["policy"], identity["split_seed"])
        if (
            arm in run_ids
            or selector in selectors
            or identity["arm"] != arm
            or identity["policy"] not in {"S1", "S2", "S3"}
            or identity["population"] not in {"historical", "expanded"}
            or identity["split_seed"] not in range(5)
        ):
            raise ValueError("Mixed or duplicate run identity")
        run_ids.add(arm)
        selectors.add(selector)
        unique(rows, "frame_id")
        if not rows:
            raise ValueError("Empty experimental population")
        if identity["population"] == "historical" and len(rows) != expected_historical:
            raise ValueError("Historical occurrence inventory does not match expected count")
        labeled_ids = {r["record_id"] for r in org_records.values() if r.get("cohort") == "labeled"}
        if identity["population"] == "historical" and {r["frame_id"] for r in rows} != labeled_ids:
            raise ValueError("Historical occurrence identities differ from organization inventory")
        members = []
        test = []
        for row in rows:
            frame, content = row["frame_id"], row["content_id"]
            partition = row["new_split"]
            if partition not in {"train", "val", "test", "quarantine", None}:
                raise ValueError("Unknown experimental partition")
            org = org_records.get(frame)
            if org and org["content_id"] != content:
                raise ValueError("Occurrence/content binding mismatch")
            source_ref = f"publication/runs/{arm}/occurrence_memberships.parquet"
            temporal = (
                row.get("temporal_provenance") or (org or {}).get("temporal_source") or "unknown"
            )
            filename = row.get("basename") or (org or {}).get("filename") or frame
            position = index_value(row.get("possible_frame_index"))
            family = row.get("possible_sequence")
            timeline = timelines.get((org or {}).get("timeline_id"))
            reason = None
            if not timeline:
                candidates = [
                    t
                    for t in timelines.values()
                    if family
                    and t["inferred_family"] == family
                    and t["source_archive"] == row.get("source_archive")
                ]
                if len(candidates) > 1:
                    reason = "ambiguous_association"
                elif candidates:
                    timeline = candidates[0]
            axis = axes.get(timeline["timeline_id"]) if timeline else None
            if not axis:
                reason = reason or "unknown_video"
            elif family and timeline["inferred_family"] and family != timeline["inferred_family"]:
                reason = "ambiguous_association"
            if position is None and org:
                position = index_value(org.get("frame_index"))
            if not reason and position is None:
                reason = "unknown_order"
            org_index = index_value((org or {}).get("frame_index"))
            if not reason and org_index is not None and org_index != position:
                reason = "index_conflict"
            if not reason and temporal == "unknown":
                reason = "unknown_order"
            preview = media.get(contents.get(content, {}).get("preview_key"))
            if preview and preview["content_id"] != content:
                raise ValueError("Preview/content binding mismatch")
            record = {
                "frame_id": frame,
                "content_id": content,
                "filename": filename,
                "axis_id": axis["axis_id"] if axis else None,
                "position": position,
                "temporal_provenance": temporal,
                "order_basis": "filename_inference"
                if "filename" in temporal
                else "sampled_grid"
                if "sample" in temporal
                else "reported_index",
                "source_reference": source_ref,
                "sequence_id": None,
                "reason": reason,
                "preview_path": preview["relative_path"]
                if preview and preview["media_available"]
                else None,
            }
            # Across seeds/policies every occurrence must retain the SAME temporal identity.
            if frame in records:
                before = {k: v for k, v in records[frame].items() if k != "source_reference"}
                after = {k: v for k, v in record.items() if k != "source_reference"}
                if before != after:
                    raise ValueError("Occurrence metadata changes between experimental runs")
            else:
                records[frame] = record
            members.append({"frame_id": frame, "new_split": partition or "unassigned"})
            if partition == "test":
                test.append((frame, content))
        test = sorted(test)
        if common_test is not None and test != common_test:
            raise ValueError("Common protected test differs between runs")
        common_test = test
        runs.append(
            {
                "run_id": arm,
                "population": identity["population"],
                "policy": identity["policy"],
                "seed": identity["split_seed"],
                "candidate_id": meta["candidate_id"],
                "membership_id": meta["membership_id"],
                "split_space_id": meta.get("split_space_id"),
                "test_id": identity["test_id"],
                "state": native.get("state", "PROVISIONAL_VERIFIED"),
                "source_reference": source_ref,
                "members": sorted(members, key=lambda m: m["frame_id"]),
            }
        )

    positions = defaultdict(list)
    for record in records.values():
        if record["reason"] is None:
            positions[(record["axis_id"], record["position"])].append(record)
    for rows in positions.values():
        if len({r["content_id"] for r in rows}) > 1:
            for record in rows:
                record["reason"] = "index_conflict"

    used_axes = {r["axis_id"] for r in records.values() if r["reason"] is None}
    intervals = []
    for axis_id in sorted(used_axes):
        rows = [r for r in records.values() if r["axis_id"] == axis_id and r["reason"] is None]
        # Existing filename-family grouping only. Extents summarize observed indices;
        # they do not discover a boundary or turn color transitions into sequences.
        intervals.append(
            {
                "interval_id": "candidate:" + axis_id,
                "axis_id": axis_id,
                "sequence_id": None,
                "kind": "candidate_span",
                "review": "not_reviewed",
                "start": min(r["position"] for r in rows),
                "end": max(r["position"] for r in rows),
                "source_reference": axes[axis_id]["source_reference"],
                "provenance": "Observed extent of an existing temporal namespace; not a confirmed sequence.",
            }
        )
    return {
        "records": sorted(records.values(), key=lambda r: r["frame_id"]),
        "axes": [axes[k] for k in sorted(used_axes)],
        "runs": sorted(runs, key=lambda r: r["run_id"]),
        "intervals": intervals,
        "cluster_memberships": sorted(
            memberships.values(), key=lambda m: (m["cluster_run_id"], m["content_id"])
        ),
        "clustering_runs": [
            {
                k: c[k]
                for k in (
                    "cluster_run_id",
                    "source_kind",
                    "full_clustering_artifact_available",
                    "membership_consistency_verified",
                    "source_membership_checksums",
                )
            }
            for c in configurations.values()
        ],
    }


def coverage(data, run):
    records = {r["frame_id"]: r for r in data["records"]}
    rows = [records[m["frame_id"]] for m in run["members"]]
    visible = [r for r in rows if r["reason"] is None]
    excluded = [r for r in rows if r["reason"] is not None]
    return {
        "run_id": run["run_id"],
        "population": run["population"],
        "policy": run["policy"],
        "seed": run["seed"],
        "total": len(rows),
        "displayed": len(visible),
        "excluded": len(excluded),
        "exclusion_reasons": dict(Counter(r["reason"] for r in excluded)),
        "by_axis": dict(Counter(r["axis_id"] for r in visible)),
        "order_basis": dict(Counter(r["order_basis"] for r in visible)),
        "partitions": dict(Counter(m["new_split"] for m in run["members"])),
    }


def export(publication, organization, output, *, expected_historical=1657):
    publication, organization, output = (
        Path(p).resolve() for p in (publication, organization, output)
    )
    if output.exists() or any(output.is_relative_to(p) for p in (publication, organization)):
        raise ValueError("Output must be new and outside both read-only sources")
    receipt_raw = (publication / "receipt.json").read_bytes()
    receipt = json.loads(receipt_raw)
    manifest_raw = (organization / "manifest.json").read_bytes()
    manifest = json.loads(manifest_raw)
    if (
        receipt["schema_version"] != "fixed_test_train_val_v1"
        or manifest["schema_version"] != "organization-evidence-v2"
    ):
        raise ValueError("Unsupported source contract")
    consumed = {}
    source_paths = {}

    def read(root, name, expected, prefix, parquet=False):
        p = (root / name).resolve()
        if not p.is_relative_to(root) or name not in expected:
            raise ValueError("Unbound source path")
        raw = p.read_bytes()
        if digest(raw) != expected[name]:
            raise ValueError("Source checksum mismatch: " + prefix + name)
        consumed[prefix + name] = digest(raw)
        source_paths[prefix + name] = p
        if parquet:
            import pyarrow.parquet as pq

            source = pq.ParquetFile(BytesIO(raw))
            required = {"frame_id", "content_id", "new_split"}
            if not required <= set(source.schema_arrow.names):
                raise ValueError("Missing occurrence identity or partition")
            allowed = required | {
                "basename",
                "possible_sequence",
                "possible_frame_index",
                "source_archive",
                "temporal_provenance",
            }
            return source.read(columns=sorted(allowed & set(source.schema_arrow.names))).to_pylist()
        return json.loads(raw)

    org = {
        name: read(organization, name + ".json", manifest["file_sha256"], "organization/")
        for name in [
            "records",
            "contents",
            "timelines",
            "cluster_memberships",
            "split_memberships",
            "clusters",
            "boundary_zones",
            "clustering_configurations",
            "media",
        ]
    }
    if len(org["records"]) != manifest["record_count"]:
        raise ValueError("Organization occurrence inventory mismatch")
    summary = read(publication, "summary.json", receipt["artifacts"], "publication/")
    provenance = read(publication, "provenance.json", receipt["artifacts"], "publication/")
    if summary["state"] != "PROVISIONAL_CANDIDATES" or summary["ready_for_training"] is not False:
        raise ValueError("This adapter supports only the provisional fixed-test publication")
    natives = []
    for run in summary["runs"]:
        prefix = "runs/" + run["arm"] + "/"
        meta = read(publication, prefix + "metadata.json", receipt["artifacts"], "publication/")
        if (
            meta["candidate_id"] != run["candidate_id"]
            or meta["membership_id"] != run["membership_id"]
        ):
            raise ValueError("Published run identity mismatch")
        identity = meta["identity"]["arm"]
        if (identity["population"], identity["policy"], identity["split_seed"]) != (
            run["population"],
            run["policy"],
            run["seed"],
        ):
            raise ValueError("Published selector mismatch")
        rows = read(
            publication,
            prefix + "occurrence_memberships.parquet",
            receipt["artifacts"],
            "publication/",
            True,
        )
        natives.append({"arm": run["arm"], "rows": rows, "metadata": meta, "state": run["state"]})
    data = build_projection(org, natives, expected_historical=expected_historical)
    data = {
        "schema_version": "scientific-timeline-v1",
        "origin": "existing_evidence_export",
        "source_receipt_sha256": digest(receipt_raw),
        "organization_manifest_sha256": digest(manifest_raw),
        "source_commit": provenance["git_commit"],
        "source_dirty": provenance["git_worktree_dirty"],
        "test_status": "PROPOSED",
        "ready_for_training": False,
        "source_checksums": consumed,
        "limitations": [
            "Local checksum verification; remote Hypatia publication was not revalidated.",
            "Filename families are inferred axes, not verified identities of original videos or timestamps.",
            "Candidate extents are not confirmed sequences. Boundary zones are not converted to sequence instances.",
            "Clustering is an independently selected reference; it does not determine S1/S2/S3 partitions.",
            "Missing original clustering diagnostics are not reconstructed from frozen memberships.",
            "Native membership_id is preserved; missing split_space_id stays null, never renamed.",
            "No protected annotations, detector predictions, selection metrics or source images are exported.",
        ],
        **data,
    }
    summaries = [coverage(data, r) for r in data["runs"]]
    if any(r["total"] != r["displayed"] + r["excluded"] for r in summaries):
        raise ValueError("Occurrence accounting failed")
    if (publication / "receipt.json").read_bytes() != receipt_raw or (
        organization / "manifest.json"
    ).read_bytes() != manifest_raw:
        raise ValueError("Source envelope changed during export")
    if any(digest(p.read_bytes()) != consumed[key] for key, p in source_paths.items()):
        raise ValueError("Consumed source changed during export")
    output.mkdir(parents=True, exist_ok=False)
    (output / "index.json").write_bytes(encoded(data))
    (output / "coverage.json").write_bytes(encoded({"runs": summaries}))
    records = {r["frame_id"]: r for r in data["records"]}
    fields = [
        "run_id",
        "population",
        "policy",
        "seed",
        "frame_id",
        "content_id",
        "filename",
        "reason",
        "source_reference",
        "temporal_provenance",
        "axis_id",
        "position",
        "new_split",
    ]
    for name, excluded in [("timeline_unassigned.csv", True), ("timeline_assigned.csv", False)]:
        with (output / name).open("x", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for run in data["runs"]:
                for member in run["members"]:
                    row = records[member["frame_id"]]
                    if (row["reason"] is not None) != excluded:
                        continue
                    values = {
                        **{k: run[k] for k in ["run_id", "population", "policy", "seed"]},
                        **{k: row[k] for k in fields if k in row},
                        "new_split": member["new_split"],
                        "source_reference": run["source_reference"],
                    }
                    writer.writerow(
                        {
                            k: (
                                "'" + v
                                if isinstance(v, str) and v.startswith(("=", "+", "-", "@"))
                                else v
                            )
                            for k, v in values.items()
                        }
                    )
    checks = {p.name: digest(p.read_bytes()) for p in output.iterdir() if p.is_file()}
    (output / "receipt.json").write_bytes(
        encoded(
            {
                "schema_version": "scientific-timeline-export-receipt-v1",
                "files": checks,
                "source_checksums": consumed,
            }
        )
    )
    print(
        json.dumps(
            {
                "runs": len(data["runs"]),
                "unique_occurrences": len(data["records"]),
                "coverage": summaries,
                "output_files": list(checks),
            }
        )
    )
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("publication", type=Path)
    parser.add_argument("organization", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expected-historical", type=int, default=1657)
    args = parser.parse_args()
    export(
        args.publication,
        args.organization,
        args.output,
        expected_historical=args.expected_historical,
    )
