# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas==3.0.5", "pyarrow==25.0.1"]
# ///
"""Read-only presentation export of an existing fixed-test publication.

No backend imports, solver, image/label decoding, metrics calculation or replay.
Only consumed receipt-bound tables are checked; this is not scientific admission.
"""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def digest(data):
    return hashlib.sha256(data).hexdigest()


def export(publication, output):
    publication, output = Path(publication).resolve(), Path(output).resolve()
    if output.exists() or output.is_relative_to(publication):
        raise ValueError("Output must be new and outside the source publication")
    receipt_bytes = (publication / "receipt.json").read_bytes()
    receipt = json.loads(receipt_bytes)
    if receipt["schema_version"] != "fixed_test_train_val_v1":
        raise ValueError("Unsupported publication")
    consumed = {}

    def read(name, parquet=False):
        path = (publication / name).resolve()
        if not path.is_relative_to(publication) or name not in receipt["artifacts"]:
            raise ValueError("Unbound source")
        raw = path.read_bytes()
        sha = digest(raw)
        if sha != receipt["artifacts"][name]:
            raise ValueError(f"Checksum mismatch: {name}")
        consumed[name] = sha
        if parquet:
            from io import BytesIO

            return pd.read_parquet(BytesIO(raw))
        return json.loads(raw)

    summary, suite = read("summary.json"), read("suite.json")
    provenance = read("provenance.json")
    if summary["ready_for_training"] is not False or summary["state"] != "PROVISIONAL_CANDIDATES":
        raise ValueError("Only provisional fixed-test evidence is supported")
    labels = read("sources/audit/annotation_audit.json")
    relations = read("sources/audit/relations.json")
    classes = {}
    for row in labels["records"]:
        for frame_id in row["frame_ids"]:
            value = sorted(int(key) for key in row["class_counts"])
            if frame_id in classes and classes[frame_id] != value:
                raise ValueError("Ambiguous occurrence class metadata")
            classes[frame_id] = value
    records, runs, common_test = {}, [], None
    for run in summary["runs"]:
        arm = run["arm"]
        prefix = f"runs/{arm}/"
        metadata = read(prefix + "metadata.json")
        metrics = read(prefix + "metrics.json")
        rows = read(prefix + "occurrence_memberships.parquet", True)
        groups = read(prefix + "atomic_groups.parquet", True)
        identity = metadata["identity"]["arm"]
        if (
            identity["arm"] != arm
            or metadata["candidate_id"] != run["candidate_id"]
            or metrics["ready_for_training"] is not False
            or metrics["state"] != "PROVISIONAL_VERIFIED"
            or metadata["membership_id"] != run["membership_id"]
        ):
            raise ValueError("Inconsistent provisional identity")
        if rows.frame_id.duplicated().any() or groups.content_id.duplicated().any():
            raise ValueError("Duplicate member identity")
        group_map = dict(zip(groups.content_id, groups.group_id, strict=True))
        members, test_rows = [], []
        for row in rows.to_dict("records"):
            if row["new_split"] not in ("train", "val", "test", "quarantine"):
                raise ValueError("Unknown partition")
            record = {
                "record_id": row["frame_id"],
                "content_id": row["content_id"],
                "filename": row["basename"],
                "source": row["source_archive"],
                "cohort": row["cohort"],
                "class_ids": classes[row["frame_id"]],
                "label_sha256": row["label_sha256"],
            }
            previous = records.setdefault(row["frame_id"], record)
            if previous != record:
                raise ValueError("Occurrence metadata differs between arms")
            group = group_map.get(row["content_id"])
            if row["new_split"] in ("train", "val") and group is None:
                raise ValueError("Missing development allocation group")
            members.append(
                {"record_id": row["frame_id"], "partition": row["new_split"], "group_id": group}
            )
            if row["new_split"] == "test":
                test_rows.append((row["frame_id"], row["content_id"], row["label_sha256"]))
        test_rows.sort()
        if common_test is not None and test_rows != common_test:
            raise ValueError("Common test identity differs")
        common_test = test_rows
        runs.append(
            {
                "id": metadata["candidate_id"],
                "membership_id": metadata["membership_id"],
                "policy": identity["policy"],
                "population": identity["population"],
                "seed": identity["split_seed"],
                "test_id": identity["test_id"],
                "state": metrics["state"],
                "ready_for_training": False,
                "metrics": metrics,
                "members": members,
            }
        )
    if len(runs) != summary["configurations"] or len(runs) != summary["verified_candidates"]:
        raise ValueError("Incomplete candidate suite")
    if len({r["id"] for r in runs}) != len(runs):
        raise ValueError("Duplicate candidate identity")
    edges = []
    for edge in relations["records"]:
        if edge["left_frame_id"] not in records or edge["right_frame_id"] not in records:
            raise ValueError("Unknown relation endpoint")
        edges.append(
            {key: edge[key] for key in ("relation_id", "left_frame_id", "right_frame_id", "state")}
        )
    result = {
        "schema_version": "candidate-splits-view-v1",
        "test_status": "PROPOSED",
        "ready_for_training": False,
        "dataset_id": labels["dataset_id"],
        "source_receipt_sha256": digest(receipt_bytes),
        "consumed_sha256": consumed,
        "source_commit": provenance["git_commit"],
        "source_dirty": provenance["git_worktree_dirty"],
        "plan_id": suite["plan_id"],
        "blockers": summary["blockers"],
        "verification_scope": "consumed-file hashes and presentation identities; no scientific replay",
        "records": list(records.values()),
        "runs": runs,
        "relations": edges,
    }
    for name, sha in consumed.items():
        if digest((publication / name).read_bytes()) != sha:
            raise ValueError("Source changed during export")
    if (publication / "receipt.json").read_bytes() != receipt_bytes:
        raise ValueError("Receipt changed during export")
    raw = (
        json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"
    ).encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(raw)
    print(f"Presentation only: {len(runs)} runs, {len(records)} records; SHA256 {digest(raw)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("publication", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    export(args.publication, args.output)
