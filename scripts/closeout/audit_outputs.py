"""Independent replay: standard-library/Pillow checks, no production importer use.

Audit raw TAR entries against the published inventory, recompute every label/CSV
count, and independently check any published partition's graph constraints.
"""

import argparse
import csv
import hashlib
import io
import json
import tarfile
from collections import Counter
from pathlib import Path

from PIL import Image


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def audit(tar_path, dataset, split=None):
    inventory = csv_rows(dataset / "members_inventory.csv")
    by_order = {int(row["tar_order"]): row for row in inventory}
    counts, decoded, seen = Counter(), 0, set()
    with tarfile.open(tar_path) as archive:
        for order, member in enumerate(archive):
            if member.isdir():
                continue
            require(
                order in by_order and member.isfile(),
                "TAR member missing from inventory",
            )
            raw = archive.extractfile(member).read()
            record = by_order[order]
            require(
                record["member"] == member.name and sha(raw) == record["sha256"],
                "Raw TAR identity mismatch",
            )
            require(
                (dataset / record["output"]).read_bytes() == raw,
                "Published member bytes differ",
            )
            seen.add(order)
            if "/images/" in member.name:
                with Image.open(io.BytesIO(raw)) as im:
                    im.load()
                decoded += 1
            if "/labels/" in member.name:
                for line in raw.decode("utf-8-sig").splitlines():
                    if line.strip():
                        counts[int(line.split()[0])] += 1
    require(seen == set(by_order), "Extra output inventory rows")
    correspondence = csv_rows(dataset / "correspondencia.csv")
    manifest = csv_rows(dataset / "manifest.csv")
    require(
        decoded == len(correspondence) == len(manifest), "Occurrence count mismatch"
    )
    names = ["vehicle", "building", "road", "river", "maquinaria"]
    require(
        [counts[i] for i in range(5)]
        == [sum(int(r[n]) for r in correspondence) for n in names],
        "Class support differs from CSV",
    )
    require(
        len({r["frame_id"] for r in manifest}) == len(manifest),
        "Frame identity collision",
    )
    receipt = json.loads((dataset / "receipt.json").read_text(encoding="utf-8"))
    for name, expected in receipt["files"].items():
        require(sha((dataset / name).read_bytes()) == expected, "Sealed output changed")
    result = {
        "state": "INDEPENDENT_REPLAY_PASSED",
        "regular_members": len(seen),
        "images_decoded": decoded,
        "correspondence_rows": len(correspondence),
        "class_support": dict(counts),
        "production_importer_called": False,
    }
    if split:
        membership = csv_rows(split / "memberships.csv")
        status = json.loads((split / "audit.json").read_text(encoding="utf-8"))["state"]
        if status == "BLOCKED":
            require(
                all(r["new_split"] == "UNASSIGNED" for r in membership),
                "Blocked split contains assignments",
            )
            require(
                not (split / "dataset.yaml").exists(),
                "Blocked split exposes training YAML",
            )
            result["split"] = "BLOCKED correctly; no materialized split"
        else:
            require(
                {r["frame_id"] for r in membership}
                == {r["frame_id"] for r in manifest},
                "Split population changed",
            )
            for key in ("group_id", "content_id", "source_frame_key"):
                assignments = {}
                for row in membership:
                    require(
                        assignments.setdefault(row[key], row["new_split"])
                        == row["new_split"],
                        f"Cross-split {key}",
                    )
            assignments = {r["frame_id"]: r["new_split"] for r in membership}
            for edge in csv_rows(split / "relations.csv"):
                require(
                    assignments[edge["left"]] == assignments[edge["right"]],
                    "Cross-split registered relation",
                )
            result["split"] = (
                "All registered relations and groups indivisible; independence unproven"
            )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("tar", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--split", type=Path)
    args = parser.parse_args()
    result = audit(args.tar, args.dataset, args.split)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))
