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
            require(len(assignments) == len(membership), "Duplicate split occurrence")
            require(
                set(assignments.values()) <= {"train", "val", "test", "quarantine"},
                "Invalid split destination",
            )
            for row in membership:
                for kind, column, expected in (
                    ("images", "image_path", row["content_id"]),
                    ("labels", "label_path", row["label_sha256"]),
                ):
                    copied = split / kind / row["new_split"] / Path(row[column]).name
                    require(sha(copied.read_bytes()) == expected, "Split copy changed")
            for edge in csv_rows(split / "relations.csv"):
                require(
                    assignments[edge["left"]] == assignments[edge["right"]],
                    "Cross-split registered relation",
                )
            result["split"] = (
                "All registered relations and groups indivisible; independence unproven"
            )
    return result


def audit_graph(dataset, directory, stores_file):
    """Reconstruct candidate edges independently, including failed MILP graphs."""
    import re
    from collections import defaultdict

    import numpy as np
    import pandas as pd

    source = csv_rows(dataset / "manifest.csv")
    metadata = json.loads((directory / "preassignment.json").read_text())
    groups = csv_rows(directory / "preassignment_groups.csv")
    config = metadata["configuration"]
    require(len(groups) == len(source), "Graph changed occurrence count")
    members = {r["frame_id"]: r["group_id"] for r in groups}
    require(set(members) == {r["frame_id"] for r in source}, "Graph changed population")
    expected = set()

    def edge(a, b, kind, evidence):
        if a != b:
            expected.add((*sorted((a, b)), kind, evidence))

    for key, kind in (
        ("content_id", "exact_content"),
        ("source_frame_key", "provider_same_source_frame"),
    ):
        buckets = defaultdict(list)
        for row in source:
            buckets[row[key]].append(row["frame_id"])
        for ids in buckets.values():
            for other in sorted(ids)[1:]:
                edge(min(ids), other, kind, key)
    scenes, videos = defaultdict(list), defaultdict(list)
    for row in source:
        videos[row["archivo_video"]].append(row)
        match = re.search(r"(?:^|_)scene(\d+)(?:\.|_|$)", row["nombre_original"], re.I)
        if match:
            scenes[(row["archivo_video"], row["dataset_origen"], match[1])].append(
                row["frame_id"]
            )
    for ids in scenes.values():
        for other in sorted(ids)[1:]:
            edge(
                min(ids),
                other,
                "nominal_scene",
                "filename heuristic; not verified scene ground truth",
            )
    window = config["temporal_window_seconds"]
    for rows in videos.values():
        rows.sort(key=lambda r: (float(r["segundo"]), r["frame_id"]))
        for a, b in zip(rows, rows[1:], strict=False):
            if float(b["segundo"]) - float(a["segundo"]) <= window:
                edge(
                    a["frame_id"],
                    b["frame_id"],
                    "temporal_window",
                    f"provider seconds; <= {window}; transitive closure",
                )
    stores = json.loads(stores_file.read_text())
    representatives = {}
    for row in sorted(source, key=lambda r: r["frame_id"]):
        representatives.setdefault(row["content_id"], row["frame_id"])
    for encoder in ("dinov2", "clip"):
        path = Path(stores[encoder]["path"])
        index = pd.read_parquet(path / "content_index.parquet")
        raw = np.load(path / "embeddings_raw.npy", allow_pickle=False)
        normalized = np.load(path / "embeddings_l2.npy", allow_pickle=False)
        require(
            set(index.content_id) == set(representatives), "Feature population differs"
        )
        require(
            np.array_equal(index.embedding_row, np.arange(len(index))),
            "Feature row order differs",
        )
        require(
            raw.shape
            == normalized.shape
            == (len(index), 384 if encoder == "dinov2" else 512),
            "Feature shape differs",
        )
        require(
            np.isfinite(raw).all() and np.isfinite(normalized).all(),
            "Nonfinite features",
        )
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        require(
            (norms > 0).all() and np.allclose(raw / norms, normalized, atol=1e-6),
            "L2 differs from raw",
        )
        for name, value in metadata["features"][encoder]["files"].items():
            require(
                sha((path / name).read_bytes()) == value, "Bound feature file changed"
            )
        threshold = config["cosine_thresholds"][encoder]
        vectors = normalized.astype(np.float64)
        # Independent row-wise dot products, no production edge generator.
        for i in range(len(index)):
            scores = vectors[i + 1 :] @ vectors[i]
            for offset in np.flatnonzero(scores >= threshold):
                j = i + 1 + int(offset)
                edge(
                    representatives[index.iloc[i].content_id],
                    representatives[index.iloc[j].content_id],
                    "visual_threshold",
                    f"{encoder} cosine >= {threshold}",
                )
    observed = {
        (r["left"], r["right"], r["kind"], r["evidence"])
        for r in csv_rows(directory / "relations.csv")
    }
    require(
        observed == expected, "Dependency edges differ from independent reconstruction"
    )
    parent = {key: key for key in members}

    def root(key):
        while parent[key] != key:
            key = parent[key]
        return key

    for a, b, _, _ in expected:
        parent[root(a)] = root(b)
        require(members[a] == members[b], "Dependency spans atomic groups")
    rebuilt = defaultdict(set)
    for key in members:
        rebuilt[root(key)].add(key)
    actual = defaultdict(set)
    for key, group in members.items():
        actual[group].add(key)
    require(
        {frozenset(v) for v in actual.values()}
        == {frozenset(v) for v in rebuilt.values()},
        "Atomic groups differ from connected components",
    )
    return {
        "state": "INDEPENDENT_GRAPH_REPLAY_PASSED",
        "records": len(members),
        "edges": len(expected),
        "groups": len(actual),
        "largest_group": max(map(len, actual.values())),
        "feature_raw_l2_and_bindings_verified": True,
        "production_graph_builder_called": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("tar", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--split", type=Path)
    parser.add_argument("--graph", type=Path)
    parser.add_argument("--stores", type=Path)
    args = parser.parse_args()
    if bool(args.graph) != bool(args.stores):
        parser.error("--graph and --stores must be supplied together")
    result = audit(args.tar, args.dataset, args.split)
    if args.graph:
        result["graph"] = audit_graph(args.dataset, args.graph, args.stores)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))
