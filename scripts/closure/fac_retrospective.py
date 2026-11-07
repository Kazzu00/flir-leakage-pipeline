"""Read-only retrospective audit of distinct FAC populations; no split changes.

Shared candidate-group co-support is an explicit derived hypothesis diagnostic,
not a confirmed dependency. It is never passed to the partitioning engine here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from flir_pipeline.data.identity import dataset_id_from_manifest
from flir_pipeline.general.publication import read_json, write_json
from flir_pipeline.utils.hashing import sha256_file


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def audit(repo: Path, data: Path, prior: Path, output: Path):
    output.mkdir(parents=True, exist_ok=False)
    receipt = {}

    def bind(path):
        if path.is_relative_to(repo):
            key = "repo:" + path.relative_to(repo).as_posix()
        else:
            key = "data:" + path.relative_to(data).as_posix()
        receipt[key] = sha256_file(path)
        return path

    release = read_json(bind(prior / "release_receipt.json"))
    for name, checksum in release["files_sha256"].items():
        if sha256_file(prior / name) != checksum:
            raise ValueError(f"Prior release changed: {name}")
    manifest = pd.read_parquet(
        bind(repo / "data/manifests/flir_canonical_candidate_v1.parquet")
    )
    meta = read_json(bind(repo / "reports/data_manifest/manifest_metadata.json"))
    if dataset_id_from_manifest(manifest) != meta["dataset_id"]:
        raise ValueError("Historical manifest dataset identity changed")
    archives = {
        r["filename"]: r
        for r in csv_rows(bind(repo / "reports/data_inventory/archives.csv"))
    }
    for filename, path_col, hash_col in (
        ("Imagenes.zip", "source_member_path", "image_sha256"),
        ("Etiquetas.zip", "relative_label_path", "label_sha256"),
    ):
        path = bind(data / "fac" / filename)
        if sha256_file(path) != archives[filename]["archive_sha256"]:
            raise ValueError("Historical source archive changed")
        with zipfile.ZipFile(path) as source:
            names = Counter(source.namelist())
            for row in manifest.to_dict("records"):
                if (
                    names[row[path_col]] != 1
                    or hashlib.sha256(source.read(row[path_col])).hexdigest()
                    != row[hash_col]
                ):
                    raise ValueError("Historical occurrence/member identity changed")
    content_splits = {
        c: set(g.original_split) for c, g in manifest.groupby("content_id")
    }
    duplicated = manifest.groupby("content_id").filter(lambda g: len(g) > 1)
    cross = {c for c, splits in content_splits.items() if len(splits) > 1}
    pair_counts = [
        {
            "left_split": a,
            "right_split": b,
            "shared_exact_contents": sum(
                a in s and b in s for s in content_splits.values()
            ),
        }
        for a, b in itertools.combinations(("train", "val", "test"), 2)
    ]
    contamination = []
    for split, group in manifest.groupby("original_split"):
        in_train = group.content_id.isin(
            set(manifest.loc[manifest.original_split == "train", "content_id"])
        )
        contamination.append(
            {
                "split": split,
                "occurrences": len(group),
                "cross_split_exact_occurrences": int(
                    group.content_id.isin(cross).sum()
                ),
                "exact_occurrences_also_in_train": int(in_train.sum())
                if split != "train"
                else None,
                "fraction_also_in_train": float(in_train.mean())
                if split != "train"
                else None,
            }
        )
    # Persist the actual exact evidence separately from any nonexact hypothesis.
    exact_rows = manifest.loc[
        manifest.content_id.isin(cross),
        ["frame_id", "content_id", "original_split", "image_sha256", "label_sha256"],
    ]
    exact_rows.sort_values(["content_id", "frame_id"]).to_csv(
        output / "historical_exact_crossings.csv", index=False
    )
    support = csv_rows(
        bind(
            data / "operations/hypatia-sequence-evidence/shared_group_rank_support.csv"
        )
    )
    groups = defaultdict(set)
    for row in support:
        if row["labeled_content_id"] in content_splits:
            groups[row["visual_dependency_group_id"]].add(row["labeled_content_id"])
    derived_pairs = set()
    group_rows = []
    for group, contents in sorted(groups.items()):
        pairs = set(itertools.combinations(sorted(contents), 2))
        derived_pairs.update(pairs)
        splits = set().union(*(content_splits[c] for c in contents))
        group_rows.append(
            {
                "candidate_group": group,
                "covered_contents": len(contents),
                "historical_splits": sorted(splits),
                "spans_splits": len(splits) > 1,
                "induced_content_pairs": len(pairs),
            }
        )
    pair_crossings = sum(
        any(a != b for a in content_splits[left] for b in content_splits[right])
        for left, right in derived_pairs
    )
    covered_contents = set().union(*groups.values()) if groups else set()
    decisions = csv_rows(
        bind(
            data
            / "operations/sequence-linkage-assessment-20261105-v2/historical_decisions.csv"
        )
    )
    assessment = read_json(
        bind(data / "operations/sequence-linkage-assessment-20261105-v2/summary.json")
    )
    current_review = data / "operations/jp-deliverables-20261105-v2/review_jp"
    acceptance = read_json(bind(current_review / "acceptance_summary.json"))
    current_decisions = csv_rows(bind(current_review / "group_decisions.csv"))
    direct = csv_rows(bind(current_review / "historical_candidate_edges.csv"))
    direct_covered = [
        r
        for r in direct
        if r["labeled_content_id"] in content_splits
        and r["video_content_id"] in content_splits
    ]
    direct_cross = sum(
        any(
            a != b
            for a in content_splits[r["labeled_content_id"]]
            for b in content_splits[r["video_content_id"]]
        )
        for r in direct_covered
    )
    declared_current = sum(bool(r["decision"].strip()) for r in current_decisions)
    if (
        declared_current
        or acceptance["scientifically_accepted_current_dependencies"] != 0
    ):
        raise ValueError(
            "New human decisions require explicit audit of their authority; prior zero-decision report cannot be reused"
        )
    new = read_json(bind(prior / "fac-final/analysis.json"))
    current_manifest = read_json(bind(prior / "fac-final/manifest.json"))
    legacy = {
        "population": "Historical canonical candidate v1; never pooled with the later delivered variants",
        "dataset_id": meta["dataset_id"],
        "occurrences": len(manifest),
        "contents": len(content_splits),
        "split_counts": {
            str(k): int(v) for k, v in manifest.original_split.value_counts().items()
        },
        "exact": {
            "duplicate_content_groups": int(duplicated.content_id.nunique()),
            "cross_split_content_groups": len(cross),
            "cross_split_occurrences": len(exact_rows),
            "denominator_contents": len(content_splits),
            "denominator_occurrences": len(manifest),
            "split_pairs": pair_counts,
            "per_split": contamination,
            "duplicate_groups_with_different_annotation_bytes": int(
                (duplicated.groupby("content_id").label_sha256.nunique() > 1).sum()
            ),
            "semantic_annotation_conflict_adjudicated_here": False,
        },
        "confirmed_nonexact_dependencies": {
            "crossing": None,
            "available_accepted_relations": 0,
            "state": "not_established_with_current_authoritative_evidence",
            "absence_of_dependency_inferred": False,
        },
        "direct_candidate_edges": {
            "crossing": direct_cross,
            "evaluable": len(direct_covered),
            "total": len(direct),
            "unknown": len(direct) - len(direct_covered),
            "reason": "Video-content endpoints lack historical canonical split assignments",
        },
        "derived_shared_candidate_group_diagnostic": {
            "definition": "Distinct canonical content pairs sharing a retrieval candidate group; an exploratory co-support hypothesis only",
            "crossing_content_pairs": pair_crossings,
            "covered_content_pairs": len(derived_pairs),
            "covered_contents": len(covered_contents),
            "unknown_contents": len(content_splits) - len(covered_contents),
            "denominator_contents": len(content_splits),
            "candidate_groups": group_rows,
            "accepted_as_split_constraint": False,
            "ground_truth": False,
        },
        "historical_reviews": {
            "rows": len(decisions),
            "status_counts": dict(Counter(r["manual_decision"] for r in decisions)),
            "ambiguous_occurrence_support": {
                split: int(
                    manifest.loc[manifest.original_split == split, "content_id"]
                    .isin(
                        {
                            r["labeled_content_id"]
                            for r in decisions
                            if r["manual_decision"] == "ambiguous"
                        }
                    )
                    .sum()
                )
                for split in ("train", "val", "test")
            },
            "current_sequence_assignments_authorized": assessment[
                "sequence_assignments_authorized"
            ],
            "historical_supported_not_promoted": True,
        },
    }
    delivery = {
        "population": "Later delivered inventory: original/processed primary occurrences and preserved references/diagnostics",
        "dataset_id": new["dataset_id"],
        "occurrences": len(current_manifest["records"]),
        "historical": new["historical_diagnostics"],
        "provisional_population": new["population"],
        "temporal_coverage": new["temporal_coverage"],
        "policies_equivalent": new["equivalence"],
        "runs": [
            {
                k: r[k]
                for k in (
                    "policy",
                    "seed",
                    "state",
                    "solver",
                    "membership_sha256",
                    "diagnostics",
                )
            }
            for r in new["runs"]
        ],
    }
    result = {
        "historical_canonical": legacy,
        "later_delivery": delivery,
        "scientific_conclusion": "Exact historical cross-split overlap observed. Nonexact current dependencies remain unadjudicated; no detector effect is estimated.",
        "test_protected_modified": False,
        "training_executed": False,
        "nominal_timestamps_used_as_temporal_truth": False,
    }
    write_json(output / "retrospective.json", result)
    write_json(output / "input_receipt.json", receipt)
    print(
        json.dumps(
            {
                "historical_occurrences": len(manifest),
                "exact_cross_split_groups": len(cross),
                "exact_split_pairs": pair_counts,
                "derived_candidate_pairs": len(derived_pairs),
                "derived_candidate_crossings": pair_crossings,
                "current_accepted_nonexact_dependencies": 0,
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--prior", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    audit(*(p.resolve() for p in (args.repo, args.data, args.prior, args.output)))
