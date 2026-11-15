"""Provisional paired assignments with explicit relation and exclusion semantics.

Both policies share the same allocator, seed, ratios and eligible population.
Exact duplicates and structural pairs are precautionary minimum units. Only
explicitly accepted, attributed relations extend grouped-policy units. No
unreviewed historical sequence or whole video becomes an atomic group.
"""

from __future__ import annotations

import platform
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from flir_pipeline.explorer.paired_inputs import (
    digest,
    read_json,
    rows_csv,
    safe_relative,
    write_csv,
    write_json,
)
from flir_pipeline.explorer.paired_timeline import public_row, render_html
from flir_pipeline.splitting.construction import atomic_random_assignment
from flir_pipeline.utils.hashing import sha256_file


def relation_rows(cfg, root, pairs):
    relations = (
        rows_csv(safe_relative(root, cfg["relations"])) if cfg.get("relations") else []
    )
    ids = {p["pair_id"] for p in pairs}
    seen = set()
    for r in relations:
        if (
            r["relation_id"] in seen
            or r["left_pair_id"] not in ids
            or r["right_pair_id"] not in ids
        ):
            raise ValueError("Duplicate relation or unknown endpoint")
        seen.add(r["relation_id"])
        if r["status"] not in {"accepted", "candidate", "ambiguous", "rejected"}:
            raise ValueError("Unknown relation status")
        if r["kind"] not in {"sequence", "dependency", "pair"}:
            raise ValueError("Unknown relation kind")
        if r["status"] == "accepted":
            if not all(
                r.get(k)
                for k in ("reviewer", "decision_at", "evidence_path", "evidence_sha256")
            ):
                raise ValueError(
                    "Accepted relation requires attributed source-bound evidence"
                )
            if (
                sha256_file(safe_relative(root, r["evidence_path"]))
                != r["evidence_sha256"]
            ):
                raise ValueError("Accepted relation evidence changed")
    return relations


def groups_for(pairs, relations, grouped):
    """Connected closure preserves transitive atomicity across both variants."""
    ids = sorted(p["pair_id"] for p in pairs)
    index = {key: i for i, key in enumerate(ids)}
    edges = []
    content_pairs = defaultdict(list)
    for p in pairs:
        for side in ("original", "no_hud"):
            content_pairs[p[side]["sha256"]].append(p["pair_id"])
    for members in content_pairs.values():
        edges.extend((members[0], other) for other in members[1:])
    if grouped:
        edges.extend(
            (r["left_pair_id"], r["right_pair_id"])
            for r in relations
            if r["status"] == "accepted"
        )
    rr = [index[a] for a, b in edges]
    cc = [index[b] for a, b in edges]
    graph = coo_matrix(
        (np.ones(len(edges)), (rr, cc)), shape=(len(ids), len(ids))
    ).tocsr()
    _, labels = connected_components(graph, directed=False)
    components = defaultdict(list)
    for key, label in zip(ids, labels, strict=True):
        components[int(label)].append(key)
    return {key: digest(members) for members in components.values() for key in members}


def eligible_population(pairs, relations):
    """Exclusion propagates through accepted and exact units for BOTH policies."""
    groups = groups_for(pairs, relations, True)
    blocked = {groups[p["pair_id"]] for p in pairs if p["status"] != "eligible"}
    result = []
    for p in pairs:
        p = dict(p)
        if p["status"] == "eligible" and groups[p["pair_id"]] in blocked:
            p["status"] = "quarantine"
            p["reasons"] = [*p["reasons"], "atomic_group_contains_excluded_member"]
        result.append(p)
    return result


def assign(pairs, relations, cfg):
    population = eligible_population(pairs, relations)
    eligible = [p for p in population if p["status"] == "eligible"]
    eligible_ids = {p["pair_id"] for p in eligible}
    eligible_relations = [
        r
        for r in relations
        if r["left_pair_id"] in eligible_ids and r["right_pair_id"] in eligible_ids
    ]
    assignments, memberships = {}, {}
    for policy in ("random", "grouped"):
        membership = groups_for(eligible, eligible_relations, policy == "grouped")
        # Count each image pair once, so two visual variants do not double class weight.
        counts = Counter(membership.values())
        units = pd.DataFrame(
            [
                {"group_id": k, "record_count": v, "content_count": v}
                for k, v in sorted(counts.items())
            ]
        )
        mapping, _ = atomic_random_assignment(
            units, np.asarray(cfg["ratios"]), cfg["seed"]
        )
        assignments[policy] = {key: mapping[group] for key, group in membership.items()}
        memberships[policy] = membership
    records = []
    for p in population:
        for side in ("original", "no_hud"):
            source = p[side]
            record = {
                "pair_id": p["pair_id"],
                "filename": p["filename"],
                "original_name": p["original_name"],
                "variant": side,
                "occurrence_id": source["inventory_occurrence_id"],
                "content_id": source["sha256"],
                "source": source["source"],
                "member": source["member"],
                "label_sha256": source["label_sha256"],
                "reference_content_id": p["reference_content_id"],
                "video": p["source_video"],
                "sequence_id": p["sequence_id"],
                "temporal_status": p["temporal_status"],
                "historical_split": p["historical_split"],
                "status": p["status"],
                "reasons": p["reasons"],
                "label_empty": p["label_empty"],
                "classes": p["classes"],
                "num_objects": p["num_objects"],
                "training_allowed": False,
            }
            for policy in assignments:
                record[f"{policy}_group"] = memberships[policy].get(p["pair_id"], "")
                record[f"{policy}_split"] = assignments[policy].get(p["pair_id"], "")
            records.append(record)
    return sorted(records, key=lambda r: (r["pair_id"], r["variant"])), population


def verify_records(records, pairs, relations):
    """Independent checks use content/edges directly, never trust emitted group IDs."""
    expected = {
        p[s]["inventory_occurrence_id"] for p in pairs for s in ("original", "no_hud")
    }
    if (
        len(records) != len(expected)
        or {r["occurrence_id"] for r in records} != expected
    ):
        raise ValueError("Occurrence coverage mismatch")
    source_rows = {
        p[s]["inventory_occurrence_id"]: (p, s)
        for p in pairs
        for s in ("original", "no_hud")
    }
    by_pair = defaultdict(list)
    for r in records:
        p, side = source_rows[r["occurrence_id"]]
        if (r["pair_id"], r["variant"], r["content_id"], r["label_sha256"]) != (
            p["pair_id"],
            side,
            p[side]["sha256"],
            p[side]["label_sha256"],
        ):
            raise ValueError("Manifest identity or label differs from source")
        if p["status"] != "eligible" and r["status"] == "eligible":
            raise ValueError("Source exclusion removed")
        by_pair[r["pair_id"]].append(r)
        for policy in ("random", "grouped"):
            assigned = r[f"{policy}_split"]
            if r["status"] == "eligible" and assigned not in {"train", "val", "test"}:
                raise ValueError("Eligible member missing assignment")
            if r["status"] != "eligible" and assigned:
                raise ValueError("Excluded member assigned")
    for policy in ("random", "grouped"):
        content_splits = defaultdict(set)
        for group in by_pair.values():
            if (
                len(group) != 2
                or {r["variant"] for r in group} != {"original", "no_hud"}
                or len({r[f"{policy}_split"] for r in group}) != 1
            ):
                raise ValueError("Pair indivisibility failed")
            for r in group:
                if r[f"{policy}_split"]:
                    content_splits[r["content_id"]].add(r[f"{policy}_split"])
        if any(len(v) > 1 for v in content_splits.values()):
            raise ValueError("Exact content crosses splits")
    crossings = {policy: Counter() for policy in ("random", "grouped")}
    for relation in relations:
        a, b = (
            by_pair[relation["left_pair_id"]][0],
            by_pair[relation["right_pair_id"]][0],
        )
        for policy in crossings:
            left, right = a[f"{policy}_split"], b[f"{policy}_split"]
            if left and right and left != right:
                crossings[policy][relation["status"]] += 1
                if policy == "grouped" and relation["status"] == "accepted":
                    raise ValueError("Accepted relation crosses grouped splits")
            if (
                policy == "grouped"
                and relation["status"] == "accepted"
                and bool(left) != bool(right)
            ):
                raise ValueError("Accepted group partly assigned")
    for p in pairs:
        if p["protected_reference"] and any(
            by_pair[p["pair_id"]][0][f"{policy}_split"]
            for policy in ("random", "grouped")
        ):
            raise ValueError("Protected reference reassigned")
    return {
        "occurrence_coverage": True,
        "pair_indivisibility": True,
        "exact_content_cross_split": 0,
        "accepted_grouped_cross_split": 0,
        "relation_crossings": {k: dict(v) for k, v in crossings.items()},
        "protected_references_reassigned": 0,
    }


def build_splits(pairs, cfg, root, output):
    if (output / "split_manifest.csv").exists():
        raise ValueError("Split output already exists; use a fresh delivery")
    relations = relation_rows(cfg, root, pairs)
    first, population = assign(pairs, relations, cfg)
    second, _ = assign(list(reversed(pairs)), list(reversed(relations)), cfg)
    verification = verify_records(first, pairs, relations)
    if first != second:
        raise ValueError("Repeated assignments differ")
    verification.update(
        status="provisional",
        training_allowed=False,
        repetitions=2,
        identical_memberships=True,
        membership_sha256=digest(first),
        second_membership_sha256=digest(second),
        accepted_relations=sum(r["status"] == "accepted" for r in relations),
        eligibility=dict(Counter(p["status"] for p in population)),
        exclusions=dict(Counter(reason for p in population for reason in p["reasons"])),
    )
    comparisons = []
    by_pair = {r["pair_id"]: r for r in first}
    candidate_crossings = {}
    for policy in ("random", "grouped"):
        candidate_groups = defaultdict(list)
        for p in population:
            for group in p["candidate_dependency_groups"]:
                candidate_groups[group].append(p)
        cross = {
            g: sorted(
                {
                    by_pair[p["pair_id"]][f"{policy}_split"]
                    for p in pp
                    if by_pair[p["pair_id"]][f"{policy}_split"]
                }
            )
            for g, pp in candidate_groups.items()
        }
        candidate_crossings[policy] = {g: s for g, s in cross.items() if len(s) > 1}
        for split in ("train", "val", "test"):
            subset = [
                p
                for p in population
                if by_pair[p["pair_id"]][f"{policy}_split"] == split
            ]
            comparisons.append(
                {
                    "policy": policy,
                    "split": split,
                    "pairs": len(subset),
                    "occurrences": 2 * len(subset),
                    "empty_pairs": sum(p["label_empty"] for p in subset),
                    "objects_per_variant": sum(p["num_objects"] for p in subset),
                    "class_objects_per_variant": dict(
                        Counter(str(b[0]) for p in subset for b in p["boxes"])
                    ),
                    "class_positive_pairs": dict(
                        Counter(str(c) for p in subset for c in p["classes"])
                    ),
                }
            )
    verification["historical_candidate_group_crossings"] = candidate_crossings
    verification["same_allocator_for_policies"] = True
    verification["policies_identical"] = all(
        r["random_split"] == r["grouped_split"] for r in first
    )
    verification["interpretation"] = (
        "No grouping benefit can be evaluated without current accepted sequence/dependency relations; equality is expected when no additional relations are authorized. Eligible means eligible for provisional simulation, never training admission."
    )
    write_csv(output / "split_manifest.csv", first)
    write_csv(output / "split_manifest_repeat.csv", second)
    write_csv(output / "policy_comparison.csv", comparisons)
    write_json(output / "relations.json", relations)
    write_json(output / "split_verification.json", verification)
    source_root = Path(__file__).resolve().parents[3]
    source_files = [
        Path(__file__),
        Path(__file__).with_name("construction.py"),
        source_root / "uv.lock",
        source_root / "src/flir_pipeline/explorer/paired_inputs.py",
        source_root / "src/flir_pipeline/explorer/paired_timeline.py",
        source_root / "src/flir_pipeline/explorer/paired_delivery.py",
    ]
    write_json(
        output / "split_provenance.json",
        {
            "python": platform.python_version(),
            "code_sha256": {
                str(p.relative_to(source_root)): sha256_file(p) for p in source_files
            },
            "input_timeline_sha256": sha256_file(output / "timeline_records.json"),
            "configuration_sha256": digest(cfg),
            "relation_evidence_sha256": {
                r["evidence_path"]: r["evidence_sha256"]
                for r in relations
                if r["status"] == "accepted"
            },
            "source_receipt": "provenance.json",
            "training_allowed": False,
        },
    )
    (output / "split_config.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8"
    )
    for p in population:
        r = by_pair[p["pair_id"]]
        p.update(
            random_split=r["random_split"],
            grouped_split=r["grouped_split"],
            candidate_cross_split={
                policy: [
                    g
                    for g in p["candidate_dependency_groups"]
                    if g in candidate_crossings[policy]
                ]
                for policy in candidate_crossings
            },
        )
    # Preserve the validated initial timeline and publish an enriched sibling.
    render_html(population, output / "timeline_with_splits.html")
    write_csv(
        output / "timeline_with_splits_manifest.csv",
        [public_row(p) for p in population],
    )
    write_json(
        output / "split_receipt.json",
        {
            name: sha256_file(output / name)
            for name in [
                "split_manifest.csv",
                "split_manifest_repeat.csv",
                "relations.json",
                "split_config.yaml",
                "policy_comparison.csv",
                "split_verification.json",
                "split_provenance.json",
                "timeline_with_splits.html",
                "timeline_with_splits_manifest.csv",
            ]
        },
    )
    return verification


def verify_saved(output: Path):
    for name, checksum in read_json(output / "split_receipt.json").items():
        if sha256_file(output / name) != checksum:
            raise ValueError(f"Split artifact changed: {name}")
    rows = rows_csv(output / "split_manifest.csv")
    if sha256_file(output / "split_manifest.csv") != sha256_file(
        output / "split_manifest_repeat.csv"
    ):
        raise ValueError("CSV repeats differ")
    verification = verify_records(
        rows,
        read_json(output / "timeline_records.json"),
        read_json(output / "relations.json"),
    )
    replayed, _ = assign(
        read_json(output / "timeline_records.json"),
        read_json(output / "relations.json"),
        yaml.safe_load((output / "split_config.yaml").read_text(encoding="utf-8")),
    )
    replay_hash = digest(replayed)
    if (
        replay_hash
        != read_json(output / "split_verification.json")["membership_sha256"]
    ):
        raise ValueError("Independent process replay differs")
    verification.update(independent_cli_replay=True, membership_sha256=replay_hash)
    write_json(output / "independent_replay.json", verification)
    return verification
