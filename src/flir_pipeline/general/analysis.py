"""Dependency closure and comparable policies over a shared occurrence population."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict

import numpy as np
import pandas as pd

from flir_pipeline.general.contract import Manifest, digest
from flir_pipeline.splitting.base import SPLITS, SplitConfig
from flir_pipeline.splitting.construction import milp_assignment


def dependency_groups(manifest: Manifest):
    parent = {r.content_id: r.content_id for r in manifest.records}

    def find(c):
        while parent[c] != c:
            parent[c] = parent[parent[c]]
            c = parent[c]
        return c

    for r in sorted(manifest.relations, key=lambda r: r.relation_id):
        if r.accepted:
            a, b = sorted((find(r.left), find(r.right)))
            parent[b] = a
    groups = {c: find(c) for c in sorted(parent)}
    conflicts = [
        {
            "relation_id": r.relation_id,
            "left": r.left,
            "right": r.right,
            "component": groups[r.left],
            "reason": "rejected_inside_accepted_component",
        }
        for r in sorted(manifest.relations, key=lambda r: r.relation_id)
        if r.status == "rejected" and groups[r.left] == groups[r.right]
    ]
    return groups, conflicts


def leakage_report(
    manifest: Manifest, assignments: dict[str, str], groups: dict
) -> dict:
    """Edges use content endpoints; occurrence duplicates have separate denominators."""
    by_content, by_group = defaultdict(set), defaultdict(set)
    occurrences, sizes, annotations = Counter(), Counter(), defaultdict(set)
    for r in manifest.records:
        annotations[r.content_id].add(r.label_sha256)
        if r.frame_id in assignments:
            split = assignments[r.frame_id]
            by_content[r.content_id].add(split)
            by_group[groups[r.content_id]].add(split)
            occurrences[r.content_id] += 1
            sizes[groups[r.content_id]] += 1
    duplicates = [c for c, n in occurrences.items() if n > 1]
    relations = {}
    for state in ("confirmed", "candidate", "ambiguous", "rejected", "accepted"):
        edges = [
            r
            for r in manifest.relations
            if (r.accepted if state == "accepted" else r.status == state)
        ]
        covered = [r for r in edges if r.left in by_content and r.right in by_content]
        crossed = [
            r
            for r in covered
            if any(a != b for a in by_content[r.left] for b in by_content[r.right])
        ]
        relations[state] = {
            "crossing": len(crossed),
            "evaluable": len(covered),
            "total": len(edges),
            "unknown": len(edges) - len(covered),
            "relation_ids": [r.relation_id for r in crossed],
        }
    distributions = []
    for s in sorted(set(assignments.values())):
        rows = [r for r in manifest.records if assignments.get(r.frame_id) == s]
        distributions.append(
            {
                "split": s,
                "occurrences": len(rows),
                "contents": len({r.content_id for r in rows}),
                "annotation_status": dict(Counter(r.annotation_status for r in rows)),
                "class_objects": {
                    str(c.class_id): sum(
                        b[0] == c.class_id
                        for r in rows
                        for b in r.boxes
                        if r.annotation_status == "valid"
                    )
                    for c in manifest.classes
                },
            }
        )
    return {
        "coverage": {
            "assigned": len(assignments),
            "total": len(manifest.records),
            "unknown_or_excluded": len(manifest.records) - len(assignments),
        },
        "exact_duplicates": {
            "crossing_content_groups": sum(len(by_content[c]) > 1 for c in duplicates),
            "duplicated_content_groups": len(duplicates),
            "occurrences_in_duplicate_groups": sum(occurrences[c] for c in duplicates),
        },
        "relations": relations,
        "fragmentation": {
            "crossing_groups": sum(len(s) > 1 for s in by_group.values()),
            "evaluable_groups": len(by_group),
        },
        "group_occurrence_size_histogram": dict(
            sorted(Counter(sizes.values()).items())
        ),
        "annotation_byte_conflicts": {
            "content_groups": sum(len(v) > 1 for v in annotations.values()),
            "denominator": len(annotations),
            "semantic_conflict_inferred": False,
        },
        "distributions": distributions,
        "interpretation": "Zero accepted crossings validates only the declared constraints, not universal absence of leakage.",
    }


def analyze(manifest: Manifest, config: SplitConfig) -> dict:
    # Canonical order must govern diagnostics as well as solver membership.
    # Relation ID lists otherwise differ after saving/reloading a shuffled input.
    manifest = Manifest.model_validate(manifest.canonical())
    if config.strategy != "cluster_aware":
        raise ValueError("General policies use the shared balanced MILP strategy")
    if config.target_ratios is None:
        raise ValueError(
            "General policies require explicit ratios, never infer them from historical assignments"
        )
    if config.seeds != (0, 1, 2, 3, 4):
        raise ValueError("Closure protocol requires seeds 0 through 4")
    groups, conflicts = dependency_groups(manifest)
    # A held-out/invalid occurrence protects its exact content and accepted closure
    # in BOTH policies so their experimental populations remain identical.
    blocked = {
        groups[r.content_id]
        for r in manifest.records
        if r.exclusion_reasons or r.annotation_status == "invalid"
    }
    population = [
        r
        for r in sorted(manifest.records, key=lambda r: r.frame_id)
        if groups[r.content_id] not in blocked
    ]
    class_ids = sorted(c.class_id for c in manifest.classes)
    annotated = any(r.annotation_status in ("valid", "empty") for r in population)
    columns = ("record_count",)
    weights = (config.record_weight,)
    if annotated:
        columns += tuple(f"class_{c}" for c in class_ids) + ("empty_annotation_count",)
        weights += tuple(config.class_weight / len(class_ids) for _ in class_ids) + (
            config.empty_weight,
        )
    runs = []
    for policy in ("exact_content_balanced", "dependency_group_balanced"):
        membership = {
            c: (groups[c] if policy == "dependency_group_balanced" else c)
            for c in groups
        }
        data = []
        for r in population:
            counts = Counter(b[0] for b in r.boxes)
            data.append(
                {
                    "frame_id": r.frame_id,
                    "content_id": r.content_id,
                    "group_id": membership[r.content_id],
                    "record_count": 1,
                    "empty_annotation_count": int(r.annotation_status == "empty"),
                    **{f"class_{c}": counts[c] for c in class_ids},
                }
            )
        units = (
            pd.DataFrame(data)
            .groupby("group_id", sort=True)[list(columns)]
            .sum()
            .reset_index()
            if data
            else pd.DataFrame()
        )
        for seed in config.seeds:
            result = {
                "policy": policy,
                "seed": seed,
                "assignments": {},
                "membership_sha256": None,
            }
            if conflicts and policy == "dependency_group_balanced":
                result.update(state="blocked_conflicting_constraints", solver=None)
            elif len(units) < 3:
                result.update(
                    state="infeasible",
                    solver={
                        "status": "precheck_infeasible",
                        "reason": "fewer_than_three_atomic_groups",
                        "atomic_groups": len(units),
                    },
                )
            else:
                try:
                    assigned, solver = milp_assignment(
                        units,
                        np.array(config.target_ratios),
                        seed,
                        config,
                        balance_columns=columns,
                        family_weights=weights,
                    )
                except ValueError as error:
                    result.update(
                        state="solver_no_valid_assignment",
                        solver=getattr(
                            error,
                            "solver_result",
                            {"error": str(error), "heuristic_substituted": False},
                        ),
                    )
                else:
                    solver.pop(
                        "solve_seconds"
                    )  # runtime is operational, excluded from replay identity
                    result.update(
                        state="assigned",
                        solver=solver,
                        assignments={
                            r.frame_id: assigned[membership[r.content_id]]
                            for r in population
                        },
                    )
                    result["membership_sha256"] = digest(result["assignments"])
                    result["diagnostics"] = leakage_report(
                        manifest, result["assignments"], groups
                    )
                    if result["diagnostics"]["exact_duplicates"][
                        "crossing_content_groups"
                    ]:
                        raise ValueError("Solver split an exact content")
                    if (
                        policy == "dependency_group_balanced"
                        and result["diagnostics"]["relations"]["accepted"]["crossing"]
                    ):
                        raise ValueError("Solver split an accepted dependency")
            runs.append(result)
    historical = {
        r.frame_id: r.original_split
        for r in manifest.records
        if r.original_split is not None
    }
    return {
        "schema_version": "general_analysis_v1",
        "dataset_id": manifest.dataset_id(),
        "evidence_snapshot_sha256": digest(manifest.canonical()),
        "configuration": asdict(config),
        "effective_balance_columns": columns,
        "effective_family_weights": weights,
        "graph": {"membership": groups, "conflicts": conflicts},
        "population": {
            "eligible_occurrences": len(population),
            "total_occurrences": len(manifest.records),
            "excluded_frame_ids": sorted(
                set(r.frame_id for r in manifest.records)
                - set(r.frame_id for r in population)
            ),
        },
        "historical_assignments": historical,
        "historical_diagnostics": leakage_report(manifest, historical, groups),
        "runs": runs,
        "equivalence": [
            {
                "seed": seed,
                "memberships_equal": runs[i]["assignments"]
                == runs[i + 5]["assignments"]
                if runs[i]["state"] == runs[i + 5]["state"] == "assigned"
                else None,
            }
            for i, seed in enumerate(config.seeds)
        ],
        "temporal_coverage": dict(
            Counter(
                "unknown"
                if not r.positions
                else "multiple_confirmed"
                if sum(
                    p.kind == "decoded_pts" and p.status == "confirmed"
                    for p in r.positions
                )
                > 1
                else "unique_confirmed"
                if any(
                    p.kind == "decoded_pts" and p.status == "confirmed"
                    for p in r.positions
                )
                else "unconfirmed"
                for r in manifest.records
            )
        ),
        "residual_similarity": {"state": "not_evaluated_no_features_supplied"},
        "training_allowed": False,
    }


def residual_similarity(
    manifest: Manifest,
    result: dict,
    ids: list[str],
    matrix: np.ndarray,
    feature_space_id: str,
) -> dict:
    """Exact bounded-memory cross-split maxima on unique L2 contents; no threshold acceptance."""
    if (
        not feature_space_id
        or len(ids) != len(set(ids))
        or not set(ids) <= {r.content_id for r in manifest.records}
        or matrix.ndim != 2
        or matrix.shape[0] != len(ids)
        or matrix.shape[1] == 0
        or not np.isfinite(matrix).all()
        or not np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=1e-5)
    ):
        raise ValueError(
            "Features require unique known content IDs and finite unit L2 vectors"
        )
    rows = []
    for run in result["runs"]:
        if run["state"] != "assigned":
            continue
        mapping = {
            r.content_id: run["assignments"][r.frame_id]
            for r in manifest.records
            if r.frame_id in run["assignments"]
        }
        for split in SPLITS:
            own = [i for i, c in enumerate(ids) if mapping.get(c) == split]
            other = [
                i for i, c in enumerate(ids) if c in mapping and mapping[c] != split
            ]
            maxima = []
            if other:
                for start in range(0, len(own), 128):
                    maxima.extend(
                        np.clip(
                            matrix[own[start : start + 128]] @ matrix[other].T, -1, 1
                        )
                        .max(axis=1)
                        .tolist()
                    )
            rows.append(
                {
                    "policy": run["policy"],
                    "seed": run["seed"],
                    "split": split,
                    "covered_contents": len(own),
                    "assigned_contents": sum(v == split for v in mapping.values()),
                    "cross_split_comparisons": len(own) * len(other),
                    "mean_max_cross_split_cosine": float(np.mean(maxima))
                    if maxima
                    else None,
                    "maximum_cross_split_cosine": max(maxima) if maxima else None,
                }
            )
    return {
        "state": "evaluated",
        "feature_space_id": feature_space_id,
        "contents_with_features": len(ids),
        "rows": rows,
    }
