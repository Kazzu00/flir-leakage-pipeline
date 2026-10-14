"""Two-destination construction on proposed development pools; no training admission."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from flir_pipeline.detection.common_test import class_support, verify_arm_assignments
from flir_pipeline.similarity.storage import stable_id
from flir_pipeline.splitting.base import BALANCE_COLUMNS, SplitConfig
from flir_pipeline.splitting.construction import (
    AssignmentFailure,
    aggregate_groups,
    milp_assignment,
    propagate,
    random_assignment,
)
from flir_pipeline.splitting.labeled import _require
from flir_pipeline.splitting.labeled_storage import _relation_metrics
from flir_pipeline.splitting.metrics import balance_metrics

VERSION = "fixed_test_train_val_v1"
DESTINATIONS = ("train", "val")
POPULATIONS = ("historical", "expanded")
POLICIES = ("S1", "S2", "S3")
BLOCKERS = [
    "human_annotation_adjudication_missing",
    "methodological_class_sufficiency_decision_missing",
    "dependency_policy_approval_missing",
    "test_approval_and_verified_freeze_missing",
    "validated_materialization_adapter_missing",
    "detector_configuration_and_environment_freeze_missing",
]


@dataclass(frozen=True)
class FixedTestConfig:
    """Both phases use identical solver limits, weights and historical ratio rule."""

    seeds: tuple[int, ...] = (0,)
    node_limit: int = 32
    mip_rel_gap: float = 0.001
    record_weight: float = 1.0
    class_weight: float = 1.0
    empty_weight: float = 1.0

    def __post_init__(self):
        object.__setattr__(self, "seeds", tuple(self.seeds))
        if self.seeds not in ((0,), tuple(range(5))):
            raise ValueError("v1 requires pilot seed 0 or complete seeds 0..4")
        self.balance()

    def balance(self):
        return SplitConfig(**asdict(self))

    def mathematical_settings(self):
        result = asdict(self)
        result.pop("seeds")
        return {
            **result,
            "destinations": list(DESTINATIONS),
            "ratio_rule": "historical_train_val_counts_renormalized",
            "milp_positive_class_coverage": True,
            "random_class_repair": False,
        }


def prepared_arm(source, plan, tables, arm):
    population = arm["population"]
    manifest = (
        source.manifest
        if population == "expanded"
        else source.manifest.loc[source.manifest.cohort == "historical"]
    )
    development = tables[f"development_{population}.parquet"]
    stats = source.statistics.loc[source.statistics.frame_id.isin(development.frame_id)]
    groups = tables[f"development_groups_{population}_{arm['policy']}.parquet"]
    units = aggregate_groups(stats, groups)
    history = source.manifest.loc[
        source.manifest.cohort == "historical"
    ].original_split.value_counts()
    ratios = np.array([history[s] for s in DESTINATIONS], dtype=float)
    ratios /= ratios.sum()
    _require(
        np.array_equal(ratios, plan["pools"][population]["train_val_target_ratios"]),
        "Proposal ratio mismatch",
    )
    return manifest, development, stats, groups, units, ratios


def construct_arm(source, plan, tables, arm, config):
    manifest, _, stats, groups, units, ratios = prepared_arm(source, plan, tables, arm)
    if len(units) < 2:
        raise AssignmentFailure(
            "Fewer than two development groups",
            {
                "failure": "INFEASIBLE_GROUP_SUPPORT",
                "primal_feasibility_verified": False,
            },
        )
    if arm["policy"] == "S1":
        assignment, solver = random_assignment(
            units, ratios, arm["split_seed"], destinations=DESTINATIONS
        )
    else:
        assignment, solver = milp_assignment(
            units,
            ratios,
            arm["split_seed"],
            config.balance(),
            destinations=DESTINATIONS,
            require_class_coverage=True,
        )
    contents, records = propagate(stats, groups, assignment)
    mapping = records.set_index("frame_id").new_split.to_dict()
    mapping.update({f: "test" for f in tables["test_records.parquet"].frame_id})
    full = manifest.copy()
    full["new_split"] = full.frame_id.map(mapping).fillna("quarantine")
    full = full.sort_values("frame_id").reset_index(drop=True)
    solver.pop(
        "solve_seconds", None
    )  # Operational wall time is not reproducibility evidence.
    return full, contents, solver


def inspect_arm(source, plan, tables, arm, rows, solver, config):
    """Recompute all hard invariants independently of solver status or receipt hashes.

    Missing class support is a recorded failed candidate; identity/atomicity failures
    are corruption and abort verification. Quarantine may share exact test content.
    """
    manifest, development, stats, groups, units, ratios = prepared_arm(
        source, plan, tables, arm
    )
    rows = rows.sort_values("frame_id").reset_index(drop=True)
    expected = manifest.sort_values("frame_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(rows.drop(columns="new_split"), expected)
    _require(
        rows.frame_id.is_unique
        and rows.new_split.isin([*DESTINATIONS, "test", "quarantine"]).all(),
        "Invalid occurrences/roles",
    )
    test = (
        rows.loc[rows.new_split == "test"]
        .drop(columns="new_split")
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(
        test,
        tables["test_records.parquet"].sort_values("frame_id").reset_index(drop=True),
    )
    protected = set(tables["protected_records.parquet"].content_id)
    dev = rows.loc[rows.new_split.isin(DESTINATIONS)]
    _require(
        set(dev.frame_id) == set(development.frame_id)
        and not set(dev.content_id) & protected,
        "Protected content or development coverage changed",
    )
    expected_quarantine = (
        set(manifest.frame_id) - set(development.frame_id) - set(test.frame_id)
    )
    _require(
        set(rows.loc[rows.new_split == "quarantine", "frame_id"])
        == expected_quarantine,
        "Quarantine changed",
    )
    scored = rows.loc[rows.new_split != "quarantine"]
    exact_cross = int((scored.groupby("content_id").new_split.nunique() > 1).sum())
    joined = dev.merge(
        groups[["content_id", "group_id"]], on="content_id", validate="many_to_one"
    )
    fractures = int((joined.groupby("group_id").new_split.nunique() > 1).sum())
    _require(exact_cross == fractures == 0, "Exact content or mandatory group fracture")
    selected = set(plan["protection_selection"]["imposed_relation_ids"])
    protected_cross = [
        e["relation_id"]
        for e in source.relations
        if e["relation_id"] in selected
        and (
            (e["left_content_id"] in protected) != (e["right_content_id"] in protected)
        )
    ]
    _require(not protected_cross, "Selected dependency crosses protection closure")
    mapping = rows.set_index("frame_id").new_split.to_dict()
    imposed = set(arm["imposed_development_relation_ids"])
    _require(
        all(
            mapping[e["left_frame_id"]] == mapping[e["right_frame_id"]]
            for e in source.relations
            if e["relation_id"] in imposed
        ),
        "Imposed dependency fracture",
    )
    balance, classes = balance_metrics(dev, stats, ratios, destinations=DESTINATIONS)
    support = {
        s: class_support(rows.loc[rows.new_split == s], source.statistics)
        for s in (*DESTINATIONS, "test", "quarantine")
    }
    coverage = all(
        c["instances"] > 0 for s in (*DESTINATIONS, "test") for c in support[s]
    )
    if arm["policy"] != "S1":
        verify_solver(solver, dev, stats, units, ratios, arm["split_seed"], config)
        _require(balance["all_classes_covered"], "MILP violates hard class coverage")
    else:
        _require(
            solver
            == {
                "method": "seeded_content_permutation_record_cuts",
                "seed": arm["split_seed"],
                "class_balancing": False,
            },
            "Random solver metadata changed",
        )
    if coverage:
        verify_arm_assignments(
            source, {**plan, "arms": [arm]}, tables, {arm["arm"]: rows}
        )
    metrics = {
        "state": "PROVISIONAL_VERIFIED" if coverage else "FAILED_CLASS_COVERAGE",
        "ready_for_training": False,
        "blockers": BLOCKERS,
        "exact_cross_split_contents_excluding_quarantine": exact_cross,
        "mandatory_group_fractures": fractures,
        "protected_dependency_crossings": len(protected_cross),
        "test_identity_preserved": True,
        "occurrence_conservation_verified": True,
        "atomic_groups": len(units),
        "balance": balance,
        "class_support": support,
        "development_relations_by_state": _relation_metrics(dev, source.relations),
        "imposed_development_relation_ids": sorted(imposed),
        "hypothetical_relations_promoted": False,
        "independence_demonstrated": False,
        "cosine_similarity_evaluated": False,
        "role_sizes": {
            s: {
                "occurrences": int((rows.new_split == s).sum()),
                "contents": int(rows.loc[rows.new_split == s, "content_id"].nunique()),
            }
            for s in (*DESTINATIONS, "test", "quarantine")
        },
    }
    return metrics, classes


def verify_solver(solver, dev, stats, units, ratios, seed, config):
    actual = dev[["frame_id", "new_split"]].merge(
        stats, on="frame_id", validate="one_to_one"
    )
    counts = (
        actual.groupby("new_split")[list(BALANCE_COLUMNS)]
        .sum()
        .reindex(DESTINATIONS)
        .to_numpy()
    )
    targets = ratios[:, None] * stats[list(BALANCE_COLUMNS)].sum().to_numpy()[None, :]
    weights = np.array(
        [config.record_weight, *([config.class_weight / 5] * 5), config.empty_weight]
    )
    objective = float(
        (abs(counts - targets) * weights[None, :] / (2 * np.maximum(targets, 1))).sum()
    )
    _require(
        solver["method"] == "scipy.optimize.milp/HiGHS"
        and solver["seed"] == seed
        and solver["atomic_group_count"] == len(units)
        and solver["profile_count"]
        == len(np.unique(units[list(BALANCE_COLUMNS)].to_numpy(), axis=0))
        and solver["primal_feasibility_verified"] is True
        and solver["unique_optimum_proven"] is False,
        "Solver identity/feasibility mismatch",
    )
    _require(
        np.isclose(solver["objective"], objective, rtol=1e-6, atol=1e-6),
        "Solver objective disagrees with assignments",
    )
    bound, gap = solver["dual_bound"], solver["mip_gap"]
    _require(
        np.isfinite([objective, bound, gap]).all()
        and bound >= -1e-6
        and bound <= objective + 1e-6
        and gap >= 0,
        "Invalid solver bound/gap",
    )
    expected_gap = abs(objective - bound) / abs(objective) if objective else 0.0
    _require(
        np.isclose(gap, expected_gap, atol=1e-5, rtol=1e-5),
        "Solver gap disagrees with objective/bound",
    )
    _require(
        solver["status"] in (0, 1, 4)
        and solver["optimal_within_tolerance"] == (solver["status"] == 0),
        "Solver status/optimality mismatch",
    )
    _require(
        not solver["optimal_within_tolerance"] or gap <= config.mip_rel_gap + 1e-6,
        "Unsupported optimality claim",
    )


def membership_id(rows):
    return stable_id(
        {
            "rows": rows.sort_values("frame_id")[
                ["frame_id", "content_id", "new_split"]
            ].values.tolist()
        }
    )


def suite_summary(entries):
    aliases = {}
    for entry in entries:
        if entry["membership_id"] is not None:
            aliases.setdefault(entry["membership_id"], []).append(entry["arm"])
    passed = sum(e["state"] == "PROVISIONAL_VERIFIED" for e in entries)
    return {
        "schema_version": VERSION,
        "state": "PROVISIONAL_CANDIDATES",
        "verdict": "APROBADO PARA REVISIÓN DE PARTICIONES"
        if passed == len(entries)
        else "REQUIERE CORRECCIONES",
        "ready_for_training": False,
        "blockers": BLOCKERS,
        "configurations": len(entries),
        "verified_candidates": passed,
        "failed_configurations": len(entries) - passed,
        "distinct_memberships": len(aliases),
        "membership_groups": aliases,
        "aliases": [v for v in aliases.values() if len(v) > 1],
        "distinct_memberships_by_population": {
            p: len(
                {
                    e["membership_id"]
                    for e in entries
                    if e["population"] == p and e["membership_id"] is not None
                }
            )
            for p in POPULATIONS
        },
        "cross_population_interpretation": "different_development_sets_with_identical_proposed_test",
        "runs": entries,
    }
