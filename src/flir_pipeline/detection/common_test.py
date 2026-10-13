"""Pre-training common-test proposals. No approval, freeze, fitting or runtime imports."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import pandas as pd

from flir_pipeline.similarity.storage import stable_id
from flir_pipeline.splitting.base import CLASS_COLUMNS
from flir_pipeline.splitting.construction import aggregate_groups
from flir_pipeline.splitting.labeled import (
    LabeledConfig,
    LabeledSource,
    sensitivity_groups,
)

VERSION = "detector_common_test_proposal_v1"


@dataclass(frozen=True)
class CommonTestConfig:
    """Candidate design, not authorization to execute a detector experiment."""

    protocol: str = VERSION
    option: str = "internal"
    anchor_rule: str = "original_historical_test"
    protection_states: tuple[str, ...] = (
        "confirmed_provenance",
        "visual_candidate",
        "unresolved",
    )
    split_seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    detector_seeds: tuple[int, ...] = (42, 43, 44)

    def __post_init__(self):
        for field in ("protection_states", "split_seeds", "detector_seeds"):
            object.__setattr__(self, field, tuple(getattr(self, field)))
        if self.protocol != VERSION or self.option not in {"internal", "prospective"}:
            raise ValueError("Unsupported common-test proposal")
        expected_anchor = (
            "original_historical_test" if self.option == "internal" else "none"
        )
        if self.anchor_rule != expected_anchor:
            raise ValueError(
                "Internal option uses historical anchors; prospective requires none without admitted data"
            )
        if (
            set(self.protection_states)
            != {"confirmed_provenance", "visual_candidate", "unresolved"}
            or len(self.protection_states) != 3
        ):
            raise ValueError(
                "v1 uniformly protects confirmed and all provisional states"
            )
        if self.split_seeds != tuple(range(5)) or self.detector_seeds != (42, 43, 44):
            raise ValueError(
                "v1 proposal retains five split seeds and three detector seeds"
            )


def class_support(records: pd.DataFrame, statistics: pd.DataFrame) -> list[dict]:
    """Occurrence counts retain conflicts; unique content support is not independence."""
    stats = statistics.set_index("frame_id").loc[records.frame_id]
    return [
        {
            "class_id": c,
            "instances": int(stats[f"class_{c}"].sum()),
            "images": int((stats[f"class_{c}"] > 0).sum()),
            "contents": int(stats.loc[stats[f"class_{c}"] > 0, "content_id"].nunique()),
        }
        for c in range(5)
    ]


def readiness_gate(
    *, option_available: bool, conflicts: int, test_covered: bool
) -> dict:
    """An accepted source/split review is never approval of a detector protocol.

    This release intentionally cannot accept approval or freeze inputs. Future
    transitions require an attributed signed-off protocol and a verified adapter.
    """
    blockers = [
        "protocol_approval_missing",
        "annotation_quality_review_missing",
        "common_test_freeze_missing",
        "train_val_assignments_not_built_or_verified",
        "initial_checkpoint_checksum_missing",
        "batch_and_environment_not_frozen",
        "training_adapter_not_implemented",
    ]
    if not option_available:
        blockers.append("prospective_annotated_independent_sources_unavailable")
    if conflicts:
        blockers.append("protected_annotation_conflicts_unresolved")
    if not test_covered:
        blockers.append("test_class_support_incomplete")
    return {
        "state": "PROPOSED" if option_available else "UNAVAILABLE",
        "review_state": "PENDING",
        "approval_state": "NOT_APPROVED",
        "freeze_state": "NOT_FROZEN",
        "ready_for_training": False,
        "blockers": blockers,
    }


def derive_proposal(
    source: LabeledSource, config: CommonTestConfig
) -> tuple[dict, dict[str, pd.DataFrame]]:
    """Reserve the full expanded graph before intersecting each development cohort.

    Test anchors are historical occurrences common to both populations. Other
    protected occurrences are quarantine, not fabricated evaluation examples.
    Components remain wholly RESERVED vs DEVELOPMENT; scoring is an explicit
    subset of reserved records and cannot release quarantine to training.
    """
    manifest, stats = source.manifest, source.statistics
    empty = manifest.iloc[:0].copy()
    if config.option == "prospective":
        return {
            "schema_version": VERSION,
            "option": "prospective",
            "scientific_claim": "no_external_population_supplied_or_admitted",
            "test_id": None,
            "test_occurrences": 0,
            "arms": [],
            "readiness": readiness_gate(
                option_available=False, conflicts=0, test_covered=False
            ),
            "prospective_admission_requirements": [
                "versioned_annotated_image_manifest_and_byte_checksums",
                "verified_acquisition_provenance",
                "attributed_non_exposure_review_train_val_test_selection_detector_design",
                "exact_and_cross_source_dependence_audit",
                "annotation_quality_adjudication",
                "five_class_support_in_independent_units",
                "explicit_protocol_approval_before_freeze",
            ],
        }, {
            "test_records.parquet": empty,
            "protected_records.parquet": empty,
            "quarantine_records.parquet": empty,
        }

    labeled = LabeledConfig(sensitivity_states=config.protection_states)
    groups, _, selection = sensitivity_groups(manifest, source.relations, "S3", labeled)
    anchors = manifest.loc[
        (manifest.cohort == "historical") & (manifest.original_split == "test")
    ].copy()
    if anchors.empty:
        raise ValueError("No historical test anchors; cannot fabricate a common test")
    anchor_contents = set(anchors.content_id)
    reserved_groups = set(
        groups.loc[groups.content_id.isin(anchor_contents), "group_id"]
    )
    protected_ids = set(groups.loc[groups.group_id.isin(reserved_groups), "content_id"])
    protected = manifest.loc[manifest.content_id.isin(protected_ids)].copy()
    quarantine = protected.loc[~protected.frame_id.isin(anchors.frame_id)].copy()
    # Test identity binds annotations and all recorded physical/historical provenance.
    test_id = stable_id({"records": anchors.to_dict(orient="records")})
    conflicts = [
        e
        for e in source.relations
        if e["state"] == "exact_bytes"
        and e["left_content_id"] in protected_ids
        and e["annotation_difference"]["annotation_conflict"]
    ]
    conflict_groups = {e["left_content_id"] for e in conflicts}
    support = class_support(anchors, stats)
    tables = {
        "test_records.parquet": anchors,
        "protected_records.parquet": protected,
        "quarantine_records.parquet": quarantine,
        "protection_groups.parquet": groups,
    }
    arms, pool_reports = [], {}
    for pop in ("historical", "expanded"):
        members = (
            manifest
            if pop == "expanded"
            else manifest.loc[manifest.cohort == "historical"]
        )
        development = members.loc[~members.content_id.isin(protected_ids)].copy()
        tables[f"development_{pop}.parquet"] = development
        history = manifest.loc[manifest.cohort == "historical"]
        sizes = history.original_split.value_counts()
        trainval_ratios = [
            float(sizes[s] / (sizes["train"] + sizes["val"])) for s in ("train", "val")
        ]
        pool_reports[pop] = {
            "population_occurrences": len(members),
            "development_occurrences": len(development),
            "development_contents": development.content_id.nunique(),
            "protected_occurrences": int(members.content_id.isin(protected_ids).sum()),
            "support": class_support(development, stats),
            "train_val_target_ratios": trainval_ratios,
        }
        for policy in ("S1", "S2", "S3"):
            devgroups, _, devselection = sensitivity_groups(
                development, source.relations, policy, labeled
            )
            devstats = stats.loc[stats.frame_id.isin(development.frame_id)]
            units = aggregate_groups(devstats, devgroups)
            support_groups = [int((units[col] > 0).sum()) for col in CLASS_COLUMNS]
            necessary = len(units) >= 2 and min(support_groups) >= 2
            tables[f"development_groups_{pop}_{policy}.parquet"] = devgroups
            for seed in config.split_seeds:
                arms.append(
                    {
                        "arm": f"{pop}-{policy}-{seed}",
                        "population": pop,
                        "policy": policy,
                        "split_seed": seed,
                        "test_id": test_id,
                        "test_occurrences": len(anchors),
                        "protection_id": stable_id({"contents": sorted(protected_ids)}),
                        "atomic_group_count": len(units),
                        "class_support_atomic_groups": support_groups,
                        "necessary_train_val_conditions_pass": necessary,
                        "assignment_feasibility": "NOT_SOLVED"
                        if necessary
                        else "INFEASIBLE_CLASS_OR_GROUP_SUPPORT",
                        "assignment_state": "NOT_BUILT",
                        "detector_seeds": list(config.detector_seeds),
                        "imposed_development_relation_ids": devselection[
                            "imposed_relation_ids"
                        ],
                    }
                )
    candidate = [
        e for e in source.relations if e["state"] in {"visual_candidate", "unresolved"}
    ]
    touching = [
        e
        for e in candidate
        if e["left_frame_id"] in set(anchors.frame_id)
        or e["right_frame_id"] in set(anchors.frame_id)
    ]
    previous_train = set(manifest.loc[manifest.original_split == "train", "content_id"])
    payload = {
        "schema_version": VERSION,
        "option": "internal",
        "scientific_claim": "retrospectively_informed_internal_holdout_candidate",
        "test_id": test_id,
        "test_occurrences": len(anchors),
        "test_contents": len(anchor_contents),
        "test_support": support,
        "protected_occurrences": len(protected),
        "protected_contents": len(protected_ids),
        "quarantine_occurrences": len(quarantine),
        "protected_annotation_conflict_groups": len(conflict_groups),
        "protected_annotation_conflict_relation_ids": sorted(
            e["relation_id"] for e in conflicts
        ),
        "historical_test_contents_in_historical_train": len(
            anchor_contents & previous_train
        ),
        "historical_test_candidate_evidence_records": len(touching),
        "historical_test_candidate_pairs": len(
            {tuple(sorted((e["left_frame_id"], e["right_frame_id"]))) for e in touching}
        ),
        "protection_selection": selection,
        "pools": pool_reports,
        "arms": arms,
        "readiness": readiness_gate(
            option_available=True,
            conflicts=len(conflict_groups),
            test_covered=all(s["instances"] > 0 for s in support),
        ),
        "inference_limits": [
            "test_not_external_or_prospectively_independent",
            "candidate_edges_are_hypotheses",
            "filenames_are_not_temporal_truth",
            "development_pool_support_does_not_prove_train_val_feasibility",
            "legacy_membership_aliases_do_not_predict_future_assignments",
        ],
    }
    return payload, tables


def verify_arm_assignments(
    source: LabeledSource,
    proposal: dict,
    tables: dict,
    assignments: dict[str, pd.DataFrame],
) -> dict:
    """Future handoff gate: check full real assignments, never synthesize validation.

    This helper does not authorize training. All required arms must be supplied,
    even if their identical memberships will later share one experiment identity.
    Quarantine is separate from scored test but remains fully outside development.
    """
    arms = {a["arm"]: a for a in proposal["arms"]}
    if not arms or set(assignments) != set(arms):
        raise ValueError("Complete proposed arm coverage required")
    memberships = defaultdict(list)
    expected_test = (
        tables["test_records.parquet"].sort_values("frame_id").reset_index(drop=True)
    )
    protected = set(tables["protected_records.parquet"].content_id)
    for name, arm in arms.items():
        rows = assignments[name].sort_values("frame_id").reset_index(drop=True)
        manifest = (
            source.manifest
            if arm["population"] == "expanded"
            else source.manifest.loc[source.manifest.cohort == "historical"]
        )
        if (
            not rows.frame_id.is_unique
            or set(rows.frame_id) != set(manifest.frame_id)
            or not rows.new_split.isin(["train", "val", "test", "quarantine"]).all()
        ):
            raise ValueError("Invalid arm coverage or partitions")
        pd.testing.assert_frame_equal(
            rows.drop(columns="new_split"),
            manifest.sort_values("frame_id").reset_index(drop=True),
        )
        test = (
            rows.loc[rows.new_split == "test"]
            .drop(columns="new_split")
            .reset_index(drop=True)
        )
        if not test.equals(expected_test):
            raise ValueError("Common test or annotations differ between arms")
        development = rows.loc[rows.new_split.isin(["train", "val"])]
        if set(development.content_id) & protected:
            raise ValueError("Protected content intersects train/validation")
        if set(development.frame_id) != set(
            tables[f"development_{arm['population']}.parquet"].frame_id
        ):
            raise ValueError("Development coverage changed or quarantine released")
        groups = tables[
            f"development_groups_{arm['population']}_{arm['policy']}.parquet"
        ]
        joined = development.merge(
            groups[["content_id", "group_id"]], on="content_id", validate="many_to_one"
        )
        if (
            joined.group_id.isna().any()
            or (joined.groupby("group_id").new_split.nunique() > 1).any()
        ):
            raise ValueError("Mandatory development group fracture")
        for split in ("train", "val", "test"):
            if any(
                s["instances"] == 0
                for s in class_support(
                    rows.loc[rows.new_split == split], source.statistics
                )
            ):
                raise ValueError(f"Missing class support in {split}")
        key = stable_id({"membership": rows[["frame_id", "new_split"]].values.tolist()})
        memberships[key].append(name)
    return {
        "assignments_valid": True,
        "distinct_memberships": len(memberships),
        "aliases": list(memberships.values()),
        "ready_for_training": False,
    }
