"""Fresh labeled candidate suites, source snapshots and semantic verification."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

from flir_pipeline.data.identity import dataset_id_from_manifest
from flir_pipeline.similarity.storage import (
    execution_provenance,
    file_sha256,
    read_json,
    stable_id,
    write_json,
)
from flir_pipeline.splitting.base import BALANCE_COLUMNS
from flir_pipeline.splitting.construction import (
    aggregate_groups,
    milp_assignment,
    propagate,
    random_assignment,
)
from flir_pipeline.splitting.labeled import (
    STATES,
    VERSION,
    LabeledConfig,
    _require,
    load_labeled_audit,
    population,
    sensitivity_groups,
    verify_original_sources,
)
from flir_pipeline.splitting.metrics import balance_metrics, verify_assignments

RUN_FILES = {
    "content_memberships.parquet",
    "occurrence_memberships.parquet",
    "atomic_groups.parquet",
    "group_balance.parquet",
    "class_balance.parquet",
    "metrics.json",
    "metadata.json",
}


def _implementation():
    root = Path(__file__).parents[1]
    paths = [
        *(
            Path(__file__).parent / name
            for name in (
                "labeled.py",
                "labeled_storage.py",
                "construction.py",
                "base.py",
                "metrics.py",
            )
        ),
        root / "data" / "identity.py",
        root / "data" / "yolo_labels.py",
    ]
    return {
        "source_sha256": {
            p.relative_to(root).as_posix(): file_sha256(p) for p in paths
        },
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
    }


def _identity(source, manifest, config, ratios, name, policy, seed, implementation):
    settings = asdict(config)
    settings["balance"].pop("seeds")
    return {
        "protocol_version": VERSION,
        "dataset_id": source.summary["dataset_id"],
        "population_dataset_id": dataset_id_from_manifest(manifest),
        "audit_id": source.summary["audit_id"],
        "source_sha256": source.signatures,
        "population": name,
        "policy": policy,
        "seed": seed,
        "configuration": settings,
        "target_ratios": ratios.tolist(),
        "implementation": implementation,
    }


def _relation_metrics(records, relations):
    membership = records.set_index("frame_id").new_split.to_dict()
    results = {}
    for state in sorted(STATES):
        all_edges = [e for e in relations if e["state"] == state]
        edges = [
            e
            for e in all_edges
            if e["left_frame_id"] in membership and e["right_frame_id"] in membership
        ]
        pairs = {
            tuple(sorted((e["left_frame_id"], e["right_frame_id"]))) for e in edges
        }
        crossing = {
            pair for pair in pairs if membership[pair[0]] != membership[pair[1]]
        }
        results[state] = {
            "evidence_records": len(edges),
            "excluded_evidence_records": len(all_edges) - len(edges),
            "occurrence_pairs": len(pairs),
            "cross_split_occurrence_pairs": len(crossing),
            "cross_split_fraction": len(crossing) / len(pairs) if pairs else None,
            "content_pairs": len(
                {
                    tuple(sorted((e["left_content_id"], e["right_content_id"])))
                    for e in edges
                }
            ),
            "cross_split_content_pairs": len(
                {
                    tuple(sorted((e["left_content_id"], e["right_content_id"])))
                    for e in edges
                    if membership[e["left_frame_id"]] != membership[e["right_frame_id"]]
                }
            ),
        }
    return results


def evaluate_candidate(
    source,
    manifest,
    stats,
    ratios,
    groups,
    labels,
    selection,
    contents,
    records,
    policy,
    config,
):
    """Reuse invariant/balance checks; report measured evidence, not independence."""
    quality = verify_assignments(
        contents,
        records,
        manifest,
        "cluster_aware" if policy == "S3" else "random_content",
        labels,
    )
    quality["mandatory_group_fracture_count"] = int(
        (contents.groupby("group_id").new_split.nunique() > 1).sum()
    )
    _require(quality["mandatory_group_fracture_count"] == 0, "Mandatory group fracture")
    # The occurrence table is the complete input manifest plus one assignment.
    expected = manifest.sort_values("frame_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(
        records.drop(columns="new_split").reset_index(drop=True), expected
    )
    balance, classes = balance_metrics(records, stats, ratios)
    if config.max_relative_record_deviation is not None:
        _require(
            balance["max_relative_record_deviation"]
            <= config.max_relative_record_deviation,
            "Candidate fails configured balance publication ceiling",
        )
    units = aggregate_groups(stats, groups)
    mapping = records.set_index("frame_id").new_split
    imposed = set(selection["imposed_relation_ids"])
    _require(
        all(
            mapping[e["left_frame_id"]] == mapping[e["right_frame_id"]]
            for e in source.relations
            if e["relation_id"] in imposed
        ),
        "Imposed relation fracture",
    )
    cohort = []
    for name in ("historical", "adendo"):
        for split in ("train", "val", "test"):
            subset = records.loc[
                (records.cohort == name) & (records.new_split == split)
            ]
            cohort.append(
                {
                    "cohort": name,
                    "split": split,
                    "occurrences": len(subset),
                    "contents": subset.content_id.nunique(),
                }
            )
    metrics = {
        "quality": quality,
        "balance": balance,
        "cohort_distribution": cohort,
        "relations": _relation_metrics(records, source.relations),
        "selection": selection,
        "atomic_group_count": len(units),
        "group_sizes_occurrences": {
            str(k): int(v)
            for k, v in units.record_count.value_counts().sort_index().items()
        },
        "group_sizes_contents": {
            str(k): int(v)
            for k, v in units.content_count.value_counts().sort_index().items()
        },
        "cosine_similarity_evaluated": False,
        "temporal_dependence_evaluated": False,
        "absence_of_leakage_demonstrated": False,
    }
    return metrics, classes, units


def _historical_reference(source, config):
    manifest, stats, ratios = population(source, "historical", config)
    records = manifest.assign(new_split=manifest.original_split)
    balance, table = balance_metrics(records, stats, ratios)
    return {
        "description": "Descriptive original memberships; unchanged, non-atomic reference",
        "occurrences": len(records),
        "contents": records.content_id.nunique(),
        "exact_duplicate_cross_split_count": int(
            (records.groupby("content_id").new_split.nunique() > 1).sum()
        ),
        "balance": balance,
        "class_balance": table.to_dict(orient="records"),
        "relations": _relation_metrics(records, source.relations),
    }


def _verify_solver(solver, records, statistics, units, ratios, seed, config):
    """Check the reported objective against assigned occurrence-level counts."""
    _require(
        solver["seed"] == seed and solver["method"] == "scipy.optimize.milp/HiGHS",
        "Solver method/seed mismatch",
    )
    _require(solver["atomic_group_count"] == len(units), "Solver group count mismatch")
    totals = statistics[list(BALANCE_COLUMNS)].sum().to_numpy(dtype=float)
    targets = ratios[:, None] * totals[None, :]
    actual = records[["frame_id", "new_split"]].merge(
        statistics, on="frame_id", validate="one_to_one"
    )
    counts = (
        actual.groupby("new_split")[list(BALANCE_COLUMNS)]
        .sum()
        .reindex(["train", "val", "test"])
        .to_numpy()
    )
    weights = np.array(
        [
            config.balance.record_weight,
            *([config.balance.class_weight / 5] * 5),
            config.balance.empty_weight,
        ]
    )
    objective = float(
        (abs(counts - targets) * weights[None, :] / (3 * np.maximum(targets, 1))).sum()
    )
    _require(
        np.isfinite(solver["objective"])
        and np.isclose(solver["objective"], objective, atol=1e-6, rtol=1e-6),
        "Solver objective disagrees with memberships",
    )
    _require(
        np.isfinite(solver["dual_bound"])
        and solver["dual_bound"] <= objective + 1e-6
        and np.isfinite(solver["mip_gap"])
        and solver["mip_gap"] >= 0,
        "Invalid solver bound/gap",
    )


def _row(meta, metrics):
    identity, balance = meta["identity_payload"], metrics["balance"]
    return {
        "split_id": meta["split_id"],
        "population": identity["population"],
        "policy": identity["policy"],
        "seed": identity["seed"],
        "occurrences": metrics["quality"]["record_count"],
        "contents": metrics["quality"]["content_count"],
        "atomic_groups": metrics["atomic_group_count"],
        "exact_cross_split": metrics["quality"]["exact_duplicate_cross_split_count"],
        "mandatory_fractures": metrics["quality"]["mandatory_group_fracture_count"],
        "candidate_cross_pairs": metrics["relations"]["visual_candidate"][
            "cross_split_occurrence_pairs"
        ],
        "candidate_available_pairs": metrics["relations"]["visual_candidate"][
            "occurrence_pairs"
        ],
        "max_relative_record_deviation": balance["max_relative_record_deviation"],
        "class_deviation_pp": balance["mean_absolute_class_distribution_deviation_pp"],
        "all_classes_covered": balance["all_classes_covered"],
        "solver_status": meta["solver"].get("status"),
        "solver_gap": meta["solver"].get("mip_gap"),
        "solver_optimal_within_tolerance": meta["solver"].get(
            "optimal_within_tolerance"
        ),
        **{f"{r['split']}_occurrences": r["records"] for r in balance["sizes"]},
    }


def build_labeled_candidates(
    audit: Path,
    output: Path,
    config: LabeledConfig,
    *,
    populations=("historical", "expanded"),
    policies=("S1", "S2", "S3"),
    expected_dataset_id=None,
    expected_audit_id=None,
    source_root: Path | None = None,
    evidence: Path | None = None,
) -> dict:
    """Create a new suite; a final receipt exists only after semantic verification.

    Failed/incomplete output is preserved for inspection, never reused. This
    namespace has no dependency on historical experiment plans or detector runs.
    """
    repo = Path(__file__).resolve().parents[3]
    target, original = output.resolve(), audit.resolve()
    _require(
        not output.exists()
        and not target.is_relative_to(repo)
        and not target.is_relative_to(original)
        and not original.is_relative_to(target),
        "Output must be fresh, outside the repository and separate from the audit",
    )
    _require(
        bool(populations)
        and len(set(populations)) == len(populations)
        and set(populations) <= {"historical", "expanded"},
        "Invalid populations",
    )
    _require(
        bool(policies)
        and len(set(policies)) == len(policies)
        and set(policies) <= {"S1", "S2", "S3"},
        "Invalid policies",
    )
    source = load_labeled_audit(
        audit,
        expected_dataset_id=expected_dataset_id,
        expected_audit_id=expected_audit_id,
    )
    source_check = (
        verify_original_sources(source, source_root, evidence)
        if source_root
        else {"original_archives_rehashed": False}
    )
    # Validate all population/group configurations before writing or solving.
    prepared = {}
    for name in populations:
        manifest, stats, ratios = population(source, name, config)
        for policy in policies:
            groups, labels, selection = sensitivity_groups(
                manifest, source.relations, policy, config
            )
            units = aggregate_groups(stats, groups)
            _require(len(units) >= 3, "At least three atomic groups required")
            prepared[name, policy] = (
                manifest,
                stats,
                ratios,
                groups,
                labels,
                selection,
                units,
            )
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "source_audit"
    snapshot.mkdir()
    for name in sorted(source.signatures):
        shutil.copyfile(audit / name, snapshot / name)
    _require(
        load_labeled_audit(snapshot).signatures == source.signatures,
        "Source changed while snapshotting",
    )
    write_json(output / "configuration.json", asdict(config))
    write_json(
        output / "historical_reference.json", _historical_reference(source, config)
    )
    implementation = _implementation()
    provenance = execution_provenance()
    provenance["source_sha256"] = implementation["source_sha256"]
    write_json(
        output / "provenance.json",
        {**provenance, "original_source_verification": source_check},
    )
    rows, runs = [], []
    for (name, policy), (
        manifest,
        stats,
        ratios,
        groups,
        labels,
        selection,
        units,
    ) in prepared.items():
        for seed in config.balance.seeds:
            assignment, solver = (
                random_assignment(units, ratios, seed)
                if policy == "S1"
                else milp_assignment(units, ratios, seed, config.balance)
            )
            solver["configured_limits"] = (
                {
                    "node_limit": config.balance.node_limit,
                    "mip_rel_gap": config.balance.mip_rel_gap,
                }
                if policy != "S1"
                else None
            )
            contents, propagated = propagate(stats, groups, assignment)
            records = (
                manifest.merge(
                    propagated[["frame_id", "new_split"]],
                    on="frame_id",
                    validate="one_to_one",
                )
                .sort_values("frame_id")
                .reset_index(drop=True)
            )
            metrics, classes, units = evaluate_candidate(
                source,
                manifest,
                stats,
                ratios,
                groups,
                labels,
                selection,
                contents,
                records,
                policy,
                config,
            )
            identity = _identity(
                source, manifest, config, ratios, name, policy, seed, implementation
            )
            split_id = stable_id(identity)
            directory = output / "runs" / split_id
            directory.mkdir(parents=True, exist_ok=False)
            tables = {
                "content_memberships.parquet": contents,
                "occurrence_memberships.parquet": records,
                "atomic_groups.parquet": groups,
                "group_balance.parquet": units,
                "class_balance.parquet": classes,
            }
            for filename, table in tables.items():
                table.to_parquet(directory / filename, index=False)
            write_json(directory / "metrics.json", metrics)
            meta = {
                "split_id": split_id,
                "identity_payload": identity,
                "solver": solver,
                "membership_id": stable_id(
                    {
                        "membership": records[
                            ["frame_id", "content_id", "new_split"]
                        ].values.tolist()
                    }
                ),
            }
            write_json(directory / "metadata.json", meta)
            rows.append(_row(meta, metrics))
            runs.append(
                {
                    "split_id": split_id,
                    "output_sha256": {
                        f: file_sha256(directory / f) for f in sorted(RUN_FILES)
                    },
                }
            )
            print(
                f"{name}/{policy} seed={seed}: atomicity verified; solver={solver.get('status', 'random')}",
                flush=True,
            )
    pd.DataFrame(rows).to_csv(output / "comparison.csv", index=False)
    receipt = {
        "schema_version": VERSION,
        "dataset_id": source.summary["dataset_id"],
        "audit_id": source.summary["audit_id"],
        "populations": list(populations),
        "policies": list(policies),
        "seeds": list(config.balance.seeds),
        "candidate_count": len(runs),
        "runs": runs,
        "source_sha256": source.signatures,
        "output_sha256": {
            name: file_sha256(output / name)
            for name in (
                "configuration.json",
                "historical_reference.json",
                "provenance.json",
                "comparison.csv",
            )
        },
    }
    # Verify from persisted bytes before publishing the completion receipt.
    verify_labeled_candidates(output, receipt=receipt)
    _require(
        all(
            file_sha256(audit / name) == digest
            for name, digest in source.signatures.items()
        ),
        "Audit changed during construction",
    )
    write_json(output / "summary.json", receipt)
    return receipt


def verify_labeled_candidates(
    directory: Path,
    *,
    audit: Path | None = None,
    replay: bool = False,
    receipt: dict | None = None,
) -> dict:
    """Recompute lineage, groups, all metrics and identities; optionally replay solver.

    Archive hash evidence in provenance describes build-time validation. Standalone
    verification binds the complete audit snapshot; it does not rehash ZIPs.
    """
    receipt = receipt if receipt is not None else read_json(directory / "summary.json")
    _require(receipt["schema_version"] == VERSION, "Unsupported candidate suite")
    _require(
        set(receipt["output_sha256"])
        == {
            "configuration.json",
            "historical_reference.json",
            "provenance.json",
            "comparison.csv",
        },
        "Incomplete suite receipt",
    )
    for name, digest in receipt["output_sha256"].items():
        _require(
            file_sha256(directory / name) == digest, f"Suite checksum mismatch: {name}"
        )
    source = load_labeled_audit(
        directory / "source_audit",
        expected_dataset_id=receipt["dataset_id"],
        expected_audit_id=receipt["audit_id"],
    )
    _require(
        source.signatures == receipt["source_sha256"], "Snapshot signatures mismatch"
    )
    if audit is not None:
        _require(
            load_labeled_audit(audit).signatures == source.signatures,
            "External audit differs from snapshot",
        )
    config = LabeledConfig(**read_json(directory / "configuration.json"))
    _require(
        receipt["seeds"] == list(config.balance.seeds), "Seed configuration mismatch"
    )
    _require(
        read_json(directory / "historical_reference.json")
        == _historical_reference(source, config),
        "Historical reference changed",
    )
    expected = {
        (n, p, s)
        for n in receipt["populations"]
        for p in receipt["policies"]
        for s in config.balance.seeds
    }
    _require(
        receipt["candidate_count"] == len(receipt["runs"]) == len(expected),
        "Candidate coverage mismatch",
    )
    observed, rows = set(), []
    for run in receipt["runs"]:
        split_id = run["split_id"]
        _require(
            isinstance(split_id, str)
            and len(split_id) == 16
            and all(c in "0123456789abcdef" for c in split_id),
            "Unsafe split identity",
        )
        path = directory / "runs" / split_id
        _require(set(run["output_sha256"]) == RUN_FILES, "Incomplete run receipt")
        for filename, digest in run["output_sha256"].items():
            _require(
                file_sha256(path / filename) == digest,
                f"Run checksum mismatch: {filename}",
            )
        meta = read_json(path / "metadata.json")
        identity = meta["identity_payload"]
        name, policy, seed = (
            identity["population"],
            identity["policy"],
            identity["seed"],
        )
        _require(
            (name, policy, seed) in expected and (name, policy, seed) not in observed,
            "Unexpected or duplicate run",
        )
        observed.add((name, policy, seed))
        manifest, stats, ratios = population(source, name, config)
        payload = _identity(
            source,
            manifest,
            config,
            ratios,
            name,
            policy,
            seed,
            identity["implementation"],
        )
        _require(
            json.loads(json.dumps(payload)) == identity
            and split_id == meta["split_id"] == stable_id(payload),
            "Split identity mismatch",
        )
        groups, labels, selection = sensitivity_groups(
            manifest, source.relations, policy, config
        )
        pd.testing.assert_frame_equal(
            groups, pd.read_parquet(path / "atomic_groups.parquet")
        )
        contents = pd.read_parquet(path / "content_memberships.parquet")
        records = pd.read_parquet(path / "occurrence_memberships.parquet")
        # Parquet list columns return ndarrays; normalize through the source schema.
        records["label_ids"] = records.label_ids.map(list)
        metrics, classes, units = evaluate_candidate(
            source,
            manifest,
            stats,
            ratios,
            groups,
            labels,
            selection,
            contents,
            records,
            policy,
            config,
        )
        _require(
            metrics == read_json(path / "metrics.json"), "Recomputed metrics differ"
        )
        pd.testing.assert_frame_equal(
            classes, pd.read_parquet(path / "class_balance.parquet")
        )
        pd.testing.assert_frame_equal(
            units, pd.read_parquet(path / "group_balance.parquet")
        )
        _require(
            meta["membership_id"]
            == stable_id(
                {
                    "membership": records[
                        ["frame_id", "content_id", "new_split"]
                    ].values.tolist()
                }
            ),
            "Membership identity mismatch",
        )
        if policy != "S1":
            solver = meta["solver"]
            _verify_solver(solver, records, stats, units, ratios, seed, config)
            _require(
                solver["primal_feasibility_verified"] is True
                and solver["unique_optimum_proven"] is False,
                "Solver feasibility claim invalid",
            )
            _require(
                solver["configured_limits"]
                == {
                    "node_limit": config.balance.node_limit,
                    "mip_rel_gap": config.balance.mip_rel_gap,
                },
                "Solver limits mismatch",
            )
            _require(
                not solver["optimal_within_tolerance"] or solver["status"] == 0,
                "Unsupported solver optimality claim",
            )
        if replay:
            _require(
                identity["implementation"] == _implementation(),
                "Replay requires the recorded implementation and versions",
            )
            assignment, _ = (
                random_assignment(units, ratios, seed)
                if policy == "S1"
                else milp_assignment(units, ratios, seed, config.balance)
            )
            replay_contents, _ = propagate(stats, groups, assignment)
            pd.testing.assert_frame_equal(replay_contents, contents)
        rows.append(_row(meta, metrics))
    _require(observed == expected, "Missing candidate runs")
    expected_table = pd.DataFrame(rows).astype(object)
    actual_table = pd.read_csv(directory / "comparison.csv").astype(object)
    # CSV has one missing-value spelling for null solver fields in random runs.
    pd.testing.assert_frame_equal(
        expected_table.where(expected_table.notna(), None),
        actual_table.where(actual_table.notna(), None),
        check_dtype=False,
        atol=1e-12,
        rtol=1e-12,
    )
    return {
        "quality_valid": True,
        "candidate_count": len(rows),
        "all_metrics_recomputed": True,
        "external_audit_verified": audit is not None,
        "assignment_replayed": replay,
    }
