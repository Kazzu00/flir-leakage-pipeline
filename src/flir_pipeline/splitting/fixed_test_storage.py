"""Immutable provisional suites, source binding, failure records and solver replay."""

from __future__ import annotations

import hashlib
import io
import shutil
import zipfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

from flir_pipeline.detection.annotation_review import validate_decisions
from flir_pipeline.detection.common_test import CommonTestConfig, derive_proposal
from flir_pipeline.detection.common_test_storage import signatures, verify_common_test
from flir_pipeline.similarity.storage import (
    execution_provenance,
    file_sha256,
    read_json,
    stable_id,
    write_json,
)
from flir_pipeline.splitting.construction import AssignmentFailure
from flir_pipeline.splitting.fixed_test import (
    POLICIES,
    POPULATIONS,
    VERSION,
    FixedTestConfig,
    construct_arm,
    inspect_arm,
    membership_id,
    prepared_arm,
    suite_summary,
)
from flir_pipeline.splitting.labeled import _require, load_labeled_audit
from flir_pipeline.splitting.labeled_storage import verify_labeled_candidates


def _implementation():
    root = Path(__file__).parents[1]
    names = (
        "splitting/fixed_test.py",
        "splitting/fixed_test_storage.py",
        "splitting/construction.py",
        "splitting/base.py",
        "splitting/metrics.py",
        "splitting/labeled.py",
        "splitting/labeled_storage.py",
        "detection/common_test.py",
        "detection/common_test_storage.py",
    )
    return {
        "source_sha256": {n: file_sha256(root / n) for n in names},
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
    }


def review_transport(review):
    """Validate optional redundant transport without rewriting the source receipt.

    Every unique cases.zip member must equal a receipted case file. This extra
    archive is separately bound, never treated as new annotation evidence.
    """
    receipt = read_json(review / "receipt.json")
    actual = signatures(review)
    actual.pop("receipt.json")
    declared = receipt["artifacts"]
    _require(
        all(actual.get(n) == digest for n, digest in declared.items()),
        "Annotation review checksum mismatch",
    )
    extras = set(actual) - set(declared)
    _require(extras <= {"cases.zip"}, "Unknown unreceipted review supplement")
    if not extras:
        return {}
    expected = {n: h for n, h in declared.items() if n.startswith("cases/")}
    observed = {}
    with zipfile.ZipFile(review / "cases.zip") as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            name = member.filename.replace("\\", "/")
            if not name.startswith("cases/"):
                name = "cases/" + name
            _require(
                name in expected
                and name not in observed
                and (member.external_attr >> 16) & 0o170000 != 0o120000,
                "Unsafe, duplicate or unexpected transport member",
            )
            observed[name] = hashlib.sha256(archive.read(member)).hexdigest()
    _require(
        observed == expected, "Transport archive differs from receipted case bytes"
    )
    return {
        "cases.zip": {
            "sha256": actual["cases.zip"],
            "role": "verified_redundant_transport",
            "case_files_verified": len(observed),
            "annotation_evidence_added": False,
        }
    }


def load_sources(audit, previous, proposal, review):
    """Validate immutable evidence without consuming detector results for assignment.

    Legacy splits are verified for provenance only, never used as new assignments.
    Review receipts bind source bytes; this stage does not rerender image evidence
    or rehash original ZIPs. Labels are reconstructed by the existing audit loader.
    """
    source = load_labeled_audit(audit)
    verify_labeled_candidates(previous, audit=audit)
    verify_common_test(proposal)
    _require(
        signatures(audit) == signatures(proposal / "sources/audit")
        and signatures(previous) == signatures(proposal / "sources/splits"),
        "Proposal source binding differs",
    )
    stored = read_json(proposal / "proposal.json")
    config = CommonTestConfig(**stored["identity"]["configuration"])
    _require(
        config.option == "internal" and stored["readiness"]["state"] == "PROPOSED",
        "Expected internal PROPOSED test",
    )
    plan, tables = derive_proposal(source, config)
    plan["plan_id"] = stored["plan_id"]
    # Review is evidence for pending adjudication, never authorization.
    receipt = read_json(review / "receipt.json")
    review_transport(review)
    evidence = read_json(review / "manifest.json")
    identity = evidence["identity"]
    _require(
        evidence["schema_version"] == "protected_annotation_review_v1"
        and stable_id(identity) == evidence["review_id"] == receipt["review_id"]
        and identity["plan_id"] == stored["plan_id"]
        and identity["test_id"] == plan["test_id"]
        and identity["sources"]["audit"] == signatures(audit)
        and identity["sources"]["proposal"] == signatures(proposal),
        "Annotation review identity/source mismatch",
    )
    _require(
        evidence["state"] == "PENDING_HUMAN_REVIEW"
        and evidence["protocol_state"] == "PROPOSED"
        and evidence["ready_for_training"] is False
        and evidence["adjudicated_labels_applied"] is False,
        "Annotation evidence cannot authorize training",
    )
    decisions = validate_decisions(
        read_json(review / "decisions.template.json"), evidence
    )
    _require(
        decisions["pending"]
        == len(evidence["cases"])
        == plan["protected_annotation_conflict_groups"],
        "Expected pending protected conflicts",
    )
    frames = source.manifest.set_index("frame_id").to_dict(orient="index")
    test_ids = set(tables["test_records.parquet"].frame_id)
    quarantine = set(tables["quarantine_records.parquet"].frame_id)
    conflict_ids = set(plan["protected_annotation_conflict_relation_ids"])
    expected_contents = {
        e["left_content_id"]
        for e in source.relations
        if e["relation_id"] in conflict_ids
    }
    _require(
        {c["content_id"] for c in evidence["cases"]} == expected_contents,
        "Review conflict scope changed",
    )
    for case in evidence["cases"]:
        for key, pool in (("A", test_ids), ("B", quarantine)):
            occurrence = case[key]["occurrence"]
            frame = occurrence["frame_id"]
            _require(
                frame in pool
                and frames[frame]["content_id"] == case["content_id"]
                and frames[frame]["label_sha256"]
                == occurrence["label_sha256"]
                == case[key]["annotation"]["label_sha256"],
                "Review annotations/roles changed",
            )
    return source, plan, tables


def _arms(plan, config):
    arms = [a for a in plan["arms"] if a["split_seed"] in config.seeds]
    _require(
        {(a["population"], a["policy"], a["split_seed"]) for a in arms}
        == {
            (p, s, seed) for p in POPULATIONS for s in POLICIES for seed in config.seeds
        },
        "Incomplete arm matrix",
    )
    return arms


def _identity(binding, config, implementation, arm):
    return {
        "schema_version": VERSION,
        "sources": binding,
        "settings": config.mathematical_settings(),
        "implementation": implementation,
        "arm": arm,
    }


def _entry(arm, meta, metrics):
    return {
        "arm": arm["arm"],
        "population": arm["population"],
        "policy": arm["policy"],
        "seed": arm["split_seed"],
        "candidate_id": meta["candidate_id"],
        "membership_id": meta["membership_id"],
        "state": metrics["state"],
        "max_relative_record_deviation": metrics.get("balance", {}).get(
            "max_relative_record_deviation"
        ),
        "class_deviation_pp": metrics.get("balance", {}).get(
            "mean_absolute_class_distribution_deviation_pp"
        ),
        "solver_status": meta["solver"].get("status"),
        "solver_gap": meta["solver"].get("mip_gap"),
        "solver_objective": meta["solver"].get("objective"),
        "solver_primal_feasible": meta["solver"].get("primal_feasibility_verified"),
        "solver_optimal_within_tolerance": meta["solver"].get(
            "optimal_within_tolerance"
        ),
    }


def _snapshot_binding(root):
    return {
        name: signatures(root / name)
        for name in ("audit", "previous", "proposal", "review")
    }


def build_fixed_test(
    audit: Path,
    previous: Path,
    proposal: Path,
    review: Path,
    output: Path,
    config: FixedTestConfig,
    *,
    pilot: Path | None = None,
):
    roots = {
        "audit": audit,
        "previous": previous,
        "proposal": proposal,
        "review": review,
    }
    repo, target = Path(__file__).resolve().parents[3], output.resolve()
    _require(
        not output.exists()
        and not target.is_relative_to(repo)
        and all(
            not target.is_relative_to(p.resolve())
            and not p.resolve().is_relative_to(target)
            for p in [*roots.values(), *([pilot] if pilot else [])]
        ),
        "Fresh external output disjoint from sources required",
    )
    binding = {n: signatures(p) for n, p in roots.items()}
    source, plan, tables = load_sources(**roots)
    implementation = _implementation()
    pilot_binding = None
    if config.seeds != (0,):
        _require(pilot is not None, "Full matrix requires a verified pilot")
        checked = verify_fixed_test(pilot, replay=True)
        pilot_meta = read_json(pilot / "suite.json")
        _require(
            checked["failed_configurations"] == 0
            and checked["configurations"] == 6
            and pilot_meta["sources"] == binding
            and FixedTestConfig(**pilot_meta["configuration"]).mathematical_settings()
            == config.mathematical_settings()
            and pilot_meta["implementation"] == implementation,
            "Pilot is not eligible for this full matrix",
        )
        pilot_binding = signatures(pilot)
    else:
        _require(pilot is None, "A pilot cannot consume another pilot")
    output.mkdir(parents=True)
    for name, path in roots.items():
        shutil.copytree(path, output / "sources" / name)
    if pilot:
        shutil.copytree(pilot, output / "pilot")
    _require(
        _snapshot_binding(output / "sources") == binding,
        "Sources changed while snapshotting",
    )
    write_json(
        output / "suite.json",
        {
            "schema_version": VERSION,
            "configuration": asdict(config),
            "implementation": implementation,
            "sources": binding,
            "pilot_sha256": pilot_binding,
            "review_transport": review_transport(review),
            "plan_id": plan["plan_id"],
            "test_id": plan["test_id"],
            "ratio_derivation": {
                "rule": "historical train/(train+val), val/(train+val); applied to development only",
                "historical_counts": source.manifest.loc[
                    source.manifest.cohort == "historical"
                ]
                .original_split.value_counts()
                .to_dict(),
                "ratios": plan["pools"]["historical"]["train_val_target_ratios"],
            },
        },
    )
    write_json(
        output / "provenance.json",
        {
            **execution_provenance(),
            "source_scope": "verified immutable snapshots; original ZIPs not rehashed in this stage",
            "annotation_adjudication_applied": False,
        },
    )
    entries = []
    for arm in _arms(plan, config):
        identity = _identity(binding, config, implementation, arm)
        meta = {
            "candidate_id": stable_id(identity),
            "identity": identity,
            "membership_id": None,
        }
        path = output / "runs" / arm["arm"]
        path.mkdir(parents=True)
        _, _, _, groups, units, _ = prepared_arm(source, plan, tables, arm)
        groups.to_parquet(path / "atomic_groups.parquet", index=False)
        units.to_parquet(path / "group_balance.parquet", index=False)
        try:
            rows, contents, solver = construct_arm(source, plan, tables, arm, config)
        except AssignmentFailure as exc:
            meta["solver"] = exc.evidence
            metrics = {
                "state": "FAILED_CONSTRUCTION",
                "reason": str(exc),
                "ready_for_training": False,
            }
        else:
            metrics, classes = inspect_arm(
                source, plan, tables, arm, rows, solver, config
            )
            meta.update(solver=solver, membership_id=membership_id(rows))
            rows.to_parquet(path / "occurrence_memberships.parquet", index=False)
            contents.to_parquet(
                path / "development_content_memberships.parquet", index=False
            )
            classes.to_parquet(path / "class_balance.parquet", index=False)
        write_json(path / "metadata.json", meta)
        write_json(path / "metrics.json", metrics)
        entries.append(_entry(arm, meta, metrics))
        print(
            f"{arm['arm']}: {metrics['state']}; gap={meta['solver'].get('mip_gap')}",
            flush=True,
        )
    summary = suite_summary(entries)
    write_json(output / "summary.json", summary)
    pd.DataFrame(entries).to_csv(output / "comparison.csv", index=False)
    _require(
        binding == {n: signatures(p) for n, p in roots.items()},
        "Sources changed during construction",
    )
    receipt = {"schema_version": VERSION, "artifacts": signatures(output)}
    verify_fixed_test(output, receipt=receipt)
    write_json(output / "receipt.json", receipt)
    return summary


def verify_fixed_test(directory: Path, *, replay=False, receipt=None):
    receipt = receipt if receipt is not None else read_json(directory / "receipt.json")
    actual = signatures(directory)
    actual.pop("receipt.json", None)
    _require(
        receipt["schema_version"] == VERSION and actual == receipt["artifacts"],
        "Fixed-test checksum/file inventory mismatch",
    )
    meta = read_json(directory / "suite.json")
    config = FixedTestConfig(**meta["configuration"])
    binding = _snapshot_binding(directory / "sources")
    _require(
        binding == meta["sources"] and meta["schema_version"] == VERSION,
        "Fixed-test source binding mismatch",
    )
    source, plan, tables = load_sources(
        **{n: directory / "sources" / n for n in binding}
    )
    _require(
        meta["review_transport"] == review_transport(directory / "sources/review"),
        "Review transport declaration differs",
    )
    _require(
        meta["plan_id"] == plan["plan_id"] and meta["test_id"] == plan["test_id"],
        "Fixed test identity changed",
    )
    history = (
        source.manifest.loc[source.manifest.cohort == "historical"]
        .original_split.value_counts()
        .to_dict()
    )
    _require(
        meta["ratio_derivation"]
        == {
            "rule": "historical train/(train+val), val/(train+val); applied to development only",
            "historical_counts": history,
            "ratios": plan["pools"]["historical"]["train_val_target_ratios"],
        },
        "Ratio derivation changed",
    )
    if replay:
        _require(
            meta["implementation"] == _implementation(),
            "Replay requires recorded implementation/environment",
        )
    if config.seeds != (0,):
        _require(
            meta["pilot_sha256"] == signatures(directory / "pilot"),
            "Pilot receipt changed",
        )
        pilot_check = verify_fixed_test(directory / "pilot", replay=False)
        pilot_meta = read_json(directory / "pilot/suite.json")
        _require(
            pilot_check["configurations"] == 6
            and pilot_check["failed_configurations"] == 0
            and pilot_meta["sources"] == binding
            and pilot_meta["implementation"] == meta["implementation"]
            and FixedTestConfig(**pilot_meta["configuration"]).mathematical_settings()
            == config.mathematical_settings(),
            "Invalid pilot gate",
        )
    else:
        _require(
            meta["pilot_sha256"] is None and not (directory / "pilot").exists(),
            "Unexpected pilot dependency",
        )
    entries = []
    arms = _arms(plan, config)
    _require(
        {p.name for p in (directory / "runs").iterdir()} == {a["arm"] for a in arms},
        "Missing or extra arms",
    )
    for arm in arms:
        path = directory / "runs" / arm["arm"]
        run = read_json(path / "metadata.json")
        identity = _identity(binding, config, meta["implementation"], arm)
        _require(
            run["identity"] == identity and run["candidate_id"] == stable_id(identity),
            "Candidate identity mismatch",
        )
        _, _, _, groups, units, _ = prepared_arm(source, plan, tables, arm)
        pd.testing.assert_frame_equal(
            pd.read_parquet(path / "atomic_groups.parquet"), groups
        )
        pd.testing.assert_frame_equal(
            pd.read_parquet(path / "group_balance.parquet"), units
        )
        metrics = read_json(path / "metrics.json")
        files = {
            "metadata.json",
            "metrics.json",
            "atomic_groups.parquet",
            "group_balance.parquet",
        }
        if metrics["state"] == "FAILED_CONSTRUCTION":
            _require(run["membership_id"] is None, "Failed solver has a membership")
            # Failure is a scientific result too: always replay it, even in basic verification.
            try:
                construct_arm(source, plan, tables, arm, config)
            except AssignmentFailure as exc:
                _require(
                    metrics
                    == {
                        "state": "FAILED_CONSTRUCTION",
                        "reason": str(exc),
                        "ready_for_training": False,
                    }
                    and run["solver"] == exc.evidence,
                    "Failure evidence differs on replay",
                )
            else:
                raise ValueError("Recorded failure no longer reproducible")
        else:
            files |= {
                "occurrence_memberships.parquet",
                "development_content_memberships.parquet",
                "class_balance.parquet",
            }
            rows = pd.read_parquet(path / "occurrence_memberships.parquet")
            rows["label_ids"] = rows.label_ids.map(list)
            recomputed, classes = inspect_arm(
                source, plan, tables, arm, rows, run["solver"], config
            )
            _require(
                recomputed == metrics and membership_id(rows) == run["membership_id"],
                "Metrics/membership identity mismatch",
            )
            pd.testing.assert_frame_equal(
                classes, pd.read_parquet(path / "class_balance.parquet")
            )
            contents = pd.read_parquet(path / "development_content_memberships.parquet")
            expected = groups.copy()
            expected["new_split"] = expected.content_id.map(
                rows.loc[rows.new_split.isin(["train", "val"])]
                .drop_duplicates("content_id")
                .set_index("content_id")
                .new_split
            )
            pd.testing.assert_frame_equal(
                contents, expected.sort_values("content_id").reset_index(drop=True)
            )
            if replay:
                replay_rows, replay_contents, solver = construct_arm(
                    source, plan, tables, arm, config
                )
                pd.testing.assert_frame_equal(replay_rows, rows)
                pd.testing.assert_frame_equal(replay_contents, contents)
                _require(solver == run["solver"], "Solver replay differs")
        _require(
            {p.name for p in path.iterdir()} == files, "Run file inventory differs"
        )
        entries.append(_entry(arm, run, metrics))
    expected_summary = suite_summary(entries)
    _require(
        expected_summary == read_json(directory / "summary.json"),
        "Provisional state, failures or aliases changed",
    )
    comparison = pd.read_csv(directory / "comparison.csv")
    pd.testing.assert_frame_equal(
        comparison,
        pd.read_csv(io.StringIO(pd.DataFrame(entries).to_csv(index=False))),
        check_dtype=False,
    )
    return {
        **expected_summary,
        "source_verification": "snapshots_and_identities",
        "replayed": replay,
    }
