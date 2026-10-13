"""Verified source bundles and immutable pre-training proposals, separate from freezes."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from flir_pipeline.detection.common_test import (
    VERSION,
    CommonTestConfig,
    derive_proposal,
)
from flir_pipeline.detection.export_contract import DetectionExport
from flir_pipeline.detection.protocol import load_config
from flir_pipeline.similarity.storage import (
    execution_provenance,
    file_sha256,
    read_json,
    stable_id,
    write_json,
)
from flir_pipeline.splitting.labeled import load_labeled_audit, verify_original_sources
from flir_pipeline.splitting.labeled_storage import verify_labeled_candidates


def _safe(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if (
        not isinstance(name, str)
        or "\\" in name
        or ":" in name
        or not path.is_relative_to(root.resolve())
        or path == root.resolve()
    ):
        raise ValueError("Unsafe receipt member")
    return path


def signatures(root: Path) -> dict:
    return {
        p.relative_to(root).as_posix(): file_sha256(p)
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def verify_receipt(root: Path, receipt: str, key: str) -> dict:
    payload = read_json(root / receipt)
    if not isinstance(payload[key], dict) or not payload[key]:
        raise ValueError("Empty or invalid source receipt")
    for name, digest in payload[key].items():
        if file_sha256(_safe(root, name)) != digest:
            raise ValueError(f"Source checksum mismatch: {receipt}/{name}")
    return payload


def audit_export(root: Path) -> dict:
    manifest = verify_receipt(root, "manifest.json", "file_sha256")
    names = set(DetectionExport.model_fields) - {"manifest"}
    if set(manifest["file_sha256"]) != {f"{n}.json" for n in names} | {
        "schema/detection-export-v1.schema.json"
    }:
        raise ValueError("Incomplete historical export receipt")
    bundle = DetectionExport.model_validate(
        {"manifest": manifest, **{n: read_json(root / f"{n}.json") for n in names}}
    )
    return {
        "aggregate_export_valid": True,
        "exported_run_count": len(bundle.runs),
        "exported_split_count": len(bundle.splits),
        "exported_plan_id": manifest["plan_id"],
        "producer_reports_source_verification": manifest[
            "generated_from_verified_artifacts"
        ],
        "original_hypatia_runs_reverified": False,
        "per_image_predictions_available_from_export": False,
        "granular_fp_fn_localization_diagnosis": "PENDING_ORIGINAL_PREDICTIONS_AND_ANNOTATIONS",
        "scope": "public_aggregate_checksums_and_contract_only",
    }


def source_evidence(audit: Path, splits: Path, review: Path, export: Path) -> tuple:
    source = load_labeled_audit(audit)
    verify_labeled_candidates(splits, audit=audit)
    verify_receipt(review, "review_receipt.json", "artifacts")
    accepted = verify_receipt(review / "results", "acceptance.json", "artifacts")
    if accepted.get("verdict") != "APROBADO":
        raise ValueError("Labeled split review has not been accepted")
    before = read_json(review / "results" / "before_sha256.json")
    if before["audit"] != signatures(audit) or before["suite"] != signatures(splits):
        raise ValueError("Methodological review is bound to different audit/splits")
    membership_groups = {}
    coverage_memberships = set()
    test_sets = []
    for run in read_json(splits / "summary.json")["runs"]:
        path = splits / "runs" / run["split_id"]
        metrics = read_json(path / "metrics.json")
        records = pd.read_parquet(path / "occurrence_memberships.parquet").sort_values(
            "frame_id"
        )
        membership = stable_id(
            {"rows": records[["frame_id", "content_id", "new_split"]].values.tolist()}
        )
        membership_groups.setdefault(membership, []).append(run["split_id"])
        if metrics["balance"]["all_classes_covered"]:
            coverage_memberships.add(membership)
        test_sets.append(set(records.loc[records.new_split == "test", "frame_id"]))
    facts = {
        "configurations": len(test_sets),
        "distinct_memberships": len(membership_groups),
        "distinct_support_eligible_memberships": len(coverage_memberships),
        "duplicate_membership_groups": [
            v for v in membership_groups.values() if len(v) > 1
        ],
        "common_test_intersection_occurrences": len(set.intersection(*test_sets)),
        "distinct_test_memberships": len({tuple(sorted(s)) for s in test_sets}),
    }
    for field in (
        "configurations",
        "distinct_memberships",
        "distinct_support_eligible_memberships",
    ):
        if accepted[field] != facts[field]:
            raise ValueError(
                "Acceptance review facts disagree with reconstructed memberships"
            )
    return source, {
        "labeled_candidates": facts,
        "historical_detector": audit_export(export),
    }


def _reconstruct(snapshot: Path, config: CommonTestConfig):
    source, evidence = source_evidence(
        *(snapshot / n for n in ("audit", "splits", "review", "export"))
    )
    reference = load_config(snapshot / "reference_yolo11n.yaml")
    if (
        reference["ultralytics_version"] != "8.3.203"
        or reference["torch_version"] != "2.7.1"
        or reference["train"]["epochs"] != 50
        or reference["train"]["imgsz"] != 640
        or reference["training_seeds"] != [42, 43, 44]
    ):
        raise ValueError(
            "The v1 proposal must retain the explicitly specified historical reference"
        )
    proposal, tables = derive_proposal(source, config)
    proposal["source_evidence"] = evidence
    proposal["detector_protocol_proposal"] = {
        "protocol": "common_test_yolo11n_v1",
        "reference_configuration": reference,
        "initialization": {
            "source": reference["model"],
            "sha256": None,
            "legacy_finetuned_weights_allowed": False,
        },
        "training_budget": "50_epochs_per_arm; unequal_dataset_sizes_mean_unequal_optimizer_steps",
        "batch": None,
        "batch_rule": "one_explicit_batch_for_all_arms_after_non_test_hardware_probe",
        "checkpoint_selection": "validation_macro_map50_95_only_with_fixed_five_classes",
        "test_queries_for_selection_allowed": False,
        "primary_metric": "macro_map50_95",
        "secondary_metrics": ["map50", "precision", "recall"],
        "missing_class_metric": "undefined_not_zero_or_reduced_macro",
        "uncertainty": "paired_test_predictions; detector_seeds_nested_in_membership; validated_dependence_blocks_when_available",
        "blocked_bootstrap_claim": "exact_or_candidate_groups_do_not_certify_temporal_independence",
        "adapter_state": "NOT_IMPLEMENTED",
    }
    # Dataframe positions are not scientific identity and are not serialized.
    return proposal, {
        name: table.reset_index(drop=True) for name, table in tables.items()
    }


def build_common_test(
    audit: Path,
    splits: Path,
    review: Path,
    export: Path,
    reference: Path,
    output: Path,
    config: CommonTestConfig,
    *,
    purpose="plan",
    source_root: Path | None = None,
    evidence_file: Path | None = None,
    legacy_artifacts: Path | None = None,
) -> dict:
    repo = Path(__file__).resolve().parents[3]
    inputs = {"audit": audit, "splits": splits, "review": review, "export": export}
    target = output.resolve()
    if purpose not in {"plan", "readiness"}:
        raise ValueError("Unsupported planning purpose")
    if (
        output.exists()
        or target.is_relative_to(repo)
        or any(
            target.is_relative_to(p.resolve()) or p.resolve().is_relative_to(target)
            for p in inputs.values()
        )
    ):
        raise ValueError(
            "Fresh output outside repository and immutable sources required"
        )
    source, _ = source_evidence(**inputs)
    load_config(reference)
    original_checks = (
        verify_original_sources(source, source_root, evidence_file)
        if source_root
        else {"scope": "verified_audit_snapshot_only"}
    )
    original_signatures = {name: signatures(path) for name, path in inputs.items()}
    ref_hash = file_sha256(reference)
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "sources"
    snapshot.mkdir()
    for name, path in inputs.items():
        shutil.copytree(path, snapshot / name)
        if signatures(snapshot / name) != original_signatures[name]:
            raise ValueError("Source changed while snapshotting")
    shutil.copyfile(reference, snapshot / "reference_yolo11n.yaml")
    proposal, tables = _reconstruct(snapshot, config)
    binding = {**original_signatures, "reference_yolo11n.yaml": ref_hash}
    identity = {
        "protocol": VERSION,
        "configuration": asdict(config),
        "sources": binding,
        "purpose": purpose,
        "implementation_sha256": {
            p.name: file_sha256(p)
            for p in Path(__file__).parent.glob("common_test*.py")
        },
    }
    proposal.update(plan_id=stable_id(identity), identity=identity)
    write_json(output / "proposal.json", proposal)
    for name, table in tables.items():
        table.to_parquet(output / name, index=False)
    local = legacy_artifacts or repo / "artifacts" / "detection"
    availability = {
        "inspection_scope": "explicit_or_default_local_artifact_root",
        "per_image_evidence_verified": False,
        "files_present_not_validated": {
            key: any(local.glob(pattern)) if local.exists() else False
            for key, pattern in {
                "runs": "runs/**/metadata.json",
                "image_statistics": "runs/**/*.npz",
                "materialized_views": "views/**/records.parquet",
                "weights": "runs/**/*.pt",
                "freeze": "protocol/runtime_freeze.json",
            }.items()
        },
    }
    write_json(
        output / "provenance.json",
        {
            **execution_provenance(),
            "original_source_checks": original_checks,
            "legacy_local_availability": availability,
            "planning_source_sha256": {
                p.name: file_sha256(p)
                for p in Path(__file__).parent.glob("common_test*.py")
            },
        },
    )
    receipt = {
        "schema_version": VERSION,
        "plan_id": proposal["plan_id"],
        "sources": binding,
        "output_sha256": {
            p.name: file_sha256(p) for p in output.iterdir() if p.is_file()
        },
    }
    verify_common_test(output, receipt=receipt)
    if (
        any(
            signatures(path) != original_signatures[name]
            for name, path in inputs.items()
        )
        or file_sha256(reference) != ref_hash
    ):
        raise ValueError("Sources changed during planning")
    write_json(output / "summary.json", receipt)
    return proposal


def verify_common_test(directory: Path, *, receipt: dict | None = None) -> dict:
    receipt = receipt if receipt is not None else read_json(directory / "summary.json")
    if receipt["schema_version"] != VERSION:
        raise ValueError("Not a common-test proposal")
    for name, digest in receipt["output_sha256"].items():
        if file_sha256(_safe(directory, name)) != digest:
            raise ValueError("Proposal output checksum mismatch")
    snapshot = directory / "sources"
    for name in ("audit", "splits", "review", "export"):
        if signatures(snapshot / name) != receipt["sources"][name]:
            raise ValueError("Source snapshot checksum mismatch")
    if (
        file_sha256(snapshot / "reference_yolo11n.yaml")
        != receipt["sources"]["reference_yolo11n.yaml"]
    ):
        raise ValueError("Reference detector configuration changed")
    stored = read_json(directory / "proposal.json")
    identity = stored["identity"]
    config = CommonTestConfig(**identity["configuration"])
    if (
        identity["protocol"] != VERSION
        or identity["purpose"] not in {"plan", "readiness"}
        or identity["sources"] != receipt["sources"]
        or stable_id(identity) != stored["plan_id"]
        or stored["plan_id"] != receipt["plan_id"]
    ):
        raise ValueError("Proposal identity mismatch")
    rebuilt, tables = _reconstruct(snapshot, config)
    rebuilt.update(plan_id=stored["plan_id"], identity=identity)
    if set(receipt["output_sha256"]) != {"proposal.json", "provenance.json", *tables}:
        raise ValueError("Incomplete proposal artifact receipt")
    if json.loads(json.dumps(rebuilt)) != stored:
        raise ValueError(
            "Recomputed proposal differs; approval/freeze/training readiness cannot be asserted"
        )
    for name, expected in tables.items():
        actual = pd.read_parquet(directory / name)
        if "label_ids" in actual:
            actual["label_ids"] = actual.label_ids.map(list)
        pd.testing.assert_frame_equal(actual, expected)
    return {
        "proposal_valid": True,
        "plan_id": stored["plan_id"],
        "state": stored["readiness"]["state"],
        "ready_for_training": False,
        "arm_test_identity_equality": len({a["test_id"] for a in stored["arms"]}) == 1
        if stored["arms"]
        else None,
        "actual_train_val_assignments_verified": False,
    }
