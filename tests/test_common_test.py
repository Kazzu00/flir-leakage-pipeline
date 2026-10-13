"""Offline common-test proposal gates; synthetic data, no detector runtime."""

from pathlib import Path

import pandas as pd
import pytest
from test_detector_association import project as project
from test_detector_final_report import build
from test_detector_final_report import complete_project as complete_project
from test_labeled_splits import audited as audited
from typer.testing import CliRunner

from flir_pipeline.cli import app
from flir_pipeline.detection.common_test import (
    CommonTestConfig,
    derive_proposal,
    readiness_gate,
    verify_arm_assignments,
)
from flir_pipeline.detection.common_test_storage import (
    audit_export,
    build_common_test,
    signatures,
    source_evidence,
    verify_common_test,
    verify_receipt,
)
from flir_pipeline.detection.protocol import verify_plan
from flir_pipeline.similarity.storage import file_sha256, read_json, write_json
from flir_pipeline.splitting.base import SplitConfig
from flir_pipeline.splitting.construction import aggregate_groups
from flir_pipeline.splitting.labeled import LabeledConfig, load_labeled_audit
from flir_pipeline.splitting.labeled_storage import build_labeled_candidates


@pytest.fixture
def source(audited):
    return load_labeled_audit(audited[0])


def synthetic_assignments(source, proposal, tables):
    """Only test fixtures. Production proposals do not assign development rows."""
    assignments = {}
    for arm in proposal["arms"]:
        manifest = (
            source.manifest
            if arm["population"] == "expanded"
            else source.manifest.loc[source.manifest.cohort == "historical"]
        )
        rows = manifest.copy()
        rows["new_split"] = "quarantine"
        rows.loc[
            rows.frame_id.isin(tables["test_records.parquet"].frame_id), "new_split"
        ] = "test"
        development = tables[f"development_{arm['population']}.parquet"]
        groups = tables[
            f"development_groups_{arm['population']}_{arm['policy']}.parquet"
        ]
        stats = source.statistics.loc[
            source.statistics.frame_id.isin(development.frame_id)
        ]
        units = aggregate_groups(stats, groups)
        val_groups = {
            units.loc[units[f"class_{c}"] > 0, "group_id"].iloc[0] for c in range(5)
        }
        val_ids = set(groups.loc[groups.group_id.isin(val_groups), "content_id"])
        rows.loc[rows.frame_id.isin(development.frame_id), "new_split"] = "train"
        rows.loc[
            rows.frame_id.isin(development.frame_id) & rows.content_id.isin(val_ids),
            "new_split",
        ] = "val"
        assignments[arm["arm"]] = rows
    return assignments


def test_internal_and_prospective_never_freeze(source):
    plan, tables = derive_proposal(source, CommonTestConfig())
    assert plan["readiness"]["state"] == "PROPOSED"
    assert not plan["readiness"]["ready_for_training"]
    assert len({a["test_id"] for a in plan["arms"]}) == 1
    for pop in ("historical", "expanded"):
        assert not set(tables[f"development_{pop}.parquet"].content_id) & set(
            tables["protected_records.parquet"].content_id
        )
    unavailable, empty = derive_proposal(
        source, CommonTestConfig(option="prospective", anchor_rule="none")
    )
    assert (
        unavailable["readiness"]["state"] == "UNAVAILABLE"
        and unavailable["test_id"] is None
    )
    assert empty["test_records.parquet"].empty
    assert not readiness_gate(option_available=True, conflicts=0, test_covered=True)[
        "ready_for_training"
    ]


def test_transitive_uniform_protection_and_conflicts_preserved(source):
    frame = source.manifest.set_index("basename")
    a = frame.loc["a0"]
    anchor = frame.loc["h5"]
    edge = {
        **next(r for r in source.relations if r["state"] == "visual_candidate"),
        "relation_id": "synthetic_link_to_test",
        "left_frame_id": a.frame_id,
        "left_content_id": a.content_id,
        "right_frame_id": anchor.frame_id,
        "right_content_id": anchor.content_id,
    }
    source.relations.append(edge)
    original = source.manifest.copy(deep=True)
    plan, tables = derive_proposal(source, CommonTestConfig())
    assert plan["protected_annotation_conflict_groups"] == 1
    assert "protected_annotation_conflicts_unresolved" in plan["readiness"]["blockers"]
    assert {"a0", "h0", "h1", "h5"} <= set(tables["protected_records.parquet"].basename)
    for pop in ("historical", "expanded"):
        assert not {"a0", "h0", "h1", "h5"} & set(
            tables[f"development_{pop}.parquet"].basename
        )
    pd.testing.assert_frame_equal(source.manifest, original)
    assert (
        tables["protected_records.parquet"]
        .loc[lambda d: d.cohort == "adendo", "original_split"]
        .isna()
        .all()
    )


def test_complete_assignment_gate_and_aliases(source):
    plan, tables = derive_proposal(source, CommonTestConfig())
    assignments = synthetic_assignments(source, plan, tables)
    result = verify_arm_assignments(source, plan, tables, assignments)
    assert result["assignments_valid"] and not result["ready_for_training"]
    assert result["distinct_memberships"] < len(assignments)


@pytest.mark.parametrize(
    "corruption",
    ["test", "annotation", "exact", "candidate", "validation", "coverage", "groups"],
)
def test_assignment_gate_rejects_invalid_handoff(source, corruption):
    # Put a candidate endpoint in the protected closure first.
    frame = source.manifest.set_index("basename")
    a, b = frame.loc["a0"], frame.loc["h5"]
    source.relations.append(
        {
            **source.relations[0],
            "state": "visual_candidate",
            "relation_id": "new",
            "left_frame_id": a.frame_id,
            "left_content_id": a.content_id,
            "right_frame_id": b.frame_id,
            "right_content_id": b.content_id,
        }
    )
    plan, tables = derive_proposal(source, CommonTestConfig())
    assignments = synthetic_assignments(source, plan, tables)
    key = "expanded-S1-0"
    rows = assignments[key]
    if corruption == "test":
        rows.loc[rows.new_split == "test", "new_split"] = "quarantine"
    elif corruption == "annotation":
        rows.loc[rows.new_split == "test", "label_sha256"] = "tampered"
    elif corruption in {"exact", "candidate"}:
        rows.loc[
            rows.basename == ("h0" if corruption == "exact" else "a0"), "new_split"
        ] = "train"
    elif corruption == "validation":
        rows.loc[rows.new_split == "val", "new_split"] = "train"
    elif corruption == "coverage":
        del assignments[key]
    else:
        # Use an unprotected duplicate and force a split fracture in a fresh plan.
        source.relations = [r for r in source.relations if r["relation_id"] != "new"]
        plan, tables = derive_proposal(source, CommonTestConfig())
        assignments = synthetic_assignments(source, plan, tables)
        rows = assignments[key]
        idx = rows.index[rows.basename == "h0"].tolist()
        rows.loc[idx[0], "new_split"], rows.loc[idx[1], "new_split"] = "train", "val"
    with pytest.raises((ValueError, AssertionError)):
        verify_arm_assignments(source, plan, tables, assignments)


def test_missing_test_class_blocks_readiness(source):
    test_ids = set(
        source.manifest.loc[source.manifest.original_split == "test", "frame_id"]
    )
    source.statistics.loc[source.statistics.frame_id.isin(test_ids), "class_4"] = 0
    proposal, _ = derive_proposal(source, CommonTestConfig())
    assert "test_class_support_incomplete" in proposal["readiness"]["blockers"]


@pytest.fixture
def bundle(audited, tmp_path, monkeypatch):
    """Storage contract fixture; aggregate export semantics have separate tests."""
    import flir_pipeline.detection.common_test_storage as storage

    audit, _, _ = audited
    splits = tmp_path / "splits"
    receipt = build_labeled_candidates(
        audit,
        splits,
        LabeledConfig(balance=SplitConfig(seeds=(0,), node_limit=2)),
        populations=("expanded",),
        policies=("S1",),
    )
    metrics = read_json(
        splits / "runs" / receipt["runs"][0]["split_id"] / "metrics.json"
    )
    review = tmp_path / "review"
    (review / "results").mkdir(parents=True)
    write_json(
        review / "results" / "before_sha256.json",
        {"audit": signatures(audit), "suite": signatures(splits)},
    )
    write_json(
        review / "results" / "acceptance.json",
        {
            "verdict": "APROBADO",
            "configurations": 1,
            "distinct_memberships": 1,
            "distinct_support_eligible_memberships": int(
                metrics["balance"]["all_classes_covered"]
            ),
            "artifacts": {
                "before_sha256.json": file_sha256(
                    review / "results" / "before_sha256.json"
                )
            },
        },
    )
    write_json(review / "review_receipt.json", {"artifacts": signatures(review)})
    export = tmp_path / "export"
    export.mkdir()
    write_json(export / "placeholder.json", {"synthetic": True})
    monkeypatch.setattr(
        storage, "audit_export", lambda p: {"synthetic_export_stub": True}
    )
    return audit, splits, review, export, Path("configs/detection/yolo11n.yaml")


def test_publication_reproducibility_and_read_only(bundle, tmp_path):
    before = [signatures(p) for p in bundle[:4]]
    a = build_common_test(*bundle, tmp_path / "first", CommonTestConfig())
    b = build_common_test(*bundle, tmp_path / "second", CommonTestConfig())
    assert a == b
    assert verify_common_test(tmp_path / "first")["proposal_valid"]
    assert before == [signatures(p) for p in bundle[:4]]
    with pytest.raises(ValueError, match="Fresh"):
        build_common_test(*bundle, tmp_path / "first", CommonTestConfig())


@pytest.mark.parametrize(
    "field,value",
    [
        ("state", "READY_FOR_TRAINING"),
        ("ready_for_training", True),
        ("approval_state", "APPROVED"),
        ("freeze_state", "FROZEN"),
    ],
)
def test_forged_readiness_fails_even_after_checksum_repair(
    bundle, tmp_path, field, value
):
    directory = tmp_path / "plan"
    build_common_test(*bundle, directory, CommonTestConfig())
    plan = read_json(directory / "proposal.json")
    plan["readiness"][field] = value
    write_json(directory / "proposal.json", plan)
    receipt = read_json(directory / "summary.json")
    receipt["output_sha256"]["proposal.json"] = file_sha256(directory / "proposal.json")
    write_json(directory / "summary.json", receipt)
    with pytest.raises(ValueError, match="approval/freeze"):
        verify_common_test(directory)


@pytest.mark.parametrize("kind", ["checksum", "missing"])
def test_incomplete_sources_rejected(bundle, tmp_path, kind):
    path = bundle[2] / "results" / "acceptance.json"
    if kind == "checksum":
        path.write_text("{}", encoding="utf8")
    else:
        path.unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        build_common_test(*bundle, tmp_path / "no-publication", CommonTestConfig())
    assert not (tmp_path / "no-publication").exists()


def test_rejected_review_cannot_authorize_source_evidence(bundle):
    review = bundle[2]
    acceptance = review / "results" / "acceptance.json"
    data = read_json(acceptance)
    data["verdict"] = "REQUIERE CORRECCIONES"
    write_json(acceptance, data)
    receipt = review / "review_receipt.json"
    linked = read_json(receipt)
    linked["artifacts"]["results/acceptance.json"] = file_sha256(acceptance)
    write_json(receipt, linked)
    with pytest.raises(ValueError, match="has not been accepted"):
        source_evidence(*bundle[:4])


def test_legacy_plan_cannot_consume_proposal(bundle, tmp_path):
    proposal = build_common_test(*bundle, tmp_path / "new-plan", CommonTestConfig())
    write_json(tmp_path / "plan.json", proposal)
    with pytest.raises(ValueError, match="approved protocol"):
        verify_plan(tmp_path)


def test_receipt_traversal_and_crlf_are_not_silently_repaired(tmp_path):
    payload = tmp_path / "file.json"
    payload.write_bytes(b"{}\n")
    digest = file_sha256(payload)
    write_json(tmp_path / "receipt.json", {"artifacts": {"file.json": digest}})
    payload.write_bytes(b"{}\r\n")
    with pytest.raises(ValueError, match="checksum"):
        verify_receipt(tmp_path, "receipt.json", "artifacts")
    write_json(tmp_path / "receipt.json", {"artifacts": {"../outside": digest}})
    with pytest.raises(ValueError, match="Unsafe"):
        verify_receipt(tmp_path, "receipt.json", "artifacts")


def test_cli_surface_and_prospective_publication(bundle, tmp_path):
    runner = CliRunner()
    for command in ("audit-readiness", "plan-common-test", "verify-common-test"):
        assert runner.invoke(app, ["detection", command, "--help"]).exit_code == 0
    output = tmp_path / "prospective"
    build_common_test(
        *bundle, output, CommonTestConfig(option="prospective", anchor_rule="none")
    )
    result = runner.invoke(app, ["detection", "verify-common-test", str(output)])
    assert result.exit_code == 0 and '"state": "UNAVAILABLE"' in result.output


def test_export_contract_is_not_a_predictions_source(complete_project, tmp_path):
    with pytest.raises(FileNotFoundError):
        audit_export(tmp_path)
    build(complete_project, tmp_path)
    result = audit_export(tmp_path / "frontend")
    assert result["aggregate_export_valid"] and result["exported_run_count"] == 48
    assert not result["original_hypatia_runs_reverified"]
    assert not result["per_image_predictions_available_from_export"]
