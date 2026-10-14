"""Offline fixed-test conservation, hard coverage, explicit failures and replay."""

import io
import shutil
import zipfile
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from test_annotation_review import inputs as inputs
from test_common_test import bundle as bundle
from typer.testing import CliRunner

from flir_pipeline.cli import app
from flir_pipeline.data.adendo import audit_adendo
from flir_pipeline.detection.annotation_review import build_review
from flir_pipeline.detection.common_test import CommonTestConfig, derive_proposal
from flir_pipeline.detection.common_test_storage import signatures
from flir_pipeline.similarity.storage import read_json, write_json
from flir_pipeline.splitting.base import SplitConfig
from flir_pipeline.splitting.construction import (
    AssignmentFailure,
    milp_assignment,
    random_assignment,
)
from flir_pipeline.splitting.fixed_test import (
    FixedTestConfig,
    construct_arm,
    inspect_arm,
    membership_id,
    suite_summary,
)
from flir_pipeline.splitting.fixed_test_storage import (
    build_fixed_test,
    review_transport,
    verify_fixed_test,
)
from flir_pipeline.splitting.labeled import load_labeled_audit


@pytest.fixture
def audited(tmp_path):
    def png(seed):
        stream = io.BytesIO()
        Image.fromarray(
            np.random.default_rng(seed).integers(0, 256, (16, 16), dtype=np.uint8)
        ).save(stream, format="PNG")
        return stream.getvalue()

    # Every ordinary synthetic image contains all classes: coverage cannot depend
    # on a lucky S1 seed in tests of the storage/pilot workflow.
    text = "".join(f"{c} .5 .5 .2 .2\n" for c in range(5))
    with (
        zipfile.ZipFile(tmp_path / "images.zip", "w") as images,
        zipfile.ZipFile(tmp_path / "labels.zip", "w") as labels,
    ):
        for i in range(30):
            split = ("train", "val", "test")[i % 3]
            images.writestr(f"{split}/h{i}.png", png(i))
            labels.writestr(f"{split}/h{i}.txt", text)
        images.writestr("train/h2.png", png(2))
        labels.writestr("train/h2.txt", "")
    with zipfile.ZipFile(tmp_path / "adendo.zip", "w") as archive:
        archive.writestr("a.png", png(100))
        archive.writestr("a.txt", text)
    audit = tmp_path / "audit"
    audit_adendo(
        tmp_path / "images.zip",
        tmp_path / "labels.zip",
        tmp_path / "adendo.zip",
        audit,
        phash_threshold=0,
    )
    return audit, tmp_path, None


def units(vectors):
    return pd.DataFrame(
        [
            {
                "group_id": f"g{i}",
                "record_count": 1,
                "content_count": 1,
                **{f"class_{c}": int(c in classes) for c in range(5)},
                "empty_annotation_count": int(not classes),
            }
            for i, classes in enumerate(vectors)
        ]
    )


def test_two_destinations_deterministic_and_coverage():
    data = units([set(range(5))] * 12 + [set()] * 3)
    ratios = np.array([0.8, 0.2])
    settings = SplitConfig(node_limit=4)
    first, solver = milp_assignment(
        data,
        ratios,
        2,
        settings,
        destinations=("train", "val"),
        require_class_coverage=True,
    )
    again, _ = milp_assignment(
        data.sample(frac=1, random_state=42),
        ratios,
        2,
        settings,
        destinations=("train", "val"),
        require_class_coverage=True,
    )
    assert first == again and set(first.values()) == {"train", "val"}
    assert solver["primal_feasibility_verified"]
    a, _ = random_assignment(data, ratios, 0, destinations=("train", "val"))
    b, _ = random_assignment(
        data.sample(frac=1, random_state=43), ratios, 0, destinations=("train", "val")
    )
    assert a == b
    with pytest.raises(ValueError, match="destinations"):
        milp_assignment(
            data, np.array([0.8, 0.1]), 0, settings, destinations=("train", "val")
        )


def test_incompatible_class_constraints_have_no_fallback():
    # Each class occurs in >=2 groups, yet classes 0..2 form a non-2-colorable cycle.
    data = units([{0, 2, 3, 4}, {0, 1, 3, 4}, {1, 2, 3, 4}])
    with pytest.raises(AssignmentFailure) as failed:
        milp_assignment(
            data,
            np.array([0.5, 0.5]),
            0,
            SplitConfig(),
            destinations=("train", "val"),
            require_class_coverage=True,
        )
    assert failed.value.evidence["status"] == 2
    assert not failed.value.evidence["primal_feasibility_verified"]
    with pytest.raises(AssignmentFailure, match="class support"):
        milp_assignment(
            units([{0, 1, 2, 3, 4}, {1, 2, 3, 4}]),
            np.array([0.5, 0.5]),
            0,
            SplitConfig(),
            destinations=("train", "val"),
            require_class_coverage=True,
        )


def test_protection_atomicity_and_random_failure(audited):
    source = load_labeled_audit(audited[0])
    # Candidate transitively protects an otherwise-development content, without
    # promoting the edge to confirmed provenance.
    anchor = source.manifest.loc[source.manifest.original_split == "test"].iloc[0]
    other = source.manifest.loc[source.manifest.basename == "h0"].iloc[0]
    source.relations.append(
        {
            "relation_id": "synthetic",
            "state": "visual_candidate",
            "left_frame_id": anchor.frame_id,
            "right_frame_id": other.frame_id,
            "left_content_id": anchor.content_id,
            "right_content_id": other.content_id,
        }
    )
    plan, tables = derive_proposal(source, CommonTestConfig())
    assert other.frame_id in set(tables["quarantine_records.parquet"].frame_id)
    arm = plan["arms"][0]
    config = FixedTestConfig()
    rows, _, solver = construct_arm(source, plan, tables, arm, config)
    metrics, _ = inspect_arm(source, plan, tables, arm, rows, solver, config)
    assert (
        metrics["state"] == "PROVISIONAL_VERIFIED" and not metrics["ready_for_training"]
    )
    changed = rows.copy()
    changed.loc[changed.frame_id == other.frame_id, "new_split"] = "train"
    with pytest.raises(ValueError, match="Protected"):
        inspect_arm(source, plan, tables, arm, changed, solver, config)
    changed = rows.copy()
    changed.loc[changed.new_split == "test", "label_sha256"] = "silently changed"
    with pytest.raises(AssertionError):
        inspect_arm(source, plan, tables, arm, changed, solver, config)
    # S1 is not repaired when a class has no validation support.
    validation = set(rows.loc[rows.new_split == "val", "frame_id"])
    source.statistics.loc[source.statistics.frame_id.isin(validation), "class_4"] = 0
    metrics, _ = inspect_arm(source, plan, tables, arm, rows, solver, config)
    assert metrics["state"] == "FAILED_CLASS_COVERAGE"


def test_aliases_are_measured_with_population_scope():
    rows = pd.DataFrame(
        {
            "frame_id": ["f1", "f2"],
            "content_id": ["c1", "c2"],
            "new_split": ["train", "val"],
        }
    )
    digest = membership_id(rows)
    assert digest == membership_id(rows.iloc[::-1])
    assert digest != membership_id(
        pd.concat(
            [
                rows,
                pd.DataFrame(
                    {"frame_id": ["f3"], "content_id": ["c3"], "new_split": ["val"]}
                ),
            ]
        )
    )
    entries = [
        {
            "arm": f"historical-{p}-0",
            "population": "historical",
            "membership_id": digest,
            "state": "PROVISIONAL_VERIFIED",
        }
        for p in ("S2", "S3")
    ]
    summary = suite_summary(entries)
    assert summary["configurations"] == 2 and summary["distinct_memberships"] == 1
    assert len(summary["aliases"]) == 1 and not summary["ready_for_training"]


def test_storage_pilot_full_replay_and_tampering(inputs, bundle, tmp_path):
    audit, proposal, acceptance, root = inputs
    review = tmp_path.parent / (tmp_path.name + "-review")
    build_review(audit, proposal, acceptance, root, review, expected_cases=1)
    paths = (audit, bundle[1], proposal, review)
    before = [signatures(p) for p in paths]
    pilot = tmp_path.parent / (tmp_path.name + "-pilot")
    result = build_fixed_test(*paths, pilot, FixedTestConfig(node_limit=2))
    assert result["verified_candidates"] == 6
    assert verify_fixed_test(pilot, replay=True)["replayed"]
    full = tmp_path.parent / (tmp_path.name + "-full")
    result = build_fixed_test(
        *paths, full, FixedTestConfig(seeds=tuple(range(5)), node_limit=2), pilot=pilot
    )
    assert result["configurations"] == result["verified_candidates"] == 30
    assert verify_fixed_test(full, replay=True)["replayed"]
    assert before == [signatures(p) for p in paths]
    with pytest.raises(ValueError, match="requires a verified pilot"):
        build_fixed_test(
            *paths,
            tmp_path.parent / (tmp_path.name + "-missing"),
            FixedTestConfig(seeds=tuple(range(5))),
        )
    # Repaired checksums must not bypass semantics or blocked scientific states.
    for kind in ("ready", "alias", "assignment", "checksum", "review"):
        changed = tmp_path.parent / (tmp_path.name + "-tampered-" + kind)
        shutil.copytree(pilot, changed)
        if kind in {"ready", "alias"}:
            summary = read_json(changed / "summary.json")
            summary[
                "ready_for_training" if kind == "ready" else "distinct_memberships"
            ] = True if kind == "ready" else 999
            write_json(changed / "summary.json", summary)
        elif kind in {"assignment", "checksum"}:
            table_path = changed / "runs/historical-S1-0/occurrence_memberships.parquet"
            rows = pd.read_parquet(table_path)
            rows.loc[rows.new_split == "quarantine", "new_split"] = "train"
            rows.to_parquet(table_path, index=False)
        else:
            manifest = read_json(changed / "sources/review/manifest.json")
            manifest["ready_for_training"] = True
            write_json(changed / "sources/review/manifest.json", manifest)
        if kind != "checksum":
            artifacts = signatures(changed)
            artifacts.pop("receipt.json")
            write_json(
                changed / "receipt.json",
                {"schema_version": "fixed_test_train_val_v1", "artifacts": artifacts},
            )
        with pytest.raises((ValueError, AssertionError)):
            verify_fixed_test(changed)


@pytest.mark.parametrize(
    "command", ["fixed-test-train-val", "verify-fixed-test-train-val"]
)
def test_cli_help(command):
    assert CliRunner().invoke(app, ["splitting", command, "--help"]).exit_code == 0


def test_redundant_transport_requires_exact_bytes(tmp_path):
    cases = tmp_path / "cases/example"
    cases.mkdir(parents=True)
    (cases / "A.txt").write_bytes(b"0 .5 .5 .2 .2\n")
    write_json(tmp_path / "receipt.json", {"artifacts": signatures(tmp_path)})
    assert review_transport(tmp_path) == {}
    with zipfile.ZipFile(tmp_path / "cases.zip", "w") as archive:
        archive.write(cases / "A.txt", "cases/example/A.txt")
    checked = review_transport(tmp_path)
    assert checked["cases.zip"]["case_files_verified"] == 1
    assert not checked["cases.zip"]["annotation_evidence_added"]
    with zipfile.ZipFile(tmp_path / "cases.zip", "w") as archive:
        archive.writestr("cases/example/A.txt", "different")
    with pytest.raises(ValueError, match="differs"):
        review_transport(tmp_path)
    with zipfile.ZipFile(tmp_path / "cases.zip", "w") as archive:
        archive.writestr("../A.txt", "unsafe")
    with pytest.raises(ValueError, match="Unsafe"):
        review_transport(tmp_path)


def test_s3_development_group_cannot_be_fractured(audited):
    source = load_labeled_audit(audited[0])
    first = source.manifest.loc[source.manifest.basename == "h0"].iloc[0]
    second = source.manifest.loc[source.manifest.basename == "h1"].iloc[0]
    source.relations.append(
        {
            "relation_id": "synthetic-development",
            "state": "visual_candidate",
            "left_frame_id": first.frame_id,
            "right_frame_id": second.frame_id,
            "left_content_id": first.content_id,
            "right_content_id": second.content_id,
        }
    )
    plan, tables = derive_proposal(source, CommonTestConfig())
    arm = next(
        a
        for a in plan["arms"]
        if a["population"] == "historical" and a["policy"] == "S3"
    )
    rows, _, solver = construct_arm(source, plan, tables, arm, FixedTestConfig())
    indices = rows.frame_id == first.frame_id
    rows.loc[indices, "new_split"] = (
        "val" if rows.loc[indices, "new_split"].iloc[0] == "train" else "train"
    )
    with pytest.raises(ValueError, match="group fracture"):
        inspect_arm(source, plan, tables, arm, rows, solver, FixedTestConfig())


def test_failed_solver_published_without_dropping_arms(
    inputs, bundle, tmp_path, monkeypatch
):
    import flir_pipeline.splitting.construction as construction

    audit, proposal, acceptance, root = inputs
    review = tmp_path.parent / (tmp_path.name + "-review")
    build_review(audit, proposal, acceptance, root, review, expected_cases=1)
    monkeypatch.setattr(
        construction,
        "milp",
        lambda *a, **k: SimpleNamespace(
            x=None, status=2, message="synthetic infeasible"
        ),
    )
    paths = (audit, bundle[1], proposal, review)
    output = tmp_path.parent / (tmp_path.name + "-failed-pilot")
    result = build_fixed_test(*paths, output, FixedTestConfig(node_limit=2))
    assert result["configurations"] == 6 and result["failed_configurations"] == 4
    assert result["verdict"] == "REQUIERE CORRECCIONES"
    assert (output / "receipt.json").exists()
    assert verify_fixed_test(output, replay=True)["failed_configurations"] == 4
    with pytest.raises(ValueError, match="Pilot is not eligible"):
        build_fixed_test(
            *paths,
            tmp_path.parent / (tmp_path.name + "-forbidden-full"),
            FixedTestConfig(seeds=tuple(range(5)), node_limit=2),
            pilot=output,
        )
