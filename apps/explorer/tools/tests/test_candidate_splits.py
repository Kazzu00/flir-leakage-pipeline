"""Synthetic, offline tests of the presentation-only importer."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location(
    "candidate_export", Path(__file__).parents[1] / "export_candidate_splits.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf8")


def receipt(root):
    write_json(
        root / "receipt.json",
        {
            "schema_version": "fixed_test_train_val_v1",
            "artifacts": {
                p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in root.rglob("*")
                if p.is_file() and p.name != "receipt.json"
            },
        },
    )


@pytest.fixture
def publication(tmp_path):
    root = tmp_path / "source"
    runs = []
    for seed in [0, 1]:
        arm = f"historical-S1-{seed}"
        run = {"arm": arm, "candidate_id": f"candidate-{seed}", "membership_id": "same-membership"}
        runs.append(run)
        base = root / "runs" / arm
        write_json(
            base / "metadata.json",
            {
                "candidate_id": run["candidate_id"],
                "membership_id": run["membership_id"],
                "identity": {
                    "arm": {
                        "arm": arm,
                        "policy": "S1",
                        "population": "historical",
                        "split_seed": seed,
                        "test_id": "test-synthetic",
                    }
                },
            },
        )
        write_json(
            base / "metrics.json",
            {
                "ready_for_training": False,
                "state": "PROVISIONAL_VERIFIED",
                "class_support": {},
                "blockers": ["human_review_missing"],
            },
        )
        rows = [
            {
                "frame_id": f"f{i}",
                "content_id": f"c{i}",
                "basename": f"synthetic{i}",
                "source_archive": "synthetic",
                "cohort": "historical",
                "label_sha256": f"label{i}",
                "new_split": partition,
            }
            for i, partition in enumerate(["train", "val", "test", "quarantine"])
        ]
        pd.DataFrame(rows).to_parquet(base / "occurrence_memberships.parquet")
        pd.DataFrame({"content_id": ["c0", "c1"], "group_id": ["g0", "g1"]}).to_parquet(
            base / "atomic_groups.parquet"
        )
    write_json(
        root / "summary.json",
        {
            "ready_for_training": False,
            "state": "PROVISIONAL_CANDIDATES",
            "configurations": 2,
            "verified_candidates": 2,
            "runs": runs,
            "blockers": ["human_review_missing"],
        },
    )
    write_json(root / "suite.json", {"plan_id": "synthetic-plan"})
    write_json(
        root / "provenance.json", {"git_commit": "synthetic-commit", "git_worktree_dirty": True}
    )
    write_json(
        root / "sources/audit/annotation_audit.json",
        {
            "dataset_id": "synthetic-dataset",
            "records": [{"frame_ids": [f"f{i}"], "class_counts": {"0": 1}} for i in range(4)],
        },
    )
    write_json(
        root / "sources/audit/relations.json",
        {
            "records": [
                {
                    "relation_id": "edge",
                    "left_frame_id": "f0",
                    "right_frame_id": "f1",
                    "state": "visual_candidate",
                }
            ]
        },
    )
    receipt(root)
    return root


def test_preserves_membership_state_and_sources(publication, tmp_path):
    before = {p: p.read_bytes() for p in publication.rglob("*") if p.is_file()}
    output = tmp_path / "view.json"
    module.export(publication, output)
    result = json.loads(output.read_text())
    assert result["test_status"] == "PROPOSED"
    assert result["ready_for_training"] is False
    assert result["source_dirty"] is True
    assert len(result["records"]) == 4
    assert result["runs"][0]["members"][-1]["partition"] == "quarantine"
    assert result["runs"][0]["members"][-1]["group_id"] is None
    assert all(p.read_bytes() == raw for p, raw in before.items())
    module.export(publication, tmp_path / "second.json")
    assert output.read_bytes() == (tmp_path / "second.json").read_bytes()
    with pytest.raises(ValueError, match="new and outside"):
        module.export(publication, output)


def test_rejects_changed_source(publication, tmp_path):
    (publication / "summary.json").write_text("{}")
    with pytest.raises(ValueError, match="Checksum"):
        module.export(publication, tmp_path / "view.json")


def test_rejects_changed_common_test_even_with_new_receipt(publication, tmp_path):
    path = publication / "runs/historical-S1-1/occurrence_memberships.parquet"
    rows = pd.read_parquet(path)
    rows.loc[rows.new_split == "test", "new_split"] = "quarantine"
    rows.to_parquet(path)
    receipt(publication)
    with pytest.raises(ValueError, match="Common test"):
        module.export(publication, tmp_path / "view.json")


def test_rejects_missing_group(publication, tmp_path):
    path = publication / "runs/historical-S1-0/atomic_groups.parquet"
    pd.read_parquet(path).iloc[:1].to_parquet(path)
    receipt(publication)
    with pytest.raises(ValueError, match="allocation group"):
        module.export(publication, tmp_path / "view.json")


def test_rejects_self_approved_state(publication, tmp_path):
    path = publication / "summary.json"
    data = json.loads(path.read_text())
    data["ready_for_training"] = True
    write_json(path, data)
    receipt(publication)
    with pytest.raises(ValueError, match="provisional"):
        module.export(publication, tmp_path / "view.json")
