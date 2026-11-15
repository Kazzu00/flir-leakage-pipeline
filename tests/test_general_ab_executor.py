"""Synthetic-only integration checks: no model, weights download or real test."""

import copy
import hashlib
import io
import zipfile
from pathlib import Path

import pandas as pd
import pytest
import yaml
from PIL import Image
from test_scientific_executor import approved_payload as approved_payload
from typer.testing import CliRunner

from flir_pipeline.cli import app
from flir_pipeline.detection import admission
from flir_pipeline.detection import common_executor as executor
from flir_pipeline.detection import general_ab as ab
from flir_pipeline.detection import scientific_executor as scientific
from flir_pipeline.similarity.storage import file_sha256, write_json

CONFIG = Path("configs/detection/common_test_yolo11m_v2.yaml")


def bind(path):
    return {"path": str(path.resolve()), "sha256": file_sha256(path)}


def replace_table(root, descriptor, name, frame):
    path = root / descriptor[name]["path"]
    frame.to_csv(path, index=False)
    descriptor[name]["sha256"] = file_sha256(path)
    write_json(root / "bundle.json", descriptor)


@pytest.fixture
def bundle(tmp_path):
    root, sources = tmp_path / "bundle", tmp_path / "sources"
    root.mkdir()
    sources.mkdir()
    records = []
    with zipfile.ZipFile(sources / "data.zip", "w") as archive:
        for i in range(8):
            origin = {1: 0, 5: 4}.get(i, i)
            stream = io.BytesIO()
            Image.new("RGB", (12, 12), (origin * 20, 13, 29)).save(stream, format="PNG")
            image = stream.getvalue()
            label = "\n".join(f"{c} 0.5 0.5 0.1 0.1" for c in range(5)).encode()
            archive.writestr(f"images/f{i}.png", image)
            archive.writestr(f"labels/f{i}.txt", label)
            records.append(
                {
                    "frame_id": f"f{i}",
                    "content_id": hashlib.sha256(image).hexdigest(),
                    "image_sha256": hashlib.sha256(image).hexdigest(),
                    "label_sha256": hashlib.sha256(label).hexdigest(),
                    "image_source": "raw",
                    "image_member": f"images/f{i}.png",
                    "label_source": "raw",
                    "label_member": f"labels/f{i}.txt",
                }
            )
    records = pd.DataFrame(records)
    groups = records[["content_id"]].drop_duplicates().copy()
    groups["group_id"] = groups.content_id
    groups.loc[groups.content_id.isin(records.iloc[2:4].content_id), "group_id"] = (
        "constructed-dependent-pair"
    )
    frames = {
        "records": records,
        "test_ids": pd.DataFrame({"frame_id": ["f0"]}),
        "protected_contents": records.iloc[[0]][["content_id"]],
        "accepted_groups": groups,
    }
    descriptor = {
        "schema_version": ab.FORMAT,
        "dataset_id": "a" * 64,
        "population_id": "synthetic-original-identity",
        "test_status": "PROPOSED",
        "classes": {str(k): v for k, v in executor.NAMES.items()},
        "sources": {
            "raw": {"path": "data.zip", "sha256": file_sha256(sources / "data.zip")}
        },
        "arms": [],
    }
    for name, frame in frames.items():
        frame.to_csv(root / f"{name}.csv", index=False)
        descriptor[name] = {
            "path": f"{name}.csv",
            "sha256": file_sha256(root / f"{name}.csv"),
        }
    for policy in ab.POLICIES:
        for seed in range(5):
            mapping = [
                "test",
                "quarantine",
                "train",
                "val",
                "train",
                "train",
                "quarantine",
                "val",
            ]
            if policy == ab.POLICIES[1]:
                mapping = [
                    "test",
                    "quarantine",
                    "train",
                    "train",
                    "val",
                    "val",
                    "quarantine",
                    "val",
                ]
            frame = pd.DataFrame({"frame_id": records.frame_id, "new_split": mapping})
            arm_id = f"{policy}-{seed}"
            path = root / f"{arm_id}.csv"
            frame.to_csv(path, index=False)
            descriptor["arms"].append(
                {
                    "arm_id": arm_id,
                    "policy": policy,
                    "seed": seed,
                    "assignments": {"path": path.name, "sha256": file_sha256(path)},
                }
            )
    write_json(root / "bundle.json", descriptor)
    return root, sources, descriptor


def test_explicit_identity_and_existing_cli(bundle, tmp_path):
    root, _, descriptor = bundle
    _, summary, assignments, test, _, _ = ab.load(root)
    assert len(summary["runs"]) == 10 and list(test.frame_id) == ["f0"]
    assert set(assignments[descriptor["arms"][0]["arm_id"]].frame_id) == {
        f"f{i}" for i in range(8)
    }
    result = CliRunner().invoke(
        app,
        [
            "detection",
            "common-executor",
            "dry-run",
            str(root),
            str(tmp_path / "plan"),
            "--input-format",
            ab.FORMAT,
        ],
    )
    assert result.exit_code == 0, result.output
    plan = executor.verify_executor_plan(tmp_path / "plan")
    assert (
        plan["execution_count"] == 6
    )  # Ten logical arms; two memberships, three detector seeds.
    assert {a["policy"] for r in plan["runs"] for a in r["aliases"]} == set(ab.POLICIES)
    assert all(
        r["identity"]["configuration"]["model"] == "yolo11m.pt" for r in plan["runs"]
    )
    assert not plan["ready_for_training"]


@pytest.mark.parametrize(
    "corruption",
    [
        "missing",
        "duplicate",
        "test",
        "protected",
        "quarantine",
        "group",
        "identity_column",
    ],
)
def test_invalid_assignments_are_rejected(bundle, corruption):
    root, _, descriptor = bundle
    arm = descriptor["arms"][5]  # B must preserve the constructed dependency.
    path = root / arm["assignments"]["path"]
    frame = pd.read_csv(path)
    if corruption == "missing":
        frame = frame.iloc[1:]
    elif corruption == "duplicate":
        frame = pd.concat([frame, frame.iloc[[0]]])
    elif corruption == "identity_column":
        frame["content_id"] = "untrusted alternate identity"
    else:
        idx, value = {
            "test": (0, "val"),
            "protected": (1, "train"),
            "quarantine": (6, "train"),
            "group": (3, "val"),
        }[corruption]
        frame.loc[idx, "new_split"] = value
    frame.to_csv(path, index=False)
    arm["assignments"]["sha256"] = file_sha256(path)
    write_json(root / "bundle.json", descriptor)
    with pytest.raises((ValueError, AssertionError)):
        ab.load(root)


def test_checksum_and_catalogue_are_not_reinterpreted(bundle):
    root, _, descriptor = bundle
    descriptor["classes"]["4"] = "silently renamed class"
    write_json(root / "bundle.json", descriptor)
    with pytest.raises(ValueError, match="catalogue"):
        ab.load(root)
    descriptor["classes"]["4"] = executor.NAMES[4]
    write_json(root / "bundle.json", descriptor)
    (root / "records.csv").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        ab.load(root)


def test_protected_transitive_closure_required(bundle):
    root, _, descriptor = bundle
    frame = pd.read_csv(root / "accepted_groups.csv")
    frame.loc[:1, "group_id"] = "test-and-development-dependency"
    replace_table(root, descriptor, "accepted_groups", frame)
    with pytest.raises(ValueError, match="closure"):
        ab.load(root)


def test_materialization_reproduces_bytes_and_excludes_test_quarantine(
    bundle, tmp_path
):
    root, sources, _ = bundle
    before = file_sha256(sources / "data.zip")
    first = executor.prepare(root, sources, tmp_path / "one", input_format=ab.FORMAT)
    second = executor.prepare(root, sources, tmp_path / "two", input_format=ab.FORMAT)
    assert (
        first["views"] == second["views"]
        and file_sha256(sources / "data.zip") == before
    )
    for out in (tmp_path / "one", tmp_path / "two"):
        assert {p.stem for p in (out / "store/images").iterdir()} == {
            "f2",
            "f3",
            "f4",
            "f5",
            "f7",
        }
        executor.verify_preparation(out)
        for config in (out / "views").glob("*/dataset.yaml"):
            assert "test" not in yaml.safe_load(config.read_text())
    for file in (tmp_path / "one/store").rglob("*"):
        if file.is_file():
            assert (
                file.read_bytes()
                == (
                    tmp_path / "two/store" / file.relative_to(tmp_path / "one/store")
                ).read_bytes()
            )


def test_annotation_class_rejected_even_with_updated_source_hashes(bundle, tmp_path):
    root, sources, descriptor = bundle
    old = sources / "data.zip"
    with zipfile.ZipFile(old) as z:
        items = {n: z.read(n) for n in z.namelist()}
    items["labels/f2.txt"] = b"5 0.5 0.5 0.2 0.2"
    with zipfile.ZipFile(old, "w") as z:
        for name, raw in items.items():
            z.writestr(name, raw)
    frame = pd.read_csv(root / "records.csv")
    frame.loc[frame.frame_id == "f2", "label_sha256"] = hashlib.sha256(
        items["labels/f2.txt"]
    ).hexdigest()
    descriptor["sources"]["raw"]["sha256"] = file_sha256(old)
    replace_table(root, descriptor, "records", frame)
    with pytest.raises(ValueError, match="annotation"):
        executor.prepare(root, sources, tmp_path / "bad", input_format=ab.FORMAT)


def test_detector_configuration_and_initialization_bind_identity(bundle, tmp_path):
    root, _, _ = bundle
    one = executor.dry_run(root, CONFIG, tmp_path / "one", "1" * 64, ab.FORMAT)
    two = executor.dry_run(root, CONFIG, tmp_path / "two", "2" * 64, ab.FORMAT)
    assert one["plan_id"] != two["plan_id"]
    assert {str(r["identity"]["configuration"]) for r in one["runs"]} == {
        str(r["identity"]["configuration"]) for r in two["runs"]
    }


def test_signed_admission_still_rejects_unapproved_test(
    bundle, tmp_path, approved_payload
):
    approved_payload.update(
        input_format=ab.FORMAT,
        population="synthetic-original-identity",
        dataset_id="a" * 64,
        policies=ab.POLICIES,
        split_seeds=list(range(5)),
        max_runs=6,
    )
    approved_payload["test"]["status"] = "PROPOSED"
    with pytest.raises(ValueError, match="Provisional"):
        admission.training_admission(tmp_path, tmp_path)
    approved_payload["test"]["status"] = "APPROVED"  # Synthetic gate fixture only.
    assert admission.training_admission(tmp_path, tmp_path)["max_runs"] == 6


def test_admitted_context_preserves_aliases_and_rejects_identity_changes(
    bundle, tmp_path, approved_payload
):
    root, sources, _ = bundle
    plan_path, prep, out = (tmp_path / n for n in ("plan", "prepared", "runs"))
    plan = executor.dry_run(
        root, CONFIG, plan_path, approved_payload["weights"]["sha256"], ab.FORMAT
    )
    receipt = executor.prepare(root, sources, prep, input_format=ab.FORMAT)
    approved_payload.update(
        input_format=ab.FORMAT,
        population="synthetic-original-identity",
        dataset_id="a" * 64,
        policies=ab.POLICIES,
        split_seeds=list(range(5)),
        max_runs=6,
        plan=bind(plan_path / "plan.json"),
        preparation=bind(prep / "preparation.json"),
        config=bind(CONFIG),
        output_root=str(out.resolve()),
    )
    approved_payload["test"]["records_id"] = receipt["input_contract"][
        "test_records_id"
    ]
    document = tmp_path / "authority" / "signed-fixture.json"
    document.parent.mkdir()
    write_json(document, {"synthetic": True})
    context = scientific.context(document, tmp_path / "fixture.sig", plan_path, out)
    assert len(context["runs"]) == 6
    assert sum(len(r["aliases"]) for r in context["runs"]) == 30
    assert context["input_contract"] == plan["input_contract"]
    approved_payload["dataset_id"] = "b" * 64
    with pytest.raises(ValueError, match="dataset"):
        scientific.context(document, tmp_path / "fixture.sig", plan_path, out)


def test_equal_arms_cannot_authorize_redundant_training():
    selected = [
        {
            "records_id": "same",
            "detector_seed": d,
            "aliases": [
                {"policy": p, "split_seed": s} for p in ab.POLICIES for s in range(5)
            ],
        }
        for d in (42, 43, 44)
    ]
    expected = {(p, s, d) for p in ab.POLICIES for s in range(5) for d in (42, 43, 44)}
    with pytest.raises(ValueError, match="No distinct"):
        ab.validate_selected(selected, expected)
    partial = copy.deepcopy(selected)
    partial[0]["aliases"].pop()
    with pytest.raises(ValueError, match="Incomplete"):
        ab.validate_selected(partial, expected)


def test_released_test_uses_same_bound_records_only(bundle, tmp_path):
    root, sources, _ = bundle
    _, summary, _, test, _, _ = ab.load(root)
    ctx = {
        "input_contract": summary["input_contract"],
        "admission": {
            "test": {"records_id": executor.records_identity(test)},
            "input_format": ab.FORMAT,
            "variant": {"kind": "original"},
        },
    }
    permission = {
        "test_sources": {
            "general_ab_bundle": bind(root / "bundle.json"),
            "source_root": str(sources),
        }
    }
    result = ab.materialize_released_test(
        permission, ctx, tmp_path / "released-fixture"
    )
    assert list(result.frame_id) == ["f0"]
    assert len(list((tmp_path / "released-fixture/original/images").iterdir())) == 1
    scientific.write_view(result, tmp_path / "released-fixture/view")
    scientific.verify_test_view(tmp_path / "released-fixture", permission, ctx)
    (tmp_path / "released-fixture/view/labels/f0.txt").write_text("changed")
    with pytest.raises(ValueError, match="checksum"):
        scientific.verify_test_view(tmp_path / "released-fixture", permission, ctx)
