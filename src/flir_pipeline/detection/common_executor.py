"""Technical handoff of provisional assignments; scientific admission is closed.

No legacy detector plan or freeze is accepted here. Run equivalence binds full
occurrences (including quarantine), original annotations and configuration.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import yaml

from flir_pipeline.detection.annotation_review import source_bytes
from flir_pipeline.detection.common_test_storage import signatures
from flir_pipeline.detection.materialization import _write_checked
from flir_pipeline.similarity.storage import (
    execution_provenance,
    file_sha256,
    read_json,
    stable_id,
    write_json,
)
from flir_pipeline.splitting.fixed_test_storage import verify_fixed_test
from flir_pipeline.splitting.labeled import load_labeled_audit, verify_original_sources

VERSION = "common_test_yolo11m_v2"
NAMES = {0: "vehicle", 1: "building", 2: "road", 3: "river", 4: "SDZI"}
CLASS_DISCREPANCY = "Original ID 4=SDZI; internal catalogue=Heavy Machinery; equivalence NOT adjudicated"
BLOCKERS = [
    "verified_human_protocol_review_missing",
    "annotation_and_class_adjudication_missing",
    "common_test_approval_and_freeze_missing",
    "methodological_freeze_missing",
    "environment_and_common_batch_freeze_missing",
    "authenticated_admission_contract_not_implemented",
]


def external_output(output: Path, *sources: Path) -> None:
    target = output.resolve()
    repo = Path(__file__).resolve().parents[3]
    if target.is_relative_to(repo) or any(
        target.is_relative_to(p.resolve()) or p.resolve().is_relative_to(target)
        for p in sources
    ):
        raise ValueError("Output must be external and disjoint from immutable sources")


def load_executor_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    # Reuse historical validation without broadening its public admission contract.
    from flir_pipeline.detection.protocol import validate_config

    if config["protocol"] != VERSION or config["model"] != "yolo11m.pt":
        raise ValueError("Expected independent YOLO11m v2 protocol")
    validate_config(
        {**config, "protocol": "controlled_detector_v1", "model": "yolo11n.pt"}
    )
    if (
        config["training_seeds"] != [42, 43, 44]
        or config["ultralytics_version"] != "8.3.203"
        or config["torch_version"] != "2.7.1"
        or config["train"]["epochs"] != 50
        or config["train"]["imgsz"] != 640
        or config["train"]["batch"] is not None
        or config["compute"] != {
            "cpu_batch_candidates": [1, 2],
            "cuda_batch_candidates": [1, 2],
            "synthetic_epochs": 1,
            "synthetic_images_per_partition": 10,
            "real_pilot_authorized": False,
        }
    ):
        raise ValueError(
            "v2 proposal requires pinned libraries, 50 epochs, 640 and unresolved batch"
        )
    return config


def active_records(rows: pd.DataFrame, expected_test: pd.DataFrame) -> pd.DataFrame:
    if (
        not rows.frame_id.is_unique
        or not rows.new_split.isin(["train", "val", "test", "quarantine"]).all()
    ):
        raise ValueError("Invalid occurrence assignments")
    test = rows.loc[rows.new_split == "test"].drop(columns="new_split")
    pd.testing.assert_frame_equal(
        test.sort_values("frame_id").reset_index(drop=True),
        expected_test.sort_values("frame_id").reset_index(drop=True),
    )
    active = rows.loc[rows.new_split != "quarantine"].copy()
    if active.groupby("content_id").new_split.nunique().gt(1).any():
        raise ValueError("Exact content crosses active partitions")
    if set(active.new_split) != {"train", "val", "test"}:
        raise ValueError("Empty active partition")
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", str(f)) for f in rows.frame_id):
        raise ValueError("Unsafe frame identity")
    return active.sort_values("frame_id").reset_index(drop=True)


def records_identity(rows: pd.DataFrame) -> str:
    # JSON serialization also normalizes parquet list columns and missing provenance.
    import json

    return stable_id(
        {"records": json.loads(rows.sort_values("frame_id").to_json(orient="records"))}
    )


def load_publication(directory: Path):
    binding = file_sha256(directory / "receipt.json")
    verify_fixed_test(directory)
    summary = read_json(directory / "summary.json")
    if summary["configurations"] != 30 or summary["failed_configurations"]:
        raise ValueError("Complete verified 30-arm publication required")
    test = pd.read_parquet(directory / "sources/proposal/test_records.parquet")
    if len(test) != 372:
        raise ValueError("The proposed common test must retain 372 occurrences")
    assignments = {}
    for arm in summary["runs"]:
        rows = pd.read_parquet(
            directory / "runs" / arm["arm"] / "occurrence_memberships.parquet"
        )
        active_records(rows, test)
        assignments[arm["arm"]] = rows
    if file_sha256(directory / "receipt.json") != binding:
        raise ValueError("Publication changed during verification")
    return summary, assignments, test


def plan_runs(
    summary: dict,
    assignments: dict,
    config: dict,
    source_id: str,
    checkpoint_sha256: str | None = None,
) -> dict:
    if checkpoint_sha256 is not None and not re.fullmatch(
        r"[0-9a-f]{64}", checkpoint_sha256
    ):
        raise ValueError("Invalid initialization SHA256")
    runs = {}
    for arm in sorted(summary["runs"], key=lambda a: a["arm"]):
        rows = assignments[arm["arm"]]
        for seed in config["training_seeds"]:
            identity = {
                "protocol": VERSION,
                "source_id": source_id,
                "records_id": records_identity(rows),
                "configuration": config,
                "initialization": {
                    "model": "yolo11m.pt",
                    "sha256": checkpoint_sha256,
                    "origin": "official_ultralytics_release",
                    "verified": False,
                },
                "detector_seed": seed,
            }
            run_id = stable_id(identity)
            entry = runs.setdefault(
                run_id,
                {
                    "run_id": run_id,
                    "identity": identity,
                    "aliases": [],
                    "state": "pending",
                    "blockers": [*BLOCKERS, "official_checkpoint_provenance_pending"],
                    "resources": {
                        "gpu_count": 1,
                        "cpu_count": None,
                        "vram_bytes": None,
                        "storage_bytes": None,
                        "duration_seconds": None,
                    },
                },
            )
            entry["aliases"].append(
                {
                    "arm": arm["arm"],
                    "population": arm["population"],
                    "policy": arm["policy"],
                    "split_seed": arm["seed"],
                    "candidate_id": arm["candidate_id"],
                }
            )
    payload = {
        "schema_version": VERSION,
        "source_id": source_id,
        "arm_count": len(summary["runs"]),
        "experimental_identities": len(summary["runs"]) * len(config["training_seeds"]),
        "execution_count": len(runs),
        "runs": sorted(runs.values(), key=lambda r: r["run_id"]),
        "ready_for_training": False,
        "class_discrepancy": CLASS_DISCREPANCY,
        "deduplication": "candidate equivalence only until initialization and batch are frozen",
    }
    return {"plan_id": stable_id(payload), **payload}


def dry_run(
    publication: Path,
    config_path: Path,
    output: Path,
    checkpoint_sha256: str | None = None,
) -> dict:
    external_output(output, publication)
    if output.exists():
        raise ValueError("Plan destination exists; never overwrite evidence")
    summary, assignments, _ = load_publication(publication)
    config = load_executor_config(config_path)
    binding = file_sha256(publication / "receipt.json")
    plan = plan_runs(summary, assignments, config, binding, checkpoint_sha256)
    output.mkdir(parents=True)
    write_json(output / "plan.json", plan)
    write_json(output / "provenance.json", execution_provenance())
    write_json(output / "receipt.json", {"artifacts": signatures(output)})
    return plan


def verify_executor_plan(directory: Path) -> dict:
    receipt = read_json(directory / "receipt.json")
    actual = signatures(directory)
    actual.pop("receipt.json", None)
    if actual != receipt["artifacts"]:
        raise ValueError("Executor plan checksum mismatch")
    plan = read_json(directory / "plan.json")
    payload = {k: v for k, v in plan.items() if k != "plan_id"}
    if stable_id(payload) != plan["plan_id"] or plan["schema_version"] != VERSION:
        raise ValueError("Executor plan identity mismatch")
    return plan


def prepare(
    publication: Path, source_root: Path, output: Path, evidence: Path | None = None
) -> dict:
    """Materialize development only; test remains bound in metadata, never in YAML.

    Occurrence labels may conflict. Shared storage reuses each occurrence across
    all arms, without merging its annotations with another occurrence.
    """
    external_output(output, publication, source_root)
    if output.exists():
        raise ValueError("Preparation destination exists; inspect partial output")
    summary, assignments, test = load_publication(publication)
    audit = publication / "sources/audit"
    source = load_labeled_audit(audit)
    original = verify_original_sources(source, source_root, evidence)
    labels = {
        r["label_id"]: r for r in read_json(audit / "annotation_audit.json")["records"]
    }
    output.mkdir(parents=True)
    views = {}
    for arm in summary["runs"]:
        rows = assignments[arm["arm"]]
        identity = records_identity(rows)
        if identity in views:
            views[identity]["aliases"].append(arm["arm"])
            continue
        active = active_records(rows, test)
        development = active.loc[active.new_split.isin(["train", "val"])].copy()
        image_paths = []
        for record in development.to_dict(orient="records"):
            suffix = Path(record["zip_member_name"]).suffix.lower()
            if suffix not in {
                ".png",
                ".jpg",
                ".jpeg",
                ".bmp",
                ".tif",
                ".tiff",
                ".webp",
            }:
                raise ValueError("Unsupported image format")
            image = output / "store/images" / (record["frame_id"] + suffix)
            label_path = output / "store/labels" / (record["frame_id"] + ".txt")
            label = labels[record["label_ids"][0]]
            _write_checked(
                image,
                source_bytes(source_root, record, record["image_sha256"]),
                record["image_sha256"],
            )
            _write_checked(
                label_path,
                source_bytes(source_root, label, record["label_sha256"]),
                record["label_sha256"],
            )
            image_paths.append(str(image.resolve()))
        development["image_path"] = image_paths
        view = output / "views" / identity
        view.mkdir(parents=True)
        rows.to_parquet(view / "assignments.parquet", index=False)
        development.to_parquet(view / "records.parquet", index=False)
        for split in ("train", "val"):
            (view / f"{split}.txt").write_text(
                "\n".join(development.loc[development.new_split == split, "image_path"])
                + "\n",
                encoding="utf-8",
            )
        (view / "dataset.yaml").write_text(
            yaml.safe_dump(
                {
                    "path": str(output.resolve()),
                    "nc": 5,
                    "names": NAMES,
                    **{s: str((view / f"{s}.txt").resolve()) for s in ("train", "val")},
                }
            ),
            encoding="utf-8",
        )
        views[identity] = {"aliases": [arm["arm"]], "test_id": records_identity(test)}
    receipt = {
        "schema_version": VERSION,
        "source_id": file_sha256(publication / "receipt.json"),
        "views": views,
        "original_sources": original,
        "ready_for_training": False,
        "class_discrepancy": CLASS_DISCREPANCY,
        "test_materialized": False,
        "provenance": execution_provenance(),
        "artifacts": signatures(output),
    }
    write_json(output / "preparation.json", receipt)
    verify_preparation(output)
    return receipt


def verify_preparation(output: Path) -> dict:
    receipt = read_json(output / "preparation.json")
    actual = signatures(output)
    actual.pop("preparation.json", None)
    if actual != receipt["artifacts"]:
        raise ValueError("Prepared data checksum mismatch")
    for identity, entry in receipt["views"].items():
        view = output / "views" / identity
        rows = pd.read_parquet(view / "assignments.parquet")
        test = rows.loc[rows.new_split == "test"].drop(columns="new_split")
        if (
            records_identity(rows) != identity
            or records_identity(test) != entry["test_id"]
        ):
            raise ValueError("Prepared membership identity mismatch")
        active_records(rows, test)
        records = pd.read_parquet(view / "records.parquet")
        expected = (
            rows.loc[rows.new_split.isin(["train", "val"])]
            .sort_values("frame_id")
            .reset_index(drop=True)
        )
        pd.testing.assert_frame_equal(records.drop(columns="image_path"), expected)
        config = yaml.safe_load((view / "dataset.yaml").read_text(encoding="utf-8"))
        if "test" in config or config["names"] != NAMES or config["nc"] != 5:
            raise ValueError("Test exposure or class mapping changed")
        for row in records.itertuples():
            image = Path(row.image_path)
            label = image.parent.parent / "labels" / f"{row.frame_id}.txt"
            if (
                not image.resolve().is_relative_to(output.resolve())
                or image.stem != row.frame_id
            ):
                raise ValueError("Invalid prepared locator")
            if (
                file_sha256(image) != row.image_sha256
                or file_sha256(label) != row.label_sha256
            ):
                raise ValueError("Prepared image/annotation changed")
        for split in ("train", "val"):
            path = view / f"{split}.txt"
            if (
                Path(config[split]).resolve() != path.resolve()
                or path.read_text(encoding="utf-8").splitlines()
                != records.loc[records.new_split == split, "image_path"].tolist()
            ):
                raise ValueError("Prepared split list changed")
    return receipt


def require_admission() -> None:
    """Do not interpret self-authored JSON as attributed scientific authorization."""
    raise ValueError("Real training/resume BLOCKED: " + "; ".join(BLOCKERS))


def train(plan: Path, output: Path, index: int, resume: bool = False) -> None:
    # Must precede imports, checkpoint loading, output creation and all GPU work.
    require_admission()
