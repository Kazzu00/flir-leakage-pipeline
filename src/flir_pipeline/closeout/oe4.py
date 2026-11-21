"""CPU-only OE4 freeze, audit and authorization bindings; no torch at import time."""

from __future__ import annotations

import importlib.metadata
import os
import uuid
from pathlib import Path

import pandas as pd
import yaml

from flir_pipeline.closeout.common_test import (
    development_records,
    materialize,
    overlap_audit,
    publication_context,
    verify_test,
)
from flir_pipeline.closeout.io import (
    environment,
    read_json,
    seal,
    transaction,
    verify,
    write_json,
)
from flir_pipeline.detection.common_admission import weights_receipt
from flir_pipeline.detection.runtime import parse_training_history
from flir_pipeline.detection.scientific_executor import verify_execution
from flir_pipeline.similarity.storage import file_sha256

PROTOCOL = "oe4_closeout_v1"
POLICIES = ("historica", "aleatoria", "c10")
AUG_ZERO = (
    "hsv_h",
    "hsv_s",
    "hsv_v",
    "degrees",
    "translate",
    "scale",
    "shear",
    "perspective",
    "flipud",
    "fliplr",
    "bgr",
    "mosaic",
    "mixup",
    "cutmix",
    "copy_paste",
    "erasing",
    "close_mosaic",
)
OPERATIONAL = {
    "model",
    "data",
    "project",
    "name",
    "save_dir",
    "exist_ok",
    "device",
    "resume",
}
REQUIRED = {
    "epochs": 50,
    "seed": 42,
    "imgsz": 640,
    "batch": 1,
    "optimizer": "SGD",
    "task": "detect",
    "mode": "train",
    "split": "val",
    "resume": False,
}
EVAL = {
    "imgsz": 640,
    "batch": 1,
    "conf": 0.001,
    "iou": 0.7,
    "max_det": 300,
    "half": False,
    "rect": False,
    "augment": False,
    "agnostic_nms": False,
    "classes": None,
    "workers": 0,
}


def path(value):
    expanded = os.path.expandvars(str(value))
    if "$HOME" in expanded or "${HOME}" in expanded:
        expanded = expanded.replace("${HOME}", str(Path.home())).replace(
            "$HOME", str(Path.home())
        )
    if "$" in expanded:
        raise ValueError("Unresolved environment variable in configured path")
    return Path(expanded).expanduser().resolve()


def load_config(config_path):
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if (
        cfg["protocol"] != PROTOCOL
        or tuple(cfg["runs"]) != POLICIES
        or cfg["evaluation"] != EVAL
    ):
        raise ValueError("Unexpected OE4 protocol/policy/evaluation configuration")
    return cfg


def aug_off(baseline: dict) -> dict:
    required_keys = {
        *AUG_ZERO,
        "auto_augment",
        "copy_paste_mode",
        "multi_scale",
        "lr0",
        "lrf",
        "momentum",
        "weight_decay",
        "warmup_epochs",
        "warmup_momentum",
        "warmup_bias_lr",
        "nbs",
        "patience",
        "amp",
        "deterministic",
        "workers",
        "cache",
        "rect",
        "fraction",
        "pretrained",
        "freeze",
        "single_cls",
        "box",
        "cls",
        "dfl",
        "time",
        "cfg",
        "val",
        "save",
        "classes",
        "augment",
        *REQUIRED,
    }
    if not required_keys <= set(baseline):
        raise ValueError(
            f"Missing recorded controls; no defaults inferred: {sorted(required_keys - set(baseline))}"
        )
    if any(baseline[k] != v for k, v in REQUIRED.items()):
        raise ValueError("Baseline differs from 50-epoch seed42 SGD YOLO11m protocol")
    if (
        baseline["time"] is not None
        or baseline["cfg"] is not None
        or baseline["classes"] is not None
        or not baseline["val"]
        or not baseline["save"]
        or baseline["augment"]
    ):
        raise ValueError(
            "Unexpected runtime override/class restriction/test augmentation"
        )
    if baseline["multi_scale"] or baseline["pretrained"] is not True:
        raise ValueError(
            "Baseline multiscale/pretrained controls incompatible with matched AUG_OFF"
        )
    if not any(
        float(baseline[k]) > 0 for k in AUG_ZERO if k not in ("close_mosaic", "erasing")
    ):
        raise ValueError("Baseline is not AUG_ON")
    return {
        **{k: v for k, v in baseline.items() if k not in OPERATIONAL},
        **dict.fromkeys(AUG_ZERO, 0),
        "auto_augment": None,
        "multi_scale": False,
    }


def check_last_checkpoint_metadata(checkpoint: dict, history: pd.DataFrame) -> None:
    """Ultralytics final stripping sets epoch=-1; require its full bound history.

    This validates the receipt-selected last.pt, not an epoch inferred from file size.
    Best can also contain final history, so checkpoint selection remains path/hash-bound.
    """
    if checkpoint.get("version") != "8.3.203" or checkpoint.get("epoch") not in (
        -1,
        49,
    ):
        raise ValueError("Unexpected checkpoint version or epoch")
    embedded = pd.DataFrame(checkpoint.get("train_results", {}))
    embedded.columns = embedded.columns.str.strip()
    history = history.copy()
    history.columns = history.columns.str.strip()
    if len(embedded) != 50 or embedded.get(
        "epoch", pd.Series(dtype=int)
    ).tolist() != list(range(1, 51)):
        raise ValueError("Checkpoint lacks all 50 completed epochs")
    if set(embedded.columns) != set(history.columns):
        raise ValueError("Checkpoint history columns differ from bound CSV")
    pd.testing.assert_frame_equal(
        embedded[history.columns], history, check_dtype=False, rtol=1e-6, atol=1e-8
    )
    args = checkpoint.get("train_args", {})
    if any(
        args.get(k) != v for k, v in REQUIRED.items() if k not in ("resume", "mode")
    ):
        raise ValueError("Checkpoint training controls differ from frozen OE4 protocol")


def code_hashes():
    root = Path(__file__).resolve().parents[3]
    result = {
        p.relative_to(root).as_posix(): file_sha256(p)
        for p in sorted((root / "src/flir_pipeline").rglob("*.py"))
    }
    for p in (root / "pyproject.toml", root / "uv.lock"):
        result[p.name] = file_sha256(p)
    return result


def preflight(config_path: Path):
    cfg = load_config(config_path)
    destination, common = path(cfg["freeze"]), path(cfg["common_test"])
    if destination.exists():
        return verify_freeze(config_path)
    blockers, checks, runs = [], {}, {}
    publication, source = path(cfg["publication"]), path(cfg["source_root"])
    # Preserve independent failures in one report rather than asking for manual debugging.
    try:
        test, _, binding = publication_context(publication, source)
        manifest = pd.read_parquet(
            publication / "sources/audit/candidate_manifest.parquet"
        )
        relations = read_json(publication / "sources/audit/relations.json")["records"]
        groups = pd.read_parquet(
            publication / "sources/proposal/protection_groups.parquet"
        )
        groups = manifest[["frame_id", "content_id"]].merge(
            groups[["content_id", "group_id"]],
            on="content_id",
            how="inner",
            validate="many_to_one",
        )
        checks["publication"] = binding
        if common.exists():
            verify_test(common)
            if (
                read_json(common / "provenance.json")["test_records_sha256"]
                != binding["test_records_sha256"]
            ):
                raise ValueError("Existing common-test source differs")
        else:
            materialize(publication, source, common)
        checks["common_test_receipt_sha256"] = file_sha256(common / "receipt.json")
    except (OSError, ValueError, KeyError) as exc:
        blockers.append(f"test/publication: {type(exc).__name__}: {exc}")
        manifest = None
    try:
        initial = path(cfg["initial_weights"])
        receipt = weights_receipt(path(cfg["weights_receipt"]))
        if file_sha256(initial) != receipt["weights"]["sha256"]:
            raise ValueError("Initial weights differ from official receipt")
        checks["initial_weights_sha256"] = file_sha256(initial)
        checks["weights_receipt_sha256"] = file_sha256(path(cfg["weights_receipt"]))
    except (OSError, ValueError, KeyError) as exc:
        blockers.append(f"initial weights: {type(exc).__name__}: {exc}")
    for policy, location in cfg["runs"].items():
        try:
            root = path(location)
            state = verify_execution(root, {"state": "completed"})
            needed = {
                "train/args.yaml",
                "train/results.csv",
                "train/weights/last.pt",
                "train/weights/best.pt",
            }
            if not needed <= set(state["files"]) or state.get("test_used") is not False:
                raise ValueError("Unbound baseline checkpoint/history or test usage")
            history = parse_training_history(root / "train/results.csv", 50)
            args = yaml.safe_load(
                (root / "train/args.yaml").read_text(encoding="utf-8")
            )
            treatment = aug_off(args)
            model = path(args["model"])
            if model.name != "yolo11m.pt" or file_sha256(model) != checks.get(
                "initial_weights_sha256"
            ):
                raise ValueError(
                    "Baseline initialization differs from supplied original weights"
                )
            dataset = path(args["data"])
            if manifest is None:
                raise ValueError("Source identity audit unavailable")
            dev = development_records(dataset, manifest)
            overlap = overlap_audit(
                test, dev, relations, cfg["temporal_window_frames"], groups
            )
            if not overlap["hard_identity_disjoint"]:
                blockers.append(
                    f"{policy}: confirmed identity/group overlap with protected test; preserved in residual audit"
                )
            runs[policy] = {
                "root": str(root),
                "execution_sha256": file_sha256(root / "execution.json"),
                "dataset_yaml": str(dataset),
                "dataset_yaml_sha256": file_sha256(dataset),
                "baseline_args": args,
                "aug_off_args": treatment,
                "baseline_label": "AUG_ON / validation epoch 50",
                "history": history,
                "last_sha256": file_sha256(root / "train/weights/last.pt"),
                "best_sha256": file_sha256(root / "train/weights/best.pt"),
                "overlap": overlap,
                "development": dev,
            }
        except (OSError, ValueError, KeyError) as exc:
            blockers.append(f"{policy}: {type(exc).__name__}: {exc}")
    if len(runs) == 3:
        controls = [
            {k: v for k, v in r["baseline_args"].items() if k not in OPERATIONAL}
            for r in runs.values()
        ]
        if any(v != controls[0] for v in controls[1:]):
            blockers.append("Nonoperational baseline arguments differ between policies")
    versions = {}
    for name, expected in cfg["versions"].items():
        try:
            versions[name] = importlib.metadata.version(name)
            if versions[name].split("+")[0] != expected:
                blockers.append(
                    f"Version {name}: expected {expected}, observed {versions[name]}"
                )
        except importlib.metadata.PackageNotFoundError:
            blockers.append(f"Missing runtime dependency: {name}")
    report_destination = (
        destination
        if not blockers
        else destination.with_name(destination.name + ".blocked-" + uuid.uuid4().hex)
    )
    with transaction(
        report_destination,
        publication,
        source,
        common,
        *[path(v) for v in cfg["runs"].values()],
    ) as stage:
        for policy, item in runs.items():
            item.pop("development").to_parquet(
                stage / f"development_{policy}.parquet", index=False
            )
        report = {
            "protocol": PROTOCOL,
            "state": "BLOCKED" if blockers else "READY",
            "blockers": blockers,
            "configuration": cfg,
            "configuration_sha256": file_sha256(config_path),
            "checks": checks,
            "runs": runs,
            "versions": versions,
            "code": code_hashes(),
            "gpu_checked": False,
            "scientific_authorization": "Not asserted; signed new-protocol scope required at worker",
            **environment(),
        }
        write_json(stage / "preflight.json", report)
        seal(stage, {"state": report["state"]})
    return report


def verify_freeze(config_path: Path):
    cfg = load_config(config_path)
    root = path(cfg["freeze"])
    verify(root)
    report = read_json(root / "preflight.json")
    if (
        report["configuration_sha256"] != file_sha256(config_path)
        or report["code"] != code_hashes()
    ):
        raise ValueError(
            "Frozen config/code changed; use a new immutable operation root"
        )
    if report["state"] != "READY":
        return report
    if (
        file_sha256(path(cfg["initial_weights"]))
        != report["checks"]["initial_weights_sha256"]
    ):
        raise ValueError("Frozen initial checkpoint changed")
    verify_test(path(cfg["common_test"]))
    if (
        file_sha256(path(cfg["common_test"]) / "receipt.json")
        != report["checks"]["common_test_receipt_sha256"]
    ):
        raise ValueError("Frozen protected test changed")
    for policy, run in report["runs"].items():
        directory = Path(run["root"])
        verify_execution(directory, {"state": "completed"})
        if (
            file_sha256(directory / "execution.json") != run["execution_sha256"]
            or file_sha256(Path(run["dataset_yaml"])) != run["dataset_yaml_sha256"]
        ):
            raise ValueError("Frozen baseline receipt/development YAML changed")
        dev = pd.read_parquet(root / f"development_{policy}.parquet")
        for row in dev.itertuples():
            if (
                file_sha256(Path(row.image_path)) != row.image_sha256
                or file_sha256(Path(row.label_path)) != row.label_sha256
            ):
                raise ValueError("Frozen development content changed")
    return report


def authorize(config_path: Path, phase: str):
    from flir_pipeline.detection.admission import authorized

    cfg = load_config(config_path)
    freeze = verify_freeze(config_path)
    if freeze["state"] != "READY":
        raise ValueError(f"BLOCKED: {freeze['blockers']}")
    permission = authorized(
        path(cfg[f"{phase}_admission"]), path(cfg[f"{phase}_signature"]), phase
    )
    if permission.get("protocol") != PROTOCOL or permission.get(
        "freeze_sha256"
    ) != file_sha256(path(cfg["freeze"]) / "receipt.json"):
        raise ValueError("Admission does not bind this new OE4 protocol/freeze")
    if phase == "evaluate":
        completed = {}
        for policy in POLICIES:
            directory = path(cfg["aug_off_output"]) / policy
            receipt = verify(directory)
            if (
                receipt["state"] != "COMPLETED_AUG_OFF"
                or receipt["freeze_sha256"] != permission["freeze_sha256"]
            ):
                raise ValueError(
                    "All three matched AUG_OFF runs must complete before test"
                )
            completed[policy] = file_sha256(directory / "receipt.json")
        if permission.get("aug_off_receipts") != completed:
            raise ValueError(
                "Evaluation release does not bind completed AUG_OFF matrix"
            )
    return cfg, freeze
