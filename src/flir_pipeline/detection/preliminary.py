"""Separate, content-level pilot: original nonempty labels, never protected test.

The user's explicit policy accepts original annotations for a preliminary run,
not as visually adjudicated ground truth. Final scientific admission is unchanged.
"""

from __future__ import annotations

import csv
import io
import shutil
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image

from flir_pipeline.data.yolo_labels import validate_label_bytes
from flir_pipeline.detection.common_executor import (
    NAMES,
    external_output,
    load_executor_config,
)
from flir_pipeline.detection.curated_development import source_path
from flir_pipeline.similarity.storage import file_sha256, read_json, write_json

VERSION = "original_nonempty_preliminary_v1"
POLICY = "user_accepts_original_nonempty_visual_review_pending"


def signatures(root: Path) -> dict:
    if any(p.is_symlink() for p in root.rglob("*")):
        raise ValueError("Links are forbidden")
    return {
        p.relative_to(root).as_posix(): file_sha256(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and p != root / "receipt.json"
    }


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_rows(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def prepare(
    source: Path,
    receipt_sha256: str,
    config: Path,
    output: Path,
    weights_sha256: str,
    *,
    policy: str = "S3",
    batch: int = 1,
    accept_original_nonempty: bool = False,
) -> dict:
    """Freeze an explicitly accepted pilot subset without constructing new splits."""
    external_output(output, source)
    if (
        output.exists()
        or not accept_original_nonempty
        or policy not in ("S1", "S2", "S3")
    ):
        raise ValueError(
            "Fresh output, explicit nonempty acceptance and existing policy required"
        )
    if (
        batch < 1
        or len(weights_sha256) != 64
        or any(c not in "0123456789abcdef" for c in weights_sha256)
    ):
        raise ValueError(
            "Positive fixed batch and SHA256 of local initial weights required"
        )
    source_receipt = source / "development_receipt.json"
    if file_sha256(source_receipt) != receipt_sha256:
        raise ValueError("Source receipt mismatch")
    binding = read_json(source_receipt)
    if binding["schema_version"] != "curated_development_v1":
        raise ValueError("Not a curated preparation")
    protection = binding["protected_exclusion"]
    if (
        protection.get("status") != "EXACT_EXCLUSION_VERIFIED_AGAINST_REFERENCE"
        or protection["overlap"]
    ):
        raise ValueError("Verified protected exclusion required")
    actual = signatures(source)
    actual.pop("development_receipt.json", None)
    if actual != binding["artifacts"]:
        raise ValueError("Source files changed")
    config_data = load_executor_config(config)
    rows = read_rows(source / "manifest.csv")
    selected, excluded, support, baseline = [], [], Counter(), Counter()
    copies = []
    for row in rows:
        split = row[f"{policy}_split"]
        if split not in ("train", "val"):
            raise ValueError("Test/quarantine forbidden")
        frame = row["frame_id"]
        ext = Path(row["image_path"]).suffix.lower()
        image = source_path(source, f"canonical/images/{frame}{ext}")
        label = source_path(source, f"canonical/labels/{frame}.txt")
        qa = validate_label_bytes(label.read_bytes())
        if not qa.label_valid or not set(qa.classes_present) <= set(NAMES):
            raise ValueError("Invalid original annotation")
        for box in qa.boxes:
            baseline[(split, box[0])] += 1
        if qa.label_empty:
            excluded.append(
                dict(
                    frame_id=frame,
                    content_id=row["content_id"],
                    split=split,
                    reason="empty_label_pending_review_not_assumed_wrong",
                )
            )
            continue
        image_name, label_name = (
            f"images/{split}/{frame}{ext}",
            f"labels/{split}/{frame}.txt",
        )
        selected.append(
            {
                **row,
                "split": split,
                "image_path": image_name,
                "label_path": label_name,
                "acceptance": POLICY,
            }
        )
        copies.extend([(image, image_name), (label, label_name)])
        for box in qa.boxes:
            support[(split, box[0])] += 1
    if {r["split"] for r in selected} != {"train", "val"}:
        raise ValueError("Nonempty train and validation required")
    train = {**config_data["train"], "batch": batch}
    experiment = dict(
        schema_version=VERSION,
        purpose="preliminary_positive_only_BASE",
        acceptance_policy=POLICY,
        split_policy=policy,
        split_seed=0,
        detector_seed=42,
        model="yolo11m.pt",
        weights_sha256=weights_sha256,
        train=train,
        test_access=False,
        paired_experiment=False,
        limitations=[
            "original_annotations_not_visually_readjudicated",
            "pending_empty_labels_excluded_selection_bias",
            "near_duplicates_and_temporal_dependence_not_ruled_out",
        ],
    )
    output.mkdir(parents=True)
    for path, name in copies:
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    write_rows(output / "manifest.csv", selected, list(selected[0]))
    write_rows(
        output / "exclusions.csv",
        excluded,
        ["frame_id", "content_id", "split", "reason"],
    )
    write_json(output / "experiment.json", experiment)
    shutil.copyfile(config, output / "environment-protocol.yaml")
    (output / "data.yaml").write_text(
        yaml.safe_dump(dict(train="images/train", val="images/val", nc=5, names=NAMES)),
        encoding="utf-8",
    )
    comparison = [
        dict(
            split=s,
            class_id=c,
            original_instances=baseline[(s, c)],
            retained_instances=support[(s, c)],
        )
        for s in ("train", "val")
        for c in NAMES
    ]
    write_json(
        output / "selection_impact.json",
        dict(
            before=dict(Counter(r[f"{policy}_split"] for r in rows)),
            after=dict(Counter(r["split"] for r in selected)),
            excluded=dict(Counter(r["split"] for r in excluded)),
            classes=comparison,
            interpretation="Removing empty labels changes background prevalence, not annotated instance counts",
        ),
    )
    receipt = dict(
        schema_version=VERSION,
        source_receipt_sha256=receipt_sha256,
        protected_exclusion=protection,
        acceptance_policy=POLICY,
        ready_for_preliminary_training=True,
        final_scientific_admission=False,
        counts=dict(Counter(r["split"] for r in selected)),
        artifacts=signatures(output),
    )
    write_json(output / "receipt.json", receipt)
    return verify(output, file_sha256(output / "receipt.json"))


def verify(package: Path, receipt_sha256: str) -> dict:
    """Portable payload and scientific invariants; no torch import or downloads."""
    if file_sha256(package / "receipt.json") != receipt_sha256:
        raise ValueError("Pilot receipt mismatch")
    receipt = read_json(package / "receipt.json")
    if (
        receipt["schema_version"] != VERSION
        or signatures(package) != receipt["artifacts"]
    ):
        raise ValueError("Pilot artifacts changed")
    config = read_json(package / "experiment.json")
    if (
        config["schema_version"] != VERSION
        or config["acceptance_policy"] != POLICY
        or config["test_access"]
        or config["paired_experiment"]
    ):
        raise ValueError("Not the accepted preliminary BASE experiment")
    rows = read_rows(package / "manifest.csv")
    for key in ("frame_id", "content_id", "image_sha256", "image_path", "label_path"):
        if len({r[key] for r in rows}) != len(rows):
            raise ValueError(f"Duplicate {key}")
    for row in rows:
        if row["split"] not in ("train", "val") or row["acceptance"] != POLICY:
            raise ValueError("Non-development or unaccepted observation")
        if row["split"] != row[f"{config['split_policy']}_split"]:
            raise ValueError("Original assignment changed")
        expected_stem = row["frame_id"]
        if (
            row["image_path"]
            != f"images/{row['split']}/{expected_stem}{Path(row['image_path']).suffix}"
            or row["label_path"] != f"labels/{row['split']}/{expected_stem}.txt"
        ):
            raise ValueError("Pairing or partition path mismatch")
        image, label = (
            source_path(package, row[k]) for k in ("image_path", "label_path")
        )
        if (
            row["content_id"] != row["image_sha256"]
            or file_sha256(image) != row["image_sha256"]
        ):
            raise ValueError("Image identity mismatch")
        if file_sha256(label) != row["label_sha256"]:
            raise ValueError("Annotation changed")
        qa = validate_label_bytes(label.read_bytes())
        if (
            not qa.label_valid
            or qa.label_empty
            or not set(qa.classes_present) <= set(NAMES)
            or len(set(qa.boxes)) != len(qa.boxes)
            or qa.num_objects != int(row["num_objects"])
        ):
            raise ValueError("Invalid or empty pilot annotation")
        with Image.open(io.BytesIO(image.read_bytes())) as decoded:
            decoded.load()
            if decoded.size != (int(row["width"]), int(row["height"])):
                raise ValueError("Dimension mismatch")
    counts = dict(Counter(r["split"] for r in rows))
    if set(counts) != {"train", "val"} or counts != receipt["counts"]:
        raise ValueError("Partition counts changed")
    data = yaml.safe_load((package / "data.yaml").read_text(encoding="utf-8"))
    if data != dict(train="images/train", val="images/val", nc=5, names=NAMES):
        raise ValueError("Unexpected dataset paths or test exposure")
    return dict(
        valid=True,
        counts=counts,
        exact_overlap=0,
        preliminary_only=True,
        acceptance_policy=POLICY,
        final_scientific_admission=False,
    )


def train(package: Path, receipt_sha256: str, weights: Path, output: Path) -> dict:
    """One fresh single-GPU pilot; no resume, final-test release or final admission."""
    verify(package, receipt_sha256)
    external_output(output, package, weights)
    if output.exists():
        raise ValueError("Use a fresh output; previous attempts must be preserved")
    config = read_json(package / "experiment.json")
    if file_sha256(weights) != config["weights_sha256"]:
        raise ValueError("Initial weights checksum mismatch")
    from flir_pipeline.detection.common_runtime import preflight
    from flir_pipeline.detection.runtime import (
        check_loader_annotations,
        parse_training_history,
        provenance,
    )
    from flir_pipeline.detection.scientific_runtime import (
        check_arguments,
        offline_runtime,
        termination_receipt,
    )

    output.mkdir(parents=True)
    state = dict(
        state="preflight",
        preliminary_only=True,
        test_used=False,
        receipt_sha256=receipt_sha256,
        config=config,
        provenance=provenance(),
    )
    try:
        state["environment"] = preflight(
            package / "environment-protocol.yaml", output / "environment.json"
        )
        # Runtime cache writes must not mutate the transferred immutable package.
        runtime = output / "development"
        shutil.copytree(package / "images", runtime / "images")
        shutil.copytree(package / "labels", runtime / "labels")
        dataset = runtime / "data.yaml"
        dataset.write_text(
            yaml.safe_dump(
                dict(
                    path=str(runtime.resolve()),
                    train="images/train",
                    val="images/val",
                    names=NAMES,
                    nc=5,
                )
            ),
            encoding="utf-8",
        )
        rows = read_rows(package / "manifest.csv")
        expected = {r["frame_id"]: int(r["num_objects"]) for r in rows}
        initial_runtime = signatures(runtime)
        with offline_runtime(), termination_receipt():
            from ultralytics import YOLO

            model = YOLO(str(weights.resolve()))
            if model.task != "detect" or model.model.yaml.get("scale") != "m":
                raise ValueError("Not a pretrained YOLO11m detector")

            def guard(trainer):
                check_arguments(vars(trainer.args), config, 42, dataset)
                if int(trainer.batch_size) != config["train"][
                    "batch"
                ] or trainer.data.get("test"):
                    raise ValueError("Batch changed or test exposed")

            def start(trainer):
                guard(trainer)
                check_loader_annotations(trainer, expected)
                for split, loader in (
                    ("train", trainer.train_loader),
                    ("val", trainer.test_loader),
                ):
                    observed = sorted(
                        Path(r["im_file"]).stem for r in loader.dataset.labels
                    )
                    if observed != sorted(
                        r["frame_id"] for r in rows if r["split"] == split
                    ):
                        raise ValueError("Loader changed membership")

            model.add_callback("on_train_start", start)
            model.add_callback("on_train_batch_end", guard)
            state["state"] = "running"
            write_json(output / "execution.json", state)
            model.train(
                data=str(dataset.resolve()),
                project=str(output.resolve()),
                name="train",
                exist_ok=False,
                device="0",
                seed=42,
                split="val",
                **config["train"],
            )
        for name, sha in initial_runtime.items():
            if file_sha256(runtime / name) != sha:
                raise ValueError("Runtime changed an input")
        verify(package, receipt_sha256)
        state["history"] = parse_training_history(
            output / "train/results.csv", config["train"]["epochs"]
        )
        for name in ("best.pt", "last.pt"):
            if not (output / "train/weights" / name).is_file():
                raise ValueError("Completed training lacks checkpoints")
        state["state"] = "completed"
        state["outputs"] = signatures(output / "train")
        return state
    except BaseException as exc:
        state.update(
            state="failed_or_interrupted", error=f"{type(exc).__name__}: {exc}"
        )
        raise
    finally:
        write_json(output / "execution.json", state)
