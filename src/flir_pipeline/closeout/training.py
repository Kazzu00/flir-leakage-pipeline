"""Shared native runtime-view contract for OE4 and isolated denoised exploration."""

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import yaml
from PIL import Image

from flir_pipeline.closeout.io import write_json
from flir_pipeline.closeout.oe4 import AUG_ZERO, OPERATIONAL
from flir_pipeline.detection.common_executor import NAMES
from flir_pipeline.detection.runtime import check_loader_annotations
from flir_pipeline.detection.scientific_executor import write_view
from flir_pipeline.similarity.storage import file_sha256

REQUIRED_COLUMNS = {
    "frame_id",
    "new_split",
    "image_path",
    "label_path",
    "image_sha256",
    "label_sha256",
}


def prepare(
    rows: pd.DataFrame, output: Path, controls: dict, initial: Path, names=None
):
    """Read counts from the native materialized view, as scientific_runtime does.

    Source manifests need not contain num_objects. write_view checks SHA256 and
    parses the actual YOLO bytes, then emits that derived field in records.parquet.
    Repeated content is allowed; repeated occurrence IDs are rejected before dicts.
    """
    if not REQUIRED_COLUMNS <= set(rows):
        raise ValueError(
            f"Missing development fields: {sorted(REQUIRED_COLUMNS - set(rows))}"
        )
    if rows.frame_id.isna().any() or not rows.frame_id.is_unique:
        raise ValueError("Development frame_id must identify unique occurrences")
    if set(rows.new_split) != {"train", "val"}:
        raise ValueError("Training requires nonempty train/val and no test")
    for row in rows.itertuples():
        with Image.open(row.image_path) as image:
            image.load()
    output.mkdir(parents=True, exist_ok=False)
    dataset = write_view(rows, output / "development", strict_geometry=False)
    # This is the historical executor's actual contract, not the input manifest.
    runtime_rows = pd.read_parquet(dataset.parent / "records.parquet")
    if not runtime_rows.frame_id.is_unique or set(runtime_rows.frame_id) != set(
        rows.frame_id
    ):
        raise ValueError("Runtime occurrence population changed")
    counts = runtime_rows.set_index("frame_id")["num_objects"].astype(int).to_dict()
    if names is not None:
        data = yaml.safe_load(dataset.read_text(encoding="utf-8"))
        data["names"] = names
        dataset.write_text(yaml.safe_dump(data), encoding="utf-8")
    settings = {
        **{k: v for k, v in controls.items() if k not in OPERATIONAL},
        "data": str(dataset.resolve()),
        "model": str(initial.resolve()),
        "device": "0",
        "project": str(output.resolve()),
        "name": "train",
        "exist_ok": False,
        "resume": False,
    }
    write_json(output / "planned_arguments.json", settings)
    result = {
        "dataset": dataset,
        "rows": runtime_rows,
        "counts": counts,
        "settings": settings,
        "output": output,
        "names": names or NAMES,
    }
    write_json(
        output / "preparation.json",
        {
            "state": "VIEW_PREPARED",
            "source_columns": sorted(rows.columns),
            "runtime_columns": sorted(runtime_rows.columns),
            "records": len(rows),
            "counts_from": "development/records.parquet parsed YOLO bytes",
            "objects": sum(counts.values()),
            "splits": rows.new_split.value_counts().to_dict(),
            "runtime_records_sha256": file_sha256(dataset.parent / "records.parquet"),
            "training_executed": False,
        },
    )
    return result


def guard(trainer, prepared, check_membership=True):
    actual = vars(trainer.args)
    if Path(actual["data"]).resolve() != prepared["dataset"].resolve():
        raise ValueError("Effective development dataset locator changed")
    for key, value in prepared["settings"].items():
        if key not in OPERATIONAL and actual.get(key) != value:
            raise ValueError(f"Effective training control changed: {key}")
    if trainer.data.get("test") or int(trainer.batch_size) != 1:
        raise ValueError("Test exposed or batch changed")
    if trainer.data["names"] != prepared["names"]:
        raise ValueError("Effective class catalogue changed")
    if not check_membership:
        return
    check_loader_annotations(trainer, prepared["counts"])
    for split, loader in (
        ("train", trainer.train_loader),
        ("val", trainer.test_loader),
    ):
        observed = sorted(
            Path(label["im_file"]).stem for label in loader.dataset.labels
        )
        expected = sorted(
            prepared["rows"].loc[prepared["rows"].new_split == split, "frame_id"]
        )
        if observed != expected:
            raise ValueError(
                "Loader dropped, duplicated or reassigned development members"
            )


def check_cpu(prepared, recorded: dict):
    """Real pinned Ultralytics config and label loader; no model/forward/training."""
    import ultralytics
    from ultralytics.cfg import DEFAULT_CFG_DICT, get_cfg
    from ultralytics.data.dataset import YOLODataset

    if ultralytics.__version__ != "8.3.203":
        raise ValueError("Pinned Ultralytics 8.3.203 required for CPU preparation")
    if set(DEFAULT_CFG_DICT) - set(recorded) or set(recorded) - set(
        DEFAULT_CFG_DICT
    ) - {"save_dir"}:
        raise ValueError("Recorded args do not cover pinned runtime keys exactly")
    effective = get_cfg(overrides=prepared["settings"])
    data = yaml.safe_load(prepared["dataset"].read_text(encoding="utf-8"))
    loaders = {}
    for split in ("train", "val"):
        # Scans the actual copied PNG/JPEG and YOLO files and detects upstream
        # dropping/deduplication. Caches stay in this isolated disposable view.
        dataset = YOLODataset(
            img_path=data[split],
            data=data,
            imgsz=effective.imgsz,
            batch_size=effective.batch,
            augment=split == "train",
            hyp=effective,
            rect=effective.rect,
            cache=False,
            fraction=effective.fraction,
            classes=effective.classes,
            single_cls=effective.single_cls,
            prefix=f"CPU {split}: ",
        )
        loaders[split] = SimpleNamespace(dataset=dataset)
    trainer = SimpleNamespace(
        args=effective,
        data=data,
        batch_size=effective.batch,
        train_loader=loaders["train"],
        test_loader=loaders["val"],
    )
    guard(trainer, prepared)
    result = {
        "state": "READY_TO_INVOKE_TRAIN",
        "ultralytics": ultralytics.__version__,
        "records": len(prepared["rows"]),
        "objects": sum(prepared["counts"].values()),
        "arguments_checked": True,
        "actual_yolo_loader_checked": True,
        "training_executed": False,
        "inference_executed": False,
        "gpu_used": False,
        "effective_augmentation": {
            key: getattr(effective, key)
            for key in (*AUG_ZERO, "auto_augment", "multi_scale")
        },
    }
    write_json(prepared["output"] / "cpu_validation.json", result)
    return result
