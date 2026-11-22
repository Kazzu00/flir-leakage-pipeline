"""Occurrence-preserving common test, verified against the original publication."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pandas as pd
import yaml

from flir_pipeline.closeout.denoised import labels
from flir_pipeline.closeout.io import (
    read_json,
    seal,
    transaction,
    verify,
    write_csv,
    write_json,
)
from flir_pipeline.detection.annotation_review import source_bytes
from flir_pipeline.detection.common_executor import NAMES
from flir_pipeline.similarity.storage import file_sha256
from flir_pipeline.splitting.fixed_test_storage import verify_fixed_test
from flir_pipeline.splitting.labeled import load_labeled_audit


def publication_context(publication: Path, source_root: Path):
    verification = verify_fixed_test(publication)
    test = pd.read_parquet(publication / "sources/proposal/test_records.parquet")
    if len(test) != 372 or not test.frame_id.is_unique or test.frame_id.isna().any():
        raise ValueError("Protected test must contain exactly 372 unique occurrences")
    audit = publication / "sources/audit"
    source = load_labeled_audit(audit)
    label_rows = {
        row["label_id"]: row
        for row in read_json(audit / "annotation_audit.json")["records"]
    }
    expected = {
        v["name"]: v["sha256"]
        for v in source.summary["configuration"]["sources"].values()
    }
    consumed = set(test.source_archive)
    for row in test.to_dict("records"):
        if len(row["label_ids"]) != 1:
            raise ValueError("Ambiguous protected annotation locator")
        consumed.add(label_rows[row["label_ids"][0]]["source_archive"])
    bindings = {}
    for name in sorted(consumed):
        if (
            Path(name).name != name
            or name not in expected
            or file_sha256(source_root / name) != expected[name]
        ):
            raise ValueError("Consumed original ZIP checksum mismatch")
        bindings[name] = expected[name]
    return (
        test,
        label_rows,
        {
            "publication_receipt_sha256": file_sha256(publication / "receipt.json"),
            "test_records_sha256": file_sha256(
                publication / "sources/proposal/test_records.parquet"
            ),
            "source_zip_sha256": bindings,
            "publication_verification": verification,
            "scope": "Publication snapshots and all ZIPs consumed by test verified; unused adendo/evidence original files not reread",
        },
    )


def write_test_view(
    rows: list[dict],
    destination: Path,
    source_root: Path | None = None,
    selected_classes=None,
):
    """Relative dataset tree is portable; labels/images remain paired by frame_id."""
    if destination.exists():
        raise FileExistsError("Test view already exists; refusing overwrite")
    if source_root and (
        destination.resolve().is_relative_to(source_root.resolve())
        or source_root.resolve().is_relative_to(destination.resolve())
    ):
        raise ValueError("Test runtime view must be disjoint from immutable source")
    if not rows or len({row["frame_id"].casefold() for row in rows}) != len(rows):
        raise ValueError("Test view requires distinct occurrence filenames")
    result = []
    for row in rows:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", row["frame_id"]):
            raise ValueError("Unsafe test frame identity")
        suffix = Path(row["image_path"]).suffix.lower()
        image_rel = f"images/test/{row['frame_id']}{suffix}"
        label_rel = f"labels/test/{row['frame_id']}.txt"
        image, label = destination / image_rel, destination / label_rel
        image.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        ip, lp = Path(row["image_path"]), Path(row["label_path"])
        if source_root:
            ip, lp = source_root / ip, source_root / lp
        if (
            file_sha256(ip) != row["image_sha256"]
            or file_sha256(lp) != row["label_sha256"]
        ):
            raise ValueError("Test source copy checksum mismatch")
        raw = lp.read_bytes()
        labels(raw)
        if selected_classes is not None:
            lines = [
                line
                for line in raw.decode("utf-8-sig").splitlines()
                if line.strip() and int(float(line.split()[0])) in selected_classes
            ]
            raw = ("\n".join(lines) + ("\n" if lines else "")).encode()
        shutil.copyfile(ip, image)
        label.write_bytes(raw)
        counts, _ = labels(raw)
        result.append(
            {
                **row,
                "image_path": image_rel,
                "label_path": label_rel,
                "label_sha256": file_sha256(label),
                "num_objects": sum(counts),
                **{f"class_{i}": n for i, n in enumerate(counts)},
            }
        )
    # ./ entries are resolved relative to the list by pinned Ultralytics.
    (destination / "test.txt").write_text(
        "".join(f"./{r['image_path']}\n" for r in result), encoding="utf-8"
    )
    config = {
        "train": "test.txt",
        "val": "test.txt",
        "test": "test.txt",
        "nc": 5,
        "names": NAMES,
    }
    (destination / "dataset.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    write_csv(destination / "records.csv", result)
    return result


def materialize(publication: Path, source: Path, output: Path):
    test, label_rows, provenance = publication_context(publication, source)
    with transaction(output, publication, source) as stage:
        records = []
        for row in test.to_dict("records"):
            if not re.fullmatch(r"[A-Za-z0-9_-]+", row["frame_id"]):
                raise ValueError("Unsafe frozen frame identity")
            image = f"images/test/{row['frame_id']}{Path(row['zip_member_name']).suffix.lower()}"
            label = f"labels/test/{row['frame_id']}.txt"
            for relative, raw in (
                (image, source_bytes(source, row, row["image_sha256"])),
                (
                    label,
                    source_bytes(
                        source, label_rows[row["label_ids"][0]], row["label_sha256"]
                    ),
                ),
            ):
                target = stage / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(raw)
            counts, boundary = labels((stage / label).read_bytes())
            if sum(counts) != row["num_objects"]:
                raise ValueError("Frozen test annotation support differs")
            records.append(
                {
                    "frame_id": row["frame_id"],
                    "content_id": row["content_id"],
                    "image_sha256": row["image_sha256"],
                    "label_sha256": row["label_sha256"],
                    "image_path": image,
                    "label_path": label,
                    "num_objects": sum(counts),
                    "boundary_overflow": boundary,
                    **{f"class_{i}": n for i, n in enumerate(counts)},
                }
            )
        write_csv(stage / "records.csv", records)
        test.to_parquet(stage / "source_records.parquet", index=False)
        (stage / "test.txt").write_text(
            "".join(f"./{r['image_path']}\n" for r in records), encoding="utf-8"
        )
        (stage / "dataset.yaml").write_text(
            yaml.safe_dump(
                {
                    "train": "test.txt",
                    "val": "test.txt",
                    "test": "test.txt",
                    "nc": 5,
                    "names": NAMES,
                }
            ),
            encoding="utf-8",
        )
        provenance.update(
            state="MATERIALIZED_NOT_RELEASED",
            records=len(records),
            class_counts={
                str(i): sum(r[f"class_{i}"] for r in records) for i in range(5)
            },
            evaluated=False,
            independent_test=False,
            limits=[
                "Internal retrospective protected test; no new scientific release",
                "Development overlap audit must precede evaluation",
                "YAML train/val aliases satisfy loader schema; this view is forbidden for training",
            ],
        )
        write_json(stage / "provenance.json", provenance)
        seal(stage, {"state": "MATERIALIZED_NOT_RELEASED", "source": provenance})
    return provenance


def development_records(dataset_yaml: Path, manifest: pd.DataFrame):
    """Resolve actual train/val lists and bind both image and label to source occurrence."""
    cfg = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    if cfg.get("test") or cfg.get("nc") != 5 or cfg.get("names") != NAMES:
        raise ValueError("Expected original five-class development YAML without test")
    root = Path(cfg.get("path") or dataset_yaml.parent)
    if not root.is_absolute():
        root = dataset_yaml.parent / root
    if manifest.frame_id.isna().any() or not manifest.frame_id.is_unique:
        raise ValueError("Source frame_id is not a unique occurrence identity")
    lookup = manifest.set_index("frame_id")
    result = []
    for split in ("train", "val"):
        value = cfg[split]
        locations = value if isinstance(value, list) else [value]
        paths = []
        for location in locations:
            p = Path(location)
            p = p if p.is_absolute() else root / p
            if p.is_dir():
                paths.extend(
                    sorted(
                        x
                        for x in p.rglob("*")
                        if x.suffix.lower() in (".png", ".jpg", ".jpeg")
                    )
                )
            else:
                for line in p.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        item = Path(line.strip())
                        paths.append(item if item.is_absolute() else p.parent / item)
        if not paths:
            raise ValueError("Empty development partition")
        for image in paths:
            frame = image.stem
            if frame not in lookup.index:
                raise ValueError(f"Unmapped development origin: {frame}")
            row = lookup.loc[frame].to_dict()
            parts = list(image.parts)
            if "images" not in parts:
                raise ValueError("Cannot derive YOLO label locator")
            parts[len(parts) - 1 - parts[::-1].index("images")] = "labels"
            label = Path(*parts).with_suffix(".txt")
            if (
                file_sha256(image) != row["image_sha256"]
                or file_sha256(label) != row["label_sha256"]
            ):
                raise ValueError("Development bytes differ from original occurrence")
            labels(label.read_bytes())
            result.append(
                {
                    **row,
                    "frame_id": frame,
                    "new_split": split,
                    "image_path": str(image.resolve()),
                    "label_path": str(label.resolve()),
                }
            )
    result = pd.DataFrame(result)
    if not result.frame_id.is_unique:
        raise ValueError("Development occurrence repeated across lists")
    return result


def overlap_audit(
    test: pd.DataFrame,
    development: pd.DataFrame,
    relation_rows: list[dict],
    temporal_window_frames=60,
    protection_groups: pd.DataFrame | None = None,
):
    """Confirmed identities and heuristic temporal evidence are distinct outputs."""
    test_ids, dev_ids = set(test.frame_id), set(development.frame_id)
    exact = sorted(set(test.content_id) & set(development.content_id))
    frame_overlap = sorted(test_ids & dev_ids)
    relations = [
        r
        for r in relation_rows
        if (r.get("left_frame_id") in test_ids and r.get("right_frame_id") in dev_ids)
        or (r.get("right_frame_id") in test_ids and r.get("left_frame_id") in dev_ids)
    ]
    temporal = []
    columns = {"possible_sequence", "possible_frame_index"}
    unknown = {"test": len(test), "development": len(development)}
    if columns <= set(test) and columns <= set(development):
        a = test.dropna(subset=list(columns))
        b = development.dropna(subset=list(columns))
        a = a.loc[a.possible_sequence.astype(str).ne("")]
        b = b.loc[b.possible_sequence.astype(str).ne("")]
        unknown = {"test": len(test) - len(a), "development": len(development) - len(b)}
        joined = a[["frame_id", *sorted(columns)]].merge(
            b[["frame_id", *sorted(columns)]],
            on="possible_sequence",
            suffixes=("_test", "_dev"),
        )
        gap = (
            joined.possible_frame_index_test.astype(float)
            - joined.possible_frame_index_dev.astype(float)
        ).abs()
        temporal = joined.loc[gap <= temporal_window_frames].to_dict("records")
    group_overlap = []
    nominal_origins = []
    if "basename" in test and "basename" in development:
        left = {Path(str(v)).stem for v in test.basename.dropna() if str(v)}
        right = {Path(str(v)).stem for v in development.basename.dropna() if str(v)}
        nominal_origins = sorted(left & right)
    constrained_crossings = [
        r for r in relations if r.get("split_constraint_applied") is True
    ]
    if protection_groups is not None:
        if not {"frame_id", "group_id"} <= set(protection_groups):
            raise ValueError("Unknown protection group schema")
        group_overlap = sorted(
            set(
                protection_groups.loc[
                    protection_groups.frame_id.isin(test_ids), "group_id"
                ]
            )
            & set(
                protection_groups.loc[
                    protection_groups.frame_id.isin(dev_ids), "group_id"
                ]
            )
        )
    return {
        "frame_id_overlap": frame_overlap,
        "exact_content_overlap": exact,
        "registered_relation_crossings": relations,
        "protected_group_overlap": group_overlap,
        "applied_constraint_crossings": constrained_crossings,
        "nominal_origin_overlap": nominal_origins,
        "nominal_origin_status": "Original basename overlap is a provenance candidate, not independently verified source identity",
        "temporal_candidates": temporal,
        "temporal_window_frames": temporal_window_frames,
        "unknown_temporal_provenance": unknown,
        "temporal_status": "filename heuristic; not confirmed leakage",
        "origin_identity_status": "Known occurrence IDs checked; source-video identity may remain unknown",
        "independence_proven": False,
        "hard_identity_disjoint": not bool(
            frame_overlap or exact or group_overlap or constrained_crossings
        ),
    }


def verify_test(directory: Path):
    receipt = verify(directory)
    rows = pd.read_csv(directory / "records.csv")
    if (
        len(rows) != 372
        or not rows.frame_id.is_unique
        or receipt["state"] != "MATERIALIZED_NOT_RELEASED"
    ):
        raise ValueError("Invalid common test receipt")
    return rows
