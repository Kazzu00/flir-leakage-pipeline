"""Hash-checked generated YOLO views retaining occurrence-specific annotations."""

from __future__ import annotations

import hashlib
import re
import zipfile
from contextlib import ExitStack
from pathlib import Path

import pandas as pd
import yaml

from flir_pipeline.data.classes import DETECTION_CLASSES, class_name
from flir_pipeline.data.identity import dataset_id_from_manifest
from flir_pipeline.similarity.storage import file_sha256, read_json, write_json
from flir_pipeline.splitting.storage import verify_split

SPLITS = ("train", "val", "test")


def plan_records(manifest: pd.DataFrame, assignments: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """A content can have conflicting historical annotations; never collapse records."""
    if manifest.frame_id.duplicated().any() or assignments.frame_id.duplicated().any():
        raise ValueError("Duplicate frame identity")
    if set(assignments.frame_id) != set(manifest.frame_id) or not set(assignments.new_split) <= set(SPLITS):
        raise ValueError("Split coverage or membership invalid")
    joined = manifest.merge(assignments[["frame_id", "content_id", "new_split"]], on=["frame_id", "content_id"], validate="one_to_one")
    if len(joined) != len(manifest):
        raise ValueError("Content identity mismatch")
    if strategy == "historical":
        if not (joined.new_split == joined.original_split).all():
            raise ValueError("Historical membership was altered")
    elif joined.groupby("content_id").new_split.nunique().max() != 1:
        raise ValueError("Exact content crosses a new split")
    if not joined.label_exists.all() or not joined.label_valid.all() or not joined.image_decode_valid.all():
        raise ValueError("Missing/invalid image or label must be resolved explicitly")
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", str(v)) for v in joined.frame_id):
        raise ValueError("Unsafe occurrence identifier")
    return joined.sort_values("frame_id").reset_index(drop=True)


def dataset_yaml(root: Path, directory: Path) -> dict:
    return {"path": str(root.resolve()), **{s: str((directory/f"{s}.txt").resolve()) for s in SPLITS},
            "nc": len(DETECTION_CLASSES), "names": {i: class_name(i) for i in DETECTION_CLASSES}}


def _write_checked(path: Path, content: bytes, expected: str) -> None:
    if hashlib.sha256(content).hexdigest() != expected:
        raise ValueError("Source member checksum differs from canonical manifest")
    if path.exists():
        if file_sha256(path) != expected:
            raise ValueError("Generated content changed; refusing silent replacement")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(content)


def materialize(split_directory: Path, manifest_path: Path, source_root: Path, output: Path) -> Path:
    """Extract only canonical members into an ignored generated store; ZIPs remain read-only.

    Each frame gets its own filename and original annotation bytes. All views reuse
    that immutable store through lists, so 16 splits do not create 16 image copies.
    """
    if not verify_split(split_directory)["quality_valid"]:
        raise ValueError("Invalid source split")
    meta = read_json(split_directory/"metadata.json")
    identity = meta["identity_payload"]
    manifest = pd.read_parquet(manifest_path)
    if dataset_id_from_manifest(manifest) != identity["dataset_id"] or file_sha256(manifest_path) != meta["input_signatures"]["manifest"]:
        raise ValueError("Manifest does not match frozen split source")
    assignments = pd.read_parquet(split_directory/"record_split_assignments.parquet")
    records = plan_records(manifest, assignments, identity["configuration"]["strategy"])
    store = output/"data"/identity["dataset_id"]
    view = output/"views"/meta["split_space_id"]
    # Even resume validates existing files: a receipt is not proof of unchanged bytes.
    archives = {}
    with ExitStack() as stack:
        for name in sorted(set(records.source_archive) | {"Etiquetas.zip"}):
            path = (source_root/name).resolve()
            if not path.is_relative_to(source_root.resolve()) or path.suffix.lower() != ".zip":
                raise ValueError("Invalid archive path")
            archives[name] = stack.enter_context(zipfile.ZipFile(path, "r"))
            if len(archives[name].namelist()) != len(set(archives[name].namelist())):
                raise ValueError("Ambiguous duplicate archive member")
        image_paths = []
        for row in records.itertuples():
            suffix = Path(row.source_member_path).suffix.lower()
            if suffix not in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"):
                raise ValueError("Unsupported image format")
            image = store/"images"/f"{row.frame_id}{suffix}"
            label = store/"labels"/f"{row.frame_id}.txt"
            _write_checked(image, archives[row.source_archive].read(row.source_member_path), row.image_sha256)
            _write_checked(label, archives["Etiquetas.zip"].read(row.relative_label_path), row.label_sha256)
            if (not label.read_text(encoding="utf-8-sig").strip()) != bool(row.label_empty):
                raise ValueError("Empty-label interpretation changed")
            image_paths.append(str(image.resolve()))
    records["image_path"] = image_paths
    view.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        values = records.loc[records.new_split == split, "image_path"].tolist()
        if not values:
            raise ValueError("Empty dataset partition")
        content = "\n".join(values)+"\n"
        path = view/f"{split}.txt"
        if path.exists() and path.read_text(encoding="utf-8") != content:
            raise ValueError("Existing materialized membership differs")
        path.write_text(content, encoding="utf-8")
    (view/"dataset.yaml").write_text(yaml.safe_dump(dataset_yaml(store, view), sort_keys=False, allow_unicode=True), encoding="utf-8")
    records[["frame_id", "content_id", "new_split", "image_sha256", "label_sha256", "label_empty", "num_objects", "image_path"]].to_parquet(view/"records.parquet", index=False)
    receipt = {"dataset_id": identity["dataset_id"], "split_space_id": meta["split_space_id"],
               "split_metadata_sha256": file_sha256(split_directory/"metadata.json"),
               "manifest_sha256": file_sha256(manifest_path), "counts": records.groupby("new_split").size().to_dict(),
               "empty_labels": int(records.label_empty.sum()), "objects": int(records.num_objects.sum()),
               "source_archives_read_only": True, "original_bytes_preserved": True,
               "files": {p.name: file_sha256(p) for p in view.iterdir() if p.name in ("dataset.yaml", "records.parquet", "train.txt", "val.txt", "test.txt")}}
    write_json(view/"materialization.json", receipt)
    verify_view(view)
    return view


def verify_view(view: Path) -> dict:
    receipt = read_json(view/"materialization.json")
    required = {"dataset.yaml", "records.parquet", "train.txt", "val.txt", "test.txt"}
    if set(receipt["files"]) != required or any(file_sha256(view/n) != h for n, h in receipt["files"].items()):
        raise ValueError("Materialized view checksum mismatch")
    config = yaml.safe_load((view/"dataset.yaml").read_text(encoding="utf-8"))
    if config["names"] != {i: class_name(i) for i in DETECTION_CLASSES} or config["nc"] != 5:
        raise ValueError("Canonical class mapping changed")
    records = pd.read_parquet(view/"records.parquet")
    if records.frame_id.duplicated().any() or records.image_path.duplicated().any():
        raise ValueError("Materialized occurrence collision")
    for row in records.itertuples():
        image = Path(row.image_path)
        label = image.parent.parent/"labels"/f"{image.stem}.txt"
        if image.stem != row.frame_id or file_sha256(image) != row.image_sha256 or file_sha256(label) != row.label_sha256:
            raise ValueError("Materialized image/label bytes changed")
    for split in SPLITS:
        paths = (view/f"{split}.txt").read_text(encoding="utf-8").splitlines()
        expected = records.loc[records.new_split == split, "image_path"].tolist()
        if paths != expected or len(paths) != receipt["counts"][split] or Path(config[split]).resolve() != (view/f"{split}.txt").resolve():
            raise ValueError("Materialized split membership/count changed")
    return receipt
