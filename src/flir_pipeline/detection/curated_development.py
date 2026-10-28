"""Manifest-driven development views, deliberately outside scientific admission.

No split solver, protected payload reader, approval transition or training entry
point belongs here. A local receipt establishes integrity, never authority.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from pathlib import Path, PurePosixPath

import pandas as pd
import yaml
from PIL import Image

from flir_pipeline.data.yolo_labels import validate_label_bytes
from flir_pipeline.detection.common_executor import NAMES, external_output
from flir_pipeline.detection.materialization import _write_checked
from flir_pipeline.similarity.storage import file_sha256, read_json

VERSION = "curated_development_v1"
STATE = "DEVELOPMENT_NOT_ADMITTED"
POLICIES = ("S1", "S2", "S3")


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def encoded(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def source_path(root: Path, value: str) -> Path:
    """Reject platform-dependent paths and links before reading source bytes."""
    parts = PurePosixPath(value)
    if (not value or "\\" in value or ":" in value or parts.is_absolute()
            or any(p in (".", "..", "") for p in value.split("/"))):
        raise ValueError("Unsafe relative source path")
    path = root.joinpath(*parts.parts)
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Source links are not supported")
    if not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("Source missing or outside dataset")
    return path


def protected_check(rows: list[dict], records: Path | None, receipt: Path | None) -> dict:
    if records is None and receipt is None:
        return {"status": "PENDING_REFERENCE_MISSING", "test_status": "NOT_APPROVED"}
    if records is None or receipt is None:
        raise ValueError("Protected records and publication receipt required together")
    relative = records.resolve().relative_to(receipt.parent.resolve()).as_posix()
    binding = file_sha256(records)
    if read_json(receipt)["artifacts"].get(relative) != binding:
        raise ValueError("Protected reference checksum mismatch")
    protected = pd.read_parquet(records, columns=["frame_id", "content_id", "image_sha256"])
    if protected.empty or protected.isna().any().any() or not protected.frame_id.is_unique:
        raise ValueError("Invalid protected reference")
    for key in ("frame_id", "content_id", "image_sha256"):
        if not protected[key].map(lambda x: bool(re.fullmatch(r"[a-f0-9]{64}", str(x)))).all():
            raise ValueError("Invalid protected identity")
        if {r[key] for r in rows} & set(protected[key]):
            raise ValueError("Protected overlap")
    return {"status": "EXACT_EXCLUSION_VERIFIED_AGAINST_REFERENCE",
            "test_status": "NOT_APPROVED", "reference_sha256": binding,
            "receipt_sha256": file_sha256(receipt), "occurrences": len(protected),
            "contents": protected.content_id.nunique(), "overlap": 0,
            "scope": "supplied metadata only; completeness, authority and OS isolation unverified"}


def prepare_curated(
    source_root: Path, manifest: Path, matrix: Path, provenance: Path,
    output: Path, linux_root: str, protected_records: Path | None = None,
    protected_receipt: Path | None = None,
) -> dict:
    """Reconstruct the exact expected tree; repeated calls verify without writing.

    Linux locators are explicit and frozen separately from population identity.
    Interrupted outputs are retained and rejected, never repaired silently.
    """
    sources = [source_root, manifest, matrix, provenance]
    sources += [p for p in (protected_records, protected_receipt) if p is not None]
    external_output(output, *sources)
    if any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError("Output links are not supported")
    runtime = PurePosixPath(linux_root)
    if (not runtime.is_absolute() or str(runtime) != linux_root or linux_root == "/"
            or any(c in linux_root for c in ("\\", ":", "\n", "\r"))
            or ".." in runtime.parts):
        raise ValueError("Explicit normalized absolute Linux output root required")
    raw = manifest.read_bytes()
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
    required = {"frame_id", "content_id", "image_sha256", "label_sha256", "image_path",
                "label_path", "width", "height", "num_objects", "label_empty",
                *(f"{p}_split" for p in POLICIES)}
    if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
        raise ValueError("Invalid manifest header")
    if not required <= set(reader.fieldnames):
        raise ValueError("Missing manifest columns")
    rows = list(reader)
    if not rows or any(None in r or None in r.values() for r in rows):
        raise ValueError("Empty or malformed manifest")
    for key in ("frame_id", "content_id", "image_sha256", "image_path", "label_path"):
        if len({r[key] for r in rows}) != len(rows):
            raise ValueError(f"Duplicate canonical {key}")
    for row in rows:
        if any(not re.fullmatch(r"[a-f0-9]{64}", row[k])
               for k in ("frame_id", "content_id", "image_sha256", "label_sha256")):
            raise ValueError("Invalid identity/hash")
        if row["content_id"] != row["image_sha256"]:
            raise ValueError("Canonical content identity must equal image SHA256")
    proposal, origin = read_json(matrix), read_json(provenance)
    if origin.get("source_manifest_sha256") != digest(raw):
        raise ValueError("Provenance manifest binding mismatch")
    expected = {(p, 0, s, f"{p}_seed0/data.yaml") for p in POLICIES for s in (42, 43, 44)}
    runs = proposal.get("runs", [])
    if (proposal.get("state") != "PROPOSED_NOT_ADMITTED_NOT_SUBMITTED" or len(runs) != 9
            or {(r["policy"], r["split_seed"], r["detector_seed"], r["data_yaml"])
                for r in runs} != expected):
        raise ValueError("Invalid proposed matrix")
    protection = protected_check(rows, protected_records, protected_receipt)
    # Small metadata is held in memory; image payloads are streamed one at a time.
    metadata = {"manifest.csv": raw, "matrix.json": matrix.read_bytes(),
                "provenance.json": provenance.read_bytes()}
    copies = {}
    counts = {}
    for row in rows:
        image = source_path(source_root, row["image_path"])
        label = source_path(source_root, row["label_path"])
        if image.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"):
            raise ValueError("Unsupported image extension")
        if label.suffix != ".txt" or image.stem != label.stem:
            raise ValueError("Image/label pairing mismatch")
        if file_sha256(image) != row["image_sha256"] or file_sha256(label) != row["label_sha256"]:
            raise ValueError("Source checksum mismatch")
        qa = validate_label_bytes(label.read_bytes())
        if (not qa.label_valid or not set(qa.classes_present) <= set(NAMES)
                or len(set(qa.boxes)) != len(qa.boxes)
                or qa.num_objects != int(row["num_objects"])
                or row["label_empty"] != str(qa.label_empty)):
            raise ValueError("Annotation validation/metadata mismatch")
        with Image.open(image) as decoded:
            decoded.load()
            if decoded.size != (int(row["width"]), int(row["height"])) or min(decoded.size) < 10:
                raise ValueError("Image dimension mismatch")
        name = row["frame_id"]
        row["prepared_image"] = f"canonical/images/{name}{image.suffix.lower()}"
        copies[row["prepared_image"]] = (image, row["image_sha256"])
        copies[f"canonical/labels/{name}.txt"] = (label, row["label_sha256"])
    for policy in POLICIES:
        key = f"{policy}_split"
        if {r[key] for r in rows} != {"train", "val"}:
            raise ValueError("Both development partitions required; test forbidden")
        counts[policy] = {}
        config = {"path": str(runtime / f"{policy}_seed0"), "nc": 5, "names": NAMES}
        for split in ("train", "val"):
            selected = sorted((r for r in rows if r[key] == split), key=lambda r: r["frame_id"])
            counts[policy][split] = len(selected)
            name = f"{policy}_seed0/{split}.txt"
            metadata[name] = ("\n".join(str(runtime / r["prepared_image"]) for r in selected) + "\n").encode()
            config[split] = str(runtime / name)
        metadata[f"{policy}_seed0/data.yaml"] = yaml.safe_dump(config, sort_keys=True).encode()
    bindings = {"manifest_sha256": digest(raw), "matrix_sha256": digest(metadata["matrix.json"]),
                "provenance_sha256": digest(metadata["provenance.json"])}
    identity = digest(encoded({"schema_version": VERSION, "manifest_sha256": digest(raw)}))
    artifacts = {name: sha for name, (_, sha) in copies.items()}
    artifacts.update({name: digest(value) for name, value in metadata.items()})
    result = {"schema_version": VERSION, "state": STATE, "ready_for_training": False,
              "curated_population_id": identity, "inputs": bindings, "counts": counts,
              "canonical_contents": len(rows), "linux_root": linux_root,
              "protected_exclusion": protection, "artifacts": artifacts,
              "implementation_sha256": file_sha256(Path(__file__)),
              "blockers": ["curated_scientific_contract_not_implemented",
                           "signed_decisions_and_test_approval_required",
                           "worker_test_isolation_not_verified"]}
    metadata["development_receipt.json"] = encoded(result)
    expected_files = {**artifacts, "development_receipt.json": digest(metadata["development_receipt.json"])}
    if output.exists():
        entries = list(output.rglob("*"))
        if any(p.is_symlink() for p in entries):
            raise ValueError("Output links are not supported")
        actual = {p.relative_to(output).as_posix(): file_sha256(p) for p in entries if p.is_file()}
        if actual != expected_files:
            raise ValueError("Existing output differs or is partial; preserve and investigate")
        return result
    output.mkdir(parents=True, exist_ok=False)
    for name, (path, sha) in copies.items():
        _write_checked(output / name, path.read_bytes(), sha)
    for name, value in metadata.items():
        _write_checked(output / name, value, digest(value))
    return result
