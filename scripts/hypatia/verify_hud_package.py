"""Verify the curated transfer bytes; integrity never grants training admission.

Works with the Python standard library on the login node, without CUDA or models.
The receipt SHA256 must come from the independently transferred checksum listing.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(root, expected_receipt):
    root = Path(root).resolve()
    receipt_path = root / "development_receipt.json"
    if sha256(receipt_path) != expected_receipt:
        raise ValueError("Receipt checksum mismatch")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt["schema_version"] != "curated_development_v1":
        raise ValueError("Unsupported package")
    artifacts = receipt["artifacts"]
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("Package contains links")
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    if actual != set(artifacts) | {"development_receipt.json"}:
        raise ValueError("Missing or additional package files")
    for name, expected in artifacts.items():
        parts = PurePosixPath(name)
        if parts.is_absolute() or ".." in parts.parts or "\\" in name or ":" in name:
            raise ValueError("Unsafe receipt path")
        if sha256(root / name) != expected:
            raise ValueError(f"Artifact checksum mismatch: {name}")
    with (root / "manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != receipt["canonical_contents"]:
        raise ValueError("Population mismatch")
    return {
        "integrity": "PASS",
        "canonical_contents": len(rows),
        "counts": receipt["counts"],
        "empty_labels": sum(r["label_empty"] == "True" for r in rows),
        "protected_exclusion": receipt["protected_exclusion"],
        "ready_for_training": False,
        "state": "DEVELOPMENT_NOT_ADMITTED",
        "limitation": "Checksums verify transfer, not visual annotation correctness",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("--receipt-sha256", required=True)
    parser.add_argument("--require-training-ready", action="store_true")
    args = parser.parse_args()
    result = verify(args.package, args.receipt_sha256)
    print(json.dumps(result, indent=2))
    if args.require_training_ready:
        parser.exit(2, "BLOCKED: curated preparation is not an admitted training dataset\n")


if __name__ == "__main__":
    main()
