"""Publish an allowlisted projection of recovered native OE4 receipts, not payloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from flir_pipeline.closeout.final_metrics import (
    OFFICIAL_SHA256,
    POLICIES,
    TREATMENTS,
    project,
)

HASH_KEYS = (
    "authorization_sha256",
    "phase_authorization_receipt_sha256",
    "freeze_sha256",
    "training_receipt_sha256",
    "weights_sha256",
    "test_receipt_sha256",
    "test_records_sha256",
)
EVALUATION = {
    "imgsz": 640, "batch": 1, "conf": 0.001, "iou": 0.7, "max_det": 300,
    "half": False, "rect": False, "augment": False, "agnostic_nms": False,
    "classes": None, "workers": 0,
}
AGGREGATE_FILES = ("metrics.json", "five_classes.csv", "shared_01.csv")


def public_receipts(originals: dict[tuple[str, str], bytes], aggregate: dict) -> bytes:
    """Check metadata consistency without claiming checkpoint/payload validation.

    Native `files` maps contain private per-image identities. Only three aggregate
    file bindings are retained. Every original receipt stays external and is bound
    by its observed byte hash; SHA-256 integrity is not an authenticity signature.
    """
    expected = {(p, t) for p in POLICIES for t in TREATMENTS}
    if set(originals) != expected:
        raise ValueError("Exactly six native receipts required")
    if aggregate.get("state") != "AGGREGATED" or aggregate.get("protocol") != "oe4_closeout_v2":
        raise ValueError("Incompatible aggregate receipt")
    common, rows = None, []
    for policy in POLICIES:
        for treatment in TREATMENTS:
            raw = originals[policy, treatment]
            receipt = json.loads(raw)
            if (
                receipt.get("state") != "COMPLETED_EVALUATION"
                or receipt.get("protocol") != aggregate["protocol"]
                or receipt.get("policy") != policy
                or receipt.get("treatment") != treatment
                or receipt.get("split") != "test"
                or receipt.get("checkpoint_selection") != "last_epoch_50"
                or receipt.get("test_tuning") is not False
                or receipt.get("evaluation") != EVALUATION
            ):
                raise ValueError("Incompatible native evaluation metadata")
            bindings = {name: receipt.get("files", {}).get(name) for name in AGGREGATE_FILES}
            for value in [*(receipt.get(k) for k in HASH_KEYS), *bindings.values()]:
                if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                    raise ValueError("Missing or malformed native SHA-256")
            if [receipt[k] for k in ("test_receipt_sha256", "test_records_sha256")] != aggregate.get("test"):
                raise ValueError("Native test identity differs from official aggregate")
            identity = tuple(receipt[k] for k in HASH_KEYS[:3])
            if common is not None and identity != common:
                raise ValueError("Native freeze or authorization identities differ")
            common = identity
            environment = receipt.get("environment", {})
            if environment.get("ultralytics") != "8.3.203" or str(environment.get("torch")).split("+")[0] != "2.7.1":
                raise ValueError("Native pinned evaluator versions differ")
            metadata = {key: receipt[key] for key in (
                "state", "protocol", "policy", "treatment", *HASH_KEYS,
                "evaluation", "split", "checkpoint_selection", "test_tuning",
            )}
            # Never copy unknown metadata/environment fields into the public output.
            metadata["environment"] = {key: environment[key] for key in (
                "ultralytics", "torch", "cuda", "gpu", "slurm_job_id",
            )}
            rows.append({
                "source_relative_path": f"{treatment}/{policy}/receipt.json",
                "native_receipt_sha256": hashlib.sha256(raw).hexdigest(),
                "metadata": metadata,
                "aggregate_file_sha256": bindings,
            })
    return (json.dumps({
        "schema_version": "oe4_public_evaluation_receipts_v1",
        "verification_scope": "Six native receipt metadata bindings checked against the official aggregate; no checkpoint or bound payload rehash",
        "checkpoint_bytes_verified_locally": False,
        "bound_payloads_verified_locally": False,
        "original_receipts_preserved_externally": True,
        "receipts": rows,
    }, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="Directory containing the official CSV and aggregate receipt")
    parser.add_argument("evaluations", type=Path, help="External directory containing six native receipts")
    parser.add_argument("output", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    source = (args.results / "comparison.csv").read_bytes()
    if hashlib.sha256(source).hexdigest() != OFFICIAL_SHA256:
        raise ValueError("Source differs from the official CSV")
    aggregate = json.loads((args.results / "receipt.json").read_bytes())
    project(source, aggregate)
    expected = {f"{t}/{p}/receipt.json" for p in POLICIES for t in TREATMENTS}
    actual = {p.relative_to(args.evaluations).as_posix() for p in args.evaluations.glob("*/*/receipt.json")}
    if actual != expected:
        raise ValueError("Native receipt directory must contain exactly six expected cells")
    output = public_receipts({
        (p, t): (args.evaluations / t / p / "receipt.json").read_bytes()
        for p in POLICIES for t in TREATMENTS
    }, aggregate)
    if args.check:
        if args.output.read_bytes() != output:
            raise ValueError("Public receipt projection differs from recovered originals")
    else:
        with args.output.open("xb") as stream:
            stream.write(output)
    print("Verified: six native receipts, checkpoint hashes and common test/freeze metadata; payloads not rehashed")


if __name__ == "__main__":
    main()
