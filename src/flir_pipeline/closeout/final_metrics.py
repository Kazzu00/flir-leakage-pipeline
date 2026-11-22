"""Lossless OE4 table projection; no model loading or metric recomputation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from decimal import Decimal, InvalidOperation
from pathlib import Path

POLICIES = ("historica", "aleatoria", "c10")
TREATMENTS = ("AUG_ON", "AUG_OFF")
COHORTS = ("five_class_macro", "shared_01_macro")
METRICS = ("precision", "recall", "map50", "map50_95")
SOURCE_COLUMNS = ("treatment", "policy", "cohort", *METRICS)
FINAL_COLUMNS = (
    "policy",
    "augmentation",
    *(f"{metric}_5class" for metric in METRICS),
    *(f"{metric}_shared_01" for metric in METRICS),
)
OFFICIAL_SHA256 = "5e690f88ed9a9d9c4f820fd65e660394c99c8ae51109156d2d3646ee15da4bb4"


def project(source: bytes, receipt: dict) -> bytes:
    """Require all 12 cells and preserve every decimal token exactly.

    Decimal is used only for range/finite validation. Original strings are written
    unchanged; binary floating-point conversion must never round source metrics.
    The reverse projection below checks every source value, independent of order.
    """
    if (
        receipt.get("state") != "AGGREGATED"
        or receipt.get("protocol") != "oe4_closeout_v2"
        or receipt.get("files", {}).get("comparison.csv")
        != hashlib.sha256(source).hexdigest()
    ):
        raise ValueError("Incompatible receipt or comparison.csv SHA-256 mismatch")
    reader = csv.DictReader(io.StringIO(source.decode("utf-8"), newline=""))
    if reader.fieldnames != list(SOURCE_COLUMNS):
        raise ValueError("Unexpected comparison.csv columns")
    cells = {}
    for row in reader:
        if None in row or any(value is None for value in row.values()):
            raise ValueError("Malformed comparison.csv row")
        key = (row["policy"], row["treatment"], row["cohort"])
        if key in cells:
            raise ValueError("Duplicate experimental cell")
        for metric in METRICS:
            try:
                value = Decimal(row[metric])
            except InvalidOperation as exc:
                raise ValueError("Invalid metric") from exc
            if not value.is_finite() or not 0 <= value <= 1:
                raise ValueError("Metric outside finite [0, 1]")
        cells[key] = row
    expected = {(p, t, c) for p in POLICIES for t in TREATMENTS for c in COHORTS}
    if set(cells) != expected:
        raise ValueError("Expected exactly the 12 policy/treatment/cohort cells")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=FINAL_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for policy in POLICIES:
        for treatment in TREATMENTS:
            record = {"policy": policy, "augmentation": treatment}
            for cohort, suffix in zip(COHORTS, ("5class", "shared_01"), strict=True):
                record.update(
                    (f"{metric}_{suffix}", cells[policy, treatment, cohort][metric])
                    for metric in METRICS
                )
            writer.writerow(record)
    final = output.getvalue()
    restored = {}
    for row in csv.DictReader(io.StringIO(final)):
        for cohort, suffix in zip(COHORTS, ("5class", "shared_01"), strict=True):
            restored[row["policy"], row["augmentation"], cohort] = {
                metric: row[f"{metric}_{suffix}"] for metric in METRICS
            }
    if restored != {key: {m: row[m] for m in METRICS} for key, row in cells.items()}:
        raise ValueError("Lossless reverse projection failed")
    return final.encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparison", type=Path)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--check", action="store_true", help="Check existing output bytes")
    args = parser.parse_args()
    source = args.comparison.read_bytes()
    if hashlib.sha256(source).hexdigest() != OFFICIAL_SHA256:
        raise ValueError("Source differs from the official 2026-11-22 CSV")
    final = project(source, json.loads(args.receipt.read_bytes()))
    if args.check:
        if args.output.read_bytes() != final:
            raise ValueError("Final CSV differs from the exact projection")
    else:
        with args.output.open("xb") as stream:
            stream.write(final)
    print("Verified: 12 source rows -> 6 rows, 8 metrics; decimal tokens unchanged")


if __name__ == "__main__":
    main()
