"""Audit denoised-to-protected-test correspondence without promoting names to truth."""

from pathlib import Path

import pandas as pd

from flir_pipeline.closeout.io import seal, transaction, write_csv, write_json
from flir_pipeline.closeout.splits import load_manifest
from flir_pipeline.similarity.storage import file_sha256
from flir_pipeline.splitting.fixed_test_storage import verify_fixed_test


def audit(dataset: Path, publication: Path, output: Path):
    rows = load_manifest(dataset)
    verify_fixed_test(publication)
    source = pd.read_parquet(publication / "sources/audit/candidate_manifest.parquet")
    protected = pd.read_parquet(publication / "sources/proposal/test_records.parquet")
    if len(protected) != 372 or not protected.frame_id.is_unique:
        raise ValueError("Protected test identity invalid")
    by_name = {}
    for row in source.to_dict("records"):
        # Native audit stores basename without extension. Strip only the image
        # extension, never video prefixes or processing suffixes to force a match.
        name = str(row["basename"])
        key = (
            Path(name).stem
            if Path(name).suffix.lower() in {".png", ".jpg", ".jpeg"}
            else name
        )
        by_name.setdefault(key, []).append(row["frame_id"])
    test_ids = set(protected.frame_id)
    exceptions = []
    for row in rows.to_dict("records"):
        possible = sorted(by_name.get(Path(row["nombre_original"]).stem, []))
        status = (
            "UNMAPPED"
            if not possible
            else "AMBIGUOUS"
            if len(possible) > 1
            else "NOMINAL_ONLY"
        )
        exceptions.append(
            {
                "denoised_frame_id": row["frame_id"],
                "nombre_original": row["nombre_original"],
                "dataset_origen": row["dataset_origen"],
                "source_frame_key": row["source_frame_key"],
                "status": status,
                "candidate_frame_ids": ";".join(possible),
                "candidate_test_ids": ";".join(sorted(test_ids & set(possible))),
                "verified_anchor": False,
                "reason": "Original source pixels and occurrence-specific provenance binding are not supplied by matching basename; historical split is not evidence",
            }
        )
    report = {
        "state": "BLOCKED_PROTECTED_FREEZE",
        "verified_anchors": 0,
        "records": len(rows),
        "status_counts": pd.Series([r["status"] for r in exceptions])
        .value_counts()
        .to_dict(),
        "nominal_test_candidates": sum(
            bool(r["candidate_test_ids"]) for r in exceptions
        ),
        "source_dataset_receipt_sha256": file_sha256(dataset / "receipt.json"),
        "publication_receipt_sha256": file_sha256(publication / "receipt.json"),
        "protected_records_sha256": file_sha256(
            publication / "sources/proposal/test_records.parquet"
        ),
        "historical_test_reused": False,
        "new_assignments_created": False,
    }
    with transaction(output, dataset, publication) as stage:
        write_csv(stage / "correspondence_exceptions.csv", exceptions)
        write_json(stage / "audit.json", report)
        seal(stage, {"state": report["state"]})
    return report
