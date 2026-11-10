"""Versioned technical admission of original occurrences, without human assertions.

Receipts establish integrity and reproducibility, not annotation correctness or
independence of frames. Historical signed contracts remain readable separately.
"""

from __future__ import annotations

import base64
import hashlib
import urllib.request
from pathlib import Path

import pandas as pd

from flir_pipeline.data.yolo_labels import validate_label_bytes
from flir_pipeline.detection.admission import bound_file
from flir_pipeline.detection.common_executor import (
    CLASS_DISCREPANCY,
    external_output,
    load_executor_config,
    load_publication,
    plan_runs,
    records_identity,
    verify_executor_plan,
    verify_preparation,
)
from flir_pipeline.similarity.storage import file_sha256, read_json, stable_id
from flir_pipeline.splitting.labeled import load_labeled_audit, verify_original_sources

POLICY = "original_occurrences_technical_v1"
OFFICIAL_URL = (
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt"
)
LIMITATIONS = [
    "Original occurrence annotations retained, including divergent exact copies",
    "Annotation completeness and semantics are not established by technical checks",
    CLASS_DISCREPANCY,
    "Common test is an internal evaluation; visual/temporal independence unproven",
    "Partition differences are descriptive, not an identified causal leakage effect",
]


def binding(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": file_sha256(path)}


def emit(path: Path, payload: dict) -> dict:
    external_output(path)
    if path.exists():
        raise ValueError("Immutable receipt already exists")
    result = {**payload, "receipt_id": stable_id(payload)}
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents two concurrent admissions overwriting evidence.
    with path.open("x", encoding="utf-8") as stream:
        import json

        json.dump(result, stream, indent=2, sort_keys=True)
    return result


def receipt(path: Path, kind: str) -> dict:
    result = read_json(path)
    payload = {k: v for k, v in result.items() if k != "receipt_id"}
    if (
        result.get("receipt_id") != stable_id(payload)
        or result.get("policy") != POLICY
        or result.get("kind") != kind
    ):
        raise ValueError("Technical receipt identity/policy mismatch")
    return result


def verify_official_weights(checkpoint: Path, output: Path) -> dict:
    """Explicit online provenance step; stream the official release, never unpickle.

    An arbitrary user-supplied hash or a URL declaration is insufficient. Compare the
    local bytes with bytes obtained through HTTPS from the fixed upstream release.
    This operation is never called by training or CI and writes no model download.
    """
    external_output(output, checkpoint)
    if output.exists():
        raise ValueError("Weights receipt exists")
    digest = hashlib.sha256()
    size = 0
    with urllib.request.urlopen(OFFICIAL_URL, timeout=60) as response:
        if not response.geturl().startswith("https://"):
            raise ValueError("Official asset transport must remain HTTPS")
        while block := response.read(1024 * 1024):
            digest.update(block)
            size += len(block)
    if not size or digest.hexdigest() != file_sha256(checkpoint):
        raise ValueError("Local checkpoint differs from official YOLO11m release")
    return emit(
        output,
        {
            "policy": POLICY,
            "kind": "official_weights",
            "url": OFFICIAL_URL,
            "method": "https_release_byte_comparison",
            "bytes": size,
            "weights": binding(checkpoint),
        },
    )


def weights_receipt(path: Path) -> dict:
    result = receipt(path, "official_weights")
    checkpoint = bound_file(result["weights"])
    if (
        result["url"] != OFFICIAL_URL
        or result["method"] != "https_release_byte_comparison"
        or checkpoint.stat().st_size != result["bytes"]
    ):
        raise ValueError("Official checkpoint provenance mismatch")
    return result


def check_memberships(summary: dict, assignments: dict, test: pd.DataFrame) -> dict:
    expected = {
        (p, s, n)
        for p in ("historical", "expanded")
        for s in ("S1", "S2", "S3")
        for n in range(5)
    }
    arms = summary["runs"]
    if (
        len(arms) != 30
        or {(r["population"], r["policy"], r["seed"]) for r in arms} != expected
    ):
        raise ValueError("Incomplete or repeated population/strategy/seed matrix")
    if len(test) != 372 or not test.frame_id.is_unique:
        raise ValueError("Expected 372 common test occurrences")
    quarantine = set().union(
        *(
            set(rows.loc[rows.new_split == "quarantine", "frame_id"])
            for rows in assignments.values()
        )
    )
    if len(quarantine) != 142:
        raise ValueError("Global quarantine must retain 142 occurrences")
    protected_hashes = set(test.image_sha256)
    protected_contents = set(test.content_id)
    for rows in assignments.values():
        q = set(rows.loc[rows.new_split == "quarantine", "frame_id"])
        if q != quarantine & set(rows.frame_id):
            raise ValueError(
                "Global quarantine occurrence assigned to an active partition"
            )
        development = rows.loc[rows.new_split.isin(["train", "val"])]
        if (
            set(development.image_sha256) & protected_hashes
            or set(development.content_id) & protected_contents
            or set(development.frame_id) & (quarantine | set(test.frame_id))
        ):
            raise ValueError("Protected occurrence/content exposed to development")
        if development.groupby("image_sha256").new_split.nunique().gt(1).any():
            raise ValueError("Exact image bytes cross train/validation")
    return {
        "test_occurrences": len(test),
        "test_id": records_identity(test),
        "quarantine_occurrences": len(quarantine or []),
        "arms": len(arms),
    }


def plan_contract(plan: dict) -> dict:
    """Compare executable semantics while retaining legacy receipt verification.

    Historical advisory blockers changed between executor versions. The new policy
    supersedes their text, not identities, membership, seeds or configuration. The
    original plan_id and every original byte are still checked by verify_executor_plan
    and bound in the data receipt; no historical document is rewritten here.
    """
    return {
        **{k: v for k, v in plan.items() if k not in ("plan_id", "runs")},
        "runs": [
            {k: v for k, v in run.items() if k != "blockers"} for run in plan["runs"]
        ],
    }


def inspect_inputs(
    publication: Path,
    plan_path: Path,
    prepared: Path,
    config_path: Path,
    source_root: Path,
    evidence: Path | None,
) -> dict:
    summary, assignments, test = load_publication(publication)
    counts = check_memberships(summary, assignments, test)
    config = load_executor_config(config_path)
    plan = verify_executor_plan(plan_path)
    preparation = verify_preparation(prepared)
    source_id = file_sha256(publication / "receipt.json")
    if plan.get("input_contract") or preparation.get("input_contract"):
        raise ValueError(
            "This policy requires original fixed-test occurrence publication"
        )
    initializations = {r["identity"]["initialization"]["sha256"] for r in plan["runs"]}
    if len(initializations) != 1:
        raise ValueError("Mixed initializations in plan")
    reconstructed = plan_runs(
        summary, assignments, config, source_id, initializations.pop()
    )
    if (
        plan_contract(plan) != plan_contract(reconstructed)
        or preparation["source_id"] != source_id
    ):
        raise ValueError("Plan/preparation differs from verified publication")
    expected_views = {records_identity(rows) for rows in assignments.values()}
    if set(preparation["views"]) != expected_views or preparation["test_materialized"]:
        raise ValueError("Prepared views differ or expose test")
    source = load_labeled_audit(publication / "sources/audit")
    sources = verify_original_sources(source, source_root, evidence)
    if sources != preparation["original_sources"]:
        raise ValueError("Original source verification differs from preparation")
    annotations = {
        r["label_id"]: r
        for r in read_json(publication / "sources/audit/annotation_audit.json")[
            "records"
        ]
    }
    all_records = pd.concat(assignments.values()).drop_duplicates("frame_id")
    for row in all_records.itertuples():
        annotation = annotations[row.label_ids[0]]
        raw = base64.b64decode(annotation["original_bytes_base64"], validate=True)
        parsed = validate_label_bytes(raw)
        if (
            hashlib.sha256(raw).hexdigest() != row.label_sha256
            or annotation["label_sha256"] != row.label_sha256
            or row.frame_id not in annotation["frame_ids"]
            or not parsed.syntax_valid
            or not parsed.normalized_values_valid
            or not set(parsed.classes_present) <= set(range(5))
            or parsed.num_objects != row.num_objects
        ):
            raise ValueError("Original occurrence annotation identity/syntax mismatch")
    for rows in assignments.values():
        identity = records_identity(rows)
        actual = pd.read_parquet(prepared / "views" / identity / "assignments.parquet")
        if records_identity(actual) != records_identity(rows):
            raise ValueError("Materialized assignments changed")
    geometry_warnings = set()
    for view_id in sorted(expected_views):
        rows = pd.read_parquet(prepared / "views" / view_id / "records.parquet")
        for row in rows.itertuples():
            label = (
                Path(row.image_path).parent.parent / "labels" / f"{row.frame_id}.txt"
            )
            parsed = validate_label_bytes(label.read_bytes())
            if (
                not parsed.syntax_valid
                or not parsed.normalized_values_valid
                or not set(parsed.classes_present) <= set(range(5))
                or parsed.num_objects != row.num_objects
            ):
                raise ValueError("Original annotation syntax/count mismatch")
            if not parsed.geometry_valid:
                geometry_warnings.add(row.frame_id)
    return {
        "counts": {
            **counts,
            "unique_memberships": len(expected_views),
            "experimental_identities": plan["experimental_identities"],
            "executions": plan["execution_count"],
        },
        "config": config,
        "plan_id": plan["plan_id"],
        "original_sources": sources,
        "development_geometry_warnings": sorted(geometry_warnings),
    }


def admit(
    publication: Path,
    plan: Path,
    prepared: Path,
    config: Path,
    source_root: Path,
    output: Path,
    evidence: Path | None = None,
) -> dict:
    external_output(output, publication, plan, prepared, source_root)
    checked = inspect_inputs(publication, plan, prepared, config, source_root, evidence)
    return emit(
        output,
        {
            "policy": POLICY,
            "kind": "data_gate",
            **checked,
            "publication": binding(publication / "receipt.json"),
            "plan": binding(plan / "plan.json"),
            "preparation": binding(prepared / "preparation.json"),
            "configuration": binding(config),
            "source_root": str(source_root.resolve()),
            "evidence": binding(evidence) if evidence else None,
            "scientific_limitations": LIMITATIONS,
            "annotation_policy": "original_bytes_per_occurrence",
            "human_review_asserted": False,
        },
    )


def verify_admission(path: Path) -> dict:
    result = receipt(path, "data_gate")
    checked = inspect_inputs(
        bound_file(result["publication"]).parent,
        bound_file(result["plan"]).parent,
        bound_file(result["preparation"]).parent,
        bound_file(result["configuration"]),
        Path(result["source_root"]),
        bound_file(result["evidence"]) if result["evidence"] else None,
    )
    if any(result.get(k) != v for k, v in checked.items()):
        raise ValueError("Technical admission differs from current immutable inputs")
    return result
