"""Signed, externally anchored admission for common-test YOLO11m training.

No self-issued JSON, edit to the prepared dataset, or CLI flag confers approval.
The trusted signers digest is provisioned OUTSIDE this public repository.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

from flir_pipeline.detection.common_executor import (
    NAMES,
    VERSION,
    verify_executor_plan,
    verify_preparation,
)
from flir_pipeline.similarity.storage import file_sha256, stable_id

ADMISSION_VERSION = "flir_common_test_training_admission_v1"
NAMESPACE = "flir-yolo11m-common-test"
REQUIRED_DECISIONS = (
    "annotations_adjudicated",
    "class_four_interpretation_adjudicated",
    "test_holdout_frozen",
    "retrospective_exposure_limit_accepted",
    "class_support_and_dependence_reviewed",
    "train_val_and_quarantine_accepted",
    "common_test_protocol_approved",
    "no_test_use_for_selection",
    "same_training_controls_for_all_arms",
)
SHA256 = re.compile(r"^[a-f0-9]{64}$")


def require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        raise ValueError(f"Invalid {field}: expected lowercase SHA256")
    return value


def verify_signature(
    authorization: Path, signature: Path, signers: Path, identity: str
) -> None:
    """Use OpenSSH SSHSIG; trust anchor must be provisioned independently."""
    expected = os.environ.get("FLIR_TRUSTED_SIGNERS_SHA256", "")
    require_sha256(expected, "FLIR_TRUSTED_SIGNERS_SHA256")
    if file_sha256(signers) != expected:
        raise ValueError("Trusted signer file does not match external trust anchor")
    if not identity.strip() or not signers.is_file() or not signature.is_file():
        raise ValueError("Missing signer identity, signers, or signature")
    signed = authorization.read_bytes()
    try:
        check = subprocess.run(
            [
                "ssh-keygen", "-Y", "verify", "-f", str(signers.resolve()),
                "-I", identity, "-n", NAMESPACE, "-s", str(signature.resolve()),
            ],
            input=signed,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as exc:
        raise ValueError("OpenSSH signature verification unavailable") from exc
    if check.returncode != 0:
        raise ValueError("Invalid admission signature or untrusted reviewer identity")


def verify_admission(
    plan_dir: Path,
    preparation_dir: Path,
    checkpoint: Path,
    authorization: Path,
    signature: Path,
    signers: Path,
    identity: str,
) -> dict:
    """Verify signatures, cross-artifact identities, and the scientific freeze.

    This is a technical gate; the real human reviewer remains responsible for
    correctness and independence of the signed scientific decisions.
    """
    repo = Path(__file__).resolve().parents[3]
    roots = (authorization, signature, signers, checkpoint)
    if any(p.resolve().is_relative_to(repo) for p in roots):
        raise ValueError("Admission artifacts and weights must be outside the repository")
    verify_signature(authorization, signature, signers, identity)
    try:
        signed = json.loads(authorization.read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid signed authorization document") from exc
    if signed.get("schema_version") != ADMISSION_VERSION:
        raise ValueError("Unsupported scientific admission version")
    if signed.get("decision") != "approved_for_real_training":
        raise ValueError("Training was not approved")
    if signed.get("reviewer_identity") != identity:
        raise ValueError("Reviewer identity is not the authorized signer")
    if not isinstance(signed.get("review_date"), str) or not signed["review_date"]:
        raise ValueError("Attributable review date is required")
    if not isinstance(signed.get("rationale"), str) or len(signed["rationale"].strip()) < 20:
        raise ValueError("Substantive signed review rationale is required")
    decisions = signed.get("decisions")
    if not isinstance(decisions, dict) or any(
        decisions.get(field) is not True for field in REQUIRED_DECISIONS
    ):
        raise ValueError("Signed scientific decisions are incomplete")

    plan = verify_executor_plan(plan_dir)
    prepared = verify_preparation(preparation_dir)
    if signed.get("plan_id") != plan["plan_id"]:
        raise ValueError("Approval does not match this experiment plan")
    if require_sha256(signed.get("plan_receipt_sha256"), "plan receipt") != file_sha256(
        plan_dir / "receipt.json"
    ):
        raise ValueError("Approved plan checksum mismatch")
    if require_sha256(
        signed.get("preparation_receipt_sha256"), "preparation receipt"
    ) != file_sha256(preparation_dir / "preparation.json"):
        raise ValueError("Approved preparation checksum mismatch")
    if prepared["source_id"] != plan["source_id"]:
        raise ValueError("Preparation and experiment plan have different sources")
    if prepared["test_materialized"] is not False:
        raise ValueError("The prepared dataset exposes test images")
    if prepared["schema_version"] != VERSION:
        raise ValueError("Incompatible prepared dataset")
    test_ids = {view["test_id"] for view in prepared["views"].values()}
    if len(test_ids) != 1 or signed.get("frozen_test_id") not in test_ids:
        raise ValueError("Common test identity is not frozen across views")
    if len(prepared["views"]) != len({
        r["identity"]["records_id"] for r in plan["runs"]
    }):
        raise ValueError("Plan and prepared views differ")
    if not {
        r["identity"]["records_id"] for r in plan["runs"]
    }.issubset(prepared["views"]):
        raise ValueError("Missing prepared development view")

    checkpoint_hash = file_sha256(checkpoint)
    if require_sha256(signed.get("checkpoint_sha256"), "weights") != checkpoint_hash:
        raise ValueError("Signed checkpoint hash differs from local bytes")
    if any(
        r["identity"]["initialization"]["sha256"] != checkpoint_hash
        for r in plan["runs"]
    ):
        raise ValueError("Plan does not bind the approved official checkpoint")
    if signed.get("initialization_source") != "official_ultralytics_yolo11m":
        raise ValueError("Official model provenance has not been adjudicated")

    freeze = signed.get("freeze")
    config = plan["runs"][0]["identity"]["configuration"]
    if not isinstance(freeze, dict) or freeze.get("batch") not in (1, 2):
        raise ValueError("Signed common batch must be 1 or 2")
    if (
        freeze.get("device") != "0"
        or freeze.get("epochs") != config["train"]["epochs"]
        or freeze.get("imgsz") != config["train"]["imgsz"]
        or freeze.get("optimizer") != config["train"]["optimizer"]
        or freeze.get("torch") != config["torch_version"]
        or freeze.get("ultralytics") != config["ultralytics_version"]
        or freeze.get("torchvision") != "0.22.1"
        or freeze.get("amp") is not False
        or freeze.get("tf32") is not False
        or freeze.get("validated_non_test_pilot") is not True
    ):
        raise ValueError("Signed environment/protocol freeze is incomplete")
    require_sha256(freeze.get("ultralytics_defaults_sha256"), "Ultralytics defaults")
    require_sha256(freeze.get("pilot_receipt_sha256"), "approved pilot")
    if signed.get("class_names") != {str(k): v for k, v in NAMES.items()}:
        raise ValueError("Approved class mapping differs from original labels")
    return {
        "approval_id": stable_id(signed),
        "authorization_sha256": file_sha256(authorization),
        "signature_sha256": file_sha256(signature),
        "trusted_signers_sha256": file_sha256(signers),
        "plan_id": plan["plan_id"],
        "preparation_sha256": file_sha256(preparation_dir / "preparation.json"),
        "checkpoint_sha256": checkpoint_hash,
        "frozen_test_id": signed["frozen_test_id"],
        "freeze": freeze,
        "plan": plan,
        "prepared": prepared,
    }
