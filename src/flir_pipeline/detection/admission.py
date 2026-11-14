"""Fail-closed scientific authorization; authority is outside the experiment tree.

OpenSSH signs exact document bytes. Only a root-managed system allowlist is
trusted; neither a JSON flag nor a key supplied alongside a run grants authority.
This attributes a decision, not proof that its scientific judgment is correct.
"""

from __future__ import annotations

import json
import os
import re
import stat
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from flir_pipeline.similarity.storage import file_sha256

TRUST = Path("/etc/flir-pipeline/allowed_signers")
SSH_KEYGEN = "/usr/bin/ssh-keygen"
NAMESPACE = "flir-scientific-admission-v1"
REPO = Path(__file__).resolve().parents[3]
DECISIONS = {
    "annotations",
    "class_ontology",
    "class_sufficiency",
    "dependencies",
    "test_freeze",
    "protocol_budget",
    "weights_origin",
    "non_test_batch_pilot",
}


def system_trust() -> Path:
    if os.name != "posix" or os.geteuid() == 0:
        raise ValueError(
            "BLOCKED: non-root POSIX worker and administrator trust required"
        )
    for path in (TRUST, *TRUST.parents):
        info = path.lstat()
        if path.is_symlink() or info.st_uid != 0 or info.st_mode & 0o022:
            raise ValueError(
                "BLOCKED: trust root must be root-owned and not writable by workers"
            )
    if not stat.S_ISREG(TRUST.stat().st_mode):
        raise ValueError("Invalid authority allowlist")
    return TRUST


def verify_signature(document: Path, signature: Path) -> bytes:
    trust = system_trust()
    raw = document.read_bytes()
    result = subprocess.run(
        [
            SSH_KEYGEN,
            "-Y",
            "verify",
            "-f",
            str(trust),
            "-I",
            "flir-scientific-approver",
            "-n",
            NAMESPACE,
            "-s",
            str(signature),
        ],
        input=raw,
        capture_output=True,
        check=False,
        timeout=15,
    )
    if result.returncode:
        raise ValueError("BLOCKED: invalid or untrusted scientific signature")
    return raw


def authorized(document: Path | None, signature: Path | None, purpose: str) -> dict:
    if document is None or signature is None:
        raise ValueError("BLOCKED: authenticated_admission_contract required")
    try:
        payload = json.loads(verify_signature(document, signature))
        if payload["schema_version"] != NAMESPACE or payload["purpose"] != purpose:
            raise ValueError("Wrong authorization scope")
        now = datetime.now(UTC)
        if not (
            datetime.fromisoformat(payload["not_before"])
            <= now
            < datetime.fromisoformat(payload["expires_at"])
        ):
            raise ValueError("Authorization expired or not active")
        if not payload["reviewer"].strip():
            raise ValueError("Attributed reviewer required")
        return payload
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("BLOCKED: invalid admission document or authority") from exc


def bound_file(binding: dict) -> Path:
    path = Path(binding["path"])
    if not path.is_absolute() or not re.fullmatch(r"[a-f0-9]{64}", binding["sha256"]):
        raise ValueError("Absolute locator and SHA256 required")
    if (
        path.is_symlink()
        or not path.is_file()
        or file_sha256(path) != binding["sha256"]
    ):
        raise ValueError("Bound input checksum mismatch")
    return path


def verify_checkout(payload: dict) -> None:
    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(REPO), *args], text=True
        ).strip()

    if git("rev-parse", "HEAD") != payload["code_commit"] or git(
        "status", "--porcelain"
    ):
        raise ValueError("Approved clean checkout required")
    if file_sha256(REPO / "uv.lock") != payload["uv_lock_sha256"]:
        raise ValueError("Dependency lock changed")


def training_admission(document: Path | None, signature: Path | None) -> dict:
    payload = authorized(document, signature, "train")
    if set(payload["decisions"]) != DECISIONS:
        raise ValueError("All scientific decisions must be bound")
    for binding in payload["decisions"].values():
        bound_file(binding)
    if payload["test"]["status"] != "APPROVED" or not payload["test"]["records_id"]:
        raise ValueError("Provisional test cannot be trained/evaluated")
    if payload["population"] != "expanded" or payload["split_seeds"] not in (
        [0],
        [0, 1, 2, 3, 4],
    ):
        raise ValueError(
            "Only the explicit primary matrix or approved extension is supported"
        )
    if payload["detector_seeds"] != [42, 43, 44] or payload["policies"] != [
        "S1",
        "S2",
        "S3",
    ]:
        raise ValueError("Unbalanced proposed detector matrix")
    if payload["max_runs"] != 9 * len(payload["split_seeds"]):
        raise ValueError("Budget does not match matrix")
    if type(payload["batch"]) is not int or payload["batch"] < 1:
        raise ValueError("Positive common batch must be frozen")
    if payload["variant"]["kind"] not in ("original", "geometry_preserving"):
        raise ValueError("Unannotated video is not a supervised dataset")
    if not payload["variant"]["id"] or not payload["variant"]["version"]:
        raise ValueError("Dataset variant identity required")
    if payload["variant"]["kind"] == "geometry_preserving":
        if payload["variant"]["geometry"] != "identity_1to1":
            raise ValueError("Label reuse requires reviewed identity geometry")
        bound_file(payload["variant"]["mapping"])
        bound_file(payload["variant"]["review"])
    weights = payload["weights"]
    if (
        weights["url"]
        != "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt"
    ):
        raise ValueError("Official pinned initialization provenance required")
    if bound_file(weights).suffix != ".pt":
        raise ValueError("Official checkpoint must be an existing .pt file")
    verify_checkout(payload)
    return payload


def check_test_isolation(payload: dict) -> None:
    """OS isolation is required, not merely omission from a YAML file.

    A custodian must enumerate all raw/test roots and mount them outside the
    worker namespace. This cannot discover undeclared copies on other systems.
    """
    roots = payload["test"]["protected_roots"]
    if not roots:
        raise ValueError("Test/raw-source isolation roots required")
    for value in roots:
        path = Path(value)
        if (
            not path.is_absolute()
            or os.access(path, os.R_OK)
            or (path.is_dir() and os.access(path, os.X_OK))
        ):
            raise ValueError("BLOCKED: worker can access protected test/raw source")
