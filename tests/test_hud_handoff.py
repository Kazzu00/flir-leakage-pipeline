"""Transfer integrity does not bypass scientific or annotation gates."""

import importlib.util
import json
from pathlib import Path

import pytest

MODULE = Path(__file__).parents[1] / "scripts/hypatia/verify_hud_package.py"
spec = importlib.util.spec_from_file_location("handoff", MODULE)
handoff = importlib.util.module_from_spec(spec)
spec.loader.exec_module(handoff)


@pytest.fixture
def package(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("frame_id,label_empty\na,True\n", encoding="utf-8")
    receipt = tmp_path / "development_receipt.json"
    receipt.write_text(json.dumps({
        "schema_version": "curated_development_v1",
        "artifacts": {"manifest.csv": handoff.sha256(manifest)},
        "canonical_contents": 1, "counts": {}, "protected_exclusion": {},
        "ready_for_training": True,
    }), encoding="utf-8")
    return tmp_path, handoff.sha256(receipt)


def test_integrity_never_admits_even_self_declared_ready(package):
    result = handoff.verify(*package)
    assert result["integrity"] == "PASS"
    assert result["ready_for_training"] is False


def test_changed_payload_rejected(package):
    (package[0] / "manifest.csv").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        handoff.verify(*package)


def test_added_test_payload_rejected(package):
    (package[0] / "test.txt").write_text("forbidden", encoding="utf-8")
    with pytest.raises(ValueError, match="additional"):
        handoff.verify(*package)


def test_receipt_tamper_rejected(package):
    with pytest.raises(ValueError, match="Receipt checksum"):
        handoff.verify(package[0], "0" * 64)
