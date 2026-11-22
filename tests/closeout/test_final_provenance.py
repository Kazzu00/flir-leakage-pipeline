"""Synthetic provenance projection: do not publish private native file maps."""

import json

import pytest

from flir_pipeline.closeout.final_provenance import (
    AGGREGATE_FILES,
    EVALUATION,
    HASH_KEYS,
    public_receipts,
)


def inputs():
    originals = {}
    for policy in ("historica", "aleatoria", "c10"):
        for treatment in ("AUG_ON", "AUG_OFF"):
            receipt = {
                "state": "COMPLETED_EVALUATION", "protocol": "oe4_closeout_v2",
                "policy": policy, "treatment": treatment, "split": "test",
                "checkpoint_selection": "last_epoch_50", "test_tuning": False,
                "evaluation": EVALUATION,
                **dict.fromkeys(HASH_KEYS, "a" * 64),
                "environment": {"ultralytics": "8.3.203", "torch": "2.7.1+cu118",
                                "cuda": "11.8", "gpu": "SYNTHETIC", "slurm_job_id": "SYNTHETIC",
                                "private_path": "DO_NOT_PUBLISH"},
                "files": {**dict.fromkeys(AGGREGATE_FILES, "b" * 64),
                          "test-view/images/PRIVATE_CONTENT_ID.jpg": "c" * 64},
            }
            originals[policy, treatment] = json.dumps(receipt).encode()
    return originals, {"state": "AGGREGATED", "protocol": "oe4_closeout_v2", "test": ["a" * 64] * 2}


def test_six_receipts_are_bound_and_private_file_maps_are_excluded():
    originals, aggregate = inputs()
    raw = public_receipts(originals, aggregate)
    result = json.loads(raw)
    assert len(result["receipts"]) == 6
    assert result["checkpoint_bytes_verified_locally"] is False
    assert result["bound_payloads_verified_locally"] is False
    assert b"PRIVATE_CONTENT_ID" not in raw and b"DO_NOT_PUBLISH" not in raw
    for row in result["receipts"]:
        assert row["metadata"]["weights_sha256"] == "a" * 64
        assert set(row["aggregate_file_sha256"]) == set(AGGREGATE_FILES)


@pytest.mark.parametrize("change", ["missing", "policy", "test", "freeze", "hash", "tuning"])
def test_missing_or_inconsistent_native_receipt_is_rejected(change):
    originals, aggregate = inputs()
    if change == "missing":
        originals.pop(("c10", "AUG_OFF"))
    else:
        key = ("historica", "AUG_ON")
        receipt = json.loads(originals[key])
        field, value = {"policy": ("policy", "c10"), "test": ("test_records_sha256", "d" * 64),
                        "freeze": ("freeze_sha256", "d" * 64), "hash": ("weights_sha256", "invalid"),
                        "tuning": ("test_tuning", True)}[change]
        receipt[field] = value
        originals[key] = json.dumps(receipt).encode()
    with pytest.raises(ValueError):
        public_receipts(originals, aggregate)
