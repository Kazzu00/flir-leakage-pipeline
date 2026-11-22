"""Synthetic regressions for the five audited v1 defects; no external network."""

import csv
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker, ValidationError

from flir_pipeline.audit.contracts import DOCUMENTS, schema_for
from flir_pipeline.audit.demo import create_demo
from flir_pipeline.audit.importers import encode, normalize
from flir_pipeline.audit.legacy_importers import normalize as legacy_normalize
from flir_pipeline.audit.models import DecisionInput
from flir_pipeline.audit.store import Store


def candidates():
    return {
        "source_video": "SYNTHETIC",
        "sample_rate_fps": 5,
        "start_second_nominal": 100,
        "results": [
            {
                "approx_video_second": 100 + i / 5,
                "approx_video13_index": i / 5,
                "event_start_sample": i,
                "event_end_sample": i + 3,
                "low_similarity_samples": 4,
                "candidate_class": "synthetic",
                "minimum_correlation": 0.1,
            }
            for i in (5, 10)
        ],
    }


def write_csv(path, doc):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=doc["results"][0])
        writer.writeheader()
        writer.writerows(doc["results"])


def request(event_id, **kwargs):
    return {
        "event_id": event_id,
        "expected_version": 0,
        "reviewer": "SYNTHETIC",
        "justification": "Revisión sintética: región y cámara",
        **kwargs,
    }


@pytest.fixture
def paired(tmp_path):
    root = tmp_path / "inputs"
    root.mkdir()
    store = Store(tmp_path / "ledger.sqlite3")
    write_csv(root / "candidates.csv", candidates())
    store.import_directory(root)
    return store, root


@pytest.mark.parametrize(
    "kwargs",
    [
        {"status": "reviewed", "confidence": "high"},
        {"status": "reviewed", "continuity": "continuous"},
        {"status": "reviewed", "dependence": "different_region", "confidence": "low"},
    ],
)
def test_conclusions_without_evidence_rejected(paired, kwargs):
    store, _ = paired
    event = store.snapshot()["events"][0]
    with pytest.raises(ValueError, match="evidencia"):
        store.save_decision(request(event["event_id"], **kwargs))
    assert store.snapshot()["decisions"] == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"status": "pending", "confidence": "high", "continuity": "shot_change"},
        {
            "status": "reviewed",
            "continuity": "uncertain",
            "dependence": "uncertain",
            "confidence": "undetermined",
        },
    ],
)
def test_pending_and_indeterminate_remain_possible(paired, kwargs):
    store, _ = paired
    assert (
        store.save_decision(
            request(store.snapshot()["events"][0]["event_id"], **kwargs)
        )["version"]
        == 1
    )


def test_external_evidence_bound_frozen_and_roundtrip(paired, tmp_path):
    from PIL import Image

    store, root = paired
    image = tmp_path / "SYNTHETIC.png"
    Image.new("RGB", (12, 12), "red").save(image)
    event, other = store.snapshot()["events"]
    evidence = store.attach_evidence(
        event["event_id"],
        image,
        locator="SYNTHETIC record 5-8",
        basis="explicit_record_reference",
        rationale="Manual record correspondence, not proximity",
        reviewer="SYNTHETIC",
    )
    validator = Draft202012Validator(
        schema_for("audit-external-evidence-v1"), format_checker=FormatChecker()
    )
    validator.validate(evidence)
    with pytest.raises(ValidationError):
        validator.validate({**evidence, "sha256": "not-a-digest"})
    assert (
        store.attach_evidence(
            event["event_id"],
            image,
            locator="SYNTHETIC record 5-8",
            basis="explicit_record_reference",
            rationale="Manual record correspondence, not proximity",
            reviewer="SYNTHETIC",
        )
        == evidence
    )
    conclusion = request(
        event["event_id"],
        status="reviewed",
        continuity="shot_change",
        dependence="same_region",
        confidence="high",
        external_evidence_consulted=[evidence["evidence_id"]],
    )
    saved = store.save_decision(conclusion)
    with pytest.raises(ValueError, match="registered"):
        store.save_decision({**conclusion, "event_id": other["event_id"]})
    with pytest.raises(ValueError, match="registered"):
        store.save_decision(
            {
                **conclusion,
                "expected_version": 1,
                "external_evidence_consulted": ["invented"],
            }
        )
    export = tmp_path / "decisions.json"
    export.write_bytes(store.export())
    restored = Store(tmp_path / "restored.sqlite3")
    restored.import_directory(root)
    with pytest.raises(ValueError, match="registered"):
        restored.import_decisions(export)
    restored.attach_evidence(
        event["event_id"],
        image,
        locator="SYNTHETIC record 5-8",
        basis="explicit_record_reference",
        rationale="Manual record correspondence, not proximity",
        reviewer="SYNTHETIC",
    )
    restored.import_decisions(export)
    assert restored.snapshot()["decisions"] == [saved]
    # The frozen bytes remain verifiable if the separately supplied file changes.
    image.write_bytes(b"changed external file")
    assert restored.verify()["valid"]
    with restored.connect() as db:
        original_payload = db.execute(
            "SELECT payload FROM external_evidence"
        ).fetchone()[0]
        forged = json.loads(original_payload)
        forged["rationale"] = "Unlogged replacement"
        db.execute("UPDATE external_evidence SET payload=?", (encode(forged),))
    assert not restored.verify()["valid"]
    with restored.connect() as db:
        db.execute("UPDATE external_evidence SET payload=?", (original_payload,))
        db.execute("UPDATE external_evidence SET raw=?", (b"tampered",))
    assert not restored.verify()["valid"]


def test_csv_then_json_stable_ids_and_reviews(paired):
    store, root = paired
    before = store.snapshot()
    pending = store.save_decision(request(before["events"][0]["event_id"]))
    (root / "candidates.json").write_text(json.dumps(candidates()))
    for _ in range(3):
        assert all(
            f["status"] == "valid" for f in store.import_directory(root)["files"]
        )
        assert store.verify()["valid"]
    after = store.snapshot()
    assert len(after["events"]) == 2 and after["observation_count"] == 2
    assert [e["event_id"] for e in before["events"]] == [
        e["event_id"] for e in after["events"]
    ]
    assert after["decisions"] == [pending]
    assert (
        after["companions"][0]["canonical_asset_id"] == before["assets"][0]["asset_id"]
    )
    assert after["events"][0]["end_second"] == 101.6
    assert "supplemental_asset=" in after["events"][0]["temporal_basis"]


def test_companion_conflict_keeps_previous_rows_and_reviews(paired):
    store, root = paired
    saved = store.save_decision(request(store.snapshot()["events"][0]["event_id"]))
    changed = candidates()
    changed["results"][0]["minimum_correlation"] = 0.9
    (root / "candidates.json").write_text(json.dumps(changed))
    result = store.import_directory(root)
    assert all(r["status"] == "error" for r in result["files"])
    assert len(store.snapshot()["events"]) == 2 and len(store.snapshot()["assets"]) == 1
    assert store.snapshot()["decisions"] == [saved]


def test_json_first_then_csv_is_idempotent(tmp_path):
    root = tmp_path / "inputs"
    root.mkdir()
    (root / "candidates.json").write_text(json.dumps(candidates()))
    store = Store(tmp_path / "ledger.sqlite3")
    store.import_directory(root)
    ids = [e["event_id"] for e in store.snapshot()["events"]]
    write_csv(root / "candidates.csv", candidates())
    store.import_directory(root)
    store.import_directory(root)
    assert [e["event_id"] for e in store.snapshot()["events"]] == ids
    assert store.verify()["valid"]


def test_multi_sample_interval_and_unknown_precision():
    _, events, _, _, _ = normalize(json.dumps(candidates()).encode(), "candidates.json")
    event = events[0]
    assert (event["event_start_sample"], event["event_end_sample"]) == (5, 8)
    assert (event["start_second"], event["end_second"]) == (101, 101.6)
    assert (
        event["precision_seconds"] is None and "unverified" in event["temporal_basis"]
    )
    doc = candidates()
    doc.pop("sample_rate_fps")
    _, events, _, _, _ = normalize(json.dumps(doc).encode(), "candidates.json")
    assert events[0]["end_second"] is None and events[0]["event_end_sample"] == 8
    doc = candidates()
    doc["results"][0]["approx_video_second"] = 999
    with pytest.raises(ValueError, match="Ambiguous"):
        normalize(json.dumps(doc).encode(), "candidates.json")


def test_cross_split_index_anomaly_preserved_and_persisted(tmp_path):
    root = tmp_path / "inputs"
    root.mkdir()
    doc = {
        "family": "SYNTHETIC",
        "indices_unique": 2,
        "splits": {
            "train": {"count": 2, "indices": [1, 2]},
            "val": {"count": 1, "indices": [2]},
            "test": {"count": 1, "indices": [2]},
        },
    }
    (root / "historical.json").write_text(json.dumps(doc))
    store = Store(tmp_path / "ledger.sqlite3")
    result = store.import_directory(root)
    anomaly = result["files"][0]["anomalies"][0]
    assert anomaly["dataset_index"] == 2 and anomaly["splits"] == [
        "test",
        "train",
        "val",
    ]
    assert anomaly["semantic_leakage_confirmed"] is False
    assert len(anomaly["observation_ids"]) == 3
    assert Store(store.path).snapshot()["anomalies"] == [anomaly]
    assert store.snapshot()["observation_count"] == 4 and store.verify()["valid"]


def legacy_insert(store, path):
    raw = path.read_bytes()
    asset, events, observations, media, _ = legacy_normalize(raw, path.name)
    frozen = {
        "asset": asset,
        "events": events,
        "observations": observations,
        "media": media,
        "csv_companion": False,
    }
    with store.connect() as db:
        db.execute(
            "INSERT INTO assets VALUES (?,?,?,?,?)",
            (
                asset["asset_id"],
                asset["sha256"],
                str(path.resolve()),
                raw,
                encode(frozen),
            ),
        )
        for table, items, key in [
            ("events", events, "event_id"),
            ("observations", observations, "observation_id"),
        ]:
            db.executemany(
                f"INSERT INTO {table} VALUES (?,?,?)",
                [(r[key], asset["asset_id"], encode(r)) for r in items],
            )
    return asset, events


def test_legacy_duplicate_detected_explicit_reconciliation_preserves_both_histories(
    tmp_path,
):
    root = tmp_path / "inputs"
    root.mkdir()
    write_csv(root / "candidates.csv", candidates())
    (root / "candidates.json").write_text(json.dumps(candidates()))
    store = Store(tmp_path / "legacy.sqlite3")
    a, ae = legacy_insert(store, root / "candidates.csv")
    b, be = legacy_insert(store, root / "candidates.json")
    first = store.save_decision(request(ae[0]["event_id"]))
    second = store.save_decision(request(be[0]["event_id"]))
    assert not store.verify()["valid"]
    receipt = store.reconcile_companions(
        a["asset_id"], b["asset_id"], "SYNTHETIC", "Explicit equivalence review"
    )
    assert receipt["decisions_migrated"] is False
    assert store.verify()["valid"] and len(store.snapshot()["events"]) == 2
    assert sorted(store.snapshot()["decisions"], key=lambda d: d["event_id"]) == sorted(
        [first, second], key=lambda d: d["event_id"]
    )
    assert store.snapshot()["companions"][0]["event_links"][0] == {
        "canonical_event_id": ae[0]["event_id"],
        "companion_event_id": be[0]["event_id"],
    }
    with store.connect() as db:
        assert db.execute("SELECT count(*) FROM events").fetchone()[0] == 4


def test_full_document_and_message_schemas(tmp_path):
    root = tmp_path / "inputs"
    create_demo(root)
    store = Store(tmp_path / "ledger.sqlite3")
    store.import_directory(root)
    event = store.snapshot()["events"][0]
    value = request(
        event["event_id"],
        status="reviewed",
        continuity="continuous",
        evidence_consulted=event["evidence"],
    )
    saved = store.save_decision(value)
    samples = {
        "audit-evidence-v1": store.snapshot(),
        "audit-decision-input-v1": value,
        "audit-decision-export-v1": json.loads(store.export()),
        "audit-review-decision-v1": saved,
    }
    for name in DOCUMENTS:
        schema = schema_for(name)
        Draft202012Validator.check_schema(schema)
        assert (
            json.loads(
                (Path("docs/contracts") / f"{name}.schema.json").read_text(
                    encoding="utf-8"
                )
            )
            == schema
        )
        validator = Draft202012Validator(schema, format_checker=FormatChecker())
        with pytest.raises(ValidationError):
            validator.validate({})
        if name in samples:
            validator.validate(samples[name])
    bad = {**value, "evidence_consulted": []}
    with pytest.raises(ValidationError):
        Draft202012Validator(schema_for("audit-decision-input-v1")).validate(bad)
    with pytest.raises(ValueError):
        DecisionInput.model_validate(bad)
    with pytest.raises(ValidationError):
        Draft202012Validator(schema_for("audit-review-decision-v1")).validate(
            {**saved, "evidence_consulted": []}
        )
    with pytest.raises(ValidationError):
        Draft202012Validator(schema_for("audit-decision-export-v1")).validate(
            {"schema": "wrong-version", "contract_revision": "1.1", "decisions": []}
        )
    invalid = store.snapshot()
    invalid["events"][0]["event_end_sample"] = -1
    with pytest.raises(ValidationError):
        Draft202012Validator(schema_for("audit-evidence-v1")).validate(invalid)


def test_early_v1_companion_without_role_marker_remains_readable(tmp_path):
    root = tmp_path / "inputs"
    root.mkdir()
    (root / "candidates.json").write_text(json.dumps(candidates()))
    write_csv(root / "candidates.csv", candidates())
    store = Store(tmp_path / "legacy.sqlite3")
    legacy_insert(store, root / "candidates.json")
    asset, _ = legacy_insert(store, root / "candidates.csv")
    with store.connect() as db:
        frozen = json.loads(
            db.execute(
                "SELECT normalized FROM assets WHERE id=?", (asset["asset_id"],)
            ).fetchone()[0]
        )
        frozen.pop("csv_companion")
        frozen["events"], frozen["observations"] = [], []
        db.execute("DELETE FROM events WHERE asset_id=?", (asset["asset_id"],))
        db.execute("DELETE FROM observations WHERE asset_id=?", (asset["asset_id"],))
        db.execute(
            "UPDATE assets SET normalized=? WHERE id=?",
            (encode(frozen), asset["asset_id"]),
        )
    assert store.verify()["valid"]
    assert len(store.snapshot()["events"]) == 2


def test_ungrounded_v1_history_is_disclosed_not_rewritten(tmp_path):
    from datetime import UTC, datetime

    from flir_pipeline.audit.legacy_models import ReviewDecision as OldDecision

    root = tmp_path / "inputs"
    create_demo(root)
    store = Store(tmp_path / "legacy.sqlite3")
    store.import_directory(root)
    event = store.snapshot()["events"][0]
    asset = store.snapshot()["assets"][0]
    old = OldDecision(
        **request(event["event_id"], status="reviewed", confidence="high"),
        decision_id="SYNTHETIC-old",
        version=1,
        created_at=datetime.now(UTC),
        evidence_sha256=asset["sha256"],
    ).model_dump(mode="json")
    with store.connect() as db:
        db.execute(
            "INSERT INTO decisions VALUES (?,?,?,?)",
            (old["decision_id"], old["event_id"], 1, encode(old)),
        )
    assert store.verify()["valid"]
    assert not store.verify()["latest_reviews_evidence_valid"]
    assert store.snapshot()["legacy_review_warnings"] == [old["decision_id"]]
    store.save_decision(
        request(
            event["event_id"],
            expected_version=1,
            status="reviewed",
            confidence="high",
            evidence_consulted=event["evidence"],
        )
    )
    assert store.verify()["latest_reviews_evidence_valid"]
    assert store.snapshot()["decisions"][0] == old
