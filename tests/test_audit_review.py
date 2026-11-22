"""Offline synthetic audit invariants; no FAC, models, network or GPU."""

import csv
import io
import json
from zipfile import ZipFile

import pytest
from typer.testing import CliRunner

from flir_pipeline.audit.cli import app
from flir_pipeline.audit.demo import create_demo
from flir_pipeline.audit.importers import normalize
from flir_pipeline.audit.server import make_server
from flir_pipeline.audit.store import Store, atomic_export


@pytest.fixture
def ledger(tmp_path):
    root = tmp_path / "SYNTHETIC-inputs"
    create_demo(root)
    store = Store(tmp_path / "ledger.sqlite3")
    report = store.import_directory(root)
    assert all(f["status"] == "valid" for f in report["files"])
    return store, root


def decision(store, **updates):
    event = store.snapshot()["events"][0]
    return {
        "event_id": event["event_id"],
        "expected_version": 0,
        "reviewer": "SYNTHETIC-reviewer",
        "justification": "Synthetic fixture only",
        "evidence_consulted": event["evidence"],
        **updates,
    }


def profile():
    return {
        "source_video": "SYNTHETIC",
        "method": "synthetic-profile-v1",
        "frames_received": 3,
        "results": [
            dict(
                left_index=i,
                right_index=i + 1,
                left_second=i,
                right_second=i + 1,
                correlation=0.2,
                sift_good_matches=2,
                ransac_inliers=1,
                review_candidate=i == 1,
            )
            for i in (0, 1)
        ],
    }


def test_json_csv_companions_and_observations(tmp_path):
    root = tmp_path / "inputs"
    root.mkdir()
    obj = profile()
    (root / "profile.json").write_text(json.dumps(obj))
    with (root / "profile.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=obj["results"][0])
        writer.writeheader()
        writer.writerows(obj["results"])
    store = Store(tmp_path / "ledger.sqlite3")
    report = store.import_directory(root)
    assert [f["status"] for f in report["files"]] == ["valid", "valid"]
    data = store.snapshot()
    assert len(data["assets"]) == 2 and len(data["events"]) == 1
    assert data["observation_count"] == 2
    event = data["events"][0]
    assert event["method"] == "synthetic-profile-v1"
    assert event["method_version"] == "unreported"
    assert event["verification"] == "candidate_only"
    assert store.verify(originals=True)["valid"]


def test_standalone_csv_optional_fields_and_missing_files(tmp_path):
    root = tmp_path / "inputs"
    root.mkdir()
    (root / "runs.csv").write_text("start,end,length\n2,4,3\n")
    store = Store(tmp_path / "ledger.sqlite3")
    report = store.import_directory(root)
    event = store.snapshot()["events"][0]
    assert event["source_video"] is None and event["start_second"] is None
    assert event["metrics"] == {} and report["missing"]


def test_zip_idempotent_originals_and_media(ledger):
    store, root = ledger
    before = store.snapshot()
    original = (root / "synthetic_review.zip").read_bytes()
    store.import_directory(root)
    after = store.snapshot()
    for key in ("assets", "events", "observation_count"):
        assert before[key] == after[key]
    assert len(after["events"]) == 3
    assert (root / "synthetic_review.zip").read_bytes() == original
    image, mime = store.media(after["events"][0]["event_id"], 0)
    assert image.startswith(b"\x89PNG") and mime == "image/png"
    assert store.verify(originals=True)["valid"]


@pytest.mark.parametrize(
    "name", ["../evil.jpg", "/absolute.jpg", "C:/escape.jpg", "../nested/evil.jpg"]
)
def test_zip_traversal(name):
    stream = io.BytesIO()
    with ZipFile(stream, "w") as z:
        z.writestr(name, b"image")
        z.writestr("manifest.json", "[]")
    with pytest.raises(ValueError, match="Unsafe"):
        normalize(stream.getvalue(), "synthetic.zip")


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d["results"][0].pop("right_second"),
        lambda d: d["results"][0].update(correlation=float("nan")),
        lambda d: d["results"][0].update(correlation=1.1),
        lambda d: d["results"][0].update(left_index=5),
        lambda d: d.update(frames_received=55),
        lambda d: d["results"][0].update(review_candidate="maybe"),
    ],
)
def test_invalid_records(mutation):
    obj = profile()
    mutation(obj)
    with pytest.raises(ValueError):
        normalize(json.dumps(obj).encode(), "profile.json")


def test_corrupt_report_does_not_commit_partial_artifact(tmp_path):
    root = tmp_path / "inputs"
    root.mkdir()
    (root / "broken.json").write_text('{"results":')
    (root / "broken.zip").write_bytes(b"SYNTHETIC invalid ZIP")
    (root / "broken.csv").write_text('field\n"unclosed')
    store = Store(tmp_path / "ledger.sqlite3")
    report = store.import_directory(root)
    assert all(f["status"] == "error" for f in report["files"])
    assert not store.snapshot()["assets"]


def test_decisions_pending_history_conflicts_and_semantics(ledger):
    store, _ = ledger
    pending = store.save_decision(decision(store))
    assert pending["status"] == "pending" and pending["continuity"] == "uncertain"
    changed = store.save_decision(
        decision(
            store,
            expected_version=1,
            status="reviewed",
            continuity="shot_change",
            dependence="same_region",
        )
    )
    with pytest.raises(ValueError, match="conflict"):
        store.save_decision(decision(store))
    with pytest.raises(ValueError, match="evidence"):
        store.save_decision(
            decision(store, expected_version=2, evidence_consulted=["alien.png"])
        )
    reopened = Store(store.path).snapshot()
    assert reopened["decisions"] == [pending, changed]
    assert "continuity" not in reopened["events"][0]
    assert reopened["events"][0]["verification"] == "candidate_only"


@pytest.mark.parametrize("format", ["json", "csv"])
def test_complete_roundtrip_with_correction(ledger, tmp_path, format):
    store, root = ledger
    store.save_decision(
        decision(
            store, status="reviewed", continuity="continuous", dependence="same_region"
        )
    )
    store.save_decision(
        decision(
            store,
            expected_version=1,
            status="reviewed",
            continuity="shot_change",
            dependence="same_region",
        )
    )
    output = tmp_path / f"decisions.{format}"
    atomic_export(output, store.export(format))
    target = Store(tmp_path / "reopened.sqlite3")
    target.import_directory(root)
    target.import_decisions(output)
    target.import_decisions(output)
    assert target.snapshot()["decisions"] == store.snapshot()["decisions"]
    assert target.verify()["valid"]


def test_missing_image_explicit_and_no_inferred_source():
    stream = io.BytesIO()
    with ZipFile(stream, "w") as z:
        z.writestr(
            "manifest.json",
            json.dumps(
                [
                    {
                        "review_id": "SYNTHETIC",
                        "approx_index": 1,
                        "center_second": 2,
                        "image": "absent.png",
                    }
                ]
            ),
        )
    _, events, _, _, warnings = normalize(stream.getvalue(), "window.zip")
    assert events[0]["evidence"] == [] and events[0]["source_video"] is None
    assert any("missing image" in w for w in warnings)


def test_tamper_detection_and_wrong_reimport_binding(ledger, tmp_path):
    store, _ = ledger
    d = store.save_decision(decision(store))
    d["evidence_sha256"] = "0" * 64
    target = tmp_path / "wrong.json"
    target.write_text(json.dumps({"schema": "audit-evidence-v1", "decisions": [d]}))
    with pytest.raises(ValueError, match="binding"):
        store.import_decisions(target)
    with store.connect() as db:
        db.execute("UPDATE assets SET sha=?", ("0" * 64,))
    assert not store.verify()["valid"]


def test_protect_input_directory(ledger):
    _, root = ledger
    with pytest.raises(ValueError, match="outside"):
        Store(root / "unsafe.sqlite3").import_directory(root)


def test_cli_demo_and_verify(tmp_path):
    runner = CliRunner()
    output = tmp_path / "demo"
    assert runner.invoke(app, ["demo", str(output)]).exit_code == 0
    db = str(output / "audit.sqlite3")
    assert runner.invoke(app, ["summary", "--db", db]).exit_code == 0
    assert runner.invoke(app, ["verify", "--db", db, "--originals"]).exit_code == 0


def test_server_only_binds_loopback(ledger):
    store, _ = ledger
    with pytest.raises(ValueError, match="loopback"):
        make_server(store, port=0, frontend_origin="https://remote.invalid")
    server = make_server(store, port=0)
    try:
        assert server.server_address[0] == "127.0.0.1"
    finally:
        server.server_close()


def test_failed_export_does_not_publish_or_overwrite(tmp_path, monkeypatch):
    target = tmp_path / "export.json"
    target.write_bytes(b"original")
    with pytest.raises(ValueError, match="already exists"):
        atomic_export(target, b"replacement")
    assert target.read_bytes() == b"original"

    def interrupted(*args):
        raise OSError("SYNTHETIC interrupted write")

    monkeypatch.setattr("flir_pipeline.audit.store.os.replace", interrupted)
    missing = tmp_path / "new.json"
    with pytest.raises(OSError, match="interrupted"):
        atomic_export(missing, b"content")
    assert not missing.exists() and not list(tmp_path.glob(".audit-export-*"))


def test_declared_version_and_automatic_verification_not_promoted():
    doc = profile()
    doc["method_version"] = "synthetic-2"
    doc["sequence_boundaries_verified"] = True
    _, events, observations, _, _ = normalize(json.dumps(doc).encode(), "profile.json")
    assert events[0]["method_version"] == "synthetic-2"
    assert events[0]["verification"] == "candidate_only"
    assert all(o["verification"] == "unverified" for o in observations)


def test_companion_disagreement_is_reported(tmp_path):
    root = tmp_path / "inputs"
    root.mkdir()
    obj = profile()
    (root / "profile.json").write_text(json.dumps(obj))
    (root / "profile.csv").write_text(
        "left_index,right_index,left_second,right_second,review_candidate\n0,1,0,1,True\n"
    )
    result = Store(tmp_path / "ledger.sqlite3").import_directory(root)
    assert result["files"][0]["status"] == "error"


@pytest.mark.parametrize(
    "data", [b"null", b"123", b'[{"unknown": 1}]', b'{"results":[],"results":[]}']
)
def test_unrecognized_json_is_interpretable(data):
    with pytest.raises(ValueError):
        normalize(data, "bad.json")


def test_priority_selection_and_cli_refuses_input_db(tmp_path):
    root = tmp_path / "inputs"
    root.mkdir()
    (root / "video13_candidate_runs.csv").write_text("start,end,length\n1,2,2\n")
    (root / "unsupported.json").write_text("null")
    store = Store(tmp_path / "ledger.sqlite3")
    result = store.import_directory(root, priority_only=True)
    assert result["excluded_by_priority_selection"] == ["unsupported.json"]
    assert result["files"][0]["status"] == "valid"
    unsafe = root / "unsafe.sqlite3"
    response = CliRunner().invoke(app, ["import", str(root), "--db", str(unsafe)])
    assert response.exit_code != 0 and not unsafe.exists()
