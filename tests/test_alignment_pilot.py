"""Synthetic-only persistence, review guards, selection and descriptive analysis."""

import io
import json
import sqlite3
from pathlib import Path

import pytest
from PIL import Image

from flir_pipeline.alignment.inventory import encoded, file_sha
from flir_pipeline.alignment.pilot import load_pilot, select_pilot
from flir_pipeline.alignment.pilot_server import make_server
from flir_pipeline.alignment.pilot_store import PilotStore


def synthetic_pilot(root):
    root = Path(root)
    root.mkdir(parents=True)
    (root / "media").mkdir()
    cases, media = [], {}
    indices = [
        199,
        200,
        349,
        350,
        399,
        400,
        499,
        501,
        *range(101, 113),
        600,
        610,
        620,
        630,
    ]
    for number, index in enumerate(indices):

        def picture(key, number=number):
            path = root / "media" / key
            image = Image.new("RGB", (96, 64), (number * 10, 80, 140))
            image.save(path)
            media[key] = {"path": f"media/{key}", "sha256": file_sha(path)}
            return key

        original = picture(f"{number}-original.png")
        alternatives = []
        for rank in range(1, 6):
            alternatives.append(
                {
                    "rank": rank,
                    "frame_media": picture(f"{number}-{rank}.png"),
                    "lines_media": None,
                    "overlay_media": None,
                    "close_to_best_inliers": rank > 1,
                    "geometry": {"synthetic": True},
                    "metrics": {
                        "pts": rank,
                        "time_base": "1/25",
                        "second": rank / 25,
                        "inliers": 50 - rank,
                        "matches": 60,
                        "coverage": 0.4,
                        "correlation": 0.8,
                        "dhash_distance": 4,
                        "reprojection_median_px": 0.2,
                    },
                }
            )
        cases.append(
            {
                "alignment_id": f"SYNTHETIC-{number}",
                "image_id": f"SYNTHETIC-IMAGE-{number}",
                "original": {
                    "dataset_index": index,
                    "original_split": ("train", "val", "test")[number % 3],
                    "source_archive_or_path": "SYNTHETIC",
                },
                "selection_reason": "SYNTHETIC fixture",
                "original_media": original,
                "alternatives": alternatives,
                "control": {
                    "media": picture(f"{number}-control.png"),
                    "metrics": {"second": 0},
                },
                "candidate_interval": [0.04, 0.08],
                "automatic_status": "not_found" if index == 630 else "candidate",
                "support_band": "automatic_abstention"
                if index == 630
                else "inliers_ge40",
                "audit_event_links": [],
            }
        )
    data = {
        "schema": "alignment-pilot-v1",
        "pilot_id": "SYNTHETIC",
        "synthetic": True,
        "source_video": "SYNTHETIC.mkv",
        "source_video_sha256": "0" * 64,
        "cases": cases,
        "media": media,
        "selection": {"absent_index_500": True},
    }
    (root / "pilot.json").write_bytes(encoded(data))
    (root / "checksums.json").write_bytes(
        encoded(
            {
                p.relative_to(root).as_posix(): file_sha(p)
                for p in root.rglob("*")
                if p.is_file()
            }
        )
    )
    return data


@pytest.fixture
def store(tmp_path):
    synthetic_pilot(tmp_path / "pilot")
    return PilotStore(tmp_path / "pilot", tmp_path / "reviews.sqlite3")


def decision(store, index=0, **overrides):
    case = store.data["cases"][index]
    return {
        "alignment_id": case["alignment_id"],
        "expected_version": 0,
        "status": "reviewed",
        "label": "correct_match",
        "confidence": "high",
        "reviewer": "SYNTHETIC reviewer",
        "justification": "Synthetic known identity, not FAC evidence",
        "evidence_consulted": [
            case["original_media"],
            case["alternatives"][0]["frame_media"],
        ],
        **overrides,
    }


@pytest.mark.parametrize(
    "overrides",
    [
        {"evidence_consulted": []},
        {"reviewer": " "},
        {"justification": ""},
        {"label": "continuous"},
        {"confidence": "undetermined"},
        {"better_rank": 2},
        {"no_correspondence_in_presented_evidence": True},
        {"control_label": "incorrect_match"},
    ],
)
def test_review_guards(store, overrides):
    with pytest.raises(ValueError):
        store.save(decision(store, **overrides))
    assert not store.snapshot()["history"]


def test_pending_uncertain_cross_case_and_integrity(store):
    case = store.data["cases"][0]
    saved = store.save({"alignment_id": case["alignment_id"], "expected_version": 0})
    assert saved["status"] == "pending" and not saved["human_verified_top1"]
    store.save(
        decision(
            store, expected_version=1, label="uncertain", confidence="undetermined"
        )
    )
    assert store.report()["overall"]["uncertain"] == 1
    with pytest.raises(ValueError, match="belong"):
        store.save(decision(store, 1, evidence_consulted=[case["original_media"]]))
    path = store.root / store.data["media"][case["original_media"]]["path"]
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        store.save(decision(store, expected_version=2))


def test_history_restart_export_and_descriptive_denominators(store):
    for i in range(6):
        store.save(
            decision(
                store,
                i,
                label="correct_match"
                if i < 3
                else "incorrect_match"
                if i < 5
                else "uncertain",
            )
        )
    case = store.data["cases"][0]
    store.save(
        decision(
            store,
            expected_version=1,
            label="incorrect_match",
            better_rank=2,
            evidence_consulted=[
                case["original_media"],
                case["alternatives"][0]["frame_media"],
                case["alternatives"][1]["frame_media"],
                case["control"]["media"],
            ],
            control_label="incorrect_match",
        )
    )
    report = store.report()
    assert report["overall"]["correct_match"] == 2
    assert report["overall"]["correct_among_reviewed"] == 2 / 6
    assert report["overall"]["correct_among_conclusive"] == 2 / 5
    assert report["overall"]["better_than_top1"] == 1
    assert report["controls"]["human_rejected"] == 1
    assert report["by_split"]["train"]["correct_among_reviewed"] is None
    restarted = PilotStore(store.root, store.path)
    assert restarted.export() == store.export()
    assert len(restarted.snapshot()["history"]) == 7
    import csv

    csv_rows = list(csv.DictReader(io.StringIO(restarted.export("csv").decode())))
    assert [
        json.loads(r["decision_json"])
        for r in csv_rows
        if r["record_type"] == "history"
    ] == json.loads(restarted.export())["history"]
    with pytest.raises(ValueError, match="conflict"):
        restarted.save(decision(store))


def test_initial_report_and_foreign_ledger(store, tmp_path):
    exported = json.loads(store.export())
    assert len(exported["initial_pending"]) == 24 and not exported["history"]
    assert store.report()["overall"]["pending"] == 24
    assert store.report()["overall"]["correct_among_reviewed"] is None
    db = tmp_path / "historical.sqlite3"
    with sqlite3.connect(db) as connection:
        connection.execute("CREATE TABLE original (value TEXT)")
    before = file_sha(db)
    with pytest.raises(ValueError, match="foreign"):
        PilotStore(store.root, db)
    assert file_sha(db) == before
    assert load_pilot(store.root)["pilot_id"] == "SYNTHETIC"


def test_abstention_needs_explicit_human_reference(store):
    report = store.report()
    assert report["abstention"]["automatic_count"] == 1
    assert (
        report["abstention"]["fraction_abstained_among_human_no_correspondence"] is None
    )
    store.save(
        decision(
            store,
            23,
            label="incorrect_match",
            no_correspondence_in_presented_evidence=True,
        )
    )
    report = store.report(minimum=1)
    assert report["abstention"]["reference_table"]["abstained_no_correspondence"] == 1
    assert report["abstention"]["fraction_abstained_among_human_no_correspondence"] == 1
    assert (
        store.report()["abstention"]["fraction_abstained_among_human_no_correspondence"]
        is None
    )


def test_loopback_handler_restart_without_network(store):
    class RequestSocket:
        def __init__(self, raw):
            self.raw = io.BytesIO(raw)
            self.output = bytearray()

        def makefile(self, *args):
            return self.raw

        def sendall(self, data):
            self.output.extend(data)

    def request(server, path, body=None, origin=None):
        raw = encoded(body) if body is not None else b""
        headers = f"{'POST' if body is not None else 'GET'} {path} HTTP/1.0\r\nHost: 127.0.0.1:{server.server_port}\r\nContent-Type: application/json\r\nX-Alignment-Review: 1\r\nContent-Length: {len(raw)}\r\n"
        if origin:
            headers += f"Origin: {origin}\r\n"
        socket = RequestSocket(headers.encode() + b"\r\n" + raw)
        server.RequestHandlerClass(socket, ("127.0.0.1", 1), server)
        status, payload = bytes(socket.output).split(b"\r\n\r\n", 1)
        return int(status.split()[1]), json.loads(payload)

    server = make_server(store, 0)
    try:
        status, data = request(server, "/api/pilot")
        assert status == 200 and len(data["pilot"]["cases"]) == 24
        assert (
            request(
                server, "/api/decisions", decision(store), "https://external.invalid"
            )[0]
            == 403
        )
        status, data = request(server, "/api/decisions", decision(store))
        assert status == 201 and data["version"] == 1
    finally:
        server.server_close()
    restarted = make_server(PilotStore(store.root, store.path), 0)
    try:
        status, data = request(restarted, "/api/pilot")
        assert status == 200 and data["history"][0]["label"] == "correct_match"
    finally:
        restarted.server_close()


def test_selection_balanced_unique_and_missing():
    cases, rows = [], []
    diagnostics = {
        199: "train",
        200: "test",
        349: "test",
        350: "val",
        399: "val",
        400: "test",
        499: "test",
        501: "train",
        630: "train",
    }
    for i, (index, split) in enumerate(
        [
            *diagnostics.items(),
            *[(1000 + i, s) for i in range(15) for s in ("train", "val", "test")],
        ]
    ):
        rows.append(
            {"image_id": str(i), "dataset_index": index, "original_split": split}
        )
        cases.append(
            {
                "image_id": str(i),
                "alignment_id": str(i),
                "top_k": [{"inliers": 50}, {"inliers": 49}],
            }
        )
    first = select_pilot({"cases": cases}, {"images": rows})
    assert first == select_pilot({"cases": cases}, {"images": rows})
    assert len({r["image_id"] for r in first["selected"]}) == 24
    for split in ("train", "val", "test"):
        assert (
            sum(
                r["selection_reason"] == f"additional_{split}"
                for r in first["selected"]
            )
            == 4
        )
    assert sum(r["selection_reason"] == "diagnostic" for r in first["selected"]) == 8
    assert first["absent_index_500"]
    with pytest.raises(ValueError, match="Diagnostic"):
        select_pilot({"cases": cases[1:]}, {"images": rows})
