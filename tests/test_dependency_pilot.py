"""Offline pair semantics, independent annotation, checksums and persistence."""

import itertools
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from flir_pipeline.alignment.dependency import (
    BOUNDARIES,
    compare,
    describe,
    load,
    select,
)
from flir_pipeline.alignment.dependency_store import (
    DependencyStore,
    PairDecision,
    summarize,
)
from flir_pipeline.alignment.inventory import encoded, file_sha


def synthetic_pilot(root):
    root = Path(root)
    root.mkdir(parents=True)
    media = {}
    for i in range(8):
        path = root / f"synthetic-{i}.png"
        Image.new("RGB", (40, 30), (i * 30, 50, 150)).save(path)
        media[f"m{i}"] = {"path": path.name, "sha256": file_sha(path)}
    pairs = []
    for i in range(4):

        def side(n):
            return {
                "image_id": f"image-{n}",
                "original": {
                    "dataset_index": n,
                    "original_split": "train" if n % 2 else "test",
                },
                "media": [f"m{n}"],
                "alignment_id": f"alignment-{n}",
                "alignment_metrics": {"second": n * 2},
                "time_uncertainty": "synthetic",
            }

        pairs.append(
            {
                "pair_id": f"pair-{i}",
                "category": "diagnostic",
                "a": side(i * 2),
                "b": side(i * 2 + 1),
                "media": [f"m{i * 2}", f"m{i * 2 + 1}"],
                "metrics": {},
                "secondary": {"cross_split": True},
                "status": "pending",
            }
        )
    data = {
        "schema": "dependency-pilot-v1",
        "pairs": pairs,
        "media": media,
        "second_reviewer_pairs": [p["pair_id"] for p in pairs[:2]],
        "method": {},
        "source_video": "synthetic.avi",
    }
    (root / "pilot.json").write_bytes(encoded(data))
    (root / "checksums.json").write_bytes(
        encoded({p.name: file_sha(p) for p in root.iterdir() if p.is_file()})
    )
    return root


@pytest.fixture
def store(tmp_path):
    return DependencyStore(synthetic_pilot(tmp_path / "pilot"), tmp_path / "r1.sqlite3")


def decision(**updates):
    return dict(
        pair_id="pair-0",
        expected_version=0,
        status="reviewed",
        continuity="shot_change",
        spatial="same_region",
        confidence="medium",
        justification="Synthetic terrain reference",
        reviewer="synthetic reviewer",
        evidence_consulted=["m0", "m1"],
        **updates,
    )


def test_axes_and_restart(store):
    first = store.save(decision())
    row = decision()
    row.update(
        expected_version=1,
        continuity="uncertain",
        spatial="uncertain",
        confidence="undetermined",
    )
    store.save(row)
    restarted = DependencyStore(store.root, store.path)
    assert restarted.history()[0] == first
    assert len(restarted.history()) == 2
    assert b"shot_change" in restarted.export("csv")
    assert json.loads(restarted.export("json"))["initial_pending"] == [
        "pair-1",
        "pair-2",
        "pair-3",
    ]
    assert restarted.report()["agreement"] is None


@pytest.mark.parametrize(
    "change",
    [
        {"evidence_consulted": []},
        {"evidence_consulted": ["m0"]},
        {"evidence_consulted": ["m0", "m2"]},
        {"confidence": "undetermined"},
        {"spatial": "correct_match"},
        {"continuity": "same_region"},
        {"justification": " "},
    ],
)
def test_invalid_reviews(store, change):
    row = decision()
    row.update(change)
    with pytest.raises(ValueError):
        store.save(row)
    assert store.history() == []


def test_pending_uncertain_conflict(store):
    store.save({"pair_id": "pair-0", "expected_version": 0})
    with pytest.raises(ValueError, match="conflict"):
        store.save(decision())
    row = decision()
    row.update(
        expected_version=1,
        continuity="uncertain",
        spatial="uncertain",
        confidence="undetermined",
    )
    store.save(row)


def test_blind_and_adjudication(store, tmp_path):
    first = store.save(decision())
    second = DependencyStore(store.root, tmp_path / "r2.sqlite3", "reviewer2")
    assert second.snapshot()["history"] == []
    assert second.snapshot()["adjudication_sources"] == []
    assert len(second.snapshot()["pilot"]["pairs"]) == 2
    with pytest.raises(ValueError, match="assigned"):
        second.save({"pair_id": "pair-3", "expected_version": 0})
    with pytest.raises(ValueError, match="assigned"):
        second.media("m7")
    with pytest.raises(ValueError):
        DependencyStore(store.root, store.path, "reviewer2")
    row = decision()
    row["spatial"] = "different_region"
    latter = second.save(row)
    report = summarize(store.data, store.history() + second.history())
    assert report["agreement"]["spatial_fraction"] == 0
    adjudicator = DependencyStore(
        store.root, tmp_path / "a.sqlite3", "adjudicator", (store.path, second.path)
    )
    with pytest.raises(ValueError):
        adjudicator.save(decision())
    row["adjudicates"] = [first["id"], latter["id"]]
    adjudicator.save(row)
    assert len(store.history()) == len(second.history()) == 1


def test_media_tampering_and_foreign_database(store, tmp_path):
    (store.root / "synthetic-0.png").write_bytes(b"changed")
    with pytest.raises(ValueError):
        store.save(decision())
    with pytest.raises(ValueError):
        load(store.root)


def test_selection_reproducible_exclusive_diverse():
    indices = [
        6,
        48,
        88,
        168,
        199,
        200,
        263,
        349,
        350,
        351,
        357,
        364,
        372,
        399,
        400,
        410,
        421,
        436,
        490,
        499,
        501,
        539,
        630,
        736,
    ]
    cases = [
        {
            "original": {
                "dataset_index": i,
                "original_split": ("train", "val", "test")[n % 3],
            },
            "alternatives": [{"metrics": {"second": float(i)}}],
        }
        for n, i in enumerate(indices)
    ]
    metrics = {
        p: {
            "inliers": sum(p) % 15,
            "mean_change": abs(p[0] - p[1]),
            "contrast_change": sum(p) % 7,
        }
        for p in itertools.combinations(range(24), 2)
    }
    selected, deficits = select(cases, metrics, 42)
    assert (selected, deficits) == select(cases, metrics, 42)
    assert not deficits and len(selected) == len({p for p, _ in selected}) == 24
    assert {
        tuple(sorted(indices[i] for i in p)) for p, c in selected if c == "diagnostic"
    } == set(BOUNDARIES)
    assert len({i for p, c in selected if c == "negative_candidate" for i in p}) == 8
    assert 500 not in [indices[i] for p, _ in selected for i in p]


def test_no_sift_does_not_make_spatial_label():
    feature = describe(np.zeros((216, 384, 3), np.uint8))
    result = compare(feature, feature, 1)
    assert result["inliers"] == 0 and not result["overlay_valid"]
    assert "spatial" not in result
    assert not feature["mask"][:43].any()


def test_decision_schema():
    from jsonschema import Draft202012Validator

    schema = PairDecision.model_json_schema()
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    validator.validate(decision())
    bad = decision()
    bad["continuity"] = "correct_match"
    assert list(validator.iter_errors(bad))


def test_published_bundle_and_decision_schema(store):
    from jsonschema import Draft202012Validator

    base = Path(__file__).parents[1] / "docs/contracts"
    export_schema = json.loads(
        (base / "dependency-review-export-v1.schema.json").read_bytes()
    )
    Draft202012Validator.check_schema(export_schema)
    store.save(decision())
    Draft202012Validator(export_schema).validate(json.loads(store.export("json")))
    for name, valid in (
        ("dependency-pilot-v1", store.data),
        ("dependency-decision-input-v1", decision()),
    ):
        schema = json.loads((base / f"{name}.schema.json").read_bytes())
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(schema)
        validator.validate(valid)
        invalid = dict(valid)
        del invalid["pairs" if name == "dependency-pilot-v1" else "pair_id"]
        assert list(validator.iter_errors(invalid))


def test_reference_history_remains_unchanged(tmp_path):
    import runpy

    from flir_pipeline.alignment.dependency import references
    from flir_pipeline.alignment.pilot_store import PilotStore

    fixture = runpy.run_path(str(Path(__file__).with_name("test_alignment_pilot.py")))
    root = tmp_path / "alignment"
    fixture["synthetic_pilot"](root)
    old = PilotStore(root, tmp_path / "alignment.sqlite3")
    old.save(fixture["decision"](old))
    export = tmp_path / "reference.json"
    export.write_bytes(old.export())
    before = file_sha(export), file_sha(old.path)
    _, latest, accepted = references(root, export)
    assert len(latest) == len(accepted) == 1
    assert (file_sha(export), file_sha(old.path)) == before
    data = json.loads(export.read_bytes())
    data["history"][0]["evidence_hashes"][
        next(iter(data["history"][0]["evidence_hashes"]))
    ] = "0" * 64
    export.write_bytes(encoded(data))
    with pytest.raises(ValueError, match="hash"):
        references(root, export)


def test_dependency_http_handler(store):
    import io

    from flir_pipeline.alignment.pilot_server import make_server

    class RequestSocket:
        def __init__(self, raw):
            self.raw = io.BytesIO(raw)
            self.output = bytearray()

        def makefile(self, *_args):
            return self.raw

        def sendall(self, data):
            self.output.extend(data)

    server = make_server(store, 0, "dependency_ui.html")
    try:

        def request(body, origin):
            raw = encoded(body)
            headers = f"POST /api/decisions HTTP/1.0\r\nHost: 127.0.0.1:{server.server_port}\r\nOrigin: {origin}\r\nContent-Type: application/json\r\nX-Alignment-Review: 1\r\nContent-Length: {len(raw)}\r\n\r\n"
            sock = RequestSocket(headers.encode() + raw)
            server.RequestHandlerClass(sock, ("127.0.0.1", 1), server)
            head, payload = bytes(sock.output).split(b"\r\n\r\n", 1)
            return int(head.split()[1]), json.loads(payload)

        assert request(decision(), "https://external.invalid")[0] == 403
        row = decision()
        row["evidence_consulted"] = []
        assert request(row, f"http://127.0.0.1:{server.server_port}")[0] == 400
        status, saved = request(decision(), f"http://127.0.0.1:{server.server_port}")
        assert (
            status == 201
            and saved["continuity"] == "shot_change"
            and saved["spatial"] == "same_region"
        )
        assert request(decision(), f"http://127.0.0.1:{server.server_port}")[0] == 409
    finally:
        server.server_close()
