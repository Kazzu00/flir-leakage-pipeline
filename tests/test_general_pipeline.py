"""Independent, offline non-FAC populations and scientific evidence invariants."""

import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest
import yaml
from PIL import Image
from pydantic import ValidationError
from typer.testing import CliRunner

from flir_pipeline.general.analysis import (
    analyze,
    dependency_groups,
    leakage_report,
    residual_similarity,
)
from flir_pipeline.general.cli import app
from flir_pipeline.general.contract import (
    Evidence,
    Manifest,
    Position,
    Relation,
    digest,
)
from flir_pipeline.general.ingestion import ingest, verify_sources
from flir_pipeline.general.publication import build, verify
from flir_pipeline.general.timeline import render_timeline
from flir_pipeline.splitting.base import SplitConfig

CFG = SplitConfig(target_ratios=(0.6, 0.2, 0.2))


def fixture_dataset(root, nclasses=2, labeled=True, videos=0):
    root.mkdir(parents=True, exist_ok=True)
    classes = (
        [{"class_id": 10 + 3 * i, "name": f"category {i}"} for i in range(nclasses)]
        if labeled
        else []
    )
    rows = []
    for i in range(12):
        image = f"observation-{i}.png"
        # Last occurrence duplicates the first bytes, without collapsing IDs.
        Image.new("RGB", (9, 7), (i % 11, 29, 51)).save(root / image)
        label = f"observation-{i}.txt"
        if labeled:
            (root / label).write_text(
                ""
                if i % 4 == 0
                else f"{classes[i % nclasses]['class_id']} 0.5 0.5 0.2 0.2\n",
                encoding="utf8",
            )
        rows.append(
            {
                "frame_id": f"occurrence-{i}",
                "image": image,
                "label": label if labeled else None,
                "original_split": ["train", "val", "test"][i % 3],
            }
        )
    ev = root / "evidence.json"
    ev.write_text(
        '{"fixture": "synthetic evidence, not real decoded video"}', encoding="utf8"
    )
    evidence = Evidence(
        path=ev.name,
        sha256=hashlib.sha256(ev.read_bytes()).hexdigest(),
        method="synthetic_test",
        version="1",
        provenance="offline_fixture",
    )
    vrows = []
    for i in range(videos):
        path = root / f"camera-{i}.bin"
        path.write_bytes(f"mock source video {i}".encode())
        vrows.append(
            {
                "video_id": f"camera-{i}",
                "path": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    if videos:
        for i, row in enumerate(rows):
            row["positions"] = [
                Position(
                    video_id=f"camera-{i % videos}",
                    status="confirmed",
                    kind="decoded_pts",
                    seconds=float(i),
                    pts=i * 25,
                    time_base="1/25",
                    decoded_ordinal=i * 25,
                    decoded_rgb_sha256="a" * 64,
                    evidence=evidence,
                ).model_dump()
            ]
    path = root / "input.yaml"
    path.write_text(
        yaml.safe_dump({"classes": classes, "images": rows, "videos": vrows}),
        encoding="utf8",
    )
    return ingest(path, root), evidence


@pytest.mark.parametrize(
    "nclasses,labeled,videos", [(2, True, 0), (7, True, 4), (0, False, 0), (2, True, 2)]
)
def test_independent_datasets_replay_and_coverage(tmp_path, nclasses, labeled, videos):
    manifest, _ = fixture_dataset(tmp_path / "input", nclasses, labeled, videos)
    result = analyze(manifest, CFG)
    assert result["population"]["eligible_occurrences"] == 12
    assert all(r["state"] == "assigned" for r in result["runs"])
    assert all(r["memberships_equal"] for r in result["equivalence"])
    for run in result["runs"]:
        assert len(run["assignments"]) == 12
        assert run["assignments"]["occurrence-0"] == run["assignments"]["occurrence-11"]
        assert run["diagnostics"]["exact_duplicates"]["crossing_content_groups"] == 0
    assert (
        result["historical_diagnostics"]["exact_duplicates"]["crossing_content_groups"]
        == 1
    )
    reordered = Manifest.model_validate(
        {
            **manifest.model_dump(),
            "records": list(reversed(manifest.records)),
            "classes": list(reversed(manifest.classes)),
        }
    )
    assert digest(analyze(reordered, CFG)) == digest(result)
    assert manifest.dataset_id() == reordered.dataset_id()
    assert (render_timeline(manifest, result) is not None) == bool(videos)
    if not labeled:
        assert result["effective_balance_columns"] == ("record_count",)
        assert all(r.annotation_status == "missing" for r in manifest.records)
    output = tmp_path / "publication"
    build(manifest, CFG, tmp_path / "input", output)
    assert verify(output, tmp_path / "input")["verified"]
    with pytest.raises(ValueError, match="already exists"):
        build(manifest, CFG, tmp_path / "input", output)


def relation(manifest, ev, a, b, status="confirmed", accepted=True):
    evidence = ev.model_copy(
        update={"reviewer": "synthetic reviewer", "reviewed_at": "2026-01-01T00:00:00Z"}
    )
    return Relation(
        relation_id=f"edge-{a}-{b}-{status}",
        left=manifest.records[a].content_id,
        right=manifest.records[b].content_id,
        kind="visual",
        status=status,
        accepted=accepted,
        evidence=evidence,
    )


def test_accepted_rejected_and_transitive_conflicts(tmp_path):
    m, ev = fixture_dataset(tmp_path)
    m.relations = [
        relation(m, ev, 0, 1),
        relation(m, ev, 1, 2),
        relation(m, ev, 0, 2, "rejected", False),
    ]
    groups, conflicts = dependency_groups(m)
    assert len(conflicts) == 1
    assert groups[m.records[0].content_id] == groups[m.records[2].content_id]
    result = analyze(m, CFG)
    assert all(
        r["state"] == "blocked_conflicting_constraints" for r in result["runs"][5:]
    )
    assert all(r["state"] == "assigned" for r in result["runs"][:5])


def test_candidate_edges_never_group_and_crossings_measured(tmp_path):
    m, ev = fixture_dataset(tmp_path, labeled=False)
    m.relations = [
        relation(m, ev, 0, 1, "candidate", False),
        relation(m, ev, 1, 2, "ambiguous", False),
    ]
    result = analyze(m, CFG)
    assert all(x["memberships_equal"] for x in result["equivalence"])
    diag = result["historical_diagnostics"]
    assert diag["relations"]["candidate"]["crossing"] == 1
    assert diag["relations"]["accepted"]["total"] == 0
    diag = leakage_report(m, {"occurrence-0": "train"}, result["graph"]["membership"])
    assert diag["relations"]["candidate"]["unknown"] == 1


def test_accepted_atomicity_shared_exclusions_and_missingness(tmp_path):
    m, ev = fixture_dataset(tmp_path)
    m.relations = [relation(m, ev, 0, 1)]
    result = analyze(m, CFG)
    for run in result["runs"][5:]:
        assert run["assignments"]["occurrence-0"] == run["assignments"]["occurrence-1"]
    m.records[0].exclusion_reasons = ["protected_reference"]
    result = analyze(m, CFG)
    assert set(result["population"]["excluded_frame_ids"]) == {
        "occurrence-0",
        "occurrence-1",
        "occurrence-11",
    }
    assert all(len(r["assignments"]) == 9 for r in result["runs"])


def test_empty_conflicts_multiple_positions_and_infeasibility(tmp_path):
    m, ev = fixture_dataset(tmp_path, videos=2)
    m.records[0].positions.append(
        m.records[0]
        .positions[0]
        .model_copy(update={"pts": 250, "decoded_ordinal": 250, "seconds": 10.0})
    )
    m.relations = [relation(m, ev, 0, i) for i in range(1, 11)]
    result = analyze(m, CFG)
    assert result["temporal_coverage"]["multiple_confirmed"] == 1
    assert (
        result["historical_diagnostics"]["annotation_byte_conflicts"]["content_groups"]
        == 1
    )
    assert all(r["state"] == "infeasible" for r in result["runs"][5:])
    assert result["runs"][5]["assignments"] == {}
    assert any(r.annotation_status == "empty" for r in m.records)


@pytest.mark.parametrize(
    "mutation",
    [
        "unknown_class",
        "bad_hash",
        "duplicate_id",
        "nan",
        "missing_label",
        "false_pts",
        "nominal_confirmation",
        "unknown_video",
        "unknown_endpoint",
        "candidate_acceptance",
        "unattributed_acceptance",
    ],
)
def test_corrupt_contracts_fail_closed(tmp_path, mutation):
    m, ev = fixture_dataset(tmp_path, videos=1)
    data = m.model_dump()
    if mutation == "unknown_class":
        data["records"][1]["boxes"][0][0] = 999
    elif mutation == "bad_hash":
        data["records"][0]["content_id"] = "b" * 64
    elif mutation == "duplicate_id":
        data["records"][1]["frame_id"] = data["records"][0]["frame_id"]
    elif mutation == "nan":
        data["records"][1]["boxes"][0][1] = float("nan")
    elif mutation == "missing_label":
        data["records"][1]["label_sha256"] = None
    elif mutation == "false_pts":
        data["records"][1]["positions"][0]["seconds"] = 99
    elif mutation == "nominal_confirmation":
        data["records"][0]["positions"][0]["kind"] = "nominal_sample"
    elif mutation == "unknown_video":
        data["records"][0]["positions"][0]["video_id"] = "unknown"
    else:
        edge = relation(m, ev, 0, 1).model_dump()
        if mutation == "unknown_endpoint":
            edge["right"] = "f" * 64
        elif mutation == "candidate_acceptance":
            edge["status"] = "candidate"
        else:
            edge["evidence"]["reviewer"] = None
        data["relations"] = [edge]
    with pytest.raises((ValueError, ValidationError)):
        Manifest.model_validate(data)


def test_corrupt_files_and_unchanged_annotations(tmp_path):
    m, _ = fixture_dataset(tmp_path)
    labels = {p.name: p.read_bytes() for p in tmp_path.glob("*.txt")}
    analyze(m, CFG)
    assert labels == {p.name: p.read_bytes() for p in tmp_path.glob("*.txt")}
    (tmp_path / "observation-0.png").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        verify_sources(m, tmp_path)
    with pytest.raises(OSError):
        ingest(tmp_path / "input.yaml", tmp_path)


def test_invalid_geometry_retained_and_excluded(tmp_path):
    fixture_dataset(tmp_path)
    (tmp_path / "observation-1.txt").write_text(
        "10 0.99 0.99 0.5 0.5\n", encoding="utf8"
    )
    m = ingest(tmp_path / "input.yaml", tmp_path)
    assert m.records[1].annotation_status == "invalid"
    assert m.records[1].boxes == [[10, 0.99, 0.99, 0.5, 0.5]]
    assert "occurrence-1" in analyze(m, CFG)["population"]["excluded_frame_ids"]


def test_residual_similarity_and_optional_npz(tmp_path):
    m, _ = fixture_dataset(tmp_path / "input")
    ids = sorted({r.content_id for r in m.records})
    matrix = np.eye(len(ids), dtype=np.float64)
    report = residual_similarity(m, analyze(m, CFG), ids, matrix, "synthetic_space")
    assert all(r["maximum_cross_split_cosine"] == 0 for r in report["rows"])
    with pytest.raises(ValueError, match="L2"):
        residual_similarity(m, analyze(m, CFG), ids, matrix * 0, "synthetic_space")
    features = tmp_path / "features.npz"
    np.savez(
        features,
        content_ids=ids,
        embeddings_l2=matrix,
        feature_space_id="synthetic_space",
    )
    build(m, CFG, tmp_path / "input", tmp_path / "pub", [features])
    assert verify(tmp_path / "pub", tmp_path / "input")["verified"]
    (tmp_path / "pub/analysis.json").write_text("{}", encoding="utf8")
    with pytest.raises(ValueError, match="checksum"):
        verify(tmp_path / "pub", tmp_path / "input")


def test_cli_schema_and_build_verify(tmp_path):
    m, _ = fixture_dataset(tmp_path / "input")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(m.canonical()), encoding="utf8")
    config = tmp_path / "split.yaml"
    config.write_text("target_ratios: [0.6, 0.2, 0.2]\n", encoding="utf8")
    runner = CliRunner()
    assert runner.invoke(app, ["schema"]).exit_code == 0
    args = [str(manifest), str(config), str(tmp_path / "input"), str(tmp_path / "pub")]
    result = runner.invoke(app, ["build", *args])
    assert result.exit_code == 0, result.exception
    assert (
        runner.invoke(
            app, ["verify", str(tmp_path / "pub"), str(tmp_path / "input")]
        ).exit_code
        == 0
    )
    with pytest.raises(ValueError, match="seeds"):
        analyze(m, replace(CFG, seeds=(42,)))


def test_decoded_localization_retains_multiple_and_unknown(tmp_path, monkeypatch):
    from fractions import Fraction

    from flir_pipeline.general.localization import localize

    m, _ = fixture_dataset(tmp_path, videos=2)
    for r in m.records:
        r.positions = []
    rgb = np.array(Image.open(tmp_path / "observation-0.png"))

    def frames(_path):
        yield 0, 4, Fraction(1, 2), rgb
        yield 1, 8, Fraction(1, 2), rgb
        yield 2, None, Fraction(1, 2), rgb

    monkeypatch.setattr("flir_pipeline.general.localization.decoded_frames", frames)
    localized = localize(m, tmp_path, tmp_path / "operations/exact.json")
    assert len(localized.records[0].positions) == 4
    assert len(localized.records[11].positions) == 4
    assert localized.records[1].positions == []
    assert m.dataset_id() == localized.dataset_id()
    assert all(p.seconds in (2, 4) for p in localized.records[0].positions)
    assert (
        json.loads((tmp_path / "operations/exact.json").read_text())["videos"][
            "camera-0"
        ]["frames_without_pts"]
        == 1
    )


def test_real_synthetic_video_decode_when_optional_av_available(tmp_path):
    av = pytest.importorskip("av")
    from fractions import Fraction

    from flir_pipeline.general.contract import Video
    from flir_pipeline.general.localization import localize

    m, _ = fixture_dataset(tmp_path)
    path = tmp_path / "synthetic.avi"
    rgb = np.array(Image.open(tmp_path / "observation-0.png"))
    with av.open(str(path), mode="w") as out:
        stream = out.add_stream("ffv1", rate=2)
        stream.width, stream.height, stream.pix_fmt = 9, 7, "bgr0"
        for i in range(3):
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            frame.pts, frame.time_base = i, Fraction(1, 2)
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    m.videos = [
        Video(
            video_id="synthetic",
            path=path.name,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
    ]
    localized = localize(m, tmp_path, tmp_path / "observations.json")
    assert [p.seconds for p in localized.records[0].positions] == [0, 0.5, 1]
    assert localized.records[1].positions == []


def test_solver_failure_preserved_without_fallback(tmp_path, monkeypatch):
    from types import SimpleNamespace

    m, _ = fixture_dataset(tmp_path)
    monkeypatch.setattr(
        "flir_pipeline.splitting.construction.milp",
        lambda *a, **k: SimpleNamespace(
            x=None, status=2, message="synthetic infeasible"
        ),
    )
    result = analyze(m, CFG)
    assert all(
        r["solver"]["status"] == 2 and not r["assignments"] for r in result["runs"]
    )


def test_original_split_never_enters_solver(tmp_path):
    m, _ = fixture_dataset(tmp_path)
    before = analyze(m, CFG)
    for r in m.records:
        r.original_split = None
    after = analyze(m, CFG)
    assert [r["membership_sha256"] for r in before["runs"]] == [
        r["membership_sha256"] for r in after["runs"]
    ]
    assert after["historical_diagnostics"]["coverage"]["assigned"] == 0


def test_manifest_false_member_identity_rejected_even_with_valid_source_receipt(
    tmp_path,
):
    m, _ = fixture_dataset(tmp_path)
    m.records[0].content_id = m.records[0].image_sha256 = "a" * 64
    with pytest.raises(ValueError, match="Occurrence byte identity"):
        verify_sources(m, tmp_path)


def test_legacy_and_explicit_five_class_objective_equivalent():
    import pandas as pd

    from flir_pipeline.splitting.base import BALANCE_COLUMNS
    from flir_pipeline.splitting.construction import milp_assignment

    units = pd.DataFrame(
        [
            {
                "group_id": str(i),
                "record_count": 1,
                **{f"class_{c}": int(c == i % 5) for c in range(5)},
                "empty_annotation_count": 0,
            }
            for i in range(12)
        ]
    )
    for seed in range(5):
        original, _ = milp_assignment(units, np.array(CFG.target_ratios), seed, CFG)
        explicit, _ = milp_assignment(
            units,
            np.array(CFG.target_ratios),
            seed,
            CFG,
            balance_columns=BALANCE_COLUMNS,
            family_weights=(1, 0.2, 0.2, 0.2, 0.2, 0.2, 1),
        )
        assert original == explicit


def test_shuffled_relations_have_identical_full_replay(tmp_path):
    m, ev = fixture_dataset(tmp_path / "input")
    m.relations = [relation(m, ev, 0, i, "candidate", False) for i in range(1, 11)]
    original = analyze(m, CFG)
    m.relations.reverse()
    assert digest(analyze(m, CFG)) == digest(original)
    build(m, CFG, tmp_path / "input", tmp_path / "publication")
    assert verify(tmp_path / "publication", tmp_path / "input")["verified"]
