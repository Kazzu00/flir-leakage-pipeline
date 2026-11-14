"""Offline scientific invariants for paired timelines and proposed partitions."""

import copy
import io
import tarfile

import numpy as np
import pytest
from PIL import Image

from flir_pipeline.explorer.paired_inputs import digest, safe_relative
from flir_pipeline.explorer.paired_timeline import (
    TarImages,
    pixel_metrics,
    render_html,
    sample_pairs,
)
from flir_pipeline.splitting.paired import (
    assign,
    build_splits,
    groups_for,
    relation_rows,
    verify_records,
    verify_saved,
)


def pairs_fixture(n=12):
    pairs = []
    for i in range(n):
        row = {
            "pair_id": f"p{i:02}",
            "filename": f"image{i}.png",
            "original_name": f"old{i}.jpg",
            "source_video": f"video{i % 2}",
            "video": f"v{i % 2}",
            "frame_index": i,
            "timestamp_seconds": float(i),
            "sequence_id": "unresolved",
            "temporal_status": "supplier_reported",
            "reference_content_id": f"reference{i}",
            "historical_split": "train",
            "status": "eligible",
            "reasons": [],
            "label_empty": i % 3 == 0,
            "classes": [] if i % 3 == 0 else [i % 5],
            "boxes": [] if i % 3 == 0 else [[i % 5, 0.5, 0.5, 0.2, 0.2]],
            "num_objects": int(i % 3 != 0),
            "width": 8,
            "height": 6,
            "protected_reference": False,
            "candidate_dependency_groups": [],
            "historical_decisions": [],
            "thumbnails": {
                "original": "data:image/jpeg;base64,",
                "no_hud": "data:image/jpeg;base64,",
            },
            "pixel_metrics": {"boxes_intersecting_changed_pixels": i % 2},
        }
        for s in ("original", "no_hud"):
            row[s] = {
                "inventory_occurrence_id": f"{s}{i}",
                "source": f"{s}.tar",
                "member": f"images/{i}.png",
                "sha256": f"{s}{i}",
                "label_sha256": f"label{i}",
                "label_members": [f"labels/{i}.txt"],
            }
        pairs.append(row)
    return pairs


def relation(a, b, status="accepted"):
    return {
        "relation_id": a + b,
        "left_pair_id": a,
        "right_pair_id": b,
        "status": status,
        "kind": "dependency",
    }


CFG = {"ratios": [0.7, 0.15, 0.15], "seed": 42}


def test_exact_duplicate_transitive_closure_across_variants():
    p = pairs_fixture()
    p[1]["no_hud"]["sha256"] = p[0]["original"]["sha256"]
    groups = groups_for(p, [relation("p01", "p02")], True)
    assert groups["p00"] == groups["p01"] == groups["p02"]


@pytest.mark.parametrize("status", ["candidate", "ambiguous", "rejected"])
def test_candidate_relations_are_not_accepted(status):
    groups = groups_for(pairs_fixture(), [relation("p00", "p01", status)], True)
    assert groups["p00"] != groups["p01"]


def test_no_whole_video_grouping_and_determinism():
    p = pairs_fixture()
    first, _ = assign(p, [], CFG)
    second, _ = assign(list(reversed(p)), [], CFG)
    assert first == second
    assert len({r["grouped_group"] for r in first}) == len(p)
    assert all(r["random_split"] == r["grouped_split"] for r in first)
    assert verify_records(first, p, [])["pair_indivisibility"]


def test_protection_and_shared_population_propagate():
    p = pairs_fixture()
    p[0].update(status="quarantine", reasons=["protected"], protected_reference=True)
    rel = [relation("p00", "p01")]
    records, population = assign(p, rel, CFG)
    assert population[1]["status"] == "quarantine"
    assert all(
        not r["random_split"] and not r["grouped_split"]
        for r in records
        if r["pair_id"] in {"p00", "p01"}
    )
    verify_records(records, p, rel)


def test_independent_verifier_detects_tampering():
    p = pairs_fixture()
    records, _ = assign(p, [], CFG)
    records[0]["grouped_split"] = (
        "test" if records[0]["grouped_split"] != "test" else "train"
    )
    with pytest.raises(ValueError, match="Pair indivisibility"):
        verify_records(records, p, [])


def test_changed_pixels_and_boxes_not_mask_or_acceptance():
    a = np.zeros((10, 10, 3), dtype=np.uint8)
    b = a.copy()
    b[:2, :2] = 255
    metrics = pixel_metrics(a, b, [[0, 0.1, 0.1, 0.2, 0.2], [1, 0.8, 0.8, 0.2, 0.2]])
    assert metrics["changed_pixel_fraction"] == 0.04
    assert metrics["box_changed_fractions"] == [1.0, 0.0]
    assert metrics["boxes_intersecting_changed_pixels"] == 1


def test_sample_strata_are_order_independent():
    p = pairs_fixture(30)
    sample = sample_pairs(p, 42, 8)
    assert sample == sample_pairs(list(reversed(p)), 42, 8)
    assert all(
        any(f"{v}:empty" in reasons for reasons in sample.values())
        for v in ["video0", "video1"]
    )


def test_html_is_offline_and_escapes_supplier_text(tmp_path):
    p = pairs_fixture()
    p[0]["filename"] = "</script><script>alert(1)</script>"
    render_html(p, tmp_path / "timeline.html")
    html = (tmp_path / "timeline.html").read_text(encoding="utf-8")
    assert "<script>alert(1)" not in html
    assert "https://" not in html and "fetch(" not in html
    assert "r.source_video===video.value" in html


def test_safe_root_rejects_escape(tmp_path):
    with pytest.raises(ValueError):
        safe_relative(tmp_path, "../outside")


def test_tar_reader_checks_bytes_without_extracting(tmp_path):
    import hashlib

    data = io.BytesIO()
    Image.new("RGB", (8, 6)).save(data, format="PNG")
    raw = data.getvalue()
    with tarfile.open(tmp_path / "source.tar", "w") as archive:
        info = tarfile.TarInfo("images/a.png")
        info.size = len(raw)
        archive.addfile(info, io.BytesIO(raw))
    row = {
        "source": "source.tar",
        "member": "images/a.png",
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    reader = TarImages(tmp_path)
    try:
        assert reader.read(row).size == (8, 6)
        with pytest.raises(ValueError, match="checksum"):
            reader.read({**row, "sha256": digest("wrong")})
    finally:
        reader.close()
    assert not (tmp_path / "images").exists()


def test_invalid_grouped_edge_does_not_silently_drop():
    p = pairs_fixture()
    with pytest.raises(KeyError):
        groups_for(p, [relation("p00", "unknown")], True)


def test_saved_delivery_integration_and_replay(tmp_path):
    from flir_pipeline.explorer.paired_inputs import write_json

    pairs = pairs_fixture()
    write_json(tmp_path / "timeline_records.json", pairs)
    report = build_splits(pairs, CFG, tmp_path, tmp_path)
    assert report["identical_memberships"]
    assert verify_saved(tmp_path)["independent_cli_replay"]
    (tmp_path / "split_manifest.csv").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact changed"):
        verify_saved(tmp_path)


def test_relation_acceptance_requires_evidence(tmp_path):
    from flir_pipeline.explorer.paired_inputs import write_csv

    write_csv(tmp_path / "relations.csv", [relation("p00", "p01")])
    with pytest.raises(ValueError, match="attributed"):
        relation_rows({"relations": "relations.csv"}, tmp_path, pairs_fixture())


def test_independent_verifier_detects_content_relabeling():
    pairs = pairs_fixture()
    records, _ = assign(pairs, [], CFG)
    records[0]["content_id"] = "fake"
    with pytest.raises(ValueError, match="identity"):
        verify_records(records, pairs, [])


def test_independent_verifier_detects_accepted_crossing():
    p = pairs_fixture()
    records, _ = assign(p, [], CFG)
    left = records[0]
    right = next(r for r in records if r["grouped_split"] != left["grouped_split"])
    with pytest.raises(ValueError, match="Accepted relation"):
        verify_records(
            copy.deepcopy(records), p, [relation(left["pair_id"], right["pair_id"])]
        )
