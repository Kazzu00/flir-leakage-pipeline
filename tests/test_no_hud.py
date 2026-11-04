"""Offline scientific gates: similarity never authorizes annotation transfer."""

import hashlib
import io
import zipfile

import numpy as np
import pytest
from PIL import Image

from flir_pipeline.data.no_hud import (
    enforce_identity_splits,
    geometry,
    inventory,
    provenance_valid,
    transfer,
)


def raster():
    return Image.fromarray(
        np.random.default_rng(9).integers(0, 255, (80, 100, 3), dtype=np.uint8)
    )


def test_identity_and_scale_preserve_normalized_bytes():
    im = raster()
    raw = b"0 0.5 0.5 0.2 0.2\n"
    for size in (im.size, (50, 40)):
        target = im.resize(size, Image.Resampling.BILINEAR)
        geom = geometry(im, target, [])
        assert geom["valid"]
        assert transfer(raw, hashlib.sha256(raw).hexdigest(), geom) == (raw, [])


def test_changed_region_box_warning_and_unexplained_changes():
    im = raster()
    target = im.copy()
    target.paste((0, 0, 0), (0, 0, 20, 20))
    assert not geometry(im, target, [])["valid"]
    geom = geometry(im, target, [[0, 0, 0.2, 0.25]])
    raw = b"1 0.15 0.15 0.2 0.2\n"
    assert transfer(raw, hashlib.sha256(raw).hexdigest(), geom)[1] == [0]


@pytest.mark.parametrize(
    "raw", [b"5 .5 .5 .1 .1", b"0 0 0 .2 .2", b"0 nan .5 .1 .1", b"0 .5 .5 0 .1"]
)
def test_invalid_annotations_not_repaired(raw):
    geom = geometry(raster(), raster(), [])
    with pytest.raises(ValueError):
        transfer(raw, hashlib.sha256(raw).hexdigest(), geom)


def test_checksum_empty_labels_and_geometry_fail_closed():
    geom = geometry(raster(), raster(), [])
    assert transfer(b"", hashlib.sha256(b"").hexdigest(), geom) == (b"", [])
    with pytest.raises(ValueError, match="checksum"):
        transfer(b"", "wrong", geom)
    with pytest.raises(ValueError, match="geometry"):
        transfer(b"", hashlib.sha256(b"").hexdigest(), {"valid": False})
    assert not geometry(raster(), raster().crop((1, 0, 100, 80)), [])["valid"]
    assert not geometry(raster(), raster(), [[0, 0, 1, 0.5]])["valid"]
    im = raster()
    im.getexif()[274] = 6
    assert not geometry(im, raster(), [])["valid"]


def test_provenance_is_required_even_for_identical_pixels():
    src, dst = {"content_id": "a"}, {"sha256": "b"}
    assert not provenance_valid({}, src, dst)
    evidence = dict(
        reviewer="synthetic",
        evidence_reference="fixture",
        modified_regions_complete=True,
        source=dict(image_sha256="a", video_sha256="c" * 64, native_frame_index=1),
        target=dict(image_sha256="b", video_sha256="c" * 64, native_frame_index=1),
    )
    assert provenance_valid(evidence, src, dst)
    evidence["target"]["native_frame_index"] = 2
    assert not provenance_valid(evidence, src, dst)


def test_future_identity_gate_blocks_cross_split_protected_and_missing():
    edges = [dict(source_frame_id="s", target_id="t", protected=True)]
    enforce_identity_splits(edges, {"s": "test", "t": "test"})
    for assignments in (
        {"s": "test", "t": "train"},
        {"s": "train", "t": "train"},
        {"s": "test"},
    ):
        with pytest.raises(ValueError):
            enforce_identity_splits(edges, assignments)


def test_inventory_retains_corrupt_and_existing_annotation(tmp_path):
    path = tmp_path / "source.zip"
    b = io.BytesIO()
    raster().save(b, format="PNG")
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("frame_001.png", b.getvalue())
        z.writestr("frame_001.txt", "")
        z.writestr("frame_002.png", b"bad")
    before = path.read_bytes()
    rows, features, labels = inventory([path])
    assert len(rows) == 2 and len(labels) == 1 and features.shape == (2, 768)
    assert rows[0]["existing_label_candidates"] == 1
    assert rows[1]["decode_error"]
    assert path.read_bytes() == before


def test_zip_traversal_refused(tmp_path):
    path = tmp_path / "source.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("../frame.png", b"bad")
    with pytest.raises(ValueError, match="Unsafe"):
        inventory([path])


def test_publication_replay_no_automatic_promotion_and_bound_transfer(
    tmp_path, monkeypatch
):
    """Use real audit/ZIP/parser/receipt paths; stub only unrelated split solver."""
    import json
    import shutil

    import pandas as pd

    from flir_pipeline.data import no_hud
    from flir_pipeline.data.adendo import audit_adendo

    root = tmp_path / "originals"
    root.mkdir()
    b = io.BytesIO()
    raster().save(b, format="PNG")
    payload = b.getvalue()
    raw = b"0 .5 .5 .2 .2\n"
    for name, entries in {
        "images.zip": {"train/frame_001.png": payload},
        "labels.zip": {"train/frame_001.txt": raw},
        "supplement.zip": {"frame_002.png": payload, "frame_002.txt": raw},
        "nohud.zip": {"frame_001.png": payload},
    }.items():
        with zipfile.ZipFile(root / name, "w") as z:
            for member, value in entries.items():
                z.writestr(member, value)
    audit = tmp_path / "audit"
    audit_adendo(
        root / "images.zip", root / "labels.zip", root / "supplement.zip", audit
    )
    proposal = tmp_path / "proposal"
    (proposal / "sources").mkdir(parents=True)
    shutil.copytree(audit, proposal / "sources" / "audit")
    manifest = json.loads((audit / "candidate_manifest.json").read_text())["records"]
    pd.DataFrame(manifest[:1]).to_parquet(proposal / "protected_records.parquet")
    monkeypatch.setattr(no_hud, "verify_common_test", lambda _: {"valid": True})
    out = tmp_path / "pending"
    summary = no_hud.prepare(root, audit, proposal, ["nohud.zip"], out)
    assert summary["verified_images"] == summary["candidate_labels"] == 0
    assert no_hud.verify(out, replay=True)["replayed"]
    target = json.loads((out / "inventory.json").read_text())["no_hud"][0]
    src = manifest[0]
    evidence = tmp_path / "evidence.json"
    evidence.write_text(
        json.dumps(
            {
                "pairs": [
                    dict(
                        target_id=target["target_id"],
                        source_frame_id=src["frame_id"],
                        reviewer="synthetic",
                        evidence_reference="fixture",
                        modified_regions_complete=True,
                        modified_regions=[],
                        source=dict(
                            image_sha256=src["content_id"],
                            video_sha256="a" * 64,
                            native_frame_index=7,
                        ),
                        target=dict(
                            image_sha256=target["sha256"],
                            video_sha256="a" * 64,
                            native_frame_index=7,
                        ),
                    )
                ]
            }
        )
    )
    verified = tmp_path / "verified"
    summary = no_hud.prepare(
        root, audit, proposal, ["nohud.zip"], verified, evidence=evidence
    )
    assert summary["candidate_labels"] == summary["verified_protected_images"] == 1
    assert summary["ready_for_training"] is False
    assert no_hud.verify(verified, replay=True)["replayed"]
    edges = json.loads((verified / "frame_identity_relations.json").read_text())[
        "relations"
    ]
    assert sum(e["relation"] == "same_native_frame" for e in edges) == 1
    assert sum(e["relation"] == "exact_content_dependency" for e in edges) == 1
    label = next((verified / "candidate_labels").glob("*.txt"))
    assert label.read_bytes() == raw
    label.write_bytes(b"")
    with pytest.raises(ValueError, match="changed transfer"):
        no_hud.verify(verified)

