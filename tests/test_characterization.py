import hashlib
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import pytest

from flir_pipeline.data.annotations import audit_annotations
from flir_pipeline.data.inventory import filename_patterns, inspect_archive
from flir_pipeline.data.temporal import audit_temporal_lineage
from flir_pipeline.features.visualization import generate_feature_engineering_report


def annotation_fixture(tmp_path: Path) -> tuple[pd.DataFrame, Path]:
    labels = {
        "train/a.txt": b"0 0.5 0.5 0.2 0.2\n0 0.2 0.2 0.1 0.1\n1 0.5 0.5 0.2 0.2\n",
        "val/b.txt": b"0 0.5 0.5 0.2 0.2\n1 0.5 0.5 0.2 0.2\n",
        "test/c.txt": b"",
        "test/orphan.txt": b"4 0.5 0.5 0.2 0.2\n4 0.3 0.3 0.1 0.1\n",
    }
    archive = tmp_path / "labels.zip"
    with ZipFile(archive, "w") as output:
        for name, content in labels.items():
            output.writestr(name, content)
    rows = []
    for i, (name, content) in enumerate(list(labels.items())[:3]):
        rows.append({
            "frame_id": f"frame-{i}", "content_id": "same" if i < 2 else "other",
            "image_sha256": "same" if i < 2 else "other", "label_sha256": hashlib.sha256(content).hexdigest(),
            "relative_label_path": name, "source_archive": "images.zip", "source_member_path": name.replace(".txt", ".png"),
            "original_split": name.split("/")[0], "label_empty": i == 2, "label_valid": True,
            "num_objects": [3, 2, 0][i], "classes_present": "0|1" if i < 2 else "",
            "possible_sequence": "synthetic", "possible_frame_index": i,
            "temporal_inference_confidence": "medium",
        })
    return pd.DataFrame(rows), archive


def test_annotation_instances_orphans_conflicts_and_byte_integrity(tmp_path: Path) -> None:
    manifest, archive = annotation_fixture(tmp_path)
    original = archive.read_bytes()
    audit = audit_annotations(manifest, archive)
    counts = audit.classes.set_index("class_id")
    assert counts.loc[0].to_dict() == {"images_containing_class": 2, "object_instances": 3}
    assert audit.summary["candidate_objects"] == 5
    assert audit.summary["orphan_objects"] == 2 and audit.summary["archive_objects"] == 7
    assert audit.summary["orphan_labels"] == 1 and audit.summary["candidate_empty_labels"] == 1
    assert audit.summary["conflicting_duplicate_groups"] == 1
    assert audit.summary["consistent_duplicate_groups"] == 0
    assert archive.read_bytes() == original
    manifest.loc[0, "label_sha256"] = "changed"
    with pytest.raises(ValueError, match="bytes differ"):
        audit_annotations(manifest, archive)


def test_temporal_heuristics_are_not_timestamps_or_exact_identity(tmp_path: Path) -> None:
    archive = tmp_path / "synthetic.zip"
    with ZipFile(archive, "w") as output:
        for name in ("v1_frame_001.png", "frame_002.png", "unknown.png"):
            output.writestr(name, b"synthetic")
    _, members = inspect_archive(archive)
    guesses = {row["basename"]: row for row in filename_patterns(members)}
    assert guesses["v1_frame_001"]["inference_confidence"] == "medium"
    assert guesses["frame_002"]["possible_sequence"] == ""
    assert guesses["frame_002"]["inference_confidence"] == "low"
    manifest = pd.DataFrame([
        {"frame_id": "a", "content_id": "x", "original_split": "train", "possible_sequence": "v1", "possible_frame_index": 1},
        {"frame_id": "b", "content_id": "x", "original_split": "val", "possible_sequence": "v1", "possible_frame_index": 1},
        {"frame_id": "c", "content_id": "y", "original_split": "test", "possible_sequence": "v1", "possible_frame_index": 2},
        {"frame_id": "d", "content_id": "z", "original_split": "train", "possible_sequence": "", "possible_frame_index": 3},
    ])
    audit = audit_temporal_lineage(manifest)
    assert audit.summary["records_with_inferred_order"] == 3
    assert audit.summary["records_without_inferred_order"] == 1
    assert audit.summary["records_with_verified_timestamps"] == 0
    assert not audit.lineage["timestamp_available"].any()
    assert audit.summary["cross_split_neighbor_pairs"] == 3
    assert audit.summary["neighbor_pairs_exact_content"] == 1
    assert audit.summary["neighbor_pairs_different_content"] == 2
    assert len(audit_temporal_lineage(manifest, max_frame_gap=0).neighbors) == 1
    separate = manifest.copy()
    separate["source_archive"] = ["one.zip", "two.zip", "three.zip", "one.zip"]
    assert audit_temporal_lineage(separate).neighbors.empty


def test_report_builds_annotation_and_temporal_evidence_without_closing_smoke(tmp_path: Path) -> None:
    manifest, archive = annotation_fixture(tmp_path)
    manifest_path = tmp_path / "manifest.parquet"
    manifest.to_parquet(manifest_path)
    diagnostics_path = tmp_path / "diagnostics.parquet"
    pd.DataFrame({"content_id": ["same", "other"], "width": [8, 8], "height": [6, 6], "aspect_ratio": [4/3, 4/3]}).to_parquet(diagnostics_path)
    output = tmp_path / "report"
    result = generate_feature_engineering_report(manifest_path, diagnostics_path, output, labels_archive=archive)
    assert result["metadata"]["annotation_summary"]["candidate_objects"] == 5
    assert not result["metadata"]["feature_engineering_completed"]
    assert (output / "figures/15_class_instances.png").is_file()
    assert (output / "figures/16_temporal_lineage.png").is_file()
    baseline = pd.read_csv(output / "tables/historical_baseline.csv")
    pd.testing.assert_frame_equal(baseline, manifest[["frame_id", "content_id", "original_split"]])
    empty = pd.read_csv(output / "tables/empty_annotations.csv")
    assert len(empty) == 2 and empty["count"].sum() == 3
    assert empty["percentage"].sum() == pytest.approx(100)
