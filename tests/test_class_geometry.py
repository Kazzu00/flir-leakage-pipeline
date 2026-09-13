import hashlib
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest
import yaml

from flir_pipeline.data.annotations import audit_annotations
from flir_pipeline.data.classes import (
    DETECTION_CLASSES,
    class_catalog_rows,
    class_display_name,
    class_name,
    verify_class_config,
)
from flir_pipeline.data.yolo_labels import validate_label_bytes
from flir_pipeline.features.visualization import _bbox_geometry_figures


@pytest.mark.parametrize("class_id,name", list(enumerate([
    "Vehicles", "Buildings", "Roads", "Rivers", "Heavy Machinery",
])))
def test_class_mapping_preserves_ids_and_original_name(class_id, name):
    assert class_name(class_id) == name
    assert class_display_name(class_id) == f"{name} ({class_id})"
    assert DETECTION_CLASSES[4].source_name == "SDZI"
    assert len(class_catalog_rows()) == 5
    with pytest.raises(ValueError, match="Unrecognized"):
        class_name(5)  # No background detection class.


@pytest.mark.parametrize("invalid", [False, True])
def test_source_yaml_verification_rejects_offsets(tmp_path, invalid):
    names = {i + int(invalid): item.source_name for i, item in DETECTION_CLASSES.items()}
    archive = tmp_path / "source.zip"
    with ZipFile(archive, "w") as output:
        output.writestr("dataset_split_completo/dataset.yaml", yaml.safe_dump({
            "nc": 5, "names": names, "path": "private-path-must-not-be-reported",
        }))
    original = archive.read_bytes()
    if invalid:
        with pytest.raises(ValueError, match="differs"):
            verify_class_config(archive)
    else:
        evidence = verify_class_config(archive)
        assert evidence["source_config_validated"]
        assert evidence["academic_mapping_validated"]
        assert "private-path" not in str(evidence)
    assert archive.read_bytes() == original


def _audit(tmp_path, labels):
    archive = tmp_path / "labels.zip"
    rows = []
    with ZipFile(archive, "w") as output:
        for i, text in enumerate(labels):
            content = text.encode()
            output.writestr(f"{i}.txt", content)
            parsed = validate_label_bytes(content)
            rows.append({"frame_id": f"f{i}", "content_id": f"c{i}",
                         "relative_label_path": f"{i}.txt", "label_sha256": hashlib.sha256(content).hexdigest(),
                         "num_objects": parsed.num_objects})
    return audit_annotations(pd.DataFrame(rows), archive)


def test_geometry_uses_individual_normalized_boxes_and_source_precision(tmp_path):
    audit = _audit(tmp_path, [
        "2 0.5 0.5 0.4 0.2\n2 0.5 0.5 0.2 0.4\n",
        "2 0.5 0.5 0.6 0.2\n1 0.5 0.5 0.1234567890123 0.3\n",
        "",
    ])
    roads = audit.instances.query("class_id == 2")
    np.testing.assert_allclose(roads.normalized_area, [0.08, 0.08, 0.12])
    np.testing.assert_allclose(roads.aspect_ratio, [2, 0.5, 3])
    stats = audit.geometry.set_index(["class_id", "metric"]).loc[(2, "aspect_ratio")]
    assert stats["count"] == 3
    assert stats["mean"] == pytest.approx(5.5 / 3)
    assert stats["std"] == pytest.approx(np.std([2, 0.5, 3], ddof=1))
    assert (stats["median"], stats["Q1"], stats["Q3"], stats["min"], stats["max"]) == pytest.approx((2, 1.25, 2.5, 0.5, 3))
    singleton = audit.instances.query("class_id == 1").iloc[0]
    assert singleton.normalized_width == float("0.1234567890123")
    assert audit.geometry.query("class_id == 1")["std"].isna().all()
    assert audit.classes.set_index("class_id").loc[2, "images_containing_class"] == 2
    assert audit.summary["candidate_empty_labels"] == 1
    assert len(audit.instances) == 4


def test_background_only_adds_no_class_boxes_or_statistics(tmp_path):
    audit = _audit(tmp_path, ["", "\n  \n"])
    assert audit.classes.empty and audit.instances.empty and audit.geometry.empty
    assert audit.summary["candidate_empty_labels"] == 2
    assert audit.summary["candidate_objects"] == 0
    _bbox_geometry_figures(audit.instances, tmp_path)
    assert (tmp_path / "figures/06_bbox_normalized_area_by_class.png").is_file()


def test_invalid_geometry_is_audited_without_division_by_zero_or_repair(tmp_path):
    audit = _audit(tmp_path, ["0 0.5 0.5 0.2 0\n", "1 0.9 0.5 0.4 0.2\n"])
    assert audit.summary["candidate_invalid_labels"] == 2
    assert audit.summary["candidate_objects"] == 2
    assert audit.summary["geometry_excluded_invalid_label_objects"] == 2
    assert audit.instances.empty and audit.geometry.empty
