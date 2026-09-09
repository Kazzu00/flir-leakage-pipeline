from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from PIL import Image

from flir_pipeline.data.identity import dataset_id_from_manifest
from flir_pipeline.data.manifest import build_manifest
from flir_pipeline.data.yolo_labels import validate_label_text


def _image_bytes(color: str) -> bytes:
    stream = BytesIO()
    Image.new("RGB", (8, 6), color=color).save(stream, format="PNG")
    return stream.getvalue()


def _write_zip(path: Path, members: dict[str, bytes]) -> None:
    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        for name, content in members.items():
            archive.writestr(name, content)


def test_yolo_validation_cases() -> None:
    valid = validate_label_text("1 0.5 0.5 0.2 0.4\n0 0.2 0.3 0.1 0.1")
    assert valid.label_valid
    assert valid.num_objects == 2
    assert valid.classes_present == [0, 1]

    assert not validate_label_text("-1 0.5 0.5 0.2 0.4").syntax_valid
    assert not validate_label_text("0 1.2 0.5 0.2 0.4").normalized_values_valid
    assert not validate_label_text("0 0.5 0.5 0 0.4").normalized_values_valid
    assert not validate_label_text("0 nan 0.5 0.2 0.4").syntax_valid
    assert not validate_label_text("0 0.5 0.5 0.2").syntax_valid
    empty = validate_label_text("\n")
    assert empty.label_empty and empty.label_valid and empty.num_objects == 0

    first = validate_label_text("0 0.5 0.5 0.2 0.2\n1 0.2 0.2 0.1 0.1")
    second = validate_label_text("1 0.2 0.2 0.1 0.1  \n0 0.5 0.5 0.2 0.2")
    assert first.canonical == second.canonical


def test_build_manifest_preserves_occurrences_and_orphans(tmp_path: Path) -> None:
    images = tmp_path / "Imagenes.zip"
    labels = tmp_path / "Etiquetas.zip"
    image = _image_bytes("red")
    _write_zip(
        images,
        {
            "Imagenes/train/v1_frame_000001.png": image,
            "Imagenes/val/v1_frame_000001.png": image,
        },
    )
    _write_zip(
        labels,
        {
            "Etiquetas/train/v1_frame_000001.txt": b"0 0.5 0.5 0.2 0.2\n",
            "Etiquetas/validation/v1_frame_000001.txt": b"0 0.5 0.5 0.2 0.2\n",
            "Etiquetas/train/orphan.txt": b"1 0.2 0.2 0.1 0.1\n",
        },
    )
    manifest_path = tmp_path / "manifest.parquet"
    report_path = tmp_path / "reports"

    summary = build_manifest(images, labels, manifest_path, report_path, tmp_path)

    assert summary["total_records"] == 2
    assert summary["unique_content_ids"] == 1
    assert summary["cross_split_duplicate_records"] == 2
    assert summary["orphan_labels"] == 1
    import pandas as pd

    dataframe = pd.read_parquet(manifest_path)
    assert len(dataframe) == 2
    assert dataframe["frame_id"].nunique() == 2
    assert dataframe["content_id"].nunique() == 1
    assert set(dataframe["original_split"]) == {"train", "val"}
    assert dataframe["label_valid"].all()
    assert dataframe["cross_split_exact_duplicate"].all()
    assert dataset_id_from_manifest(dataframe) == summary["dataset_id"]
    assert dataset_id_from_manifest(dataframe.iloc[::-1]) == summary["dataset_id"]
    dataframe.loc[0, "label_sha256"] = "changed-label"
    assert dataset_id_from_manifest(dataframe) != summary["dataset_id"]
    assert all("C:\\" not in value for value in dataframe["source_member_path"])
    assert len(pd.read_csv(report_path / "orphan_labels.csv")) == 1
    assert len(pd.read_csv(report_path / "duplicate_annotation_consistency.csv")) == 1
