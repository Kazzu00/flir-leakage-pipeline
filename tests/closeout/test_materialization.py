import io
import zipfile

import pandas as pd
import pytest
from PIL import Image

from flir_pipeline.closeout import common_test
from flir_pipeline.closeout.io import digest, read_json, verify


def protected_fixture(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    rows, label_rows = [], {}
    image = io.BytesIO()
    Image.new("RGB", (4, 4), "gray").save(image, "PNG")
    raw = image.getvalue()
    with (
        zipfile.ZipFile(source / "images.zip", "w") as archive,
        zipfile.ZipFile(source / "labels.zip", "w") as label_archive,
    ):
        for index in range(372):
            frame = f"frame_{index}"
            member = frame + ".png"
            label = f"{index % 5} 0.5 0.5 0.1 0.1\n".encode()
            archive.writestr(member, raw)
            label_archive.writestr(frame + ".txt", label)
            rows.append(
                {
                    "frame_id": frame,
                    "content_id": digest(raw),
                    "source_archive": "images.zip",
                    "zip_member_name": member,
                    "zip_header_offset": archive.getinfo(member).header_offset,
                    "image_sha256": digest(raw),
                    "label_sha256": digest(label),
                    "num_objects": 1,
                    "label_ids": [frame],
                }
            )
            label_rows[frame] = {
                "source_archive": "labels.zip",
                "zip_member_name": frame + ".txt",
                "zip_header_offset": label_archive.getinfo(
                    frame + ".txt"
                ).header_offset,
            }
    return source, pd.DataFrame(rows), label_rows


def test_materialize_exact_occurrences_and_immutable_sources(tmp_path, monkeypatch):
    source, rows, label_rows = protected_fixture(tmp_path)
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    monkeypatch.setattr(
        common_test,
        "publication_context",
        lambda *args: (rows, label_rows, {"test_records_sha256": "synthetic"}),
    )
    output = tmp_path / "test"
    result = common_test.materialize(tmp_path / "publication", source, output)
    assert result["records"] == 372 and result["evaluated"] is False
    assert sum(result["class_counts"].values()) == 372
    assert len(list((output / "images/test").glob("*.png"))) == 372
    assert before == {p.name: p.read_bytes() for p in source.iterdir()}
    verify(output)
    assert len(common_test.verify_test(output)) == 372
    assert read_json(output / "receipt.json")["state"] == "MATERIALIZED_NOT_RELEASED"
    with pytest.raises(FileExistsError):
        common_test.materialize(tmp_path / "publication", source, output)


def test_materialize_bad_locator_never_publishes(tmp_path, monkeypatch):
    source, rows, label_rows = protected_fixture(tmp_path)
    rows.loc[10, "zip_header_offset"] += 1
    monkeypatch.setattr(
        common_test, "publication_context", lambda *args: (rows, label_rows, {})
    )
    with pytest.raises(ValueError, match="locator"):
        common_test.materialize(tmp_path / "publication", source, tmp_path / "test")
    assert not (tmp_path / "test").exists()
    assert list(tmp_path.glob(".test.partial-*/FAILURE.json"))
