from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from flir_pipeline.data.inventory import (
    compare_archive_members,
    cross_split_exact_duplicate_analysis,
    inspect_archive,
    match_images_labels,
    run_inventory,
    temporal_neighbors,
)
from flir_pipeline.utils.hashing import sha256_file, sha256_zip_member


def _write_zip(path: Path, members: dict[str, bytes]) -> None:
    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        for name, content in members.items():
            archive.writestr(name, content)


def test_archive_inspection_and_hashing(tmp_path: Path) -> None:
    archive_path = tmp_path / "images.zip"
    _write_zip(
        archive_path,
        {
            "train/frame_000001.jpg": b"image-one",
            "val/frame_000002.jpg": b"image-two",
            "test/frame_000003.jpg": b"image-three",
            "train/frame_000001.txt": b"",
            "odd/../frame_000004.jpg": b"image-four",
        },
    )

    inspection, members = inspect_archive(archive_path, hash_members=True)

    assert inspection.readable
    assert inspection.total_files == 5
    assert inspection.extension_counts[".jpg"] == 4
    assert sha256_file(archive_path)
    image = next(member for member in members if member.extension == ".jpg")
    assert image.sha256 == sha256_zip_member(archive_path, image.member_path)
    assert all("\\" not in member.member_path for member in members)


def test_matching_and_temporal_neighbors(tmp_path: Path) -> None:
    archive_path = tmp_path / "synthetic.zip"
    _write_zip(
        archive_path,
        {
            "train/v1_frame_000001.jpg": b"one",
            "val/v1_frame_000002.jpg": b"two",
            "test/v1_frame_000003.jpg": b"three",
            "train/v1_frame_000001.txt": b"0 0.5 0.5 1 1\n",
            "test/orphan.txt": b"",
        },
    )
    _, members = inspect_archive(archive_path, hash_members=True)

    matching = match_images_labels(members)
    assert sum(row["match_status"] == "matched" for row in matching) == 1
    assert sum(row["match_status"] == "label_without_image" for row in matching) == 1
    neighbors = temporal_neighbors(members)
    assert neighbors[0]["delta_frame_index"] == 1
    assert neighbors[0]["possible_sequence"] == "v1"


def test_relationships_identical_and_subset(tmp_path: Path) -> None:
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    _write_zip(first, {"a.jpg": b"a", "b.jpg": b"b"})
    _write_zip(second, {"nested/a.jpg": b"a", "nested/b.jpg": b"b", "c.jpg": b"c"})
    _, first_members = inspect_archive(first, hash_members=True)
    _, second_members = inspect_archive(second, hash_members=True)

    overlap, _ = compare_archive_members({"first.zip": first_members, "second.zip": second_members})
    image_row = next(row for row in overlap if row["comparable_file_type"] == "image")
    assert image_row["exact_hash_overlap"] == 2
    assert image_row["relationship"] == "SUBSET"


def test_run_inventory_is_local_and_read_only(tmp_path: Path) -> None:
    _write_zip(tmp_path / "one.zip", {"train/a.jpg": b"a"})
    before = (tmp_path / "one.zip").read_bytes()
    output = tmp_path / "reports"

    summary = run_inventory(tmp_path, output, hash_members=True)

    assert summary["read_only"] is True
    assert (tmp_path / "one.zip").read_bytes() == before
    assert (output / "summary.json").is_file()
    assert (output / "findings.md").is_file()


def test_corrupt_zip_is_reported(tmp_path: Path) -> None:
    corrupt = tmp_path / "broken.zip"
    corrupt.write_bytes(b"not a zip")

    inspection, members = inspect_archive(corrupt)

    assert not inspection.readable
    assert inspection.error
    assert members == []


def test_cross_split_exact_duplicate_analysis(tmp_path: Path) -> None:
    archive_path = tmp_path / "Imagenes.zip"
    _write_zip(
        archive_path,
        {
            "Imagenes/train/v1_frame_000001.jpg": b"same",
            "Imagenes/val/v1_frame_000001.jpg": b"same",
            "Imagenes/test/v1_frame_000002.jpg": b"different",
        },
    )
    _, members = inspect_archive(archive_path, hash_members=True)
    neighbors = temporal_neighbors(members)

    pairs, duplicates, summary = cross_split_exact_duplicate_analysis(
        members, neighbors
    )

    zero_delta = next(row for row in pairs if row["delta_frame_index"] == 0)
    assert zero_delta["exact_content_duplicate"] is True
    assert zero_delta["size_bytes_a"] == 4
    assert zero_delta["size_bytes_b"] == 4
    assert len(duplicates) == 2
    assert summary["same_split_duplicate_hashes"] == 0
    assert summary["train_val_hashes"] == 1
    assert summary["train_test_hashes"] == 0
    assert summary["val_test_hashes"] == 0


def test_cross_split_analysis_rejects_sequence_mismatch(tmp_path: Path) -> None:
    archive_path = tmp_path / "Imagenes.zip"
    _write_zip(
        archive_path,
        {
            "Imagenes/train/v1_frame_000001.jpg": b"same",
            "Imagenes/val/v2_frame_000001.jpg": b"same",
        },
    )
    _, members = inspect_archive(archive_path, hash_members=True)
    neighbors = [
        {
            "frame_a": "Imagenes/train/v1_frame_000001.jpg",
            "split_a": "train",
            "frame_b": "Imagenes/val/v2_frame_000001.jpg",
            "split_b": "val",
            "delta_frame_index": 0,
        }
    ]

    pairs, _, _ = cross_split_exact_duplicate_analysis(members, neighbors)

    assert pairs[0]["sequence_match"] is False
    assert pairs[0]["exact_content_duplicate"] is False
