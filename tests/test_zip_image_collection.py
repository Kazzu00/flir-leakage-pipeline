"""Small offline ZIP fixtures exercise entry identity and bounded decoding."""

import io
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import pytest
from PIL import Image

from flir_pipeline.data.video_variant_contract import Limits
from flir_pipeline.data.zip_image_collection import (
    EntryLimitError,
    check_archive,
    image_properties,
    read_entry,
)


def png(color="red", size=(6, 4)):
    stream = io.BytesIO()
    Image.new("RGB", size, color).save(stream, format="PNG")
    return stream.getvalue()


def test_duplicate_member_names_read_the_exact_zipinfo(tmp_path):
    path = tmp_path / "same-names.zip"
    with ZipFile(path, "w") as archive:
        archive.writestr("frame_1.png", png("red"))
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr("frame_1.png", png("blue"))
    with ZipFile(path) as archive:
        first, second = [
            read_entry(archive, info, Limits()) for info in archive.infolist()
        ]
    assert first[0] != second[0] and first[1] != second[1]


@pytest.mark.parametrize(
    "name",
    [
        "../frame_1.png",
        "/frame_1.png",
        "C:/frame_1.png",
        "folder/../frame_1.png",
        "folder\\frame_1.png",
        "frame_1.png:stream",
    ],
)
def test_unsafe_member_paths_are_never_normalized_away(tmp_path, name):
    path = tmp_path / "unsafe.zip"
    with ZipFile(path, "w") as archive:
        info = ZipInfo("placeholder.png")
        info.filename = info.orig_filename = name
        archive.writestr(info, png())
    with ZipFile(path) as archive, pytest.raises(ValueError, match="Unsafe"):
        read_entry(archive, archive.infolist()[0], Limits())


def test_size_compression_and_archive_limits(tmp_path):
    path = tmp_path / "limits.zip"
    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        archive.writestr("frame_1.png", b"0" * 10000)
        archive.writestr("frame_2.png", png())
    with ZipFile(path) as archive:
        infos = archive.infolist()
        with pytest.raises(EntryLimitError, match="max_member_bytes"):
            read_entry(archive, infos[0], Limits(max_member_bytes=10))
        with pytest.raises(EntryLimitError, match="max_compression_ratio"):
            read_entry(archive, infos[0], Limits(max_compression_ratio=2.0))
        with pytest.raises(EntryLimitError, match="max_archive_entries"):
            check_archive(infos, Limits(max_archive_entries=1))
        with pytest.raises(EntryLimitError, match="max_archive_uncompressed_bytes"):
            check_archive(infos, Limits(max_archive_uncompressed_bytes=20))


def test_actual_decompressed_size_is_bounded_even_if_declaration_lies():
    from zipfile import ZipInfo

    class FakeArchive:
        def open(self, *_):
            return io.BytesIO(b"x" * 100)

    info = ZipInfo("frame_1.png")
    info.file_size, info.compress_size = 1, 1
    with pytest.raises(EntryLimitError, match="Decompressed"):
        read_entry(FakeArchive(), info, Limits(max_member_bytes=10))


@pytest.mark.parametrize(
    "limits", [Limits(max_width=3), Limits(max_height=2), Limits(max_pixels=10)]
)
def test_dimensions_checked_before_pixel_decode(limits):
    properties = image_properties(png(), limits, "PNG")
    assert properties["width"] == 6 and properties["height"] == 4
    assert not properties["image_decode_valid"]
    assert "dimensions" in properties["image_error"]


def test_decode_errors_and_wrong_formats_are_explicit():
    invalid = image_properties(b"corrupt image", Limits(), "PNG")
    assert not invalid["image_decode_valid"] and invalid["image_error"]
    stream = io.BytesIO()
    Image.new("RGB", (3, 2)).save(stream, format="JPEG")
    wrong = image_properties(stream.getvalue(), Limits(), "PNG")
    assert wrong["image_format"] == "JPEG" and not wrong["image_decode_valid"]
