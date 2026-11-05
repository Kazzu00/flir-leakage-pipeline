import pytest

from flir_pipeline.data.name_lineage import parse_name_fields


@pytest.mark.parametrize(
    "name,index,scene",
    [
        ("v1_frame_000258_scene074.jpg", 258, 74),
        ("v12_frame_000022_scene009.png", 22, 9),
        ("frame_000258_scene074.jpg", 258, 74),
    ],
)
def test_prefix_is_not_the_index(name, index, scene):
    parsed = parse_name_fields(name)
    assert parsed["nominal_index"] == index
    assert parsed["scene_token"] == scene
    assert not parsed["temporal_identity_verified"]


def test_video_name_does_not_assign_a_video_or_timestamp():
    parsed = parse_name_fields("nested/video_11min_258.jpg")
    assert parsed["family"] == "video_11min"
    assert parsed["nominal_index"] == 258
    assert "video_id" not in parsed and "timestamp" not in parsed
    assert not parsed["temporal_identity_verified"]


def test_unknown_name_stays_unknown():
    assert parse_name_fields("prefix123.jpg")["nominal_index"] is None
