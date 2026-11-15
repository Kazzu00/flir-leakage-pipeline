"""Synthetic scientific invariants for content-only localization."""

import numpy as np
import pytest

from flir_pipeline.alignment.visual_localization import (
    diverse_candidates,
    pixel_key,
    raster_digest,
    temporal_state,
)


def test_sparse_collision_never_implies_raster_identity():
    a = np.zeros((32, 32, 3), dtype=np.uint8)
    b = a.copy()
    b[1, 1] = 100
    assert pixel_key(a) == pixel_key(b)
    assert raster_digest(a) != raster_digest(b)


def test_uniqueness_requires_complete_search_and_preserves_recurrences():
    assert temporal_state([], search_complete=True) == "NOT_VERIFIED"
    assert temporal_state([{}], search_complete=False) == "NOT_VERIFIED"
    assert temporal_state([{}], search_complete=True) == "VIDEO_VERIFIED"
    assert temporal_state([{}, {}], search_complete=True) == "VIDEO_AMBIGUOUS"


def test_retrieval_preserves_other_videos_and_distinct_time_modes():
    rows = [
        {"video_sha256": "a", "second": 10.0},
        {"video_sha256": "a", "second": 10.1},
        {"video_sha256": "b", "second": 10.0},
        {"video_sha256": "a", "second": 100.0},
    ]
    assert diverse_candidates(np.array([1, 0.99, 0.9, 0.8]), rows) == [0, 2, 3]


def test_decoded_video_occurrences_preserve_real_pts_and_repeated_rasters(tmp_path):
    av = pytest.importorskip("av")
    rng = np.random.default_rng(7)
    first = rng.integers(0, 256, (32, 64, 3), dtype=np.uint8)
    second = rng.integers(0, 256, (32, 64, 3), dtype=np.uint8)
    video = tmp_path / "nominal_frame_999999.mkv"
    with av.open(str(video), "w") as container:
        stream = container.add_stream("ffv1", rate=5)
        stream.width, stream.height, stream.pix_fmt = 64, 32, "bgr0"
        for raster in (first, second, first):
            for packet in stream.encode(
                av.VideoFrame.from_ndarray(raster, format="rgb24")
            ):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    hits = {raster_digest(first): [], raster_digest(second): []}
    with av.open(str(video)) as container:
        for ordinal, frame in enumerate(container.decode(video=0)):
            digest = raster_digest(frame.to_ndarray(format="rgb24"))
            hits[digest].append((ordinal, frame.pts, frame.time_base))
    repeated = hits[raster_digest(first)]
    assert [row[0] for row in repeated] == [0, 2]
    assert all(pts is not None for _, pts, _ in repeated)
    assert repeated[1][1] * repeated[1][2] > repeated[0][1] * repeated[0][2]
    assert temporal_state(repeated, search_complete=True) == "VIDEO_AMBIGUOUS"
    assert (
        temporal_state(hits[raster_digest(second)], search_complete=True)
        == "VIDEO_VERIFIED"
    )
