"""Synthetic strict video-admission contract; no pixels or scientific execution."""

import copy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "verified_timeline", Path(__file__).parents[1] / "export_verified_video_timeline.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture
def inputs():
    data = {
        "records": [
            {
                "frame_id": "f1",
                "content_id": "c1",
                "filename": "one.jpg",
                "axis_id": "inferred",
                "position": 999,
                "reason": None,
            }
        ],
        "runs": [
            {
                "run_id": "historical-S2-3",
                "population": "historical",
                "membership_id": "m",
                "candidate_id": "c",
                "members": [{"frame_id": "f1", "new_split": "train"}],
            }
        ],
    }
    rows = [
        {
            "frame_id": "f1",
            "content_id": "c1",
            "filename": "one.jpg",
            "source_video_id": "",
            "appearance_order": "",
            "match_status": "NOT_VERIFIED_IN_EXISTING_SCOPE",
            "evidence_reference": "synthetic/proof.json",
        }
    ]
    videos = [
        {
            "source_video_id": f"v{i}",
            "sha256": str(i) * 64,
            "unchanged": True,
            "decoded_frames_in_existing_complete_scan": 100,
        }
        for i in range(3)
    ]
    delivery = {"package_id": "p"}
    audit = {
        "package_id": "p",
        "source_membership_id": "m",
        "source_candidate_id": "c",
        "clustering_memberships_nonnull_in_source_groups": 0,
    }
    return data, rows, videos, delivery, audit


def accepted(inputs):
    inputs = copy.deepcopy(inputs)
    inputs[1][0].update(
        source_video_id="v0",
        source_video_sha256="0" * 64,
        appearance_order="0",
        match_status="VIDEO_VERIFIED",
        match_method="complete_rgb_raster_equality_via_existing_completed_scan",
        existing_exact_occurrence_count="1",
        canonical_rgb_sha256="a" * 64,
    )
    return inputs


def test_unknowns_are_not_projected_and_all_three_videos_remain(inputs):
    original = copy.deepcopy(inputs[0])
    data, admission = module.project(*inputs)
    assert inputs[0] == original
    assert len(data["axes"]) == 3 and admission["accepted_records"] == []
    assert data["records"][0]["position"] is None
    assert data["records"][0]["axis_id"] is None
    assert data["intervals"] == data["cluster_memberships"] == []
    assert data["runs"] == original["runs"]
    assert admission["absence_from_videos_claimed"] is False


def test_exact_unique_position_zero_is_admitted_without_new_split(inputs):
    arguments = accepted(inputs)
    data, admission = module.project(*arguments)
    assert data["records"][0]["position"] == 0
    assert data["records"][0]["reason"] is None
    assert admission["accepted_records"][0]["appearance_order"] == 0
    assert data["runs"] == inputs[0]["runs"]


def test_legacy_basename_display_is_expanded_only_after_exact_identity_join(inputs):
    inputs[0]["records"][0]["filename"] = "one"
    data, _ = module.project(*inputs)
    assert data["records"][0]["filename"] == "one.jpg"


@pytest.mark.parametrize(
    "kind",
    [
        "identity",
        "filename",
        "hash",
        "perceptual",
        "repeated",
        "negative",
        "wrong-video",
        "membership",
        "missing",
        "candidate-position",
        "clustering",
    ],
)
def test_rejects_unjustified_or_mismatched_evidence(inputs, kind):
    args = accepted(inputs)
    row = args[1][0]
    if kind == "identity":
        row["content_id"] = "different"
    if kind == "filename":
        row["filename"] = "different.jpg"
    if kind == "hash":
        row["source_video_sha256"] = "bad"
    if kind == "perceptual":
        row["match_method"] = "CLIP_cosine"
    if kind == "repeated":
        row["existing_exact_occurrence_count"] = "8"
    if kind == "negative":
        row["appearance_order"] = "-1"
    if kind == "wrong-video":
        row["source_video_id"] = "unauthorized"
    if kind == "membership":
        args[4]["source_membership_id"] = "different"
    if kind == "missing":
        args[1].clear()
    if kind == "candidate-position":
        row["match_status"] = "CANDIDATE"
    if kind == "clustering":
        args[4]["clustering_memberships_nonnull_in_source_groups"] = 1
    with pytest.raises(ValueError):
        module.project(*args)
