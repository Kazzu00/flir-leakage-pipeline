"""Exercise the existing linkage/review storage contracts through export adapters."""

from dataclasses import asdict

import pandas as pd
import pytest
import test_linkage_review

from flir_pipeline.explorer.organization import (
    content_records,
    occurrence_records,
    preview_media,
    scientific_tables,
    timelines,
)
from flir_pipeline.explorer.organization_sources import (
    CandidateEvidence,
    Sources,
    load_candidates,
)
from flir_pipeline.linkage.review_sources import (
    load_candidate_context,
    load_linkage_context,
)
from flir_pipeline.similarity.storage import read_json, write_json


@pytest.fixture(scope="module")
def review_template(tmp_path_factory):
    return test_linkage_review.review_template.__wrapped__(tmp_path_factory)


def test_shared_candidate_reader_preserves_review_contract(review_template):
    _, paths, _ = review_template
    shared = load_candidate_context(
        paths.linkage, paths.labeled_manifest, paths.sequence_set
    )
    original = load_linkage_context(paths)
    for a, b in zip(shared[:4], original[:4], strict=True):
        pd.testing.assert_frame_equal(a, b, check_exact=True)
    assert original[4].keys() - shared[4].keys() == {"calibration_sample_sha256"}


def test_real_storage_linkage_and_manual_reviews_remain_external_evidence(
    review_template, tmp_path
):
    root, paths, revision = review_template
    metadata = read_json(revision / "metadata.json")
    source_map = tmp_path / "sources.json"
    write_json(
        source_map,
        {
            "calibrations": {
                metadata["calibration_id"]: {
                    k: str(v) for k, v in asdict(paths).items()
                }
            }
        },
    )
    manifest = pd.read_parquet(paths.labeled_manifest)
    sources = Sources()
    result = load_candidates(
        [paths.linkage, revision],
        paths.labeled_manifest,
        manifest,
        sources,
        sequence_root=paths.sequence_set,
        review_source_map=source_map,
    )
    candidates = pd.read_parquet(paths.linkage / "content_candidates.parquet")
    assert {r["linkage_group_id"] for r in result.groups} == set(
        candidates.candidate_id
    )
    assert all(r["kind"] == "candidate_pair" for r in result.groups)
    assert len(result.reviews) == 3
    assert all(r["decision"]["manual_decision"] == "" for r in result.reviews)
    tables, raw = scientific_tables(manifest, [], {}, result)
    assert len(tables["records"]) == len(manifest) + 360
    assert len(tables["contents"]) < len(tables["records"])
    assert all(g["member_count"] == 2 for g in tables["linkage_groups"])
    assert len(tables["timelines"]) == 1
    assert tables["timelines"][0]["source_video_id"] is not None
    assert len(tables["timelines"][0]["points"]) == 360
    assert all(
        p["timestamp_seconds"] is not None for p in tables["timelines"][0]["points"]
    )
    media_root = tmp_path / "media"
    media_root.mkdir()
    previews = preview_media(
        tables["contents"],
        raw,
        manifest,
        media_root,
        include=True,
        data_root=None,
        video_images_root=root / "video-images",
    )
    assert any(p["media_available"] for p in previews)
    assert any(not p["media_available"] for p in previews)
    sources.unchanged()


def test_linkage_needs_resolvable_sequence_sources(review_template):
    _, paths, _ = review_template
    with pytest.raises(ValueError, match="sequence-root"):
        load_candidates(
            [paths.linkage],
            paths.labeled_manifest,
            pd.read_parquet(paths.labeled_manifest),
            Sources(),
        )


def test_same_content_multiple_source_videos_never_collapses_time():
    manifest = pd.DataFrame(
        [
            dict(
                frame_id="labeled",
                content_id="same",
                image_sha256="same",
                label_sha256="annotation",
                source_archive="synthetic.zip",
                original_split="train",
            )
        ]
    )
    videos = pd.DataFrame(
        [
            dict(
                frame_id=f"video-{i}",
                content_id="same",
                image_sha256="same",
                video_id=f"video-{i}",
                sample_index=i * 4,
                timestamp_seconds=float(i * 4),
                source_frame_index_estimate=i * 120,
                image_path=f"video-{i}/frame.jpg",
            )
            for i in (1, 2)
        ]
    )
    records, raw = occurrence_records(
        manifest, CandidateEvidence(video_records=[videos])
    )
    contents = content_records(records, raw, {})
    assert len(contents) == 1
    assert contents[0]["source_video_id"] is None
    assert contents[0]["frame_index"] is None
    assert contents[0]["timestamp_seconds"] is None
    assert len(contents[0]["record_ids"]) == 3
    lines = timelines(records, contents)
    assert {t["source_video_id"] for t in lines} == {"video-1", "video-2"}
    assert [t["points"][0]["frame_index"] for t in lines] == [4, 8]
    assert all(t["temporal_source"] == "sampled_video_grid" for t in lines)


def test_explicit_unknown_artifact_is_rejected(tmp_path):
    write_json(
        tmp_path / "metadata.json",
        {"artifact_id": "unknown", "artifact_kind": "unrecognized"},
    )
    manifest_path = tmp_path / "manifest.parquet"
    pd.DataFrame().to_parquet(manifest_path)
    with pytest.raises(ValueError, match="Unsupported evidence kind"):
        load_candidates([tmp_path], manifest_path, pd.DataFrame(), Sources())


def test_manual_review_requires_source_map(review_template):
    _, paths, revision = review_template
    with pytest.raises(ValueError, match="review-source-map"):
        load_candidates(
            [revision],
            paths.labeled_manifest,
            pd.read_parquet(paths.labeled_manifest),
            Sources(),
        )
