"""Synthetic/offline scientific invariants; no real video/review identifiers."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from flir_pipeline.cli import app
from flir_pipeline.data.identity import dataset_id_from_manifest
from flir_pipeline.features.storage import feature_space_id
from flir_pipeline.sequences.base import SequenceConfig
from flir_pipeline.sequences.construction import exact_dependencies
from flir_pipeline.sequences.detection import (
    centroid_changes,
    detect_tables,
    localize_events,
    multiscale_scores,
    stable_percentiles,
)
from flir_pipeline.sequences.storage import (
    build_to_store,
    detect_to_store,
    load_sources,
    summarize_directory,
    verify_directory,
)
from flir_pipeline.sequences.validation import (
    REVIEW_COLUMNS,
    load_confirmed_validation,
)
from flir_pipeline.similarity.storage import file_sha256, read_json, write_json


def synthetic_sources(root, sizes=(360, 80, 1)):
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for v, n in enumerate(sizes):
        for i in range(n):
            # Exact duplicate block and an isolated duplicate; all occurrences stay.
            content_index = i if i < 180 else i - 180
            if i == 359:
                content_index = 3
            content = hashlib.sha256(
                f"content-{v}-{content_index}".encode()
            ).hexdigest()
            rows.append(
                {
                    "manifest_version": "flir_video_samples_v1",
                    "frame_id": f"f-{v}-{i}",
                    "content_id": content,
                    "image_sha256": content,
                    "label_sha256": "",
                    "video_id": f"video-{v}",
                    "source_video": f"video-{v}.mp4",
                    "source_video_sha256": hashlib.sha256(
                        f"source-{v}".encode()
                    ).hexdigest(),
                    "sample_index": i,
                    "timestamp_seconds": float(i),
                    "sample_fps": 1.0,
                    "source_frame_index_estimate": i * 30,
                    "image_path": f"video-{v}/{i}.jpg",
                }
            )
    manifest = pd.DataFrame(rows).astype(
        {"sample_index": "Int64", "source_frame_index_estimate": "Int64"}
    )
    manifest_path = root / "manifest.parquet"
    manifest.to_parquet(manifest_path, index=False)
    directories = []
    for encoder in ("clip", "dinov2"):
        directory = root / encoder
        directory.mkdir()
        directories.append(directory)
        index = manifest.drop_duplicates("content_id").sort_values(
            "content_id", ascending=encoder == "clip"
        )
        index = (
            index[["content_id", "frame_id", "sample_index"]]
            .rename(columns={"frame_id": "representative_frame_id"})
            .reset_index(drop=True)
        )
        index["embedding_row"] = np.arange(len(index))
        # Orthogonal plateaus: 60 samples per scene; no scientific model downloads.
        x = np.eye(8, dtype=np.float32)[
            (index.sample_index.to_numpy(dtype=int) // 60) % 8
        ]
        index.drop(columns="sample_index").to_parquet(
            directory / "content_index.parquet", index=False
        )
        record_index = manifest[["frame_id", "content_id"]].copy()
        record_index["embedding_row"] = record_index.content_id.map(
            index.set_index("content_id").embedding_row
        )
        record_index.iloc[::-1].to_parquet(
            directory / "record_index.parquet", index=False
        )
        for name in ("embeddings_raw", "embeddings_l2"):
            np.save(directory / f"{name}.npy", x)
        config = {
            "extractor": encoder,
            "model_id": f"synthetic/{encoder}",
            "model_revision": "b" * 40,
            "pooling_strategy": "projected_pooler_output"
            if encoder == "clip"
            else "cls_token",
            "preprocessing": {"synthetic": True},
            "normalization_policy": "raw_and_l2_float32",
        }
        write_json(
            directory / "metadata.json",
            {
                **config,
                "feature_space_id": feature_space_id(config),
                "resolved_model_revision": "b" * 40,
                "dataset_id": dataset_id_from_manifest(manifest),
                "total_records": len(manifest),
                "selected_content_ids": len(index),
                "unique_content_ids": len(index),
                "embedding_dimension": 8,
            },
        )
    return manifest_path, *directories


def synthetic_review(root, events, reject=(1,)):
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, event in enumerate(events.itertuples()):
        decision = "reject" if i in reject else "accept"
        rows.append(
            {
                "event_id": i + 40,
                "video_id": event.video_id,
                "coarse_sample_index": event.coarse_sample_index,
                "search_start": event.search_start_sample_index,
                "search_end": event.search_end_sample_index,
                "localized_sample_index": event.localized_sample_index,
                "localization_shift": event.localized_sample_index
                - event.coarse_sample_index,
                "f3_score": event.localized_F3,
                "persistent_stable_min": event.coarse_score,
                "persistent_stable_median": event.coarse_median,
                "high_confidence": event.high_confidence,
                "repeat_partner_sample_index": None,
                "decision": decision,
                "boundary_type": "reject"
                if decision == "reject"
                else ("scene_change", "degradation_transition", "transition_interval")[
                    i % 3
                ],
                "notes": "Synthetic confirmed review; not ground truth.",
                "review_status": "confirmed_manual_review",
                "sequence_boundary_committed": False,
                "manual_confirmation": True,
            }
        )
    review = pd.DataFrame(rows, columns=REVIEW_COLUMNS)
    write_review(root, review)
    return root


def write_review(root, review):
    accepted = review.loc[review.decision.eq("accept")]
    review.to_csv(root / "boundary_validation.csv", index=False)
    accepted.to_csv(root / "accepted_boundary_candidates.csv", index=False)
    write_json(
        root / "metadata.json",
        {
            "artifact_kind": "confirmed_manual_boundary_validation",
            "ground_truth": False,
            "diagnostic_source": "synthetic_boundary_diagnostic",
            "review_status": "confirmed_manual_review",
            "manual_confirmation_complete": True,
            "manual_confirmation_required_before_sequence_commit": False,
            "sequence_boundaries_committed": False,
            "confirmation_timestamp_utc": "2025-01-01T00:00:00+00:00",
            "event_count": len(review),
            "accepted_count": len(accepted),
            "rejected_count": len(review) - len(accepted),
            "boundary_type_counts": {
                str(k): int(v) for k, v in review.boundary_type.value_counts().items()
            },
            "source_checksums": {
                "provisional_boundary_validation_csv": "a" * 64,
                "provisional_metadata_json": "b" * 64,
            },
        },
    )


@pytest.fixture
def detected(tmp_path):
    sources = synthetic_sources(tmp_path / "inputs")
    output = detect_to_store(*sources, SequenceConfig(), tmp_path / "output")
    events = pd.read_parquet(output / "candidate_events.parquet")
    assert len(events) >= 3
    validation = synthetic_review(tmp_path / "review", events)
    return sources, output, validation


def test_stable_rounding_ties_and_video_denominator():
    values = np.array([0.1, 0.1 + 2e-14, 0.2, np.nan])
    np.testing.assert_array_equal(stable_percentiles(values)[:3], [0.5, 0.5, 1.0])
    assert np.isnan(stable_percentiles(values)[3])
    np.testing.assert_array_equal(stable_percentiles(np.ones(4)), np.full(4, 0.625))
    with pytest.raises(ValueError):
        stable_percentiles(np.array([np.inf]))


def test_centroid_prefix_windows_edges_repeats_and_zero_norm():
    rng = np.random.default_rng(7)
    block = rng.normal(size=(60, 8)).astype(np.float32)
    block /= np.linalg.norm(block, axis=1, keepdims=True)
    x = np.vstack([block, block, block])
    changes = centroid_changes(x)
    for w in (1, 3, 5, 10, 20):
        observed = changes[w]
        assert np.isnan(observed[: w - 1]).all()
        assert np.isnan(observed[len(x) - w :]).all()
        for t in range(w, len(x) - w + 1):
            a = x[t - w : t].astype(float).sum(axis=0)
            b = x[t : t + w].astype(float).sum(axis=0)
            expected = 1 - np.dot(a / np.linalg.norm(a), b / np.linalg.norm(b))
            assert observed[t - 1] == pytest.approx(expected, abs=1e-14)
        ranks = stable_percentiles(observed)
        np.testing.assert_array_equal(
            ranks[w - 1 : 60 - w], ranks[60 + w - 1 : 120 - w]
        )
    alternating = np.tile([[1.0, 0.0], [-1.0, 0.0]], (10, 1))
    assert np.isnan(centroid_changes(alternating, windows=(2,))[2]).all()
    assert all(
        np.isnan(v).all()
        for v in centroid_changes(np.ones((2, 2)), (3, 5, 10, 20)).values()
    )


def test_multiscale_rule_requires_both_encoders_every_scale():
    table = pd.DataFrame(
        {
            f"{e}_percentile_w{w}": [1.0, 1.0, 1.0, 1.0]
            for e in ("clip", "dinov2")
            for w in (1, 3, 5, 10, 20)
        }
    )
    table["clip_percentile_w1"] = 0.01
    table.loc[0, "dinov2_percentile_w5"] = 0.94
    table.loc[1, "clip_percentile_w10"] = 0.92
    table.loc[2, "dinov2_percentile_w20"] = np.nan
    table.loc[3, "clip_percentile_w3"] = 0.1
    scores = multiscale_scores(table)
    assert scores.S[:2].tolist() == [0.94, 0.92]
    assert np.isnan(scores.S[2])
    assert scores.S[3] == 1.0 and scores.F3[3] == 0.1
    assert scores.P1.eq(0.01).all()


def controlled_scores(n=100):
    return pd.DataFrame(
        {
            "video_id": "synthetic",
            "sample_index": np.arange(1, n),
            "S": 0.0,
            "coarse_median": 0.0,
            "P1": 0.0,
            "F3": 0.1,
        }
    )


def test_merge_gap_coarse_tiebreak_high_confidence_is_annotation():
    scores = controlled_scores()
    for t, s, median, p1 in [
        (30, 0.96, 0.97, 0.2),
        (33, 0.96, 0.98, 0.2),
        (36, 0.96, 0.98, 0.3),
        (39, 0.96, 0.98, 0.3),
        (43, 0.99, 0.99, 0.5),
    ]:
        scores.loc[scores.sample_index.eq(t), ["S", "coarse_median", "P1"]] = [
            s,
            median,
            p1,
        ]
    events = localize_events(scores, SequenceConfig(), "synthetic")
    assert events.candidate_count.tolist() == [4, 1]
    assert events.coarse_sample_index.tolist() == [36, 43]
    assert events.coarse_P1.tolist() == [0.3, 0.5]
    assert events.high_confidence.tolist() == [False, True]
    assert events.status.eq("candidate").all()


def test_midpoint_excludes_shared_peak_and_f3_ties_use_nearest_then_lower():
    scores = controlled_scores()
    scores.loc[scores.sample_index.isin([40, 50]), "S"] = 0.99
    scores.loc[scores.sample_index.eq(45), "F3"] = 1.0
    scores.loc[scores.sample_index.isin([48, 52]), "F3"] = 0.98
    events = localize_events(scores, SequenceConfig(), "synthetic")
    assert events.localized_sample_index.tolist() == [45, 48]
    assert events.search_end_sample_index[0] == 45
    assert events.search_start_sample_index[1] == 46
    # P1/F1 cannot override final F3 localization or admit a non-candidate cut.
    scores.loc[scores.sample_index.eq(49), "P1"] = 1.0
    pd.testing.assert_frame_equal(
        events, localize_events(scores, SequenceConfig(), "synthetic")
    )


def test_coarse_P1_is_invariant_to_raw_encoder_scales_with_equivalent_ranks():
    samples = np.arange(1, 100)
    clip = np.linspace(0.0001, 0.002, len(samples))
    dino = np.linspace(0.01, 0.2, len(samples))
    candidates = np.isin(samples, [30, 33])
    clip[candidates] = [0.009, 0.008]
    dino[candidates] = [0.4, 1.2]
    # A non-candidate temporal cut changes the DINO percentile denominator/order.
    dino[samples == 80] = 0.7
    expected = None
    raw_min_winners = set()
    for clip_scale, dino_scale in ((1.0, 1.0), (100.0, 1.0), (1.0, 0.01)):
        table = pd.DataFrame({"video_id": "synthetic", "sample_index": samples})
        for encoder, raw in (
            ("clip", clip * clip_scale),
            ("dinov2", dino * dino_scale),
        ):
            table[f"{encoder}_change_w1"] = raw
            table[f"{encoder}_percentile_w1"] = stable_percentiles(raw)
            for w in (3, 5, 10, 20):
                table[f"{encoder}_percentile_w{w}"] = np.where(candidates, 0.96, 0.2)
            table.loc[samples == 34, f"{encoder}_percentile_w3"] = 1.0
        scores = multiscale_scores(table)
        events = localize_events(scores, SequenceConfig(), "synthetic-scale-test")
        assert events.coarse_sample_index.tolist() == [33]
        assert events.coarse_P1.iloc[0] == pytest.approx(98 / 99)
        assert events.localized_sample_index.tolist() == [34]
        if expected is not None:
            pd.testing.assert_frame_equal(events, expected)
        expected = events
        # This same fixture exposes the rejected policy: its winner switches
        # under scale changes that preserve both encoders' percentile ordering.
        raw_min = np.minimum(
            clip[candidates] * clip_scale, dino[candidates] * dino_scale
        )
        raw_min_winners.add(int(samples[candidates][np.argmax(raw_min)]))
    assert raw_min_winners == {30, 33}


def test_confirmed_csv_needs_no_P1_field_when_third_tie_does_not_decide(tmp_path):
    scores = controlled_scores()
    scores.loc[scores.sample_index.eq(30), ["S", "coarse_median", "P1"]] = [
        0.99,
        0.99,
        0.1,
    ]
    scores.loc[scores.sample_index.eq(33), ["S", "coarse_median", "P1"]] = [
        0.98,
        0.99,
        1.0,
    ]
    scores.loc[scores.sample_index.eq(32), "F3"] = 1.0
    initial = localize_events(
        scores, SequenceConfig(), "synthetic-review-compatibility"
    )
    validation = synthetic_review(tmp_path / "review", initial, reject=())
    scores["P1"] = 1 - scores.P1
    revised = localize_events(
        scores, SequenceConfig(), "synthetic-review-compatibility"
    )
    assert initial.coarse_P1.iloc[0] != revised.coarse_P1.iloc[0]
    assert revised.coarse_sample_index.tolist() == initial.coarse_sample_index.tolist()
    assert (
        revised.localized_sample_index.tolist()
        == initial.localized_sample_index.tolist()
    )
    review, signature = load_confirmed_validation(validation, revised)
    assert set(review) == set(REVIEW_COLUMNS)
    assert review.decision.eq("accept").all()
    assert not review.sequence_boundary_committed.any()
    assert signature["metadata"]["ground_truth"] is False


def test_per_video_ranking_independent_and_record_index_alignment(tmp_path):
    sources = synthetic_sources(tmp_path / "source")
    loaded = load_sources(*sources)
    records = loaded.records
    assert not records.clip_embedding_row.equals(records.dinov2_embedding_row)
    scores, events = detect_tables(
        records, loaded.embeddings, SequenceConfig(), "synthetic"
    )
    for w in (1, 3, 5, 10, 20):
        np.testing.assert_array_equal(
            scores[f"clip_percentile_w{w}"], scores[f"dinov2_percentile_w{w}"]
        )
        for _, video in scores.groupby("video_id"):
            for encoder in ("clip", "dinov2"):
                np.testing.assert_array_equal(
                    video[f"{encoder}_percentile_w{w}"],
                    stable_percentiles(video[f"{encoder}_change_w{w}"].to_numpy()),
                )
    assert "instantaneous_consensus" not in scores
    np.testing.assert_array_equal(
        scores.P1, np.minimum(scores.clip_percentile_w1, scores.dinov2_percentile_w1)
    )
    subset = records.loc[records.video_id.eq("video-0")]
    smaller, _ = detect_tables(subset, loaded.embeddings, SequenceConfig(), "synthetic")
    pd.testing.assert_frame_equal(
        scores.loc[scores.video_id.eq("video-0")].reset_index(drop=True), smaller
    )
    assert events.status.eq("candidate").all()


def test_build_full_coverage_cut_semantics_duplicates_and_reject(detected, tmp_path):
    sources, detection, validation = detected
    output = build_to_store(detection, *sources, validation, tmp_path / "built")
    result = verify_directory(output, *sources, validation)
    assert result["quality_valid"], result
    source = pd.read_parquet(sources[0])
    records = pd.read_parquet(output / "occurrence_assignments.parquet")
    instances = pd.read_parquet(output / "sequence_instances.parquet")
    boundaries = pd.read_parquet(output / "boundaries.parquet")
    review = pd.read_parquet(output / "manual_review.parquet")
    assert len(records) == len(source) and records.frame_id.is_unique
    assert (
        records.content_id.value_counts()
        .sort_index()
        .equals(source.content_id.value_counts().sort_index())
    )
    assert (
        len(instances) == source.video_id.nunique() + review.decision.eq("accept").sum()
    )
    assert (
        instances.sequence_id.is_unique
        and records.groupby("sequence_id").video_id.nunique().eq(1).all()
    )
    assert set(boundaries.event_id) == set(
        review.loc[review.decision.eq("accept"), "event_id"]
    )
    assert not set(review.loc[review.decision.eq("reject"), "event_id"]) & set(
        boundaries.event_id
    )
    assert (
        boundaries.sequence_boundary_committed.all()
        and not boundaries.ground_truth.any()
    )
    assert not boundaries.input_sequence_boundary_committed.any()
    for b in boundaries.itertuples():
        seqs = instances.loc[instances.video_id.eq(b.video_id)]
        assert seqs.end_sample_index.eq(b.localized_sample_index - 1).sum() == 1
        assert seqs.start_sample_index.eq(b.localized_sample_index).sum() == 1
        before, after = (
            records.loc[records.video_id.eq(b.video_id)]
            .set_index("sample_index")
            .loc[
                [b.localized_sample_index - 1, b.localized_sample_index], "sequence_id"
            ]
        )
        assert before != after
    assert len(pd.read_parquet(output / "dependency_edges.parquet")) > 0
    assert not {
        "cluster_id",
        "split_id",
        "new_split",
        "original_split",
        "reduction_space_id",
    } & set(records)
    assert build_to_store(detection, *sources, validation, tmp_path / "built") == output
    assert not verify_directory(output, *sources)["quality_valid"]


def test_dependency_star_components_transitive_and_singletons():
    facts = [
        ("a", "block0"),
        ("a", "block1"),
        ("b", "block0"),
        ("b", "block1"),
        ("c", "block0"),
        ("c", "isolated"),
        ("d", "isolated"),
        ("e", "unique"),
    ]
    records = pd.DataFrame(
        [
            {
                "sequence_id": s,
                "content_id": c,
                "frame_id": f"frame-{i}",
                "video_id": "synthetic",
                "sample_index": i,
                "timestamp_seconds": float(i),
            }
            for i, (s, c) in enumerate(facts)
        ]
    )
    edges, support, membership = exact_dependencies(records, list("abcde"), "set")
    assert len(edges) == 4  # block0 star=2, block1=1, isolated=1.
    assert len(support) == 7
    assert len({membership[s] for s in "abcd"}) == 1
    assert membership["e"] != membership["a"]
    again = exact_dependencies(records.iloc[::-1], list("edcba"), "set")
    pd.testing.assert_frame_equal(edges, again[0])
    pd.testing.assert_frame_equal(support, again[1])
    assert membership == again[2]
    for edge in edges.itertuples():
        for side in (edge.left_sequence_id, edge.right_sequence_id):
            assert (
                (support.sequence_id == side) & (support.content_id == edge.content_id)
            ).any()


@pytest.mark.parametrize(
    "field,value",
    [
        ("artifact_kind", "provisional_boundary_validation"),
        ("ground_truth", True),
        ("review_status", "provisional"),
        ("manual_confirmation_complete", False),
        ("manual_confirmation_required_before_sequence_commit", True),
        ("accepted_count", 999),
        ("event_count", 999),
        ("rejected_count", 999),
        ("boundary_type_counts", {}),
        ("ground_truth", "false"),
    ],
)
def test_refuse_provisional_invalid_metadata(detected, field, value, tmp_path):
    sources, detection, validation = detected
    meta = read_json(validation / "metadata.json")
    meta[field] = value
    write_json(validation / "metadata.json", meta)
    with pytest.raises(ValueError):
        build_to_store(detection, *sources, validation, tmp_path / "built")
    assert not (tmp_path / "built").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("decision", "maybe"),
        ("boundary_type", "cluster"),
        ("manual_confirmation", False),
        ("review_status", "provisional"),
        ("localized_sample_index", 7),
        ("search_start", 2),
        ("f3_score", 0.01),
        ("high_confidence", False),
    ],
)
def test_refuse_invalid_rows_and_stale_localization(detected, field, value):
    _, detection, validation = detected
    review = pd.read_csv(validation / "boundary_validation.csv")
    if field == "high_confidence":
        value = not bool(review.loc[0, field])
    review.loc[0, field] = value
    write_review(validation, review)
    with pytest.raises((ValueError, AssertionError)):
        load_confirmed_validation(
            validation, pd.read_parquet(detection / "candidate_events.parquet")
        )


def test_prefiltered_accepted_file_never_authorizes_rejected_rows(detected):
    _, detection, validation = detected
    review = pd.read_csv(validation / "boundary_validation.csv")
    review.to_csv(validation / "accepted_boundary_candidates.csv", index=False)
    with pytest.raises(AssertionError):
        load_confirmed_validation(
            validation, pd.read_parquet(detection / "candidate_events.parquet")
        )


@pytest.mark.parametrize(
    "change",
    [
        "gap",
        "fps",
        "historical_split",
        "cluster",
        "wrong_encoder",
        "mapping",
        "not_l2",
        "feature_id",
    ],
)
def test_source_contract_invalid_inputs(tmp_path, change):
    sources = synthetic_sources(tmp_path / "source", sizes=(80,))
    manifest = pd.read_parquet(sources[0])
    if change == "gap":
        manifest = manifest.iloc[1:]
    elif change == "fps":
        manifest["sample_fps"] = 2.0
        manifest["timestamp_seconds"] = manifest.sample_index.astype(float) / 2
    elif change == "historical_split":
        manifest["original_split"] = "train"
    elif change == "cluster":
        manifest["cluster_id"] = 0
    elif change in {"wrong_encoder", "feature_id"}:
        meta = read_json(sources[1] / "metadata.json")
        meta["extractor" if change == "wrong_encoder" else "feature_space_id"] = "wrong"
        write_json(sources[1] / "metadata.json", meta)
    elif change == "mapping":
        records = pd.read_parquet(sources[1] / "record_index.parquet")
        records.loc[0, "embedding_row"] = 2
        records.to_parquet(sources[1] / "record_index.parquet", index=False)
    else:
        x = np.load(sources[1] / "embeddings_l2.npy")
        x[0] = 0
        np.save(sources[1] / "embeddings_l2.npy", x)
    manifest.to_parquet(sources[0], index=False)
    with pytest.raises(ValueError):
        load_sources(*sources)


def test_deterministic_ids_outputs_and_read_only_sources(detected, tmp_path):
    sources, detection, validation = detected
    paths = [
        sources[0],
        *(p for d in sources[1:] for p in d.iterdir()),
        *validation.iterdir(),
    ]
    before = {str(p): file_sha256(p) for p in paths}
    duplicate = detect_to_store(*sources, SequenceConfig(), tmp_path / "second")
    a, b = (
        read_json(detection / "metadata.json"),
        read_json(duplicate / "metadata.json"),
    )
    assert (
        a["artifact_id"] == b["artifact_id"]
        and a["output_checksums"] == b["output_checksums"]
    )
    first = build_to_store(detection, *sources, validation, tmp_path / "first_build")
    second = build_to_store(duplicate, *sources, validation, tmp_path / "second_build")
    a, b = read_json(first / "metadata.json"), read_json(second / "metadata.json")
    assert (
        a["artifact_id"] == b["artifact_id"]
        and a["output_checksums"] == b["output_checksums"]
    )
    assert before == {str(p): file_sha256(p) for p in paths}
    meta = read_json(validation / "metadata.json")
    meta["diagnostic_source"] = "another confirmed provenance"
    write_json(validation / "metadata.json", meta)
    different = build_to_store(
        detection, *sources, validation, tmp_path / "first_build"
    )
    assert different != first
    assert not verify_directory(first, *sources, validation)["quality_valid"]


@pytest.mark.parametrize(
    "table,column",
    [
        ("sequence_instances", "end_sample_index"),
        ("sequence_instances", "sequence_id"),
        ("occurrence_assignments", "sequence_id"),
        ("boundaries", "event_id"),
        ("dependency_edges", "content_id"),
        ("sequence_instances", "exact_duplicate_dependency_group_id"),
        ("dependency_support", "frame_id"),
        ("temporal_scores", "S"),
    ],
)
def test_verifier_catches_tampering_even_with_updated_checksums(
    detected, tmp_path, table, column
):
    sources, detection, validation = detected
    output = build_to_store(detection, *sources, validation, tmp_path / "built")
    path = output / f"{table}.parquet"
    data = pd.read_parquet(path)
    data.loc[0, column] = (
        999 if column in {"end_sample_index", "event_id", "S"} else "tampered"
    )
    data.to_parquet(path, index=False)
    metadata = read_json(output / "metadata.json")
    metadata["output_checksums"][path.name] = file_sha256(path)
    write_json(output / "metadata.json", metadata)
    result = verify_directory(output, *sources, validation)
    assert not result["quality_valid"] and "error" in result


def test_summary_is_lightweight_and_cli_formal_verification(
    detected, monkeypatch, tmp_path
):
    sources, detection, validation = detected
    runner = CliRunner()
    source_flags = [
        "--manifest",
        str(sources[0]),
        "--clip-features",
        str(sources[1]),
        "--dinov2-features",
        str(sources[2]),
    ]
    assert runner.invoke(app, ["sequences", "--help"]).exit_code == 0
    result = runner.invoke(app, ["sequences", "verify", str(detection), *source_flags])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["quality_valid"]
    result = runner.invoke(
        app,
        [
            "sequences",
            "build",
            "--detection",
            str(detection),
            *source_flags,
            "--validation",
            str(validation),
            "--output",
            str(tmp_path / "built"),
        ],
    )
    assert result.exit_code == 0, result.output
    output = Path(result.output.strip())
    assert (
        runner.invoke(
            app, ["sequences", "verify", str(output), *source_flags]
        ).exit_code
        == 1
    )
    assert (
        runner.invoke(
            app,
            [
                "sequences",
                "verify",
                str(output),
                *source_flags,
                "--validation",
                str(validation),
            ],
        ).exit_code
        == 0
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Summary must not load arrays/tables or hash sources")

    monkeypatch.setattr(pd, "read_parquet", forbidden)
    monkeypatch.setattr(np, "load", forbidden)
    assert summarize_directory(output)["quality_valid"] is None
    result = runner.invoke(app, ["sequences", "summary", str(output)])
    assert (
        result.exit_code == 0
        and not json.loads(result.output)["verification_performed"]
    )


def test_no_events_builds_one_instance_per_source_without_inventing_boundaries(
    tmp_path,
):
    sources = synthetic_sources(tmp_path / "source", sizes=(20, 1))
    detection = detect_to_store(*sources, SequenceConfig(), tmp_path / "output")
    events = pd.read_parquet(detection / "candidate_events.parquet")
    assert events.empty
    validation = synthetic_review(tmp_path / "review", events)
    output = build_to_store(detection, *sources, validation, tmp_path / "output")
    result = verify_directory(output, *sources, validation)
    assert result["quality_valid"], result
    instances = pd.read_parquet(output / "sequence_instances.parquet")
    assert len(instances) == 2
    assert instances.start_sample_index.eq(0).all()


def test_config_rejects_foreign_policies_and_partial_outputs(detected, tmp_path):
    sources, detection, _ = detected
    for overrides in (
        {"cluster_id": 0},
        {"windows": [1, 5, 10, 20]},
        {"rounding_decimals": 10},
        {"merge_gap": -1},
        {"candidate_threshold": np.nan},
        {"sample_fps": 2.0},
        {"coarse_tie_policy": "score_median_instantaneous_raw_consensus_lower_index"},
    ):
        with pytest.raises(ValueError):
            SequenceConfig(**overrides)
    metadata = read_json(detection / "metadata.json")
    partial = tmp_path / "partial" / "candidates" / metadata["artifact_id"]
    partial.mkdir(parents=True)
    (partial / "preserved.partial").write_text(
        "potentially resumable", encoding="utf-8"
    )
    with pytest.raises(FileNotFoundError):
        detect_to_store(*sources, SequenceConfig(), tmp_path / "partial")
    assert (partial / "preserved.partial").read_text(
        encoding="utf-8"
    ) == "potentially resumable"


def test_accepted_low_confidence_boundaries_are_committed(tmp_path):
    sources = synthetic_sources(tmp_path / "source")
    config = SequenceConfig(high_confidence_threshold=1.0)
    detection = detect_to_store(*sources, config, tmp_path / "output")
    events = pd.read_parquet(detection / "candidate_events.parquet")
    assert not events.high_confidence.all()
    validation = synthetic_review(tmp_path / "review", events, reject=())
    output = build_to_store(detection, *sources, validation, tmp_path / "output")
    boundaries = pd.read_parquet(output / "boundaries.parquet")
    assert len(boundaries) == len(events)
    assert (~boundaries.high_confidence).sum() == (~events.high_confidence).sum()


def test_detection_never_calls_pairwise_or_reduction_clustering(tmp_path, monkeypatch):
    import ast

    import flir_pipeline.sequences as package
    import flir_pipeline.similarity.cosine as cosine

    def forbidden(*args, **kwargs):
        raise AssertionError("Sequences cannot use an NxN similarity matrix")

    monkeypatch.setattr(cosine, "compute_cosine_similarity", forbidden)
    monkeypatch.setattr(np, "triu_indices", forbidden)
    sources = synthetic_sources(tmp_path / "source")
    detection = detect_to_store(*sources, SequenceConfig(), tmp_path / "output")
    assert verify_directory(detection, *sources)["quality_valid"]
    for path in Path(package.__file__).parent.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            imports = (
                [node.module]
                if isinstance(node, ast.ImportFrom)
                else (
                    [n.name for n in node.names] if isinstance(node, ast.Import) else []
                )
            )
            assert not any(
                name
                and (
                    "flir_pipeline.clustering" in name
                    or "flir_pipeline.reduction" in name
                    or "flir_pipeline.splitting" in name
                )
                for name in imports
            )


@pytest.mark.parametrize("change", ["identity", "policy", "checksum", "source_binding"])
def test_candidate_verifier_refuses_metadata_tampering(detected, change):
    sources, detection, _ = detected
    metadata = read_json(detection / "metadata.json")
    if change == "identity":
        metadata["artifact_id"] = "0" * 16
    elif change == "policy":
        metadata["semantics"]["ground_truth"] = True
    elif change == "checksum":
        metadata["output_checksums"]["candidate_events.parquet"] = "0" * 64
    else:
        metadata["identity"]["sources"]["manifest_sha256"] = "0" * 64
    write_json(detection / "metadata.json", metadata)
    assert not verify_directory(detection, *sources)["quality_valid"]
