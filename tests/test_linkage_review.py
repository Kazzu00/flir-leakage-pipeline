"""Synthetic manual calibration; decisions are never inferred from scores."""

import hashlib
import io
import json
import shutil
from dataclasses import replace

import pandas as pd
import pytest
from PIL import Image
from test_linkage import labeled_inputs
from test_sequences import synthetic_review, synthetic_sources
from typer.testing import CliRunner

from flir_pipeline.cli import app
from flir_pipeline.data.identity import dataset_id_from_manifest
from flir_pipeline.linkage.base import LinkageConfig
from flir_pipeline.linkage.review_model import (
    MANUAL_COLUMNS,
    ReviewConfig,
    apply_decisions,
    decision_summary,
    validate_sample,
)
from flir_pipeline.linkage.review_sources import ReviewPaths
from flir_pipeline.linkage.review_storage import (
    init_review,
    inspect_snapshot,
    record_review,
    summarize_review,
    verify_review,
)
from flir_pipeline.linkage.sources import InputPaths
from flir_pipeline.linkage.storage import build_to_store as build_linkage
from flir_pipeline.sequences.base import SequenceConfig
from flir_pipeline.sequences.storage import build_to_store as build_sequences
from flir_pipeline.sequences.storage import detect_to_store
from flir_pipeline.similarity.storage import file_sha256, read_json, write_json


@pytest.fixture
def blank_review():
    return pd.DataFrame(
        [
            {
                "labeled_content_id": "q1",
                "proposed_visual_dependency_group_id": "g1",
                "review_stratum": "agreement",
                "review_query_id": "r1",
            },
            {
                "labeled_content_id": "q2",
                "proposed_visual_dependency_group_id": "g2",
                "review_stratum": "agreement",
                "review_query_id": "r2",
            },
            {
                "labeled_content_id": "q3",
                "proposed_visual_dependency_group_id": "g2",
                "review_stratum": "disagreement",
                "review_query_id": "r3",
            },
            {
                "labeled_content_id": "q4",
                "proposed_visual_dependency_group_id": "g3",
                "review_stratum": "disagreement",
                "review_query_id": "r4",
            },
        ]
    ).assign(**dict.fromkeys(MANUAL_COLUMNS, ""))


def decisions(*rows):
    return pd.DataFrame(
        rows,
        columns=[
            "labeled_content_id",
            "proposed_visual_dependency_group_id",
            "manual_decision",
            "manual_notes",
        ],
    )


def apply(review, changes, **kwargs):
    return apply_decisions(
        review,
        changes,
        reviewer="synthetic-reviewer",
        source="manual-contact-sheets",
        timestamp="2026-01-01T00:00:00+00:00",
        **kwargs,
    )


def test_group_diverse_summary_denominators_and_unresolved(blank_review):
    updated, events = apply(
        blank_review,
        decisions(
            ("q1", "g1", "supported", "group evidence"),
            ("q2", "g2", "ambiguous", "multiple plausible groups"),
            ("q3", "g2", "unsupported", "different scene"),
        ),
    )
    summary = decision_summary(updated)
    assert summary["overall"] == {
        "query_count": 4,
        "distinct_group_count": 3,
        "decision_counts": {
            "supported": 1,
            "ambiguous": 1,
            "unsupported": 1,
            "blank": 1,
        },
        "decision_rates": {
            "supported": 0.25,
            "ambiguous": 0.25,
            "unsupported": 0.25,
            "blank": 0.25,
        },
        "unresolved_count": 2,
    }
    assert summary["by_stratum"][0]["query_count"] == 2
    assert summary["by_stratum"][0]["distinct_group_count"] == 2
    assert summary["by_visual_dependency_group"][1]["query_count"] == 2
    assert len(summary["by_stratum_and_group"]) == 4
    assert summary["semantics"]["confirmed_matches_created"] is False
    assert summary["semantics"]["ground_truth"] is False
    assert summary["semantics"]["split_created"] is False
    assert "accuracy" not in summary["overall"]
    assert len(events) == 3
    assert blank_review.manual_decision.eq("").all()


@pytest.mark.parametrize(
    "decision",
    ["accept", "reject", "confirmed", "true", "Supported", " supported", "NaN"],
)
def test_invalid_manual_vocabulary(blank_review, decision):
    with pytest.raises(ValueError, match="vocabulary"):
        apply(blank_review, decisions(("q1", "g1", decision, "")))


@pytest.mark.parametrize(
    ("query", "group"), [("unknown", "g1"), ("q1", "unknown"), ("q1", "g2")]
)
def test_reject_unknown_queries_or_groups(blank_review, query, group):
    with pytest.raises(ValueError, match="Unknown"):
        apply(blank_review, decisions((query, group, "supported", "")))


def test_blank_ambiguous_clear_and_readonly_provenance(blank_review):
    updated, _ = apply(
        blank_review, decisions(("q1", "g1", "ambiguous", "Not an exact-frame claim"))
    )
    cleared, changes = apply(
        updated, decisions(("q1", "g1", "", "Needs another reviewer"))
    )
    assert cleared.manual_decision.eq("").all()
    assert changes[0]["before"]["manual_decision"] == "ambiguous"
    assert changes[0]["after"]["manual_decision"] == ""
    assert cleared.loc[0, "reviewer"] == "synthetic-reviewer"
    imported = blank_review.copy()
    imported.loc[0, "review_stratum"] = "silently-reassigned"
    with pytest.raises(ValueError, match="readonly"):
        apply(blank_review, imported)
    imported = blank_review.copy()
    imported.loc[0, "reviewer"] = "invented-reviewer"
    with pytest.raises(ValueError, match="readonly"):
        apply(blank_review, imported)


def test_duplicate_query_or_pair_rejected(blank_review):
    repeated = pd.concat([blank_review, blank_review.iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match="one proposed group"):
        validate_sample(repeated)
    imported = decisions(("q1", "g1", "supported", ""), ("q1", "g1", "ambiguous", ""))
    with pytest.raises(ValueError, match="unique"):
        apply(blank_review, imported)


def test_noop_is_deterministic_and_future_provenance_not_silently_overwritten(
    blank_review,
):
    updated, changes = apply(blank_review, decisions(("q1", "g1", "", "")))
    pd.testing.assert_frame_equal(updated, blank_review)
    assert changes == []
    original, _ = apply(
        blank_review, decisions(("q1", "g1", "supported", "manual evidence"))
    )
    with pytest.raises(ValueError, match="predates"):
        apply_decisions(
            original,
            decisions(("q1", "g1", "unsupported", "changed")),
            reviewer="another",
            source="second-review",
            timestamp="2025-01-01T00:00:00+00:00",
        )
    assert 'confirmed_matches_created": false' in json.dumps(decision_summary(original))


def bind_synthetic_images(manifest_path, features, image_root, *, offset=0):
    manifest = pd.read_parquet(manifest_path)
    path_column = "image_path" if "image_path" in manifest else "source_member_path"
    if path_column not in manifest:
        manifest[path_column] = manifest.image_member_path
    mapping, encoded = {}, {}
    for i, cid in enumerate(sorted(manifest.content_id.unique())):
        buffer = io.BytesIO()
        Image.new("RGB", (24, 18), ((i + offset) % 256, (i + offset) // 256, 57)).save(
            buffer, format="PNG"
        )
        encoded[cid] = buffer.getvalue()
        mapping[cid] = hashlib.sha256(encoded[cid]).hexdigest()
    for row in manifest.to_dict("records"):
        path = image_root / row[path_column]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(encoded[row["content_id"]])
    manifest.content_id = manifest.content_id.map(mapping)
    manifest.image_sha256 = manifest.content_id
    manifest.to_parquet(manifest_path, index=False)
    for feature in features:
        for name in ("content_index", "record_index"):
            index = pd.read_parquet(feature / f"{name}.parquet")
            index.content_id = index.content_id.map(mapping)
            index.to_parquet(feature / f"{name}.parquet", index=False)
        meta = read_json(feature / "metadata.json")
        meta["dataset_id"] = dataset_id_from_manifest(manifest)
        write_json(feature / "metadata.json", meta)


@pytest.fixture(scope="module")
def review_template(tmp_path_factory):
    root = tmp_path_factory.mktemp("manual-calibration")
    video, clip, dino = synthetic_sources(root / "video", sizes=(360,))
    bind_synthetic_images(video, (clip, dino), root / "video-images")
    detected = detect_to_store(video, clip, dino, SequenceConfig(), root / "sequences")
    review = synthetic_review(
        root / "boundary-review",
        pd.read_parquet(detected / "candidate_events.parquet"),
        reject=(),
    )
    sequences = build_sequences(detected, video, clip, dino, review, root / "sequences")
    labeled, labeled_clip, labeled_dino = labeled_inputs(root / "labeled", (clip, dino))
    bind_synthetic_images(
        labeled, (labeled_clip, labeled_dino), root / "labeled-images", offset=1000
    )
    linkage = build_linkage(
        InputPaths(labeled, video, labeled_clip, labeled_dino, clip, dino, sequences),
        LinkageConfig(top_k=2),
        root / "linkage",
    )
    instances = pd.read_parquet(sequences / "sequence_instances.parquet")
    membership = instances[["sequence_id"]].copy()
    # The isolated repeated frame also joins two plateaus: the synthetic manual
    # groups must preserve that existing exact-copy constraint.
    group_names = {
        key: f"group-{i}"
        for i, key in enumerate(
            sorted(instances.exact_duplicate_dependency_group_id.unique())
        )
    }
    membership["visual_dependency_group_id"] = (
        instances.exact_duplicate_dependency_group_id.map(group_names)
    )
    dependencies = root / "visual-dependencies"
    dependencies.mkdir()
    membership.to_csv(dependencies / "membership.csv", index=False)
    write_json(
        dependencies / "metadata.json",
        {
            "artifact_kind": "synthetic_confirmed_visual_dependencies",
            "artifact_id": "synthetic-dependencies-v1",
            "ground_truth": False,
            "review_status": "confirmed_manual_review",
            "manual_confirmation_complete": True,
            "split_created": False,
            "sequence_instances_merged": False,
            "sequence_set_id": read_json(sequences / "metadata.json")["artifact_id"],
            "output_checksums": {
                "membership.csv": file_sha256(dependencies / "membership.csv")
            },
        },
    )
    candidates = pd.read_parquet(linkage / "content_candidates.parquet")
    occurrences = pd.read_parquet(linkage / "candidate_occurrences.parquet").merge(
        membership, on="sequence_id", validate="many_to_one"
    )
    sample = []
    for i, (query, pairs) in enumerate(
        candidates.groupby("labeled_content_id", sort=True)
    ):
        groups = sorted(
            occurrences.loc[
                occurrences.video_content_id.isin(pairs.video_content_id),
                "visual_dependency_group_id",
            ].unique()
        )
        sample.append(
            {
                "labeled_content_id": query,
                "proposed_visual_dependency_group_id": groups[-1],
                "review_stratum": "agreement"
                if pairs.both_topk.all()
                else "disagreement",
                "sampling_note": f"Synthetic stratified query {i}; not representative",
            }
        )
    sample_path = root / "sample.csv"
    pd.DataFrame(sample).to_csv(sample_path, index=False)
    paths = ReviewPaths(sample_path, linkage, labeled, sequences, dependencies)
    artifact = init_review(
        paths,
        ReviewConfig(),
        root / "reviews",
        labeled_images_root=root / "labeled-images",
        video_images_root=root / "video-images",
    )
    return root, paths, artifact


@pytest.fixture
def review_publication(review_template, tmp_path):
    root, paths, artifact = review_template
    copied = tmp_path / "source"
    shutil.copytree(root, copied)
    paths = replace(
        paths,
        **{
            key: copied / getattr(paths, key).relative_to(root)
            for key in (
                "calibration_sample",
                "linkage",
                "labeled_manifest",
                "sequence_set",
                "visual_dependencies",
            )
        },
    )
    return copied, paths, copied / artifact.relative_to(root)


def test_init_preserves_ambiguity_lineage_and_has_no_auto_decisions(review_publication):
    _, paths, artifact = review_publication
    checks = verify_review(artifact, paths)
    assert checks["quality_valid"], checks
    meta, review, history = inspect_snapshot(artifact)
    assert len(review) == 3 and history == []
    assert review.manual_decision.eq("").all()
    assert review.proposed_visual_dependency_group_id.nunique() >= 2
    candidates = pd.read_parquet(artifact / "candidates.parquet")
    occurrences = pd.read_parquet(artifact / "candidate_occurrences.parquet")
    labeled = pd.read_parquet(artifact / "labeled_occurrences.parquet")
    assert len(labeled) == 4 and labeled.content_id.nunique() == 3
    assert (
        labeled[labeled.content_id.duplicated(keep=False)].label_sha256.nunique() == 2
    )
    assert occurrences.groupby("video_content_id").sequence_id.nunique().gt(1).any()
    assert occurrences.video_content_id.duplicated().any()
    context = pd.read_parquet(artifact / "temporal_context.parquet")
    for query in review.itertuples():
        selected = candidates[
            candidates.labeled_content_id.eq(query.labeled_content_id)
        ]
        centers = occurrences[
            occurrences.video_content_id.isin(selected.video_content_id)
        ]
        actual = context[context.review_query_id.eq(query.review_query_id)]
        assert set(actual.center_video_frame_id) == set(centers.video_frame_id)
        assert actual.contact_sheet.nunique() == len(centers)
        assert actual.relative_seconds.between(-3, 3).all()
        details = json.loads(query.candidate_details_json)
        assert sum(len(item["occurrences"]) for item in details) == len(centers)
    assert context.is_center.groupby(context.contact_sheet).sum().eq(1).all()
    assert all((artifact / name).is_file() for name in context.contact_sheet)
    assert (
        meta["ground_truth"] is False
        and meta["confirmed_matches_created"] is False
        and meta["split_created"] is False
    )


def test_deterministic_contact_sheets_and_init_reuse(review_publication):
    root, paths, artifact = review_publication
    before = {
        p.relative_to(artifact).as_posix(): file_sha256(p)
        for p in artifact.rglob("*")
        if p.is_file()
    }
    again = init_review(
        paths,
        ReviewConfig(),
        artifact.parent,
        labeled_images_root=root / "labeled-images",
        video_images_root=root / "video-images",
    )
    assert again == artifact
    assert before == {
        p.relative_to(artifact).as_posix(): file_sha256(p)
        for p in artifact.rglob("*")
        if p.is_file()
    }
    rebuilt = init_review(
        paths,
        ReviewConfig(),
        root / "rebuilt",
        labeled_images_root=root / "labeled-images",
        video_images_root=root / "video-images",
    )
    assert rebuilt.name == artifact.name
    first, second = (
        read_json(artifact / "metadata.json"),
        read_json(rebuilt / "metadata.json"),
    )
    assert first["artifact_id"] == second["artifact_id"]
    assert first["output_checksums"] == second["output_checksums"]


def test_record_is_immutable_replayable_and_does_not_promote_supported(
    review_publication,
):
    root, paths, artifact = review_publication
    original_checks = {p.name: file_sha256(p) for p in paths.linkage.iterdir()}
    frame = pd.read_csv(artifact / "review.csv", dtype=str, keep_default_na=False)
    frame.loc[0, ["manual_decision", "manual_notes"]] = [
        "supported",
        "Group evidence only; not exact sequence",
    ]
    frame.loc[1, "manual_decision"] = "ambiguous"
    imported = root / "manual.csv"
    frame.to_csv(imported, index=False)
    revision = record_review(
        artifact,
        imported,
        reviewer="human",
        source="contact sheets",
        timestamp="2026-01-01T00:00:00+00:00",
    )
    assert revision != artifact
    assert inspect_snapshot(artifact)[1].manual_decision.eq("").all()
    meta, updated, history = inspect_snapshot(revision)
    assert updated.manual_decision.tolist() == ["supported", "ambiguous", ""]
    assert len(history) == 1 and len(history[0]["changes"]) == 2
    assert (
        revision / "imports" / f"{history[0]['event_id']}.csv"
    ).read_bytes() == imported.read_bytes()
    assert verify_review(revision, paths)["quality_valid"]
    assert meta["confirmed_matches_created"] is False
    assert meta["split_created"] is False
    assert meta["ground_truth"] is False
    assert summarize_review(revision)["overall"]["unresolved_count"] == 2
    assert original_checks == {p.name: file_sha256(p) for p in paths.linkage.iterdir()}
    assert (
        record_review(
            artifact,
            imported,
            reviewer="human",
            source="contact sheets",
            timestamp="2026-01-01T00:00:00+00:00",
        )
        == revision
    )


@pytest.mark.parametrize(
    "fault",
    [
        "auto_decision",
        "lost_occurrence",
        "modified_notes",
        "confirmation_flag",
        "contact_sheet",
    ],
)
def test_review_verification_fails_on_tampering(review_publication, fault):
    _, paths, artifact = review_publication
    meta = read_json(artifact / "metadata.json")
    if fault in ("auto_decision", "modified_notes"):
        target = artifact / (
            "initial_review.csv" if fault == "auto_decision" else "review.csv"
        )
        table = pd.read_csv(target, dtype=str, keep_default_na=False)
        table.loc[
            0, "manual_decision" if fault == "auto_decision" else "manual_notes"
        ] = "supported" if fault == "auto_decision" else "unaudited annotation"
        table.to_csv(target, index=False)
    elif fault == "lost_occurrence":
        target = artifact / "candidate_occurrences.parquet"
        table = pd.read_parquet(target).drop_duplicates("video_content_id")
        table.to_parquet(target, index=False)
    elif fault == "confirmation_flag":
        meta["confirmed_matches_created"] = True
        target = None
    else:
        name = next(name for name in meta["output_checksums"] if name.endswith(".png"))
        target = artifact / name
        target.write_bytes(b"changed image")
    if target:
        meta["output_checksums"][target.relative_to(artifact).as_posix()] = file_sha256(
            target
        )
    write_json(artifact / "metadata.json", meta)
    assert not verify_review(artifact, paths)["quality_valid"]


@pytest.mark.parametrize(
    "fault",
    [
        "unconfirmed",
        "wrong_sequence_set",
        "unknown_group",
        "missing_sequence",
        "duplicate_sequence",
    ],
)
def test_invalid_external_membership_or_sample_refused(review_publication, fault):
    root, paths, artifact = review_publication
    meta = read_json(paths.visual_dependencies / "metadata.json")
    if fault == "unconfirmed":
        meta["manual_confirmation_complete"] = False
    elif fault == "wrong_sequence_set":
        meta["sequence_set_id"] = "foreign-sequences"
    elif fault == "unknown_group":
        sample = pd.read_csv(paths.calibration_sample)
        sample.loc[0, "proposed_visual_dependency_group_id"] = "unknown"
        sample.to_csv(paths.calibration_sample, index=False)
    else:
        table_path = paths.visual_dependencies / "membership.csv"
        table = pd.read_csv(table_path)
        table = (
            table.iloc[1:]
            if fault == "missing_sequence"
            else pd.concat([table, table.iloc[:1]], ignore_index=True)
        )
        table.to_csv(table_path, index=False)
        meta["output_checksums"]["membership.csv"] = file_sha256(table_path)
    write_json(paths.visual_dependencies / "metadata.json", meta)
    assert not verify_review(artifact, paths)["quality_valid"]
    with pytest.raises(ValueError):
        init_review(
            paths,
            ReviewConfig(),
            root / "invalid",
            labeled_images_root=root / "labeled-images",
            video_images_root=root / "video-images",
        )


def test_cli_review_roundtrip(review_publication):
    root, paths, artifact = review_publication
    runner = CliRunner()
    flags = [
        part
        for key in (
            "calibration_sample",
            "linkage",
            "labeled_manifest",
            "sequence_set",
            "visual_dependencies",
        )
        for part in ("--" + key.replace("_", "-"), str(getattr(paths, key)))
    ]
    result = runner.invoke(
        app,
        [
            "linkage",
            "review",
            "init",
            *flags,
            "--labeled-images-root",
            str(root / "labeled-images"),
            "--video-images-root",
            str(root / "video-images"),
            "--output",
            str(artifact.parent),
        ],
    )
    assert result.exit_code == 0, result.output


def test_cli_review_verification_and_record(review_publication):
    root, paths, artifact = review_publication
    runner = CliRunner()
    flags = [
        part
        for key in (
            "calibration_sample",
            "linkage",
            "labeled_manifest",
            "sequence_set",
            "visual_dependencies",
        )
        for part in ("--" + key.replace("_", "-"), str(getattr(paths, key)))
    ]
    result = runner.invoke(app, ["linkage", "review", "verify", str(artifact), *flags])
    assert result.exit_code == 0, result.output
    result = runner.invoke(app, ["linkage", "review", "summary", str(artifact)])
    assert result.exit_code == 0 and '"unresolved_count": 3' in result.output
    frame = pd.read_csv(artifact / "review.csv", dtype=str, keep_default_na=False)
    frame.loc[0, "manual_decision"] = "unsupported"
    path = root / "cli-decisions.csv"
    frame.to_csv(path, index=False)
    result = runner.invoke(
        app,
        [
            "linkage",
            "review",
            "record",
            str(artifact),
            "--decisions",
            str(path),
            "--reviewer",
            "human",
            "--source",
            "manual calibration",
        ],
    )
    assert result.exit_code == 0, result.output


def test_second_revision_preserves_prior_import_and_audits_clearing(review_publication):
    root, paths, artifact = review_publication
    _, initial, _ = inspect_snapshot(artifact)
    first = initial.copy()
    first.loc[0, "manual_decision"] = "supported"
    import_path = root / "first.csv"
    first.to_csv(import_path, index=False)
    revision = record_review(
        artifact,
        import_path,
        reviewer="first reviewer",
        source="manual sheets",
        timestamp="2026-01-01T00:00:00+00:00",
    )
    _, second, _ = inspect_snapshot(revision)
    second.loc[0, ["manual_decision", "manual_notes"]] = ["", "Pending adjudication"]
    second_path = root / "second.csv"
    second.to_csv(second_path, index=False)
    final = record_review(
        revision,
        second_path,
        reviewer="second reviewer",
        source="manual recheck",
        timestamp="2026-01-02T00:00:00+00:00",
    )
    _, final_rows, history = inspect_snapshot(final)
    assert len(history) == 2
    assert history[1]["changes"][0]["before"]["manual_decision"] == "supported"
    assert history[1]["changes"][0]["after"]["manual_decision"] == ""
    assert len(list((final / "imports").glob("*.csv"))) == 2
    assert final_rows.loc[0, "reviewer"] == "second reviewer"
    assert verify_review(final, paths)["quality_valid"]


def test_image_hash_failure_preserves_sources_and_refuses_completion(
    review_publication,
):
    root, paths, _ = review_publication
    labeled = pd.read_parquet(paths.labeled_manifest).sort_values("frame_id")
    (root / "labeled-images" / labeled.source_member_path.iloc[0]).write_bytes(
        b"wrong pixels"
    )
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        init_review(
            paths,
            ReviewConfig(),
            root / "invalid-images",
            labeled_images_root=root / "labeled-images",
            video_images_root=root / "video-images",
        )
    assert not list((root / "invalid-images").rglob("metadata.json"))


def test_record_cannot_publish_inside_linkage_source(review_publication):
    root, paths, artifact = review_publication
    _, frame, _ = inspect_snapshot(artifact)
    frame.loc[0, "manual_decision"] = "supported"
    imported = root / "decisions.csv"
    frame.to_csv(imported, index=False)
    with pytest.raises(ValueError, match="existing publication"):
        record_review(
            artifact, imported, reviewer="human", source="manual", output=paths.linkage
        )


def test_context_crosses_sequence_boundaries_but_never_source_videos():
    from flir_pipeline.linkage.review_media import context_plan

    rows = []
    for video in ("a", "b"):
        for t in range(7):
            rows.append(
                {
                    "frame_id": f"{video}-{t}",
                    "content_id": f"c-{t}",
                    "video_id": video,
                    "timestamp_seconds": float(t),
                    "sample_index": t,
                    "sequence_id": f"seq-{video}-{t // 3}",
                    "visual_dependency_group_id": f"g-{t // 3}",
                    "image_path": f"{video}/{t}.png",
                    "image_sha256": f"c-{t}",
                }
            )
    sequences = pd.DataFrame(rows)
    review = pd.DataFrame(
        [
            {
                "labeled_content_id": "q",
                "proposed_visual_dependency_group_id": "g-1",
                "review_stratum": "ambiguous-candidates",
                "review_query_id": "qid",
            }
        ]
    )
    candidates = pd.DataFrame([{"labeled_content_id": "q", "video_content_id": "c-3"}])
    centers = sequences[sequences.frame_id.eq("a-3")].rename(
        columns={"frame_id": "video_frame_id", "content_id": "video_content_id"}
    )
    context = context_plan(review, candidates, centers, sequences, ReviewConfig())
    assert context.sample_index.tolist() == list(range(7))
    assert context.video_id.eq("a").all()
    assert context.sequence_id.nunique() == 3
    assert context.visual_dependency_group_id.nunique() == 3


@pytest.fixture(scope="module")
def confirmed_v1_template(review_template, tmp_path_factory):
    original, original_paths, _ = review_template
    root = tmp_path_factory.mktemp("confirmed-visual-v1") / "sources"
    shutil.copytree(original, root)
    paths = replace(
        original_paths,
        **{
            key: root / getattr(original_paths, key).relative_to(original)
            for key in (
                "calibration_sample",
                "linkage",
                "labeled_manifest",
                "sequence_set",
                "visual_dependencies",
            )
        },
    )
    directory = root / "confirmed-visual-dependency-validation-v1"
    directory.mkdir()
    instances = pd.read_parquet(paths.sequence_set / "sequence_instances.parquet")
    membership = instances[
        [
            "sequence_id",
            "video_id",
            "source_video",
            "start_sample_index",
            "end_sample_index",
            "exact_duplicate_dependency_group_id",
        ]
    ].merge(
        pd.read_csv(paths.visual_dependencies / "membership.csv"),
        on="sequence_id",
        validate="one_to_one",
    )
    membership["visual_dependency_status"] = "synthetic_confirmed"
    membership["decision_basis"] = "Synthetic manually confirmed evidence"
    membership["ground_truth"] = False
    membership.to_csv(directory / "visual_dependency_membership.csv", index=False)
    membership.groupby("visual_dependency_group_id").size().rename(
        "sequence_count"
    ).reset_index().to_csv(directory / "visual_dependency_groups.csv", index=False)
    # Exact producer metadata schema, with synthetic identities. No producer ID
    # or output checksums exist, and the consumer must not add them externally.
    write_json(
        directory / "metadata.json",
        {
            "artifact_kind": "confirmed_manual_visual_dependency_validation",
            "artifact_version": 1,
            "ground_truth": False,
            "sequence_set_id": read_json(paths.sequence_set / "metadata.json")[
                "artifact_id"
            ],
            "manual_confirmation_complete": True,
            "review_status": "confirmed_manual_review",
            "semantic_role": "must_link_constraint_for_leakage_safe_split",
            "sequence_instances_merged": False,
            "exact_duplicate_dependencies_preserved": True,
            "split_created": False,
        },
    )
    paths = replace(
        paths,
        visual_dependencies=directory,
        membership_table="visual_dependency_membership.csv",
    )
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    artifact = init_review(
        paths,
        ReviewConfig(),
        root / "v1-reviews",
        labeled_images_root=root / "labeled-images",
        video_images_root=root / "video-images",
    )
    assert before == {p.name: p.read_bytes() for p in directory.iterdir()}
    return root, paths, artifact


@pytest.fixture
def confirmed_v1_publication(confirmed_v1_template, tmp_path):
    root, paths, artifact = confirmed_v1_template
    directory = tmp_path / "confirmed-source"
    shutil.copytree(paths.visual_dependencies, directory)
    # Mutate only a private copy of the three external source files. All other
    # sources and the existing consumer publication remain read-only.
    return root, replace(paths, visual_dependencies=directory), artifact


def test_confirmed_v1_consumer_identity_and_verify_without_rerender(
    confirmed_v1_template, monkeypatch
):
    from flir_pipeline.linkage.review_model import canonical_json

    _, paths, artifact = confirmed_v1_template
    original = {p.name: p.read_bytes() for p in paths.visual_dependencies.iterdir()}
    producer = read_json(paths.visual_dependencies / "metadata.json")
    assert "artifact_id" not in producer and "output_checksums" not in producer
    meta, rows, _ = inspect_snapshot(artifact)
    source = meta["identity"]["calibration"]["sources"]
    assert "visual_dependency_artifact_id" not in source
    identity = source["visual_dependency_consumer_source_identity"]
    assert identity["artifact_kind"] == producer["artifact_kind"]
    assert identity["artifact_version"] == 1
    assert identity["sequence_set_id"] == producer["sequence_set_id"]
    expected = {
        name: file_sha256(paths.visual_dependencies / name)
        for name in (
            "metadata.json",
            "visual_dependency_groups.csv",
            "visual_dependency_membership.csv",
        )
    }
    assert source["visual_dependency_files"] == identity["source_checksums"] == expected
    assert (
        source["visual_dependency_consumer_source_fingerprint"]
        == hashlib.sha256(canonical_json(identity).encode("utf-8")).hexdigest()
    )
    snapshot = pd.read_parquet(artifact / "visual_dependency_membership.parquet")
    assert {
        "video_id",
        "source_video",
        "start_sample_index",
        "end_sample_index",
        "exact_duplicate_dependency_group_id",
        "visual_dependency_status",
        "decision_basis",
        "ground_truth",
    } <= set(snapshot)
    assert rows.manual_decision.eq("").all()

    def forbidden(*args, **kwargs):
        pytest.fail("verify must not rerender or reopen original images")

    monkeypatch.setattr("flir_pipeline.linkage.review_storage.render_sheets", forbidden)
    monkeypatch.setattr(
        "flir_pipeline.features.image_source.ImageSource.decode", forbidden
    )
    result = verify_review(artifact, paths)
    assert result["quality_valid"], result
    assert not result["confirmed_matches_created"] and not result["split_created"]
    assert original == {
        p.name: p.read_bytes() for p in paths.visual_dependencies.iterdir()
    }


@pytest.mark.parametrize(
    "filename",
    [
        "metadata.json",
        "visual_dependency_groups.csv",
        "visual_dependency_membership.csv",
    ],
)
def test_confirmed_v1_verify_rehashes_every_external_file(
    confirmed_v1_publication, filename
):
    _, paths, artifact = confirmed_v1_publication
    path = paths.visual_dependencies / filename
    # A semantic no-op still changes the exact frozen external bytes.
    path.write_bytes(path.read_bytes() + b"\n")
    result = verify_review(artifact, paths)
    assert not result["quality_valid"]
    assert "bound source artifacts" in result["error"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("artifact_version", 2),
        ("artifact_version", True),
        ("artifact_version", "1"),
        ("ground_truth", True),
        ("ground_truth", 0),
        ("review_status", "provisional"),
        ("manual_confirmation_complete", False),
        ("manual_confirmation_complete", 1),
        ("sequence_set_id", "foreign-sequences"),
        ("semantic_role", "cluster"),
        ("sequence_instances_merged", True),
        ("sequence_instances_merged", 0),
        ("split_created", True),
        ("exact_duplicate_dependencies_preserved", False),
    ],
)
def test_confirmed_v1_refuses_invalid_metadata(confirmed_v1_template, field, value):
    from flir_pipeline.linkage.review_sources import visual_dependency_binding

    _, paths, _ = confirmed_v1_template
    metadata = read_json(paths.visual_dependencies / "metadata.json")
    sequence_id = metadata["sequence_set_id"]
    metadata[field] = value
    with pytest.raises(ValueError):
        visual_dependency_binding(paths, metadata, sequence_id)


@pytest.mark.parametrize(
    "field",
    [
        "artifact_version",
        "ground_truth",
        "review_status",
        "manual_confirmation_complete",
        "sequence_set_id",
        "semantic_role",
        "sequence_instances_merged",
        "split_created",
        "exact_duplicate_dependencies_preserved",
    ],
)
def test_confirmed_v1_requires_each_semantic_declaration(confirmed_v1_template, field):
    from flir_pipeline.linkage.review_sources import visual_dependency_binding

    _, paths, _ = confirmed_v1_template
    metadata = read_json(paths.visual_dependencies / "metadata.json")
    sequence_id = metadata["sequence_set_id"]
    metadata.pop(field)
    with pytest.raises(ValueError):
        visual_dependency_binding(paths, metadata, sequence_id)


@pytest.mark.parametrize(
    "fault", ["missing_sequence", "duplicate_sequence", "extra_sequence"]
)
def test_confirmed_v1_requires_exact_membership_coverage(
    confirmed_v1_publication, fault
):
    from flir_pipeline.linkage.review_sources import load_review_sources

    _, paths, _ = confirmed_v1_publication
    path = paths.visual_dependencies / paths.membership_table
    membership = pd.read_csv(path)
    if fault == "missing_sequence":
        membership = membership.iloc[1:]
    else:
        membership = pd.concat([membership, membership.iloc[:1]], ignore_index=True)
        if fault == "extra_sequence":
            membership.loc[len(membership) - 1, "sequence_id"] = "unknown-sequence"
    membership.to_csv(path, index=False)
    with pytest.raises(ValueError, match="cover each sequence exactly once"):
        load_review_sources(paths)


@pytest.mark.parametrize("field", ["artifact_id", "output_checksums"])
def test_normalized_adapter_still_requires_producer_identity_and_checksums(
    review_template, field
):
    from flir_pipeline.linkage.review_sources import visual_dependency_binding

    _, paths, _ = review_template
    metadata = read_json(paths.visual_dependencies / "metadata.json")
    metadata.pop(field)
    with pytest.raises(ValueError):
        visual_dependency_binding(paths, metadata, metadata["sequence_set_id"])


def test_confirmed_v1_explicit_membership_selection_and_deterministic_binding(
    confirmed_v1_publication,
):
    from flir_pipeline.linkage.review_sources import visual_dependency_binding

    _, paths, _ = confirmed_v1_publication
    metadata = read_json(paths.visual_dependencies / "metadata.json")
    first = visual_dependency_binding(paths, metadata, metadata["sequence_set_id"])
    assert first == visual_dependency_binding(
        paths, metadata, metadata["sequence_set_id"]
    )
    selected = paths.visual_dependencies / "selected_membership.csv"
    selected.write_bytes(
        (paths.visual_dependencies / paths.membership_table).read_bytes()
    )
    alternate = replace(paths, membership_table=selected.name)
    result = visual_dependency_binding(alternate, metadata, metadata["sequence_set_id"])
    assert set(result["visual_dependency_files"]) == {
        "metadata.json",
        "visual_dependency_groups.csv",
        selected.name,
    }
    assert (
        result["visual_dependency_consumer_source_fingerprint"]
        != first["visual_dependency_consumer_source_fingerprint"]
    )
    with pytest.raises(ValueError):
        visual_dependency_binding(
            replace(paths, membership_table="missing.csv"),
            metadata,
            metadata["sequence_set_id"],
        )
    with pytest.raises(ValueError):
        visual_dependency_binding(
            replace(paths, membership_table="visual_dependency_groups.csv"),
            metadata,
            metadata["sequence_set_id"],
        )
