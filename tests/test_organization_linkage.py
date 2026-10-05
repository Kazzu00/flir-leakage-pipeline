"""Exercise the existing linkage/review storage contracts through export adapters."""

import json
import shutil
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import pytest
import test_linkage_review
from jsonschema import Draft202012Validator
from test_explorer import publish
from test_organization_export import freeze, payload, snapshot

from flir_pipeline.data.identity import dataset_id_from_manifest
from flir_pipeline.explorer.organization import (
    content_records,
    export_organization,
    occurrence_records,
    preview_media,
    scientific_tables,
    timelines,
)
from flir_pipeline.explorer.organization_contract import OrganizationExport, schema
from flir_pipeline.explorer.organization_sources import (
    CandidateEvidence,
    Sources,
    load_candidates,
)
from flir_pipeline.linkage.review_sources import (
    load_candidate_context,
    load_linkage_context,
)
from flir_pipeline.similarity.storage import file_sha256, read_json, write_json
from flir_pipeline.splitting.base import SplitConfig, identity_payload, split_space_id


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
    assert set(shared[4]) == {
        "labeled_manifest_sha256",
        "linkage_files",
        "linkage_id",
        "sequence_files",
        "sequence_set_id",
    }


@pytest.fixture
def linkage_export_args(review_template, tmp_path):
    """Real storage readers with a synthetic historical-only frozen plan."""
    _, paths, _ = review_template
    manifest_path = tmp_path / "current.parquet"
    shutil.copy2(paths.labeled_manifest, manifest_path)
    manifest = pd.read_parquet(manifest_path)
    dataset = dataset_id_from_manifest(manifest)
    config = SplitConfig(strategy="historical", seeds=(0,))
    identity = identity_payload(dataset, config, np.array([0.5, 0.25, 0.25]), 0, None)
    sid = split_space_id(identity)
    records = manifest[["frame_id", "content_id", "original_split"]].copy()
    records["new_split"] = records.original_split
    groups = pd.DataFrame({"content_id": sorted(manifest.content_id.unique())})
    groups["cluster_id"] = None
    groups["group_id"] = groups.content_id
    groups["group_type"] = "content_singleton"
    contents = groups.copy()
    memberships = records.groupby("content_id").new_split.agg(lambda v: sorted(set(v)))
    contents["new_split"] = contents.content_id.map(
        memberships.map(lambda v: v[0] if len(v) == 1 else None)
    )
    contents["split_membership_set"] = contents.content_id.map(
        memberships.map(json.dumps)
    )
    split = tmp_path / "splits" / sid
    publish(
        split,
        {
            "artifact_kind": "split_run",
            "split_space_id": sid,
            "identity_payload": identity,
        },
        {
            "record_split_assignments.parquet": records,
            "source_groups.parquet": groups,
            "split_assignments.parquet": contents,
            "quality.json": {"quality_valid": True},
        },
    )
    plan = tmp_path / "plan"
    plan.mkdir()
    freeze(
        plan,
        [
            dict(
                strategy="historical",
                split_seed=0,
                split_space_id=sid,
                dataset_id=dataset,
                clustering_space_id=None,
                split_metadata_sha256=file_sha256(split / "metadata.json"),
                record_counts={
                    p: int(records.new_split.eq(p).sum())
                    for p in ("train", "val", "test")
                },
            )
        ],
    )
    return dict(
        manifest_path=manifest_path,
        plan_directory=plan,
        split_root=split.parent,
        clustering_root=tmp_path / "absent",
        linkage_root=paths.linkage,
        sequence_root=paths.sequence_set,
        output=tmp_path / "export",
        media_output=tmp_path / "media",
    ), paths


@pytest.mark.parametrize("reserialized", [False, True])
def test_organization_manifest_serialization_provenance_and_readonly_export(
    linkage_export_args,
    reserialized,
):
    args, paths = linkage_export_args
    manifest = pd.read_parquet(args["manifest_path"])
    if reserialized:
        manifest.iloc[::-1].to_parquet(
            args["manifest_path"], index=False, compression="gzip"
        )
    historical = file_sha256(paths.labeled_manifest)
    current = file_sha256(args["manifest_path"])
    assert (historical != current) is reserialized
    before = {
        key: snapshot(args[key])
        for key in ("plan_directory", "split_root", "linkage_root", "sequence_root")
    }
    manifest_before = (current, args["manifest_path"].stat().st_mtime_ns)
    export_organization(**args)
    data = payload(args)
    linkage_id = read_json(paths.linkage / "metadata.json")["artifact_id"]
    receipt = next(
        s
        for s in data["manifest"]["evidence_sources"]
        if s["artifact_id"] == linkage_id
    )
    assert receipt["labeled_manifest_binding"] == {
        "historical_manifest_sha256": historical,
        "current_manifest_sha256": current,
        "exact_tabular_identity_verified": True,
        "source_manifest_reserialized": reserialized,
    }
    assert (
        any("different serialized" in s for s in data["manifest"]["limitations"])
        is reserialized
    )
    assert receipt["output_checksums"]["labeled_occurrences.parquet"] == file_sha256(
        paths.linkage / "labeled_occurrences.parquet"
    )
    Draft202012Validator(schema()).validate(data)
    OrganizationExport.model_validate(data)
    scientific = {
        p.name: p.read_bytes()
        for p in args["output"].glob("*.json")
        if p.name != "manifest.json"
    }
    export_organization(**args)
    assert scientific == {
        p.name: p.read_bytes()
        for p in args["output"].glob("*.json")
        if p.name != "manifest.json"
    }
    assert (
        data["manifest"]["evidence_sources"]
        == payload(args)["manifest"]["evidence_sources"]
    )
    assert before == {key: snapshot(args[key]) for key in before}
    assert manifest_before == (
        file_sha256(args["manifest_path"]),
        args["manifest_path"].stat().st_mtime_ns,
    )


def test_reserialization_remains_opt_in_for_candidate_and_review_readers(
    review_template, tmp_path
):
    _, paths, _ = review_template
    current = tmp_path / "current.parquet"
    pd.read_parquet(paths.labeled_manifest).to_parquet(
        current, index=False, compression="gzip"
    )
    assert file_sha256(current) != file_sha256(paths.labeled_manifest)
    with pytest.raises(ValueError, match="Labeled manifest differs"):
        load_candidate_context(paths.linkage, current, paths.sequence_set)
    with pytest.raises(ValueError, match="Labeled manifest differs"):
        load_linkage_context(replace(paths, labeled_manifest=current))


@pytest.mark.parametrize(
    "change",
    [
        "value",
        "dtype",
        "missing_row",
        "extra_row",
        "missing_column",
        "extra_column",
        "column_order",
        "content_id",
        "label",
    ],
)
def test_organization_reserialization_rejects_any_tabular_change(
    review_template, tmp_path, change
):
    _, paths, _ = review_template
    manifest = pd.read_parquet(paths.labeled_manifest)
    if change == "value":
        manifest.loc[0, "class_counts"] = "{}"
    elif change == "dtype":
        manifest["label_sha256"] = manifest.label_sha256.astype("string")
    elif change == "missing_row":
        manifest = manifest.iloc[1:]
    elif change == "extra_row":
        manifest = pd.concat([manifest, manifest.iloc[:1]], ignore_index=True)
    elif change == "missing_column":
        manifest = manifest.drop(columns="class_counts")
    elif change == "extra_column":
        manifest["unexpected"] = 1
    elif change == "column_order":
        manifest = manifest[manifest.columns[::-1]]
    else:
        manifest.loc[0, "content_id" if change == "content_id" else "label_sha256"] = (
            "changed"
        )
    current = tmp_path / "current.parquet"
    manifest.to_parquet(current, index=False, compression="gzip")
    assert file_sha256(current) != file_sha256(paths.labeled_manifest)
    with pytest.raises(AssertionError):
        load_candidates(
            [paths.linkage],
            current,
            manifest,
            Sources(),
            sequence_root=paths.sequence_set,
        )


@pytest.mark.parametrize(
    "failure",
    [
        "dataset_id",
        "snapshot",
        "metadata",
        "other_output",
        "sequence",
        "snapshot_order",
    ],
)
def test_reserialization_keeps_publication_dataset_and_sequence_checks(
    review_template, tmp_path, failure
):
    _, paths, _ = review_template
    linkage = tmp_path / "linkage"
    shutil.copytree(paths.linkage, linkage)
    sequence = tmp_path / "sequence"
    shutil.copytree(paths.sequence_set, sequence)
    manifest = pd.read_parquet(paths.labeled_manifest)
    current = tmp_path / "current.parquet"
    manifest.to_parquet(current, index=False, compression="gzip")
    meta_path = linkage / "metadata.json"
    meta = read_json(meta_path)
    if failure == "dataset_id":
        # Keep the tables exactly equal, exercising the independent dataset check.
        meta["labeled_dataset_id"] = "different-dataset"
        write_json(meta_path, meta)
    elif failure == "metadata":
        meta["artifact_id"] = "wrong-identity"
        write_json(meta_path, meta)
    elif failure in {"snapshot", "snapshot_order"}:
        target = linkage / "labeled_occurrences.parquet"
        stored = pd.read_parquet(target)
        stored.iloc[::-1].to_parquet(target, index=False, compression="gzip")
        if failure == "snapshot_order":
            # Even checksum-bound row order must retain the existing canonical contract.
            meta["output_checksums"][target.name] = file_sha256(target)
            write_json(meta_path, meta)
    else:
        target = (
            linkage / "summary.json"
            if failure == "other_output"
            else sequence / "occurrence_assignments.parquet"
        )
        with target.open("ab") as stream:
            stream.write(b"changed")
    with pytest.raises((ValueError, AssertionError)):
        load_candidates([linkage], current, manifest, Sources(), sequence_root=sequence)


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_manifest_reserialized", True),
        ("exact_tabular_identity_verified", False),
        ("current_manifest_sha256", "0" * 64),
    ],
)
def test_organization_contract_rejects_incorrect_manifest_binding(
    linkage_export_args, field, value
):
    args, _ = linkage_export_args
    export_organization(**args)
    data = payload(args)
    receipt = next(
        s
        for s in data["manifest"]["evidence_sources"]
        if s["labeled_manifest_binding"] is not None
    )
    receipt["labeled_manifest_binding"][field] = value
    if field == "current_manifest_sha256":
        receipt["labeled_manifest_binding"]["source_manifest_reserialized"] = True
    with pytest.raises(ValueError):
        OrganizationExport.model_validate(data)


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
