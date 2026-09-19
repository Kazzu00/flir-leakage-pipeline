"""Synthetic offline invariants for human inspection, not Streamlit or real FLIR."""

import hashlib
import io
import json
import zipfile
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from flir_pipeline.data.identity import dataset_id_from_manifest
from flir_pipeline.explorer.data import (
    cluster_members,
    compare_partitions,
    filter_split,
    load_cluster,
    load_split,
    with_split,
)
from flir_pipeline.explorer.discovery import (
    candidate_labels,
    compatible_splits,
    discover_runs,
    sha256,
)
from flir_pipeline.explorer.frames import FrameReader
from flir_pipeline.explorer.preview import cache_key, render_preview
from flir_pipeline.explorer.timeline import index_gaps, ordered_contents, preview_plans
from flir_pipeline.similarity.temporal_display import GAP_BINS, temporal_display_table


def publish(directory, metadata, tables, arrays=None):
    directory.mkdir(parents=True)
    for name, value in tables.items():
        if name.endswith(".parquet"):
            value.to_parquet(directory / name, index=False)
        else:
            (directory / name).write_text(json.dumps(value), encoding="utf-8")
    for name, value in (arrays or {}).items():
        np.save(directory / name, value, allow_pickle=False)
    metadata["output_sha256"] = {p.name: sha256(p) for p in directory.iterdir()}
    (directory / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


@pytest.fixture
def experiment(tmp_path):
    rows, contents = [], []
    source = tmp_path / "sources"
    source.mkdir()
    with zipfile.ZipFile(source / "synthetic.zip", "w") as archive:
        for i, (sequence, index) in enumerate([("A", 1), ("A", 3), ("A", 40), ("B", 2), ("B", 4), ("C", 1)]):
            output = io.BytesIO()
            Image.new("RGB", (24, 16), (i * 30, 80, 150)).save(output, format="PNG")
            data = output.getvalue()
            content_id = hashlib.sha256(data).hexdigest()
            member = f"frames/{i}.png"
            archive.writestr(member, data)
            row = {"frame_id": f"frame-{i}", "content_id": content_id,
                   "image_sha256": content_id, "label_sha256": f"label-{i}",
                   "source_archive": "synthetic.zip", "source_member_path": member,
                   "possible_sequence": sequence, "possible_frame_index": index,
                   "original_split": "train", "temporal_inference_confidence": "filename"}
            rows.append(row)
            contents.append({k: row[k] for k in ["content_id", "image_sha256", "source_archive", "source_member_path"]} | {"representative_frame_id": row["frame_id"], "embedding_row": i})
        rows.append({**rows[0], "frame_id": "duplicate", "original_split": "test"})
        rows.append({**rows[5], "frame_id": "ambiguous", "possible_frame_index": 2, "original_split": "val"})
    manifest = pd.DataFrame(rows)
    index = pd.DataFrame(contents)
    labels = np.array([0, 0, 0, 0, -1, 1])
    dataset = dataset_id_from_manifest(manifest)
    clusters = tmp_path / "clusters"
    publish(clusters / "run", {"artifact_kind": "clustering_run", "clustering_space_id": "synthetic-clustering", "dataset_id": dataset,
                              "extractor": "synthetic", "representation": "original_l2", "algorithm": "dbscan", "N": 6},
            {"content_index.parquet": index,
             "cluster_summary.parquet": pd.DataFrame([{"cluster_id": 0, "n_members": 4, "medoid_content_id": index.content_id[1]},
                                                       {"cluster_id": 1, "n_members": 1, "medoid_content_id": index.content_id[5]}]),
             "metrics.json": {"n_clusters_excluding_noise": 2, "noise_fraction": 1 / 6}, "quality.json": {"quality_valid": True}},
            {"cluster_labels.npy": labels})
    split_root = tmp_path / "splits"
    for strategy in ("historical", "random_content", "cluster_aware"):
        groups = index[["content_id"]].copy()
        groups["cluster_id"] = labels if strategy == "cluster_aware" else None
        groups["group_id"] = ["cluster-0"] * 4 + ["noise-4", "cluster-1"] if strategy == "cluster_aware" else [f"content-{i}" for i in range(6)]
        groups["group_type"] = ["cluster"] * 4 + ["noise_singleton", "cluster"] if strategy == "cluster_aware" else "content_singleton"
        records = manifest[["frame_id", "content_id", "original_split"]].copy()
        mapping = dict(zip(index.content_id, ["train", "train", "train", "train", "test", "val"], strict=True))
        records["new_split"] = records.original_split if strategy == "historical" else records.content_id.map(mapping)
        publish(split_root / strategy, {"artifact_kind": "split_run", "split_space_id": strategy,
                                       "identity_payload": {"dataset_id": dataset, "clustering_space_id": "synthetic-clustering" if strategy == "cluster_aware" else None,
                                                            "configuration": {"strategy": strategy}, "seed": 0}},
                {"record_split_assignments.parquet": records, "source_groups.parquet": groups,
                 "split_assignments.parquet": groups, "quality.json": {"quality_valid": True}})
    return tmp_path, source, manifest, discover_runs(clusters)[0][0], discover_runs(split_root)[0]


def test_discovery_requires_real_complete_unambiguous_artifacts(experiment):
    root, _, _, cluster, splits = experiment
    assert len(splits) == 3
    partial = root / "clusters" / "paused.partial"
    partial.mkdir()
    (partial / "metadata.json").write_text("{}")
    assert discover_runs(root / "clusters")[0] == [cluster]
    (cluster.directory / "cluster_labels.npy").unlink()
    runs, issues = discover_runs(root / "clusters")
    assert not runs and len(issues) == 1
    assert partial.exists()


def test_aliases_are_optional_not_run_logic(experiment):
    root, _, _, cluster, _ = experiment
    path = root / "aliases.csv"
    pd.DataFrame([{"clustering_space_id": cluster.space_id, "candidate_label": "Synthetic candidate"}]).to_csv(path, index=False)
    aliases = candidate_labels(path)
    assert discover_runs(root / "clusters", aliases)[0][0].candidate == "Synthetic candidate"
    assert candidate_labels(root / "absent.csv") == {}


def test_membership_mapping_multiple_sequences_and_historical_occurrences(experiment):
    _, _, manifest, run, splits = experiment
    cluster = load_cluster(run, manifest.sample(frac=1, random_state=7))
    assert len(cluster.contents) == 6 and len(cluster.records) == 8
    members = cluster_members(cluster.contents, 0)
    assert len(members) == 4 and members.occurrence_count.sum() == 5
    assert members.sequence_id.nunique() == 2
    historical = load_split(next(r for r in splits if r.strategy == "historical"), manifest)
    contents, records = with_split(cluster, historical)
    assert contents.iloc[0].new_splits == ("train", "test")
    assert len(filter_split(contents, "test")) == 1
    assert len(records) == 8


def test_order_gaps_preview_separation_and_unknown_provenance(experiment):
    _, _, manifest, run, _ = experiment
    contents = load_cluster(run, manifest).contents
    members = cluster_members(contents, 0).sample(frac=1, random_state=4)
    assert ordered_contents(members).frame_index.tolist() == [1, 3, 40, 2]
    gaps = index_gaps(members)
    assert gaps.index_gap.dropna().tolist() == [2, 37]
    plans = preview_plans(members, run.space_id, 0, gap_threshold=25)
    assert [p.indices for p in plans] == [(1, 3), (40,), (2,)]
    assert len({p.sequence_key for p in plans}) == 2
    assert len(preview_plans(members, run.space_id, 0, gap_threshold=0)) == 2
    assert len(preview_plans(members, run.space_id, 0, max_frames=1)) == 4
    ambiguous = cluster_members(contents, 1)
    assert not ambiguous.frame_index_valid.any()
    assert not preview_plans(ambiguous, run.space_id, 1)


def test_noise_singletons_and_split_binding(experiment):
    _, _, manifest, run, runs = experiment
    cluster = load_cluster(run, manifest)
    aware = load_split(next(r for r in runs if r.strategy == "cluster_aware"), manifest)
    contents, _ = with_split(cluster, aware)
    noise = cluster_members(contents, -1)
    assert len(noise) == 1 and noise.iloc[0].new_splits == ("test",)
    assert noise.iloc[0].group_type == "noise_singleton"
    assert len(compatible_splits(run, runs)) == 3
    alien = replace(run, space_id="another-clustering")
    assert len(compatible_splits(alien, runs)) == 2
    cluster.run = alien
    with pytest.raises(ValueError, match="different source clustering"):
        with_split(cluster, aware)


def test_comparison_keeps_all_occurrences(experiment):
    _, _, manifest, run, runs = experiment
    cluster = load_cluster(run, manifest)
    left = load_split(next(r for r in runs if r.strategy == "historical"), manifest)
    right = load_split(next(r for r in runs if r.strategy == "cluster_aware"), manifest)
    rows = compare_partitions(cluster, left, right)
    assert len(rows) == len(manifest)
    duplicate = rows.loc[rows.frame_id.eq("duplicate")].iloc[0]
    assert duplicate.left == "test" and duplicate.right == "train" and duplicate.right_cluster == 0


def test_overlay_rejects_cluster_fracture_even_with_separate_group_ids(experiment):
    _, _, manifest, run, runs = experiment
    cluster = load_cluster(run, manifest)
    aware = load_split(next(r for r in runs if r.strategy == "cluster_aware"), manifest)
    aware.records.loc[aware.records.frame_id.eq("frame-1"), "new_split"] = "test"
    with pytest.raises(ValueError, match="fractures"):
        with_split(cluster, aware)


def test_noise_cannot_share_a_non_noise_group(experiment):
    _, _, manifest, run, runs = experiment
    cluster = load_cluster(run, manifest)
    aware = load_split(next(r for r in runs if r.strategy == "cluster_aware"), manifest)
    aware.groups.loc[aware.groups.cluster_id.eq(-1), "group_id"] = "cluster-0"
    with pytest.raises(ValueError, match="singleton"):
        with_split(cluster, aware)


def test_corrupt_artifacts_and_wrong_manifest_rejected(experiment):
    _, _, manifest, run, _ = experiment
    changed = manifest.copy()
    changed.loc[0, "label_sha256"] = "changed"
    with pytest.raises(ValueError, match="identities"):
        load_cluster(run, changed)
    (run.directory / "cluster_labels.npy").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        load_cluster(run, manifest)


def test_gif_is_read_only_and_cache_binds_order_fps_content(experiment):
    _, source, manifest, run, _ = experiment
    contents = load_cluster(run, manifest).contents
    plan = preview_plans(cluster_members(contents, 0), run.space_id, 0)[0]
    original_hash = sha256(source / "synthetic.zip")
    original_files = sorted(source.iterdir())
    with FrameReader(source, manifest) as reader:
        gif = render_preview(plan, reader)
    with Image.open(io.BytesIO(gif)) as image:
        assert image.format == "GIF" and image.n_frames == 2
        assert image.info["duration"] == 250
    assert sha256(source / "synthetic.zip") == original_hash
    assert sorted(source.iterdir()) == original_files
    assert cache_key(plan) == cache_key(replace(plan))
    for variation in [replace(plan, fps=8), replace(plan, frame_ids=plan.frame_ids[::-1]),
                      replace(plan, width=320), replace(plan, gap_threshold=50),
                      replace(plan, clustering_space_id="another"), replace(plan, content_ids=("changed",) * 2)]:
        assert cache_key(plan) != cache_key(variation)


def test_frame_reader_rejects_path_escape_and_byte_mismatch(experiment):
    _, source, manifest, _, _ = experiment
    broken = manifest.copy()
    broken.loc[0, "source_archive"] = "../synthetic.zip"
    with FrameReader(source, broken) as reader, pytest.raises(ValueError, match="FLIR_DATA_ROOT"):
        reader.image("frame-0")
    broken = manifest.copy()
    broken.loc[0, "image_sha256"] = "invalid"
    with FrameReader(source, broken) as reader, pytest.raises(ValueError, match="bytes differ"):
        reader.image("frame-0")


def test_temporal_display_preserves_recorded_values_and_empty_bins():
    summary = pd.DataFrame([{"extractor": "synthetic", "frame_delta_bin": b, "count": 5, "Q1": .2, "median": .4, "Q3": .6} for b in GAP_BINS])
    summary.loc[2, "count"] = 0
    displayed = temporal_display_table(summary, "synthetic")
    assert displayed.frame_delta_bin.tolist() == list(GAP_BINS)
    assert displayed.pair_count.tolist() == [5, 5, 0, 5, 5, 5, 5]
    assert pd.isna(displayed.iloc[2]["median"])
    assert displayed.iloc[1]["median"] == .4
    summary.loc[1, "Q1"] = .9
    with pytest.raises(ValueError, match="quartiles"):
        temporal_display_table(summary, "synthetic")
