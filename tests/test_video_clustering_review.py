"""Synthetic JPEGs, real source verification and occurrence-preserving reports."""

import hashlib
import io
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from PIL import Image
from test_video_similarity import video_config, video_features
from typer.testing import CliRunner

from flir_pipeline.cli import app
from flir_pipeline.clustering.algorithms import ClusteringResult
from flir_pipeline.clustering.base import ClusteringConfig
from flir_pipeline.clustering.experiments import (
    comparison_to_store,
    load_families,
    screening_to_store,
    verify_collection,
)
from flir_pipeline.clustering.metrics import evaluate_clustering
from flir_pipeline.clustering.storage import run_to_store
from flir_pipeline.clustering.video_review import (
    aligned_video_records,
    gallery_selections,
    generate_video_clustering_review,
    inspection_tables,
)
from flir_pipeline.data.identity import dataset_id_from_manifest
from flir_pipeline.features.image_source import ImageSource
from flir_pipeline.reduction.base import ReductionConfig
from flir_pipeline.reduction.benchmark import benchmark_to_store
from flir_pipeline.reduction.storage import load_inputs
from flir_pipeline.similarity.storage import (
    compute_to_store,
    file_sha256,
    read_json,
    write_json,
)


def _hashes(root):
    return {p.relative_to(root).as_posix(): file_sha256(p) for p in root.rglob("*") if p.is_file()}


class LocalLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.targets = []

    def handle_starttag(self, tag, attrs):
        self.targets.extend(value for key, value in attrs if (tag, key) in {("a", "href"), ("img", "src")})


@pytest.fixture(scope="module")
def sources(tmp_path_factory):
    """Real verified artifacts over fabricated embeddings/JPEGs, never FLIR data."""
    pytest.importorskip("pacmap")
    root = tmp_path_factory.mktemp("video-review")
    feature, manifest_path = video_features(root, n=48)
    # Duplicate occurrences share one content row; distinct rows need not have
    # the deliberately identical vectors used by the similarity-specific helper.
    vectors = np.random.default_rng(42).normal(size=(48, 8)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    for name in ("embeddings_raw.npy", "embeddings_l2.npy"):
        np.save(feature/name, vectors)
    manifest = pd.read_parquet(manifest_path)
    images = root/"samples"
    payloads, content_hashes = {}, {}
    for i, content in enumerate(sorted(manifest.content_id.unique())):
        buffer = io.BytesIO()
        pixels = np.random.default_rng(i).integers(0, 255, (24, 32, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(buffer, format="JPEG")
        payloads[content] = buffer.getvalue()
        content_hashes[content] = hashlib.sha256(buffer.getvalue()).hexdigest()
    for row in manifest.itertuples():
        target = images/row.image_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payloads[row.content_id])
    for filename in ("content_index.parquet", "record_index.parquet"):
        frame = pd.read_parquet(feature/filename)
        frame["content_id"] = frame.content_id.map(content_hashes)
        if "image_sha256" in frame:
            frame["image_sha256"] = frame.content_id
        frame.to_parquet(feature/filename, index=False)
    manifest["content_id"] = manifest.content_id.map(content_hashes)
    manifest["image_sha256"] = manifest.content_id
    manifest.to_parquet(manifest_path, index=False)
    meta = read_json(feature/"metadata.json")
    meta["dataset_id"] = dataset_id_from_manifest(manifest)
    write_json(feature/"metadata.json", meta)
    similarity = compute_to_store(feature, manifest_path, video_config(top_k=20), root/"similarity")
    inputs = load_inputs(feature, similarity, manifest_path)
    grid = [ReductionConfig(method, seed=seed, hyperparameters=params)
            for method, params in (("tsne", {"perplexity": 5, "max_iter": 300}), ("pacmap", {"num_iters": [10, 10, 20]}))
            for seed in (0, 1, 2)]
    benchmark = benchmark_to_store(inputs, grid, root/"reduction")
    spec = root/"inputs.yaml"
    spec.write_text(yaml.safe_dump({"manifest": str(manifest_path), "encoders": {"synthetic": {
        "feature_directory": str(feature), "similarity_directory": str(similarity), "reduction_benchmark": str(benchmark)}}}), encoding="utf-8")
    return root, spec, images, manifest_path


@pytest.fixture
def family(sources):
    return load_families(sources[1])["synthetic"]


def _publish(family, labels, output, monkeypatch):
    from flir_pipeline.clustering import storage

    # Controlled synthetic assignments exercise report edge cases. Scientific
    # metrics, medoids, checksums and source verification use production code.
    def fit(values, ids, config, parameters):
        return ClusteringResult(labels, None, {"effective_parameters": parameters, "fit_seconds": 0.}, {})

    monkeypatch.setattr(storage, "fit_clustering", fit)
    return run_to_store(family, family.spaces[("original_l2", None)],
                        ClusteringConfig("dbscan", {"min_samples": 3, "eps_quantile": .5}), output)


def test_occurrences_cross_video_noise_transitions_and_display_only(sources, family):
    manifest = pd.read_parquet(sources[3])
    records = aligned_video_records(manifest, family)
    labels = np.array([0]*10+[1]*10+[-1]*28, dtype=np.int32)
    _, summary = evaluate_clustering(labels, family.context, family.context.original_distances)
    tables = inspection_tables(labels, family.source.content_index, records, summary, family.context.neighbors)
    occurrences = tables["occurrences"]
    assert len(occurrences) == len(manifest) == 51
    assert occurrences.frame_id.is_unique
    assert occurrences.is_noise.sum() == 28
    first = family.context.content_ids[0]
    repeated = occurrences.loc[occurrences.content_id.eq(first)]
    assert repeated.sample_index.tolist() == [0, 9, 80]
    assert repeated.video_id.tolist() == ["v1", "v1", "v2"]
    assert repeated.cluster_id.eq(0).all()
    pd.testing.assert_frame_equal(occurrences[records.columns].sort_values("frame_id").reset_index(drop=True), records)
    assert tables["clusters"].occurrences.sum() == 51
    assert tables["clusters"].unique_contents.sum() == 48
    assert tables["source_video_membership"].occurrences.sum() == 51
    assert tables["source_video_membership"].unique_contents.sum() > 48
    assert tables["clusters"].loc[lambda x: x.is_noise, "mean_intra_cosine"].isna().all()
    for _, group in occurrences.groupby("video_id"):
        assert group.sample_index.is_monotonic_increasing
        assert group.timestamp_seconds.is_monotonic_increasing
        assert pd.isna(group.iloc[0].assignment_changed)
        assert group.iloc[0].transition_kind == "start"
    row = occurrences.loc[occurrences.frame_id.eq("f001")].iloc[0]
    assert row.sample_index_gap == 9 and row.missing_grid_positions == 8
    assert row.timestamp_gap_seconds == 4.5
    assert tables["transitions"].assignment_changed.all()
    assert "to_noise" in set(tables["transitions"].transition_kind)
    selected = gallery_selections(labels, summary, family, occurrences)
    assert selected.display_only.all()
    assert selected.loc[selected.cluster_id.eq(-1), "role"].eq("noise_sample_content_id_order").all()
    assert selected.loc[selected.role.eq("medoid"), "content_id"].tolist() == summary.medoid_content_id.tolist()
    # Changing display order never flattens the complete scientific provenance.
    pd.testing.assert_frame_equal(selected, gallery_selections(labels, summary, family, occurrences.iloc[::-1]))
    for table in tables.values():
        assert not {"sequence_id", "sequence_key", "scene_id", "split_id", "original_split", "new_split", "labels"} & set(table)
    assert occurrences.sequence_status.eq("unknown").all()


@pytest.mark.parametrize("pattern", ["mixed", "all_noise", "one_cluster"])
def test_run_review_is_verified_local_read_only_and_has_complete_artifacts(sources, family, tmp_path, monkeypatch, pattern):
    labels = {"mixed": [0]*10+[1]*10+[-1]*28, "all_noise": [-1]*48, "one_cluster": [0]*48}[pattern]
    run = _publish(family, np.asarray(labels, dtype=np.int32), tmp_path/"clustering", monkeypatch)
    before, scientific = _hashes(sources[0]), _hashes(run)
    report = tmp_path/"review"
    result = generate_video_clustering_review(run, sources[1], sources[2], report)
    assert result["source_verification"]["quality_valid"]
    assert result["semantics"]["sequence_identity"] == "unknown"
    assert result["semantics"]["clusters_are_sequences"] is False
    assert result["validated_image_occurrences"] == 51
    assert _hashes(sources[0]) == before and _hashes(run) == scientific
    assert (report/"index.html").exists()
    sub = report/"runs"/"run-0000"
    occurrences = pd.read_parquet(sub/"occurrences.parquet")
    assert len(occurrences) == 51
    selected = pd.read_parquet(sub/"display_selections.parquet")
    assert selected.display_only.all()
    for relative in selected.thumbnail:
        with Image.open(report/relative) as thumbnail:
            assert thumbnail.width <= 320 and thumbnail.height <= 240
    for page in report.rglob("*.html"):
        links = LocalLinks()
        links.feed(page.read_text(encoding="utf-8"))
        assert links.targets
        for target in links.targets:
            resolved = (page.parent/target).resolve()
            assert resolved.is_relative_to(report.resolve()) and resolved.is_file()
    if pattern == "all_noise":
        assert selected.cluster_id.eq(-1).all() and not selected.role.eq("medoid").any()
        assert len(pd.read_parquet(sub/"transitions.parquet")) == 0
    if pattern == "one_cluster":
        assert selected.role.eq("medoid").sum() == 1
        assert len(pd.read_parquet(sub/"transitions.parquet")) == 0
    html = (sub/"index.html").read_text(encoding="utf-8")
    assert "display-only" in html and "Secuencias desconocidas" in html
    assert "no una secuencia" in html and "no son límites de escena" in html
    assert "train" not in html and "validation" not in html
    assert len(list(sub.glob("video-*.svg"))) == 4
    assert len(list(sub.glob("video-*.csv"))) == 4
    assert all(file_sha256(report/name) == digest for name, digest in result["output_sha256"].items())
    with pytest.raises(ValueError, match="Existing report preserved"):
        generate_video_clustering_review(run, sources[1], sources[2], report)


def test_cli_and_comparison_cover_all_runs(sources, family, tmp_path, monkeypatch):
    _publish(family, np.array([0]*10+[1]*10+[-1]*28, dtype=np.int32), tmp_path/"stub", monkeypatch)
    configs = [ClusteringConfig("dbscan", {"min_samples": 3, "eps_quantile": q}) for q in (.5, .8)]
    root = tmp_path/"clustering"
    screening = screening_to_store({"synthetic": family}, configs, root)
    comparison = comparison_to_store(screening, {"synthetic": family}, root)
    before = _hashes(root)
    report = tmp_path/"review"
    result = CliRunner().invoke(app, ["clustering", "video-review", str(comparison), "--inputs", str(sources[1]),
                                    "--images-root", str(sources[2]), "--output", str(report)])
    assert result.exit_code == 0, result.output
    assert _hashes(root) == before
    meta = read_json(comparison/"metadata.json")
    reviewed = pd.read_csv(report/"runs.csv")
    assert set(reviewed.clustering_space_id) == {r["clustering_space_id"] for r in meta["runs"]}
    assert len(list((report/"runs").glob("*/occurrences.parquet"))) == len(meta["runs"])
    assert read_json(report/"report_metadata.json")["source_verification"]["comparison"]["quality_valid"]
    with pytest.raises(ValueError, match="separate from read-only"):
        generate_video_clustering_review(comparison, sources[1], sources[2], screening/"review")
    assert _hashes(root) == before
    # Treat None/NaN as unavailable at CSV verification, but never accept zero
    # as an available sequence metric, even with coherent output checksums.
    path = screening/"screening.csv"
    table = pd.read_csv(path)
    table["weighted_dominant_sequence_fraction"] = 0.
    table.to_csv(path, index=False)
    metadata = read_json(screening/"metadata.json")
    metadata["output_sha256"][path.name] = file_sha256(path)
    write_json(screening/"metadata.json", metadata)
    assert not verify_collection(screening, {"synthetic": family})["quality_valid"]


@pytest.mark.parametrize("field", ["sample_index", "content_id", "frame_id"])
def test_provenance_alignment_rejects_changed_occurrences(sources, family, field):
    manifest = pd.read_parquet(sources[3])
    if field == "sample_index":
        manifest.loc[0, "sample_index"] = 1
        manifest.loc[0, "timestamp_seconds"] = .5
    elif field == "content_id":
        manifest.loc[0, ["content_id", "image_sha256"]] = manifest.iloc[4].content_id
    else:
        manifest.loc[0, field] = "changed-frame"
    with pytest.raises(ValueError, match="misaligned"):
        aligned_video_records(manifest, family)


@pytest.mark.parametrize("unsafe", ["../outside.jpg", "/absolute.jpg", "C:/private.jpg", "v1/../escape.jpg", "v1\\escape.jpg", "v1/file.jpg:stream"])
def test_image_source_rejects_unsafe_paths_for_video_review(sources, unsafe):
    manifest = pd.read_parquet(sources[3])
    manifest.loc[1, "image_path"] = unsafe  # Repeated occurrence, not necessarily displayed.
    with ImageSource(None, sources[2]) as source, pytest.raises(ValueError, match="Unsafe relative"):
        source.validate(manifest)


def test_bad_duplicate_image_hash_and_output_overlap_fail_before_publication(sources, family, tmp_path, monkeypatch):
    run = _publish(family, np.array([-1]*48, dtype=np.int32), tmp_path/"clustering", monkeypatch)
    for output in (sources[2]/"review", run/"review", sources[0]/"features"/"review"):
        with pytest.raises(ValueError, match="separate from read-only"):
            generate_video_clustering_review(run, sources[1], sources[2], output)
        assert not output.exists()
    # Redirect just a duplicated occurrence to bad bytes without touching fixtures.
    from flir_pipeline.data import local_images
    from flir_pipeline.features import image_source

    original = local_images.declared_file
    bad = tmp_path/"bad.jpg"
    bad.write_bytes(b"synthetic-corruption")

    def resolve(root, relative):
        return bad if relative == "v1/9.jpg" else original(root, relative)

    monkeypatch.setattr(image_source, "declared_file", resolve)
    report = tmp_path/"review"
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        generate_video_clustering_review(run, sources[1], sources[2], report)
    assert not report.exists()


def test_symlink_escape_is_rejected(tmp_path):
    root = tmp_path/"root"
    root.mkdir()
    outside = tmp_path/"outside.jpg"
    outside.write_bytes(b"synthetic")
    try:
        (root/"escape.jpg").symlink_to(outside)
    except OSError:
        pytest.skip("Host does not permit creating symlinks")
    with ImageSource(None, root) as images, pytest.raises(ValueError, match="escapes root"):
        images.read("escape.jpg", hashlib.sha256(b"synthetic").hexdigest())


def test_expanded_image_root_cannot_be_used_as_report_output(sources, family, tmp_path, monkeypatch):
    run = _publish(family, np.full(48, -1, dtype=np.int32), tmp_path/"clustering", monkeypatch)
    monkeypatch.setenv("USERPROFILE", str(sources[0]))
    monkeypatch.setenv("HOME", str(sources[0]))
    before = _hashes(sources[2])
    with pytest.raises(ValueError, match="separate from read-only"):
        generate_video_clustering_review(run, sources[1], Path("~/samples"), sources[2]/"review")
    assert _hashes(sources[2]) == before


def test_historical_inputs_are_rejected_without_changing_historical_report(tmp_path):
    manifest = tmp_path/"manifest.parquet"
    pd.DataFrame({"content_id": ["historical"], "original_split": ["train"]}).to_parquet(manifest)
    spec = tmp_path/"inputs.yaml"
    spec.write_text(yaml.safe_dump({"manifest": str(manifest), "encoders": {}}))
    with pytest.raises(ValueError, match="Expected sampled_video_grid"):
        # Minimal metadata permits the report to reach its early manifest guard.
        artifact = tmp_path/"artifact"
        artifact.mkdir()
        write_json(artifact/"metadata.json", {"artifact_kind": "clustering_run"})
        generate_video_clustering_review(artifact, spec, tmp_path/"images", tmp_path/"review")
    assert not (tmp_path/"review").exists()


def test_noise_entry_exit_and_cluster_changes_remain_observed_only(sources, family):
    records = aligned_video_records(pd.read_parquet(sources[3]), family)
    labels = np.zeros(48, dtype=np.int32)
    labels[1] = -1
    labels[5] = 1
    _, summary = evaluate_clustering(labels, family.context, family.context.original_distances)
    tables = inspection_tables(labels, family.source.content_index, records, summary, family.context.neighbors)
    assert set(tables["transitions"].transition_kind) == {"to_noise", "from_noise", "cluster_to_cluster"}
    for cluster in (0, 1):
        edges = family.context.neighbors.loc[family.context.neighbors.neighbor_rank.le(10)]
        a, b = edges.query_row.to_numpy(), edges.neighbor_row.to_numpy()
        expected = int(((labels[a] == cluster) & (labels[b] == cluster)).sum())/int((labels == cluster).sum()*10)
        assert tables["clusters"].set_index("cluster_id").loc[cluster, "visual_neighbor_coherence@10"] == expected


def test_corrupt_run_fails_before_report_and_render_failure_has_no_completion(sources, family, tmp_path, monkeypatch):
    run = _publish(family, np.array([-1]*48, dtype=np.int32), tmp_path/"clustering", monkeypatch)
    before = _hashes(run)

    def fail(*args):
        raise ValueError("synthetic decode failure")

    monkeypatch.setattr(ImageSource, "decode", fail)
    partial = tmp_path/"partial"
    with pytest.raises(ValueError, match="synthetic decode failure"):
        generate_video_clustering_review(run, sources[1], sources[2], partial)
    assert partial.exists() and not (partial/"report_metadata.json").exists()
    assert before == _hashes(run)
    values = np.load(run/"cluster_labels.npy")
    values[0] = -2
    np.save(run/"cluster_labels.npy", values)
    with pytest.raises(ValueError, match="verification failed"):
        generate_video_clustering_review(run, sources[1], sources[2], tmp_path/"bad")
    assert not (tmp_path/"bad").exists()
