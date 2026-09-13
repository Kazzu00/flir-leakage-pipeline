import hashlib
import io
import json
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from typer.testing import CliRunner

from flir_pipeline.cli import app
from flir_pipeline.data.identity import dataset_id_from_manifest
from flir_pipeline.similarity.comparison import (
    compare_neighbor_spaces,
    compare_to_store,
)
from flir_pipeline.similarity.cosine import (
    compute_cosine_similarity,
    compute_topk_neighbors,
    distribution_summary,
)
from flir_pipeline.similarity.storage import (
    SimilarityConfig,
    compute_to_store,
    similarity_space_id,
    verify_similarity_directory,
)
from flir_pipeline.similarity.temporal import (
    analyze_temporal_neighbors,
    annotate_pairs,
    build_content_provenance,
    summarize_pair_relations,
)


def synthetic_feature_space(tmp_path: Path) -> tuple[Path, Path]:
    """Distinct contents can share a vector; duplicated occurrences stay in mapping."""
    feature = tmp_path/"features"
    feature.mkdir()
    raw = np.array([[1, 0], [1, 0], [0, 1], [-1, 0]], dtype=np.float32)
    np.save(feature/"embeddings_raw.npy", raw)
    np.save(feature/"embeddings_l2.npy", raw)
    contents = pd.DataFrame({"content_id": ["a", "b", "c", "d"], "embedding_row": [0, 1, 2, 3],
                             "image_sha256": ["a", "b", "c", "d"], "representative_frame_id": ["f0", "f2", "f3", "f4"],
                             "source_archive": "images.zip", "source_member_path": ["a.png", "b.png", "c.png", "d.png"]})
    contents.to_parquet(feature/"content_index.parquet", index=False)
    manifest = pd.DataFrame({"frame_id": [f"f{i}" for i in range(5)], "content_id": ["a", "a", "b", "c", "d"],
                             "image_sha256": ["a", "a", "b", "c", "d"], "label_sha256": "label",
                             "original_split": ["train", "val", "train", "test", "train"],
                             "source_archive": "images.zip", "source_member_path": ["a.png", "a-copy.png", "b.png", "c.png", "d.png"],
                             "possible_sequence": ["one", "one", "one", "two", ""],
                             "possible_frame_index": [1, 1, 3, 1, None], "temporal_inference_confidence": "medium"})
    records = manifest[["frame_id", "content_id"]].assign(embedding_row=[0, 0, 1, 2, 3])
    records.to_parquet(feature/"record_index.parquet", index=False)
    metadata = {"extractor": "synthetic", "model_id": "synthetic/unit-vectors", "model_revision": "a"*40,
                "resolved_model_revision": "a"*40, "feature_space_id": "synthetic-space", "pooling_strategy": "synthetic",
                "dataset_id": dataset_id_from_manifest(manifest), "total_records": 5, "selected_content_ids": 4,
                "unique_content_ids": 4, "embedding_dimension": 2}
    (feature/"metadata.json").write_text(json.dumps(metadata))
    manifest_path = tmp_path/"manifest.parquet"
    manifest.to_parquet(manifest_path, index=False)
    return feature, manifest_path


def test_cosine_known_values_symmetry_diagonal_and_no_silent_normalization():
    x = np.array([[1, 0], [0, 1], [-1, 0]], dtype=np.float32)
    original = x.copy()
    matrix = compute_cosine_similarity(x)
    np.testing.assert_array_equal(matrix, [[1, 0, -1], [0, 1, 0], [-1, 0, 1]])
    np.testing.assert_array_equal(x, original)
    assert matrix.dtype == np.float32
    with pytest.raises(ValueError, match="L2-normalized"):
        compute_cosine_similarity(2*x)
    with pytest.raises(ValueError, match="L2-normalized"):
        compute_cosine_similarity(np.array([[np.nan, 0], [1, 0]], dtype=np.float32))


def test_posthoc_metadata_never_changes_numerical_neighbors_and_invalidates_cache(tmp_path):
    feature, manifest = synthetic_feature_space(tmp_path)
    config = SimilarityConfig(top_k=3)
    original = compute_to_store(feature, manifest, config, tmp_path/"original")
    altered = pd.read_parquet(manifest)
    altered["original_split"] = "train"
    altered["possible_sequence"] = "one"
    altered["possible_frame_index"] = altered.content_id.map({"a": 1, "b": 2, "c": 3, "d": 4})
    altered.to_parquet(manifest, index=False)
    with pytest.raises(ValueError, match="mismatches inputs"):
        compute_to_store(feature, manifest, config, tmp_path/"original")
    updated = compute_to_store(feature, manifest, config, tmp_path/"altered")
    assert original.name == updated.name  # Metadata-only changes do not change the numerical space.
    np.testing.assert_array_equal(np.load(original/"cosine_similarity.npy"), np.load(updated/"cosine_similarity.npy"))
    columns = ["query_content_id", "neighbor_rank", "neighbor_content_id", "cosine_similarity"]
    before, after = (pd.read_parquet(p/"nearest_neighbors.parquet") for p in (original, updated))
    pd.testing.assert_frame_equal(before[columns], after[columns])
    assert before.historical_cross_split.fillna(False).any()
    assert not after.historical_cross_split.any()
    assert after.same_sequence.all()
    altered["label_sha256"] = "a-different-annotation-version"
    altered.to_parquet(manifest, index=False)
    quality = verify_similarity_directory(updated, manifest_path=manifest)
    assert not quality["canonical_dataset_id_matches"] and not quality["quality_valid"]


def test_report_uses_verified_sources_and_keeps_private_ids_out_of_display_tables(tmp_path):
    """Exercise the complete reporting boundary with tiny synthetic images, no models."""
    from flir_pipeline.similarity.reporting import (
        FIGURE_NAMES,
        generate_similarity_report,
    )

    n = 21
    image_rows = []
    archive = tmp_path/"images.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        for i in range(n):
            buffer = io.BytesIO()
            Image.new("RGB", (12, 8), (i*11, i*7, i*3)).save(buffer, "PNG")
            data = buffer.getvalue()
            digest = hashlib.sha256(data).hexdigest()
            stream.writestr(f"{i}.png", data)
            image_rows.append({"frame_id": f"f{i}", "content_id": digest, "image_sha256": digest,
                               "label_sha256": "synthetic-label", "source_archive": archive.name,
                               "source_member_path": f"{i}.png", "possible_sequence": f"seq{i%2}",
                               "possible_frame_index": i, "temporal_inference_confidence": "medium",
                               "original_split": "train" if i%3 else "test"})
    manifest = pd.DataFrame(image_rows)
    manifest_path = tmp_path/"manifest.parquet"
    manifest.to_parquet(manifest_path, index=False)
    outputs = []
    for seed, name in enumerate(("dinov2", "clip")):
        source = tmp_path/name
        source.mkdir()
        raw = np.random.default_rng(seed).normal(size=(n, 8)).astype(np.float32)
        np.save(source/"embeddings_raw.npy", raw)
        np.save(source/"embeddings_l2.npy", raw/np.linalg.norm(raw, axis=1, keepdims=True))
        contents = manifest[["content_id", "image_sha256", "source_archive", "source_member_path"]].assign(embedding_row=np.arange(n), representative_frame_id=manifest.frame_id)
        contents.to_parquet(source/"content_index.parquet", index=False)
        manifest[["frame_id", "content_id"]].assign(embedding_row=np.arange(n)).to_parquet(source/"record_index.parquet", index=False)
        metadata = {"extractor": name, "model_id": "synthetic/unit-vectors", "model_revision": "a"*40,
                    "resolved_model_revision": "a"*40, "feature_space_id": f"synthetic-{name}",
                    "pooling_strategy": "synthetic", "dataset_id": dataset_id_from_manifest(manifest),
                    "total_records": n, "selected_content_ids": n, "unique_content_ids": n, "embedding_dimension": 8}
        (source/"metadata.json").write_text(json.dumps(metadata))
        outputs.append(compute_to_store(source, manifest_path, SimilarityConfig(), tmp_path/"artifacts"))
    comparison = compare_to_store(*outputs, tmp_path/"comparison")
    saved = (comparison/"metadata.json").read_text()
    incomplete = json.loads(saved)
    incomplete["output_sha256"] = {}
    (comparison/"metadata.json").write_text(json.dumps(incomplete))
    with pytest.raises(ValueError, match="integrity"):
        compare_to_store(*outputs, tmp_path/"comparison")
    (comparison/"metadata.json").write_text(saved)
    report = tmp_path/"report"
    receipt = generate_similarity_report(*outputs, comparison, archive, report, examples=1)
    assert all((report/"figures"/name).is_file() for name in FIGURE_NAMES)
    assert len(pd.read_csv(report/"tables/global_similarity.csv")) == 2
    for path in (report/"tables").glob("*.csv"):
        text = path.read_text()
        assert all(digest not in text for digest in manifest.content_id)
        assert str(tmp_path) not in text
    selection = pd.read_parquet(report/"tables/visual_selection.parquet")
    assert len(selection) == 12
    assert selection.groupby("extractor").query_content_id.first().nunique() == 1
    assert receipt["gallery_seed"] == 0
    assert len(pd.read_parquet(report/"tables/near_unit_inspection.parquet")) == 2
    # Replacing a source member must fail at the image-identity boundary.
    bad = tmp_path/"bad"
    bad.mkdir()
    with zipfile.ZipFile(bad/archive.name, "w") as stream:
        for row in image_rows:
            stream.writestr(row["source_member_path"], b"changed")
    with pytest.raises(ValueError, match="no longer matches"):
        generate_similarity_report(*outputs, comparison, bad/archive.name, tmp_path/"bad-report", examples=1)


def test_topk_ties_self_exclusion_determinism_and_row_permutation():
    x = np.array([[1, 0], [1, 0], [0, 1], [0, -1]], dtype=np.float32)
    ids = ["d", "a", "c", "b"]
    matrix = compute_cosine_similarity(x)
    neighbors = compute_topk_neighbors(matrix, ids, 3)
    pd.testing.assert_frame_equal(neighbors, compute_topk_neighbors(matrix, ids, 3))
    assert not neighbors.query_content_id.eq(neighbors.neighbor_content_id).any()
    assert neighbors.query("query_content_id == 'd'").neighbor_content_id.tolist() == ["a", "b", "c"]
    order = [3, 1, 0, 2]
    shuffled = compute_topk_neighbors(matrix[np.ix_(order, order)], [ids[i] for i in order], 3)
    cols = ["query_content_id", "neighbor_rank", "neighbor_content_id", "cosine_similarity"]
    pd.testing.assert_frame_equal(neighbors[cols].sort_values(cols[:2]).reset_index(drop=True), shuffled[cols].sort_values(cols[:2]).reset_index(drop=True))
    with pytest.raises(ValueError, match="N-1"):
        compute_topk_neighbors(matrix, ids, 4)


def test_experiment_identity_configuration_stability():
    cfg = SimilarityConfig()
    original = similarity_space_id("dataset", "feature", cfg)
    assert original == similarity_space_id("dataset", "feature", SimilarityConfig(**json.loads(json.dumps(cfg.__dict__))))
    for other in (replace(cfg, top_k=10), replace(cfg, quantiles=(.9,)), replace(cfg, frame_delta_upper_bounds=(0, 1, 10))):
        assert original != similarity_space_id("dataset", "feature", other)
    assert original != similarity_space_id("another-dataset", "feature", cfg)
    assert original != similarity_space_id("dataset", "another-feature", cfg)
    # Operational settings are not accepted into the scientific config.
    with pytest.raises(TypeError):
        SimilarityConfig(device="cpu")


def test_temporal_consensus_unknowns_split_sets_and_cross_split_rule(tmp_path):
    feature, manifest_path = synthetic_feature_space(tmp_path)
    m = pd.read_parquet(manifest_path)
    ci = pd.read_parquet(feature/"content_index.parquet")
    contents, records = build_content_provenance(m, ci)
    assert len(records) == 5 and len(contents) == 4
    assert json.loads(contents.iloc[0].split_membership_set) == ["train", "val"]
    pairs = annotate_pairs(pd.DataFrame({"query_row": [0, 0, 0, 1], "neighbor_row": [1, 2, 3, 3]}), contents)
    assert pairs.same_sequence.tolist()[:2] == [True, False]
    assert pairs.frame_delta.iloc[0] == 2
    assert pairs.frame_delta.iloc[1:].isna().all()
    assert pairs.same_sequence.iloc[2:].isna().all()
    assert pairs.historical_cross_split.tolist() == [True, True, True, False]
    # A duplicated content with conflicting sequence/index is not reduced to a representative.
    m.loc[1, ["possible_sequence", "possible_frame_index"]] = ["other", 9]
    conflicting, _ = build_content_provenance(m, ci)
    assert not conflicting.iloc[0].sequence_provenance_valid and pd.isna(conflicting.iloc[0].frame_index)
    m.loc[1, "possible_sequence"] = "one"
    conflicting, _ = build_content_provenance(m, ci)
    assert conflicting.iloc[0].sequence_provenance_valid and pd.isna(conflicting.iloc[0].frame_index)
    m.loc[1, "possible_frame_index"] = 1
    m.loc[2, "source_archive"] = "other.zip"
    separate, _ = build_content_provenance(m, ci)
    assert not annotate_pairs(pd.DataFrame({"query_row": [0], "neighbor_row": [1]}), separate).same_sequence.iloc[0]


def test_same_multisplit_sets_can_cross_and_missing_membership_stays_unknown(tmp_path):
    feature, manifest_path = synthetic_feature_space(tmp_path)
    m = pd.read_parquet(manifest_path)
    extra = m.iloc[[2]].assign(frame_id="extra", original_split="val")
    m = pd.concat([m, extra], ignore_index=True)
    ci = pd.read_parquet(feature/"content_index.parquet")
    contents, _ = build_content_provenance(m, ci)
    pair = pd.DataFrame({"query_row": [0], "neighbor_row": [1]})
    assert annotate_pairs(pair, contents).historical_cross_split.iloc[0]
    m.loc[m.content_id == "b", "original_split"] = None
    unknown, _ = build_content_provenance(m, ci)
    assert pd.isna(annotate_pairs(pair, unknown).historical_cross_split.iloc[0])


def test_quantile_cohorts_are_unique_pairs_and_include_ties(tmp_path):
    feature, manifest = synthetic_feature_space(tmp_path)
    matrix = compute_cosine_similarity(np.load(feature/"embeddings_l2.npy"))
    a, b = np.triu_indices(4, 1)
    contents, _ = build_content_provenance(pd.read_parquet(manifest), pd.read_parquet(feature/"content_index.parquet"))
    pairs = annotate_pairs(pd.DataFrame({"query_row": a, "neighbor_row": b, "cosine_similarity": matrix[a, b]}), contents)
    assert len(pairs) == 6
    tables = summarize_pair_relations(pairs, (.5, .9), (0, 1, 5))
    q = tables["quantile_candidates"]
    assert q.threshold_cosine.tolist() == pytest.approx([0, .5])
    assert q.pair_count.tolist() == [4, 1]  # More than half at the median because of ties.
    assert q.cross_split_count.iloc[1] == 1
    assert tables["sequence_similarity"]["count"].sum() == 6
    assert tables["frame_delta_similarity"]["count"].sum() == 1
    assert distribution_summary(np.array([]))["mean"] is None
    neighbors = annotate_pairs(compute_topk_neighbors(matrix, contents.content_id.tolist(), 3), contents)
    stats = analyze_temporal_neighbors(neighbors)
    assert stats["topk"]["count"] == 12
    assert stats["rank1"]["count"] == 4
    assert stats["topk"]["unknown_sequence_count"] > 0


def test_jaccard_known_neighbor_sets_and_query_alignment():
    def table(a, b):
        return pd.DataFrame({"query_content_id": ["q"]*3+["r"]*3, "neighbor_rank": [1, 2, 3]*2,
                             "neighbor_content_id": a+b})
    left = table(["a", "b", "c"], ["a", "b", "c"])
    right = table(["a", "c", "d"], ["b", "a", "c"]).iloc[::-1]
    per_content, summary = compare_neighbor_spaces(left, right, (1, 2, 3))
    assert summary.iloc[0].mean_jaccard == .5
    assert summary.iloc[0].exact_neighbor_match_percentage == 50
    assert per_content.query("query_content_id == 'q' and k == 2").jaccard.iloc[0] == pytest.approx(1/3)
    assert per_content.query("query_content_id == 'q' and k == 3").jaccard.iloc[0] == .5
    with pytest.raises(ValueError, match="same nonempty"):
        compare_neighbor_spaces(left, right.iloc[:3], (1,))


def test_store_verifier_cli_and_cache_integrity(tmp_path):
    feature, manifest = synthetic_feature_space(tmp_path)
    cfg = SimilarityConfig(top_k=3)
    output = compute_to_store(feature, manifest, cfg, tmp_path/"similarity")
    before = (output/"metadata.json").read_bytes()
    assert verify_similarity_directory(output, feature, manifest)["quality_valid"]
    assert compute_to_store(feature, manifest, cfg, tmp_path/"similarity") == output
    assert (output/"metadata.json").read_bytes() == before
    other = compute_to_store(feature, manifest, cfg, tmp_path/"another-runtime-path")
    assert other.name == output.name  # Location/time do not enter identity.
    result = CliRunner().invoke(app, ["similarity", "verify", str(output), "--feature-directory", str(feature), "--manifest", str(manifest)])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["quality_valid"]
    summary = json.loads((output/"similarity_summary.json").read_text())
    assert summary["global_similarity"]["count"] == 6
    assert summary["near_unit_pair_count"] == 1  # Different contents, equal vectors.
    neighbors = pd.read_parquet(output/"nearest_neighbors.parquet")
    neighbors.loc[0, "neighbor_content_id"] = neighbors.loc[0, "query_content_id"]
    neighbors.to_parquet(output/"nearest_neighbors.parquet", index=False)
    assert not verify_similarity_directory(output)["quality_valid"]
    assert CliRunner().invoke(app, ["similarity", "verify", str(output)]).exit_code == 1
    with pytest.raises(ValueError, match="failed QA"):
        compute_to_store(feature, manifest, cfg, tmp_path/"similarity")


@pytest.mark.parametrize("corruption", ["matrix", "metadata", "index", "rank", "posterior", "missing"])
def test_verifier_detects_invalid_artifacts(tmp_path, corruption):
    feature, manifest = synthetic_feature_space(tmp_path)
    output = compute_to_store(feature, manifest, SimilarityConfig(top_k=3), tmp_path/"similarity")
    if corruption == "matrix":
        matrix = np.load(output/"cosine_similarity.npy")
        matrix[0, 1] = np.nan
        np.save(output/"cosine_similarity.npy", matrix)
    elif corruption == "metadata":
        p = output/"metadata.json"
        meta = json.loads(p.read_text())
        meta["feature_space_id"] = "wrong"
        p.write_text(json.dumps(meta))
    elif corruption == "index":
        p = output/"content_index.parquet"
        frame = pd.read_parquet(p).iloc[::-1]
        frame.to_parquet(p, index=False)
    elif corruption in {"rank", "posterior"}:
        p = output/"nearest_neighbors.parquet"
        frame = pd.read_parquet(p)
        frame.loc[0, "neighbor_rank" if corruption == "rank" else "frame_delta"] = 99
        frame.to_parquet(p, index=False)
    else:
        (output/"quality.json").unlink()
    assert not verify_similarity_directory(output)["quality_valid"]
