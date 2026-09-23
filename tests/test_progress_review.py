"""Synthetic checks for the evidence-only project report boundary."""

import json
import runpy
from pathlib import Path

import pandas as pd
import pytest

BUILDER = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/build_progress_review.py"))
Review = BUILDER["Review"]
SOURCES = BUILDER["SOURCES"]


def write_source(root, key, value):
    path = root / SOURCES[key]
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".csv":
        pd.DataFrame(value).to_csv(path, index=False)
    elif path.suffix == ".parquet":
        pd.DataFrame(value).to_parquet(path, index=False)
    else:
        path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_absent_artifacts_remain_missing_without_fabricated_results(tmp_path):
    review = Review(tmp_path)
    section = review.section(19)
    assert "missing" in section
    assert "DONE" not in section
    assert "PENDING_FINAL_EXPERIMENT" not in section
    assert len(review.issues) == len(SOURCES)
    assert list(tmp_path.iterdir()) == []


def test_report_receipt_rejects_tampering_and_never_repairs_sources(tmp_path):
    table = write_source(tmp_path, "v_global_similarity", [{"extractor": "synthetic", "mean": 0.125}])
    write_source(tmp_path, "v_meta", {"output_sha256": {"tables/global_similarity.csv": BUILDER["digest"](table)}})
    before = Review(tmp_path)
    assert "v_global_similarity" not in before.issues
    write_source(tmp_path, "v_global_similarity", [{"extractor": "synthetic", "mean": 0.875}])
    assert not before.unchanged()
    after = Review(tmp_path)
    assert after.inventory["v_global_similarity"]["state"] == "invalid"
    assert "0.875" not in after.section(7)
    assert pd.read_csv(table)["mean"].iloc[0] == 0.875


def test_smoke_cannot_close_full_features_and_private_columns_are_not_displayed(tmp_path):
    write_source(tmp_path, "manifest", [
        {"frame_id": "synthetic-a", "content_id": "synthetic-content-a", "original_split": "train"},
        {"frame_id": "synthetic-b", "content_id": "synthetic-content-b", "original_split": "test"},
    ])
    audit = {"full_features": {}}
    for encoder, dimension in [("dinov2", 384), ("clip", 512)]:
        row = {"model_id": f"synthetic/{encoder}", "reproducible_full_dataset_valid": True,
               "content_embedding_count": 2, "record_count": 2, "embedding_dimension": dimension,
               "feature_space_id": f"synthetic-{encoder}", "pooling_strategy": "synthetic pooling",
               "feature_directory": "private-field-must-not-appear"}
        write_source(tmp_path, f"f_embedding_health_{encoder}", [row])
        audit["full_features"][encoder] = {"reproducible_full_dataset_valid": True, "feature_space_id": row["feature_space_id"]}
    write_source(tmp_path, "f_audit", audit)
    good = Review(tmp_path).section(6)
    assert "VALIDADO" in good
    assert "private-field-must-not-appear" not in good
    assert "synthetic-content-a" not in good
    row["content_embedding_count"] = 1
    row["reproducible_full_dataset_valid"] = False
    write_source(tmp_path, "f_embedding_health_clip", [row])
    blocked = Review(tmp_path).section(6)
    assert "invalid" in blocked
    assert "VALIDADO" not in blocked


def test_manifest_summary_mismatch_is_not_published(tmp_path):
    write_source(tmp_path, "manifest", [{"frame_id": "a", "content_id": "x", "original_split": "train"}])
    write_source(tmp_path, "f_dataset_summary", [{"total_records": 2, "unique_content_ids": 1}])
    write_source(tmp_path, "f_historical_split_summary", [{"original_split": "train", "records": 1}])
    review = Review(tmp_path)
    assert "f_dataset_summary" in review.issues
    assert review.inventory["manifest"]["state"] == "invalid"


def test_random_uses_recorded_aggregate_and_representative_uses_seed_zero(tmp_path):
    review = Review(tmp_path)
    metrics = ["dinov2_nn_mean", "clip_nn_mean", "dinov2_top001_pairs", "clip_top001_pairs", "temporal_at5"]
    review.data["s_runs"] = pd.DataFrame([
        {"strategy": "historical", "candidate_label": "historical", "seed": 0, **dict.fromkeys(metrics, 0.21)},
        {"strategy": "cluster_aware", "candidate_label": "C10", "seed": 0, **dict.fromkeys(metrics, 0.32)},
        {"strategy": "cluster_aware", "candidate_label": "C10", "seed": 1, **dict.fromkeys(metrics, 0.99)},
    ])
    review.data["s_metric_variation"] = pd.DataFrame([
        {"strategy": "random_content", "metric": metric, "n": 5, "mean": 0.43} for metric in metrics
    ])
    body, _, _ = review._section(15)
    assert "0.210000" in body and "0.320000" in body and "0.430000" in body
    assert "0.990000" not in body
    assert "media seeds 0–4" in body and "C10: seed 0" in body
    review.data["s_metric_variation"]["n"] = 1
    with pytest.raises(ValueError, match="cinco semillas"):
        review._section(15)


def test_changed_detector_scientific_state_requires_narrative_review(tmp_path):
    budget = {"expected_full_runs": 48}
    metadata = {"scientific_state": "PENDING_FINAL_EXPERIMENT", "fair_comparison": {"completed_runs": 0},
                "small_pilot_count": 1, "compute_budget": budget}
    write_source(tmp_path, "d_meta", metadata)
    write_source(tmp_path, "d_budget", budget)
    write_source(tmp_path, "d_verify", {"scientific_runs_completed": 0})
    write_source(tmp_path, "d_pilot_validation", [{"scientific_result": False}])
    assert "d_meta" not in Review(tmp_path).issues
    metadata["scientific_state"] = "FINAL_COMPLETE"
    metadata["fair_comparison"]["completed_runs"] = 48
    write_source(tmp_path, "d_meta", metadata)
    assert "d_meta" in Review(tmp_path).issues


@pytest.mark.parametrize("body,ids", [
    ("<p>C:/private/example</p>", []),
    ("<p>/home/private/example</p>", []),
    ("<p>" + "a" * 64 + "</p>", []),
    ('<img alt="synthetic-sensitive-id">', ["synthetic-sensitive-id"]),
    ('<div class="jp-CodeMirrorEditor"><pre>print(1)</pre></div>', []),
])
def test_visible_privacy_and_code_audit(body, ids):
    with pytest.raises(ValueError):
        BUILDER["audit_html"](body, ids)


def test_lab_markdown_is_visible_narrative_not_code():
    result = BUILDER["audit_html"]('<div class="jp-InputArea"><div class="jp-RenderedMarkdown"><h2>1. Investigación</h2></div></div>')
    assert result["code_hidden"] and result["sections"] == 1


def test_progress_source_is_clean_spanish_with_all_sections():
    root = Path(__file__).resolve().parents[1]
    source = root / "notebooks/progress_review.ipynb"
    runpy.run_path(str(root / "scripts/check_notebook_source.py"))["check_notebook_source"](source)
    nb = json.loads(source.read_text(encoding="utf-8"))
    assert "— Project Report" in "".join(nb["cells"][0]["source"])
    sections = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown" and "".join(c["source"]).startswith("## ")]
    assert len(sections) == 22
    assert all("**Pregunta / objetivo:** ¿" in section for section in sections)
    assert "Próximos pasos" in sections[-1]
