"""Offline inspection boundaries over completed synthetic clustering publications."""

import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import test_clustering
from typer.testing import CliRunner

from flir_pipeline.cli import app
from flir_pipeline.clustering import distances, experiments, metrics, selection, storage
from flir_pipeline.clustering.base import ClusteringConfig
from flir_pipeline.clustering.inspection import inspect_clustering
from flir_pipeline.clustering.metrics import EvaluationContext
from flir_pipeline.similarity.storage import file_sha256, read_json, write_json


@pytest.fixture(scope="module", params=["historical", "video"])
def publications(request, tmp_path_factory):
    pytest.importorskip("sklearn")
    root = tmp_path_factory.mktemp(f"inspection-{request.param}")
    family = test_clustering.family.__wrapped__(root)
    if request.param == "video":
        context = family.context
        provenance = pd.DataFrame({"content_id": context.content_ids,
                                   "temporal_source": "sampled_video_grid"})
        family = replace(family, context=EvaluationContext.create(
            context.content_ids, context.original_distances, context.cosine,
            context.neighbors, provenance,
        ))
    configs = [ClusteringConfig("dbscan", {"min_samples": 5, "eps_quantile": q})
               for q in (.7, .8, .9)]
    screening = experiments.screening_to_store({"synthetic": family}, configs, root)
    comparison = experiments.comparison_to_store(screening, {"synthetic": family}, root)
    run = root / read_json(screening / "metadata.json")["runs"][0]["path"]
    return {"run": run, "screening": screening, "comparison": comparison,
            "mode": request.param}


def snapshot(directory):
    return {p.name: (file_sha256(p), p.stat().st_mtime_ns)
            for p in directory.iterdir() if p.is_file()}


def reject_scientific_work(monkeypatch, directory):
    def reject(*args, **kwargs):
        raise AssertionError("Lightweight inspection attempted scientific work")

    for module, names in (
        (distances, ("pdist", "squareform", "distance_matrix")),
        (distances.EuclideanDistances, ("materialize",)),
        (np, ("load", "memmap")),
        (pd, ("read_parquet",)),
        (storage, ("verify_run", "load_family", "load_inputs", "evaluate_clustering", "fit_clustering")),
        (experiments, ("verify_collection", "verify_run", "load_families", "run_row",
                       "assignment_agreement", "shortlist_screening", "final_candidates")),
        (metrics, ("evaluate_clustering", "assignment_agreement")),
        (selection, ("shortlist_screening", "final_candidates")),
    ):
        for name in names:
            monkeypatch.setattr(module, name, reject)
    original_open = Path.open

    def selected_read_only(path, mode="r", *args, **kwargs):
        assert path.resolve().parent == directory.resolve(), "Traversed a scientific run/source"
        assert not set(mode) & set("wax+"), "Inspection attempted a write"
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", selected_read_only)


@pytest.mark.parametrize("kind", ["run", "screening", "comparison"])
def test_inspect_is_bounded_read_only_and_never_loads_arrays(publications, kind, tmp_path, monkeypatch):
    # A portable leaf copy has no scientific inputs, sibling runs or screening.
    directory = tmp_path / "selected"
    shutil.copytree(publications[kind], directory)
    before = snapshot(directory)
    saved = read_json(directory / ("metrics.json" if kind == "run" else "summary.json"))
    with monkeypatch.context() as guarded:
        reject_scientific_work(guarded, directory)
        result = inspect_clustering(directory, limit=1)
        cli = CliRunner().invoke(app, ["clustering", "inspect", str(directory), "--limit", "1"])
        assert cli.exit_code == 0, cli.output
        assert json.loads(cli.stdout) == result
    assert snapshot(directory) == before
    assert result["inspection_mode"] == "lightweight"
    assert "NOT full scientific verification" in result["notice"]
    assert result["validation"]["scientific_verification_performed"] is False
    assert "quality_valid" not in result
    assert result["validation"]["checked_output_files"] == sorted(read_json(directory / "metadata.json")["output_sha256"])
    assert result["metrics" if kind == "run" else "summary"] == saved
    for table in (("shortlist",) if kind == "screening" else ("references", "candidates") if kind == "comparison" else ()):
        preview = result[table]
        expected = pd.read_csv(directory / f"{table}.csv")
        assert preview["row_count"] == len(expected)
        assert preview["shown_rows"] == min(1, len(expected))
        assert preview["truncated"] == (len(expected) > 1)
        assert len(preview["columns"]) < len(expected.columns)
        assert [r["clustering_space_id"] for r in preview["rows"]] == expected.clustering_space_id.head(1).tolist()
    if kind != "run":
        # Existing commands retain their strict requirement for referenced runs.
        for command in ("summary", "verify"):
            cli = CliRunner().invoke(app, ["clustering", command, str(directory)])
            assert cli.exit_code == 1, cli.output


def test_video_unavailable_metrics_stay_none(publications):
    if publications["mode"] != "video":
        return
    result = inspect_clustering(publications["run"])
    assert result["metadata"]["provenance_mode"] == "sampled_video_grid"
    saved = result["metrics"]
    unknown = [key for key, value in saved.items() if value is None]
    assert "temporal_pairs@5" in unknown and "historical_train_only_clusters" in unknown
    assert saved["temporal_recall@5"] is None
    assert saved["weighted_dominant_sequence_fraction"] is None
    assert saved["historical_multisplit_clusters"] is None
    for kind, tables in (("screening", ("shortlist",)), ("comparison", ("references", "candidates"))):
        inspected = inspect_clustering(publications[kind])
        for name in tables:
            assert inspected[name]["rows"], "Synthetic video case must exercise CSV missing metrics"
            for row in inspected[name]["rows"]:
                assert row["temporal_recall@5"] is None
                assert row["weighted_dominant_sequence_fraction"] is None
                assert row["weighted_sequence_entropy_bits"] is None
                assert row["historical_multisplit_clusters"] is None
                assert row["temporal_provenance_mode"] == "sampled_video_grid"
                assert row["historical_split_evaluation"] == "unavailable"


@pytest.mark.parametrize("kind,filename", [
    ("run", "metrics.json"), ("run", "cluster_labels.npy"),
    ("screening", "summary.json"), ("screening", "shortlist.csv"),
    ("comparison", "summary.json"), ("comparison", "references.csv"),
    ("comparison", "candidates.csv"), ("comparison", "assignment_comparisons.parquet"),
])
@pytest.mark.parametrize("damage", ["bytes", "checksum", "missing"])
def test_corrupt_or_missing_outputs_fail_clearly(publications, tmp_path, kind, filename, damage):
    directory = tmp_path / "selected"
    shutil.copytree(publications[kind], directory)
    path = directory / filename
    if damage == "bytes":
        path.write_bytes(path.read_bytes() + b"corrupt")
    elif damage == "checksum":
        meta = read_json(directory / "metadata.json")
        meta["output_sha256"][filename] = "0" * 64
        write_json(directory / "metadata.json", meta)
    else:
        path.unlink()
    before = snapshot(directory)
    cli = CliRunner().invoke(app, ["clustering", "inspect", str(directory)])
    assert cli.exit_code == 1, cli.output
    assert "Lightweight inspection failed" in cli.output and filename in cli.output
    assert ("Missing inspection file" if damage == "missing" else "checksum mismatch") in cli.output
    assert snapshot(directory) == before


@pytest.mark.parametrize("kind", ["run", "screening", "comparison"])
@pytest.mark.parametrize("damage", ["missing", "invalid_json", "identity", "missing_checksum", "malformed_checksum"])
def test_invalid_selected_metadata_is_rejected(publications, tmp_path, kind, damage):
    directory = tmp_path / "selected"
    shutil.copytree(publications[kind], directory)
    path = directory / "metadata.json"
    if damage == "missing":
        path.unlink()
    elif damage == "invalid_json":
        path.write_text("[", encoding="utf-8")
    else:
        meta = read_json(path)
        filename = "metrics.json" if kind == "run" else "summary.json"
        if damage == "identity":
            meta["clustering_space_id" if kind == "run" else "collection_id"] = "invalid"
        elif damage == "missing_checksum":
            del meta["output_sha256"][filename]
        else:
            meta["output_sha256"][filename] = "invalid"
        write_json(path, meta)
    cli = CliRunner().invoke(app, ["clustering", "inspect", str(directory)])
    assert cli.exit_code == 1, cli.output
    assert "metadata.json" in cli.output


@pytest.mark.parametrize("filename", ["summary.json", "references.csv"])
def test_malformed_inspected_payload_with_updated_checksum_is_rejected(publications, tmp_path, filename):
    directory = tmp_path / "selected"
    shutil.copytree(publications["comparison"], directory)
    path = directory / filename
    path.write_text("[]" if filename.endswith("json") else "wrong,columns\n1,2\n", encoding="utf-8")
    meta = read_json(directory / "metadata.json")
    meta["output_sha256"][filename] = file_sha256(path)
    write_json(directory / "metadata.json", meta)
    with pytest.raises(ValueError, match=filename):
        inspect_clustering(directory)


def test_output_checksums_cannot_redirect_inspection(publications, tmp_path):
    directory = tmp_path / "selected"
    shutil.copytree(publications["screening"], directory)
    external = tmp_path / "external.json"
    external.write_text("{}", encoding="utf-8")
    meta = read_json(directory / "metadata.json")
    meta["output_sha256"]["../external.json"] = file_sha256(external)
    write_json(directory / "metadata.json", meta)
    with pytest.raises(ValueError, match="filename"):
        inspect_clustering(directory)


def test_empty_saved_candidate_table_is_valid(publications, tmp_path):
    directory = tmp_path / "selected"
    shutil.copytree(publications["comparison"], directory)
    path = directory / "candidates.csv"
    pd.read_csv(path).head(0).to_csv(path, index=False)
    meta = read_json(directory / "metadata.json")
    meta["output_sha256"][path.name] = file_sha256(path)
    write_json(directory / "metadata.json", meta)
    result = inspect_clustering(directory)
    assert result["candidates"]["row_count"] == 0
    assert result["candidates"]["rows"] == []
    assert result["candidates"]["truncated"] is False
    # Inspection is byte integrity, not a reconstruction of saved selection.
    assert result["validation"]["scientific_verification_performed"] is False


def test_missing_directory_and_invalid_limit_fail(tmp_path):
    cli = CliRunner().invoke(app, ["clustering", "inspect", str(tmp_path / "absent")])
    assert cli.exit_code == 1 and "Missing inspection file: metadata.json" in cli.output
    cli = CliRunner().invoke(app, ["clustering", "inspect", str(tmp_path), "--limit", "0"])
    assert cli.exit_code == 2
