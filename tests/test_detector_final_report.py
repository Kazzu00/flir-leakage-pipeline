"""Final publication exercised against real synthetic evidence, never FLIR data."""

import copy
import json
import shutil
from pathlib import Path

import jsonschema
import numpy as np
import pandas as pd
import pytest
import yaml
from pydantic import ValidationError
from test_detector_association import project as project  # shared synthetic producer
from test_detector_association import publish
from typer.testing import CliRunner

from flir_pipeline.cli import app
from flir_pipeline.detection.association import load_evidence, load_registry
from flir_pipeline.detection.export_contract import DetectionExport, export_schema
from flir_pipeline.detection.final_report import (
    _publish,
    analysis_tables,
    association_results,
    centered_correlation,
    check_destinations,
    correlation,
    generate_final_report,
)
from flir_pipeline.similarity.storage import file_sha256, read_json, write_json


@pytest.fixture
def complete_project(project):
    for split in range(len(project[2]["identity"]["splits"])):
        for seed in project[2]["identity"]["config"]["training_seeds"]:
            publish(project, split, seed)
    return project


def digest(root):
    return {
        p.relative_to(root).as_posix(): file_sha256(p)
        for p in root.rglob("*")
        if p.is_file()
    }


def build(project, output_root):
    artifacts, protocol, _, _ = project
    return generate_final_report(
        protocol,
        artifacts,
        output_root / "splitting",
        output_root / "analysis",
        output_root / "frontend",
    )


def load_bundle(frontend):
    return {p.stem: read_json(p) for p in sorted(frontend.glob("*.json"))}


def test_complete_publication_schema_source_integrity_and_determinism(
    complete_project, tmp_path
):
    artifacts, protocol, plan, freeze = complete_project
    # Stage A shares the same identity/cell; this receipt adds no new observations.
    write_json(
        protocol / "pilot_validation.json", {"controlled": True, "completed_runs": 4}
    )
    pilot = artifacts / "runs" / "small-pilot"
    pilot.mkdir()
    write_json(
        pilot / "metadata.json",
        {"scientific_result": False, "state": "SMALL_PILOT_VALIDATED"},
    )
    before = digest(artifacts)
    manifest = build(complete_project, tmp_path)
    bundle = load_bundle(tmp_path / "frontend")
    schema = read_json(tmp_path / "frontend/schema/detection-export-v1.schema.json")
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(bundle, schema)
    DetectionExport.model_validate(bundle)
    assert manifest["plan_id"] == plan["plan_id"]
    assert manifest["model_config_id"] == freeze["model_config_id"]
    assert manifest["completed_runs"] == manifest["expected_runs"] == 48
    assert manifest["split_count"] == 16 and manifest["excluded_small_pilots"] == 1
    assert len(bundle["runs"]) == 48 and len(bundle["support"]) == 80
    assert {row["support"] for row in bundle["support"]} == {1}
    assert len(bundle["bootstrap"]) == 48 * 6 * 4
    assert all(r["training_seconds"] is None for r in bundle["runs"])
    assert all(a["n_splits"] == len(a["points"]) == 16 for a in bundle["associations"])
    assert all(
        a["global_pearson"] is None for a in bundle["associations"]
    )  # constant fixture outcomes
    historical = next(s for s in bundle["strategies"] if s["strategy"] == "historical")
    assert historical["std_between_splits"] is None
    assert all(
        c["std_map50_95"] is None
        for c in bundle["classes"]
        if c["strategy"] == "historical"
    )
    assert (
        next(v for v in bundle["variance"] if v["strategy"] == "historical")[
            "between_vs_within_ratio"
        ]
        is None
    )
    assert before == digest(artifacts)
    assert read_json(tmp_path / "analysis/receipt.json")["sources_unchanged"] is True
    assert (
        "matplotlib"
        in read_json(tmp_path / "analysis/receipt.json")["provenance"][
            "report_package_versions"
        ]
    )
    detail = pd.read_parquet(
        tmp_path / "analysis/strategy_level_metrics.parquet"
    ).set_index("strategy")
    assert bundle["summary"]["headline_metrics"] == detail.mean_map50_95.to_dict()
    for name in (
        "run_level_metrics",
        "split_level_metrics",
        "class_level_metrics",
        "test_support",
        "variance_summary",
        "prespecified_association_results",
    ):
        assert (tmp_path / f"analysis/{name}.csv").exists()
        assert (tmp_path / f"analysis/{name}.parquet").exists()
    assert len(list((tmp_path / "analysis/figures").glob("*.png"))) == 6
    assert len(list((tmp_path / "analysis/figures").glob("*.svg"))) == 6
    assert "no causales" in (tmp_path / "analysis/REPORT.md").read_text(
        encoding="utf-8"
    )
    assert {p.suffix for p in (tmp_path / "frontend").rglob("*") if p.is_file()} == {
        ".json"
    }
    serialized = json.dumps(bundle)
    assert "frame-" not in serialized and str(tmp_path) not in serialized
    assert not any(
        word in serialized
        for word in ("test_image_stats", "bootstrap_samples", "effective_arguments")
    )
    assert (
        sum(p.stat().st_size for p in (tmp_path / "frontend").rglob("*.json"))
        < 1_000_000
    )
    original = digest(tmp_path / "frontend")
    figures = digest(tmp_path / "analysis/figures")
    build(complete_project, tmp_path)
    assert {k: v for k, v in original.items() if k != "manifest.json"} == {
        k: v for k, v in digest(tmp_path / "frontend").items() if k != "manifest.json"
    }
    assert figures == digest(tmp_path / "analysis/figures")
    assert before == digest(artifacts)
    invalid = copy.deepcopy(bundle)
    invalid["strategies"][0]["mean_map50_95"] = 1.5
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(invalid, schema)
    invalid = copy.deepcopy(bundle)
    next(s for s in invalid["strategies"] if s["strategy"] == "historical")[
        "std_between_splits"
    ] = 0
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(invalid, schema)
    with pytest.raises(ValidationError):
        DetectionExport.model_validate(invalid)
    # A later failed attempt must retain the entire previously valid export/report.
    saved_frontend, saved_analysis = (
        digest(tmp_path / "frontend"),
        digest(tmp_path / "analysis"),
    )
    run = next(p for p in (artifacts / "runs").iterdir() if p.name != "small-pilot")
    (run / "metrics.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="requires complete controlled"):
        build(complete_project, tmp_path)
    assert digest(tmp_path / "frontend") == saved_frontend
    assert digest(tmp_path / "analysis") == saved_analysis


@pytest.mark.parametrize(
    "failure",
    [
        "missing",
        "duplicate",
        "checksum",
        "other_plan",
        "freeze",
        "pilot_instead_of_run",
    ],
)
def test_invalid_evidence_cannot_publish(project, tmp_path, failure):
    artifacts, protocol, _, _ = project
    run = publish(project)
    if failure == "duplicate":
        shutil.copytree(run, artifacts / "runs/duplicate")
    elif failure == "checksum":
        (run / "metrics.json").write_text("{}", encoding="utf-8")
    elif failure in {"other_plan", "pilot_instead_of_run"}:
        meta = read_json(run / "metadata.json")
        meta.update(
            {"plan_id": "other"}
            if failure == "other_plan"
            else {"scientific_result": False}
        )
        write_json(run / "metadata.json", meta)
    elif failure == "freeze":
        freeze = read_json(protocol / "runtime_freeze.json")
        freeze["plan_id"] = "other"
        write_json(protocol / "runtime_freeze.json", freeze)
    before = digest(artifacts)
    with pytest.raises(ValueError):
        build(project, tmp_path)
    assert not (tmp_path / "frontend").exists()
    assert not (tmp_path / "analysis").exists()
    assert digest(artifacts) == before


def test_split_units_centering_registry_and_order(complete_project):
    artifacts, protocol, _, _ = complete_project
    evidence = load_evidence(protocol, artifacts)
    registry = load_registry()
    assert set(registry.association_id) == {
        "primary_dinov2",
        "primary_clip",
        "extreme_dinov2",
        "extreme_clip",
        "temporal_at5",
        "domain_dinov2",
        "domain_clip",
    }
    # Deliberately distinct seed outcomes: aggregating seeds must retain their mean.
    rows = evidence.associations
    rows["map50_95"] = rows.split_seed * 0.07 + rows.detector_seed.map(
        {42: 0.1, 43: 0.2, 44: 0.6}
    )
    tables = analysis_tables(evidence)
    assert tables["split_level_metrics"].n_detector_runs.eq(3).all()
    assert (
        tables["split_level_metrics"]
        .query("split_seed == 0")
        .mean_map50_95.eq(0.3)
        .all()
    )
    result = association_results(evidence, registry)
    primary = next(a for a in result if a["association_id"] == "primary_dinov2")
    assert primary["n_splits"] == primary["n_valid_splits"] == 16
    assert all(
        p["y"] == pytest.approx(0.3) for p in primary["points"] if p["split_seed"] == 0
    )
    evidence.associations = rows.sample(frac=1, random_state=10)
    # Canonical sorting is required even when upstream table order changes.
    assert association_results(evidence, registry.iloc[::-1]) == result
    for name, table in analysis_tables(evidence).items():
        pd.testing.assert_frame_equal(table, tables[name])
    subset = registry.iloc[[0]].copy()
    subset["association_id"] = "registry_controls_name"
    assert [a["association_id"] for a in association_results(evidence, subset)] == [
        "registry_controls_name"
    ]
    rows.loc[rows.detector_seed.eq(44), "dinov2_nn_mean"] = 0.1
    evidence.associations = rows
    with pytest.raises(ValueError, match="context varies"):
        association_results(evidence, registry)


def test_centering_matches_manual_calculation_and_missing_pairs():
    points = pd.DataFrame(
        {
            "strategy": ["historical", "C10", "C10", "C10", "C12", "C12", "C12", "C12"],
            "x": [3, 0.1, 0.2, 0.4, 0.6, 0.7, 0.9, np.nan],
            "y": [0.2, 0.4, 0.3, 0.2, 0.5, 0.9, 0.8, 0.1],
        }
    )
    valid = points.dropna().copy()
    x = valid.x - valid.groupby("strategy").x.transform("mean")
    y = valid.y - valid.groupby("strategy").y.transform("mean")
    assert centered_correlation(points) == pytest.approx(np.corrcoef(x, y)[0, 1])
    assert correlation(points.iloc[:2]) is None
    assert correlation(points.assign(y=0.2)) is None
    constant_groups = pd.DataFrame(
        {
            "strategy": ["C10"] * 3 + ["C12"] * 3,
            "x": [0.1] * 3 + [0.2] * 3,
            "y": [0.2, 0.3, 0.4, 0.1, 0.3, 0.9],
        }
    )
    assert centered_correlation(constant_groups) is None
    tied = pd.DataFrame({"x": [1, 1, 3, 4], "y": [2, 4, 4, 1]})
    assert correlation(tied, "spearman") == pytest.approx(
        tied.x.rank().corr(tied.y.rank())
    )


def test_generic_matrix_is_not_hardcoded_to_48(project, tmp_path):
    artifacts, protocol, plan, freeze = project
    from flir_pipeline.similarity.storage import stable_id

    plan["identity"]["splits"] = plan["identity"]["splits"][:2]
    plan["plan_id"] = stable_id(plan["identity"])
    freeze["plan_id"] = plan["plan_id"]
    write_json(protocol / "plan.json", plan)
    write_json(protocol / "runtime_freeze.json", freeze)
    for split in range(2):
        for seed in (42, 43, 44):
            publish(project, split, seed)
    manifest = build(project, tmp_path)
    assert manifest["expected_runs"] == 6 and manifest["split_count"] == 2
    assert all(
        s["std_between_splits"] is None
        for s in load_bundle(tmp_path / "frontend")["strategies"]
    )


@pytest.mark.parametrize(
    "destination", ["protocol", "runs", "views", "splitting", "root"]
)
def test_write_boundary_rejects_sources_and_ancestors(tmp_path, destination):
    artifacts = tmp_path / "detection"
    paths = {
        "protocol": artifacts / "protocol",
        "runs": artifacts / "runs/subdir",
        "views": artifacts / "views",
        "splitting": tmp_path / "splitting",
        "root": tmp_path,
    }
    with pytest.raises(ValueError, match="disjoint"):
        check_destinations(
            artifacts / "protocol",
            artifacts,
            tmp_path / "splitting",
            paths[destination],
            tmp_path / "frontend",
        )


def test_publication_rolls_back_both_outputs(tmp_path, monkeypatch):
    pairs = []
    for name in ("analysis", "frontend"):
        destination, stage = tmp_path / name, tmp_path / f"{name}-stage"
        destination.mkdir()
        stage.mkdir()
        (destination / "old.txt").write_text("old", encoding="utf-8")
        (stage / "new.txt").write_text("new", encoding="utf-8")
        pairs.append((stage, destination))
    rename = Path.rename

    def fail_second(path, target):
        if path == pairs[1][0]:
            raise OSError("synthetic publication failure")
        return rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_second)
    with pytest.raises(OSError, match="publication failure"):
        _publish(pairs)
    for _, destination in pairs:
        assert (destination / "old.txt").read_text() == "old"
        assert not (destination / "new.txt").exists()


def test_cli_help_and_incomplete_exit(project, tmp_path):
    runner = CliRunner()
    help_result = runner.invoke(app, ["detection", "report", "--help"])
    assert help_result.exit_code == 0
    artifacts, protocol, _, _ = project
    result = runner.invoke(
        app,
        [
            "detection",
            "report",
            "--plan",
            str(protocol),
            "--artifacts",
            str(artifacts),
            "--split-root",
            str(tmp_path / "splitting"),
            "--analysis-output",
            str(tmp_path / "analysis"),
            "--frontend-output",
            str(tmp_path / "frontend"),
        ],
    )
    assert result.exit_code == 2 and "0/48" in result.output
    assert not (tmp_path / "frontend").exists()


def test_versioned_schema_matches_producer():
    assert (
        read_json(
            Path("exports/frontend/detection/schema/detection-export-v1.schema.json")
        )
        == export_schema()
    )


def test_registry_is_loaded_from_config(tmp_path):
    registry = yaml.safe_load(
        Path("configs/detection/associations.yaml").read_text(encoding="utf-8")
    )
    registry["associations"] = registry["associations"][:1]
    path = tmp_path / "registry.yaml"
    path.write_text(yaml.safe_dump(registry), encoding="utf-8")
    assert len(load_registry(path)) == 1


def test_unrelated_output_files_are_preserved(project, tmp_path):
    output = tmp_path / "analysis"
    output.mkdir()
    (output / "user-notes.md").write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="unrelated"):
        build(project, tmp_path)
    assert (output / "user-notes.md").read_text() == "keep"


def test_source_change_during_rendering_refuses_publication(
    complete_project, tmp_path, monkeypatch
):
    artifacts, protocol, _, _ = complete_project

    def concurrent_change(*args):
        (protocol / "unexpected.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        "flir_pipeline.detection.final_report_plot.generate_figures", concurrent_change
    )
    with pytest.raises(ValueError, match="Source artifacts changed"):
        build(complete_project, tmp_path)
    assert not (tmp_path / "frontend").exists()
    assert not (tmp_path / "analysis").exists()
    assert (artifacts / "runs").exists()


@pytest.mark.parametrize("change", ["missing", "reversed", "resamples"])
def test_invalid_stored_intervals_refuse_publication(
    complete_project, tmp_path, change
):
    artifacts, _, _, _ = complete_project
    run = next((artifacts / "runs").iterdir())
    bootstrap = read_json(run / "bootstrap.json")
    if change == "missing":
        bootstrap["intervals"].pop()
    elif change == "reversed":
        bootstrap["intervals"][0].update(lower=0.8, upper=0.2)
    else:
        bootstrap["intervals"][0]["valid_resamples"] = 3
    # Deliberately checksum-bound but semantically invalid synthetic source.
    write_json(run / "bootstrap.json", bootstrap)
    meta = read_json(run / "metadata.json")
    meta["output_sha256"]["bootstrap.json"] = file_sha256(run / "bootstrap.json")
    write_json(run / "metadata.json", meta)
    with pytest.raises(ValueError):
        build(complete_project, tmp_path)
    assert not (tmp_path / "frontend").exists()
