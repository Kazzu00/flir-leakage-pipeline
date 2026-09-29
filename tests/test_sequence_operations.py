"""Offline operational invariants: real-schema adapters tested separately."""

import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest
import yaml
from test_sequence_experiment_review import reviewed_source
from test_sequence_experiments import small_config, synthetic_memory, zones
from test_sequences import synthetic_sources

from flir_pipeline.sequences.experiments import slurm
from flir_pipeline.sequences.experiments.artifacts import tables
from flir_pipeline.sequences.experiments.config import (
    ClusterGrid,
    ReductionGrid,
    SuiteConfig,
)
from flir_pipeline.sequences.experiments.inputs import (
    Inputs,
    load_profile,
    resolve_inputs,
)
from flir_pipeline.sequences.experiments.real_evidence import (
    FAMILIES,
    discover,
    import_real,
)
from flir_pipeline.sequences.experiments.review import import_decisions
from flir_pipeline.sequences.experiments.runner import run_clustering, suite
from flir_pipeline.sequences.experiments.sources import load_sources
from flir_pipeline.sequences.experiments.stages import (
    execute_stage,
    graph,
    require_review,
    review_checkpoint,
)
from flir_pipeline.sequences.experiments.structure import (
    expected_binding,
    import_evidence,
)
from flir_pipeline.similarity.storage import file_sha256, read_json, write_json


@pytest.fixture
def run_plan(tmp_path, monkeypatch):
    for key in (
        "MANIFEST",
        "CLIP_FEATURES",
        "DINOV2_FEATURES",
        "IMAGES_ROOT",
        "IMAGES_ARCHIVE",
    ):
        monkeypatch.delenv("FLIR_" + key, raising=False)
    paths = synthetic_sources(tmp_path / "inputs", sizes=(72,))
    source = load_sources(*paths)
    intervals = zones(source)
    intervals["timeline_id"] = "video-0"
    producer = tmp_path / "review.csv"
    producer.write_text("synthetic evidence")
    envelope = tmp_path / "review.json"
    write_json(
        envelope,
        dict(
            artifact_kind="video11_sequence_structure_v1",
            **expected_binding(source),
            producer_files={producer.name: file_sha256(producer)},
            reviewer="synthetic",
            reviewed_at="2026-09-01T00:00:00Z",
            ground_truth=False,
            split_created=False,
            automatic_confirmation=False,
            intervals=intervals.to_dict("records"),
        ),
    )
    evidence = import_evidence(envelope, source, tmp_path / "evidence")
    image_root = tmp_path / "images"
    image_root.mkdir()
    profile = tmp_path / "profile.yaml"
    config = small_config()
    profile.write_text(
        yaml.safe_dump(
            {
                **config.model_dump(mode="json"),
                "inputs": dict(
                    manifest=str(paths[0]),
                    clip_features=str(paths[1]),
                    dinov2_features=str(paths[2]),
                    images_root=str(image_root),
                ),
                "slurm": dict(run_root=str(tmp_path / "runs")),
            }
        )
    )
    return slurm.plan_run(profile, "all", evidence), profile, paths


def test_resolution_explicit_env_metadata_ambiguity_and_corruption(
    tmp_path, monkeypatch
):
    for key in (
        "MANIFEST",
        "CLIP_FEATURES",
        "DINOV2_FEATURES",
        "IMAGES_ROOT",
        "IMAGES_ARCHIVE",
    ):
        monkeypatch.delenv("FLIR_" + key, raising=False)
    paths = synthetic_sources(tmp_path / "source", sizes=(72,))
    inputs = Inputs(search_roots=(str(tmp_path / "source"),))
    resolved, source = resolve_inputs(inputs, "all")
    assert resolved["clip_features"] == paths[1]
    assert source.signature["checksums"]["manifest_sha256"] == file_sha256(paths[0])
    duplicate = paths[1].parent / "second_clip"
    shutil.copytree(paths[1], duplicate)
    with pytest.raises(ValueError, match="found 2"):
        resolve_inputs(inputs, "all")
    monkeypatch.setenv("FLIR_CLIP_FEATURES", str(paths[1]))
    assert resolve_inputs(inputs, "all")[0]["clip_features"] == paths[1]
    array = paths[1] / "embeddings_l2.npy"
    array.write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        resolve_inputs(inputs, "all")


def test_operational_settings_do_not_enter_scientific_configuration(run_plan):
    manifest, profile, _ = run_plan
    config, _, _ = load_profile(profile)
    assert config.model_dump(mode="json") == small_config().model_dump(mode="json")
    assert "inputs" not in config.model_dump()
    assert manifest["plan"]["identity"]["source_signature"]["feature_spaces"]


def test_dag_dry_run_has_no_submission_or_files(run_plan, monkeypatch):
    manifest, _, _ = run_plan
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: pytest.fail("dry run submitted")
    )
    result = slurm.submit(manifest, dry_run=True)
    assert not Path(result["plan"]["manifest"]).exists()
    stages = result["plan"]["stages"]
    seen = set()
    for spec in stages:
        assert set(spec["dependencies"]) <= seen
        assert spec["command"] and spec["resources"] and spec["expected_artifact"]
        seen.add(spec["stage"])
    roots = [s for s in stages if not s["dependencies"]]
    assert {"boundary", "representation", "direct_recurrence"} <= {
        s["kind"] for s in roots
    }
    assert len([s for s in roots if s["kind"] == "representation"]) == 2
    assert [s for s in stages if s["stage"] == "recurrence"][0]["dependencies"] == [
        "clustering",
        "direct_recurrence",
    ]


def test_submission_receipts_afterok_and_status_failures(run_plan, monkeypatch):
    manifest, _, _ = run_plan
    calls = []

    def sbatch(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, f"{1000 + len(calls)};test\n", "")

    monkeypatch.setattr(subprocess, "run", sbatch)
    submitted = slurm.submit(manifest)
    stored = slurm.read_run(submitted["plan"]["manifest"])
    assert stored["submissions"] == submitted["submissions"]
    assert len(calls) == len(stored["plan"]["stages"])
    for spec, entry, call in zip(
        stored["plan"]["stages"], stored["submissions"], calls, strict=True
    ):
        assert entry["output_log"] == spec["log"]
        if spec["dependencies"]:
            assert (
                "--dependency=afterok:" + ":".join(entry["dependency_job_ids"]) in call
            )
    states = {entry["job_id"]: "PENDING" for entry in stored["submissions"]}
    failed = next(r for r in stored["submissions"] if r["stage"].startswith("fit_"))
    states[failed["job_id"]] = "FAILED"
    result = slurm.status(stored, states=states)
    assert result["counts"]["failed"] == 1
    assert (
        next(r for r in result["stages"] if r["stage"] == "clustering")["state"]
        == "blocked_by_dependency"
    )
    recovery = slurm.resubmission_plan(stored, states=states)
    assert (
        next(r for r in recovery["stages"] if r["stage"] == failed["stage"])["action"]
        == "resubmit_after_dependencies"
    )
    assert (
        next(r for r in recovery["stages"] if r["stage"] == "clustering")["action"]
        == "reconcile_existing_job_first"
    )
    assert recovery == slurm.resubmission_plan(stored, states=states)
    with pytest.raises(ValueError, match="already exists"):
        slurm.submit(manifest)


def test_submission_failure_preserves_history_and_unknown_acceptance(
    run_plan, monkeypatch
):
    manifest, _, _ = run_plan
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: (_ for _ in ()).throw(OSError("transport lost")),
    )
    with pytest.raises(ValueError, match="Submission stopped"):
        slurm.submit(manifest)
    stored = slurm.read_run(manifest["plan"]["manifest"])
    assert stored["submissions"][0]["state"] == "SUBMISSION_UNKNOWN"
    assert (
        slurm.resubmission_plan(stored, states={})["stages"][0]["action"]
        == "reconcile_existing_job_first"
    )


def test_worker_refuses_login_reuses_verified_and_rejects_corrupt(
    run_plan, monkeypatch
):
    manifest, _, _ = run_plan
    manifest_path = Path(manifest["plan"]["manifest"])
    manifest_path.parent.mkdir(parents=True)
    (manifest_path.parent / "receipts").mkdir()
    slurm.atomic_json(manifest_path, manifest)
    stage = manifest["plan"]["stages"][0]["stage"]
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(ValueError, match="login node"):
        slurm.worker(manifest_path, stage)
    monkeypatch.setenv("SLURM_JOB_ID", "synthetic")
    artifact = slurm.worker(manifest_path, stage)
    monkeypatch.setattr(
        slurm, "execute_stage", lambda *a, **k: pytest.fail("recomputed verified stage")
    )
    assert slurm.worker(manifest_path, stage) == artifact
    assert (
        slurm.resubmission_plan(manifest, states={})["stages"][0]["action"]
        == "reuse_verified"
    )
    (artifact / "summary.json").write_text("corrupt")
    assert slurm.status(manifest, states={})["stages"][0]["state"] == "corrupt"
    assert (
        slurm.resubmission_plan(manifest, states={})["stages"][0]["action"]
        == "refuse_artifact_reuse"
    )
    with pytest.raises(ValueError):
        slurm.worker(manifest_path, stage)


def test_parallel_fits_equal_serial_with_one_reduction_per_space(tmp_path, monkeypatch):
    from flir_pipeline.sequences.experiments import fitting

    source = synthetic_memory()
    config = SuiteConfig(
        reductions=(
            ReductionGrid(
                method="tsne", seeds=(0,), parameters={"perplexity": 5, "max_iter": 300}
            ),
        ),
        clustering=(
            ClusterGrid(algorithm="agglomerative", grid={"n_clusters": [2, 3]}),
            ClusterGrid(
                algorithm="dbscan", parameters={"min_samples": 3, "eps_quantile": 0.9}
            ),
        ),
    )
    serial = run_clustering(source, config, tmp_path / "serial")
    original = fitting.make_reducer
    calls = []

    def reducer(c):
        calls.append(c)
        return original(c)

    monkeypatch.setattr(fitting, "make_reducer", reducer)
    paths = {}
    for spec in graph(config):
        if spec["kind"] not in {"representation", "fit", "merge"}:
            continue
        paths[spec["stage"]] = execute_stage(
            spec,
            source,
            config,
            None,
            {d: paths[d] for d in spec["dependencies"]},
            tmp_path / "parallel" / spec["stage"],
            {},
        )
    assert len(calls) == 2
    for name, expected in tables(serial).items():
        pd.testing.assert_frame_equal(tables(paths["clustering"])[name], expected)
    assert read_json(serial / "backend_details.json") == read_json(
        paths["clustering"] / "backend_details.json"
    )


def test_combined_manual_checkpoint_and_final_summary(tmp_path):
    source, evidence, images = reviewed_source(tmp_path)
    config = small_config()
    # Feed existing serial artifacts to the same post-compute stages used by SLURM.
    serial = suite(source, config, tmp_path / "products", evidence)
    children = tables(serial)["artifacts"]
    paths = {
        row.role: tmp_path / "products" / row.relative_directory
        for row in children.itertuples()
    }
    paths["suite"] = serial
    specs = {s["stage"]: s for s in graph(config)}
    spec = specs["review_package"]
    package = execute_stage(
        spec,
        source,
        config,
        evidence,
        {d: paths[d] for d in spec["dependencies"]},
        tmp_path / "reviews",
        {"images_root": images},
    )
    paths["review_package"] = package
    assert review_checkpoint(package)["manual_review_required"]
    with pytest.raises(ValueError, match="Blocked by manual review"):
        require_review(package, None)
    data = tables(package)
    assert set(data["queries"].kind) == {
        "boundary_zone_candidate",
        "recurrence_pair_candidate",
    }
    assert any("cluster_recurrence" in name for name in data)
    assert data["context"].role.str.contains("strongest_clip").any()
    assert data["context"].role.str.contains("dinov2_medoid").any()
    spec = specs["final_summary"]
    final = execute_stage(
        spec,
        source,
        config,
        evidence,
        {d: paths[d] for d in spec["dependencies"]},
        tmp_path / "final",
        {},
    )
    summary = read_json(final / "summary.json")
    assert summary["manual_review_required"] and not summary["split_created"]
    assert not summary["visual_dependency_groups_created"]
    decisions = pd.DataFrame(
        dict(
            review_query_id=data["queries"].review_query_id,
            decision="ambiguous",
            reviewer="synthetic",
            reviewed_at="2026-09-01T00:00:00Z",
            notes="synthetic",
        )
    )
    csv = tmp_path / "decisions.csv"
    decisions.iloc[:1].to_csv(csv, index=False)
    partial = import_decisions(package, csv, tmp_path / "reviews")
    with pytest.raises(ValueError, match="Blocked"):
        require_review(package, partial)
    decisions.to_csv(csv, index=False)
    complete = import_decisions(package, csv, tmp_path / "reviews", previous=partial)
    assert not require_review(package, complete)["manual_review_required"]


def test_discovery_uses_content_and_rejects_ambiguous_or_changed_producers(tmp_path):
    source = synthetic_memory()
    root = tmp_path / "reports"
    for index, kind in enumerate(FAMILIES):
        folder = root / f"unrelated-directory-{index}"
        folder.mkdir(parents=True)
        producer = folder / "review.csv"
        producer.write_text("synthetic producer")
        write_json(
            folder / "metadata.json",
            dict(
                schema_version="sequence_evidence_import_v1",
                artifact_kind=kind,
                **expected_binding(source),
                producer_files={producer.name: file_sha256(producer)},
                reviewer="synthetic",
                reviewed_at="2026-09-01T00:00:00Z",
                ground_truth=False,
                split_created=False,
                automatic_confirmation=False,
                intervals=zones(source).to_dict("records") if index == 2 else [],
            ),
        )
    found = discover(root, source)
    assert set(found) == set(FAMILIES)
    normalized = import_real(root, source, root / "normalized")
    assert len(tables(normalized)["cores"]) == 3
    assert set(discover(root, source)) == set(FAMILIES)
    duplicate = root / "copy"
    shutil.copytree(root / "unrelated-directory-0", duplicate)
    with pytest.raises(ValueError, match="found 2"):
        discover(root, source)
    selected = {FAMILIES[0]: root / "unrelated-directory-0"}
    assert set(discover(root, source, selected)) == set(FAMILIES)
    (root / "unrelated-directory-2" / "review.csv").write_text("changed")
    with pytest.raises(ValueError, match="producer changed"):
        discover(root, source, selected)


def test_discovery_directory_name_never_implies_schema(tmp_path):
    source = synthetic_memory()
    folder = tmp_path / FAMILIES[0]
    folder.mkdir()
    write_json(folder / "summary.json", {"count": 12})
    with pytest.raises(ValueError, match="found 0"):
        discover(tmp_path, source)
    write_json(folder / "summary.json", {"artifact_kind": FAMILIES[0], "count": 12})
    with pytest.raises(ValueError, match="Uninspected producer schema"):
        discover(tmp_path, source)


def test_scheduler_accounting_and_pending_reason_are_not_completion(
    run_plan, monkeypatch
):
    manifest, _, _ = run_plan
    spec = manifest["plan"]["stages"][0]
    manifest["submissions"] = [
        dict(stage=spec["stage"], job_id="123", state="SUBMITTED")
    ]

    def command(args, **kwargs):
        return subprocess.CompletedProcess(
            args,
            0,
            "123|COMPLETED|1:0\n123.batch|FAILED|1:0\n" if args[0] == "sacct" else "",
            "",
        )

    monkeypatch.setattr(subprocess, "run", command)
    assert slurm.status(manifest)["stages"][0]["state"] == "failed"


def test_missing_receipt_with_orphaned_publication_requires_reconciliation(run_plan):
    manifest, _, _ = run_plan
    spec = manifest["plan"]["stages"][0]
    root = Path(spec["output_root"])
    root.mkdir(parents=True)
    (root / "partial-diagnostic.json").write_text("{}")
    assert (
        slurm.resubmission_plan(manifest, states={})["stages"][0]["action"]
        == "refuse_artifact_reuse"
    )


def test_direct_recurrence_cache_preserves_scientific_results(tmp_path, monkeypatch):
    from flir_pipeline.sequences.experiments import runner

    source, evidence, _ = reviewed_source(tmp_path)
    config = small_config()
    fitted = run_clustering(source, config, tmp_path / "fits")
    serial = runner.run_recurrence(
        source, evidence, config.recurrence, tmp_path / "serial", fitted
    )
    direct = runner.run_recurrence(
        source, evidence, config.recurrence, tmp_path / "direct"
    )
    monkeypatch.setattr(
        runner,
        "recurrence_tables",
        lambda *a, **k: pytest.fail("recomputed direct recurrence"),
    )
    staged = runner.run_recurrence(
        source, evidence, config.recurrence, tmp_path / "staged", fitted, direct
    )
    for name, expected in tables(serial).items():
        pd.testing.assert_frame_equal(tables(staged)[name], expected)


def test_changed_inputs_block_resubmission(run_plan):
    manifest, _, paths = run_plan
    (paths[1] / "metadata.json").write_text("changed")
    with pytest.raises(ValueError, match="inputs/evidence changed"):
        slurm.resubmission_plan(manifest, states={})
