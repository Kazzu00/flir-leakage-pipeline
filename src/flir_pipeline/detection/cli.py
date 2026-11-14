"""Controlled detector commands; optional runtime loads only on execution."""

from pathlib import Path

import typer

from flir_pipeline.detection.preliminary_cli import app as preliminary_app

app = typer.Typer(help="Frozen historical/random/cluster detector comparison.")
app.add_typer(preliminary_app, name="preliminary-base")
common_executor_app = typer.Typer(help="YOLO11m common-test operations with versioned technical admission; signed scientific operations remain separate.")
app.add_typer(common_executor_app, name="common-executor")


@common_executor_app.command("prepare-curated-development")
def common_prepare_curated_development(
    source_root: Path, manifest: Path, matrix: Path, provenance: Path,
    output: Path, linux_root: str = typer.Option(...),
    protected_records: Path | None = None, protected_receipt: Path | None = None,
) -> None:
    """Prepare or reverify curated development only. Does not admit training."""
    from flir_pipeline.detection.curated_development import prepare_curated

    result = prepare_curated(source_root, manifest, matrix, provenance, output,
                             linux_root, protected_records, protected_receipt)
    typer.echo(f"{result['state']}: {result['canonical_contents']} contents; "
               f"protection={result['protected_exclusion']['status']}")

def _prime_cuda_import_order() -> None:
    """Load GPU dependencies before the detection runtime."""
    import torch  # noqa: F401
    from ultralytics import settings  # noqa: F401


@common_executor_app.command("dry-run")
def common_executor_plan(publication: Path, output: Path,
                         config: Path = Path("configs/detection/common_test_yolo11m_v2.yaml"),
                         checkpoint_sha256: str | None = None,
                         input_format: str = "legacy_fixed_test") -> None:
    import json

    from flir_pipeline.detection.common_executor import dry_run
    typer.echo(json.dumps(dry_run(publication, config, output, checkpoint_sha256, input_format), indent=2))


@common_executor_app.command("prepare")
def common_executor_prepare(publication: Path, source_root: Path, output: Path,
                            evidence: Path | None = None,
                            input_format: str = "legacy_fixed_test") -> None:
    from flir_pipeline.detection.common_executor import prepare
    result = prepare(publication, source_root, output, evidence, input_format)
    typer.echo(f"Prepared {len(result['views'])} development views; operational freeze pending")


@common_executor_app.command("verify")
def common_executor_verify(directory: Path, kind: str = "plan") -> None:
    from flir_pipeline.detection.common_executor import (
        verify_executor_plan,
        verify_preparation,
    )
    from flir_pipeline.detection.common_runtime import verify_operation
    checks = {"plan": verify_executor_plan, "preparation": verify_preparation, "operation": verify_operation}
    if kind not in checks:
        raise typer.BadParameter("kind must be plan, preparation or operation")
    result = checks[kind](directory)
    typer.echo(f"Verified {kind}: {result.get('state', result.get('plan_id', 'technical_only'))}")


@common_executor_app.command("preflight")
def common_executor_preflight(output: Path, cpu: bool = False,
                              config: Path = Path("configs/detection/common_test_yolo11m_v2.yaml")) -> None:
    _prime_cuda_import_order()
    from flir_pipeline.detection.common_runtime import preflight
    typer.echo(preflight(config, output, cpu))


@common_executor_app.command("synthetic-smoke")
def common_executor_smoke(output: Path, batch: int = 1, cpu: bool = False,
                          config: Path = Path("configs/detection/common_test_yolo11m_v2.yaml")) -> None:
    _prime_cuda_import_order()
    from flir_pipeline.detection.common_runtime import synthetic_smoke
    typer.echo(synthetic_smoke(config, output, batch, cpu))


@common_executor_app.command("probe")
def common_executor_probe(output: Path,
                          config: Path = Path("configs/detection/common_test_yolo11m_v2.yaml")) -> None:
    _prime_cuda_import_order()
    from flir_pipeline.detection.common_runtime import probe
    typer.echo(probe(config, output))


@common_executor_app.command("train")
def common_executor_train(plan: Path, output: Path, index: int = 0, resume: bool = False,
                          admission: Path | None = None, signature: Path | None = None,
                          resume_admission: Path | None = None, resume_signature: Path | None = None,
                          technical_freeze: Path | None = None) -> None:
    from flir_pipeline.detection.common_executor import train
    try:
        result = train(plan, output, index, resume, admission, signature, resume_admission, resume_signature, technical_freeze)
        typer.echo(result["state"])
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc


@common_executor_app.command("legacy-admission-check", hidden=True)
def common_admission_check(plan: Path, output: Path, admission: Path, signature: Path) -> None:
    """Check signed matrix and immutable inputs; no model import or training."""
    from flir_pipeline.detection.scientific_executor import context
    checked = context(admission, signature, plan, output)
    typer.echo(len(checked["runs"]))


@common_executor_app.command("legacy-prepare-test", hidden=True)
def common_prepare_test(admission: Path, signature: Path, training_admission: Path,
                        training_signature: Path, output: Path) -> None:
    """Custodian operation after a signed release of the entire completed matrix."""
    from flir_pipeline.detection.scientific_executor import prepare_test
    typer.echo(prepare_test(admission, signature, training_admission, training_signature, output)["state"])


@common_executor_app.command("legacy-evaluate", hidden=True)
def common_evaluate(admission: Path, signature: Path, training_admission: Path,
                    training_signature: Path, test_root: Path, output: Path, index: int = 0) -> None:
    from flir_pipeline.detection.scientific_executor import evaluate_run
    typer.echo(evaluate_run(admission, signature, training_admission, training_signature, test_root, output, index)["state"])


@common_executor_app.command("admit")
def common_technical_admit(publication: Path, plan: Path, prepared: Path,
                           source_root: Path, output: Path,
                           config: Path = Path("configs/detection/common_test_yolo11m_v2.yaml"),
                           evidence: Path | None = None) -> None:
    """Verify original bytes and the complete matrix, without human assertions."""
    from flir_pipeline.detection.common_admission import admit
    typer.echo(admit(publication, plan, prepared, config, source_root, output, evidence)["counts"])


@common_executor_app.command("verify-weights")
def common_verify_weights(checkpoint: Path, output: Path) -> None:
    """Explicit HTTPS comparison with official release; no model execution."""
    from flir_pipeline.detection.common_admission import verify_official_weights
    typer.echo(verify_official_weights(checkpoint, output)["receipt_id"])


@common_executor_app.command("pilot")
def common_pilot(gate: Path, weights: Path, probe: Path, output: Path,
                 batch: int = 1, index: int = 0, resume: bool = False) -> None:
    """Operational development-only epoch; never counts toward scientific matrix."""
    from flir_pipeline.detection.common_training import pilot
    typer.echo(pilot(gate, weights, probe, output, batch, index, resume)["state"])


@common_executor_app.command("freeze")
def common_freeze(gate: Path, weights: Path, probe: Path, pilot: Path,
                  output: Path, batch: int = 1) -> None:
    from flir_pipeline.detection.common_training import freeze
    typer.echo(len(freeze(gate, weights, probe, pilot, output, batch)["runs"]))


@common_executor_app.command("admission-check")
def common_automatic_check(freeze: Path) -> None:
    """Recheck full technical freeze and return the derived unique execution count."""
    from flir_pipeline.detection.common_training import frozen_context
    typer.echo(len(frozen_context(freeze)["runs"]))


@common_executor_app.command("release-test")
def common_release_test(freeze: Path, training: Path, output: Path) -> None:
    from flir_pipeline.detection.common_evaluation import release
    typer.echo(release(freeze, training, output)["receipt_id"])


@common_executor_app.command("prepare-test")
def common_automatic_test(release: Path, output: Path) -> None:
    from flir_pipeline.detection.common_evaluation import prepare_test
    typer.echo(prepare_test(release, output)["state"])


@common_executor_app.command("evaluate")
def common_automatic_evaluate(release: Path, test: Path, output: Path, index: int = 0) -> None:
    from flir_pipeline.detection.common_evaluation import evaluate_run
    typer.echo(evaluate_run(release, test, output, index)["state"])


@common_executor_app.command("compare")
def common_compare(release: Path, test: Path, evaluations: Path, output: Path) -> None:
    from flir_pipeline.detection.common_evaluation import compare
    typer.echo(compare(release, test, evaluations, output)["state"])


@common_executor_app.command("status")
def common_executor_status(plan: Path) -> None:
    from flir_pipeline.detection.common_executor import verify_executor_plan
    checked = verify_executor_plan(plan)
    for index, run in enumerate(checked["runs"]):
        typer.echo(f"{index}\t{run['run_id']}\tcandidate\tautomatic data/probe/pilot/freeze checks required; use admission-check for operational readiness")


@app.command("ground-truth-quality")
def ground_truth_quality_command(
    audit: Path = typer.Option(...), proposal: Path = typer.Option(...),
    review: Path = typer.Option(...), partitions: Path = typer.Option(...),
    source_root: Path = typer.Option(...), output: Path = typer.Option(...),
    observations: Path | None = typer.Option(None),
    evidence: Path | None = typer.Option(None),
) -> None:
    """Audit 372 proposed test images and 30 existing train/val arms; never approve."""
    from flir_pipeline.detection.ground_truth_quality_storage import build_quality

    try:
        result = build_quality(audit, proposal, review, partitions, source_root, output,
                               observations=observations, evidence=evidence)
    except (ValueError, OSError, KeyError, TypeError, AssertionError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"{result['verdict']}; READY_FOR_TRAINING=false; {output / 'index.html'}")


@app.command("verify-ground-truth-quality")
def verify_ground_truth_quality_command(
    directory: Path, audit: Path = typer.Option(...), proposal: Path = typer.Option(...),
    review: Path = typer.Option(...), partitions: Path = typer.Option(...),
    source_root: Path = typer.Option(...),
    evidence: Path | None = typer.Option(None),
) -> None:
    """Reproduce tables and visual evidence from original sources; no adjudication."""
    import json

    from flir_pipeline.detection.ground_truth_quality_storage import verify_quality

    try:
        result = verify_quality(directory, audit, proposal, review, partitions, source_root, evidence=evidence)
    except (ValueError, OSError, KeyError, TypeError, AssertionError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(json.dumps(result, indent=2))


@app.command("validate-ground-truth-review")
def validate_ground_truth_review_command(directory: Path, decisions: Path) -> None:
    """Validate attributed observations in a separate form; never apply labels."""
    import json

    from flir_pipeline.detection.common_test_storage import signatures
    from flir_pipeline.detection.ground_truth_quality import validate_human_review
    from flir_pipeline.similarity.storage import read_json

    try:
        receipt = read_json(directory / "receipt.json")
        actual = signatures(directory)
        actual.pop("receipt.json", None)
        if actual != receipt["artifacts"]:
            raise ValueError("Quality evidence checksum mismatch")
        manifest = read_json(directory / "manifest.json")
        frames = [r["frame_id"] for r in read_json(directory / "test_inventory.json")["records"]]
        result = validate_human_review(read_json(decisions), manifest["audit_id"], frames)
    except (ValueError, OSError, KeyError, TypeError, AssertionError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(json.dumps(result, indent=2))


@app.command("protected-annotation-review")
def protected_annotation_review_command(
    audit: Path = typer.Option(...), proposal: Path = typer.Option(...),
    acceptance: Path = typer.Option(...), source_root: Path = typer.Option(...),
    output: Path = typer.Option(...),
) -> None:
    """Generate eight pending human reviews outside the repo; never adjudicate."""
    from flir_pipeline.detection.annotation_review import build_review

    try:
        result = build_review(audit, proposal, acceptance, source_root, output)
    except (ValueError, OSError, KeyError, TypeError, AssertionError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"{result['state']}: {len(result['cases'])} cases; {output / 'index.html'}")


@app.command("verify-annotation-review")
def verify_annotation_review_command(
    directory: Path, audit: Path = typer.Option(...), proposal: Path = typer.Option(...),
    acceptance: Path = typer.Option(...), source_root: Path = typer.Option(...),
) -> None:
    """Reproduce evidence from original sources, without applying human decisions."""
    import json

    from flir_pipeline.detection.annotation_review import verify_review

    try:
        result = verify_review(directory, audit, proposal, acceptance, source_root)
    except (ValueError, OSError, KeyError, TypeError, AssertionError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(json.dumps(result, indent=2))


@app.command("validate-annotation-decisions")
def validate_annotation_decisions_command(directory: Path, decisions: Path) -> None:
    """Check attributed decision proposals; grants no approval and changes no labels."""
    import json

    from flir_pipeline.detection.annotation_review import validate_decisions
    from flir_pipeline.detection.common_test_storage import verify_receipt
    from flir_pipeline.similarity.storage import read_json

    try:
        verify_receipt(directory, "receipt.json", "artifacts")
        result = validate_decisions(read_json(decisions), read_json(directory / "manifest.json"))
    except (ValueError, OSError, KeyError, TypeError, AssertionError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(json.dumps(result, indent=2))


def _common_proposal(audit, splits, review, export, reference, output, config,
                     source_root, evidence, legacy_artifacts, purpose):
    import yaml

    from flir_pipeline.detection.common_test import CommonTestConfig
    from flir_pipeline.detection.common_test_storage import build_common_test

    try:
        settings = CommonTestConfig(**yaml.safe_load(config.read_text(encoding="utf-8")))
        proposal = build_common_test(audit, splits, review, export, reference, output, settings,
                                    purpose=purpose, source_root=source_root, evidence_file=evidence,
                                    legacy_artifacts=legacy_artifacts)
    except (ValueError, OSError, KeyError, TypeError, AssertionError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"{proposal['readiness']['state']}: {proposal['plan_id']}; READY_FOR_TRAINING=false")


@app.command("plan-common-test")
def plan_common_test_command(
    audit: Path = typer.Option(...), splits: Path = typer.Option(...), review: Path = typer.Option(...),
    output: Path = typer.Option(...), config: Path = typer.Option(Path("configs/detection/common_test_v1.yaml")),
    export: Path = typer.Option(Path("exports/frontend/detection")),
    reference: Path = typer.Option(Path("configs/detection/yolo11n.yaml")),
    source_root: Path | None = typer.Option(None), evidence: Path | None = typer.Option(None),
    legacy_artifacts: Path | None = typer.Option(None),
) -> None:
    """Publish a candidate common test and development pools; never approve or train."""
    _common_proposal(audit, splits, review, export, reference, output, config,
                     source_root, evidence, legacy_artifacts, "plan")


@app.command("audit-readiness")
def audit_readiness_command(
    audit: Path = typer.Option(...), splits: Path = typer.Option(...), review: Path = typer.Option(...),
    output: Path = typer.Option(...), config: Path = typer.Option(Path("configs/detection/common_test_v1.yaml")),
    export: Path = typer.Option(Path("exports/frontend/detection")),
    reference: Path = typer.Option(Path("configs/detection/yolo11n.yaml")),
    source_root: Path | None = typer.Option(None), evidence: Path | None = typer.Option(None),
    legacy_artifacts: Path | None = typer.Option(None),
) -> None:
    """Audit source evidence, test support and blockers in a fresh external output."""
    _common_proposal(audit, splits, review, export, reference, output, config,
                     source_root, evidence, legacy_artifacts, "readiness")


@app.command("verify-common-test")
def verify_common_test_command(directory: Path) -> None:
    import json

    from flir_pipeline.detection.common_test_storage import verify_common_test

    try:
        result = verify_common_test(directory)
    except (ValueError, OSError, KeyError, TypeError, AssertionError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(json.dumps(result, indent=2))


@app.command("report")
def report_command(
    plan: Path = Path("artifacts/detection/protocol"),
    artifacts: Path = Path("artifacts/detection"),
    split_root: Path = Path("artifacts/splitting/runs"),
    analysis_output: Path = Path("artifacts/detection/final_report"),
    frontend_output: Path = Path("exports/frontend/detection"),
    associations: Path = Path("configs/detection/associations.yaml"),
) -> None:
    """Verify a complete frozen experiment and publish descriptive report/export."""
    from flir_pipeline.detection.final_report import generate_final_report

    try:
        manifest = generate_final_report(plan, artifacts, split_root, analysis_output, frontend_output, associations)
    except (OSError, ValueError, KeyError, TypeError, AssertionError) as exc:
        typer.echo(f"Final detector report refused: {exc}", err=True)
        raise typer.Exit(2) from exc
    typer.echo(f"{manifest['state']}: {manifest['completed_runs']}/{manifest['expected_runs']} verified runs; "
               f"{manifest['split_count']} splits. Report: {analysis_output}; export: {frontend_output}")


@app.command("plan")
def plan_command(comparison: Path, config: Path = Path("configs/detection/yolo11n.yaml"), output: Path = Path("artifacts/detection/protocol")) -> None:
    from flir_pipeline.detection.protocol import audit_and_plan

    plan = audit_and_plan(comparison, config, output)
    typer.echo(f"Plan {plan['plan_id']}: {len(plan['identity']['splits'])} splits; hardware probe still required")


@app.command("materialize")
def materialize_command(plan: Path = Path("artifacts/detection/protocol"), manifest: Path = Path("data/manifests/flir_canonical_candidate_v1.parquet"),
                        split_root: Path = Path("artifacts/splitting/runs"), output: Path = Path("artifacts/detection"), root: Path | None = None) -> None:
    from flir_pipeline.cli import _default_root
    from flir_pipeline.detection.materialization import materialize
    from flir_pipeline.detection.protocol import verify_plan

    source = root or _default_root()
    if source is None:
        raise typer.BadParameter("Set FLIR_DATA_ROOT or provide --root")
    for split in verify_plan(plan)["identity"]["splits"]:
        result = materialize(split_root/split["split_space_id"], manifest, source, output)
        typer.echo(f"Verified {split['strategy']} seed={split['split_seed']}: {result.name}")


@app.command("environment")
def environment_command(output: Path = Path("artifacts/detection/environment.json")) -> None:
    from flir_pipeline.detection.runtime import configure_runtime, environment
    from flir_pipeline.similarity.storage import write_json

    configure_runtime()
    receipt = environment()
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, receipt)
    typer.echo(str(receipt))


@app.command("smoke")
def smoke_command(plan: Path = Path("artifacts/detection/protocol"), output: Path = Path("artifacts/detection/runtime_smoke"), cpu: bool = False) -> None:
    from flir_pipeline.detection.runtime import synthetic_smoke

    result = synthetic_smoke(plan, output, gpu=not cpu)
    typer.echo(result["state"])
    if result["state"] == "BLOCKED_NO_CUDA":
        raise typer.Exit(2)


@app.command("probe")
def probe_command(plan: Path = Path("artifacts/detection/protocol"), output: Path = Path("artifacts/detection")) -> None:
    from flir_pipeline.detection.runtime import batch_probe

    typer.echo(str(batch_probe(plan, output)))


@app.command("pilot-small")
def pilot_small_command(plan: Path = Path("artifacts/detection/protocol"), output: Path = Path("artifacts/detection")) -> None:
    from flir_pipeline.detection.runtime import small_pilot

    typer.echo(str(small_pilot(plan, output)))


@app.command("freeze")
def freeze_command(plan: Path = Path("artifacts/detection/protocol"), output: Path = Path("artifacts/detection")) -> None:
    from flir_pipeline.detection.runtime import freeze_runtime

    result = freeze_runtime(plan, output/"batch_probe.json", output)
    typer.echo(f"Runtime frozen: {result['model_config_id']}")


@app.command("stage-a")
def stage_a_command(
    plan: Path = Path("artifacts/detection/protocol"),
    output: Path = Path("artifacts/detection"),
) -> None:
    from flir_pipeline.detection.runtime import execute_stage_a

    result = execute_stage_a(plan, output)
    typer.echo(str(result))
    if not result["controlled"]:
        raise typer.Exit(2)


@app.command("run")
def run_command(plan: Path = Path("artifacts/detection/protocol"), output: Path = Path("artifacts/detection")) -> None:
    from flir_pipeline.detection.runtime import execute_matrix

    result = execute_matrix(plan, output)
    typer.echo(str(result))
    if not result["controlled"]:
        raise typer.Exit(2)


@app.command("verify")
def verify_command(plan: Path = Path("artifacts/detection/protocol"), output: Path = Path("artifacts/detection")) -> None:
    from flir_pipeline.detection.materialization import verify_view
    from flir_pipeline.detection.protocol import verify_plan

    frozen = verify_plan(plan)
    for split in frozen["identity"]["splits"]:
        result = verify_view(output/"views"/split["split_space_id"])
        if result["split_metadata_sha256"] != split["split_metadata_sha256"]:
            raise typer.BadParameter("View differs from frozen split")
    typer.echo(f"Verified plan and {len(frozen['identity']['splits'])} views; detector training status is separate")
