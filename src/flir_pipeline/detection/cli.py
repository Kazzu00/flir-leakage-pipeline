"""Controlled detector commands; optional runtime loads only on execution."""

from pathlib import Path

import typer

app = typer.Typer(help="Frozen historical/random/cluster detector comparison.")


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
