"""Controlled detector commands; optional runtime loads only on execution."""

from pathlib import Path

import typer

app = typer.Typer(help="Frozen historical/random/cluster detector comparison.")


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
