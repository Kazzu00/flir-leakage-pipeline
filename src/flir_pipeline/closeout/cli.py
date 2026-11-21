"""Closeout commands deliberately separate preparation from GPU execution."""

import json
from pathlib import Path

import typer

app = typer.Typer(help="Denoised data audit and reproducible OE4 closeout.")


@app.command("audit-denoised-anchors")
def audit_denoised_anchors(dataset: Path, publication: Path, output: Path):
    from flir_pipeline.closeout.anchors import audit

    typer.echo(json.dumps(audit(dataset, publication, output), indent=2))


@app.command("import-denoised")
def import_denoised(
    source: Path, output: Path, expected_count: int = 1116, deep_verify: bool = False
):
    from flir_pipeline.closeout.denoised import prepare

    report = prepare(source, output, expected_count, deep_verify)
    typer.echo(json.dumps(report, indent=2, ensure_ascii=False))


@app.command("verify")
def verify_artifact(directory: Path):
    from flir_pipeline.closeout.io import verify

    result = verify(directory)
    typer.echo(
        json.dumps({"state": result["state"], "verified_files": len(result["files"])})
    )


@app.command("build-denoised-splits")
def build_splits(
    dataset: Path,
    config: Path,
    output: Path,
    dinov2: Path | None = None,
    clip: Path | None = None,
):
    from flir_pipeline.closeout.splits import build

    if (dinov2 is None) != (clip is None):
        raise typer.BadParameter("Supply both encoder stores")
    report = build(
        dataset, config, output, {"dinov2": dinov2, "clip": clip} if dinov2 else None
    )
    typer.echo(json.dumps(report, indent=2))
    if report["state"] == "BLOCKED":
        raise typer.Exit(2)


@app.command("denoised-features")
def denoised_features(
    dataset: Path, output: Path, device: str = "cpu", batch_size: int = 8
):
    from flir_pipeline.closeout.features import extract

    typer.echo(json.dumps(extract(dataset, output, device, batch_size), indent=2))


@app.command("materialize-test")
def materialize_test(publication: Path, source_root: Path, output: Path):
    from flir_pipeline.closeout.common_test import materialize

    typer.echo(json.dumps(materialize(publication, source_root, output), indent=2))


@app.command("oe4-preflight")
def oe4_preflight(config: Path):
    from flir_pipeline.closeout.oe4 import preflight

    result = preflight(config)
    typer.echo(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "state",
                    "blockers",
                    "gpu_checked",
                    "scientific_authorization",
                )
            },
            indent=2,
        )
    )
    if result["state"] != "READY":
        raise typer.Exit(2)


@app.command("oe4-train")
def oe4_train(config: Path, index: int):
    from flir_pipeline.closeout.oe4_runtime import train

    result = train(config, index)
    typer.echo(json.dumps({"state": result["state"]}))


@app.command("oe4-evaluate")
def oe4_evaluate(config: Path, index: int):
    from flir_pipeline.closeout.oe4_runtime import evaluate_policy

    typer.echo(json.dumps(evaluate_policy(config, index)))


@app.command("oe4-aggregate")
def oe4_aggregate(config: Path):
    from flir_pipeline.closeout.oe4_runtime import aggregate

    result = aggregate(config)
    typer.echo(json.dumps({"state": result["state"]}))
