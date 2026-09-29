"""Lazy CLI for candidates, explicit manual commitment and source-bound QA."""

import json
from pathlib import Path

import typer

from flir_pipeline.sequences.experiments.cli import app as experiment_app

app = typer.Typer(
    help="Occurrence-level candidates, reviewed sequences and exact-copy dependencies; no splits."
)
app.add_typer(experiment_app, name="experiment")


@app.command("detect")
def detect(
    manifest: Path = typer.Option(...),
    clip_features: Path = typer.Option(...),
    dinov2_features: Path = typer.Option(...),
    config: Path = typer.Option(Path("configs/sequences/research.yaml")),
    output: Path = typer.Option(Path("artifacts/sequences")),
) -> None:
    """Publish diagnostic boundary candidates using both original L2 feature stores."""
    from flir_pipeline.sequences.base import SequenceConfig
    from flir_pipeline.sequences.storage import detect_to_store

    try:
        typer.echo(
            detect_to_store(
                manifest,
                clip_features,
                dinov2_features,
                SequenceConfig.load(config),
                output,
            )
        )
    except (OSError, ValueError, KeyError, AssertionError) as error:
        typer.echo(f"Sequence detection failed: {error}", err=True)
        raise typer.Exit(1) from error


@app.command("build")
def build(
    detection: Path = typer.Option(...),
    manifest: Path = typer.Option(...),
    clip_features: Path = typer.Option(...),
    dinov2_features: Path = typer.Option(...),
    validation: Path = typer.Option(
        ..., help="Frozen confirmed manual validation directory."
    ),
    output: Path = typer.Option(Path("artifacts/sequences")),
) -> None:
    """Commit only manually accepted localized cuts; preserve all occurrences."""
    from flir_pipeline.sequences.storage import build_to_store

    try:
        typer.echo(
            build_to_store(
                detection, manifest, clip_features, dinov2_features, validation, output
            )
        )
    except (OSError, ValueError, KeyError, AssertionError) as error:
        typer.echo(f"Sequence build failed: {error}", err=True)
        raise typer.Exit(1) from error


@app.command("verify")
def verify(
    directory: Path,
    manifest: Path = typer.Option(...),
    clip_features: Path = typer.Option(...),
    dinov2_features: Path = typer.Option(...),
    validation: Path | None = typer.Option(None),
) -> None:
    """Reconstruct from all sources; committed sets also require confirmed validation."""
    from flir_pipeline.sequences.storage import verify_directory

    result = verify_directory(
        directory, manifest, clip_features, dinov2_features, validation
    )
    typer.echo(json.dumps(result, indent=2))
    if not result["quality_valid"]:
        raise typer.Exit(1)


@app.command("summary")
def summary(directory: Path) -> None:
    """Read aggregate metadata only; this does not verify scientific artifacts."""
    from flir_pipeline.sequences.storage import summarize_directory

    typer.echo(json.dumps(summarize_directory(directory), indent=2))
