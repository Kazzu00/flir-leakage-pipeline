"""Lazy CLI for cross-dataset candidates, source-bound QA and counts."""

import json
from pathlib import Path

import typer

app = typer.Typer(
    help="Candidate-only labeled/video visual linkage; no confirmed matches or splits."
)


@app.command("build")
def build(
    labeled_manifest: Path = typer.Option(...),
    video_manifest: Path = typer.Option(...),
    labeled_clip: Path = typer.Option(...),
    labeled_dinov2: Path = typer.Option(...),
    video_clip: Path = typer.Option(...),
    video_dinov2: Path = typer.Option(...),
    sequence_set: Path = typer.Option(...),
    output: Path = typer.Option(Path("artifacts/linkage")),
    top_k: int = typer.Option(10, min=1),
) -> None:
    """Publish the union of both encoder top-k sets and all occurrence lineage."""
    from flir_pipeline.linkage.base import LinkageConfig
    from flir_pipeline.linkage.sources import InputPaths
    from flir_pipeline.linkage.storage import build_to_store

    paths = InputPaths(
        labeled_manifest,
        video_manifest,
        labeled_clip,
        labeled_dinov2,
        video_clip,
        video_dinov2,
        sequence_set,
    )
    try:
        typer.echo(build_to_store(paths, LinkageConfig(top_k=top_k), output))
    except (
        ValueError,
        OSError,
        KeyError,
        TypeError,
        AssertionError,
        IndexError,
        OverflowError,
    ) as error:
        typer.echo(f"Linkage build failed: {error}", err=True)
        raise typer.Exit(1) from error


@app.command("verify")
def verify(
    directory: Path,
    labeled_manifest: Path = typer.Option(...),
    video_manifest: Path = typer.Option(...),
    labeled_clip: Path = typer.Option(...),
    labeled_dinov2: Path = typer.Option(...),
    video_clip: Path = typer.Option(...),
    video_dinov2: Path = typer.Option(...),
    sequence_set: Path = typer.Option(...),
) -> None:
    """Recompute from all original sources, including every candidate occurrence."""
    from flir_pipeline.linkage.sources import InputPaths
    from flir_pipeline.linkage.storage import verify_directory

    paths = InputPaths(
        labeled_manifest,
        video_manifest,
        labeled_clip,
        labeled_dinov2,
        video_clip,
        video_dinov2,
        sequence_set,
    )
    result = verify_directory(directory, paths)
    typer.echo(json.dumps(result, indent=2))
    if not result["quality_valid"]:
        raise typer.Exit(1)


@app.command("summary")
def summary(directory: Path) -> None:
    """Read candidate counts only; does not perform formal verification."""
    from flir_pipeline.linkage.storage import summarize_directory

    try:
        typer.echo(json.dumps(summarize_directory(directory), indent=2))
    except (ValueError, OSError) as error:
        typer.echo(f"Linkage summary failed: {error}", err=True)
        raise typer.Exit(1) from error
