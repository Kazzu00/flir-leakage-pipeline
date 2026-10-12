"""CLI for audited external image variants; no feature extraction or alignment fit."""

import json
from pathlib import Path

import typer

app = typer.Typer(
    help="Immutable external video-variant ingestion; candidate-only alignment."
)


@app.command("build")
def build(
    config: Path = typer.Option(
        ..., help="Explicit input/series/variant YAML configuration."
    ),
    input_root: Path | None = typer.Option(
        None, help="Read-only input root; defaults to FLIR_DATA_ROOT/.env."
    ),
    output_root: Path = typer.Option(
        Path("artifacts/video_variants"), help="Separate ignored publication root."
    ),
    ffprobe_bin: str | None = typer.Option(
        None, help="Opt into video metadata inspection; no frame decoding/alignment."
    ),
):
    from flir_pipeline.cli import _default_root
    from flir_pipeline.data.video_variant_storage import (
        build_ingestion,
        summarize_ingestion,
    )

    root = input_root or _default_root()
    if root is None:
        raise typer.BadParameter("Provide --input-root or set FLIR_DATA_ROOT.")
    try:
        destination = build_ingestion(
            config, root, output_root, ffprobe_bin=ffprobe_bin
        )
        summary = summarize_ingestion(destination)
    except (ValueError, OSError, KeyError, TypeError) as error:
        typer.echo(f"Video variant ingestion failed: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo(json.dumps({"artifact": str(destination), **summary}, indent=2))


@app.command("verify")
def verify(
    artifact: Path,
    input_root: Path | None = typer.Option(
        None,
        help="Explicit root for full original checksum/image replay; omitted means stored evidence only.",
    ),
):
    from flir_pipeline.data.video_variant_storage import verify_ingestion

    try:
        result = verify_ingestion(artifact, input_root)
    except (ValueError, OSError, KeyError, TypeError) as error:
        typer.echo(f"Video variant verification failed: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo(json.dumps(result, indent=2))


@app.command("summary")
def summary(artifact: Path):
    from flir_pipeline.data.video_variant_storage import summarize_ingestion

    try:
        result = summarize_ingestion(artifact)
    except (ValueError, OSError, KeyError, TypeError) as error:
        typer.echo(f"Video variant summary failed: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo(json.dumps(result, indent=2))
