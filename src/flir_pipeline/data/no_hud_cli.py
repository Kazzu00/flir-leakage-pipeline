"""External-only preparation and verification of no-HUD annotation candidates."""

import json
from pathlib import Path

import typer

app = typer.Typer(
    help="Prepare no-HUD candidates without approving labels or training."
)


@app.command("prepare")
def prepare_command(
    source_root: Path = typer.Option(..., exists=True, file_okay=False),
    audit: Path = typer.Option(..., exists=True, file_okay=False),
    proposal: Path = typer.Option(..., exists=True, file_okay=False),
    archive: list[str] = typer.Option(..., help="Repeat for each no-HUD ZIP basename."),
    output: Path = typer.Option(...),
    evidence: Path | None = typer.Option(None, exists=True, dir_okay=False),
    audit_evidence: Path | None = typer.Option(None, exists=True, dir_okay=False),
    gallery_limit: int = typer.Option(60, min=0),
):
    from flir_pipeline.data.no_hud import prepare

    try:
        result = prepare(
            source_root,
            audit,
            proposal,
            archive,
            output,
            evidence=evidence,
            audit_evidence=audit_evidence,
            gallery_limit=gallery_limit,
        )
    except (ValueError, OSError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(json.dumps(result, indent=2))


@app.command("verify")
def verify_command(
    publication: Path = typer.Option(..., exists=True, file_okay=False),
    replay: bool = typer.Option(
        False, help="Re-read all images and reproduce evidence."
    ),
):
    from flir_pipeline.data.no_hud import verify

    try:
        result = verify(publication, replay=replay)
    except (ValueError, OSError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(json.dumps(result, indent=2))
