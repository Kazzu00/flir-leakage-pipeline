"""Small CLI; local report is the review surface, no frontend expansion required."""

import json
from pathlib import Path

import typer

from .pilot_cli import app as pilot_app

app = typer.Typer(help="Bounded multisplit visual provenance alignment; CPU only.")
app.add_typer(pilot_app, name="pilot")


@app.command("run")
def run_command(
    root: Path = typer.Option(..., exists=True),
    video: str = typer.Option(..., help="Path relative to root."),
    output: Path = typer.Option(...),
    config: Path | None = typer.Option(None),
    audit_db: Path | None = typer.Option(None),
    sample_only: bool = False,
    dry_run: bool = False,
):
    from .models import Config
    from .pipeline import run

    value = Config.model_validate_json(config.read_bytes()) if config else Config()
    result = run(root, video, output, value, sample_only or dry_run, audit_db)
    typer.echo(json.dumps(result, indent=2))


@app.command("verify")
def verify_command(output: Path):
    from .pipeline import verify

    data = verify(output)
    typer.echo(
        json.dumps(
            {"valid": True, "run_id": data["run_id"], "cases": len(data["cases"])}
        )
    )


@app.command("review")
def review_command(output: Path, decision: Path, destination: Path):
    """Record an explicit human-authored JSON decision in a separate revision file."""
    from .pipeline import record_review

    typer.echo(
        json.dumps(
            record_review(output, json.loads(decision.read_bytes()), destination)
        )
    )
