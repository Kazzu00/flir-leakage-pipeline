"""Preparation and local service for human alignment validation."""

import json
from pathlib import Path

import typer

app = typer.Typer(help="Human alignment pilot; historical ledgers remain untouched.")


@app.command("prepare")
def prepare_command(root: Path, evidence: Path, output: Path, seed: int = 20261103):
    from .pilot import prepare

    data = prepare(root, evidence, output, seed)
    typer.echo(
        json.dumps(
            {
                "cases": len(data["cases"]),
                "by_split": data["summary"],
                "media": len(data["media"]),
            }
        )
    )


@app.command("serve")
def serve(pilot: Path, db: Path, port: int = 8766):
    from .pilot_server import make_server
    from .pilot_store import PilotStore

    server = make_server(PilotStore(pilot, db), port)
    typer.echo(f"Alignment reviewer: http://127.0.0.1:{server.server_port}")
    try:
        server.serve_forever()
    finally:
        server.server_close()


@app.command("report")
def report(pilot: Path, db: Path, minimum: int = 5):
    from .pilot_store import PilotStore

    typer.echo(json.dumps(PilotStore(pilot, db).report(minimum), indent=2))


@app.command("export")
def export(pilot: Path, db: Path, output: Path, format: str = "json"):
    from .pilot_store import PilotStore

    if format not in {"json", "csv"}:
        raise typer.BadParameter("Use json or csv")
    data = PilotStore(pilot, db).export(format)
    if output.exists() and output.read_bytes() != data:
        raise typer.BadParameter("Export path already contains different bytes")
    output.write_bytes(data)
