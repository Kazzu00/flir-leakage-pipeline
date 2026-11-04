"""Local dependency-pair pilot commands."""

import json
from pathlib import Path

import typer

app = typer.Typer(help="Independent audiovisual continuity and spatial pair review.")


@app.command("prepare")
def prepare_command(
    pilot: Path,
    reviews: Path,
    closure: Path,
    root: Path,
    output: Path,
    seed: int = 20261103,
):
    from .dependency import prepare

    data = prepare(pilot, reviews, closure, root, output, seed)
    typer.echo(
        json.dumps(
            {
                "pairs": len(data["pairs"]),
                "counts": data["counts"],
                "deficits": data["deficits"],
            }
        )
    )


@app.command("serve")
def serve(
    pilot: Path,
    db: Path,
    role: str = "reviewer1",
    port: int = 8767,
    reviewer1_db: Path | None = None,
    reviewer2_db: Path | None = None,
):
    from .dependency_store import DependencyStore
    from .pilot_server import make_server

    sources = [reviewer1_db, reviewer2_db] if role == "adjudicator" else ()
    server = make_server(
        DependencyStore(pilot, db, role, sources), port, "dependency_ui.html"
    )
    typer.echo(f"Dependency reviewer ({role}): http://127.0.0.1:{server.server_port}")
    try:
        server.serve_forever()
    finally:
        server.server_close()


@app.command("report")
def report(
    pilot: Path,
    reviewer1_db: Path,
    reviewer2_db: Path | None = None,
    adjudication_db: Path | None = None,
):
    from .dependency import load
    from .dependency_store import read_ledger, summarize
    from .inventory import file_sha

    data = load(pilot)
    binding = file_sha(pilot / "pilot.json")
    rows = []
    for path, role in (
        (reviewer1_db, "reviewer1"),
        (reviewer2_db, "reviewer2"),
        (adjudication_db, "adjudicator"),
    ):
        if path:
            rows.extend(read_ledger(path, binding, role))
    typer.echo(json.dumps(summarize(data, rows), indent=2))


@app.command("export")
def export(
    pilot: Path, db: Path, output: Path, role: str = "reviewer1", format: str = "json"
):
    from .dependency_store import DependencyStore

    data = DependencyStore(pilot, db, role).export(format)
    if output.exists():
        raise typer.BadParameter("Use a new export path")
    output.write_bytes(data)
