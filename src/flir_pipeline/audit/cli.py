"""Local audit commands; no scientific stage is run or promoted."""

import json
import tempfile
from pathlib import Path

import typer

from .store import Store, atomic_export

app = typer.Typer(
    help="Import exploratory evidence and review locally; no sequences or splits."
)


def show(value):
    typer.echo(json.dumps(value, ensure_ascii=True, indent=2))


@app.command("validate")
def validate(root: Path, priority_only: bool = False):
    """Validate inputs in a disposable external ledger, without changing originals."""
    with tempfile.TemporaryDirectory(prefix="flir-audit-validation-") as temp:
        result = Store(Path(temp) / "audit.sqlite3").import_directory(
            root, priority_only
        )
        show(result)
    if any(f["status"] == "error" for f in result["files"]):
        raise typer.Exit(1)


@app.command("import")
def import_artifacts(
    root: Path, db: Path = typer.Option(...), priority_only: bool = False
):
    if db.resolve().is_relative_to(root.resolve()):
        raise typer.BadParameter(
            "Database must be outside the original input directory"
        )
    result = Store(db).import_directory(root, priority_only)
    show(result)
    if any(f["status"] == "error" for f in result["files"]):
        raise typer.Exit(1)


@app.command("summary")
def summary(db: Path = typer.Option(..., exists=True)):
    data = Store(db).snapshot()
    show(
        {
            "schema": data["schema"],
            "assets": len(data["assets"]),
            "events": len(data["events"]),
            "observations": data["observation_count"],
            "decision_revisions": len(data["decisions"]),
            "last_import": data["last_import"],
        }
    )


@app.command("verify")
def verify(db: Path = typer.Option(..., exists=True), originals: bool = False):
    result = Store(db).verify(originals)
    show(result)
    if not result["valid"]:
        raise typer.Exit(1)


@app.command("export")
def export(
    output: Path, db: Path = typer.Option(..., exists=True), format: str = "json"
):
    if output.resolve() == db.resolve():
        raise typer.BadParameter("Export cannot overwrite the ledger")
    atomic_export(output, Store(db).export(format))


@app.command("import-decisions")
def import_decisions(source: Path, db: Path = typer.Option(..., exists=True)):
    show({"revisions": Store(db).import_decisions(source)})


@app.command("serve")
def serve(
    db: Path = typer.Option(..., exists=True),
    port: int = 8765,
    frontend_origin: str = "http://127.0.0.1:5173",
):
    from .server import make_server

    store = Store(db)
    if not store.verify()["valid"]:
        raise typer.BadParameter("Ledger verification failed")
    server = make_server(store, port, frontend_origin)
    typer.echo(
        f"Local audit API: http://127.0.0.1:{server.server_port}; frontend: {frontend_origin}/organization/sequences?mode=audit"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


@app.command("demo")
def demo(output: Path):
    """Create a new SYNTHETIC smoke workspace; refuses to overwrite existing inputs."""
    from .demo import create_demo

    create_demo(output / "inputs")
    show(Store(output / "audit.sqlite3").import_directory(output / "inputs"))


@app.command("attach-evidence")
def attach_evidence(
    event_id: str,
    image: Path,
    db: Path = typer.Option(..., exists=True),
    locator: str = typer.Option(...),
    basis: str = typer.Option(...),
    rationale: str = typer.Option(...),
    reviewer: str = typer.Option(...),
):
    """Freeze one existing image and an explicit attributable event association."""
    show(
        Store(db).attach_evidence(
            event_id,
            image,
            locator=locator,
            basis=basis,
            rationale=rationale,
            reviewer=reviewer,
        )
    )


@app.command("reconcile-companions")
def reconcile_companions(
    canonical_asset_id: str,
    companion_asset_id: str,
    db: Path = typer.Option(..., exists=True),
    reviewer: str = typer.Option(...),
    rationale: str = typer.Option(...),
):
    """Explicitly reconcile a legacy duplicate pair; retain all records and reviews."""
    show(
        Store(db).reconcile_companions(
            canonical_asset_id, companion_asset_id, reviewer, rationale
        )
    )
