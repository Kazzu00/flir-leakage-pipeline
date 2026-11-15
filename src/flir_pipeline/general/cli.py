"""General image-dataset audit and immutable dependency split publications."""

import json
from pathlib import Path

import typer
import yaml

from flir_pipeline.general.contract import Manifest
from flir_pipeline.general.ingestion import ingest
from flir_pipeline.general.publication import build, read_json, verify, write_json
from flir_pipeline.splitting.base import SplitConfig

app = typer.Typer(
    help="Dataset-independent evidence and dependency splitting; never trains models."
)


@app.command("ingest")
def ingest_command(config: Path, root: Path, output: Path):
    """Read an explicit YAML list of images and optional original YOLO labels."""
    if output.exists():
        raise typer.BadParameter("Output already exists")
    manifest = ingest(config, root)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, manifest.canonical())
    typer.echo(manifest.dataset_id())


@app.command("build")
def build_command(
    manifest: Path,
    config: Path,
    root: Path,
    output: Path,
    features: list[Path] = typer.Option(
        None, help="Optional content-level L2 NPZ, repeat per feature space."
    ),
):
    """Build both balanced policies, seeds 0–4, diagnostics and observed timeline."""
    cfg = SplitConfig(**yaml.safe_load(config.read_text(encoding="utf-8")))
    typer.echo(
        json.dumps(
            build(
                Manifest.model_validate(read_json(manifest)),
                cfg,
                root,
                output,
                features,
            )
        )
    )


@app.command("verify")
def verify_command(publication: Path, root: Path, report: Path | None = None):
    """Rehash inputs and independently replay all memberships and diagnostics."""
    if report is not None and (
        report.exists() or report.resolve().is_relative_to(publication.resolve())
    ):
        raise typer.BadParameter(
            "Replay report must be a new sibling, outside the publication"
        )
    result = verify(publication, root)
    if report is not None:
        write_json(report, result)
    typer.echo(json.dumps(result))


@app.command("schema")
def schema_command():
    """Print the versioned canonical JSON Schema without private records."""
    typer.echo(json.dumps(Manifest.model_json_schema(), indent=2))


@app.command("localize")
def localize_command(manifest: Path, root: Path, evidence: Path, output: Path):
    """Optional full video decode; exact appearances only, unknown stays unknown."""
    from flir_pipeline.general.localization import localize

    if output.exists():
        raise typer.BadParameter("Output manifest already exists")
    localized = localize(Manifest.model_validate(read_json(manifest)), root, evidence)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, localized.canonical())
    typer.echo(localized.dataset_id())


@app.command("adapt-fac")
def adapt_fac_command(config: Path, root: Path, output: Path, report: Path):
    """Adapt the existing audited FAC case; no private paths are core defaults."""
    from flir_pipeline.general.fac_adapter import adapt

    if output.exists() or report.exists():
        raise typer.BadParameter("Adapter outputs must be new files")
    manifest, audit = adapt(config, root)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, manifest.canonical())
    write_json(report, audit)
    typer.echo(json.dumps(audit))
