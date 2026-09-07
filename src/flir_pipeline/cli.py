"""Command-line entry point for the research pipeline."""

import json
import os
from pathlib import Path

import typer

app = typer.Typer(help="Reproducible FLIR leakage research pipeline.")
data_app = typer.Typer(help="Read-only dataset discovery and audit commands.")
app.add_typer(data_app, name="data")


def _placeholder(name: str) -> None:
    typer.echo(f"{name} is reserved for a future pipeline stage.")


def _default_root() -> Path | None:
    value = os.getenv("FLIR_DATA_ROOT", "").strip()
    if not value:
        env_path = Path(".env")
        if env_path.is_file():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("FLIR_DATA_ROOT="):
                    value = line.partition("=")[2].strip().strip('"\'')
                    break
    return Path(value) if value else None


@data_app.command("inventory")
def inventory(
    root: Path | None = typer.Option(None, help="External FLIR_DATA_ROOT."),
    output: Path = typer.Option(
        Path("reports/data_inventory"), help="Report directory."
    ),
    inspect_archives: bool = typer.Option(
        True, help="Inspect ZIP members without extraction."
    ),
    hash_members: bool = typer.Option(
        False, help="Hash image and label members by streaming."
    ),
) -> None:
    """Inventory files and inspect ZIP archives without modifying the data root."""
    resolved_root = root or _default_root()
    if resolved_root is None:
        raise typer.BadParameter(
            "Provide --root or set FLIR_DATA_ROOT.", param_hint="--root"
        )
    if not resolved_root.is_dir():
        raise typer.BadParameter(f"Data root does not exist: {resolved_root}")
    from flir_pipeline.data.inventory import run_inventory

    summary = run_inventory(resolved_root, output, inspect_archives, hash_members)
    typer.echo(
        f"Inventory written to {output}: {summary['file_count']} files, {summary['archive_count']} archives."
    )


@data_app.command("archive-tree")
def archive_tree(archive_path: Path) -> None:
    """Print a normalized member tree for one ZIP without extracting it."""
    if not archive_path.is_file():
        raise typer.BadParameter(f"Archive does not exist: {archive_path}")
    from flir_pipeline.data.inventory import archive_tree as build_archive_tree

    for member in build_archive_tree(archive_path):
        typer.echo(
            f"{member['member_path']} [{member['file_type']}] {member['uncompressed_size']} bytes"
        )


@data_app.command("compare-archives")
def compare_archives(
    root: Path | None = typer.Option(None, help="External FLIR_DATA_ROOT."),
    output: Path = typer.Option(
        Path("reports/data_inventory"), help="Report directory."
    ),
    hash_members: bool = typer.Option(
        True, help="Hash comparable image and label members."
    ),
) -> None:
    """Compare archive content and write lineage reports without extraction."""
    resolved_root = root or _default_root()
    if resolved_root is None:
        raise typer.BadParameter(
            "Provide --root or set FLIR_DATA_ROOT.", param_hint="--root"
        )
    if not resolved_root.is_dir():
        raise typer.BadParameter(f"Data root does not exist: {resolved_root}")
    from flir_pipeline.data.inventory import run_inventory

    summary = run_inventory(
        resolved_root, output, inspect_archives=True, hash_members=hash_members
    )
    typer.echo(
        f"Archive comparison written to {output}; hashed_members={summary['hash_members']}."
    )


@data_app.command("build-manifest")
def build_manifest_command(
    images_archive: Path | None = typer.Option(None, help="Imagenes.zip path."),
    labels_archive: Path | None = typer.Option(None, help="Etiquetas.zip path."),
    output: Path = typer.Option(
        Path("data/manifests/flir_canonical_candidate_v1.parquet"),
        help="Local manifest output.",
    ),
    report_output: Path = typer.Option(
        Path("reports/data_manifest"), help="Local report directory."
    ),
) -> None:
    """Build the candidate canonical manifest without extracting source ZIPs."""
    root = _default_root()
    resolved_images = images_archive or (root / "Imagenes.zip" if root else None)
    resolved_labels = labels_archive or (root / "Etiquetas.zip" if root else None)
    if resolved_images is None or resolved_labels is None:
        raise typer.BadParameter(
            "Provide --images-archive and --labels-archive or set FLIR_DATA_ROOT."
        )
    if not resolved_images.is_file() or not resolved_labels.is_file():
        raise typer.BadParameter("Both source archives must exist.")
    from flir_pipeline.data.manifest import build_manifest

    summary = build_manifest(
        resolved_images,
        resolved_labels,
        output,
        report_output,
        comparison_root=resolved_images.parent,
    )
    typer.echo(
        f"Manifest written to {output}: {summary['total_records']} records, "
        f"{summary['unique_content_ids']} unique contents."
    )


@data_app.command("validate-labels")
def validate_labels_command(
    labels_archive: Path | None = typer.Option(None, help="Etiquetas.zip path."),
    output: Path = typer.Option(
        Path("reports/data_manifest"), help="Local validation report directory."
    ),
) -> None:
    """Validate labels from a ZIP without modifying or extracting them."""
    root = _default_root()
    resolved_labels = labels_archive or (root / "Etiquetas.zip" if root else None)
    if resolved_labels is None or not resolved_labels.is_file():
        raise typer.BadParameter(
            "Provide --labels-archive or set FLIR_DATA_ROOT to a valid root."
        )
    from flir_pipeline.data.manifest import validate_labels_archive

    summary = validate_labels_archive(resolved_labels, output)
    typer.echo(
        f"Labels validated: {summary['total_labels']} total, "
        f"{summary['valid_labels']} valid, {summary['invalid_labels']} invalid."
    )


@data_app.command("manifest-summary")
def manifest_summary_command(manifest: Path) -> None:
    """Print a compact summary of an existing local Parquet manifest."""
    if not manifest.is_file():
        raise typer.BadParameter(f"Manifest does not exist: {manifest}")
    from flir_pipeline.data.manifest import manifest_summary

    typer.echo(json.dumps(manifest_summary(manifest), indent=2))


@app.command()
def features() -> None:
    """Visual representation commands."""
    _placeholder("features")


@app.command()
def similarity() -> None:
    """Similarity analysis commands."""
    _placeholder("similarity")


@app.command("cluster")
def cluster() -> None:
    """Clustering commands."""
    _placeholder("cluster")


@app.command()
def split() -> None:
    """Dataset split commands."""
    _placeholder("split")


@app.command()
def detect() -> None:
    """Detection commands."""
    _placeholder("detect")


@app.command()
def evaluate() -> None:
    """Evaluation commands."""
    _placeholder("evaluate")
