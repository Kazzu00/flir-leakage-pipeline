"""Command-line entry point for the research pipeline."""

import json
import os
from pathlib import Path

import typer

app = typer.Typer(help="Reproducible FLIR leakage research pipeline.")
data_app = typer.Typer(help="Read-only dataset discovery and audit commands.")
features_app = typer.Typer(help="Content-level visual feature extraction commands.")
app.add_typer(data_app, name="data")
app.add_typer(features_app, name="features")


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


def _load_yaml_config(path: Path | None) -> dict:
    if path is None:
        return {}
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise typer.BadParameter("Feature config must contain a YAML mapping.")
    return data


def _feature_extractor(name: str, config: dict, device: str | None, local_files_only: bool):
    from flir_pipeline.features.base import DeterministicFakeExtractor

    options = dict(config)
    options.pop("extractor", None)
    pooling_strategy = options.pop("pooling_strategy", None)
    options.pop("feature_type", None)
    options.pop("dtype", None)
    options.pop("store_raw", None)
    options.pop("store_l2_normalized", None)
    if name == "dinov2" and pooling_strategy not in {None, "cls_token"}:
        raise typer.BadParameter("DINOv2 currently supports pooling_strategy=cls_token")
    if device is not None:
        options["device"] = device
    options["local_files_only"] = local_files_only
    if name == "fake":
        return DeterministicFakeExtractor(options.get("embedding_dimension", 8))
    if name == "dinov2":
        from flir_pipeline.features.dinov2 import DinoV2Extractor

        return DinoV2Extractor(**options)
    if name == "clip":
        from flir_pipeline.features.clip import CLIPExtractor

        return CLIPExtractor(**options)
    raise typer.BadParameter("extractor must be fake, dinov2, or clip")


@features_app.command("extract")
def features_extract(
    manifest: Path = typer.Option(..., help="Candidate manifest Parquet."),
    extractor: str = typer.Option("dinov2", help="fake, dinov2, or clip."),
    config: Path | None = typer.Option(None, help="YAML extractor configuration."),
    images_archive: Path | None = typer.Option(None, help="Imagenes.zip path."),
    output_root: Path = typer.Option(Path("artifacts/features"), help="Artifact root."),
    device: str | None = typer.Option(None, help="cpu, cuda, or auto."),
    batch_size: int | None = typer.Option(None, help="Override batch size."),
    limit_content: int | None = typer.Option(None, help="Maximum unique contents."),
    seed: int = typer.Option(0, help="Deterministic content sample seed."),
    local_files_only: bool = typer.Option(False, help="Do not access model downloads."),
) -> None:
    """Extract raw and L2 embeddings once per unique content_id."""
    root = _default_root()
    resolved_archive = images_archive or (root / "Imagenes.zip" if root else None)
    if resolved_archive is None or not resolved_archive.is_file():
        raise typer.BadParameter("Provide --images-archive or set FLIR_DATA_ROOT.")
    settings = _load_yaml_config(config)
    selected_name = settings.pop("extractor", extractor)
    selected_extractor = _feature_extractor(
        selected_name, settings, device, local_files_only
    )
    if batch_size is None:
        batch_size = int(settings.get("batch_size", 8))
    from flir_pipeline.features.storage import extract_to_store

    feature_dir = extract_to_store(
        manifest,
        resolved_archive,
        selected_extractor,
        output_root=output_root,
        batch_size=batch_size,
        limit_content=limit_content,
        seed=seed,
    )
    typer.echo(f"Features written to {feature_dir}")


@features_app.command("diagnostics")
def features_diagnostics(
    manifest: Path = typer.Option(..., help="Candidate manifest Parquet."),
    images_archive: Path | None = typer.Option(None, help="Imagenes.zip path."),
    output_root: Path = typer.Option(
        Path("artifacts/features/diagnostics"), help="Diagnostics artifact root."
    ),
    limit_content: int | None = typer.Option(None, help="Maximum unique contents."),
    seed: int = typer.Option(0, help="Deterministic content sample seed."),
) -> None:
    """Compute independent content-level image quality diagnostics."""
    root = _default_root()
    resolved_archive = images_archive or (root / "Imagenes.zip" if root else None)
    if resolved_archive is None or not resolved_archive.is_file():
        raise typer.BadParameter("Provide --images-archive or set FLIR_DATA_ROOT.")
    from flir_pipeline.features.diagnostics import run_diagnostics

    output = run_diagnostics(
        manifest, resolved_archive, output_root, limit_content=limit_content, seed=seed
    )
    typer.echo(f"Diagnostics written to {output}")


@features_app.command("summary")
def features_summary(feature_directory: Path) -> None:
    """Print metadata and quality for a feature directory."""
    if not feature_directory.is_dir():
        raise typer.BadParameter(f"Feature directory does not exist: {feature_directory}")
    from flir_pipeline.features.storage import summarize_feature_directory

    typer.echo(json.dumps(summarize_feature_directory(feature_directory), indent=2))


@features_app.command("verify")
def features_verify(feature_directory: Path) -> None:
    """Verify arrays and indexes in a feature directory."""
    if not feature_directory.is_dir():
        raise typer.BadParameter(f"Feature directory does not exist: {feature_directory}")
    from flir_pipeline.features.storage import verify_feature_directory

    result = verify_feature_directory(feature_directory)
    typer.echo(json.dumps(result, indent=2))


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
