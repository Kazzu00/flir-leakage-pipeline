"""Command-line entry point for the research pipeline."""

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer

from flir_pipeline.detection.cli import app as detection_app
from flir_pipeline.explorer.cli import app as explorer_app
from flir_pipeline.splitting.cli import app as splitting_app

if TYPE_CHECKING:
    from flir_pipeline.features.base import FeatureExtractor
app = typer.Typer(help="Reproducible FLIR leakage research pipeline.")
data_app = typer.Typer(help="Read-only dataset discovery and audit commands.")
features_app = typer.Typer(help="Content-level visual feature extraction commands.")
similarity_app = typer.Typer(help="Content-level cosine and post-hoc temporal/historical analysis.")
reduction_app = typer.Typer(help="Reproducible t-SNE/PaCMAP, preservation metrics and seed stability.")
clustering_app = typer.Typer(help="Content-level DBSCAN/OPTICS/HDBSCAN screening and noise-aware evaluation.")
app.add_typer(data_app, name="data")
app.add_typer(features_app, name="features")
app.add_typer(similarity_app, name="similarity")
app.add_typer(reduction_app, name="reduction")
app.add_typer(clustering_app, name="clustering")
app.add_typer(splitting_app, name="splitting")
app.add_typer(detection_app, name="detection")
app.add_typer(explorer_app, name="explorer")


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
        Path("reports/data_manifest_validation"), help="Archive-wide validation directory, separate from candidate reports."
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


def _feature_extractor(
    name: str, config: dict[str, Any], device: str | None, local_files_only: bool
) -> "FeatureExtractor":
    from flir_pipeline.features.base import DeterministicFakeExtractor

    options = dict(config)
    options.pop("extractor", None)
    pooling_strategy = options.pop("pooling_strategy", None)
    feature_type = options.pop("feature_type", None)
    if feature_type not in {None, "image_embedding"}:
        raise typer.BadParameter("Only feature_type=image_embedding is supported")
    if options.pop("dtype", "float32") != "float32":
        raise typer.BadParameter("Only dtype=float32 storage is supported")
    for flag in ("store_raw", "store_l2_normalized"):
        if options.pop(flag, True) is not True:
            raise typer.BadParameter(f"{flag} must be true: raw and L2 arrays are both required")
    if name == "dinov2" and pooling_strategy not in {None, "cls_token"}:
        raise typer.BadParameter("DINOv2 currently supports pooling_strategy=cls_token")
    if name == "clip" and pooling_strategy not in {None, "projected_pooler_output"}:
        raise typer.BadParameter("CLIP currently supports pooling_strategy=projected_pooler_output")
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
    batch_size: int | None = typer.Option(None, min=1, help="Override batch size."),
    limit_content: int | None = typer.Option(None, min=1, help="Maximum unique contents."),
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
    if batch_size is not None:
        settings["batch_size"] = batch_size
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
def features_verify(
    feature_directory: Path,
    manifest: Path | None = typer.Option(None, help="Also verify full coverage of this canonical manifest."),
) -> None:
    """Verify arrays and indexes in a feature directory."""
    if not feature_directory.is_dir():
        raise typer.BadParameter(f"Feature directory does not exist: {feature_directory}")
    from flir_pipeline.features.storage import (
        verify_feature_directory,
        verify_features_against_manifest,
    )

    if manifest is None:
        result = verify_feature_directory(feature_directory)
    else:
        import pandas as pd

        result = verify_features_against_manifest(feature_directory, pd.read_parquet(manifest))
    typer.echo(json.dumps(result, indent=2))
    if not result.get("reproducible_full_dataset_valid", result["quality_valid"]):
        raise typer.Exit(code=1)


@features_app.command("visualize-data")
def features_visualize_data(
    manifest: Path = typer.Option(..., help="Candidate manifest Parquet."),
    diagnostics: Path = typer.Option(..., help="Diagnostics Parquet."),
    output: Path = typer.Option(Path("reports/feature_engineering"), help="Output directory."),
) -> None:
    """Generate descriptive visualizations for dataset composition and diagnostics."""
    if not manifest.is_file():
        raise typer.BadParameter(f"Manifest does not exist: {manifest}")
    if not diagnostics.is_file():
        raise typer.BadParameter(f"Diagnostics file does not exist: {diagnostics}")
    from flir_pipeline.features.visualization import generate_feature_engineering_report

    result = generate_feature_engineering_report(manifest, diagnostics, output)
    typer.echo(json.dumps(result["metadata"], indent=2))


@features_app.command("visualize-embeddings")
def features_visualize_embeddings(
    feature_directory: Path = typer.Option(..., help="Feature directory to visualize."),
    output: Path = typer.Option(Path("reports/feature_engineering"), help="Output directory."),
    label: str = typer.Option("feature", help="Label used in file names."),
) -> None:
    """Generate embedding health visualizations for a stored feature directory."""
    if not feature_directory.is_dir():
        raise typer.BadParameter(f"Feature directory does not exist: {feature_directory}")
    from flir_pipeline.features.visualization import visualize_embedding_health

    result = visualize_embedding_health(feature_directory, output, label=label)
    typer.echo(json.dumps(result["stats"], indent=2))


@similarity_app.command("compute")
def similarity_compute(
    feature_directory: Path = typer.Option(..., help="Explicit complete feature directory."),
    manifest: Path = typer.Option(..., help="Canonical manifest, for coverage and post-hoc provenance."),
    config: Path | None = typer.Option(None, help="Similarity YAML; defaults to cosine/top-20."),
    output_root: Path = typer.Option(Path("artifacts/similarity"), help="Local ignored artifact root."),
) -> None:
    """Compute cosine once per content, then analyze temporal and historical metadata."""
    from flir_pipeline.similarity.storage import SimilarityConfig, compute_to_store

    settings = SimilarityConfig(**_load_yaml_config(config))
    output = compute_to_store(feature_directory, manifest, settings, output_root)
    typer.echo(f"Similarity written to {output}")


@similarity_app.command("summary")
def similarity_summary(similarity_directory: Path) -> None:
    """Read an executed similarity summary after verifying stored artifacts."""
    from flir_pipeline.similarity.storage import read_json, verify_similarity_directory

    quality = verify_similarity_directory(similarity_directory)
    if not quality["quality_valid"]:
        typer.echo(json.dumps(quality, indent=2))
        raise typer.Exit(1)
    metadata = read_json(similarity_directory/"metadata.json")
    typer.echo(json.dumps({"experiment": {key: metadata[key] for key in ("extractor", "similarity_space_id", "feature_space_id", "content_count", "top_k")},
                           "summary": read_json(similarity_directory/"similarity_summary.json")}, indent=2))


@similarity_app.command("verify")
def similarity_verify(
    similarity_directory: Path,
    feature_directory: Path | None = typer.Option(None, help="Additionally verify against original L2 embeddings and indexes."),
    manifest: Path | None = typer.Option(None, help="Additionally verify against canonical occurrence provenance."),
) -> None:
    """Verify matrix, IDs, ranks, provenance, metadata and artifact fingerprints."""
    from flir_pipeline.similarity.storage import verify_similarity_directory

    result = verify_similarity_directory(similarity_directory, feature_directory, manifest)
    typer.echo(json.dumps(result, indent=2))
    if not result["quality_valid"]:
        raise typer.Exit(1)


@similarity_app.command("compare")
def similarity_compare(
    left: Path = typer.Option(..., help="Left verified similarity directory."),
    right: Path = typer.Option(..., help="Right verified similarity directory."),
    output_root: Path = typer.Option(Path("artifacts/similarity/comparisons")),
) -> None:
    """Compare neighbor sets at k=1/5/10/20 without merging embeddings."""
    from flir_pipeline.similarity.comparison import compare_to_store

    output = compare_to_store(left, right, output_root)
    typer.echo(f"Neighbor agreement written to {output}")


@reduction_app.command("run")
def reduction_run(
    feature_directory: Path = typer.Option(...),
    similarity_directory: Path = typer.Option(...),
    manifest: Path = typer.Option(...),
    config: Path = typer.Option(..., help="Reduction YAML; choose one configuration and seed below."),
    configuration_index: int = typer.Option(0, min=0, help="Zero-based position among hyperparameter configurations, excluding seeds."),
    seed: int = typer.Option(0, min=0),
    output_root: Path = typer.Option(Path("artifacts/reduction")),
) -> None:
    """Run one explicitly chosen reduction, without posterior metadata as features."""
    from dataclasses import replace

    from flir_pipeline.reduction.base import configuration_id, load_grid
    from flir_pipeline.reduction.storage import load_inputs, run_to_store

    grid = {configuration_id(c): c for c in load_grid(config)}
    if configuration_index >= len(grid):
        raise typer.BadParameter("configuration-index is outside the YAML grid")
    selected = replace(list(grid.values())[configuration_index], seed=seed)
    inputs = load_inputs(feature_directory, similarity_directory, manifest)
    typer.echo(f"Reduction written to {run_to_store(inputs, selected, output_root)}")


@reduction_app.command("benchmark")
def reduction_benchmark(
    feature_directory: Path = typer.Option(...),
    similarity_directory: Path = typer.Option(...),
    manifest: Path = typer.Option(...),
    config: list[Path] = typer.Option(..., help="Repeat --config for the t-SNE and PaCMAP grids."),
    output_root: Path = typer.Option(Path("artifacts/reduction")),
) -> None:
    """Execute a bounded grid and select exploratory references from measured criteria."""
    from flir_pipeline.reduction.base import load_grid
    from flir_pipeline.reduction.benchmark import benchmark_to_store
    from flir_pipeline.reduction.storage import load_inputs

    inputs = load_inputs(feature_directory, similarity_directory, manifest)
    grid = [configuration for path in config for configuration in load_grid(path)]
    typer.echo(f"Benchmark written to {benchmark_to_store(inputs, grid, output_root)}")


@reduction_app.command("verify")
def reduction_verify(
    directory: Path,
    feature_directory: Path | None = typer.Option(None),
    similarity_directory: Path | None = typer.Option(None),
    manifest: Path | None = typer.Option(None),
) -> None:
    """Verify a run or benchmark; all three source options also recompute preservation metrics."""
    from flir_pipeline.reduction.benchmark import verify_benchmark
    from flir_pipeline.reduction.storage import load_inputs, verify_reduction
    from flir_pipeline.similarity.storage import read_json

    options = (feature_directory, similarity_directory, manifest)
    if any(p is not None for p in options) and not all(p is not None for p in options):
        raise typer.BadParameter("Provide feature-directory, similarity-directory and manifest together")
    inputs = load_inputs(*options) if all(p is not None for p in options) else None
    kind = read_json(directory/"metadata.json").get("artifact_kind") if (directory/"metadata.json").is_file() else None
    result = verify_benchmark(directory, inputs) if kind == "reduction_benchmark" else verify_reduction(directory, inputs)
    typer.echo(json.dumps(result, indent=2))
    if not result["quality_valid"]:
        raise typer.Exit(1)


@reduction_app.command("summary")
def reduction_summary(directory: Path) -> None:
    """Read completed and verified reduction metrics or benchmark aggregates."""
    from flir_pipeline.reduction.benchmark import verify_benchmark
    from flir_pipeline.reduction.storage import verify_reduction
    from flir_pipeline.similarity.storage import read_json

    meta = read_json(directory/"metadata.json")
    is_benchmark = meta.get("artifact_kind") == "reduction_benchmark"
    quality = verify_benchmark(directory) if is_benchmark else verify_reduction(directory)
    if not quality["quality_valid"]:
        typer.echo(json.dumps(quality, indent=2))
        raise typer.Exit(1)
    payload = read_json(directory/("summary.json" if is_benchmark else "metrics.json"))
    typer.echo(json.dumps({"method": meta.get("method"), "experiment_id": meta.get("benchmark_id", meta.get("reduction_space_id")), "summary": payload}, indent=2))


@clustering_app.command("run")
def clustering_run(
    inputs: Path = typer.Option(..., help="Local YAML with manifest and exact source directories."),
    encoder: str = typer.Option(...),
    representation: str = typer.Option(..., help="original_l2, tsne or pacmap."),
    config: Path = typer.Option(...),
    configuration_index: int = typer.Option(0, min=0),
    reduction_seed: int = typer.Option(0, min=0, max=2),
    output_root: Path = typer.Option(Path("artifacts/clustering")),
) -> None:
    """Fit one conceptual configuration on a verified content representation."""
    from flir_pipeline.clustering.base import load_grid
    from flir_pipeline.clustering.experiments import load_families
    from flir_pipeline.clustering.storage import run_to_store

    families, configs = load_families(inputs), load_grid(config)
    if encoder not in families or representation not in ("original_l2", "tsne", "pacmap") or configuration_index >= len(configs):
        raise typer.BadParameter("Unknown encoder, representation or configuration index")
    family = families[encoder]
    space = family.spaces[(representation, None if representation == "original_l2" else reduction_seed)]
    typer.echo(f"Clustering written to {run_to_store(family, space, configs[configuration_index], output_root)}")


@clustering_app.command("sweep")
def clustering_sweep(
    inputs: Path = typer.Option(...),
    config: list[Path] = typer.Option(..., help="Repeat for the three algorithm YAML grids."),
    output_root: Path = typer.Option(Path("artifacts/clustering")),
) -> None:
    """Stage A: screen original L2 and seed-0 reduction candidates, then shortlist."""
    from flir_pipeline.clustering.base import load_grid
    from flir_pipeline.clustering.experiments import load_families, screening_to_store

    output = screening_to_store(load_families(inputs), [c for path in config for c in load_grid(path)], output_root)
    typer.echo(f"Screening written to {output}")


@clustering_app.command("compare")
def clustering_compare(
    screening: Path,
    inputs: Path = typer.Option(...),
    output_root: Path = typer.Option(Path("artifacts/clustering")),
) -> None:
    """Stage B: shortlist-only reduction seeds, local parameter robustness and Pareto."""
    from flir_pipeline.clustering.experiments import comparison_to_store, load_families

    typer.echo(f"Comparison written to {comparison_to_store(screening, load_families(inputs), output_root)}")


@clustering_app.command("verify")
def clustering_verify(directory: Path, inputs: Path | None = typer.Option(None, help="Also bind sources and recompute cluster metrics/medoids.")) -> None:
    """Verify assignments, indices, metadata and collections; optionally recompute evaluation."""
    from flir_pipeline.clustering.experiments import load_families, verify_collection
    from flir_pipeline.clustering.storage import verify_run
    from flir_pipeline.similarity.storage import read_json

    families = load_families(inputs) if inputs is not None else None
    meta = read_json(directory/"metadata.json") if (directory/"metadata.json").is_file() else {}
    if meta.get("artifact_kind") in ("clustering_screening", "clustering_comparison"):
        result = verify_collection(directory, families)
    else:
        result = verify_run(directory, families[meta["extractor"]] if families is not None and "extractor" in meta else None)
    typer.echo(json.dumps(result, indent=2))
    if not result["quality_valid"]:
        raise typer.Exit(1)


@clustering_app.command("summary")
def clustering_summary(directory: Path) -> None:
    """Read only completed, verified clustering results."""
    from flir_pipeline.clustering.experiments import verify_collection
    from flir_pipeline.clustering.storage import verify_run
    from flir_pipeline.similarity.storage import read_json

    meta = read_json(directory/"metadata.json")
    is_run = meta["artifact_kind"] == "clustering_run"
    quality = verify_run(directory) if is_run else verify_collection(directory)
    if not quality["quality_valid"]:
        typer.echo(json.dumps(quality, indent=2))
        raise typer.Exit(1)
    typer.echo(json.dumps(read_json(directory/("metrics.json" if is_run else "summary.json")), indent=2))
