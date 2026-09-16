"""Lazy command surface for construction, comparison, verification and later export."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import typer

app = typer.Typer(help="Atomic splits, seeded baselines and residual correlation in both encoders.")


def _inputs(spec: Path, comparison: Path, labels: Path | None):
    from flir_pipeline.cli import _default_root
    from flir_pipeline.splitting.storage import load_inputs
    root = _default_root()
    archive = labels or (root/"Etiquetas.zip" if root else None)
    if archive is None or not archive.is_file():
        raise typer.BadParameter("Provide --labels-archive or configure FLIR_DATA_ROOT")
    return load_inputs(spec, comparison, archive)


@app.command("build")
def build(inputs: Path = typer.Option(...), comparison: Path = typer.Option(...),
          config: Path = typer.Option(Path("configs/splits/cluster_aware_research.yaml")),
          labels_archive: Path | None = typer.Option(None), clustering_id: str | None = typer.Option(None),
          output: Path = typer.Option(Path("artifacts/splitting"))) -> None:
    """Construct/evaluate five runs for each selected clustering (or one selected ID)."""
    from flir_pipeline.splitting.base import SplitConfig
    from flir_pipeline.splitting.storage import build_run
    source, settings = _inputs(inputs, comparison, labels_archive), SplitConfig.load(config)
    if settings.strategy != "cluster_aware":
        raise typer.BadParameter("build requires a cluster_aware config")
    ids = [clustering_id] if clustering_id else source.candidates.clustering_space_id.tolist()
    if not set(ids) <= set(source.cluster_paths):
        raise typer.BadParameter("Clustering ID is outside the reproducibly selected candidate set")
    for cid in ids:
        for seed in settings.seeds:
            typer.echo(build_run(source, settings, seed, output, cid))


@app.command("baseline")
def baseline(inputs: Path = typer.Option(...), comparison: Path = typer.Option(...),
             config: Path = typer.Option(Path("configs/splits/random_baseline.yaml")),
             labels_archive: Path | None = typer.Option(None),
             output: Path = typer.Option(Path("artifacts/splitting"))) -> None:
    """Preserve/evaluate historical memberships and build content-random seeds."""
    from flir_pipeline.splitting.base import SplitConfig
    from flir_pipeline.splitting.storage import build_run
    source, settings = _inputs(inputs, comparison, labels_archive), SplitConfig.load(config)
    if settings.strategy != "random_content":
        raise typer.BadParameter("baseline requires a random_content config")
    typer.echo(build_run(source, replace(settings, strategy="historical"), 0, output))
    for seed in settings.seeds:
        typer.echo(build_run(source, settings, seed, output))


@app.command("compare")
def compare(inputs: Path = typer.Option(...), comparison: Path = typer.Option(...),
            cluster_config: Path = typer.Option(Path("configs/splits/cluster_aware_research.yaml")),
            random_config: Path = typer.Option(Path("configs/splits/random_baseline.yaml")),
            labels_archive: Path | None = typer.Option(None),
            output: Path = typer.Option(Path("artifacts/splitting"))) -> None:
    """Build/reuse all runs, compare seed robustness, and publish the Pareto shortlist."""
    from flir_pipeline.splitting.base import SplitConfig
    from flir_pipeline.splitting.experiments import compare_to_store
    source = _inputs(inputs, comparison, labels_archive)
    typer.echo(compare_to_store(source, SplitConfig.load(cluster_config), SplitConfig.load(random_config), output))


@app.command("verify")
def verify(directory: Path, inputs: Path | None = typer.Option(None),
           comparison: Path | None = typer.Option(None), labels_archive: Path | None = typer.Option(None)) -> None:
    """Recompute invariants; --inputs plus --comparison also re-evaluate against sources."""
    from flir_pipeline.similarity.storage import read_json
    from flir_pipeline.splitting.experiments import verify_comparison
    from flir_pipeline.splitting.storage import verify_split
    if (inputs is None) != (comparison is None):
        raise typer.BadParameter("Provide --inputs and --comparison together")
    source = _inputs(inputs, comparison, labels_archive) if inputs else None
    kind = read_json(directory/"metadata.json")["artifact_kind"]
    result = verify_comparison(directory, source) if kind == "split_comparison" else verify_split(directory, source)
    typer.echo(json.dumps(result, indent=2))
    if not result["quality_valid"]:
        raise typer.Exit(1)


@app.command("evaluate")
def evaluate(directory: Path, inputs: Path = typer.Option(...), comparison: Path = typer.Option(...),
             labels_archive: Path | None = typer.Option(None)) -> None:
    """Re-evaluate every stored metric using both verified encoder sources."""
    verify(directory, inputs, comparison, labels_archive)
    summary(directory)


@app.command("summary")
def summary(directory: Path) -> None:
    """Display an aggregate summary, without content IDs or private paths."""
    from flir_pipeline.similarity.storage import read_json
    from flir_pipeline.splitting.experiments import verify_comparison
    from flir_pipeline.splitting.storage import verify_split
    kind = read_json(directory/"metadata.json")["artifact_kind"]
    result = verify_comparison(directory) if kind == "split_comparison" else verify_split(directory)
    if not result["quality_valid"]:
        raise typer.BadParameter("Artifact verification failed")
    name = "summary.json" if kind == "split_comparison" else "split_summary.json"
    typer.echo(json.dumps(read_json(directory/name), indent=2, ensure_ascii=False))


@app.command("export-lists")
def export_lists(directory: Path, manifest: Path = typer.Option(...), image_root: Path = typer.Option(...),
                 output: Path = typer.Option(...)) -> None:
    """Later handoff: write lists for already materialized images, without moving data."""
    from flir_pipeline.splitting.experiments import export_image_lists
    export_image_lists(directory, manifest, image_root, output)
    typer.echo("train.txt, val.txt and test.txt written; image files were only referenced.")
