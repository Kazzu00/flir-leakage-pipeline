"""Local viewer commands; no optional Streamlit or scientific fitting imports."""

from pathlib import Path

import typer

app = typer.Typer(help="Local, read-only visual inspection of existing artifacts.")


@app.command("vikus-build")
def vikus_build(
    manifest: Path = typer.Option(Path("data/manifests/flir_canonical_candidate_v1.parquet")),
    candidate: str | None = typer.Option(None, help="Display alias discovered from the existing candidate table."),
    seed: int = typer.Option(0, help="Select an existing split seed; never generate one."),
    split_space_id: str | None = typer.Option(None),
    clustering_space_id: str | None = typer.Option(None, help="Clustering-only bundle, or explicit clustering for a baseline split."),
    clustering_root: Path = typer.Option(Path("artifacts/clustering")),
    split_root: Path = typer.Option(Path("artifacts/splitting")),
    reduction_root: Path = typer.Option(Path("artifacts/reduction")),
    candidates: Path = typer.Option(Path("reports/splitting/tables/clustering_candidates.csv")),
    extra_reduction: list[Path] | None = typer.Option(None, help="Optional existing aligned layout, e.g. t-SNE; explicit directory."),
    data_root: Path | None = typer.Option(None, help="Defaults to FLIR_DATA_ROOT / .env."),
    name: str | None = typer.Option(None, help="Optional local bundle folder name; never overwrites a different bundle."),
    runtime_archive: Path | None = typer.Option(None, help="Pinned upstream ZIP; useful for offline builds."),
    offline: bool = typer.Option(False, help="Require the already cached upstream archive."),
):
    import pandas as pd

    from flir_pipeline.cli import _default_root
    from flir_pipeline.explorer.data import load_cluster, load_split
    from flir_pipeline.explorer.discovery import candidate_labels, discover_runs
    from flir_pipeline.explorer.vikus import build_bundle, find_source_reduction
    from flir_pipeline.explorer.vikus_upstream import COMMIT, fetch_runtime

    if not any((candidate, split_space_id, clustering_space_id)) or candidate and (split_space_id or clustering_space_id):
        raise typer.BadParameter("Choose a candidate, or an explicit split/clustering identity")
    root = data_root or _default_root()
    if root is None:
        raise typer.BadParameter("Set FLIR_DATA_ROOT or provide --data-root")
    aliases = candidate_labels(candidates)
    clusters, cluster_issues = discover_runs(clustering_root, aliases)
    splits, split_issues = discover_runs(split_root, aliases)
    if cluster_issues or split_issues:
        typer.echo(f"Discovery excluded {len(cluster_issues) + len(split_issues)} incomplete/ambiguous entries")
    selected_split = None
    if candidate or split_space_id:
        selected = ([s for s in splits if s.candidate == candidate and s.seed == seed] if candidate
                    else [s for s in splits if s.space_id == split_space_id])
        if len(selected) != 1:
            raise typer.BadParameter("The requested existing split must be unique")
        selected_split = selected[0]
        clustering_space_id = clustering_space_id or selected_split.clustering_space_id
        if clustering_space_id is None:
            raise typer.BadParameter("A baseline overlay also requires --clustering-space-id")
    selected_clusters = [c for c in clusters if c.space_id == clustering_space_id]
    if len(selected_clusters) != 1:
        raise typer.BadParameter("The requested existing clustering must be unique")
    frame = pd.read_parquet(manifest)
    cluster = load_cluster(selected_clusters[0], frame)
    split = load_split(selected_split, frame) if selected_split else None
    source = find_source_reduction(cluster, reduction_root)
    reductions = ([(source, True)] if source else []) + [(p, False) for p in extra_reduction or []]
    archive = runtime_archive or Path(".cache/upstream") / f"vikus-{COMMIT}.zip"
    if not archive.is_file():
        if offline or runtime_archive:
            raise typer.BadParameter("Pinned VIKUS runtime archive is missing")
        archive = fetch_runtime(Path(".cache/upstream"))
    output = build_bundle(Path.cwd(), manifest, root, cluster, split, reductions, archive, name=name)
    typer.echo(f"Local VIKUS bundle: {output.relative_to(Path.cwd())}")
    typer.echo(f"Unique contents: {len(cluster.contents)}; historical occurrences: {len(frame)}")


@app.command("vikus-serve")
def vikus_serve(bundle: Path = typer.Option(...), port: int = typer.Option(8765, min=1024, max=65535)):
    from flir_pipeline.explorer.vikus_server import serve_bundle

    serve_bundle(bundle, Path.cwd(), port)
