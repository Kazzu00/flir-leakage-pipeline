"""Command-line entry point for the research pipeline."""

import typer

app = typer.Typer(help="Reproducible FLIR leakage research pipeline.")


def _placeholder(name: str) -> None:
    typer.echo(f"{name} is reserved for a future pipeline stage.")


@app.command()
def data() -> None:
    """Data ingestion and audit commands."""
    _placeholder("data")


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
