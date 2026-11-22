"""Explicit user-accepted preliminary BASE path, independent of final admission."""

from pathlib import Path

import typer

from flir_pipeline.detection import preliminary
from flir_pipeline.similarity.storage import file_sha256

app = typer.Typer(
    help="Positive-only preliminary BASE; no protected test or final admission."
)


@app.command("prepare")
def prepare(
    source: Path,
    output: Path,
    weights: Path,
    source_receipt_sha256: str = typer.Option(...),
    accept_original_nonempty: bool = typer.Option(False),
    policy: str = "S3",
    batch: int = 1,
    config: Path = Path("configs/detection/common_test_yolo11m_v2.yaml"),
):
    typer.echo(
        preliminary.prepare(
            source,
            source_receipt_sha256,
            config,
            output,
            file_sha256(weights),
            policy=policy,
            batch=batch,
            accept_original_nonempty=accept_original_nonempty,
        )
    )


@app.command("verify")
def verify(package: Path, receipt_sha256: str = typer.Option(...)):
    typer.echo(preliminary.verify(package, receipt_sha256))


@app.command("train")
def train(
    package: Path, weights: Path, output: Path, receipt_sha256: str = typer.Option(...)
):
    from flir_pipeline.detection.cli import _prime_cuda_import_order

    _prime_cuda_import_order()
    typer.echo(preliminary.train(package, receipt_sha256, weights, output))
