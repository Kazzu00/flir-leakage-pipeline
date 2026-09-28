"""Manual group-calibration CLI; no automatic decision or split operation."""

import json
from pathlib import Path

import typer

app = typer.Typer(
    help="Manual evidence calibration for proposed visual dependency groups; not ground truth."
)


def _run(function, *args, **kwargs):
    try:
        result = function(*args, **kwargs)
        typer.echo(
            json.dumps(result, indent=2, ensure_ascii=False)
            if isinstance(result, dict)
            else result
        )
        if isinstance(result, dict) and result.get("quality_valid") is False:
            raise typer.Exit(1)
    except (
        ValueError,
        OSError,
        KeyError,
        TypeError,
        AssertionError,
        IndexError,
    ) as error:
        typer.echo(f"Manual calibration failed: {error}", err=True)
        raise typer.Exit(1) from error


@app.command("init")
def initialize(
    calibration_sample: Path = typer.Option(...),
    linkage: Path = typer.Option(...),
    labeled_manifest: Path = typer.Option(...),
    sequence_set: Path = typer.Option(...),
    visual_dependencies: Path = typer.Option(...),
    membership_table: str = typer.Option(
        "membership.csv",
        help="Checksum-bound CSV/Parquet within the confirmed visual dependency artifact.",
    ),
    labeled_images_archive: Path | None = typer.Option(None),
    labeled_images_root: Path | None = typer.Option(None),
    video_images_root: Path = typer.Option(...),
    context_seconds: int = typer.Option(3, min=0, max=30),
    output: Path = typer.Option(Path("reports/linkage/manual_calibration")),
):
    """Create blank query/group decisions and all-occurrence temporal contact sheets."""
    from flir_pipeline.linkage.review_model import ReviewConfig
    from flir_pipeline.linkage.review_sources import ReviewPaths
    from flir_pipeline.linkage.review_storage import init_review

    _run(
        init_review,
        ReviewPaths(
            calibration_sample,
            linkage,
            labeled_manifest,
            sequence_set,
            visual_dependencies,
            membership_table,
        ),
        ReviewConfig(context_seconds=context_seconds),
        output,
        labeled_images_archive=labeled_images_archive,
        labeled_images_root=labeled_images_root,
        video_images_root=video_images_root,
    )


@app.command("record")
def record(
    directory: Path,
    decisions: Path = typer.Option(
        ..., help="External CSV copy with explicit manual decisions/notes."
    ),
    reviewer: str = typer.Option(...),
    source: str = typer.Option(
        ..., help="Human-readable origin of this review, not an automatic rank rule."
    ),
    timestamp_utc: str | None = typer.Option(None),
    output: Path | None = typer.Option(None),
):
    """Import decisions into a new immutable revision, retaining the original CSV and history."""
    from flir_pipeline.linkage.review_storage import record_review

    _run(
        record_review,
        directory,
        decisions,
        reviewer=reviewer,
        source=source,
        timestamp=timestamp_utc,
        output=output,
    )


@app.command("summary")
def summary(directory: Path):
    """Count query-level decisions by stratum AND group; no representative accuracy claim."""
    from flir_pipeline.linkage.review_storage import summarize_review

    _run(summarize_review, directory)


@app.command("verify")
def verify(
    directory: Path,
    calibration_sample: Path = typer.Option(...),
    linkage: Path = typer.Option(...),
    labeled_manifest: Path = typer.Option(...),
    sequence_set: Path = typer.Option(...),
    visual_dependencies: Path = typer.Option(...),
    membership_table: str = typer.Option("membership.csv"),
):
    """Rebind sources, reconstruct evidence/context, and replay explicit manual decisions."""
    from flir_pipeline.linkage.review_sources import ReviewPaths
    from flir_pipeline.linkage.review_storage import verify_review

    _run(
        verify_review,
        directory,
        ReviewPaths(
            calibration_sample,
            linkage,
            labeled_manifest,
            sequence_set,
            visual_dependencies,
            membership_table,
        ),
    )
