"""Thin CLI for source-bound experimental evidence; never creates a split."""

import contextlib
import json
import sys
from pathlib import Path

import typer

app = typer.Typer(
    help="Experimental temporal/visual evidence, manual review and coverage; no automatic confirmations or splits."
)


@app.command("register-variant")
def register_variant_command(
    manifest: Path = typer.Option(...),
    variant_name: str = typer.Option(...),
    parent: Path | None = typer.Option(
        None, help="Existing parent variant declaration."
    ),
    definition: Path | None = typer.Option(
        None,
        help="JSON description of the transformation; no image processing is performed.",
    ),
    output: Path = typer.Option(Path("artifacts/dataset_variants")),
):
    """Register a source-bound variant for later feature extraction and experiments."""
    from flir_pipeline.data.variants import register_variant

    try:
        path = register_variant(
            manifest,
            variant_name,
            output,
            parent=parent,
            definition=json.loads(definition.read_text(encoding="utf-8"))
            if definition
            else None,
        )
        typer.echo(path)
    except (OSError, ValueError, KeyError) as error:
        typer.echo(f"Variant registration failed: {error}", err=True)
        raise typer.Exit(1) from error


@app.command("import-correspondence")
def import_correspondence_command(
    suite_a: Path,
    suite_b: Path,
    correspondence: Path,
    output: Path = typer.Option(Path("artifacts/sequence_experiments")),
):
    """Validate explicit paired occurrences; retain coverage and external evidence."""
    from flir_pipeline.sequences.experiments.variant_comparison import (
        import_correspondence,
    )

    try:
        typer.echo(import_correspondence(suite_a, suite_b, correspondence, output))
    except (OSError, ValueError, KeyError, AssertionError) as error:
        typer.echo(f"Correspondence import failed: {error}", err=True)
        raise typer.Exit(1) from error


@app.command("compare-variants")
def compare_variants_command(
    suite_a: Path,
    suite_b: Path,
    correspondence: Path | None = typer.Option(
        None, help="Validated correspondence artifact."
    ),
    output: Path = typer.Option(Path("reports/sequence_variant_comparisons")),
):
    """Compare completed immutable suites; no fitting, automatic winner or split."""
    from flir_pipeline.sequences.experiments.variant_comparison import compare_variants

    try:
        typer.echo(compare_variants(suite_a, suite_b, output, correspondence))
    except (OSError, ValueError, KeyError, AssertionError) as error:
        typer.echo(f"Variant comparison failed: {error}", err=True)
        raise typer.Exit(1) from error


def command(operation):
    def execute(
        manifest: Path | None = typer.Option(None, envvar="FLIR_MANIFEST"),
        clip_features: Path | None = typer.Option(None, envvar="FLIR_CLIP_FEATURES"),
        dinov2_features: Path | None = typer.Option(
            None, envvar="FLIR_DINOV2_FEATURES"
        ),
        family: str = typer.Option("all"),
        dataset_variant: str | None = typer.Option(None, envvar="FLIR_DATASET_VARIANT"),
        profile: Path = typer.Option(Path("configs/hypatia_sequence_experiments.yaml")),
        output: Path = typer.Option(
            Path("artifacts/sequence_experiments"), envvar="FLIR_SEQUENCE_OUTPUT"
        ),
        structure: Path | None = typer.Option(
            None, help="Immutable imported structure review; post-hoc only."
        ),
        clustering: Path | None = typer.Option(
            None, help="Existing sequence clustering experiment artifact."
        ),
    ):
        from flir_pipeline.sequences.experiments import runner
        from flir_pipeline.sequences.experiments.inputs import (
            load_profile,
            resolve_inputs,
        )

        try:
            config, inputs, _ = load_profile(profile)
            _, source = resolve_inputs(
                inputs,
                family,
                manifest=manifest,
                clip_features=clip_features,
                dinov2_features=dinov2_features,
                dataset_variant=dataset_variant,
            )
            if operation in {"recurrence", "evaluate"} and structure is None:
                raise ValueError("This post-hoc command requires --structure")
            if operation in {"evaluate", "cluster-transitions"} and clustering is None:
                raise ValueError("This command requires --clustering")
            with contextlib.redirect_stdout(sys.stderr):
                if operation == "suite":
                    path = runner.suite(
                        source,
                        config,
                        output,
                        structure,
                        progress=lambda message: typer.echo(message, err=True),
                    )
                elif operation == "boundary":
                    paths = [
                        runner.run_boundary(source, boundary, output)
                        for boundary in config.boundaries()
                    ]
                    path = "\n".join(str(p) for p in paths)
                elif operation == "clustering":
                    path = runner.run_clustering(source, config, output)
                elif operation == "recurrence":
                    path = runner.run_recurrence(
                        source, structure, config.recurrence, output, clustering
                    )
                elif operation == "evaluate":
                    path = runner.run_evaluate(source, clustering, structure, output)
                else:
                    path = runner.run_transitions(
                        source, clustering, config.transitions, output, structure
                    )
            typer.echo(path)
            if operation in {"clustering", "suite"}:
                from flir_pipeline.similarity.storage import read_json

                summary = read_json(path / "summary.json")
                if not summary["all_requested_cells_succeeded"]:
                    typer.echo(
                        f"Published incomplete experiment: {summary['failed_cells']} failed cells; inspect failures.parquet.",
                        err=True,
                    )
                    raise typer.Exit(1)
        except (OSError, ValueError, KeyError, AssertionError, ImportError) as error:
            typer.echo(f"Sequence experiment failed: {error}", err=True)
            raise typer.Exit(1) from error

    return execute


for _name, _help in {
    "boundary": "Original-L2 boundary-zone candidates; configurable multiscale grids.",
    "clustering": "Fit unique visual contents only; reductions and labels are not temporal truth.",
    "recurrence": "Candidate dependencies between distinct reviewed cores; no VDG creation.",
    "cluster-transitions": "Experimental cluster timeline transitions; never scene boundaries.",
    "evaluate": "Post-hoc ARI/AMI/V-measure with exact masks and coverage.",
    "suite": "Run local experiments and ablations; no scheduling logic, split or automatic winner.",
}.items():
    app.command(_name, help=_help)(command(_name))


@app.command("import-real-evidence")
def import_real_evidence_command(
    family: str = typer.Option(...),
    dataset_variant: str | None = typer.Option(None, envvar="FLIR_DATASET_VARIANT"),
    root: Path = typer.Option(Path("reports")),
    profile: Path = typer.Option(Path("configs/hypatia_sequence_experiments.yaml")),
    manifest: Path | None = typer.Option(None, envvar="FLIR_MANIFEST"),
    clip_features: Path | None = typer.Option(None, envvar="FLIR_CLIP_FEATURES"),
    dinov2_features: Path | None = typer.Option(None, envvar="FLIR_DINOV2_FEATURES"),
    select: list[str] | None = typer.Option(
        None, help="Explicit kind=metadata-file-or-directory; repeat for ambiguity."
    ),
    output: Path = typer.Option(Path("reports/sequence_evidence")),
    dry_run: bool = typer.Option(
        False, help="Validate inputs and native consistency without publishing."
    ),
):
    """Discover typed source-bound producers, validate and publish immutable evidence."""
    from flir_pipeline.sequences.experiments.inputs import load_profile, resolve_inputs
    from flir_pipeline.sequences.experiments.real_evidence import import_real

    try:
        _, inputs, _ = load_profile(profile)
        _, source = resolve_inputs(
            inputs,
            family,
            manifest=manifest,
            clip_features=clip_features,
            dinov2_features=dinov2_features,
            dataset_variant=dataset_variant,
        )
        selections = {}
        for value in select or []:
            kind, separator, path = value.partition("=")
            if not separator or not path or kind in selections:
                raise ValueError("--select requires distinct kind=path selections")
            selections[kind] = Path(path)
        result = import_real(root, source, output, selections, dry_run=dry_run)
        typer.echo(json.dumps(result, indent=2) if dry_run else result)
    except (OSError, ValueError, KeyError, AssertionError) as error:
        typer.echo(f"Real evidence import failed: {error}", err=True)
        raise typer.Exit(1) from error


@app.command("inspect-real-evidence")
def inspect_real_evidence_command(
    family: str = typer.Option(...),
    root: Path = typer.Option(Path("reports")),
    select: list[str] | None = typer.Option(
        None, help="Repeat kind=summary-file-or-directory to resolve ambiguity."
    ),
):
    """Read-only native schema/consistency dry-run; no features or outputs required."""
    from flir_pipeline.sequences.experiments.real_evidence import inspect_real

    try:
        selections = {}
        for value in select or []:
            kind, separator, path = value.partition("=")
            if not separator or not path or kind in selections:
                raise ValueError("--select requires distinct kind=path selections")
            selections[kind] = Path(path)
        typer.echo(json.dumps(inspect_real(root, family, selections), indent=2))
    except (OSError, ValueError, KeyError, AssertionError) as error:
        typer.echo(f"Native evidence inspection failed: {error}", err=True)
        raise typer.Exit(1) from error


@app.command("import-evidence")
def import_evidence_command(
    evidence: Path = typer.Argument(
        ..., help="Explicit normalized evidence JSON binding original producer files."
    ),
    manifest: Path = typer.Option(..., envvar="FLIR_MANIFEST"),
    clip_features: Path = typer.Option(..., envvar="FLIR_CLIP_FEATURES"),
    dinov2_features: Path = typer.Option(..., envvar="FLIR_DINOV2_FEATURES"),
    family: str = typer.Option("all"),
    dataset_variant: str | None = typer.Option(None, envvar="FLIR_DATASET_VARIANT"),
    output: Path = typer.Option(
        Path("artifacts/sequence_experiments"), envvar="FLIR_SEQUENCE_OUTPUT"
    ),
):
    """Import checked external/manual intervals and observations; never exactify zones."""
    from flir_pipeline.sequences.experiments.sources import load_sources
    from flir_pipeline.sequences.experiments.structure import import_evidence

    try:
        source = load_sources(
            manifest, clip_features, dinov2_features, family, dataset_variant
        )
        typer.echo(import_evidence(evidence, source, output))
    except (OSError, ValueError, KeyError, AssertionError) as error:
        typer.echo(f"Evidence import failed: {error}", err=True)
        raise typer.Exit(1) from error


@app.command("verify")
def verify_command(
    directory: Path,
    manifest: Path = typer.Option(..., envvar="FLIR_MANIFEST"),
    clip_features: Path = typer.Option(..., envvar="FLIR_CLIP_FEATURES"),
    dinov2_features: Path = typer.Option(..., envvar="FLIR_DINOV2_FEATURES"),
    family: str = typer.Option("all"),
    dataset_variant: str | None = typer.Option(None, envvar="FLIR_DATASET_VARIANT"),
    structure: Path | None = typer.Option(None),
    clustering: Path | None = typer.Option(None),
    evidence: Path | None = typer.Option(
        None, help="Original boundary/recurrence artifact for a review publication."
    ),
    review_package: Path | None = typer.Option(
        None, help="Original immutable review package for manual decisions."
    ),
    evidence_member: list[Path] | None = typer.Option(
        None,
        help="Repeat for each member of a combined review package, in recorded order.",
    ),
):
    """Verify all files, identities, child publications and current immutable inputs."""
    from flir_pipeline.sequences.experiments.runner import verify
    from flir_pipeline.sequences.experiments.sources import load_sources

    try:
        result = verify(
            directory,
            load_sources(
                manifest, clip_features, dinov2_features, family, dataset_variant
            ),
            structure,
            clustering,
            evidence_member or evidence,
            review_package,
        )
    except (OSError, ValueError, KeyError, AssertionError) as error:
        result = {"quality_valid": False, "error": str(error)}
    typer.echo(json.dumps(result, indent=2))
    if not result["quality_valid"]:
        raise typer.Exit(1)


@app.command("summary")
def summary_command(directory: Path):
    """Read aggregate metadata only; does not verify the publication."""
    from flir_pipeline.sequences.experiments.artifacts import summary

    typer.echo(json.dumps(summary(directory), indent=2))


@app.command("review-package")
def review_package_command(
    directory: Path,
    manifest: Path = typer.Option(..., envvar="FLIR_MANIFEST"),
    clip_features: Path = typer.Option(..., envvar="FLIR_CLIP_FEATURES"),
    dinov2_features: Path = typer.Option(..., envvar="FLIR_DINOV2_FEATURES"),
    family: str = typer.Option("all"),
    dataset_variant: str | None = typer.Option(None, envvar="FLIR_DATASET_VARIANT"),
    images_root: Path | None = typer.Option(None),
    images_archive: Path | None = typer.Option(None),
    context_radius: int = typer.Option(20, min=1),
    output: Path = typer.Option(Path("reports/sequence_review")),
):
    """Boundary/pair contact sheets, encoder medoids/matches and wide context."""
    from flir_pipeline.sequences.experiments.review import create_package
    from flir_pipeline.sequences.experiments.sources import load_sources

    try:
        source = load_sources(
            manifest, clip_features, dinov2_features, family, dataset_variant
        )
        typer.echo(
            create_package(
                directory, source, output, images_root, images_archive, context_radius
            )
        )
    except (OSError, ValueError, KeyError, AssertionError) as error:
        typer.echo(f"Review package failed: {error}", err=True)
        raise typer.Exit(1) from error


@app.command("review-import")
def review_import_command(
    package: Path,
    decisions: Path,
    previous: Path | None = typer.Option(None),
    output: Path = typer.Option(Path("reports/sequence_review")),
):
    """Immutable supported/ambiguous/unsupported decisions and history; no confirmation."""
    from flir_pipeline.sequences.experiments.review import import_decisions

    try:
        typer.echo(import_decisions(package, decisions, output, previous))
    except (OSError, ValueError, KeyError, AssertionError) as error:
        typer.echo(f"Review import failed: {error}", err=True)
        raise typer.Exit(1) from error
