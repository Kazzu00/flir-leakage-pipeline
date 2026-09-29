"""Small serial/SLURM-shared stages; no scheduler-specific scientific settings."""

from pathlib import Path

import pandas as pd

from flir_pipeline.sequences.experiments import runner
from flir_pipeline.sequences.experiments.artifacts import (
    binding,
    inspect,
    publish,
    tables,
)
from flir_pipeline.sequences.experiments.config import BoundaryConfig, SuiteConfig
from flir_pipeline.sequences.experiments.fitting import clustering_tables
from flir_pipeline.sequences.experiments.review import (
    create_package,
    verify_decision_history,
)
from flir_pipeline.sequences.experiments.sources import scientific_input_id
from flir_pipeline.similarity.storage import read_json


def graph(config):
    """Canonical topological order. Each reduction is fitted once, then shared."""
    stages = []

    def add(name, kind, dependencies=(), configuration=None):
        stages.append(
            dict(
                stage=name,
                kind=kind,
                dependencies=list(dependencies),
                config=configuration,
            )
        )

    for i, boundary in enumerate(config.boundaries()):
        add(f"boundary_{i}", "boundary", configuration=boundary.model_dump(mode="json"))
    fits = []
    for encoder in config.encoders:
        spaces = [(True, [])] if config.original_l2 else []
        for grid in config.reductions:
            for reduction in grid.expand():
                spaces.append(
                    (
                        False,
                        [
                            dict(
                                method=reduction.method,
                                seeds=[reduction.seed],
                                output_dimension=reduction.output_dimension,
                                parameters=reduction.hyperparameters,
                            )
                        ],
                    )
                )
        for index, (original, reductions) in enumerate(spaces):
            selected = config.model_dump(mode="json")
            selected.update(
                encoders=[encoder], original_l2=original, reductions=reductions
            )
            name = f"representation_{encoder}_{index}"
            add(name, "representation", configuration=selected)
            # Configurations of one algorithm share a worker and pairwise distances.
            for j, grid in enumerate(config.clustering):
                fit = f"fit_{encoder}_{index}_{j}_{grid.algorithm}"
                fits.append(fit)
                add(
                    fit,
                    "fit",
                    [name],
                    {**selected, "clustering": [grid.model_dump(mode="json")]},
                )
    add("direct_recurrence", "direct_recurrence")
    add("clustering", "merge", fits)
    add("transitions", "transitions", ["clustering"])
    add("evaluation", "evaluation", ["clustering"])
    add("recurrence", "recurrence", ["clustering", "direct_recurrence"])
    boundaries = [s["stage"] for s in stages if s["kind"] == "boundary"]
    add(
        "suite",
        "suite",
        [*boundaries, "clustering", "transitions", "evaluation", "recurrence"],
    )
    add("review_package", "review_package", [*boundaries, "recurrence", "suite"])
    add(
        "final_summary",
        "final_summary",
        ["suite", "review_package", "recurrence", "transitions", "clustering"],
    )
    return stages


def merge_clustering(source, config, shards, output):
    """Concatenate checked cells in the serial loop order, without refitting."""
    parts, details = [], {"fits": {}, "spaces": {}}
    expected = [s for s in graph(config) if s["kind"] == "fit"]
    if set(shards) != {s["stage"] for s in expected}:
        raise ValueError("Incomplete clustering shard coverage")
    for spec in expected:
        path = shards[spec["stage"]]
        meta = inspect(path)
        if meta["identity"]["config"] != spec["config"]:
            raise ValueError("Clustering shard configuration changed")
        part = runner.require_clustering(path, source)
        if not part["failures"].empty:
            raise ValueError("Failed scientific cells cannot be aggregated as complete")
        selected = SuiteConfig.model_validate(spec["config"])
        expected_runs = sum(len(grid.expand()) for grid in selected.clustering)
        if len(part["runs"]) != expected_runs or not part["runs"].run_id.is_unique:
            raise ValueError("Incomplete clustering configuration coverage")
        if set(part["assignments"].run_id) != set(part["runs"].run_id):
            raise ValueError("Incomplete assignment run coverage")
        for _, group in part["assignments"].groupby("run_id"):
            if not group.content_id.is_unique or set(group.content_id) != set(
                source.contents
            ):
                raise ValueError("Clustering shard lost unique-content coverage")
        pd.testing.assert_frame_equal(part["occurrences"], source.records)
        parts.append(part)
        backend = read_json(path / "backend_details.json")
        for role in details:
            for key, value in backend[role].items():
                if key in details[role] and details[role][key] != value:
                    raise ValueError("Conflicting shared representation metadata")
                details[role][key] = value
    combined = {
        name: pd.concat(
            [p[name] for p in parts if not p[name].empty] or [parts[0][name]],
            ignore_index=True,
        )
        for name in ("runs", "assignments", "coordinates", "failures")
    }
    if not combined["runs"].run_id.is_unique:
        raise ValueError("Duplicate fitted run IDs")
    coords = combined["coordinates"].drop_duplicates().reset_index(drop=True)
    if coords.duplicated(["space_id", "content_id"]).any():
        raise ValueError("Conflicting shared coordinates")
    combined["coordinates"] = coords
    combined["occurrences"] = source.records.copy()
    source.unchanged()
    return publish(
        output,
        "sequence_clustering_experiment_v1",
        config.model_dump(mode="json"),
        {"input": source.signature},
        combined,
        dict(
            successful_runs=len(combined["runs"]),
            failed_cells=0,
            all_requested_cells_succeeded=True,
            unique_fitting_contents=len(source.contents),
            preserved_occurrences=len(source.records),
        ),
        extras={"backend_details": details},
    )


def review_checkpoint(package, review=None):
    meta = inspect(package)
    if meta["artifact_kind"] != "sequence_review_package_v1":
        raise ValueError("Expected review package")
    queries = set(tables(package)["queries"].review_query_id)
    reviewed = set()
    if review is not None:
        manual = inspect(review)
        if manual["artifact_kind"] != "sequence_manual_review_v1" or manual["identity"][
            "sources"
        ].get("package") != binding(package):
            raise ValueError("Review import does not belong to this immutable package")
        verify_decision_history(review)
        reviewed = set(tables(review)["decisions"].review_query_id)
        if not reviewed <= queries:
            raise ValueError("Review contains unknown queries")
    required = bool(queries - reviewed)
    return dict(
        manual_review_required=required,
        review_package=str(Path(package).resolve()),
        review_artifact=str(Path(review).resolve()) if review else None,
        review_queries=len(queries),
        reviewed_queries=len(reviewed),
        confirmation_dependent_stages_allowed=review is not None and not required,
        confirmation_dependent_stages_implemented=False,
    )


def require_review(package, review):
    status = review_checkpoint(package, review)
    if not status["confirmation_dependent_stages_allowed"]:
        raise ValueError(
            "Blocked by manual review: import valid decisions for every query"
        )
    return status


def final_summary(source, config, paths, output, review=None):
    package = paths["review_package"]
    checkpoint = review_checkpoint(package, review)
    recurrence = tables(paths["recurrence"])
    suite = tables(paths["suite"])
    clustering = read_json(paths["clustering"] / "summary.json")
    transition = read_json(paths["transitions"] / "summary.json")
    pairs = recurrence["pairs"]
    result = dict(
        temporal_structure={
            "cluster_transitions": transition["transitions"],
            "boundary_ablation_rows": len(suite["boundary_ablation_comparison"]),
            "exact_boundaries_created": False,
        },
        clustering_coverage={
            k: clustering[k]
            for k in (
                "successful_runs",
                "failed_cells",
                "all_requested_cells_succeeded",
            )
        },
        recurrence_candidates={
            "possible_pairs": len(pairs),
            "broad": int(pairs.broad_candidate.sum()),
            "refined": int(pairs.refined_candidate.sum()),
        },
        encoder_agreement={
            "disagreement_pairs": int(pairs.encoder_disagreement.sum()),
            "agreement_pairs": int((~pairs.encoder_disagreement).sum()),
        },
        cluster_direct_agreement={
            "agreement_rows": len(recurrence["agreement"]),
            "disagreement_rows": len(recurrence["disagreement"]),
        },
        stability={
            "comparisons": len(suite["stability"]),
            "boundary_comparisons": len(suite["boundary_stability"]),
        },
        **checkpoint,
        visual_dependency_groups_created=False,
        split_created=False,
        leakage_safe_split_exists=False,
    )
    dependencies = {role: binding(path) for role, path in sorted(paths.items())}
    if review:
        dependencies["manual_review"] = binding(review)
    return publish(
        output,
        "sequence_final_summary_v1",
        config.model_dump(mode="json"),
        {"input": source.signature, "dependencies": dependencies},
        {
            name: suite[name]
            for name in (
                "comparison",
                "stability",
                "boundary_stability",
                "boundary_ablation_comparison",
            )
        },
        result,
    )


def execute_stage(spec, source, config, evidence, dependencies, output, inputs):
    kind = spec["kind"]
    if kind == "boundary":
        return runner.run_boundary(
            source, BoundaryConfig.model_validate(spec["config"]), output
        )
    if kind in {"representation", "fit"}:
        selected = SuiteConfig.model_validate(spec["config"])
        prepared = None
        if kind == "fit":
            cache = dependencies[spec["dependencies"][0]]
            meta = inspect(cache)
            if meta["artifact_kind"] != "sequence_representation_v1" or meta[
                "identity"
            ]["sources"] != {"input": source.signature}:
                raise ValueError("Invalid representation cache binding")
            cache_config = {
                **meta["identity"]["config"],
                "clustering": spec["config"]["clustering"],
            }
            if cache_config != spec["config"]:
                raise ValueError("Prepared representation configuration changed")
            prepared = (
                tables(cache)["coordinates"],
                read_json(cache / "backend_details.json")["spaces"],
            )
        result, backend = clustering_tables(
            source,
            selected,
            scientific_input_id(source),
            prepared=prepared,
            prepare_only=kind == "representation",
        )
        source.unchanged()
        path = publish(
            output,
            "sequence_representation_v1"
            if kind == "representation"
            else "sequence_clustering_experiment_v1",
            spec["config"],
            {"input": source.signature},
            result,
            dict(
                successful_runs=len(result["runs"]),
                failed_cells=len(result["failures"]),
                all_requested_cells_succeeded=result["failures"].empty,
            ),
            extras={"backend_details": backend},
        )
        if not result["failures"].empty:
            raise ValueError(f"Scientific cells failed; immutable diagnostics: {path}")
        return path
    if kind == "merge":
        return merge_clustering(source, config, dependencies, output)
    if kind in {"direct_recurrence", "recurrence"}:
        return runner.run_recurrence(
            source,
            evidence,
            config.recurrence,
            output,
            dependencies.get("clustering"),
            dependencies.get("direct_recurrence"),
        )
    if kind == "evaluation":
        return runner.run_evaluate(source, dependencies["clustering"], evidence, output)
    if kind == "transitions":
        return runner.run_transitions(
            source, dependencies["clustering"], config.transitions, output, evidence
        )
    if kind == "suite":
        return runner.suite(source, config, output, evidence, completed=dependencies)
    if kind == "review_package":
        return create_package(
            [
                path
                for role, path in dependencies.items()
                if role.startswith("boundary_") or role == "recurrence"
            ],
            source,
            output,
            inputs.get("images_root"),
            inputs.get("images_archive"),
        )
    if kind == "final_summary":
        return final_summary(source, config, dependencies, output)
    raise ValueError(f"Unknown stage: {kind}")
