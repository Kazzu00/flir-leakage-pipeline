"""Local Python orchestration; scheduling policy does not enter scientific logic."""

from pathlib import Path

import pandas as pd

from flir_pipeline.sequences.experiments.artifacts import (
    binding,
    inspect,
    publish,
    tables,
)
from flir_pipeline.sequences.experiments.boundary import boundary_tables
from flir_pipeline.sequences.experiments.evaluation import (
    evaluation_tables,
    stability_tables,
)
from flir_pipeline.sequences.experiments.fitting import clustering_tables
from flir_pipeline.sequences.experiments.recurrence import (
    cluster_recurrence,
    recurrence_tables,
)
from flir_pipeline.sequences.experiments.sources import scientific_input_id
from flir_pipeline.sequences.experiments.structure import load_structure
from flir_pipeline.sequences.experiments.transitions import (
    ablation_tables,
    interval_stability,
    transition_tables,
)


def run_boundary(source, config, output):
    source.safe_output(output)
    result = boundary_tables(source, config, scientific_input_id(source))
    source.unchanged()
    return publish(
        output,
        "sequence_boundary_candidates_v2",
        config.model_dump(mode="json"),
        {"input": source.signature},
        result,
        {
            "occurrences": len(source.records),
            "unique_contents": len(source.contents),
            "candidate_zone_count": len(result["candidate_zones"]),
            "valid_scored_cuts": len(result["temporal_scores"]),
        },
    )


def run_clustering(source, config, output):
    source.safe_output(output)
    result, details = clustering_tables(source, config, scientific_input_id(source))
    source.unchanged()
    return publish(
        output,
        "sequence_clustering_experiment_v1",
        config.model_dump(mode="json"),
        {"input": source.signature},
        result,
        {
            "successful_runs": len(result["runs"]),
            "failed_cells": len(result["failures"]),
            "all_requested_cells_succeeded": result["failures"].empty,
            "unique_fitting_contents": len(source.contents),
            "preserved_occurrences": len(source.records),
        },
        extras={"backend_details": details},
    )


def require_clustering(directory, source):
    meta = inspect(directory)
    if meta["artifact_kind"] != "sequence_clustering_experiment_v1" or meta["identity"][
        "sources"
    ] != {"input": source.signature}:
        raise ValueError("Require clustering experiment bound to these sources")
    result = tables(directory)
    if result["runs"].empty:
        raise ValueError(
            "No successful clustering runs; inspect failures before evaluation"
        )
    return result


def run_recurrence(source, structure, config, output, clustering=None, direct=None):
    source.safe_output(output, structure)
    _, cores, intervals, review_binding = load_structure(structure, source)
    if cores.empty:
        cores = intervals.loc[
            intervals.kind.eq("sequence_instance") & intervals.decision.eq("supported")
        ].copy()
    if direct is None:
        result = recurrence_tables(source, cores, config)
    else:
        meta = inspect(direct)
        if (
            meta["artifact_kind"] != "sequence_recurrence_v1"
            or meta["identity"]["sources"]
            != {"input": source.signature, "structure": review_binding}
            or meta["identity"]["config"] != config.model_dump(mode="json")
        ):
            raise ValueError(
                "Direct recurrence cache does not match inputs/configuration"
            )
        result = tables(direct)
    result["occurrences"] = source.records.copy()
    result["structure_elements"] = cores
    sources = {"input": source.signature, "structure": review_binding}
    if clustering:
        fitted = require_clustering(clustering, source)
        result.update(
            cluster_recurrence(
                fitted["assignments"], cores, source, result["pairs"], config
            )
        )
        sources["clustering"] = binding(clustering)
    source.unchanged()
    return publish(
        output,
        "sequence_recurrence_v1",
        config.model_dump(mode="json"),
        sources,
        result,
        {
            "possible_pairs": len(result["pairs"]),
            "refined_candidates": int(result["pairs"].refined_candidate.sum()),
            "non_discriminative_warning": bool(
                result["candidate_diagnostics"].non_discriminative_warning.any()
            ),
        },
    )


def run_evaluate(source, clustering, structure, output):
    source.safe_output(output, structure, clustering)
    membership, _, _, review_binding = load_structure(structure, source)
    fitted = require_clustering(clustering, source)
    result = evaluation_tables(source, fitted["assignments"], membership)
    result["occurrences"] = source.records.copy()
    source.unchanged()
    return publish(
        output,
        "sequence_cluster_evaluation_v1",
        {"evaluation_policy": "unique_content_unanimous_occurrences_v1"},
        {
            "input": source.signature,
            "clustering": binding(clustering),
            "structure": review_binding,
        },
        result,
        {
            "runs": len(fitted["runs"]),
            "metric_rows": len(result["metrics"]),
            "boundary_zones_excluded": True,
        },
    )


def run_transitions(source, clustering, config, output, structure=None):
    source.safe_output(output, clustering)
    fitted = require_clustering(clustering, source)
    intervals, exhaustive = None, False
    sources = {"input": source.signature, "clustering": binding(clustering)}
    if structure:
        _, _, intervals, sources["structure"] = load_structure(structure, source)
        exhaustive = inspect(structure)["identity"]["config"][
            "exhaustive_boundary_review"
        ]
    result = transition_tables(
        source, fitted["assignments"], config, intervals, exhaustive
    )
    result["occurrences"] = source.records.copy()
    source.unchanged()
    return publish(
        output,
        "sequence_cluster_transition_diagnostics_v1",
        config.model_dump(mode="json"),
        sources,
        result,
        {
            "runs": len(fitted["runs"]),
            "transitions": len(result["transitions"]),
            "experimental_diagnostic": True,
        },
    )


def suite(source, config, output, structure=None, progress=None, *, completed=None):
    notify = progress or (lambda message: None)
    supplied = completed or {}

    def stage(role, action):
        if role in supplied:
            path = supplied[role]
            if inspect(path)["identity"]["sources"].get("input") != source.signature:
                raise ValueError(
                    f"Completed suite stage belongs to different sources: {role}"
                )
            return path
        if completed is not None:
            raise ValueError(f"Missing completed suite stage: {role}")
        return action()

    artifacts, boundaries = {}, {}
    for i, boundary in enumerate(config.boundaries()):
        notify(f"Boundary configuration {i + 1}")
        path = stage(
            f"boundary_{i}",
            lambda boundary=boundary: run_boundary(source, boundary, output),
        )
        artifacts[f"boundary_{i}"] = path
        zones = pd.read_parquet(path / "candidate_zones.parquet")
        for (signal, policy), group in zones.groupby(
            ["signal", "encoder_policy"], sort=True
        ):
            boundaries[f"boundary_{i}:{signal}:{policy}"] = group
        # Empty candidate sets are meaningful ablations, not missing runs.
        for signal in ("adjacent", "multiscale"):
            for policy in ("clip", "dinov2", "consensus"):
                boundaries.setdefault(f"boundary_{i}:{signal}:{policy}", zones.iloc[:0])
        boundaries.setdefault(f"boundary_{i}:current_v1:consensus", zones.iloc[:0])
    notify("Clustering visual features on unique content")
    artifacts["clustering"] = stage(
        "clustering", lambda: run_clustering(source, config, output)
    )
    fitted = require_clustering(artifacts["clustering"], source)
    notify("Cluster transitions and post-hoc evaluation")
    artifacts["transitions"] = stage(
        "transitions",
        lambda: run_transitions(
            source, artifacts["clustering"], config.transitions, output, structure
        ),
    )
    transitions = tables(artifacts["transitions"])
    recurrence, intervals, exhaustive = None, None, False
    comparison = fitted["runs"].copy()
    if structure:
        _, _, intervals, _ = load_structure(structure, source)
        exhaustive = inspect(structure)["identity"]["config"][
            "exhaustive_boundary_review"
        ]
        artifacts["evaluation"] = stage(
            "evaluation",
            lambda: run_evaluate(source, artifacts["clustering"], structure, output),
        )
        artifacts["recurrence"] = stage(
            "recurrence",
            lambda: run_recurrence(
                source, structure, config.recurrence, output, artifacts["clustering"]
            ),
        )
        recurrence = tables(artifacts["recurrence"])
        metrics = tables(artifacts["evaluation"])["metrics"]
        comparison = comparison.merge(metrics, on="run_id", validate="one_to_many")
    result = {
        "comparison": comparison,
        "failures": fitted["failures"],
        "boundary_stability": interval_stability(boundaries),
        "stability": stability_tables(
            fitted["runs"],
            fitted["assignments"],
            transitions["transitions"],
            recurrence["cluster_recurrence"] if recurrence else None,
        ),
        **ablation_tables(
            boundaries,
            transitions["transitions"],
            intervals,
            config.transitions,
            exhaustive,
        ),
    }
    # Relative locators retain relocatability; each child is independently sealed.
    result["artifacts"] = pd.DataFrame(
        [
            {
                "role": role,
                "relative_directory": path.relative_to(output).as_posix(),
                **binding(path),
            }
            for role, path in sorted(artifacts.items())
        ]
    )
    sources = {
        "input": source.signature,
        "children": {role: binding(path) for role, path in sorted(artifacts.items())},
    }
    if structure:
        sources["structure"] = binding(structure)
    source.unchanged()
    notify("Publishing evidence comparison; no automatic winner")
    return publish(
        output,
        "sequence_experiment_suite_v1",
        config.model_dump(mode="json"),
        sources,
        result,
        {
            "successful_runs": len(fitted["runs"]),
            "failed_cells": len(fitted["failures"]),
            "all_requested_cells_succeeded": fitted["failures"].empty,
            "posthoc_review_available": structure is not None,
            "automatic_winner_selected": False,
            "real_data_conclusions": "not inferred",
        },
    )


def verify(
    directory: Path,
    source,
    structure=None,
    clustering=None,
    evidence=None,
    package=None,
):
    """Validate every byte/identity, live source binding and supplied dependencies."""
    try:
        meta = inspect(directory)
        dependencies = meta["identity"]["sources"]
        if dependencies.get("input") != source.signature:
            raise ValueError("Experiment input sources changed")
        if "evidence_set" in dependencies:
            members = evidence if isinstance(evidence, (list, tuple)) else []
            if [binding(path) for path in members] != dependencies["evidence_set"]:
                raise ValueError(
                    "Supply all exact combined review evidence members to verify"
                )
            if any(
                inspect(path)["identity"]["sources"].get("input") != source.signature
                for path in members
            ):
                raise ValueError("Combined review member belongs to different sources")
        for name, path in (("evidence", evidence), ("package", package)):
            if name in dependencies:
                if path is None or binding(path) != dependencies[name]:
                    raise ValueError(
                        f"Supply the exact review {name} artifact to verify"
                    )
                if (
                    inspect(path)["identity"]["sources"].get("input")
                    != source.signature
                ):
                    raise ValueError(f"Review {name} belongs to different sources")
        if "structure" in dependencies:
            if structure is None or binding(structure) != dependencies["structure"]:
                raise ValueError("Supply the exact structure review to verify")
            load_structure(structure, source)
        if "children" in dependencies:
            artifact_table = pd.read_parquet(directory / "artifacts.parquet")
            from flir_pipeline.data.local_images import relative_posix_path

            root = directory.parent.parent
            children = {}
            for row in artifact_table.itertuples():
                relative_posix_path(row.relative_directory)
                child = root / row.relative_directory
                if not child.resolve().is_relative_to(root.resolve()):
                    raise ValueError("Child artifact locator escapes experiment root")
                children[row.role] = child
                if binding(child) != dependencies["children"][row.role]:
                    raise ValueError("Suite child was modified")
            if set(children) != set(dependencies["children"]):
                raise ValueError("Incomplete suite child coverage")
            for child in children.values():
                checked = verify(child, source, structure, children.get("clustering"))
                if not checked["quality_valid"]:
                    raise ValueError(checked["error"])
        if "clustering" in dependencies:
            if clustering is None or binding(clustering) != dependencies["clustering"]:
                raise ValueError("Supply original clustering artifact to verify")
            require_clustering(clustering, source)
        if meta["artifact_kind"] == "sequence_structure_review_v1":
            load_structure(directory, source)
        if meta["artifact_kind"] == "sequence_manual_review_v1":
            from flir_pipeline.sequences.experiments.review import (
                verify_decision_history,
            )

            verify_decision_history(directory)
        source.unchanged()
        return {
            "quality_valid": True,
            "source_bound": True,
            "artifact_id": meta["artifact_id"],
            "verification_scope": "all file/semantic identities and source binding; not a stochastic refit",
            "ground_truth": False,
            "split_created": False,
            "automatic_confirmation": False,
        }
    except (ValueError, OSError, KeyError, AssertionError) as error:
        return {
            "quality_valid": False,
            "source_bound": False,
            "error": str(error),
            "ground_truth": False,
            "split_created": False,
            "automatic_confirmation": False,
        }
