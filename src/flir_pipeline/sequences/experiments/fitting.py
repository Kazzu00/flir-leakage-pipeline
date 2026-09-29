"""Reuse vector-only reduction/density adapters; labels enter only after fit."""

import json
from dataclasses import asdict

import numpy as np
import pandas as pd

from flir_pipeline.clustering.algorithms import effective_parameters, fit_clustering
from flir_pipeline.clustering.base import ClusteringConfig
from flir_pipeline.clustering.distances import EuclideanDistances
from flir_pipeline.reduction.reducers import make_reducer
from flir_pipeline.similarity.storage import stable_id


def canonical_labels(labels, contents):
    labels = np.asarray(labels, dtype=np.int32)
    groups = sorted(
        set(labels) - {-1},
        key=lambda c: min(
            x for x, lab in zip(contents, labels, strict=True) if lab == c
        ),
    )
    mapping = {-1: -1, **{old: new for new, old in enumerate(groups)}}
    return np.asarray([mapping[int(v)] for v in labels], dtype=np.int32)


def fit_visual(values, contents, algorithm, parameters):
    """No manifest, historical assignment, manual label or zone is accepted here."""
    if algorithm == "agglomerative":
        from sklearn.cluster import AgglomerativeClustering
        from threadpoolctl import threadpool_limits

        if parameters["n_clusters"] > len(contents):
            raise ValueError("Agglomerative n_clusters exceeds unique population")
        if len(set(contents)) != len(contents) or not np.isfinite(values).all():
            raise ValueError("Require finite unique-content input")
        order = np.argsort(contents, kind="stable")
        with threadpool_limits(limits=1):
            labels = AgglomerativeClustering(
                metric="euclidean", **parameters
            ).fit_predict(values[order].astype(np.float64))
        labels = labels[np.argsort(order)]
        details = {
            "adapter": "experimental_literature_agglomerative_v1",
            "effective_parameters": parameters,
            "preregistered_density_protocol": False,
            "metric": "euclidean",
        }
    else:
        config = ClusteringConfig(algorithm, parameters)
        effective = effective_parameters(config, EuclideanDistances(values))
        fitted = fit_clustering(values, contents, config, effective)
        labels = fitted.labels
        details = {k: v for k, v in fitted.metadata.items() if k != "fit_seconds"}
        details["adapter"] = "existing_content_density_v1"
    return canonical_labels(labels, contents), details


def clustering_tables(source, config, input_id, *, prepared=None, prepare_only=False):
    runs, assignments, coordinates, failures = [], [], [], []
    fit_details, space_details = {}, {}
    for encoder in config.encoders:
        spaces = [("original_l2", None)] if config.original_l2 else []
        spaces.extend(
            (grid.method, r) for grid in config.reductions for r in grid.expand()
        )
        for representation, reduction in spaces:
            reduction_config = asdict(reduction) if reduction else None
            space_id = stable_id(
                {
                    "input": input_id,
                    "encoder": encoder,
                    "representation": representation,
                    "reduction": reduction_config,
                }
            )
            values = source.embeddings[encoder]
            if prepared is not None and reduction:
                cached, metadata = prepared
                rows = cached.loc[cached.space_id.eq(space_id)]
                if rows.content_id.tolist() != source.contents:
                    raise ValueError(
                        "Prepared coordinates differ from canonical content order"
                    )
                values = np.asarray([json.loads(v) for v in rows.coordinates_json])
                if (
                    values.shape != (len(source.contents), reduction.output_dimension)
                    or not np.isfinite(values).all()
                ):
                    raise ValueError("Invalid prepared representation shape/values")
                # JSON float roundtrips preserve the original reducer dtype explicitly.
                values = values.astype(metadata[space_id]["coordinates_dtype"])
                space_details[space_id] = metadata[space_id]
                coordinates.extend(rows.to_dict("records"))
            elif reduction:
                # Canonical content order is established at loading, before any
                # stochastic reduction. Original spaces remain the evaluation control.
                try:
                    result = make_reducer(reduction).fit_transform(values)
                    values = result.coordinates
                    if not np.isfinite(values).all():
                        raise ValueError("Reducer returned nonfinite coordinates")
                    space_details[space_id] = {
                        k: v for k, v in result.metadata.items() if k != "fit_seconds"
                    }
                    space_details[space_id]["coordinates_dtype"] = str(values.dtype)
                    for i, content in enumerate(source.contents):
                        coordinates.append(
                            {
                                "space_id": space_id,
                                "content_id": content,
                                "coordinates_json": json.dumps(values[i].tolist()),
                            }
                        )
                except (ValueError, ImportError) as error:
                    failures.append(
                        {
                            "space_id": space_id,
                            "run_id": "",
                            "stage": "reduction",
                            "configuration_json": json.dumps(
                                reduction_config, sort_keys=True
                            ),
                            "error": str(error),
                        }
                    )
                    continue
            if prepare_only:
                continue
            for grid in config.clustering:
                for parameters in grid.expand():
                    spec = {
                        "input_id": input_id,
                        "encoder": encoder,
                        "representation": representation,
                        "space_id": space_id,
                        "reduction": reduction_config,
                        "algorithm": grid.algorithm,
                        "parameters": parameters,
                    }
                    run_id = stable_id(spec)
                    try:
                        labels, details = fit_visual(
                            values, source.contents, grid.algorithm, parameters
                        )
                    except ValueError as error:
                        failures.append(
                            {
                                "space_id": space_id,
                                "run_id": run_id,
                                "stage": "clustering",
                                "configuration_json": json.dumps(spec, sort_keys=True),
                                "error": str(error),
                            }
                        )
                        continue
                    fit_details[run_id] = details
                    runs.append(
                        {
                            "run_id": run_id,
                            "space_id": space_id,
                            "encoder": encoder,
                            "representation": representation,
                            "seed": reduction.seed if reduction else -1,
                            "algorithm": grid.algorithm,
                            "parameters_json": json.dumps(parameters, sort_keys=True),
                            "reduction_json": json.dumps(
                                reduction_config, sort_keys=True
                            ),
                            "experimental_adapter": grid.algorithm == "agglomerative",
                        }
                    )
                    assignments.extend(
                        {
                            "run_id": run_id,
                            "content_id": content,
                            "cluster_id": int(label),
                        }
                        for content, label in zip(source.contents, labels, strict=True)
                    )
    tables = {
        "runs": pd.DataFrame(
            runs,
            columns=[
                "run_id",
                "space_id",
                "encoder",
                "representation",
                "seed",
                "algorithm",
                "parameters_json",
                "reduction_json",
                "experimental_adapter",
            ],
        ),
        "assignments": pd.DataFrame(
            assignments, columns=["run_id", "content_id", "cluster_id"]
        ),
        "coordinates": pd.DataFrame(
            coordinates, columns=["space_id", "content_id", "coordinates_json"]
        ),
        "failures": pd.DataFrame(
            failures,
            columns=["space_id", "run_id", "stage", "configuration_json", "error"],
        ),
        "occurrences": source.records.copy(),
    }
    return tables, {"fits": fit_details, "spaces": space_details}
