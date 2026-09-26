"""Synthetic scientific snapshot shared by the HEAD-737e588 regression tests.

No runtime, paths, receipts or private data enter this snapshot. The checked-in
oracle was captured before changing clustering at 737e588; rebuilding it is an
explicit scientific review, never an automatic test update.
"""

import hashlib
import json

import numpy as np
import pandas as pd

from flir_pipeline.clustering.base import ClusteringConfig
from flir_pipeline.similarity.storage import read_json


def configs():
    return [
        *(ClusteringConfig("dbscan", {"min_samples": 5, "eps_quantile": q}) for q in (.7, .8)),
        *(ClusteringConfig("optics", {"min_samples": 5, "min_cluster_size": 10, "xi": xi}) for xi in (.03, .05)),
        *(ClusteringConfig("hdbscan", {"min_samples": ms, "min_cluster_size": 10}) for ms in (3, 5)),
    ]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def snapshot(screening, comparison):
    root = screening.parents[1]
    meta = read_json(comparison/"metadata.json")
    runs = {}
    for ref in meta["runs"]:
        directory = root/ref["path"]
        run = read_json(directory/"metadata.json")
        key = f"{run['representation']}:{run['reduction_seed']}:{run['configuration_id']}"
        diagnostics = {}
        if (directory/"algorithm_diagnostics.npz").exists():
            with np.load(directory/"algorithm_diagnostics.npz") as arrays:
                diagnostics = {k: hashlib.sha256(arrays[k].tobytes()).hexdigest() for k in arrays.files}
        runs[key] = {
            "clustering_space_id": run["clustering_space_id"],
            "effective_parameters": run["effective_parameters"],
            "labels": np.load(directory/"cluster_labels.npy").tolist(),
            "metrics": read_json(directory/"metrics.json"),
            "summary": json.loads(pd.read_parquet(directory/"cluster_summary.parquet").to_json(orient="records", double_precision=15)),
            "probabilities": (np.load(directory/"membership_probabilities.npy").tolist()
                              if (directory/"membership_probabilities.npy").exists() else None),
            "diagnostics_sha256": diagnostics,
        }
    tables = {}
    for directory, names in ((screening, ("screening.csv", "shortlist.csv", "k_distance_quantiles.csv", "redundant_or_nonexecuted.csv")),
                             (comparison, ("evaluated_shortlist.csv", "candidates.csv", "references.csv"))):
        for name in names:
            frame = pd.read_csv(directory/name).drop(columns=["fit_seconds", "evaluation_seconds"], errors="ignore")
            tables[name] = json.loads(frame.to_json(orient="records", double_precision=15))
    tables["assignment_comparisons"] = json.loads(pd.read_parquet(comparison/"assignment_comparisons.parquet").to_json(orient="records", double_precision=15))
    return {"runs": runs, "tables": tables}
