"""Read saved clustering receipts without traversing or evaluating scientific runs.

Checksums check byte integrity relative to the selected metadata, not scientific
correctness or source provenance. Even NPY/Parquet outputs are only streamed as
bytes: no arrays, assignments, embeddings or distance matrices are loaded.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import pandas as pd

from flir_pipeline.clustering.base import ClusteringConfig, clustering_space_id
from flir_pipeline.clustering.selection import SELECTION_POLICY
from flir_pipeline.clustering.storage import RUN_FILES
from flir_pipeline.similarity.storage import file_sha256, read_json, stable_id

COLLECTION_FILES = {
    "clustering_screening": {
        "summary.json", "screening.csv", "shortlist.csv",
        "redundant_or_nonexecuted.csv", "k_distance_quantiles.csv",
    },
    "clustering_comparison": {
        "summary.json", "evaluated_shortlist.csv", "candidates.csv",
        "references.csv", "assignment_comparisons.parquet", "all_runs.csv",
    },
}
TEXT_COLUMNS = (
    "encoder", "representation", "algorithm", "clustering_space_id",
    "configuration_id", "shortlist_reason", "temporal_provenance_mode",
    "temporal_evaluation", "historical_split_evaluation",
)
PREVIEW_COLUMNS = (
    *TEXT_COLUMNS[:5], "reduction_seed", "n_clusters_excluding_noise",
    "noise_fraction", "silhouette_original_space",
    "weighted_mean_intra_cluster_similarity", "visual_neighbor_coherence@10",
    "temporal_recall@5", "weighted_dominant_sequence_fraction",
    "weighted_sequence_entropy_bits", "historical_multisplit_clusters",
    "parameters_common_clustered_ari_min", "parameters_common_clustered_ami_min",
    "seeds_common_clustered_ari_min", "seeds_common_clustered_ami_min",
    "reference_for_review", *TEXT_COLUMNS[5:],
)
METADATA_FIELDS = (
    "artifact_kind", "collection_id", "clustering_space_id", "dataset_id",
    "feature_space_id", "extractor", "model_id", "N", "input_dimension",
    "representation", "reduction_space_id", "reduction_seed", "configuration_id",
    "algorithm", "config", "effective_parameters", "implementation_versions",
    "evaluation_protocol", "provenance_mode", "unavailable_metric_groups",
    "noise_policy", "record_mapping", "selection_policy", "screening_path",
    "screening_metadata_sha256",
)


def _local_file(directory: Path, name: str) -> Path:
    # Receipts must never redirect this read-only command into another run/source.
    if (not isinstance(name, str) or not name or name in {".", ".."}
            or any(char in name for char in "/\\:\x00")):
        raise ValueError("Invalid output filename in clustering metadata")
    path = directory / name
    if not path.resolve().is_relative_to(directory.resolve()):
        raise ValueError(f"Inspection file escapes selected directory: {name}")
    if not path.is_file():
        raise ValueError(f"Missing inspection file: {name}")
    return path


def _read_object(path: Path) -> dict:
    try:
        value = read_json(path)
        if not isinstance(value, dict):
            raise ValueError("expected a JSON object")
        json.dumps(value, allow_nan=False)
    except (ValueError, UnicodeError) as error:
        raise ValueError(f"Invalid JSON in {path.name}: {error}") from error
    return value


def _digest_valid(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _validate_metadata(meta: dict) -> None:
    """Check identities from recorded settings only; never resolve referenced paths."""
    kind = meta.get("artifact_kind")
    if not isinstance(kind, str) or kind not in {"clustering_run", *COLLECTION_FILES}:
        raise ValueError("Unknown clustering artifact_kind in metadata.json")
    try:
        if kind == "clustering_run":
            config = ClusteringConfig(**meta["config"])
            expected = clustering_space_id(
                meta["dataset_id"], meta["feature_space_id"], meta["representation"],
                meta["reduction_space_id"], config, meta["effective_parameters"],
                meta["implementation_versions"],
            )
            if (meta["clustering_space_id"] != expected
                    or meta["configuration_id"] != config.configuration_id
                    or meta["algorithm"] != config.algorithm):
                raise ValueError("run identity/configuration mismatch")
            if type(meta["N"]) is not int or meta["N"] < 1:
                raise ValueError("N must be a positive integer")
            optional = meta["optional_files"]
            if (not isinstance(optional, list) or any(not isinstance(v, str) for v in optional)
                    or len(set(optional)) != len(optional) or set(optional) & set(RUN_FILES)):
                raise ValueError("invalid optional_files")
            required = {*RUN_FILES, *optional}
        else:
            if meta["selection_policy"] != SELECTION_POLICY:
                raise ValueError("selection_policy mismatch")
            runs = meta["runs"]
            if not isinstance(runs, list) or not runs:
                raise ValueError("runs must be a nonempty list of metadata references")
            for run in runs:
                if (not isinstance(run, dict)
                        or any(not isinstance(run.get(k), str) or not run[k]
                               for k in ("path", "encoder", "clustering_space_id"))
                        or not _digest_valid(run.get("metadata_sha256"))):
                    raise ValueError("invalid recorded run reference")
            ids = [run["clustering_space_id"] for run in runs]
            if len(set(ids)) != len(ids):
                raise ValueError("duplicate recorded run IDs")
            if kind == "clustering_screening":
                payload = {"runs": sorted(ids), "policy": SELECTION_POLICY}
            else:
                if (not _digest_valid(meta["screening_metadata_sha256"])
                        or not isinstance(meta["screening_path"], str) or not meta["screening_path"]):
                    raise ValueError("invalid recorded screening reference")
                payload = {"screening_metadata_sha256": meta["screening_metadata_sha256"],
                           "policy": SELECTION_POLICY}
            if meta["collection_id"] != stable_id(payload):
                raise ValueError("collection identity mismatch")
            required = COLLECTION_FILES[kind]
        outputs = meta["output_sha256"]
        if not isinstance(outputs, dict) or not required <= outputs.keys():
            missing = sorted(required - outputs.keys()) if isinstance(outputs, dict) else sorted(required)
            raise ValueError(f"missing output checksums: {', '.join(missing)}")
        if kind == "clustering_run" and set(outputs) != required:
            raise ValueError("output checksums differ from declared run files")
        if "metadata.json" in outputs or any(not _digest_valid(v) for v in outputs.values()):
            raise ValueError("invalid output SHA256 record")
    except (KeyError, TypeError, AttributeError, ValueError) as error:
        raise ValueError(f"Invalid metadata.json: {error}") from error


def _preview(path: Path, limit: int) -> dict:
    """Bound display memory and preserve saved order; missing CSV metrics become None."""
    with path.open(encoding="utf-8", newline="") as stream:
        header = next(csv.reader(stream), [])
    if len(set(header)) != len(header) or not set(TEXT_COLUMNS[:5]) <= set(header):
        raise ValueError(f"Invalid columns in {path.name}")
    columns = [name for name in PREVIEW_COLUMNS if name in header]
    rows, count = [], 0
    # IDs must remain strings even when a hexadecimal ID happens to be all digits.
    dtypes = {name: str for name in TEXT_COLUMNS if name in columns}
    try:
        with pd.read_csv(path, usecols=columns, dtype=dtypes, chunksize=256) as chunks:
            for chunk in chunks:
                count += len(chunk)
                shown = chunk.loc[:, columns].head(max(0, limit - len(rows)))
                rows.extend(shown.astype(object).where(shown.notna(), None).to_dict("records"))
        json.dumps(rows, allow_nan=False)
    except (ValueError, UnicodeError) as error:
        raise ValueError(f"Invalid table {path.name}: {error}") from error
    return {"file": path.name, "row_count": count, "shown_rows": len(rows),
            "truncated": count > len(rows), "columns": columns, "rows": rows}


def inspect_clustering(directory: Path, *, limit: int = 10) -> dict:
    """Inspect one completed publication, without claiming full verification.

    metadata.json is the completion marker. All its output checksums are checked
    using bounded byte reads, including binary outputs. Only saved JSON and small
    collection CSV views are decoded. Referenced runs, screening and input sources
    may be offline; neither their existence nor scientific validity is checked.
    """
    if type(limit) is not int or limit < 1:
        raise ValueError("Inspection limit must be a positive integer")
    meta = _read_object(_local_file(directory, "metadata.json"))
    _validate_metadata(meta)
    checked = []
    for name, digest in sorted(meta["output_sha256"].items()):
        if file_sha256(_local_file(directory, name)) != digest:
            raise ValueError(f"Output checksum mismatch: {name}")
        checked.append(name)
    result = {
        "inspection_mode": "lightweight",
        "notice": "Lightweight inspection only; NOT full scientific verification.",
        "validation": {"selected_metadata_valid": True,
                       "recorded_output_checksums_valid": True,
                       "checked_output_files": checked,
                       "scientific_verification_performed": False},
        "not_checked": ["referenced runs and screening/source bindings",
                        "scientific arrays, assignments, indices and medoids",
                        "per-run metrics, ARI/AMI and selection recomputation"],
        "metadata": {key: meta[key] for key in METADATA_FIELDS if key in meta},
    }
    kind = meta["artifact_kind"]
    if kind == "clustering_run":
        result["metrics"] = _read_object(_local_file(directory, "metrics.json"))
    else:
        result["metadata"]["recorded_run_count"] = len(meta["runs"])
        result["summary"] = _read_object(_local_file(directory, "summary.json"))
        tables = ("shortlist",) if kind == "clustering_screening" else ("references", "candidates")
        for table in tables:
            result[table] = _preview(_local_file(directory, f"{table}.csv"), limit)
    return result
