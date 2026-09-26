"""Immutable split runs, verified existing sources, and executable artifact checks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from flir_pipeline.clustering.experiments import verify_collection
from flir_pipeline.clustering.storage import verify_run as verify_cluster
from flir_pipeline.data.annotations import audit_annotations
from flir_pipeline.data.identity import dataset_id_from_manifest
from flir_pipeline.data.video_temporal import temporal_mode
from flir_pipeline.reduction.storage import load_inputs as load_visual_inputs
from flir_pipeline.similarity.storage import (
    execution_provenance,
    file_sha256,
    read_json,
    stable_id,
    write_json,
)
from flir_pipeline.splitting.base import (
    SplitConfig,
    identity_payload,
    split_space_id,
    targets_from_manifest,
)
from flir_pipeline.splitting.construction import (
    aggregate_groups,
    make_groups,
    milp_assignment,
    propagate,
    random_assignment,
    record_statistics,
)
from flir_pipeline.splitting.metrics import (
    balance_metrics,
    membership_masks,
    residual_similarity,
    temporal_metrics,
    verify_assignments,
)
from flir_pipeline.splitting.selection import select_clustering_candidates


@dataclass
class SimilaritySource:
    content_ids: list[str]
    matrix: np.ndarray
    neighbors: pd.DataFrame
    thresholds: pd.DataFrame
    provenance: pd.DataFrame
    metadata: dict


@dataclass
class SplitInputs:
    manifest: pd.DataFrame
    statistics: pd.DataFrame
    dataset_id: str
    similarities: dict[str, SimilaritySource]
    candidates: pd.DataFrame
    cluster_paths: dict[str, Path]
    signatures: dict
    annotation_summary: dict


def splitting_provenance() -> dict:
    provenance = execution_provenance()
    provenance["similarity_source_sha256"] = provenance.pop("source_sha256")
    provenance["source_sha256"] = {p.name: file_sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    return provenance


def load_inputs(spec_path: Path, comparison: Path, labels_archive: Path) -> SplitInputs:
    """Reuse the existing input spec and verify both full original spaces without fitting."""
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    if set(spec["encoders"]) != {"dinov2", "clip"}:
        raise ValueError("Splitting evaluation requires both DINOv2 and CLIP")
    manifest_path = Path(spec["manifest"])
    manifest = pd.read_parquet(manifest_path)
    if temporal_mode(manifest) == "sampled_video_grid":
        raise ValueError("Video splitting is out of scope: sequence groups and audited labels are unavailable")
    if not manifest.label_valid.all() or not manifest.label_exists.all():
        raise ValueError("Splitting needs fully audited canonical annotations")
    audit = audit_annotations(manifest, labels_archive)
    statistics = record_statistics(manifest, audit.instances)
    dataset_id = dataset_id_from_manifest(manifest)
    similarities, signatures = {}, {"manifest": file_sha256(manifest_path), "similarities": {}}
    sources = {}
    for encoder, paths in spec["encoders"].items():
        directory = Path(paths["similarity_directory"])
        source = load_visual_inputs(Path(paths["feature_directory"]), directory, manifest_path)
        if source.feature["extractor"] != encoder or source.feature["dataset_id"] != dataset_id:
            raise ValueError("Encoder or dataset source mismatch")
        thresholds = pd.read_csv(directory/"quantile_candidates.csv", float_precision="round_trip")
        if not np.allclose(thresholds["quantile"], [.9, .95, .975, .99, .995, .999], atol=1e-12, rtol=0):
            raise ValueError("The six precomputed quantile cohorts are required")
        similarities[encoder] = SimilaritySource(source.content_index.content_id.tolist(), source.cosine, source.original_neighbors,
                                                  thresholds, pd.read_parquet(directory/"content_provenance.parquet"), source.similarity)
        signatures["similarities"][encoder] = {**source.signatures, "thresholds": file_sha256(directory/"quantile_candidates.csv")}
        sources[encoder] = source
    if not verify_collection(comparison)["quality_valid"]:
        raise ValueError("Existing clustering Pareto collection failed verification")
    collection = read_json(comparison/"metadata.json")
    if collection["source_signatures"] != {e: s.signatures for e, s in sources.items()}:
        raise ValueError("Clustering collection does not match original feature/similarity sources")
    candidates = select_clustering_candidates(pd.read_csv(comparison/"candidates.csv"))
    root = comparison.parents[1]
    paths = {r["clustering_space_id"]: root/r["path"] for r in collection["runs"]}
    paths = {i: paths[i] for i in candidates.clustering_space_id}
    signatures["clustering_collection"] = file_sha256(comparison/"metadata.json")
    signatures["clustering_runs"] = {i: file_sha256(p/"metadata.json") for i, p in paths.items()}
    # Consensus must agree across encoders after an ID join, not by row position.
    columns = ["sequence_key", "sequence_provenance_valid", "frame_index", "frame_index_valid"]
    pd.testing.assert_frame_equal(similarities["dinov2"].provenance.set_index("content_id")[columns].sort_index(),
                                  similarities["clip"].provenance.set_index("content_id")[columns].sort_index())
    return SplitInputs(manifest, statistics, dataset_id, similarities, candidates, paths, signatures, audit.summary)


def cluster_labels(inputs: SplitInputs, clustering_id: str) -> pd.Series:
    directory = inputs.cluster_paths[clustering_id]
    if not verify_cluster(directory)["quality_valid"]:
        raise ValueError("Clustering source failed verification")
    meta = read_json(directory/"metadata.json")
    if meta["dataset_id"] != inputs.dataset_id or meta["clustering_space_id"] != clustering_id:
        raise ValueError("Clustering belongs to another dataset/identity")
    index = pd.read_parquet(directory/"content_index.parquet")
    return pd.Series(np.load(directory/"cluster_labels.npy", allow_pickle=False), index=index.content_id, name="cluster_id")


def evaluate_records(inputs: SplitInputs, records: pd.DataFrame, ratios: np.ndarray) -> tuple[dict, dict[str, pd.DataFrame]]:
    balance, class_table = balance_metrics(records, inputs.statistics, ratios)
    metrics, tables = {"balance": balance, "visual": {}}, {"class_balance.parquet": class_table}
    for encoder, source in inputs.similarities.items():
        masks = membership_masks(records, source.content_ids)
        result, quantiles, nn = residual_similarity(source.matrix, source.neighbors, source.thresholds, masks, source.content_ids)
        metrics["visual"][encoder] = {**result, "similarity_space_id": source.metadata["similarity_space_id"]}
        tables[f"quantile_pairs_{encoder}.parquet"] = quantiles
        tables[f"cross_split_nn_{encoder}.parquet"] = nn
    temporal = inputs.similarities["dinov2"]
    metrics["temporal"], tables["temporal_cross_split.parquet"] = temporal_metrics(temporal.provenance, membership_masks(records, temporal.content_ids))
    return metrics, tables


def build_run(inputs: SplitInputs, config: SplitConfig, seed: int,
              output_root: Path, clustering_id: str | None = None) -> Path:
    if (config.strategy == "cluster_aware") != (clustering_id is not None):
        raise ValueError("A clustering ID is required exactly for cluster-aware runs")
    ratios = targets_from_manifest(inputs.manifest, config)
    historical_id = stable_id({"membership": inputs.manifest[["frame_id", "original_split"]].sort_values("frame_id").values.tolist()})
    payload = identity_payload(inputs.dataset_id, config, ratios, seed, clustering_id, historical_id)
    run_id = split_space_id(payload)
    output = output_root/"runs"/run_id
    if output.exists():
        if not verify_split(output, inputs)["quality_valid"]:
            raise ValueError("Incomplete/incompatible split output preserved; use a separate root")
        return output
    labels = cluster_labels(inputs, clustering_id) if clustering_id else None
    groups = make_groups(sorted(inputs.manifest.content_id.unique()), labels, config.noise_policy)
    units = aggregate_groups(inputs.statistics, groups)
    if config.strategy == "historical":
        records = inputs.statistics[["frame_id", "content_id", "original_split"]].copy()
        records["new_split"] = records.original_split
        contents = groups.copy()
        memberships = records.groupby("content_id").new_split.agg(lambda v: sorted(set(v)))
        contents["new_split"] = contents.content_id.map(memberships.map(lambda v: v[0] if len(v) == 1 else None))
        contents["split_membership_set"] = contents.content_id.map(memberships.map(json.dumps))
        contents["group_type"] = "historical_content"
        solver = {"method": "preserve_all_historical_occurrences", "seed": None}
    else:
        assignments, solver = (random_assignment(units, ratios, seed) if config.strategy == "random_content"
                               else milp_assignment(units, ratios, seed, config))
        contents, records = propagate(inputs.statistics, groups, assignments)
    quality = verify_assignments(contents, records, inputs.manifest, config.strategy, labels)
    metrics, tables = evaluate_records(inputs, records, ratios)
    output.mkdir(parents=True, exist_ok=False)
    contents.to_parquet(output/"split_assignments.parquet", index=False)
    records.to_parquet(output/"record_split_assignments.parquet", index=False)
    inputs.statistics.to_parquet(output/"record_statistics.parquet", index=False)
    groups.to_parquet(output/"source_groups.parquet", index=False)
    units.to_parquet(output/"group_balance.parquet", index=False)
    for name, table in tables.items():
        table.to_parquet(output/name, index=False)
    write_json(output/"split_summary.json", {**metrics, "quality": quality})
    write_json(output/"quality.json", quality)
    meta = {**splitting_provenance(), "artifact_kind": "split_run", "split_space_id": run_id,
            "identity_payload": payload, "input_signatures": inputs.signatures, "solver": solver,
            "evaluation_protocol": "unique_content_existential_cross_membership_v1",
            "annotation_audit": inputs.annotation_summary,
            "output_sha256": {p.name: file_sha256(p) for p in output.iterdir() if p.is_file()}}
    write_json(output/"metadata.json", meta)
    if not verify_split(output, inputs)["quality_valid"]:
        raise ValueError("Split failed publication verification; artifacts preserved")
    return output


RUN_FILES = {"split_assignments.parquet", "record_split_assignments.parquet", "record_statistics.parquet",
             "source_groups.parquet", "group_balance.parquet", "split_summary.json", "quality.json", "class_balance.parquet",
             "quantile_pairs_dinov2.parquet", "quantile_pairs_clip.parquet", "cross_split_nn_dinov2.parquet",
             "cross_split_nn_clip.parquet", "temporal_cross_split.parquet"}


def verify_split(directory: Path, inputs: SplitInputs | None = None) -> dict:
    checks = {}
    try:
        meta = read_json(directory/"metadata.json")
        payload = meta["identity_payload"]
        config = SplitConfig(**payload["configuration"])
        checks["identity_valid"] = meta["split_space_id"] == split_space_id(payload)
        checks["artifact_kind_valid"] = meta["artifact_kind"] == "split_run"
        checks["checksums_valid"] = set(meta["output_sha256"]) == RUN_FILES and all(file_sha256(directory/name) == meta["output_sha256"][name] for name in RUN_FILES)
        contents, records = (pd.read_parquet(directory/name) for name in ("split_assignments.parquet", "record_split_assignments.parquet"))
        statistics = pd.read_parquet(directory/"record_statistics.parquet")
        groups = pd.read_parquet(directory/"source_groups.parquet")
        labels = groups.set_index("content_id").cluster_id.astype(int) if config.strategy == "cluster_aware" else None
        quality = verify_assignments(contents, records, statistics, config.strategy, labels)
        checks["quality_recomputed"] = quality == read_json(directory/"quality.json")
        rebuilt = aggregate_groups(statistics, groups)
        pd.testing.assert_frame_equal(rebuilt, pd.read_parquet(directory/"group_balance.parquet"))
        checks["group_balance_recomputed"] = True
        summary = read_json(directory/"split_summary.json")
        balance, class_table = balance_metrics(records, statistics, np.asarray(config.target_ratios))
        checks["balance_recomputed"] = balance == summary["balance"] and summary["quality"] == quality
        pd.testing.assert_frame_equal(class_table, pd.read_parquet(directory/"class_balance.parquet"))
        if inputs is not None:
            checks["sources_bound"] = meta["input_signatures"] == inputs.signatures and payload["dataset_id"] == inputs.dataset_id
            pd.testing.assert_frame_equal(statistics, inputs.statistics)
            expected_labels = cluster_labels(inputs, payload["clustering_space_id"]) if config.strategy == "cluster_aware" else None
            verify_assignments(contents, records, inputs.manifest, config.strategy, expected_labels)
            metrics, tables = evaluate_records(inputs, records, np.asarray(config.target_ratios))
            checks["all_metrics_recomputed"] = {**metrics, "quality": quality} == summary
            for name, table in tables.items():
                pd.testing.assert_frame_equal(table, pd.read_parquet(directory/name))
        checks["quality_valid"] = all(checks.values())
    except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError, AssertionError) as error:
        checks.update(quality_valid=False, error_type=type(error).__name__)
    return checks
