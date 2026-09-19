"""ID-based joins that preserve occurrence mappings and historical memberships."""

from __future__ import annotations

import numpy as np
import pandas as pd

from flir_pipeline.data.identity import dataset_id_from_manifest
from flir_pipeline.explorer.discovery import checked_file, read_json
from flir_pipeline.explorer.models import ClusterData, Run, SplitData
from flir_pipeline.similarity.temporal import build_content_provenance

SPLITS = ("train", "val", "test")


def load_cluster(run: Run, manifest: pd.DataFrame) -> ClusterData:
    """Load saved labels and summary; temporal consensus reuses the existing rule."""
    if run.kind != "clustering_run" or dataset_id_from_manifest(manifest) != run.dataset_id:
        raise ValueError("Clustering and manifest dataset identities differ")
    index = pd.read_parquet(checked_file(run, "content_index.parquet"))
    labels = np.load(checked_file(run, "cluster_labels.npy"), allow_pickle=False)
    summary = pd.read_parquet(checked_file(run, "cluster_summary.parquet"))
    metrics = read_json(checked_file(run, "metrics.json"))
    checked_file(run, "quality.json")
    if (not index.content_id.is_unique or index.content_id.isna().any()
            or set(index.content_id) != set(manifest.content_id)
            or not np.array_equal(index.embedding_row, np.arange(len(index)))
            or labels.shape != (len(index),) or labels.dtype.kind not in "iu"
            or (labels < -1).any() or run.metadata["N"] != len(index)):
        raise ValueError("Invalid content-level label/index mapping")
    representatives = manifest.set_index("frame_id").reindex(index.representative_frame_id)
    for column in ("content_id", "source_archive", "source_member_path", "image_sha256"):
        if not np.array_equal(representatives[column].to_numpy(), index[column].to_numpy()):
            raise ValueError("Representative frame does not match the canonical occurrence")
    if not summary.cluster_id.is_unique or set(summary.cluster_id) != set(labels) - {-1}:
        raise ValueError("Cluster summary does not cover labels")
    for row in summary.itertuples():
        members = set(index.loc[labels == row.cluster_id, "content_id"])
        if row.n_members != len(members) or row.medoid_content_id not in members:
            raise ValueError("Saved cluster summary membership mismatch")
    contents, records = build_content_provenance(manifest, index)
    contents["cluster_id"] = labels
    records["cluster_id"] = records.content_id.map(contents.set_index("content_id").cluster_id)
    return ClusterData(run, contents, records, summary, metrics)


def load_split(run: Run, manifest: pd.DataFrame) -> SplitData:
    """Preserve historical occurrences; never reduce multi-membership to one split."""
    if run.kind != "split_run" or dataset_id_from_manifest(manifest) != run.dataset_id:
        raise ValueError("Split and manifest dataset identities differ")
    records = pd.read_parquet(checked_file(run, "record_split_assignments.parquet"))
    groups = pd.read_parquet(checked_file(run, "source_groups.parquet"))
    checked_file(run, "quality.json")
    if (not records.frame_id.is_unique or set(records.frame_id) != set(manifest.frame_id)
            or not records.new_split.isin(SPLITS).all() or not groups.content_id.is_unique
            or set(groups.content_id) != set(manifest.content_id) or groups.group_id.isna().any()):
        raise ValueError("Incomplete or ambiguous split assignments")
    expected = manifest.set_index("frame_id").loc[records.frame_id]
    for column in ("content_id", "original_split"):
        if not np.array_equal(expected[column].to_numpy(), records[column].to_numpy()):
            raise ValueError("Split occurrence mapping differs from manifest")
    if run.strategy == "historical":
        if not records.new_split.equals(records.original_split):
            raise ValueError("Historical assignments changed")
    else:
        if records.groupby("content_id").new_split.nunique().gt(1).any():
            raise ValueError("New split fractures exact content")
        joined = records.merge(groups[["content_id", "group_id"]], on="content_id", validate="many_to_one")
        if joined.groupby("group_id").new_split.nunique().gt(1).any():
            raise ValueError("New split fractures source group")
    return SplitData(run, records, groups)


def with_split(cluster: ClusterData, split: SplitData | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return independent display tables; no mutation of loaded artifacts."""
    contents, records = cluster.contents.copy(), cluster.records.copy()
    if split is None:
        contents["new_splits"] = [tuple() for _ in range(len(contents))]
        records["new_split"] = "unassigned"
        return contents, records
    if split.run.dataset_id != cluster.run.dataset_id:
        raise ValueError("Cannot overlay different datasets")
    if split.run.strategy == "cluster_aware":
        if split.run.clustering_space_id != cluster.run.space_id:
            raise ValueError("Cannot overlay a different source clustering")
        actual = split.groups.set_index("content_id").loc[contents.content_id, "cluster_id"]
        if not np.array_equal(actual.to_numpy(), contents.cluster_id.to_numpy()):
            raise ValueError("Source cluster memberships differ")
        noise = split.groups.loc[split.groups.cluster_id.eq(-1)]
        group_sizes = split.groups.groupby("group_id").size()
        if (noise.group_id.duplicated().any() or not noise.group_type.eq("noise_singleton").all()
                or noise.group_id.map(group_sizes).ne(1).any()):
            raise ValueError("Noise must remain singleton without invented clusters")
        memberships = split.records.merge(split.groups[["content_id", "cluster_id"]], on="content_id", validate="many_to_one")
        if memberships.loc[memberships.cluster_id.ge(0)].groupby("cluster_id").new_split.nunique().gt(1).any():
            raise ValueError("Cluster-aware split fractures a non-noise cluster")
    records = records.merge(split.records[["frame_id", "new_split"]], on="frame_id", validate="one_to_one")
    memberships = records.groupby("content_id").new_split.agg(lambda x: tuple(s for s in SPLITS if s in set(x)))
    contents["new_splits"] = contents.content_id.map(memberships)
    contents = contents.merge(split.groups[["content_id", "group_id", "group_type"]], on="content_id", validate="one_to_one")
    return contents, records


def filter_split(contents: pd.DataFrame, split_name: str) -> pd.DataFrame:
    if split_name not in SPLITS:
        raise ValueError("Unknown split")
    return contents.loc[contents.new_splits.map(lambda values: split_name in values)].copy()


def cluster_members(contents: pd.DataFrame, cluster_id: int) -> pd.DataFrame:
    return contents.loc[contents.cluster_id.eq(cluster_id)].copy()


def compare_partitions(cluster: ClusterData, left: SplitData, right: SplitData) -> pd.DataFrame:
    """One row per historical occurrence, including duplicate cross-split records."""
    if {left.run.dataset_id, right.run.dataset_id} != {cluster.run.dataset_id}:
        raise ValueError("Partition comparison requires the same dataset")
    rows = cluster.records.copy()
    for side, split in (("left", left), ("right", right)):
        assigned = split.records[["frame_id", "new_split"]].rename(columns={"new_split": side})
        rows = rows.merge(assigned, on="frame_id", validate="one_to_one")
        group = split.groups[["content_id", "group_id", "cluster_id"]].rename(
            columns={"group_id": f"{side}_group", "cluster_id": f"{side}_cluster"})
        rows = rows.merge(group, on="content_id", validate="many_to_one")
    return rows
