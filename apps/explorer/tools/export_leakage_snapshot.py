# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.2.6", "pandas==2.2.3", "pyarrow==20.0.0"]
# ///
"""Read-only, allowlisted bridge. Never imports or executes the scientific pipeline.

Checks the bytes consumed, identity continuity and linear-size numerical/index
invariants. It does NOT certify upstream experiments or recompute scientific metrics.
Unknown strings (including names, paths and metadata descriptions) never cross
the boundary. Aliases are deterministic within a manifest, not public hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

POLICY = "allowlist-v1; all contents; declared candidates; consumed checksums; no scientific recomputation"
STATS = re.compile(r"^[a-zA-Z][a-zA-Z0-9_@. -]{0,90}$")
# Keys are allowlisted: syntactically harmless private identifiers are also private.
METRICS = {
    "count",
    "mean",
    "std",
    "min",
    "max",
    "Q1",
    "median",
    "Q3",
    "p90",
    "p95",
    "p97_5",
    "p99",
    "p99_5",
    "p99_9",
    "pair_count",
    "n_pairs",
    "query_count",
    "neighbor_count",
    "top_k",
    "k",
    "source_video_count",
    "record_count",
    "same_source_video_pairs",
    "disjoint_source_video_pairs",
    "near_unit_pair_count",
    "mean_cosine",
    "median_cosine",
    "spearman_distance",
    "trustworthiness_mean",
    "continuity_mean",
    "preservation_mean",
    "stability_mean",
    "spearman_mean",
    "fit_seconds",
    "fit_seconds_total",
    "seed_count",
    "mean_criterion_rank",
    "perplexity",
    "learning_rate",
    "early_exaggeration",
    "n_neighbors",
    "MN_ratio",
    "FP_ratio",
    "n_components",
    "eps",
    "min_samples",
    "min_cluster_size",
    "xi",
    "max_eps",
    "precision",
    "recall",
    "map50",
    "map50_95",
    "image_count",
    "n_clusters",
    "noise_count",
    "noise_fraction",
    "silhouette_cosine",
    "davies_bouldin",
    "calinski_harabasz",
    "ARI",
    "AMI",
    "ari",
    "ami",
    "n_members",
    "mean_intra_cosine",
    "mean_same_source_video",
    "sample_index_gap",
    "timestamp_gap_seconds",
    "lower",
    "upper",
    "bin_start",
    "bin_end",
    "train",
    "val",
    "validation",
    "test",
    "lower_exclusive",
    "upper_inclusive",
    "quantile",
    "top_percentage",
    "threshold_cosine",
    "same_source_video_count",
    "disjoint_source_video_count",
    "total_points",
    "clustered_points",
    "noise_points",
    "n_clusters_excluding_noise",
    "largest_cluster_fraction",
    "singleton_cluster_count",
    "silhouette_original_space",
    "silhouette_clustering_space",
    "silhouette_population",
    "visual_query_coverage",
    "weighted_mean_intra_cluster_similarity",
    "member_weighted_intra_cluster_similarity",
    "median_cluster_similarity",
    "cluster_size_min",
    "cluster_size_q1",
    "cluster_size_median",
    "cluster_size_mean",
    "cluster_size_q3",
    "cluster_size_max",
}
FEATURE_FILES = (
    "embeddings_l2.npy",
    "content_index.parquet",
    "record_index.parquet",
    "metadata.json",
)
VIDEO_COLUMNS = (
    "manifest_version",
    "frame_id",
    "content_id",
    "image_sha256",
    "image_path",
    "video_id",
    "source_video",
    "source_video_sha256",
    "sample_index",
    "timestamp_seconds",
    "sample_fps",
    "source_frame_index_estimate",
)


class ExportError(ValueError):
    """Public messages contain a stable code, never an upstream value or path."""


def require(condition, code="artifact-invalid"):
    if not condition:
        raise ExportError(code)


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checked(directory, meta, name):
    """Only fixed filenames supplied by this code can be opened through receipts."""
    path = directory / name
    expected = meta.get("output_sha256", {}).get(name)
    require(isinstance(expected, str) and digest(path) == expected, "checksum-mismatch")
    return path


def stats(value):
    if not isinstance(value, dict):
        return {}
    output = {}
    for key, v in value.items():
        if key not in METRICS and not re.fullmatch(
            r"(?:trustworthiness|continuity|jaccard)@\d+(?:_mean_jaccard)?(?:_seed_mean|_seed_std)?",
            key,
        ):
            continue
        if (
            v is None
            or isinstance(v, (int, float, np.number))
            and not isinstance(v, (bool, np.bool_))
        ):
            output[key] = float(v) if v is not None and math.isfinite(float(v)) else None
    return output


def aliases(values, prefix):
    return {v: f"{prefix}-{i:06d}" for i, v in enumerate(sorted(set(values)), 1)}


def canonical_dataset(manifest):
    require(
        set(manifest.manifest_version) == {"flir_video_samples_v1"},
        "manifest-version-mismatch",
    )
    require(
        manifest.frame_id.is_unique
        and not manifest[["frame_id", "content_id", "image_sha256", "label_sha256"]]
        .isna()
        .any()
        .any()
    )
    require(manifest.content_id.eq(manifest.image_sha256).all(), "content-identity-mismatch")
    records = sorted(
        manifest[["frame_id", "image_sha256", "label_sha256"]]
        .astype(str)
        .itertuples(index=False, name=None)
    )
    payload = "flir_video_samples_v1\x1f" + json.dumps(
        records, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(payload.encode()).hexdigest()


class Exporter:
    def __init__(self, manifest):
        self.manifest = pd.read_parquet(manifest)
        m = self.manifest
        self.dataset_id = canonical_dataset(m)
        require(set(VIDEO_COLUMNS) <= set(m), "manifest-provenance-missing")
        require(
            not m.duplicated(["video_id", "sample_index"]).any(),
            "duplicate-grid-position",
        )
        require(
            not ({"sequence_id", "cluster_id", "group_id", "new_split"} & set(m)),
            "inferred-provenance-disallowed",
        )
        require(
            m.label_sha256.eq("").all()
            and ("original_split" not in m or m.original_split.fillna("").eq("").all()),
            "historical-manifest-disallowed",
        )
        require(0 < m.content_id.nunique() <= 20000 and len(m) <= 100000, "coverage-limit")
        require(
            not m[["video_id", "sample_index", "timestamp_seconds", "sample_fps"]]
            .isna()
            .any()
            .any()
        )
        require(
            (m.sample_index >= 0).all()
            and (m.sample_index % 1 == 0).all()
            and (m.sample_fps > 0).all()
        )
        require(
            np.isfinite(m[["sample_index", "timestamp_seconds", "sample_fps"]].to_numpy()).all()
        )
        require(
            np.allclose(m.timestamp_seconds, m.sample_index / m.sample_fps),
            "sampling-grid-mismatch",
        )
        self.content = aliases(m.content_id, "content")
        self.frame = aliases(m.frame_id, "frame")
        self.video = aliases(m.video_id, "video")
        self.ids = {}
        self.indices = {}
        self.features = {}
        self.similarities = {}
        self.feature_receipts = {}
        self.similarity_receipts = {}
        self.runs = []

    def alias(self, raw, kind):
        require(isinstance(raw, str) and bool(raw), "missing-identity")
        key = (kind, raw)
        if key not in self.ids:
            self.ids[key] = f"{kind}-{sum(k[0] == kind for k in self.ids) + 1:06d}"
        return self.ids[key]

    def identity(self, meta, encoder=None):
        require(meta.get("dataset_id") == self.dataset_id, "dataset-identity-mismatch")
        if encoder:
            require(
                meta.get("feature_space_id") == self.features[encoder]["feature_space_id"],
                "feature-identity-mismatch",
            )
            require(meta.get("extractor") == encoder, "encoder-mismatch")

    def index(self, directory, meta=None, encoder=None):
        path = (
            checked(directory, meta, "content_index.parquet")
            if meta
            else directory / "content_index.parquet"
        )
        index = pd.read_parquet(path)
        require(
            index.content_id.is_unique and set(index.content_id) <= set(self.content),
            "content-index-mismatch",
        )
        require(
            np.array_equal(index.embedding_row, np.arange(len(index))),
            "embedding-index-mismatch",
        )
        if encoder:
            require(
                index.content_id.tolist() == self.indices[encoder],
                "embedding-order-mismatch",
            )
        return index.content_id.tolist()

    def base(self, meta, stage, kind, raw_id, method, encoder=None):
        run = {
            "id": self.alias(raw_id, kind),
            "datasetId": "dataset-000001",
            "stage": stage,
            "origin": "artifact",
            "encoder": {"dinov2": "DINOv2", "clip": "CLIP"}.get(encoder),
            "method": method,
            "featureSpaceId": self.alias(meta["feature_space_id"], "feature")
            if meta.get("feature_space_id")
            else None,
            "similaritySpaceId": None,
            "reductionSpaceId": None,
            "clusteringSpaceId": None,
            "splitSpaceId": None,
            "configurationId": self.alias(meta["configuration_id"], "configuration")
            if meta.get("configuration_id")
            else None,
            "seed": int(meta["seed"]) if meta.get("seed") is not None else None,
            "status": "complete",
            "integrity": "export-validated",
            "metricOrigin": "reported",
            "metrics": {},
            "parameters": {},
            "dimensions": None,
            "pooling": None,
            "coverage": len(self.content),
            "candidate": False,
            "contentIndex": [],
            "coordinates": [],
            "pairs": [],
            "summaries": [],
            "labels": [],
            "assignments": [],
        }
        require(not any(r["id"] == run["id"] for r in self.runs), "duplicate-run")
        self.runs.append(run)
        return run

    def feature(self, directory, encoder):
        directory = Path(directory)
        meta = read_json(directory / "metadata.json")
        require(meta.get("dataset_id") == self.dataset_id, "dataset-identity-mismatch")
        dimension, pooling = (
            (384, "cls_token") if encoder == "dinov2" else (512, "projected_pooler_output")
        )
        require(
            meta.get("extractor") == encoder
            and meta.get("embedding_dimension") == dimension
            and meta.get("pooling_strategy") == pooling,
            "representation-mismatch",
        )
        require(
            read_json(directory / "feature_quality.json").get("quality_valid") is True,
            "feature-quality-invalid",
        )
        index = self.index(directory)
        # A full sampled-video snapshot deliberately rejects partial smoke-test coverage.
        require(set(index) == set(self.content), "incomplete-feature-coverage")
        mapping = pd.read_parquet(directory / "record_index.parquet")
        expected_frames = self.manifest.set_index("frame_id").content_id.to_dict()
        rows = dict(zip(index, range(len(index)), strict=True))
        require(
            mapping.frame_id.is_unique and set(mapping.frame_id) == set(expected_frames),
            "record-mapping-mismatch",
        )
        require(
            all(
                expected_frames[r.frame_id] == r.content_id
                and rows[r.content_id] == r.embedding_row
                for r in mapping.itertuples()
            ),
            "record-mapping-mismatch",
        )
        for name in ("embeddings_raw.npy", "embeddings_l2.npy"):
            array = np.load(directory / name, mmap_mode="r", allow_pickle=False)
            require(
                array.shape == (len(index), dimension) and np.issubdtype(array.dtype, np.floating),
                "embedding-shape-mismatch",
            )
            for start in range(0, len(array), 2048):
                block = array[start : start + 2048]
                require(np.isfinite(block).all(), "embedding-nonfinite")
                norms = np.linalg.norm(block, axis=1)
                require((norms > 0).all(), "embedding-zero-norm")
                if name == "embeddings_l2.npy":
                    require(np.allclose(norms, 1, atol=1e-5), "embedding-l2-invalid")
        self.indices[encoder] = index
        self.features[encoder] = meta
        self.feature_receipts[encoder] = {name: digest(directory / name) for name in FEATURE_FILES}
        run = self.base(
            meta,
            "embeddings",
            "feature",
            meta["feature_space_id"],
            "DINOv2" if encoder == "dinov2" else "CLIP",
            encoder,
        )
        run.update(
            dimensions=dimension,
            pooling=pooling,
            coverage=len(index),
            metricOrigin="artifact",
        )
        run["contentIndex"] = [
            {"contentId": self.content[c], "embeddingRow": i} for i, c in enumerate(index)
        ]

    def similarity(self, directory, encoder):
        require(encoder in self.features, "missing-features")
        directory = Path(directory)
        meta = read_json(directory / "metadata.json")
        self.identity(meta, encoder)
        relevant = self.manifest.reindex(columns=VIDEO_COLUMNS).sort_values("frame_id")
        rows = [
            {
                key: None
                if pd.isna(value)
                else {"float64_hex": float(value).hex()}
                if isinstance(value, float)
                else value
                for key, value in row.items()
            }
            for row in relevant.to_dict("records")
        ]
        provenance_digest = hashlib.sha256(
            json.dumps(rows, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
        require(
            meta.get("input_signature")
            == {
                "feature_files": self.feature_receipts[encoder],
                "posthoc_manifest_sha256": provenance_digest,
            },
            "similarity-source-mismatch",
        )
        require(
            meta.get("artifact_schema") == "content_cosine_v2"
            and meta.get("provenance_mode") == "sampled_video_grid",
            "similarity-schema-mismatch",
        )
        self.index(directory, meta, encoder)
        require(
            meta.get("content_count") == len(self.content)
            and meta.get("record_count") == len(self.frame),
            "similarity-coverage-mismatch",
        )
        summary = read_json(checked(directory, meta, "similarity_summary.json"))
        pairs = pd.read_parquet(checked(directory, meta, "nearest_neighbors.parquet"))
        require(len(pairs) <= 1000000, "pair-limit")
        require(
            not pairs.duplicated(["query_content_id", "neighbor_rank"]).any(),
            "duplicate-neighbor",
        )
        require(
            len(pairs) == len(self.content) * meta["top_k"],
            "neighbor-coverage-mismatch",
        )
        run = self.base(
            meta,
            "similarity",
            "similarity",
            meta["similarity_space_id"],
            "cosine",
            encoder,
        )
        self.similarities[encoder] = meta
        self.similarity_receipts[encoder] = {
            "metadata.json": digest(directory / "metadata.json"),
            "nearest_neighbors.parquet": digest(directory / "nearest_neighbors.parquet"),
            "cosine_similarity.npy": meta["output_sha256"].get("cosine_similarity.npy"),
        }
        run["similaritySpaceId"] = run["id"]
        run["metrics"] = stats(summary.get("global_similarity", {}))
        run["parameters"] = {"top_k": meta["top_k"]}
        for p in pairs.itertuples():
            require(
                p.query_content_id in self.content
                and p.neighbor_content_id in self.content
                and p.query_content_id != p.neighbor_content_id,
                "neighbor-content-mismatch",
            )
            require(
                math.isfinite(p.cosine_similarity) and -1.00001 <= p.cosine_similarity <= 1.00001,
                "cosine-invalid",
            )
            a, b = p.query_content_id, p.neighbor_content_id
            require(
                self.indices[encoder][p.query_row] == a
                and self.indices[encoder][p.neighbor_row] == b,
                "neighbor-row-mismatch",
            )
            same = bool(p.same_source_video)
            gap = None if pd.isna(p.min_sample_index_gap) else float(p.min_sample_index_gap)
            seconds = (
                None if pd.isna(p.min_timestamp_gap_seconds) else float(p.min_timestamp_gap_seconds)
            )
            require(same or gap is None and seconds is None, "provenance-gap-mismatch")
            run["pairs"].append(
                {
                    "a": self.content[a],
                    "b": self.content[b],
                    "rank": int(p.neighbor_rank),
                    "cosine": float(p.cosine_similarity),
                    "sameSourceVideo": same,
                    "sampleGap": gap,
                    "gridSecondsGap": seconds,
                }
            )
        for name, scope in (
            ("source_video_similarity.csv", "source"),
            ("sample_index_gap_similarity.csv", "sample_gap"),
            ("timestamp_gap_seconds_similarity.csv", "grid_gap"),
            ("topk_global_summary.csv", "topk"),
        ):
            self.table(directory, meta, name, run, scope)

    def table(self, directory, meta, name, run, scope):
        table = pd.read_csv(checked(directory, meta, name))
        require(len(table) <= 20000, "summary-limit")
        for i, row in enumerate(table.to_dict("records")):
            label = row.get("relation", row.get("metric"))
            if label not in {
                "same_source_video",
                "disjoint_source_videos",
                "rank1_similarity",
                "top5_mean_similarity",
                "top10_mean_similarity",
                "top20_mean_similarity",
            }:
                label = None
            configuration = row.get("configuration_id")
            run["summaries"].append(
                {
                    "scope": scope,
                    "index": i,
                    "label": label,
                    "configurationId": self.alias(configuration, "configuration")
                    if isinstance(configuration, str)
                    else None,
                    "metrics": stats(row),
                }
            )

    def reduction(self, directory, encoder):
        require(encoder in self.similarities, "missing-similarity")
        directory = Path(directory)
        meta = read_json(directory / "metadata.json")
        self.identity(meta, encoder)
        require(
            meta.get("reference_similarity_space_id")
            == self.similarities[encoder]["similarity_space_id"],
            "similarity-identity-mismatch",
        )
        if meta.get("artifact_kind") == "reduction_benchmark":
            candidates = pd.read_csv(checked(directory, meta, "candidates.csv"))
            selected = set(candidates.reduction_space_id)
            require(0 < len(selected) <= 4 and len(meta["runs"]) <= 128, "benchmark-limit")
            require(
                selected <= {r["reduction_space_id"] for r in meta["runs"]},
                "candidate-identity-mismatch",
            )
            root = directory.parents[1].resolve()
            for entry in meta["runs"]:
                require(
                    entry["method"] in ("tsne", "pacmap")
                    and re.fullmatch(r"[a-zA-Z0-9_-]+", entry["reduction_space_id"]),
                    "unsafe-run-reference",
                )
                run_dir = root / entry["method"] / entry["reduction_space_id"]
                require(run_dir.resolve().is_relative_to(root), "unsafe-run-reference")
                require(
                    digest(run_dir / "metadata.json") == entry["metadata_sha256"],
                    "benchmark-receipt-mismatch",
                )
                run_meta = read_json(run_dir / "metadata.json")
                require(
                    run_meta.get("reduction_space_id") == entry["reduction_space_id"]
                    and run_meta.get("method") == entry["method"],
                    "benchmark-identity-mismatch",
                )
                run = self.reduction_run(run_dir, encoder, entry["reduction_space_id"] in selected)
                if run["candidate"]:
                    row = (
                        candidates.loc[candidates.reduction_space_id == entry["reduction_space_id"]]
                        .iloc[0]
                        .to_dict()
                    )
                    run["metrics"].update(stats(row))
                    self.table(directory, meta, "configuration_stability.csv", run, "stability")
                    self.table(
                        directory,
                        meta,
                        "configuration_summary.csv",
                        run,
                        "configuration",
                    )
        else:
            self.reduction_run(directory, encoder, True, candidate=False)

    def reduction_run(self, directory, encoder, coordinates, candidate=None):
        meta = read_json(directory / "metadata.json")
        self.identity(meta, encoder)
        require(
            meta.get("input_signatures")
            == {
                "features": self.feature_receipts[encoder],
                "similarity": self.similarity_receipts[encoder],
            },
            "reduction-source-mismatch",
        )
        require(
            meta.get("artifact_kind") == "reduction_run"
            and meta.get("method") in ("tsne", "pacmap"),
            "reduction-schema-mismatch",
        )
        require(
            meta.get("reference_similarity_space_id")
            == self.similarities[encoder]["similarity_space_id"],
            "similarity-identity-mismatch",
        )
        require(
            read_json(checked(directory, meta, "quality.json")).get("quality_valid") is True,
            "reduction-quality-invalid",
        )
        index = self.index(directory, meta, encoder)
        run = self.base(
            meta,
            "reduction",
            "reduction",
            meta["reduction_space_id"],
            {"tsne": "t-SNE", "pacmap": "PaCMAP"}[meta["method"]],
            encoder,
        )
        run.update(
            reductionSpaceId=run["id"],
            similaritySpaceId=self.alias(meta["reference_similarity_space_id"], "similarity"),
            candidate=coordinates if candidate is None else candidate,
            metrics=stats(read_json(checked(directory, meta, "metrics.json"))),
            parameters=stats(meta.get("hyperparameters", {})),
        )
        if coordinates:
            xy = np.load(
                checked(directory, meta, "coordinates.npy"),
                mmap_mode="r",
                allow_pickle=False,
            )
            require(
                xy.shape == (len(index), 2) and np.isfinite(xy).all(),
                "coordinate-shape-mismatch",
            )
            run["coordinates"] = [
                {
                    "contentId": self.content[c],
                    "embeddingRow": i,
                    "x": float(xy[i, 0]),
                    "y": float(xy[i, 1]),
                }
                for i, c in enumerate(index)
            ]
        return run

    def clustering(self, directory):
        directory = Path(directory)
        meta = read_json(directory / "metadata.json")
        encoder = meta.get("extractor")
        require(encoder in self.features, "missing-features")
        self.identity(meta, encoder)
        require(
            meta.get("artifact_kind") == "clustering_run"
            and str(meta.get("algorithm")).upper() in ("DBSCAN", "OPTICS", "HDBSCAN"),
            "clustering-schema-mismatch",
        )
        require(
            read_json(checked(directory, meta, "quality.json")).get("quality_valid") is True,
            "clustering-quality-invalid",
        )
        index = self.index(directory, meta, encoder)
        labels = np.load(checked(directory, meta, "cluster_labels.npy"), allow_pickle=False)
        require(
            labels.shape == (len(index),)
            and np.issubdtype(labels.dtype, np.integer)
            and (labels >= -1).all(),
            "cluster-labels-invalid",
        )
        summary = pd.read_parquet(checked(directory, meta, "cluster_summary.parquet"))
        require(
            summary.cluster_id.is_unique and set(summary.cluster_id) == set(labels) - {-1},
            "cluster-summary-coverage-mismatch",
        )
        medoids = set()
        for row in summary.itertuples():
            members = {c for c, label in zip(index, labels, strict=True) if label == row.cluster_id}
            require(
                len(members) == row.n_members and row.medoid_content_id in members,
                "cluster-summary-mismatch",
            )
            medoids.add(row.medoid_content_id)
        run = self.base(
            meta,
            "clustering",
            "clustering",
            meta["clustering_space_id"],
            meta["algorithm"].upper(),
            encoder,
        )
        reduction = meta.get("reduction_space_id")
        if reduction:
            require(("reduction", reduction) in self.ids, "missing-reduction")
            run["reductionSpaceId"] = self.alias(reduction, "reduction")
        run["labels"] = [
            {
                "contentId": self.content[c],
                "clusterId": int(label),
                "medoid": c in medoids,
            }
            for c, label in zip(index, labels, strict=True)
        ]
        run["clusteringSpaceId"] = run["id"]
        run["metrics"] = stats(read_json(checked(directory, meta, "metrics.json")))
        run["parameters"] = stats(meta.get("effective_parameters", meta.get("config", {})))

    def splitting(self, directory):
        directory = Path(directory)
        meta = read_json(directory / "metadata.json")
        payload = meta.get("identity_payload", {})
        self.identity(payload)
        require(meta.get("artifact_kind") == "split_run", "split-schema-mismatch")
        strategy = payload.get("configuration", {}).get("strategy")
        require(
            strategy in ("historical", "random_content", "cluster_aware"),
            "split-strategy-invalid",
        )
        require(
            read_json(checked(directory, meta, "quality.json")).get("quality_valid") is True,
            "split-quality-invalid",
        )
        records = pd.read_parquet(checked(directory, meta, "record_split_assignments.parquet"))
        contents = pd.read_parquet(checked(directory, meta, "split_assignments.parquet"))
        require(
            contents.content_id.is_unique and set(contents.content_id) == set(self.content),
            "split-content-mismatch",
        )
        require(
            records.frame_id.is_unique and set(records.frame_id) == set(self.frame),
            "split-frame-mismatch",
        )
        require(
            set(records.new_split) <= {"train", "val", "test"},
            "split-membership-invalid",
        )
        expected = self.manifest.set_index("frame_id").content_id.to_dict()
        require(
            all(expected[r.frame_id] == r.content_id for r in records.itertuples()),
            "split-mapping-mismatch",
        )
        clustering_id = payload.get("clustering_space_id")
        cluster_run = None
        if strategy == "cluster_aware":
            require(("clustering", clustering_id) in self.ids, "missing-clustering")
            cluster_run = next(
                r for r in self.runs if r["id"] == self.ids[("clustering", clustering_id)]
            )
        if strategy != "historical":
            require(
                (records.groupby("content_id").new_split.nunique() == 1).all(),
                "content-cross-split",
            )
            if cluster_run:
                membership = records.groupby("content_id").new_split.first()
                allocation = {}
                inverse = {v: k for k, v in self.content.items()}
                for label in cluster_run["labels"]:
                    if label["clusterId"] < 0:
                        continue
                    split = membership[inverse[label["contentId"]]]
                    require(
                        allocation.get(label["clusterId"], split) == split,
                        "cluster-cross-split",
                    )
                    allocation[label["clusterId"]] = split
        run = self.base(payload, "splits", "split", meta["split_space_id"], strategy)
        run["splitSpaceId"] = run["id"]
        run["clusteringSpaceId"] = cluster_run["id"] if cluster_run else None
        run["assignments"] = [
            {
                "frameId": self.frame[r.frame_id],
                "contentId": self.content[r.content_id],
                "split": "validation" if r.new_split == "val" else r.new_split,
            }
            for r in records.itertuples()
        ]
        run["metrics"] = stats(read_json(checked(directory, meta, "split_summary.json")))

    def detection(self, directory):
        directory = Path(directory)
        meta = read_json(directory / "metadata.json")
        self.identity(meta)
        identity = meta.get("identity", {})
        split_id = identity.get("split_space_id")
        require(("split", split_id) in self.ids, "detector-split-mismatch")
        require(
            meta.get("state") == "COMPLETE" and meta.get("test_tuning") is False,
            "detector-not-complete",
        )
        metrics = read_json(checked(directory, meta, "metrics.json"))
        require(
            metrics.get("checkpoint_selection") == "validation_only"
            and metrics.get("test_threshold_optimized") is False,
            "detector-protocol-mismatch",
        )
        run = self.base(meta, "detector", "detector", meta["detector_run_id"], "detector")
        run.update(
            splitSpaceId=self.alias(split_id, "split"),
            metrics=stats(metrics.get("overall", {})),
        )
        # Complete files remain reported evidence, never a frontend certification.

    def snapshot(self):
        records = []
        for content, group in self.manifest.groupby("content_id", sort=True):
            records.append(
                {
                    "contentId": self.content[content],
                    "occurrences": [
                        {
                            "frameId": self.frame[r.frame_id],
                            "sourceVideo": self.video[r.video_id],
                            "sampleIndex": int(r.sample_index),
                            "gridSeconds": float(r.timestamp_seconds),
                            "sampleFps": float(r.sample_fps),
                            "sequenceId": None,
                            "captureTimestamp": None,
                        }
                        for r in group.sort_values("frame_id").itertuples()
                    ],
                }
            )
        return {
            "schemaVersion": "LeakageSnapshotV1",
            "dataset": {
                "id": "dataset-000001",
                "manifestVersion": "flir_video_samples_v1",
                "origin": "artifact",
                "occurrences": len(self.frame),
                "uniqueContents": len(self.content),
                "duplicateGroups": sum(len(c["occurrences"]) > 1 for c in records),
            },
            "contents": records,
            "runs": self.runs,
            "exportPolicy": POLICY,
        }


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--manifest", type=Path, required=True)
    for encoder in ("dinov2", "clip"):
        for stage in ("features", "similarity", "reduction"):
            cli.add_argument(f"--{encoder}-{stage}", type=Path)
    for stage in ("clustering", "splitting", "detection"):
        cli.add_argument(f"--{stage}", type=Path, action="append", default=[])
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--schema-version", choices=("v1", "v2"), default="v1")
    for stage in ("sequences", "linkage", "linkage-review", "linkage-aggregate"):
        cli.add_argument(f"--{stage}", type=Path)
    for stage in ("sequence-experiment", "sequence-evidence", "dataset-variant"):
        cli.add_argument(f"--{stage}", type=Path, action="append", default=[])
    return cli


def export(args):
    require(not args.output.exists(), "output-exists")
    # Never create an output inside an input artifact directory.
    inputs = [v for k, v in vars(args).items() if k != "output" and isinstance(v, Path)]
    inputs += args.clustering + args.splitting + args.detection
    inputs += args.sequence_experiment + args.sequence_evidence + args.dataset_variant
    require(
        all(
            not args.output.resolve().is_relative_to(
                p.resolve() if p.is_dir() else p.resolve().parent
            )
            for p in inputs
        ),
        "output-inside-input",
    )
    exporter = Exporter(args.manifest)
    for encoder in ("dinov2", "clip"):
        for stage, method in (
            ("features", exporter.feature),
            ("similarity", exporter.similarity),
            ("reduction", exporter.reduction),
        ):
            path = getattr(args, f"{encoder}_{stage}")
            if path is not None:
                method(path, encoder)
    for stage, method in (
        ("clustering", exporter.clustering),
        ("splitting", exporter.splitting),
        ("detection", exporter.detection),
    ):
        for path in getattr(args, stage):
            require(path.exists(), "explicit-artifact-missing")
            method(path)
    result = exporter.snapshot()
    research_inputs = any(
        getattr(args, key)
        for key in (
            "sequences",
            "sequence_experiment",
            "sequence_evidence",
            "linkage",
            "linkage_review",
            "linkage_aggregate",
            "dataset_variant",
        )
    )
    if args.schema_version == "v2" or research_inputs:
        try:
            from tools.export_research import ResearchExporter
        except ModuleNotFoundError:
            from export_research import ResearchExporter
        research = ResearchExporter(exporter, require, read_json, digest)
        for key, method in (
            ("sequences", research.sequences),
            ("sequence_experiment", research.experiments),
            ("sequence_evidence", research.evidence),
            ("linkage", research.linkage),
            ("linkage_review", research.review),
            ("linkage_aggregate", research.aggregate),
            ("dataset_variant", research.variant),
        ):
            value = getattr(args, key)
            for path in value if isinstance(value, list) else [value] if value else []:
                require(path.exists(), "explicit-artifact-missing")
                method(path)
        result.update(schemaVersion="LeakageSnapshotV2", research=research.finish())
    require(len(result["runs"]) <= 256, "run-limit")
    payload = json.dumps(result, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    require(len(payload.encode()) <= 80 * 1024 * 1024, "snapshot-size-limit")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Publish only complete bytes, without replacing an existing snapshot.
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=args.output.parent,
        prefix=".snapshot-",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        finally:
            stream.close()
    try:
        os.link(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)
    return result


if __name__ == "__main__":
    try:
        result = export(parser().parse_args())
        print(
            f"Snapshot exported: {len(result['contents'])} contents; {len(result['runs'])} runs. No scientific stages executed."
        )
    except ExportError as error:
        print(f"Export refused: {error}. No snapshot published.", file=sys.stderr)
        sys.exit(2)
    except (
        OSError,
        ValueError,
        KeyError,
        AttributeError,
        TypeError,
        IndexError,
    ):
        # Exceptions from parsers can contain private paths or raw identifiers.
        print(
            "Export refused: input, identity, coverage or integrity validation failed. Check the explicit inputs locally; no snapshot published.",
            file=sys.stderr,
        )
        sys.exit(2)
