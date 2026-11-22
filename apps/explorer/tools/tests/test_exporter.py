"""Synthetic/offline artifact compatibility tests; no pipeline imports or models."""

import hashlib
import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SPEC = importlib.util.spec_from_file_location(
    "exporter", Path(__file__).parents[1] / "export_leakage_snapshot.py"
)
ex = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ex)


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def receipt(directory, metadata):
    metadata["output_sha256"] = {
        p.name: ex.digest(p) for p in directory.iterdir() if p.name != "metadata.json"
    }
    write_json(directory / "metadata.json", metadata)


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.inputs = self.root / "input"
        self.inputs.mkdir()
        self.manifest = self.inputs / "manifest.parquet"
        self.frame = pd.DataFrame(
            [
                {
                    "manifest_version": "flir_video_samples_v1",
                    "frame_id": f"private-frame-{i}",
                    "content_id": f"private-hash-{i % 3}",
                    "image_sha256": f"private-hash-{i % 3}",
                    "label_sha256": "",
                    "image_path": f"C:/private/person/images/{i}.jpg",
                    "video_id": "secret-video-a" if i < 3 else "secret-video-b",
                    "source_video": "private-drone.mp4",
                    "source_video_sha256": "private-source-hash",
                    "sample_index": i if i < 3 else 0,
                    "timestamp_seconds": float(i) if i < 3 else 0.0,
                    "sample_fps": 1.0,
                    "source_frame_index_estimate": i,
                    "original_split": "",
                }
                for i in range(4)
            ]
        )
        self.frame.to_parquet(self.manifest, index=False)
        self.exporter = ex.Exporter(self.manifest)
        self.dataset = self.exporter.dataset_id
        self.feature_dirs = {}
        self.similarity_dirs = {}
        self.reduction_dirs = {}

    def tearDown(self):
        self.temp.cleanup()

    def feature(self, encoder="dinov2"):
        directory = self.inputs / encoder / "features"
        directory.mkdir(parents=True)
        dimension = 384 if encoder == "dinov2" else 512
        values = np.ones((3, dimension), dtype=np.float32)
        np.save(directory / "embeddings_raw.npy", values)
        np.save(
            directory / "embeddings_l2.npy",
            values / np.linalg.norm(values, axis=1, keepdims=True),
        )
        index = pd.DataFrame(
            {
                "content_id": sorted(self.frame.content_id.unique()),
                "embedding_row": range(3),
            }
        )
        index.to_parquet(directory / "content_index.parquet", index=False)
        self.frame[["frame_id", "content_id"]].merge(index).to_parquet(
            directory / "record_index.parquet", index=False
        )
        meta = {
            "dataset_id": self.dataset,
            "feature_space_id": f"private-feature-{encoder}",
            "extractor": encoder,
            "embedding_dimension": dimension,
            "pooling_strategy": "cls_token" if encoder == "dinov2" else "projected_pooler_output",
            "private_path": "C:/Users/private_person/token",
            "seed": 0,
        }
        write_json(directory / "metadata.json", meta)
        write_json(directory / "feature_quality.json", {"quality_valid": True})
        self.exporter.feature(directory, encoder)
        self.feature_dirs[encoder] = directory
        return directory

    def similarity(self, encoder="dinov2"):
        self.feature(encoder)
        directory = self.inputs / encoder / "similarity"
        directory.mkdir()
        index = pd.read_parquet(self.feature_dirs[encoder] / "content_index.parquet")
        index.to_parquet(directory / "content_index.parquet", index=False)
        ids = index.content_id.tolist()
        pairs = [
            {
                "query_row": i,
                "neighbor_row": (i + 1) % 3,
                "query_content_id": ids[i],
                "neighbor_content_id": ids[(i + 1) % 3],
                "neighbor_rank": 1,
                "cosine_similarity": 0.8,
                "same_source_video": True,
                "min_sample_index_gap": 1 if i < 2 else 2,
                "min_timestamp_gap_seconds": 1.0 if i < 2 else 2.0,
            }
            for i in range(3)
        ]
        pd.DataFrame(pairs).to_parquet(directory / "nearest_neighbors.parquet", index=False)
        write_json(
            directory / "similarity_summary.json",
            {
                "global_similarity": {
                    "count": 3,
                    "mean": 0.8,
                    "median": 0.8,
                    "private_person": 44,
                }
            },
        )
        for name in (
            "source_video_similarity",
            "sample_index_gap_similarity",
            "timestamp_gap_seconds_similarity",
            "topk_global_summary",
        ):
            pd.DataFrame([{"count": 3, "mean": 0.8}]).to_csv(directory / f"{name}.csv", index=False)
        relevant = self.frame.reindex(columns=ex.VIDEO_COLUMNS).sort_values("frame_id")
        rows = [
            {
                key: None
                if pd.isna(v)
                else {"float64_hex": float(v).hex()}
                if isinstance(v, float)
                else v
                for key, v in row.items()
            }
            for row in relevant.to_dict("records")
        ]
        provenance_digest = hashlib.sha256(
            json.dumps(rows, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
        meta = {
            "dataset_id": self.dataset,
            "feature_space_id": self.exporter.features[encoder]["feature_space_id"],
            "extractor": encoder,
            "artifact_schema": "content_cosine_v2",
            "provenance_mode": "sampled_video_grid",
            "content_count": 3,
            "record_count": 4,
            "top_k": 1,
            "similarity_space_id": f"private-similarity-{encoder}",
            "input_signature": {
                "feature_files": self.exporter.feature_receipts[encoder],
                "posthoc_manifest_sha256": provenance_digest,
            },
        }
        receipt(directory, meta)
        # Matrix is not consumed. Its receipt binds reduction metadata without an N² read.
        meta["output_sha256"]["cosine_similarity.npy"] = "a" * 64
        write_json(directory / "metadata.json", meta)
        self.exporter.similarity(directory, encoder)
        self.similarity_dirs[encoder] = directory
        return directory

    def reduction(self, encoder="dinov2"):
        self.similarity(encoder)
        directory = self.inputs / encoder / "reduction"
        directory.mkdir()
        pd.read_parquet(self.feature_dirs[encoder] / "content_index.parquet").to_parquet(
            directory / "content_index.parquet", index=False
        )
        np.save(
            directory / "coordinates.npy",
            np.array([[-4.5, 12.0], [5.0, -1.0], [90.0, 120.0]], dtype=np.float32),
        )
        write_json(directory / "quality.json", {"quality_valid": True})
        write_json(
            directory / "metrics.json",
            {"trustworthiness@5": 0.91, "spearman_distance": 0.78},
        )
        meta = {
            "dataset_id": self.dataset,
            "feature_space_id": self.exporter.features[encoder]["feature_space_id"],
            "extractor": encoder,
            "artifact_kind": "reduction_run",
            "method": "tsne",
            "reduction_space_id": f"private-reduction-{encoder}",
            "reference_similarity_space_id": self.exporter.similarities[encoder][
                "similarity_space_id"
            ],
            "input_signatures": {
                "features": self.exporter.feature_receipts[encoder],
                "similarity": self.exporter.similarity_receipts[encoder],
            },
            "seed": 0,
            "hyperparameters": {"perplexity": 5},
        }
        receipt(directory, meta)
        self.exporter.reduction(directory, encoder)
        self.reduction_dirs[encoder] = directory
        return directory

    def test_both_encoders_preserve_coordinates_and_all_occurrences(self):
        self.reduction()
        self.reduction("clip")
        snapshot = self.exporter.snapshot()
        self.assertEqual(len(snapshot["runs"]), 6)
        self.assertEqual(snapshot["runs"][2]["coordinates"][0]["x"], -4.5)
        self.assertEqual(len(snapshot["contents"][0]["occurrences"]), 2)
        self.assertEqual(len({r["sourceVideo"] for r in snapshot["contents"][0]["occurrences"]}), 2)
        self.assertTrue(
            all(o["sequenceId"] is None for c in snapshot["contents"] for o in c["occurrences"])
        )
        text = json.dumps(snapshot)
        for secret in ("private", "secret", "C:/", ".jpg", ".mp4", "raw_dtype"):
            self.assertNotIn(secret, text)

    def test_missing_stages_stay_absent(self):
        self.feature()
        self.assertEqual([r["stage"] for r in self.exporter.snapshot()["runs"]], ["embeddings"])

    def test_shared_frontend_fixture_matches_exporter(self):
        self.reduction()
        self.reduction("clip")
        fixture = Path(__file__).parents[2] / "src/test/fixtures/leakage-synthetic.json"
        self.assertEqual(self.exporter.snapshot(), ex.read_json(fixture))

    def test_dataset_mismatch_refused(self):
        directory = self.feature()
        meta = ex.read_json(directory / "metadata.json")
        meta["dataset_id"] = "historical-dataset"
        write_json(directory / "metadata.json", meta)
        with self.assertRaisesRegex(ex.ExportError, "dataset-identity"):
            ex.Exporter(self.manifest).feature(directory, "dinov2")

    def test_nonfinite_and_zero_norm_refused(self):
        directory = self.feature()
        for value in (0.0, float("nan")):
            np.save(
                directory / "embeddings_l2.npy",
                np.full((3, 384), value, dtype=np.float32),
            )
            with self.assertRaises(ex.ExportError):
                ex.Exporter(self.manifest).feature(directory, "dinov2")

    def test_coordinate_checksum_and_row_order_refused(self):
        directory = self.reduction()
        np.save(directory / "coordinates.npy", np.zeros((3, 2)))
        with self.assertRaises(ex.ExportError):
            self.exporter.reduction_run(directory, "dinov2", True)
        pd.DataFrame(
            {
                "content_id": sorted(self.frame.content_id.unique(), reverse=True),
                "embedding_row": range(3),
            }
        ).to_parquet(directory / "content_index.parquet", index=False)
        meta = ex.read_json(directory / "metadata.json")
        receipt(directory, meta)
        with self.assertRaisesRegex(ex.ExportError, "embedding-order"):
            self.exporter.index(directory, meta, "dinov2")

    def test_temporal_change_is_bound_to_similarity_receipt(self):
        directory = self.similarity()
        self.exporter.manifest.loc[0, "video_id"] = "another-source"
        with self.assertRaisesRegex(ex.ExportError, "similarity-source"):
            self.exporter.similarity(directory, "dinov2")

    def test_export_is_read_only_and_does_not_overwrite(self):
        self.reduction()
        before = {str(p): ex.digest(p) for p in self.inputs.rglob("*") if p.is_file()}
        args = ex.parser().parse_args(
            [
                "--manifest",
                str(self.manifest),
                "--dinov2-features",
                str(self.feature_dirs["dinov2"]),
                "--dinov2-similarity",
                str(self.similarity_dirs["dinov2"]),
                "--dinov2-reduction",
                str(self.reduction_dirs["dinov2"]),
                "--output",
                str(self.root / "output" / "snapshot.json"),
            ]
        )
        result = ex.export(args)
        self.assertEqual(result["schemaVersion"], "LeakageSnapshotV1")
        self.assertEqual(
            before,
            {str(p): ex.digest(p) for p in self.inputs.rglob("*") if p.is_file()},
        )
        with self.assertRaisesRegex(ex.ExportError, "output-exists"):
            ex.export(args)

    def test_explicit_missing_stage_is_error_and_omitted_stage_stays_pending(self):
        args = ex.parser().parse_args(
            [
                "--manifest",
                str(self.manifest),
                "--clustering",
                str(self.inputs / "missing"),
                "--output",
                str(self.root / "snapshot.json"),
            ]
        )
        with self.assertRaisesRegex(ex.ExportError, "explicit-artifact-missing"):
            ex.export(args)
        self.assertFalse(args.output.exists())
        args.clustering = []
        result = ex.export(args)
        self.assertEqual(result["runs"], [])
        self.assertTrue(args.output.exists())

    def clustering(self):
        self.reduction()
        directory = self.inputs / "clustering"
        directory.mkdir()
        index = pd.read_parquet(self.feature_dirs["dinov2"] / "content_index.parquet")
        index.to_parquet(directory / "content_index.parquet", index=False)
        np.save(directory / "cluster_labels.npy", np.array([0, 0, -1]))
        pd.DataFrame(
            [{"cluster_id": 0, "n_members": 2, "medoid_content_id": index.content_id.iloc[0]}]
        ).to_parquet(directory / "cluster_summary.parquet", index=False)
        write_json(directory / "quality.json", {"quality_valid": True})
        write_json(directory / "metrics.json", {"noise_fraction": 1 / 3})
        receipt(
            directory,
            {
                "dataset_id": self.dataset,
                "feature_space_id": "private-feature-dinov2",
                "extractor": "dinov2",
                "artifact_kind": "clustering_run",
                "algorithm": "dbscan",
                "clustering_space_id": "private-clustering",
                "reduction_space_id": "private-reduction-dinov2",
            },
        )
        self.exporter.clustering(directory)
        return directory

    def splitting(self):
        self.clustering()
        directory = self.inputs / "split"
        directory.mkdir()
        records = self.frame[["frame_id", "content_id"]].copy()
        records["new_split"] = ["train", "train", "test", "train"]
        records.to_parquet(directory / "record_split_assignments.parquet", index=False)
        records[["content_id", "new_split"]].drop_duplicates().to_parquet(
            directory / "split_assignments.parquet", index=False
        )
        write_json(directory / "quality.json", {"quality_valid": True})
        write_json(directory / "split_summary.json", {})
        receipt(
            directory,
            {
                "artifact_kind": "split_run",
                "split_space_id": "private-split",
                "identity_payload": {
                    "dataset_id": self.dataset,
                    "clustering_space_id": "private-clustering",
                    "configuration": {"strategy": "cluster_aware"},
                    "seed": 42,
                },
            },
        )
        self.exporter.splitting(directory)
        return directory

    def test_clustering_noise_and_compatible_split(self):
        self.splitting()
        cluster, split = self.exporter.runs[-2:]
        self.assertEqual(cluster["labels"][-1]["clusterId"], -1)
        self.assertEqual(split["clusteringSpaceId"], cluster["id"])
        self.assertEqual(len(split["assignments"]), 4)

    def test_fractured_cluster_split_is_rejected(self):
        directory = self.splitting()
        records = pd.read_parquet(directory / "record_split_assignments.parquet")
        records.loc[records.content_id == "private-hash-1", "new_split"] = "val"
        records.to_parquet(directory / "record_split_assignments.parquet", index=False)
        receipt(directory, ex.read_json(directory / "metadata.json"))
        with self.assertRaisesRegex(ex.ExportError, "cluster-cross-split"):
            self.exporter.splitting(directory)

    def test_detector_requires_split_and_rejects_pilots(self):
        self.splitting()
        directory = self.inputs / "detector"
        directory.mkdir()
        write_json(
            directory / "metrics.json",
            {
                "checkpoint_selection": "validation_only",
                "test_threshold_optimized": False,
                "overall": {"precision": 0.7, "recall": 0.6, "map50": 0.5, "map50_95": 0.4},
            },
        )
        meta = {
            "dataset_id": self.dataset,
            "identity": {"split_space_id": "private-split"},
            "state": "SMALL_PILOT_VALIDATED",
            "test_tuning": False,
            "detector_run_id": "private-detector",
        }
        receipt(directory, meta)
        with self.assertRaisesRegex(ex.ExportError, "detector-not-complete"):
            self.exporter.detection(directory)
        meta["state"] = "COMPLETE"
        receipt(directory, meta)
        self.exporter.detection(directory)
        self.assertEqual(self.exporter.runs[-1]["metrics"]["map50"], 0.5)

    def test_benchmark_uses_declared_candidates_without_sampling_contents(self):
        directory = self.reduction()
        self.exporter.runs.pop()
        root = self.inputs / "benchmark-root"
        runs = []
        for seed in (0, 1):
            run_id = f"synthetic-reduction-{seed}"
            target = root / "tsne" / run_id
            shutil.copytree(directory, target)
            meta = ex.read_json(target / "metadata.json")
            meta.update(seed=seed, reduction_space_id=run_id)
            write_json(target / "metadata.json", meta)
            runs.append(
                {
                    "method": "tsne",
                    "reduction_space_id": run_id,
                    "metadata_sha256": ex.digest(target / "metadata.json"),
                }
            )
        benchmark = root / "benchmarks" / "synthetic-benchmark"
        benchmark.mkdir(parents=True)
        pd.DataFrame(
            [{"reduction_space_id": "synthetic-reduction-0", "stability_mean": 0.9}]
        ).to_csv(benchmark / "candidates.csv", index=False)
        for name in ("configuration_stability", "configuration_summary"):
            pd.DataFrame([{"mean": 0.9}]).to_csv(benchmark / f"{name}.csv", index=False)
        receipt(
            benchmark,
            {
                "artifact_kind": "reduction_benchmark",
                "dataset_id": self.dataset,
                "feature_space_id": "private-feature-dinov2",
                "extractor": "dinov2",
                "reference_similarity_space_id": "private-similarity-dinov2",
                "runs": runs,
            },
        )
        self.exporter.reduction(benchmark, "dinov2")
        selected, other = self.exporter.runs[-2:]
        self.assertTrue(selected["candidate"])
        self.assertEqual(len(selected["coordinates"]), 3)
        self.assertFalse(other["candidate"])
        self.assertEqual(other["coordinates"], [])


if __name__ == "__main__":
    unittest.main()
