"""Synthetic current-upstream schemas; never open real research artifacts."""

import hashlib
import importlib.util
import json
import unittest
from pathlib import Path

import pandas as pd
import test_exporter as base

ex = base.ex
write_json = base.write_json

SPEC = importlib.util.spec_from_file_location(
    "research", Path(__file__).parents[1] / "export_research.py"
)
research_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(research_module)


class ResearchTests(unittest.TestCase):
    def setUp(self):
        self.fixture = base.ExportTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.ex = self.fixture.exporter
        for encoder in ("clip", "dinov2"):
            self.fixture.feature(encoder)
        self.research = research_module.ResearchExporter(
            self.ex, ex.require, ex.read_json, ex.digest
        )

    def publish(self, name, kind, identity, tables=None, summary=None, extra=None):
        directory = self.fixture.inputs / name
        directory.mkdir()
        for key, table in (tables or {}).items():
            pd.DataFrame(table).to_parquet(directory / f"{key}.parquet", index=False)
        write_json(directory / "summary.json", summary or {})
        meta = {
            "artifact_id": f"private-{name}",
            "artifact_kind": kind,
            "identity": identity,
            "semantics": {"ground_truth": False},
            **(extra or {}),
        }
        self.seal(directory, meta)
        return directory

    def seal(self, directory, meta):
        meta["output_checksums"] = {
            p.name: ex.digest(p) for p in directory.iterdir() if p.name != "metadata.json"
        }
        write_json(directory / "metadata.json", meta)

    def signature(self):
        return {
            "dataset_id": self.ex.dataset_id,
            "feature_spaces": {
                encoder: {"feature_space_id": self.ex.features[encoder]["feature_space_id"]}
                for encoder in ("clip", "dinov2")
            },
        }

    def sequences(self, instances=False):
        events = [
            {
                "event_id": 0,
                "candidate_id": "private-event",
                "video_id": "secret-video-a",
                "search_start_sample_index": 1,
                "search_end_sample_index": 2,
            }
        ]
        detection = {"sources": self.signature()}
        tables = {"candidate_events": events}
        identity = detection
        if instances:
            identity = {
                "detection": detection,
                "validation": {
                    "metadata": {
                        "review_status": "confirmed_manual_review",
                        "ground_truth": False,
                        "manual_confirmation_complete": True,
                    }
                },
            }
            tables["manual_review"] = [
                {"event_id": 0, "decision": "reject", "manual_confirmation": True}
            ]
            tables["sequence_instances"] = [
                {
                    "sequence_id": "private-seq-a",
                    "video_id": "secret-video-a",
                    "start_sample_index": 0,
                    "end_sample_index": 2,
                },
                {
                    "sequence_id": "private-seq-b",
                    "video_id": "secret-video-b",
                    "start_sample_index": 0,
                    "end_sample_index": 0,
                },
            ]
            assignments = self.ex.manifest[
                ["frame_id", "content_id", "video_id", "sample_index"]
            ].copy()
            assignments["sequence_id"] = assignments.video_id.map(
                {"secret-video-a": "private-seq-a", "secret-video-b": "private-seq-b"}
            )
            tables["occurrence_assignments"] = assignments
        return self.publish(
            "sequences",
            "sequence_instance_set" if instances else "sequence_boundary_candidates",
            identity,
            tables,
        )

    def linkage(self):
        assignments = self.ex.manifest.loc[self.ex.manifest.content_id.eq("private-hash-0")]
        occurrences = [
            {
                "video_content_id": row.content_id,
                "video_frame_id": row.frame_id,
                "sequence_id": self.research.sequence_members[row.frame_id],
            }
            for row in assignments.itertuples()
        ]
        labeled = [
            {
                "manifest_version": "flir_canonical_candidate_v1",
                "frame_id": f"private-labeled-{suffix}",
                "content_id": "private-labeled",
                "image_sha256": "private-labeled",
                "label_sha256": "private-label",
            }
            for suffix in ("a", "b")
        ]
        payload = "flir_canonical_candidate_v1\x1f" + json.dumps(
            [(r["frame_id"], r["image_sha256"], r["label_sha256"]) for r in labeled],
            separators=(",", ":"),
            ensure_ascii=True,
        )
        return self.publish(
            "linkage",
            "labeled_video_link_candidates",
            {},
            {
                "content_candidates": [
                    {
                        "candidate_id": "private-link",
                        "labeled_content_id": "private-labeled",
                        "video_content_id": "private-hash-0",
                        "clip_cosine": 0.8,
                        "dinov2_cosine": 0.7,
                    }
                ],
                "candidate_occurrences": occurrences,
                "labeled_occurrences": labeled,
            },
            extra={
                "video_dataset_id": self.ex.dataset_id,
                "labeled_dataset_id": hashlib.sha256(payload.encode()).hexdigest(),
                "sequence_set_id": "private-sequences",
                "clip_feature_space_id": self.ex.features["clip"]["feature_space_id"],
                "dinov2_feature_space_id": self.ex.features["dinov2"]["feature_space_id"],
            },
        )

    def test_candidate_zones_do_not_become_instances(self):
        self.research.sequences(self.sequences())
        result = self.research.finish()
        zone = result["sequences"]["artifact"]["zones"][0]
        self.assertEqual(
            (zone["kind"], zone["decision"], zone["reviewId"]),
            ("candidate_zone", "candidate", None),
        )
        self.assertEqual(result["linkage"]["state"], "pending")
        self.assertNotIn("private", json.dumps(result))

    def test_linkage_retains_all_memberships_and_labeled_occurrences(self):
        self.research.sequences(self.sequences(True))
        self.research.linkage(self.linkage())
        result = self.research.finish()["linkage"]["artifact"]
        self.assertEqual(len(result["candidates"][0]["sequenceIds"]), 2)
        self.assertEqual(len(result["labeledOccurrences"]), 2)
        self.assertEqual(result["candidates"][0]["state"], "candidate")
        self.assertNotIn("private", json.dumps(result))
        snapshot = self.ex.snapshot()
        snapshot.update(schemaVersion="LeakageSnapshotV2", research=self.research.finish())
        golden = Path(__file__).parents[2] / "src/test/fixtures/leakage-v2-synthetic.json"
        self.assertEqual(snapshot, json.loads(golden.read_text(encoding="utf-8")))

    def test_labeled_identity_is_validated_before_anonymization(self):
        self.research.sequences(self.sequences(True))
        directory = self.linkage()
        meta = ex.read_json(directory / "metadata.json")
        meta["labeled_dataset_id"] = "foreign-dataset"
        write_json(directory / "metadata.json", meta)
        with self.assertRaisesRegex(ex.ExportError, "linkage-labeled-dataset-mismatch"):
            self.research.linkage(directory)

    def test_group_review_and_aggregate_do_not_confirm_candidate_links(self):
        self.research.sequences(self.sequences(True))
        self.research.linkage(self.linkage())
        sources = {"linkage_id": "private-linkage", "sequence_set_id": "private-sequences"}
        directory = self.publish(
            "review",
            "labeled_visual_dependency_manual_calibration",
            {"calibration": {"sources": sources}},
        )
        pd.DataFrame(
            [
                {
                    "review_query_id": "private-query",
                    "labeled_content_id": "private-labeled",
                    "proposed_visual_dependency_group_id": "private-group",
                    "manual_decision": "supported",
                }
            ]
        ).to_csv(directory / "review.csv", index=False)
        meta = ex.read_json(directory / "metadata.json")
        self.seal(directory, meta)
        self.research.review(directory)
        aggregate = self.publish(
            "aggregate",
            "labeled_visual_dependency_manual_calibration_aggregate",
            {},
            summary={
                "source_revision_count": 1,
                "overall_pooled_descriptive": {
                    "decision_counts": {"supported": 1, "unsupported": 0, "ambiguous": 0}
                },
            },
        )
        write_json(
            aggregate / "source_revisions.json",
            [{"source_revision_id": "private-review", "metadata": meta}],
        )
        self.seal(aggregate, ex.read_json(aggregate / "metadata.json"))
        self.research.aggregate(aggregate)
        self.assertEqual(
            self.research.result["groupReview"]["artifact"]["items"][0]["decision"], "supported"
        )
        self.assertIsNone(self.research.result["reviewAggregation"]["artifact"]["confirmed"])
        self.assertEqual(
            self.research.result["linkage"]["artifact"]["candidates"][0]["state"], "candidate"
        )

    def test_declared_media_is_never_opened(self):
        directory = self.sequences()
        meta = ex.read_json(directory / "metadata.json")
        meta["output_checksums"]["media/never-open.png"] = "not-an-access-permission"
        write_json(directory / "metadata.json", meta)
        self.research.sequences(directory)
        self.research.finish()

    def test_checksum_and_original_identity_are_checked_before_aliases(self):
        directory = self.sequences()
        meta = ex.read_json(directory / "metadata.json")
        meta["identity"]["sources"]["dataset_id"] = "foreign"
        write_json(directory / "metadata.json", meta)
        with self.assertRaisesRegex(ex.ExportError, "research-dataset-mismatch"):
            self.research.sequences(directory)
        meta["identity"]["sources"] = self.signature()
        write_json(directory / "metadata.json", meta)
        with (directory / "candidate_events.parquet").open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ex.ExportError, "research-checksum-mismatch"):
            self.research.sequences(directory)

    def test_changed_consumed_file_is_rejected_at_publication(self):
        directory = self.sequences()
        self.research.sequences(directory)
        write_json(directory / "metadata.json", {})
        with self.assertRaisesRegex(ex.ExportError, "research-input-changed"):
            self.research.finish()

    def test_collapsed_linkage_occurrences_rejected(self):
        self.research.sequences(self.sequences(True))
        directory = self.linkage()
        path = directory / "candidate_occurrences.parquet"
        pd.read_parquet(path).iloc[:1].to_parquet(path, index=False)
        self.seal(directory, ex.read_json(directory / "metadata.json"))
        with self.assertRaisesRegex(ex.ExportError, "linkage-occurrence-coverage"):
            self.research.linkage(directory)

    def test_v2_omitted_artifacts_are_pending_and_v1_default_unchanged(self):
        args = ex.parser().parse_args(
            [
                "--manifest",
                str(self.fixture.manifest),
                "--schema-version",
                "v2",
                "--output",
                str(self.fixture.root / "v2.json"),
            ]
        )
        result = ex.export(args)
        self.assertEqual(result["schemaVersion"], "LeakageSnapshotV2")
        self.assertEqual(result["research"]["sequences"], {"state": "pending", "artifact": None})
        self.assertEqual(len(result["contents"]), 3)

    def test_native_evidence_and_suite_remain_reported_not_verified(self):
        evidence = self.publish(
            "evidence",
            "sequence_structure_review_v1",
            {
                "sources": {"input": self.signature()},
                "config": {"adapter": "hypatia_legacy_evidence_v1"},
            },
            summary={"canonical_source_verified": True},
        )
        self.research.evidence(evidence)
        self.assertEqual(self.research.result["evidence"]["artifact"][0]["state"], "imported")
        experiment = self.publish(
            "experiment",
            "sequence_experiment_suite_v1",
            {"sources": {"input": self.signature()}, "config": {}},
            summary={"posthoc_review_available": False},
        )
        self.research.experiments(experiment)
        run = self.research.result["experiments"]["artifact"][0]
        self.assertIsNone(run["agreement"])
        self.assertEqual(run["reviewState"], "pending")
