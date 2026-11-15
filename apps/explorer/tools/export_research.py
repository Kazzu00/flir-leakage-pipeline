"""Read-only native research-artifact projection; no scientific pipeline imports.

Only fixed JSON/Parquet/CSV names below are consumed. Media and original evidence
snapshots are deliberately not opened. Checks cover consumed bytes, not the full
upstream publication or a scientific replay. Private strings never pass through.
"""

import hashlib
import json
from pathlib import Path

import pandas as pd


class ResearchExporter:
    def __init__(self, exporter, require, read_json, digest):
        self.ex = exporter
        self.require = require
        self.read_json = read_json
        self.digest = digest
        self.consumed = {}
        self.sequence_id = None
        self.sequence_members = {}
        self.linkage_id = None
        self.labeled = {}
        self.review_id = None
        self.result = {
            key: {"state": "pending", "artifact": None}
            for key in (
                "sequences",
                "experiments",
                "evidence",
                "linkage",
                "linkageReview",
                "groupReview",
                "reviewAggregation",
            )
        }
        self.result["variants"] = []

    def read(self, path):
        before = self.digest(path)
        value = self.read_json(path)
        self.require(before == self.digest(path), "research-input-changed")
        self.consumed[Path(path)] = before
        return value

    def metadata(self, directory, kinds):
        meta = self.read(directory / "metadata.json")
        self.require(meta.get("artifact_kind") in kinds, "research-kind-unsupported")
        self.require(
            isinstance(meta.get("artifact_id"), str) and bool(meta["artifact_id"]),
            "research-id-missing",
        )
        self.require(
            meta.get("ground_truth", meta.get("semantics", {}).get("ground_truth")) is False,
            "research-ground-truth-invalid",
        )
        return meta

    def file(self, directory, meta, name):
        # All callers supply constant names; do not follow arbitrary metadata paths.
        path = directory / name
        self.require(
            path.resolve().is_relative_to(directory.resolve()), "research-path-outside-input"
        )
        expected = meta.get("output_checksums", {}).get(name)
        self.require(
            isinstance(expected, str) and self.digest(path) == expected,
            "research-checksum-mismatch",
        )
        self.consumed[path] = expected
        return path

    def table(self, directory, meta, name, columns):
        path = self.file(directory, meta, name)
        table = pd.read_parquet(path, columns=columns)
        self.require(
            len(table) <= 100000 and not table.isna().any().any(), "research-table-invalid"
        )
        return table

    def source(self, signature):
        self.require(signature.get("dataset_id") == self.ex.dataset_id, "research-dataset-mismatch")
        spaces = signature.get("feature_spaces", {})
        self.require(set(spaces) == {"clip", "dinov2"}, "research-features-missing")
        for encoder, feature in spaces.items():
            self.require(encoder in self.ex.features, "research-features-missing")
            self.require(
                feature.get("feature_space_id") == self.ex.features[encoder]["feature_space_id"],
                "research-feature-mismatch",
            )

    def base(self, meta, kind):
        return {
            "id": self.ex.alias(meta["artifact_id"], kind),
            "datasetId": "dataset-000001",
            "origin": "artifact",
            "groundTruth": False,
            "integrity": "consumed-files-validated",
        }

    def put(self, key, artifact, state="experimental"):
        self.result[key] = {"state": state, "artifact": artifact}

    def grid(self, video, start, end):
        self.require(
            video in self.ex.video
            and isinstance(start, int)
            and isinstance(end, int)
            and 0 <= start <= end,
            "research-grid-invalid",
        )
        samples = self.ex.manifest.loc[self.ex.manifest.video_id.eq(video), "sample_index"]
        self.require(
            start in set(samples) and end in set(samples), "research-grid-outside-manifest"
        )

    def sequences(self, directory):
        meta = self.metadata(directory, {"sequence_boundary_candidates", "sequence_instance_set"})
        is_set = meta["artifact_kind"] == "sequence_instance_set"
        detection = meta["identity"]["detection"] if is_set else meta["identity"]
        self.source(detection["sources"])
        events = self.table(
            directory,
            meta,
            "candidate_events.parquet",
            [
                "event_id",
                "candidate_id",
                "video_id",
                "search_start_sample_index",
                "search_end_sample_index",
            ],
        )
        self.require(events.candidate_id.is_unique, "duplicate-sequence-candidate")
        result = {
            **self.base(meta, "sequence"),
            "zones": [],
            "reviews": [],
            "recurrence": [],
            "instanceReviewBound": False,
        }
        decisions = {}
        if is_set:
            validation = meta["identity"]["validation"]["metadata"]
            self.require(
                validation.get("review_status") == "confirmed_manual_review"
                and validation.get("ground_truth") is False
                and validation.get("manual_confirmation_complete") is True,
                "sequence-review-not-confirmed",
            )
            review = self.table(
                directory,
                meta,
                "manual_review.parquet",
                ["event_id", "decision", "manual_confirmation"],
            )
            self.require(
                review.event_id.is_unique and set(review.event_id) == set(events.event_id),
                "sequence-review-coverage",
            )
            self.require(
                set(review.decision) <= {"accept", "reject"}
                and review.manual_confirmation.eq(True).all(),
                "sequence-review-vocabulary",
            )
            decisions = dict(zip(review.event_id, review.decision, strict=True))
            result["instanceReviewBound"] = True
        for row in events.itertuples():
            self.grid(row.video_id, row.search_start_sample_index, row.search_end_sample_index)
            zid = self.ex.alias(row.candidate_id, "zone")
            decision = {"accept": "accepted", "reject": "rejected"}.get(
                decisions.get(row.event_id), "candidate"
            )
            rid = self.ex.alias(row.candidate_id, "review") if decision != "candidate" else None
            result["zones"].append(
                {
                    "id": zid,
                    "sourceVideo": self.ex.video[row.video_id],
                    "start": row.search_start_sample_index,
                    "end": row.search_end_sample_index,
                    "kind": "candidate_zone",
                    "decision": decision,
                    "reviewId": rid,
                }
            )
            if rid:
                result["reviews"].append(
                    {"id": rid, "targetId": zid, "decision": decision, "groundTruth": False}
                )
        if is_set:
            instances = self.table(
                directory,
                meta,
                "sequence_instances.parquet",
                ["sequence_id", "video_id", "start_sample_index", "end_sample_index"],
            )
            assignments = self.table(
                directory,
                meta,
                "occurrence_assignments.parquet",
                ["frame_id", "content_id", "video_id", "sample_index", "sequence_id"],
            )
            self.require(
                instances.sequence_id.is_unique
                and assignments.frame_id.is_unique
                and set(assignments.frame_id) == set(self.ex.frame),
                "sequence-occurrence-coverage",
            )
            original = self.ex.manifest.set_index("frame_id")
            lookup = instances.set_index("sequence_id")
            for row in assignments.itertuples():
                self.require(row.sequence_id in lookup.index, "sequence-membership-unknown")
                source, interval = original.loc[row.frame_id], lookup.loc[row.sequence_id]
                self.require(
                    row.content_id == source.content_id
                    and row.video_id == source.video_id == interval.video_id
                    and row.sample_index == source.sample_index
                    and interval.start_sample_index
                    <= row.sample_index
                    <= interval.end_sample_index,
                    "sequence-occurrence-mismatch",
                )
                self.sequence_members[row.frame_id] = row.sequence_id
            for row in instances.itertuples():
                self.grid(row.video_id, row.start_sample_index, row.end_sample_index)
                members = assignments.loc[assignments.sequence_id.eq(row.sequence_id)]
                self.require(
                    set(members.sample_index)
                    == set(range(row.start_sample_index, row.end_sample_index + 1)),
                    "sequence-interval-coverage",
                )
                result["zones"].append(
                    {
                        "id": self.ex.alias(row.sequence_id, "sequence"),
                        "sourceVideo": self.ex.video[row.video_id],
                        "start": row.start_sample_index,
                        "end": row.end_sample_index,
                        "kind": "sequence_instance",
                        "decision": "reviewed",
                        "reviewId": None,
                    }
                )
        self.sequence_id = meta["artifact_id"]
        self.put("sequences", result, "artifact_available" if is_set else "review_required")

    def experiments(self, directory):
        meta = self.metadata(
            directory, {"sequence_experiment_suite_v1", "sequence_clustering_experiment_v1"}
        )
        self.source(meta["identity"]["sources"]["input"])
        summary = self.read(self.file(directory, meta, "summary.json"))
        artifact = {
            **self.base(meta, "experiment"),
            "encoder": None,
            "representation": None,
            "method": None,
            "coverage": summary.get("unique_fitting_contents"),
            "agreement": None,
            "stability": None,
            "recurrenceCandidates": None,
            "reviewState": "partial" if summary.get("posthoc_review_available") else "pending",
            "reviewMaskCoverage": None,
            "ablation": bool(meta["identity"]["config"].get("ablations", False)),
            "immutable": True,
        }
        existing = self.result["experiments"]["artifact"] or []
        self.put("experiments", [*existing, artifact])
        table_name = (
            "comparison.parquet"
            if meta["artifact_kind"] == "sequence_experiment_suite_v1"
            else "runs.parquet"
        )
        if table_name in meta["output_checksums"]:
            # A suite's comparison table can have multiple evaluation masks per run.
            # Preserve those rows; never average them into a single ranking score.
            path = self.file(directory, meta, table_name)
            table = pd.read_parquet(path)
            self.require(len(table) <= 256, "experiment-row-limit")
            rows = []
            encoders = {"clip": "CLIP", "dinov2": "DINOv2"}
            representations = {"original_l2": "original_l2", "pacmap": "PaCMAP", "tsne": "t-SNE"}
            methods = {
                "dbscan": "DBSCAN",
                "optics": "OPTICS",
                "hdbscan": "HDBSCAN",
                "agglomerative": "Agglomerative",
            }
            for index, row in enumerate(table.to_dict("records")):
                self.require(
                    row.get("encoder") in encoders
                    and row.get("representation") in representations
                    and row.get("algorithm") in methods,
                    "experiment-method-unsupported",
                )
                metrics = {
                    key: None if pd.isna(row[key]) else float(row[key])
                    for key in (
                        "ari",
                        "ami",
                        "evaluated_content_coverage",
                        "evaluated_n",
                        "total_contents",
                        "cluster_count",
                        "noise_coverage",
                    )
                    if key in row
                }
                rows.append(
                    {
                        **artifact,
                        "id": self.ex.alias(
                            f"{meta['artifact_id']}:{row['run_id']}:{index}", "experiment"
                        ),
                        "encoder": encoders[row["encoder"]],
                        "representation": representations[row["representation"]],
                        "method": methods[row["algorithm"]],
                        "reportedMetrics": metrics,
                        "coverage": None
                        if pd.isna(row.get("total_contents"))
                        else int(row["total_contents"]),
                        "reviewMaskCoverage": metrics.get("evaluated_content_coverage"),
                    }
                )
            if rows:
                self.put("experiments", [*existing, *rows])

    def evidence(self, directory):
        meta = self.metadata(
            directory, {"sequence_structure_review_v1", "sequence_external_evidence_v1"}
        )
        self.source(meta["identity"]["sources"]["input"])
        summary = self.read(self.file(directory, meta, "summary.json"))
        legacy = meta["identity"]["config"].get("adapter") == "hypatia_legacy_evidence_v1"
        artifact = {
            **self.base(meta, "evidence"),
            "source": "hypatia_legacy_evidence_v1" if legacy else "external",
            "state": "imported",
            "canonicalBinding": summary.get("canonical_source_verified"),
            "reviewMode": "legacy" if legacy else "unavailable",
        }
        self.put(
            "evidence",
            [*(self.result["evidence"]["artifact"] or []), artifact],
            "artifact_available",
        )
        if "intervals.parquet" in meta["output_checksums"]:
            intervals = self.table(
                directory,
                meta,
                "intervals.parquet",
                ["element_id", "timeline_id", "start", "end", "kind", "decision"],
            )
            self.require(intervals.element_id.is_unique, "evidence-duplicate-interval")
            sequence = self.result["sequences"]["artifact"] or {
                **self.base(meta, "sequence"),
                "zones": [],
                "reviews": [],
                "recurrence": [],
                "instanceReviewBound": False,
            }
            known = {}
            for row in intervals.itertuples():
                self.grid(row.timeline_id, row.start, row.end)
                # Known source membership is provenance, not sequence evidence.
                if row.kind == "known_source_video":
                    continue
                self.require(
                    row.kind in {"boundary_zone", "sequence_core", "sequence_core_candidate"}
                    and row.decision in {"candidate", "supported", "unsupported", "ambiguous"},
                    "evidence-interval-unsupported",
                )
                zid = self.ex.alias(f"{meta['artifact_id']}:{row.element_id}", "zone")
                known[row.element_id] = zid
                rid = (
                    None
                    if row.decision == "candidate"
                    else self.ex.alias(f"{meta['artifact_id']}:{row.element_id}", "review")
                )
                sequence["zones"].append(
                    {
                        "id": zid,
                        "sourceVideo": self.ex.video[row.timeline_id],
                        "start": row.start,
                        "end": row.end,
                        "kind": "boundary_zone"
                        if row.kind == "boundary_zone"
                        else "sequence_candidate",
                        "decision": row.decision,
                        "reviewId": rid,
                    }
                )
                if rid:
                    sequence["reviews"].append(
                        {"id": rid, "targetId": zid, "decision": row.decision, "groundTruth": False}
                    )
            if "observations.parquet" in meta["output_checksums"]:
                observations = self.table(
                    directory, meta, "observations.parquet", ["category", "subject_id", "object_id"]
                )
                for row in observations.itertuples():
                    if row.category == "recurrence":
                        self.require(
                            row.subject_id in known
                            and row.object_id in known
                            and row.subject_id != row.object_id,
                            "recurrence-interval-mismatch",
                        )
                        sequence["recurrence"].append(
                            {
                                "a": known[row.subject_id],
                                "b": known[row.object_id],
                                "score": None,
                                "kind": "visual_recurrence",
                            }
                        )
            self.put("sequences", sequence)

    def linkage(self, directory):
        meta = self.metadata(directory, {"labeled_video_link_candidates"})
        self.require(
            meta.get("video_dataset_id") == self.ex.dataset_id
            and meta.get("labeled_dataset_id") != self.ex.dataset_id,
            "linkage-dataset-mismatch",
        )
        self.require(
            self.sequence_members and meta.get("sequence_set_id") == self.sequence_id,
            "linkage-sequence-set-mismatch",
        )
        for encoder in ("clip", "dinov2"):
            self.require(
                meta.get(f"{encoder}_feature_space_id")
                == self.ex.features[encoder]["feature_space_id"],
                "linkage-feature-mismatch",
            )
        pairs = self.table(
            directory,
            meta,
            "content_candidates.parquet",
            [
                "candidate_id",
                "labeled_content_id",
                "video_content_id",
                "clip_cosine",
                "dinov2_cosine",
            ],
        )
        occurrences = self.table(
            directory,
            meta,
            "candidate_occurrences.parquet",
            ["video_content_id", "video_frame_id", "sequence_id"],
        )
        labeled = self.table(
            directory,
            meta,
            "labeled_occurrences.parquet",
            ["manifest_version", "frame_id", "content_id", "image_sha256", "label_sha256"],
        )
        self.require(
            labeled.manifest_version.eq("flir_canonical_candidate_v1").all()
            and labeled.content_id.eq(labeled.image_sha256).all(),
            "linkage-labeled-manifest-invalid",
        )
        identity_rows = sorted(
            labeled[["frame_id", "image_sha256", "label_sha256"]]
            .astype(str)
            .itertuples(index=False, name=None)
        )
        identity_payload = "flir_canonical_candidate_v1\x1f" + json.dumps(
            identity_rows, separators=(",", ":"), ensure_ascii=True
        )
        self.require(
            hashlib.sha256(identity_payload.encode()).hexdigest() == meta["labeled_dataset_id"],
            "linkage-labeled-dataset-mismatch",
        )
        self.require(
            labeled.frame_id.is_unique
            and pairs.candidate_id.is_unique
            and not pairs.duplicated(["labeled_content_id", "video_content_id"]).any(),
            "linkage-duplicate-identity",
        )
        self.require(
            set(pairs.video_content_id) <= set(self.ex.content)
            and set(pairs.labeled_content_id) <= set(labeled.content_id),
            "linkage-content-mismatch",
        )
        expected = self.ex.manifest.loc[self.ex.manifest.content_id.isin(pairs.video_content_id)]
        self.require(
            occurrences.video_frame_id.is_unique
            and set(occurrences.video_frame_id) == set(expected.frame_id),
            "linkage-occurrence-coverage",
        )
        original = expected.set_index("frame_id")
        membership = {}
        for row in occurrences.itertuples():
            self.require(
                row.video_content_id == original.loc[row.video_frame_id].content_id
                and row.sequence_id == self.sequence_members[row.video_frame_id],
                "linkage-occurrence-mismatch",
            )
            membership.setdefault(row.video_content_id, set()).add(row.sequence_id)
        self.labeled = {
            value: f"content-labeled-{i:06d}"
            for i, value in enumerate(sorted(set(labeled.content_id)), 1)
        }
        result = {
            **self.base(meta, "linkage"),
            "labeledDatasetId": "dataset-labeled",
            "sequenceSetId": self.ex.alias(self.sequence_id, "sequence"),
            "labeledContents": list(self.labeled.values()),
            "labeledOccurrences": [
                {"frameId": f"frame-labeled-{i:06d}", "contentId": self.labeled[row.content_id]}
                for i, row in enumerate(labeled.sort_values("frame_id").itertuples(), 1)
            ],
            "candidates": [],
        }
        for row in pairs.itertuples():
            self.require(
                -1.00001 <= row.clip_cosine <= 1.00001 and -1.00001 <= row.dinov2_cosine <= 1.00001,
                "linkage-score-invalid",
            )
            result["candidates"].append(
                {
                    "id": self.ex.alias(row.candidate_id, "linkage"),
                    "labeledContentId": self.labeled[row.labeled_content_id],
                    "videoContentId": self.ex.content[row.video_content_id],
                    "sequenceIds": [
                        self.ex.alias(sid, "sequence")
                        for sid in sorted(membership[row.video_content_id])
                    ],
                    "clipCosine": row.clip_cosine,
                    "dinov2Cosine": row.dinov2_cosine,
                    "state": "candidate",
                    "reviewId": None,
                }
            )
        self.linkage_id = meta["artifact_id"]
        self.put("linkage", result, "review_required")

    def review(self, directory):
        meta = self.metadata(directory, {"labeled_visual_dependency_manual_calibration"})
        source = meta["identity"]["calibration"]["sources"]
        self.require(
            self.linkage_id
            and source.get("linkage_id") == self.linkage_id
            and source.get("sequence_set_id") == self.sequence_id,
            "review-linkage-mismatch",
        )
        review = pd.read_csv(
            self.file(directory, meta, "review.csv"),
            keep_default_na=False,
            usecols=[
                "review_query_id",
                "labeled_content_id",
                "proposed_visual_dependency_group_id",
                "manual_decision",
            ],
        )
        self.require(
            review.review_query_id.is_unique
            and set(review.labeled_content_id) <= set(self.labeled)
            and set(review.manual_decision) <= {"", "supported", "unsupported", "ambiguous"},
            "review-identity-or-decision-invalid",
        )
        artifact = {
            **self.base(meta, "review"),
            "linkageId": self.ex.alias(self.linkage_id, "linkage"),
            "items": [
                {
                    "id": self.ex.alias(row.review_query_id, "review"),
                    "labeledContentId": self.labeled[row.labeled_content_id],
                    "groupId": self.ex.alias(row.proposed_visual_dependency_group_id, "group"),
                    "decision": row.manual_decision or "pending",
                }
                for row in review.itertuples()
            ],
        }
        self.put("groupReview", artifact, "artifact_available")
        self.review_id = meta["artifact_id"]

    def aggregate(self, directory):
        meta = self.metadata(directory, {"labeled_visual_dependency_manual_calibration_aggregate"})
        self.require(
            self.result["groupReview"]["artifact"] is not None, "aggregate-review-required"
        )
        summary = self.read(self.file(directory, meta, "summary.json"))
        # Binding is checked against the exact supplied revision, not safe aliases.
        history = self.read(self.file(directory, meta, "source_revisions.json"))
        self.require(
            isinstance(history, list)
            and len(history) > 0
            and self.review_id in {r["source_revision_id"] for r in history},
            "aggregate-review-mismatch",
        )
        for revision in history:
            sources = revision["metadata"]["identity"]["calibration"]["sources"]
            self.require(
                sources.get("linkage_id") == self.linkage_id
                and sources.get("sequence_set_id") == self.sequence_id,
                "aggregate-linkage-mismatch",
            )
        counts = summary["overall_pooled_descriptive"]["decision_counts"]
        artifact = {
            **self.base(meta, "review"),
            "linkageId": self.ex.alias(self.linkage_id, "linkage"),
            "reviews": summary["source_revision_count"],
            "reviewedItems": sum(counts[k] for k in ("supported", "unsupported", "ambiguous")),
            "confirmed": None,
            "rejected": None,
            "ambiguous": counts["ambiguous"],
            "supported": counts["supported"],
            "unsupported": counts["unsupported"],
            "conflicts": None,
        }
        self.put("reviewAggregation", artifact, "artifact_available")

    def variant(self, path):
        meta = self.read(path)
        self.require(meta.get("dataset_id") == self.ex.dataset_id, "variant-dataset-mismatch")
        name = meta.get("variant_name")
        self.require(
            name in {"unspecified", "original_with_hud", "no_hud"}
            and isinstance(meta.get("dataset_variant_id"), str),
            "variant-schema-unsupported",
        )
        self.result["variants"].append(
            {
                "id": self.ex.alias(meta["dataset_variant_id"], "variant"),
                "datasetId": "dataset-000001",
                "name": name,
            }
        )

    def finish(self):
        for path, expected in self.consumed.items():
            self.require(self.digest(path) == expected, "research-input-changed")
        return self.result
