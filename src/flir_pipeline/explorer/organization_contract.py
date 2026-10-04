"""Typed presentation joins for existing evidence, never scientific assignments."""

from collections import Counter, defaultdict
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

SCHEMA_VERSION = "organization-evidence-v1"
ID = Annotated[str, Field(min_length=1)]
Count = Annotated[int, Field(ge=0)]
Index = Annotated[int, Field(ge=0)]
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Partition = Literal["train", "val", "test"]
TemporalSource = Literal["filename_heuristic", "sampled_video_grid", "unknown"]


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Semantics(Model):
    ground_truth_clusters: Literal[False] = False
    ground_truth_sequences: Literal[False] = False
    automatic_confirmation: Literal[False] = False
    sequence_instances_created: Literal[False] = False
    clusters_are_sequences: Literal[False] = False
    partition_vocabulary: list[Partition] = ["train", "val", "test"]
    boundary_zone: Literal["inclusive uncertainty interval, never an exact cut"] = (
        "inclusive uncertainty interval, never an exact cut"
    )


class Source(Model):
    artifact_id: ID
    artifact_kind: ID
    metadata_sha256: Digest
    output_checksums: dict[str, Digest]


class Manifest(Model):
    schema_version: Literal["organization-evidence-v1"] = SCHEMA_VERSION
    scientific_result: Literal["existing_evidence_export"] = "existing_evidence_export"
    generated_from_verified_artifacts: Literal[True] = True
    verification_scope: Literal[
        "stored identities, checksums and membership relationships"
    ] = "stored identities, checksums and membership relationships"
    generated_at: ID
    source_commit: str | None
    source_dirty: bool | None
    dataset_id: ID
    manifest_sha256: Digest
    plan_id: ID
    plan_sha256: Digest
    runtime_freeze_sha256: Digest
    record_count: Count
    unique_content_count: Count
    labeled_record_count: Count
    labeled_unique_content_count: Count
    source_video_count: Count
    timeline_count: Count
    split_count: Count
    strategy_count: Count
    clustering_configuration_ids: list[ID]
    evidence_sources: list[Source]
    semantics: Semantics = Field(default_factory=Semantics)
    limitations: list[str]
    terminology: dict[str, str]
    file_sha256: dict[str, Digest]


class Record(Model):
    record_id: ID
    content_id: ID
    cohort: Literal["labeled", "video_evidence"]
    filename: str | None
    source_archive: str | None
    source_video_id: str | None
    timeline_id: str | None
    frame_index: Index | None
    source_frame_index: Index | None = None
    source_frame_index_estimate: Index | None
    timestamp_seconds: Annotated[float, Field(ge=0)] | None
    temporal_source: TemporalSource
    original_split: Partition | None
    class_ids: list[Annotated[int, Field(ge=0, le=4)]] | None
    class_names: list[str] | None
    num_objects: Count | None
    label_empty: bool | None
    width: Annotated[int, Field(gt=0)] | None
    height: Annotated[int, Field(gt=0)] | None


class Content(Model):
    content_id: ID
    canonical_filename: str | None
    representative_record_id: ID
    source_video_id: str | None
    frame_index: Index | None
    source_frame_index: Index | None = None
    source_frame_index_estimate: Index | None
    timestamp_seconds: Annotated[float, Field(ge=0)] | None
    temporal_source: TemporalSource
    record_ids: list[ID] = Field(min_length=1)
    class_ids: list[Annotated[int, Field(ge=0, le=4)]] | None
    class_names: list[str] | None
    annotation_consensus: Literal["identical", "conflicting", "unavailable"]
    preview_key: ID
    width: Annotated[int, Field(gt=0)] | None
    height: Annotated[int, Field(gt=0)] | None


class PartitionSummary(Model):
    n_records: Count
    n_unique_contents: Count


class Split(Model):
    strategy: ID
    split_seed: Count
    split_space_id: ID
    artifact_id: ID
    source_strategy: Literal["historical", "random_content", "cluster_aware"]
    cluster_run_id: str | None
    partitions: dict[Partition, PartitionSummary]
    class_support: list[dict[str, JsonValue]] | None


class SplitMembership(Model):
    strategy: ID
    split_seed: Count
    split_space_id: ID
    partition: Partition
    record_id: ID
    content_id: ID


class ClusteringConfiguration(Model):
    cluster_run_id: ID
    dataset_id: ID
    feature_space_id: ID
    configuration_id: ID
    model_id: str | None
    strategy_labels: list[ID]
    representation: ID
    reduction_space_id: str | None
    reduction_seed: Count | None
    extractor: ID
    algorithm: Literal["dbscan", "optics", "hdbscan"]
    parameters: dict[str, JsonValue]
    effective_parameters: dict[str, JsonValue]
    n_clusters: Count
    n_noise: Count
    noise_fraction: Annotated[float, Field(ge=0, le=1)]
    ground_truth: Literal[False] = False


class TemporalSpan(Model):
    timeline_id: ID
    frame_index_min: Index | None
    frame_index_max: Index | None
    n_unique_contents: Count


class Cluster(Model):
    cluster_run_id: ID
    cluster_id: Annotated[int, Field(ge=-1)]
    size_unique_contents: Count
    is_noise: bool
    source_video_distribution: dict[str, Count]
    temporal_spans: list[TemporalSpan]
    ground_truth: Literal[False] = False
    automatic_confirmation: Literal[False] = False


class ClusterMembership(Model):
    cluster_run_id: ID
    cluster_id: Annotated[int, Field(ge=-1)]
    content_id: ID
    is_noise: bool
    probability: Annotated[float, Field(ge=0, le=1)] | None = None
    reachability: Annotated[float, Field(ge=0)] | None = None
    reachability_infinite: bool | None = None
    core_distance: Annotated[float, Field(ge=0)] | None = None
    core_distance_infinite: bool | None = None
    ordering_position: Index | None = None


class LinkageGroup(Model):
    evidence_artifact_id: ID
    linkage_group_id: ID
    kind: Literal["candidate_pair", "diagnostic_component", "candidate_core"]
    member_count: Count
    source_video_ids: list[ID]
    temporal_spans: list[TemporalSpan]
    evidence_sources: list[ID]
    upstream_metadata: dict[str, JsonValue]
    review_state: str | None
    ground_truth: Literal[False] = False
    automatic_confirmation: Literal[False] = False


class LinkageMembership(Model):
    evidence_artifact_id: ID
    linkage_group_id: ID
    content_id: ID
    record_ids: list[ID]
    source_video_ids: list[ID]
    frame_indices: list[Index]
    role: Literal["query", "candidate", "query_and_candidate", "core_member"]
    upstream_element_ids: list[ID]


class BoundaryZone(Model):
    evidence_artifact_id: ID
    element_id: ID
    upstream_timeline_id: ID
    timeline_ids: list[ID]
    start: Index
    end: Index
    kind: Literal["boundary_zone"] = "boundary_zone"
    decision: Literal["supported", "ambiguous", "unsupported", "candidate"]
    notes: str
    inclusive: Literal[True] = True
    exact_cut: Literal[False] = False

    @model_validator(mode="after")
    def interval(self):
        if self.end < self.start:
            raise ValueError("Reversed boundary zone")
        return self


class Review(Model):
    evidence_artifact_id: ID
    review_query_id: ID
    content_id: ID
    proposed_visual_dependency_group_id: ID
    decision: dict[str, JsonValue]
    ground_truth: Literal[False] = False
    automatic_confirmation: Literal[False] = False


class TimelinePoint(Model):
    content_id: ID
    record_ids: list[ID] = Field(min_length=1)
    frame_index: Index | None
    timestamp_seconds: Annotated[float, Field(ge=0)] | None
    preview_key: ID


class Timeline(Model):
    timeline_id: ID
    source_video_id: str | None
    source_archive: str | None
    inferred_family: str | None
    temporal_source: TemporalSource
    points: list[TimelinePoint]


class Media(Model):
    preview_key: ID
    content_id: ID
    relative_path: str | None
    width: Annotated[int, Field(gt=0)] | None
    height: Annotated[int, Field(gt=0)] | None
    checksum: Digest | None
    media_available: bool
    unavailable_reason: (
        Literal["not_requested", "source_unavailable", "source_invalid"] | None
    )


def unique(rows, key, name):
    result = {key(row): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate {name} primary ID")
    return result


class OrganizationExport(Model):
    manifest: Manifest
    contents: list[Content]
    records: list[Record]
    splits: list[Split]
    split_memberships: list[SplitMembership]
    clustering_configurations: list[ClusteringConfiguration]
    clusters: list[Cluster]
    cluster_memberships: list[ClusterMembership]
    linkage_groups: list[LinkageGroup]
    linkage_memberships: list[LinkageMembership]
    boundary_zones: list[BoundaryZone]
    reviews: list[Review]
    timelines: list[Timeline]
    media: list[Media]

    @model_validator(mode="after")
    def relationships(self):
        """Check relational coverage, including the historical multiset exception."""
        contents = unique(self.contents, lambda r: r.content_id, "content")
        records = unique(self.records, lambda r: r.record_id, "record")
        splits = unique(self.splits, lambda r: r.split_space_id, "split")
        configurations = unique(
            self.clustering_configurations, lambda r: r.cluster_run_id, "configuration"
        )
        clusters = unique(
            self.clusters, lambda r: (r.cluster_run_id, r.cluster_id), "cluster"
        )
        groups = unique(
            self.linkage_groups,
            lambda r: (r.evidence_artifact_id, r.linkage_group_id),
            "linkage group",
        )
        timelines = unique(self.timelines, lambda r: r.timeline_id, "timeline")
        media = unique(self.media, lambda r: r.content_id, "media content")
        sources = unique(
            self.manifest.evidence_sources, lambda r: r.artifact_id, "evidence source"
        )
        if not set(splits) | set(configurations) <= set(sources):
            raise ValueError("Missing split/clustering source receipt")
        if any(
            g.evidence_artifact_id not in sources
            or not set(g.evidence_sources) <= set(sources)
            for g in groups.values()
        ):
            raise ValueError("Missing linkage evidence source receipt")
        if any(
            r.evidence_artifact_id not in sources
            for r in [*self.reviews, *self.boundary_zones]
        ):
            raise ValueError("Missing review/boundary source receipt")
        unique(self.media, lambda r: r.preview_key, "preview")
        unique(
            self.boundary_zones,
            lambda r: (r.evidence_artifact_id, r.element_id),
            "boundary",
        )
        unique(
            self.reviews,
            lambda r: (r.evidence_artifact_id, r.review_query_id),
            "review",
        )
        by_content = defaultdict(set)
        for r in records.values():
            if r.content_id not in contents or (
                r.timeline_id is not None and r.timeline_id not in timelines
            ):
                raise ValueError("Broken record content/timeline reference")
            by_content[r.content_id].add(r.record_id)
        if set(media) != set(contents):
            raise ValueError("Media must cover every unique content")
        for c in contents.values():
            if (
                set(c.record_ids) != by_content[c.content_id]
                or len(set(c.record_ids)) != len(c.record_ids)
                or c.representative_record_id not in c.record_ids
            ):
                raise ValueError("Content occurrence mapping differs")
            m = media[c.content_id]
            if m.preview_key != c.preview_key:
                raise ValueError("Broken preview join")
            if m.media_available:
                from flir_pipeline.data.local_images import relative_posix_path

                if (
                    None in (m.relative_path, m.width, m.height, m.checksum)
                    or m.unavailable_reason is not None
                ):
                    raise ValueError("Available media needs complete receipt")
                relative_posix_path(m.relative_path)
            elif (
                any(
                    v is not None
                    for v in (m.relative_path, m.width, m.height, m.checksum)
                )
                or m.unavailable_reason is None
            ):
                raise ValueError("Missing media must remain explicitly unavailable")
        unique(
            self.split_memberships,
            lambda r: (r.split_space_id, r.record_id),
            "split record",
        )
        labeled = {r.record_id for r in records.values() if r.cohort == "labeled"}
        assigned = defaultdict(list)
        for m in self.split_memberships:
            if (
                m.split_space_id not in splits
                or m.record_id not in labeled
                or records[m.record_id].content_id != m.content_id
            ):
                raise ValueError("Broken split record/content reference")
            s = splits[m.split_space_id]
            if (m.strategy, m.split_seed) != (s.strategy, s.split_seed):
                raise ValueError("Split identity mismatch")
            assigned[m.split_space_id].append(m)
        for s in splits.values():
            rows = assigned[s.split_space_id]
            if {r.record_id for r in rows} != labeled or set(s.partitions) != {
                "train",
                "val",
                "test",
            }:
                raise ValueError("Incomplete split coverage")
            for p, counts in s.partitions.items():
                part = [r for r in rows if r.partition == p]
                if (counts.n_records, counts.n_unique_contents) != (
                    len(part),
                    len({r.content_id for r in part}),
                ):
                    raise ValueError("Partition counts differ from membership")
            if s.source_strategy == "historical":
                if any(
                    r.partition != records[r.record_id].original_split for r in rows
                ):
                    raise ValueError("Historical membership changed")
            else:
                presence = defaultdict(set)
                for r in rows:
                    presence[r.content_id].add(r.partition)
                if any(len(p) != 1 for p in presence.values()):
                    raise ValueError("Exact content fractured by new split")
            if (s.source_strategy == "cluster_aware") != (
                s.cluster_run_id in configurations
            ):
                raise ValueError("Unresolved selected clustering")
        unique(
            self.cluster_memberships,
            lambda r: (r.cluster_run_id, r.content_id),
            "cluster membership",
        )
        sizes = Counter()
        clustered = defaultdict(set)
        labels = {}
        for m in self.cluster_memberships:
            key = (m.cluster_run_id, m.cluster_id)
            if (
                key not in clusters
                or m.content_id not in contents
                or m.is_noise != (m.cluster_id == -1)
                or m.is_noise
                and m.probability not in (None, 0)
            ):
                raise ValueError("Broken cluster content reference/noise flag")
            sizes[key] += 1
            clustered[m.cluster_run_id].add(m.content_id)
            labels[m.cluster_run_id, m.content_id] = m.cluster_id
        for key, c in clusters.items():
            if (
                c.cluster_run_id not in configurations
                or c.size_unique_contents != sizes[key]
                or c.is_noise != (c.cluster_id == -1)
            ):
                raise ValueError("Cluster summary mismatch")
        for run in configurations.values():
            if clustered[run.cluster_run_id] != {
                records[r].content_id for r in labeled
            }:
                raise ValueError("Selected clustering does not cover labeled contents")
            n_noise = sizes[run.cluster_run_id, -1]
            n_clusters = sum(k[0] == run.cluster_run_id and k[1] >= 0 for k in clusters)
            if (run.n_clusters, run.n_noise) != (n_clusters, n_noise) or abs(
                run.noise_fraction - n_noise / len(clustered[run.cluster_run_id])
            ) > 1e-12:
                raise ValueError("Clustering statistics differ from saved labels")
        for s in splits.values():
            if s.cluster_run_id:
                presence = defaultdict(set)
                for r in assigned[s.split_space_id]:
                    label = labels[s.cluster_run_id, r.content_id]
                    if label >= 0:
                        presence[label].add(r.partition)
                if any(len(p) != 1 for p in presence.values()):
                    raise ValueError("Split fractures a non-noise cluster")
        unique(
            self.linkage_memberships,
            lambda r: (r.evidence_artifact_id, r.linkage_group_id, r.content_id),
            "linkage membership",
        )
        sizes = Counter()
        for m in self.linkage_memberships:
            key = m.evidence_artifact_id, m.linkage_group_id
            if (
                key not in groups
                or m.content_id not in contents
                or not set(m.record_ids) <= by_content[m.content_id]
            ):
                raise ValueError("Broken linkage content/record reference")
            linked_records = [records[rid] for rid in m.record_ids]
            if (
                len(set(m.record_ids)) != len(m.record_ids)
                or not m.record_ids
                or m.source_video_ids
                != sorted(
                    {r.source_video_id for r in linked_records if r.source_video_id}
                )
                or m.frame_indices
                != sorted(
                    {r.frame_index for r in linked_records if r.frame_index is not None}
                )
            ):
                raise ValueError(
                    "Linkage temporal metadata differs from stored occurrences"
                )
            sizes[key] += 1
        if any(g.member_count != sizes[key] for key, g in groups.items()):
            raise ValueError("Linkage summary differs from memberships")
        covered = Counter()
        for t in timelines.values():
            for point in t.points:
                if (
                    point.content_id not in contents
                    or point.preview_key != contents[point.content_id].preview_key
                ):
                    raise ValueError("Broken timeline content/preview reference")
                for rid in point.record_ids:
                    if rid not in records or (
                        records[rid].timeline_id,
                        records[rid].content_id,
                        records[rid].frame_index,
                        records[rid].timestamp_seconds,
                    ) != (
                        t.timeline_id,
                        point.content_id,
                        point.frame_index,
                        point.timestamp_seconds,
                    ):
                        raise ValueError("Timeline position differs from occurrence")
                    covered[rid] += 1
        if dict(covered) != {
            r.record_id: 1 for r in records.values() if r.timeline_id is not None
        }:
            raise ValueError("Timelines must preserve all located occurrences")
        if any(
            not set(z.timeline_ids) <= set(timelines) for z in self.boundary_zones
        ) or any(r.content_id not in contents for r in self.reviews):
            raise ValueError("Unknown boundary/review reference")
        m = self.manifest
        actual = (
            len(records),
            len(contents),
            len(labeled),
            len({records[r].content_id for r in labeled}),
            len({r.source_video_id for r in records.values() if r.source_video_id}),
            len(timelines),
            len(splits),
            len({s.strategy for s in splits.values()}),
        )
        expected = (
            m.record_count,
            m.unique_content_count,
            m.labeled_record_count,
            m.labeled_unique_content_count,
            m.source_video_count,
            m.timeline_count,
            m.split_count,
            m.strategy_count,
        )
        if actual != expected or set(m.clustering_configuration_ids) != set(
            configurations
        ):
            raise ValueError("Manifest counts/identities differ from export")
        return self


def schema():
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        **OrganizationExport.model_json_schema(),
    }
