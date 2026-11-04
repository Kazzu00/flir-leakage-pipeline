import { z } from 'zod'

export type JsonValue =
  | null
  | boolean
  | number
  | string
  | JsonValue[]
  | { [key: string]: JsonValue }

export const JsonValueSchema: z.ZodType<JsonValue> = z.lazy(() =>
  z.union([
    z.null(),
    z.boolean(),
    z.number(),
    z.string(),
    z.array(JsonValueSchema),
    z.record(z.string(), JsonValueSchema),
  ]),
)

const NonNegativeInt = z.number().int().nonnegative()
const NullableNonNegativeInt = NonNegativeInt.nullable()
const TemporalSourceSchema = z.enum([
  'filename_heuristic',
  'sampled_video_grid',
  'unknown',
])
const PartitionSchema = z.enum(['train', 'val', 'test'])

export const TemporalSpanSchema = z
  .object({
    timeline_id: z.string(),
    frame_index_min: NullableNonNegativeInt,
    frame_index_max: NullableNonNegativeInt,
    n_unique_contents: NonNegativeInt,
  })
  .strict()

export const TimelinePointSchema = z
  .object({
    content_id: z.string(),
    record_ids: z.array(z.string()),
    frame_index: NullableNonNegativeInt,
    timestamp_seconds: z.number().nullable(),
    preview_key: z.string(),
  })
  .strict()

export const TimelineSchema = z
  .object({
    timeline_id: z.string(),
    source_video_id: z.string().nullable(),
    source_archive: z.string().nullable(),
    inferred_family: z.string().nullable(),
    temporal_source: TemporalSourceSchema,
    points: z.array(TimelinePointSchema),
  })
  .strict()

export const CandidatePairSemanticsSchema = z
  .object({
    automatic_confirmation: z.literal(false).default(false),
    confirmed_dependency: z.literal(false).default(false),
    evidence_status: z.literal('candidate_only').default('candidate_only'),
    ground_truth: z.literal(false).default(false),
    sequence_identity: z.literal(false).default(false),
    split_constraint: z.literal(false).default(false),
  })
  .strict()

export const SemanticsSchema = z
  .object({
    automatic_confirmation: z.literal(false).default(false),
    boundary_zone: z
      .literal('inclusive uncertainty interval, never an exact cut')
      .default('inclusive uncertainty interval, never an exact cut'),
    candidate_pairs: CandidatePairSemanticsSchema.prefault({}),
    clusters_are_sequences: z.literal(false).default(false),
    ground_truth_clusters: z.literal(false).default(false),
    ground_truth_sequences: z.literal(false).default(false),
    partition_vocabulary: z.array(PartitionSchema).default([
      'train',
      'val',
      'test',
    ]),
    sequence_instances_created: z.literal(false).default(false),
  })
  .strict()

export const LabeledManifestBindingSchema = z
  .object({
    historical_manifest_sha256: z.string(),
    current_manifest_sha256: z.string(),
    source_manifest_reserialized: z.boolean(),
    exact_tabular_identity_verified: z.literal(true),
  })
  .strict()

export const SequenceManifestBindingSchema = z
  .object({
    historical_manifest_sha256: z.string(),
    current_manifest_sha256: z.string(),
    source_manifest_reserialized: z.boolean(),
    exact_occurrence_binding_verified: z.literal(true),
  })
  .strict()

export const SourceSchema = z
  .object({
    artifact_id: z.string(),
    artifact_kind: z.string(),
    metadata_sha256: z.string(),
    output_checksums: z.record(z.string(), z.string()),
    labeled_manifest_binding: z
      .union([
        LabeledManifestBindingSchema,
        SequenceManifestBindingSchema,
        z.null(),
      ])
      .optional(),
  })
  .strict()

export const ManifestSchema = z
  .object({
    generated_at: z.string(),
    source_commit: z.string().nullable(),
    source_dirty: z.boolean().nullable(),

    dataset_id: z.string(),
    manifest_sha256: z.string(),
    plan_id: z.string(),
    plan_sha256: z.string(),
    runtime_freeze_sha256: z.string(),

    record_count: NonNegativeInt,
    unique_content_count: NonNegativeInt,
    labeled_record_count: NonNegativeInt,
    labeled_unique_content_count: NonNegativeInt,
    source_video_count: NonNegativeInt,
    timeline_count: NonNegativeInt,
    split_count: NonNegativeInt,
    strategy_count: NonNegativeInt,
    candidate_pair_count: NonNegativeInt,

    clustering_configuration_ids: z.array(z.string()),
    evidence_sources: z.array(SourceSchema),
    limitations: z.array(z.string()),
    terminology: z.record(z.string(), z.string()),
    file_sha256: z.record(z.string(), z.string()),

    generated_from_verified_artifacts: z.literal(true).default(true),
    schema_version: z
      .literal('organization-evidence-v2')
      .default('organization-evidence-v2'),
    scientific_result: z
      .literal('existing_evidence_export')
      .default('existing_evidence_export'),
    semantics: SemanticsSchema.prefault({}),
    verification_scope: z
      .literal('stored identities, checksums and membership relationships')
      .default('stored identities, checksums and membership relationships'),
  })
  .strict()

export const ContentSchema = z
  .object({
    content_id: z.string(),
    canonical_filename: z.string().nullable(),
    representative_record_id: z.string(),
    source_video_id: z.string().nullable(),
    frame_index: NullableNonNegativeInt,
    source_frame_index: NullableNonNegativeInt.optional(),
    source_frame_index_estimate: NullableNonNegativeInt,
    timestamp_seconds: z.number().nullable(),
    temporal_source: TemporalSourceSchema,
    record_ids: z.array(z.string()),
    class_ids: z.array(NonNegativeInt).nullable(),
    class_names: z.array(z.string()).nullable(),
    annotation_consensus: z.enum([
      'identical_label_bytes',
      'different_label_bytes',
      'unavailable',
    ]),
    preview_key: z.string(),
    width: NullableNonNegativeInt,
    height: NullableNonNegativeInt,
  })
  .strict()

export const RecordSchema = z
  .object({
    record_id: z.string(),
    content_id: z.string(),
    cohort: z.enum(['labeled', 'video_evidence']),
    filename: z.string().nullable(),
    source_archive: z.string().nullable(),
    source_video_id: z.string().nullable(),
    timeline_id: z.string().nullable(),
    frame_index: NullableNonNegativeInt,
    source_frame_index: NullableNonNegativeInt.optional(),
    source_frame_index_estimate: NullableNonNegativeInt,
    timestamp_seconds: z.number().nullable(),
    temporal_source: TemporalSourceSchema,
    original_split: PartitionSchema.nullable(),
    class_ids: z.array(NonNegativeInt).nullable(),
    class_names: z.array(z.string()).nullable(),
    num_objects: NullableNonNegativeInt,
    label_empty: z.boolean().nullable(),
    width: NullableNonNegativeInt,
    height: NullableNonNegativeInt,
  })
  .strict()

export const PartitionSummarySchema = z
  .object({
    n_records: NonNegativeInt,
    n_unique_contents: NonNegativeInt,
  })
  .strict()

export const ClassSupportSchema = z
  .object({
    split: PartitionSchema,
    class_id: NonNegativeInt,
    class_name: z.string(),
    images: NonNegativeInt,
    contents: NonNegativeInt,
    instances: NonNegativeInt,
    empty_annotations: NonNegativeInt,
    instance_percentage: z.number(),
    global_instance_percentage: z.number(),
    absolute_percentage_point_deviation: z.number(),
    target_instances: z.number(),
  })
  .strict()

export const SplitSchema = z
  .object({
    strategy: z.string(),
    split_seed: NonNegativeInt,
    split_space_id: z.string(),
    artifact_id: z.string(),
    source_strategy: z.enum([
      'historical',
      'random_content',
      'cluster_aware',
    ]),
    cluster_run_id: z.string().nullable(),
    partitions: z.record(z.string(), PartitionSummarySchema),
    class_support: z.array(ClassSupportSchema).nullable(),
  })
  .strict()

export const SplitMembershipSchema = z
  .object({
    strategy: z.string(),
    split_seed: NonNegativeInt,
    split_space_id: z.string(),
    partition: PartitionSchema,
    record_id: z.string(),
    content_id: z.string(),
  })
  .strict()

export const ClusteringConfigurationSchema = z
  .object({
    cluster_run_id: z.string(),
    source_kind: z.enum([
      'full_clustering_artifact',
      'frozen_split_membership',
    ]),
    full_clustering_artifact_available: z.boolean(),
    membership_consistency_verified: z.boolean(),
    source_split_ids: z.array(z.string()),
    source_membership_checksums: z.record(z.string(), z.string()),

    dataset_id: z.string(),
    feature_space_id: z.string().nullable(),
    configuration_id: z.string().nullable(),
    model_id: z.string().nullable(),
    strategy_labels: z.array(z.string()),
    representation: z.string().nullable(),
    reduction_space_id: z.string().nullable(),
    reduction_seed: NullableNonNegativeInt,
    extractor: z.string().nullable(),
    algorithm: z.enum(['dbscan', 'optics', 'hdbscan']).nullable(),

    parameters: z
      .record(z.string(), JsonValueSchema)
      .nullable(),
    effective_parameters: z
      .record(z.string(), JsonValueSchema)
      .nullable(),

    n_clusters: NonNegativeInt,
    n_noise: NonNegativeInt,
    noise_fraction: z.number(),

    ground_truth: z.literal(false).default(false),
  })
  .strict()

export const ClusterSchema = z
  .object({
    cluster_run_id: z.string(),
    cluster_id: z.number().int(),
    size_unique_contents: NonNegativeInt,
    is_noise: z.boolean(),
    source_video_distribution: z.record(z.string(), NonNegativeInt),
    temporal_spans: z.array(TemporalSpanSchema),
    automatic_confirmation: z.literal(false).default(false),
    ground_truth: z.literal(false).default(false),
  })
  .strict()

export const ClusterMembershipSchema = z
  .object({
    cluster_run_id: z.string(),
    cluster_id: z.number().int(),
    content_id: z.string(),
    is_noise: z.boolean(),

    core_distance: z.number().nullable().optional(),
    core_distance_infinite: z.boolean().nullable().optional(),
    ordering_position: NullableNonNegativeInt.optional(),
    probability: z.number().nullable().optional(),
    reachability: z.number().nullable().optional(),
    reachability_infinite: z.boolean().nullable().optional(),
  })
  .strict()

export const CandidatePairSchema = z
  .object({
    evidence_artifact_id: z.string(),
    candidate_id: z.string(),
    labeled_content_id: z.string(),
    video_content_id: z.string(),

    clip_cosine: z.number(),
    dinov2_cosine: z.number(),
    clip_rank: NullableNonNegativeInt,
    dinov2_rank: NullableNonNegativeInt,
    clip_topk: z.boolean(),
    dinov2_topk: z.boolean(),
    both_topk: z.boolean(),
    mean_reciprocal_rank: z.number(),

    video_occurrence_count: NonNegativeInt,
    video_sequence_count: NonNegativeInt,
  })
  .strict()

export const LinkageGroupSchema = z
  .object({
    evidence_artifact_id: z.string(),
    linkage_group_id: z.string(),
    kind: z.enum(['diagnostic_component', 'candidate_core']),
    member_count: NonNegativeInt,
    source_video_ids: z.array(z.string()),
    temporal_spans: z.array(TemporalSpanSchema),
    evidence_sources: z.array(z.string()),
    upstream_metadata: z.record(z.string(), JsonValueSchema),
    review_state: z.string().nullable(),

    automatic_confirmation: z.literal(false).default(false),
    ground_truth: z.literal(false).default(false),
  })
  .strict()

export const LinkageMembershipSchema = z
  .object({
    evidence_artifact_id: z.string(),
    linkage_group_id: z.string(),
    content_id: z.string(),
    record_ids: z.array(z.string()),
    source_video_ids: z.array(z.string()),
    frame_indices: z.array(NonNegativeInt),
    role: z.literal('core_member'),
    upstream_element_ids: z.array(z.string()),
  })
  .strict()

export const BoundaryZoneSchema = z
  .object({
    evidence_artifact_id: z.string(),
    element_id: z.string(),
    upstream_timeline_id: z.string(),
    timeline_ids: z.array(z.string()),
    start: NonNegativeInt,
    end: NonNegativeInt,
    decision: z.enum([
      'supported',
      'ambiguous',
      'unsupported',
      'candidate',
    ]),
    notes: z.string(),

    exact_cut: z.literal(false).default(false),
    inclusive: z.literal(true).default(true),
    kind: z.literal('boundary_zone').default('boundary_zone'),
  })
  .strict()
  .refine((zone) => zone.start <= zone.end, 'Reversed boundary zone')

export const ReviewSchema = z
  .object({
    evidence_artifact_id: z.string(),
    review_query_id: z.string(),
    content_id: z.string(),
    proposed_visual_dependency_group_id: z.string(),
    decision: z.record(z.string(), JsonValueSchema),

    automatic_confirmation: z.literal(false).default(false),
    ground_truth: z.literal(false).default(false),
  })
  .strict()

export const MediaSchema = z
  .object({
    preview_key: z.string(),
    content_id: z.string(),
    relative_path: z.string().nullable(),
    width: NullableNonNegativeInt,
    height: NullableNonNegativeInt,
    checksum: z.string().nullable(),
    media_available: z.boolean(),
    unavailable_reason: z
      .enum(['not_requested', 'source_unavailable', 'source_invalid'])
      .nullable(),
  })
  .strict()

export const OrganizationEvidenceBundleSchema = z
  .object({
    manifest: ManifestSchema,
    contents: z.array(ContentSchema),
    records: z.array(RecordSchema),
    splits: z.array(SplitSchema),
    split_memberships: z.array(SplitMembershipSchema),
    clustering_configurations: z.array(ClusteringConfigurationSchema),
    clusters: z.array(ClusterSchema),
    cluster_memberships: z.array(ClusterMembershipSchema),
    candidate_pairs: z.array(CandidatePairSchema),
    linkage_groups: z.array(LinkageGroupSchema),
    linkage_memberships: z.array(LinkageMembershipSchema),
    boundary_zones: z.array(BoundaryZoneSchema),
    reviews: z.array(ReviewSchema),
    timelines: z.array(TimelineSchema),
    media: z.array(MediaSchema),
  })
  .strict()

export const OrganizationEvidenceResourceSchemas = {
  manifest: ManifestSchema,
  contents: z.array(ContentSchema),
  records: z.array(RecordSchema),
  splits: z.array(SplitSchema),
  split_memberships: z.array(SplitMembershipSchema),
  clustering_configurations: z.array(ClusteringConfigurationSchema),
  clusters: z.array(ClusterSchema),
  cluster_memberships: z.array(ClusterMembershipSchema),
  candidate_pairs: z.array(CandidatePairSchema),
  linkage_groups: z.array(LinkageGroupSchema),
  linkage_memberships: z.array(LinkageMembershipSchema),
  boundary_zones: z.array(BoundaryZoneSchema),
  reviews: z.array(ReviewSchema),
  timelines: z.array(TimelineSchema),
  media: z.array(MediaSchema),
} as const

export type OrganizationManifest = z.infer<typeof ManifestSchema>
export type OrganizationContent = z.infer<typeof ContentSchema>
export type OrganizationRecord = z.infer<typeof RecordSchema>
export type OrganizationSplit = z.infer<typeof SplitSchema>
export type OrganizationSplitMembership = z.infer<
  typeof SplitMembershipSchema
>
export type OrganizationClusteringConfiguration = z.infer<
  typeof ClusteringConfigurationSchema
>
export type OrganizationCluster = z.infer<typeof ClusterSchema>
export type OrganizationClusterMembership = z.infer<
  typeof ClusterMembershipSchema
>
export type OrganizationCandidatePair = z.infer<typeof CandidatePairSchema>
export type OrganizationLinkageGroup = z.infer<typeof LinkageGroupSchema>
export type OrganizationLinkageMembership = z.infer<
  typeof LinkageMembershipSchema
>
export type OrganizationBoundaryZone = z.infer<typeof BoundaryZoneSchema>
export type OrganizationReview = z.infer<typeof ReviewSchema>
export type OrganizationTimeline = z.infer<typeof TimelineSchema>
export type OrganizationMedia = z.infer<typeof MediaSchema>
export type OrganizationEvidenceBundle = z.infer<
  typeof OrganizationEvidenceBundleSchema
>
export type OrganizationEvidenceResource =
  keyof typeof OrganizationEvidenceResourceSchemas


