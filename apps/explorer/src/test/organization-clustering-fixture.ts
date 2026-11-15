import type {
  OrganizationEvidenceBundle,
  OrganizationClusteringConfiguration,
} from '@/contracts/organization-evidence-v2'

/** Synthetic, offline contract fixture. No exported scientific values or media. */
export function organizationClusteringFixture(count = 2) {
  const configuration: OrganizationClusteringConfiguration = {
    cluster_run_id: 'synthetic-run-a',
    source_kind: 'full_clustering_artifact',
    full_clustering_artifact_available: true,
    membership_consistency_verified: true,
    source_split_ids: [],
    source_membership_checksums: {},
    dataset_id: 'synthetic-dataset',
    feature_space_id: 'synthetic-feature-a',
    configuration_id: 'synthetic-config-a',
    model_id: 'synthetic-model',
    strategy_labels: [],
    representation: 'synthetic normalized embeddings',
    reduction_space_id: null,
    reduction_seed: null,
    extractor: 'synthetic-extractor',
    algorithm: 'dbscan',
    parameters: { eps: 0.2, min_samples: 2 },
    effective_parameters: { metric: 'euclidean' },
    n_clusters: 2,
    n_noise: 1,
    noise_fraction: 1 / (count + 2),
    ground_truth: false,
  }
  const data: Pick<
    OrganizationEvidenceBundle,
    | 'clustering_configurations'
    | 'clusters'
    | 'cluster_memberships'
    | 'contents'
    | 'records'
    | 'media'
  > = {
    clustering_configurations: [
      configuration,
      {
        ...configuration,
        cluster_run_id: 'synthetic-run-frozen',
        configuration_id: null,
        feature_space_id: null,
        representation: null,
        extractor: null,
        model_id: null,
        algorithm: null,
        source_kind: 'frozen_split_membership',
        full_clustering_artifact_available: false,
        membership_consistency_verified: false,
        source_split_ids: ['synthetic-split'],
        n_clusters: 1,
        n_noise: 0,
        noise_fraction: 0,
        parameters: null,
        effective_parameters: null,
      },
    ],
    clusters: [
      {
        cluster_run_id: configuration.cluster_run_id,
        cluster_id: -1,
        size_unique_contents: 1,
        is_noise: true,
        source_video_distribution: {},
        temporal_spans: [],
        automatic_confirmation: false,
        ground_truth: false,
      },
      {
        cluster_run_id: configuration.cluster_run_id,
        cluster_id: 42,
        size_unique_contents: 1,
        is_noise: false,
        source_video_distribution: {},
        temporal_spans: [],
        automatic_confirmation: false,
        ground_truth: false,
      },
      {
        cluster_run_id: configuration.cluster_run_id,
        cluster_id: 7,
        size_unique_contents: count,
        is_noise: false,
        source_video_distribution: {},
        temporal_spans: [],
        automatic_confirmation: false,
        ground_truth: false,
      },
      {
        cluster_run_id: 'synthetic-run-frozen',
        cluster_id: 19,
        size_unique_contents: 1,
        is_noise: false,
        source_video_distribution: {},
        temporal_spans: [],
        automatic_confirmation: false,
        ground_truth: false,
      },
    ],
    cluster_memberships: Array.from({ length: count + 2 }, (_, index) => ({
      cluster_run_id: configuration.cluster_run_id,
      cluster_id: index < count ? 7 : index === count ? 42 : -1,
      content_id: `synthetic-content-${index}`,
      is_noise: index === count + 1,
      ...(index === 0
        ? {
            core_distance: null,
            core_distance_infinite: true,
            reachability: 0,
            ordering_position: 0,
            probability: 0,
          }
        : {}),
    })),
    contents: Array.from({ length: count + 2 }, (_, index) => ({
      content_id: `synthetic-content-${index}`,
      canonical_filename: `SAMPLE frame ${index}.jpg`,
      representative_record_id: `synthetic-record-${index}`,
      source_video_id: 'synthetic-source',
      frame_index: index,
      source_frame_index: index * 10,
      source_frame_index_estimate: null,
      timestamp_seconds: index === 0 ? 0 : null,
      temporal_source: 'sampled_video_grid',
      record_ids:
        index === 0
          ? ['synthetic-record-0', 'synthetic-occurrence-copy']
          : [`synthetic-record-${index}`],
      class_ids: index === 0 ? [0] : null,
      class_names: index === 0 ? ['synthetic-class'] : null,
      annotation_consensus:
        index === 0 ? 'different_label_bytes' : 'unavailable',
      preview_key: `synthetic-preview-${index}`,
      width: 640,
      height: 480,
    })),
    records: [],
    media: [],
  }
  data.cluster_memberships.push({
    cluster_run_id: 'synthetic-run-frozen',
    cluster_id: 19,
    content_id: 'synthetic-content-0',
    is_noise: false,
  })
  data.media = data.contents.map((content, index) => ({
    preview_key: content.preview_key,
    content_id: content.content_id,
    relative_path: index === 1 ? null : `synthetic/SAMPLE frame ${index}.jpg`,
    width: content.width,
    height: content.height,
    checksum: null,
    media_available: index !== 1,
    unavailable_reason: index === 1 ? 'source_unavailable' : null,
  }))
  data.records = data.contents.flatMap((content) =>
    content.record_ids.map((recordId) => ({
      record_id: recordId,
      content_id: content.content_id,
      cohort: 'video_evidence',
      filename: content.canonical_filename,
      source_archive: 'synthetic-source.zip',
      source_video_id: content.source_video_id,
      timeline_id: 'synthetic-timeline',
      frame_index: content.frame_index,
      source_frame_index_estimate: null,
      timestamp_seconds: content.timestamp_seconds,
      temporal_source: content.temporal_source,
      original_split: null,
      class_ids: content.class_ids,
      class_names: content.class_names,
      num_objects: null,
      label_empty: null,
      width: content.width,
      height: content.height,
    })),
  )
  return data
}

export function clusteringFixtureResponse(
  data: ReturnType<typeof organizationClusteringFixture>,
  input: RequestInfo | URL,
) {
  const resource = String(input).split('/').at(-1)?.replace('.json', '')
  if (resource && Object.hasOwn(data, resource)) {
    return new Response(JSON.stringify(data[resource as keyof typeof data]), {
      headers: { 'Content-Type': 'application/json' },
    })
  }
  return new Response('', { status: 404 })
}
