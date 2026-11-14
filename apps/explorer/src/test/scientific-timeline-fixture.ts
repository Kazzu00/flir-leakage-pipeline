import type {
  ScientificTimelineData,
  VideoAdmission,
} from '@/data/scientific-timeline'

/** Synthetic identities only; never a stand-in for the scientific publication. */
export function scientificTimelineFixture(): ScientificTimelineData {
  const definitions = [
    ['f-late', 'c-late', 30],
    ['f-first', 'c-first', 0],
    ['f-copy', 'c-first', 0],
    ['f-middle', 'c-middle', 12],
    ['f-noise', 'c-noise', 20],
    ['f-recurrent', 'c-recurrent', 42],
    ['f-no-cluster', 'c-unassigned', 40],
    ['f-unknown', 'c-unknown', null],
  ] as const
  const members = definitions.map(([id], i) => ({
    frame_id: id,
    new_split: (i === 1
      ? 'test'
      : i === 2
        ? 'quarantine'
        : i === 3 || i === 6
          ? 'val'
          : 'train') as 'test' | 'quarantine' | 'val' | 'train',
  }))
  return {
    schema_version: 'scientific-timeline-v1',
    origin: 'synthetic_fixture',
    source_receipt_sha256: 'a'.repeat(64),
    organization_manifest_sha256: 'b'.repeat(64),
    source_commit: 'synthetic-only',
    source_dirty: false,
    source_checksums: { 'synthetic.json': 'c'.repeat(64) },
    test_status: 'PROPOSED',
    ready_for_training: false,
    limitations: ['Synthetic infrastructure fixture, no scientific evidence.'],
    axes: [
      {
        axis_id: 'synthetic-axis',
        source_video_id: null,
        label: 'synthetic-family',
        association: 'filename_inference',
        source_reference: 'synthetic/timelines.json',
      },
    ],
    records: definitions.map(([frame_id, content_id, position]) => ({
      frame_id,
      content_id,
      filename: `${frame_id}.jpg`,
      position,
      axis_id: position === null ? null : 'synthetic-axis',
      temporal_provenance: 'filename_heuristic_not_ground_truth',
      order_basis: 'filename_inference',
      source_reference: 'synthetic/occurrence_memberships.parquet',
      sequence_id: null,
      reason: position === null ? 'unknown_video' : null,
      preview_path: null,
    })),
    runs: [
      ['historical', 'S1', 0],
      ['historical', 'S1', 1],
      ['historical', 'S2', 0],
      ['expanded', 'S3', 4],
    ].map(([population, policy, seed]) => ({
      run_id: `${population}-${policy}-${seed}`,
      population: population as 'historical' | 'expanded',
      policy: policy as 'S1' | 'S2' | 'S3',
      seed: Number(seed),
      candidate_id: `candidate-${population}-${policy}-${seed}`,
      membership_id: `membership-${population}-${policy}-${seed}`,
      split_space_id: null,
      test_id: 'synthetic-test',
      state: 'PROVISIONAL_VERIFIED',
      source_reference: `synthetic/${population}-${policy}-${seed}.parquet`,
      members: members.map((m) => ({ ...m })),
    })),
    intervals: [
      {
        interval_id: 'candidate-synthetic',
        axis_id: 'synthetic-axis',
        sequence_id: null,
        kind: 'candidate_span',
        review: 'not_reviewed',
        start: 0,
        end: 42,
        source_reference: 'synthetic/timelines.json',
        provenance: 'Observed namespace extent, not a confirmed sequence.',
      },
    ],
    clustering_runs: ['synthetic-cluster-a', 'synthetic-cluster-b'].map(
      (cluster_run_id) => ({
        cluster_run_id,
        source_kind: 'frozen_split_membership',
        full_clustering_artifact_available: false,
        membership_consistency_verified: true,
        source_membership_checksums: { 'synthetic-split': 'd'.repeat(64) },
      }),
    ),
    cluster_memberships: ['synthetic-cluster-a', 'synthetic-cluster-b'].flatMap(
      (cluster_run_id, i) => [
        { cluster_run_id, content_id: 'c-first', cluster_id: i ? 3 : 0 },
        { cluster_run_id, content_id: 'c-middle', cluster_id: i ? 8 : 42 },
        { cluster_run_id, content_id: 'c-noise', cluster_id: -1 },
        { cluster_run_id, content_id: 'c-recurrent', cluster_id: i ? 9 : 7 },
        { cluster_run_id, content_id: 'c-late', cluster_id: i ? 9 : 7 },
      ],
    ),
  }
}

/** Positive exact-video contract fixture, explicitly synthetic at every boundary. */
export function verifiedVideoFixture() {
  const data = scientificTimelineFixture()
  data.axes = ['synthetic-axis', 'synthetic-empty-2', 'synthetic-empty-3'].map(
    (id) => ({
      axis_id: id,
      source_video_id: id,
      label: id,
      association: 'reported_video' as const,
      source_reference: 'synthetic/videos.json',
    }),
  )
  for (const r of data.records) {
    r.order_basis = 'reported_index'
    r.temporal_provenance =
      r.reason === null ? 'VIDEO_VERIFIED' : 'NOT_VERIFIED_IN_EXISTING_SCOPE'
  }
  const admission: VideoAdmission = {
    schema_version: 'verified-video-admission-v1',
    scope: 'exact_unique_three_authorized_videos',
    delivery_run_id: data.runs[0].run_id,
    membership_id: data.runs[0].membership_id,
    package_id: 'e'.repeat(64),
    delivery_b_status: 'BLOCKED_NO_VERIFIED_SEQUENCE_UNITS',
    absence_from_videos_claimed: false,
    authorized_videos: data.axes.map((a) => ({
      source_video_id: a.axis_id,
      sha256: 'f'.repeat(64),
    })),
    accepted_records: data.records
      .filter((r) => r.reason === null)
      .map((r) => ({
        frame_id: r.frame_id,
        content_id: r.content_id,
        source_video_id: r.axis_id!,
        appearance_order: r.position!,
        source_video_sha256: 'f'.repeat(64),
        evidence_reference: r.source_reference,
      })),
  }
  return { data, admission }
}
