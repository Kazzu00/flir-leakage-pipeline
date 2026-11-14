import { z } from 'zod'
import { useQuery } from '@tanstack/react-query'

const Id = z.string().min(1)
const Index = z.number().int().nonnegative()
const Digest = z.string().regex(/^[a-f0-9]{64}$/)
const Partition = z.enum(['train', 'val', 'test', 'quarantine', 'unassigned'])
const Reason = z.enum([
  'unknown_video',
  'ambiguous_association',
  'unknown_order',
  'index_conflict',
])
const RelativePath = Id.refine(
  (s) =>
    !s.startsWith('/') &&
    !s.includes('\\') &&
    !s.includes(':') &&
    s.split('/').every((v) => v !== '' && v !== '.' && v !== '..'),
)

export const ScientificTimelineSchema = z
  .object({
    schema_version: z.literal('scientific-timeline-v1'),
    origin: z.enum(['existing_evidence_export', 'synthetic_fixture']),
    source_receipt_sha256: Digest,
    organization_manifest_sha256: Digest,
    source_commit: Id,
    source_dirty: z.boolean(),
    test_status: z.literal('PROPOSED'),
    ready_for_training: z.literal(false),
    source_checksums: z.record(Id, Digest),
    limitations: z.array(z.string()),
    axes: z.array(
      z
        .object({
          axis_id: Id,
          source_video_id: Id.nullable(),
          label: Id,
          association: z.enum(['reported_video', 'filename_inference']),
          source_reference: Id,
        })
        .strict(),
    ),
    records: z.array(
      z
        .object({
          frame_id: Id,
          content_id: Id,
          filename: Id,
          axis_id: Id.nullable(),
          position: Index.nullable(),
          temporal_provenance: Id,
          order_basis: z.enum([
            'filename_inference',
            'sampled_grid',
            'reported_index',
          ]),
          source_reference: Id,
          sequence_id: Id.nullable(),
          reason: Reason.nullable(),
          preview_path: RelativePath.nullable(),
        })
        .strict(),
    ),
    runs: z
      .array(
        z
          .object({
            run_id: Id,
            population: z.enum(['historical', 'expanded']),
            policy: z.enum(['S1', 'S2', 'S3']),
            seed: Index.max(4),
            candidate_id: Id,
            membership_id: Id,
            split_space_id: Id.nullable(),
            test_id: Id,
            state: z.literal('PROVISIONAL_VERIFIED'),
            source_reference: Id,
            members: z.array(
              z.object({ frame_id: Id, new_split: Partition }).strict(),
            ),
          })
          .strict(),
      )
      .min(1),
    intervals: z.array(
      z
        .object({
          interval_id: Id,
          axis_id: Id,
          sequence_id: Id.nullable(),
          kind: z.enum([
            'candidate_span',
            'reviewed_sequence',
            'verified_sequence',
          ]),
          review: Id,
          start: Index,
          end: Index,
          source_reference: Id,
          provenance: Id,
        })
        .strict(),
    ),
    clustering_runs: z.array(
      z
        .object({
          cluster_run_id: Id,
          source_kind: z.enum([
            'full_clustering_artifact',
            'frozen_split_membership',
          ]),
          full_clustering_artifact_available: z.boolean(),
          membership_consistency_verified: z.literal(true),
          source_membership_checksums: z.record(Id, Digest),
        })
        .strict(),
    ),
    cluster_memberships: z.array(
      z
        .object({
          cluster_run_id: Id,
          content_id: Id,
          cluster_id: z.number().int().min(-1),
        })
        .strict(),
    ),
  })
  .strict()
  .superRefine((data, ctx) => {
    const fail = (message: string) => ctx.addIssue({ code: 'custom', message })
    const distinct = <T>(rows: T[], key: (r: T) => string) => {
      if (new Set(rows.map(key)).size !== rows.length)
        fail('Duplicate identity')
    }
    distinct(data.records, (r) => r.frame_id)
    distinct(data.axes, (r) => r.axis_id)
    distinct(data.runs, (r) => r.run_id)
    distinct(data.runs, (r) => JSON.stringify([r.population, r.policy, r.seed]))
    distinct(data.clustering_runs, (r) => r.cluster_run_id)
    distinct(data.cluster_memberships, (r) =>
      JSON.stringify([r.cluster_run_id, r.content_id]),
    )
    distinct(data.intervals, (r) => r.interval_id)
    const axes = new Map(data.axes.map((r) => [r.axis_id, r]))
    const records = new Map(data.records.map((r) => [r.frame_id, r]))
    const configs = new Map(
      data.clustering_runs.map((r) => [r.cluster_run_id, r]),
    )
    const positions = new Map<string, string>()
    for (const r of data.records) {
      if (r.reason !== null) continue
      if (r.position === null || !r.axis_id || !axes.has(r.axis_id))
        fail('Missing temporal binding')
      const key = JSON.stringify([r.axis_id, r.position])
      if (positions.has(key) && positions.get(key) !== r.content_id)
        fail('Unreported index conflict')
      positions.set(key, r.content_id)
    }
    let commonTest: string | undefined
    for (const run of data.runs) {
      distinct(run.members, (m) => m.frame_id)
      if (run.run_id !== `${run.population}-${run.policy}-${run.seed}`)
        fail('Run selector mismatch')
      if (run.members.some((m) => !records.has(m.frame_id)))
        fail('Unknown occurrence')
      const test = JSON.stringify(
        run.members
          .filter((m) => m.new_split === 'test')
          .map((m) => m.frame_id)
          .sort(),
      )
      if (commonTest !== undefined && commonTest !== test)
        fail('Protected test differs')
      commonTest = test
    }
    for (const m of data.cluster_memberships) {
      if (!configs.has(m.cluster_run_id)) fail('Unknown clustering run')
    }
    for (const c of data.clustering_runs) {
      if (
        c.source_kind === 'frozen_split_membership' &&
        !Object.keys(c.source_membership_checksums).length
      )
        fail('Missing frozen membership provenance')
    }
    for (const i of data.intervals) {
      if (!axes.has(i.axis_id) || i.start > i.end)
        fail('Invalid temporal interval')
      if (
        i.kind === 'candidate_span'
          ? i.sequence_id !== null
          : i.sequence_id === null
      )
        fail('Sequence and candidate identities confused')
    }
  })

export const VideoAdmissionSchema = z
  .object({
    schema_version: z.literal('verified-video-admission-v1'),
    scope: z.literal('exact_unique_three_authorized_videos'),
    delivery_run_id: Id,
    membership_id: Id,
    package_id: Digest,
    delivery_b_status: z.literal('BLOCKED_NO_VERIFIED_SEQUENCE_UNITS'),
    absence_from_videos_claimed: z.literal(false),
    authorized_videos: z
      .array(z.object({ source_video_id: Id, sha256: Digest }).strict())
      .length(3),
    accepted_records: z.array(
      z
        .object({
          frame_id: Id,
          content_id: Id,
          source_video_id: Id,
          appearance_order: Index,
          source_video_sha256: Digest,
          evidence_reference: Id,
        })
        .strict(),
    ),
  })
  .strict()

export type VideoAdmission = z.infer<typeof VideoAdmissionSchema>
export type ScientificTimelineData = z.infer<
  typeof ScientificTimelineSchema
> & { admission?: VideoAdmission }

/** A schema-valid inferred timeline cannot enter the verified-video mount. */
export function admitVideoTimeline(
  data: ScientificTimelineData,
  admission: VideoAdmission,
): ScientificTimelineData {
  const videos = new Map(
    admission.authorized_videos.map((v) => [v.source_video_id, v.sha256]),
  )
  const run = data.runs.find((r) => r.run_id === admission.delivery_run_id)
  if (
    videos.size !== 3 ||
    !run ||
    run.membership_id !== admission.membership_id
  )
    throw new Error('Video admission identity mismatch')
  if (
    data.axes.length !== 3 ||
    data.axes.some(
      (a) =>
        a.association !== 'reported_video' ||
        a.axis_id !== a.source_video_id ||
        !videos.has(a.axis_id),
    )
  )
    throw new Error('Unauthorized or inferred video axis')
  const accepted = new Map(
    admission.accepted_records.map((r) => [r.frame_id, r]),
  )
  const represented = data.records.filter((r) => r.reason === null)
  if (
    accepted.size !== admission.accepted_records.length ||
    accepted.size !== represented.length
  )
    throw new Error('Video admission coverage mismatch')
  const contents = new Map<string, string>()
  for (const r of represented) {
    const m = accepted.get(r.frame_id)
    if (
      !m ||
      r.content_id !== m.content_id ||
      r.axis_id !== m.source_video_id ||
      r.position !== m.appearance_order ||
      r.order_basis !== 'reported_index' ||
      r.temporal_provenance !== 'VIDEO_VERIFIED' ||
      r.source_reference !== m.evidence_reference ||
      videos.get(m.source_video_id) !== m.source_video_sha256
    )
      throw new Error('Unverified or inconsistent video correspondence')
    const position = JSON.stringify([m.source_video_id, m.appearance_order])
    if (contents.has(m.content_id) && contents.get(m.content_id) !== position)
      throw new Error('Ambiguous repeated appearance')
    contents.set(m.content_id, position)
  }
  return { ...data, admission }
}
export type TimelineRecord = ScientificTimelineData['records'][number]
export type TimelineRun = ScientificTimelineData['runs'][number]
export type TimelineInterval = ScientificTimelineData['intervals'][number]
export type TimelineFrame = TimelineRecord & {
  new_split: z.infer<typeof Partition>
  cluster_id: number | null
}

/** One row per occurrence, including repeated content. Reference clustering is independent of the split. */
export function framesForRun(
  data: ScientificTimelineData,
  run: TimelineRun,
  clusterRunId: string,
): TimelineFrame[] {
  const records = new Map(data.records.map((r) => [r.frame_id, r]))
  const clusters = new Map(
    data.cluster_memberships
      .filter((m) => m.cluster_run_id === clusterRunId)
      .map((m) => [m.content_id, m.cluster_id]),
  )
  return run.members.map((m) => ({
    ...records.get(m.frame_id)!,
    new_split: m.new_split,
    cluster_id: clusters.get(records.get(m.frame_id)!.content_id) ?? null,
  }))
}

export function temporalOrder(a: TimelineFrame, b: TimelineFrame) {
  return (
    (a.position ?? Infinity) - (b.position ?? Infinity) ||
    a.frame_id.localeCompare(b.frame_id)
  )
}

export const partitionLabels = {
  train: 'Train',
  val: 'Validation',
  test: 'Test',
  quarantine: 'Quarantine',
  unassigned: 'Sin asignación',
}
export const reasonLabels = {
  unknown_video: 'Vídeo desconocido',
  ambiguous_association: 'Asociación ambigua',
  unknown_order: 'Orden desconocido',
  index_conflict: 'Conflicto de índices',
}
export const clusterLabel = (id: number | null) =>
  id === null
    ? 'Clúster no disponible'
    : id === -1
      ? 'Ruido (−1)'
      : `Clúster ${id}`

export function clusterColor(run: string, id: number | null) {
  if (id === null) return '#758195'
  if (id === -1) return '#8d647e'
  let hash = 2166136261
  for (const ch of `${run}:${id}`)
    hash = Math.imul(hash ^ ch.charCodeAt(0), 16777619)
  return `hsl(${(hash >>> 0) % 360} 64% var(--timeline-cluster-lightness, 32%))`
}

export function summarizeFrames(frames: TimelineFrame[]) {
  return Object.keys(partitionLabels).map((partition) => {
    const members = frames.filter((r) => r.new_split === partition)
    const counts = new Map<number | null, number>()
    for (const r of members)
      counts.set(r.cluster_id, (counts.get(r.cluster_id) ?? 0) + 1)
    return {
      partition,
      count: members.length,
      clusters:
        [...counts]
          .map(([id, count]) => `${clusterLabel(id)}: ${count}`)
          .join(' · ') || '—',
    }
  })
}

/** CSV keeps one row per selected-run occurrence. Quoting also neutralizes spreadsheet formulas. */
export function timelineCsv(
  frames: TimelineFrame[],
  run: TimelineRun,
  clusterRunId: string,
) {
  const fields = [
    'run_id',
    'population',
    'policy',
    'seed',
    'frame_id',
    'content_id',
    'filename',
    'axis_id',
    'position',
    'temporal_provenance',
    'source_reference',
    'assignment_source_reference',
    'sequence_id',
    'cluster_run_id',
    'cluster_id',
    'split_space_id',
    'membership_id',
    'new_split',
    'reason',
  ] as const
  const cell = (v: unknown) => {
    const raw = v === null || v === undefined ? '' : String(v)
    const safe = typeof v === 'string' && /^[=+@-]/.test(raw) ? `'${raw}` : raw
    return `"${safe.replaceAll('"', '""')}"`
  }
  return (
    '\uFEFF' +
    [
      fields.join(','),
      ...frames.map((f) => {
        const row = {
          ...f,
          ...run,
          source_reference: f.source_reference,
          assignment_source_reference: run.source_reference,
          cluster_run_id: clusterRunId,
          cluster_id: f.cluster_id,
        }
        return fields.map((key) => cell(row[key])).join(',')
      }),
    ].join('\r\n')
  )
}

const Receipt = z
  .object({
    schema_version: z.literal('scientific-timeline-export-receipt-v1'),
    files: z.record(Id, Digest),
    source_checksums: z.record(Id, Digest),
  })
  .strict()

export async function loadScientificTimeline(
  signal?: AbortSignal,
): Promise<ScientificTimelineData> {
  // Fixed same-origin runtime mount, independent of DEMO selection and excluded from builds.
  const base = '/runtime/verified-video-timeline-v1'
  const [receiptResponse, response, admissionResponse] = await Promise.all([
    fetch(`${base}/receipt.json`, {
      signal,
      cache: 'no-store',
      credentials: 'same-origin',
    }),
    fetch(`${base}/index.json`, {
      signal,
      cache: 'no-store',
      credentials: 'same-origin',
    }),
    fetch(`${base}/admission.json`, {
      signal,
      cache: 'no-store',
      credentials: 'same-origin',
    }),
  ])
  if (!response.ok || !receiptResponse.ok || !admissionResponse.ok)
    throw new Error('Timeline publication unavailable')
  const receipt = Receipt.parse(await receiptResponse.json())
  const raw = await response.arrayBuffer()
  const admissionRaw = await admissionResponse.arrayBuffer()
  const sha256 = async (bytes: ArrayBuffer) =>
    [...new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))]
      .map((b) => b.toString(16).padStart(2, '0'))
      .join('')
  if (
    (await sha256(raw)) !== receipt.files['index.json'] ||
    (await sha256(admissionRaw)) !== receipt.files['admission.json']
  )
    throw new Error('Timeline checksum mismatch')
  const data = ScientificTimelineSchema.parse(
    JSON.parse(new TextDecoder().decode(raw)),
  )
  const entries = Object.entries(data.source_checksums)
  if (
    entries.length !== Object.keys(receipt.source_checksums).length ||
    entries.some(([k, v]) => receipt.source_checksums[k] !== v)
  )
    throw new Error('Timeline provenance mismatch')
  return admitVideoTimeline(
    data,
    VideoAdmissionSchema.parse(
      JSON.parse(new TextDecoder().decode(admissionRaw)),
    ),
  )
}

export function useScientificTimeline() {
  return useQuery({
    queryKey: ['verified-video-timeline-v1'],
    queryFn: ({ signal }) => loadScientificTimeline(signal),
    staleTime: Infinity,
    retry: false,
  })
}
