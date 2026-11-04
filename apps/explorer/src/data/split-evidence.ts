import { z } from 'zod'
import { useQuery } from '@tanstack/react-query'
import { OrganizationEvidenceAdapter } from './adapters/OrganizationEvidenceAdapter'
import { JsonValueSchema } from '@/contracts/organization-evidence-v2'

const Member = z
  .object({
    record_id: z.string(),
    partition: z.enum(['train', 'val', 'test', 'quarantine']),
    group_id: z.string().nullable(),
  })
  .strict()
const RecordSchema = z
  .object({
    record_id: z.string(),
    content_id: z.string(),
    filename: z.string(),
    source: z.string(),
    cohort: z.string(),
    class_ids: z.array(z.number().int().nonnegative()),
    label_sha256: z.string(),
  })
  .strict()
const Support = z.object({
  class_id: z.number(),
  instances: z.number(),
  images: z.number(),
  contents: z.number(),
})
export const CandidateSplitsSchema = z
  .object({
    schema_version: z.literal('candidate-splits-view-v1'),
    test_status: z.literal('PROPOSED'),
    ready_for_training: z.literal(false),
    dataset_id: z.string(),
    source_receipt_sha256: z.string().regex(/^[a-f0-9]{64}$/),
    consumed_sha256: z.record(z.string(), z.string().regex(/^[a-f0-9]{64}$/)),
    source_commit: z.string(),
    source_dirty: z.boolean(),
    plan_id: z.string(),
    blockers: z.array(z.string()),
    verification_scope: z.string(),
    records: z.array(RecordSchema),
    runs: z.array(
      z
        .object({
          id: z.string(),
          membership_id: z.string(),
          policy: z.enum(['S1', 'S2', 'S3']),
          population: z.enum(['historical', 'expanded']),
          seed: z.number().int().nonnegative(),
          test_id: z.string(),
          state: z.literal('PROVISIONAL_VERIFIED'),
          ready_for_training: z.literal(false),
          members: z.array(Member),
          metrics: z
            .object({
              state: z.literal('PROVISIONAL_VERIFIED'),
              ready_for_training: z.literal(false),
              class_support: z.record(z.string(), z.array(Support)),
              blockers: z.array(z.string()),
            })
            .catchall(JsonValueSchema),
        })
        .strict(),
    ),
    relations: z.array(
      z
        .object({
          relation_id: z.string(),
          left_frame_id: z.string(),
          right_frame_id: z.string(),
          state: z.enum([
            'exact_bytes',
            'visual_candidate',
            'confirmed_provenance',
            'unresolved',
            'rejected',
          ]),
        })
        .strict(),
    ),
  })
  .strict()
  .superRefine((data, ctx) => {
    const records = new Map(data.records.map((r) => [r.record_id, r]))
    const fail = (message: string) => ctx.addIssue({ code: 'custom', message })
    if (
      records.size !== data.records.length ||
      new Set(data.runs.map((r) => r.id)).size !== data.runs.length
    )
      fail('Duplicate identities')
    let testIdentity: string | undefined
    for (const run of data.runs) {
      const contentPartitions = new Map<string, string>()
      const groupPartitions = new Map<string, string>()
      for (const member of run.members) {
        if (member.partition === 'quarantine') continue
        const content = records.get(member.record_id)?.content_id
        if (content) {
          const previous = contentPartitions.get(content)
          if (previous && previous !== member.partition)
            fail('Exact content crosses active partitions')
          contentPartitions.set(content, member.partition)
        }
        if (member.partition === 'train' || member.partition === 'val') {
          if (!member.group_id) fail('Missing development group')
          else {
            const previous = groupPartitions.get(member.group_id)
            if (previous && previous !== member.partition)
              fail('Development group fractured')
            groupPartitions.set(member.group_id, member.partition)
          }
        }
      }
      if (
        new Set(run.members.map((m) => m.record_id)).size !==
          run.members.length ||
        run.members.some((m) => !records.has(m.record_id))
      )
        fail('Invalid member references')
      const test = JSON.stringify(
        run.members
          .filter((m) => m.partition === 'test')
          .map((m) => {
            const r = records.get(m.record_id)
            return [m.record_id, r?.content_id, r?.label_sha256]
          })
          .sort((a, b) => String(a[0]).localeCompare(String(b[0]))),
      )
      if (!run.members.some((m) => m.partition === 'test'))
        fail('Missing proposed test')
      if (testIdentity !== undefined && test !== testIdentity)
        fail('Common test differs')
      testIdentity = test
    }
    if (
      data.relations.some(
        (r) => !records.has(r.left_frame_id) || !records.has(r.right_frame_id),
      )
    )
      fail('Unknown relation endpoints')
  })

export type SplitMember = {
  recordId: string
  contentId: string
  partition: string
  group: string | null
  filename: string
  source: string
  classes: number[] | null
  preview: string | null
}
export type SplitRun = {
  id: string
  family: string
  policy: string
  population: string
  seed: number
  testStatus: string
  members: SplitMember[]
  support: {
    partition: string
    class_id: number
    images: number
    contents: number
    instances: number
  }[]
  metadata: unknown
}
export type SplitEdge = {
  id: string
  left: string
  right: string
  state: string
}
const adapter = new OrganizationEvidenceAdapter()

export async function loadHistoricalSplits(signal?: AbortSignal) {
  const [
    manifest,
    splits,
    memberships,
    records,
    contents,
    media,
    clusters,
    links,
  ] = await Promise.all([
    adapter.getResource('manifest', signal),
    adapter.getResource('splits', signal),
    adapter.getResource('split_memberships', signal),
    adapter.getResource('records', signal),
    adapter.getResource('contents', signal),
    adapter.getResource('media', signal),
    adapter.getResource('cluster_memberships', signal),
    adapter.getResource('linkage_memberships', signal),
  ])
  const byRecord = new Map(records.map((r) => [r.record_id, r]))
  const byContent = new Map(contents.map((c) => [c.content_id, c]))
  const byPreview = new Map(media.map((m) => [m.preview_key, m]))
  const byCluster = new Map(
    clusters.map((c) => [`${c.cluster_run_id}:${c.content_id}`, c]),
  )
  const previews = new Map(
    contents.map((c) => {
      const m = byPreview.get(c.preview_key)
      return [c.content_id, m?.media_available ? m.relative_path : null]
    }),
  )
  const runs: SplitRun[] = splits.map((split) => ({
    id: split.split_space_id,
    family: 'historical_yolo11n',
    policy: split.strategy,
    population: 'historical',
    seed: split.split_seed,
    testStatus: 'HISTORICAL / varies by split; not the proposed common test',
    members: memberships
      .filter((m) => m.split_space_id === split.split_space_id)
      .map((m) => {
        const r = byRecord.get(m.record_id)
        if (!r || r.content_id !== m.content_id || !byContent.has(m.content_id))
          throw new Error('Invalid historical member binding')
        const cluster = byCluster.get(`${split.cluster_run_id}:${m.content_id}`)
        if (split.cluster_run_id && !cluster)
          throw new Error('Missing frozen cluster membership')
        return {
          recordId: m.record_id,
          contentId: m.content_id,
          partition: m.partition,
          group:
            cluster && !cluster.is_noise
              ? `cluster:${split.cluster_run_id}:${cluster.cluster_id}`
              : `content:${m.content_id}`,
          filename: r.filename ?? m.record_id,
          source: r.source_archive ?? r.source_video_id ?? 'Unavailable',
          classes: r.class_ids,
          preview: previews.get(m.content_id) ?? null,
        }
      }),
    support: (split.class_support ?? []).map((s) => ({
      partition: s.split,
      class_id: s.class_id,
      images: s.images,
      contents: s.contents,
      instances: s.instances,
    })),
    metadata: {
      split,
      provenance: {
        source_commit: manifest.source_commit,
        plan_id: manifest.plan_id,
        verification_scope: manifest.verification_scope,
      },
      limitations: manifest.limitations,
    },
  }))
  if (
    runs.length !== manifest.split_count ||
    runs.some((r) => r.members.length !== manifest.labeled_record_count)
  )
    throw new Error('Incomplete historical splits')
  return { runs, previews, links, manifest }
}

export async function loadCandidateSplits(signal?: AbortSignal) {
  const response = await fetch('/runtime/candidate-splits-view-v1.json', {
    signal,
    cache: 'no-store',
    credentials: 'same-origin',
  })
  if (!response.ok) throw new Error('Candidate presentation export UNAVAILABLE')
  return CandidateSplitsSchema.parse(await response.json())
}

export function candidateRuns(
  data: z.infer<typeof CandidateSplitsSchema>,
  previews: Map<string, string | null>,
): SplitRun[] {
  const records = new Map(data.records.map((r) => [r.record_id, r]))
  return data.runs.map((run) => ({
    id: run.id,
    family: 'common_test_candidates',
    policy: run.policy,
    population: run.population,
    seed: run.seed,
    testStatus: data.test_status,
    members: run.members.map((m) => {
      const r = records.get(m.record_id)!
      return {
        recordId: r.record_id,
        contentId: r.content_id,
        partition: m.partition,
        group: m.group_id,
        filename: r.filename,
        source: r.source,
        classes: r.class_ids,
        preview: previews.get(r.content_id) ?? null,
      }
    }),
    support: Object.entries(run.metrics.class_support).flatMap(
      ([partition, rows]) => rows.map((row) => ({ partition, ...row })),
    ),
    metadata: {
      candidate_id: run.id,
      membership_id: run.membership_id,
      test_id: run.test_id,
      state: run.state,
      ready_for_training: false,
      metrics: run.metrics,
      source_commit: data.source_commit,
      source_dirty: data.source_dirty,
      source_receipt_sha256: data.source_receipt_sha256,
      verification_scope: data.verification_scope,
      blockers: data.blockers,
    },
  }))
}

export function useSplitEvidence() {
  const historical = useQuery({
    queryKey: ['historical-split-evidence'],
    queryFn: ({ signal }) => loadHistoricalSplits(signal),
    staleTime: Infinity,
  })
  const candidates = useQuery({
    queryKey: ['candidate-split-evidence'],
    queryFn: ({ signal }) => loadCandidateSplits(signal),
    staleTime: Infinity,
  })
  return { historical, candidates }
}
