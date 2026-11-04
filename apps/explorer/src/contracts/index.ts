import { z } from 'zod'
import { LeakageSnapshotSchema } from './leakage-v2'

export const VerificationState = z.enum([
  'verified',
  'complete',
  'running',
  'experimental',
  'pending',
  'invalid',
  'unavailable',
  'mock',
  'inconsistent',
])
export const ModuleId = z.enum([
  'preprocessing',
  'organization',
  'segmentation',
])
const Identity = z.string().min(1)
export const Metric = z.object({
  name: Identity,
  value: z.number().finite().nullable(),
  unit: z.string(),
  origin: z.enum(['mock', 'artifact', 'reported', 'unavailable']),
  caveat: z.string(),
})
export const ArtifactReference = z.object({
  id: Identity,
  label: Identity,
  format: Identity,
  status: VerificationState,
  description: z.string(),
})
export const PipelineStage = z.object({
  id: Identity,
  name: Identity,
  status: VerificationState,
  output: Identity,
})
export const PipelineModule = z.object({
  id: ModuleId,
  number: Identity,
  name: Identity,
  subtitle: Identity,
  description: Identity,
  status: VerificationState,
  repository: z.url(),
  evidence: z.string(),
  stages: z.array(PipelineStage),
})
export const DatasetSummary = z.object({
  id: Identity,
  name: Identity,
  origin: z.enum(['mock', 'artifact', 'reported', 'unavailable']),
  occurrences: z.number().int().nonnegative(),
  uniqueContents: z.number().int().nonnegative(),
  duplicateGroups: z.number().int().nonnegative(),
  provenance: z.string(),
  verification: VerificationState,
})
export const RunSummary = z.object({
  id: Identity,
  module: ModuleId,
  stage: Identity,
  datasetId: Identity,
  method: Identity,
  status: VerificationState,
  verification: VerificationState,
  origin: z.enum(['mock', 'artifact', 'reported', 'unavailable']),
  seed: z.number().int().nullable(),
  parameters: z.record(
    z.string(),
    z.union([z.string(), z.number(), z.boolean()]),
  ),
  metrics: z.array(Metric),
  artifacts: z.array(ArtifactReference),
  caveat: z.string(),
})
export const EmbeddingRun = RunSummary.extend({
  encoder: z.enum(['DINOv2', 'CLIP']),
  dimensions: z.number().int().positive(),
  pooling: Identity,
  featureSpaceId: Identity,
  coverage: z.number().int().nonnegative(),
})
export const ReductionRun = RunSummary.extend({
  encoder: z.enum(['DINOv2', 'CLIP']),
  method: z.enum(['t-SNE', 'PaCMAP']),
  selectedCandidate: z.boolean(),
})
export const ContentPoint = z.object({
  contentId: Identity,
  frameIds: z.array(Identity).min(1),
  embeddingRow: z.number().int().nonnegative(),
  clusterId: z.number().int().min(-1),
  groupId: Identity,
  sourceVideo: Identity,
  sequenceId: Identity.nullable(),
  sequenceVerified: z.boolean(),
  sampleIndex: z.number().int().nonnegative(),
  x: z.number().finite(),
  y: z.number().finite(),
})
export const ClusterSummary = z.object({
  id: z.number().int().min(-1),
  size: z.number().int().nonnegative(),
  medoidContentId: Identity.nullable(),
  provenance: z.string(),
  metrics: z.array(Metric),
})
export const ClusteringRun = RunSummary.extend({
  encoder: z.enum(['DINOv2', 'CLIP']),
  reduction: z.enum(['t-SNE', 'PaCMAP', 'Original L2']),
  algorithm: z.enum(['DBSCAN', 'OPTICS', 'HDBSCAN']),
  points: z.array(ContentPoint),
  clusters: z.array(ClusterSummary),
})
export const SplitName = z.enum(['train', 'validation', 'test'])
export const SplitRun = RunSummary.extend({
  strategy: z.enum(['Historical', 'Random/content', 'Cluster-aware']),
  clusteringRunId: Identity.nullable(),
  assignments: z.array(
    z.object({ contentId: Identity, splits: z.array(SplitName).min(1) }),
  ),
  occurrences: z.array(
    z.object({ frameId: Identity, contentId: Identity, split: SplitName }),
  ),
  counts: z.object({
    train: z.number().int().nonnegative(),
    validation: z.number().int().nonnegative(),
    test: z.number().int().nonnegative(),
  }),
})
export const SimilarityRun = RunSummary.extend({
  pairs: z.array(
    z.object({
      a: Identity,
      b: Identity,
      cosine: z.number().min(-1).max(1),
      sameSourceVideo: z.boolean(),
      sampleGap: z.number().int().nonnegative().nullable(),
    }),
  ),
  distribution: z.array(
    z.object({ bin: z.string(), count: z.number().int().nonnegative() }),
  ),
})
export const PredictionResult = z.object({
  id: Identity,
  contentId: Identity,
  origin: z.literal('mock'),
  metrics: z.array(Metric),
  caveat: z.string(),
})
export const SegmentationRun = RunSummary.extend({
  modelVariant: z.enum(['base', 'context_fusion']),
  predictions: z.array(PredictionResult),
  classes: z.array(z.object({ name: Identity, metrics: z.array(Metric) })),
})
export const SnapshotSchema = z
  .object({
    schemaVersion: z.literal(1),
    modules: z.array(PipelineModule),
    dataset: DatasetSummary,
    runs: z.array(RunSummary),
    embeddings: z.array(EmbeddingRun),
    reductions: z.array(ReductionRun),
    clusterings: z.array(ClusteringRun),
    splits: z.array(SplitRun),
    similarity: SimilarityRun.nullable(),
    leakage: LeakageSnapshotSchema.optional(),
    segmentation: SegmentationRun,
    quality: z.array(
      z.object({
        index: z.number().int(),
        original: z.number(),
        n2n: z.number(),
        n2v: z.number(),
      }),
    ),
  })
  .superRefine((data, ctx) => {
    for (const run of data.clusterings) {
      const ids = run.points.map((p) => p.contentId)
      const noise = run.points.filter((p) => p.clusterId === -1)
      if (new Set(noise.map((p) => p.groupId)).size !== noise.length)
        ctx.addIssue({
          code: 'custom',
          message: 'Noise requires singleton allocation groups',
        })
      const clusterGroups = new Map<number, string>()
      for (const p of run.points.filter((p) => p.clusterId !== -1)) {
        if (
          clusterGroups.has(p.clusterId) &&
          clusterGroups.get(p.clusterId) !== p.groupId
        )
          ctx.addIssue({
            code: 'custom',
            message: 'A cluster must remain one atomic group',
          })
        clusterGroups.set(p.clusterId, p.groupId)
      }
      if (new Set(ids).size !== ids.length)
        ctx.addIssue({
          code: 'custom',
          message: 'Clustering must contain unique content rows',
        })
      for (const p of run.points)
        if (p.sequenceId && !p.sequenceVerified)
          ctx.addIssue({
            code: 'custom',
            message: 'Unverified sequence must remain null',
          })
      for (const cluster of run.clusters) {
        const members = run.points.filter((p) => p.clusterId === cluster.id)
        if (
          members.length !== cluster.size ||
          (cluster.medoidContentId &&
            !members.some((p) => p.contentId === cluster.medoidContentId))
        )
          ctx.addIssue({
            code: 'custom',
            message: 'Invalid cluster summary or medoid',
          })
      }
    }
    for (const split of data.splits) {
      const run = data.clusterings.find((r) => r.id === split.clusteringRunId)
      if (!run) {
        ctx.addIssue({
          code: 'custom',
          message: 'Split requires an existing clustering reference',
        })
        continue
      }
      const map = new Map(split.assignments.map((a) => [a.contentId, a.splits]))
      const occurrenceIds = new Set(split.occurrences.map((o) => o.frameId))
      const frames = run.points.flatMap((p) => p.frameIds)
      if (
        occurrenceIds.size !== frames.length ||
        split.occurrences.length !== frames.length ||
        frames.some((f) => !occurrenceIds.has(f))
      )
        ctx.addIssue({
          code: 'custom',
          message: 'Occurrence coverage mismatch',
        })
      for (const p of run.points) {
        const rows = split.occurrences.filter(
          (o) => o.contentId === p.contentId,
        )
        const memberships = new Set(rows.map((o) => o.split))
        if (
          rows.length !== p.frameIds.length ||
          rows.some((o) => !p.frameIds.includes(o.frameId)) ||
          memberships.size !== map.get(p.contentId)?.length ||
          map.get(p.contentId)?.some((s) => !memberships.has(s))
        )
          ctx.addIssue({
            code: 'custom',
            message: 'Occurrence and content assignments disagree',
          })
      }
      for (const name of ['train', 'validation', 'test'] as const)
        if (
          split.occurrences.filter((o) => o.split === name).length !==
          split.counts[name]
        )
          ctx.addIssue({
            code: 'custom',
            message: 'Split counts must match occurrences',
          })
      if (
        map.size !== run.points.length ||
        map.size !== split.assignments.length ||
        run.points.some((p) => !map.has(p.contentId))
      )
        ctx.addIssue({
          code: 'custom',
          message: 'Split content coverage mismatch',
        })
      if (
        split.strategy !== 'Historical' &&
        split.assignments.some((a) => a.splits.length !== 1)
      )
        ctx.addIssue({
          code: 'custom',
          message: 'New splits must assign each content once',
        })
      if (split.strategy === 'Cluster-aware') {
        const groups = new Map<string, string>()
        for (const p of run.points) {
          const s = map.get(p.contentId)?.join(',') ?? ''
          if (groups.has(p.groupId) && groups.get(p.groupId) !== s)
            ctx.addIssue({
              code: 'custom',
              message: 'Atomic group fractured across splits',
            })
          groups.set(p.groupId, s)
        }
      }
    }
  })
export type Snapshot = z.infer<typeof SnapshotSchema>
export type Run = z.infer<typeof RunSummary>
export type Point = z.infer<typeof ContentPoint>
export type ClusterRun = z.infer<typeof ClusteringRun>
export type Split = z.infer<typeof SplitRun>
export type State = z.infer<typeof VerificationState>
export type MetricValue = z.infer<typeof Metric>
