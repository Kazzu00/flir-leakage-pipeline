import { z } from 'zod'

// Runtime exports are an allowlisted interchange format, never upstream metadata dumps.
const Alias = z
  .string()
  .regex(
    /^(dataset|content|frame|video|feature|similarity|reduction|clustering|split|detector|configuration)-[a-z0-9-]{1,40}$/,
  )
const Stats = z.record(
  z.string().regex(/^[a-zA-Z][a-zA-Z0-9_@. -]{0,90}$/),
  z.number().finite().nullable(),
)
const Encoder = z.enum(['DINOv2', 'CLIP'])
const Stage = z.enum([
  'embeddings',
  'similarity',
  'reduction',
  'clustering',
  'splits',
  'detector',
])
const Origin = z.enum(['mock', 'artifact', 'reported', 'unavailable'])
export const LeakageRun = z
  .object({
    id: Alias,
    datasetId: Alias,
    stage: Stage,
    origin: z.literal('artifact'),
    encoder: Encoder.nullable(),
    method: z.enum([
      'DINOv2',
      'CLIP',
      'cosine',
      't-SNE',
      'PaCMAP',
      'DBSCAN',
      'OPTICS',
      'HDBSCAN',
      'historical',
      'random_content',
      'cluster_aware',
      'detector',
    ]),
    featureSpaceId: Alias.nullable(),
    similaritySpaceId: Alias.nullable(),
    reductionSpaceId: Alias.nullable(),
    clusteringSpaceId: Alias.nullable(),
    splitSpaceId: Alias.nullable(),
    configurationId: Alias.nullable(),
    seed: z.number().int().nullable(),
    status: z.enum(['complete', 'experimental']),
    integrity: z.literal('export-validated'),
    metricOrigin: Origin,
    metrics: Stats,
    parameters: Stats,
    dimensions: z.number().int().positive().nullable(),
    pooling: z.enum(['cls_token', 'projected_pooler_output']).nullable(),
    coverage: z.number().int().nonnegative(),
    candidate: z.boolean(),
    contentIndex: z
      .array(
        z
          .object({
            contentId: Alias,
            embeddingRow: z.number().int().nonnegative(),
          })
          .strict(),
      )
      .max(20000),
    coordinates: z
      .array(
        z.object({
          contentId: Alias,
          embeddingRow: z.number().int().nonnegative(),
          x: z.number().finite(),
          y: z.number().finite(),
        }),
      )
      .max(20000),
    pairs: z
      .array(
        z.object({
          a: Alias,
          b: Alias,
          rank: z.number().int().positive(),
          cosine: z.number().min(-1.00001).max(1.00001),
          sameSourceVideo: z.boolean(),
          sampleGap: z.number().nonnegative().nullable(),
          gridSecondsGap: z.number().nonnegative().nullable(),
        }),
      )
      .max(1000000),
    summaries: z
      .array(
        z.object({
          scope: z.enum([
            'source',
            'sample_gap',
            'grid_gap',
            'topk',
            'stability',
            'configuration',
          ]),
          index: z.number().int().nonnegative(),
          label: z
            .enum([
              'same_source_video',
              'disjoint_source_videos',
              'rank1_similarity',
              'top5_mean_similarity',
              'top10_mean_similarity',
              'top20_mean_similarity',
            ])
            .nullable(),
          configurationId: Alias.nullable(),
          metrics: Stats,
        }),
      )
      .max(20000),
    labels: z
      .array(
        z.object({
          contentId: Alias,
          clusterId: z.number().int().min(-1),
          medoid: z.boolean(),
        }),
      )
      .max(20000),
    assignments: z
      .array(
        z.object({
          frameId: Alias,
          contentId: Alias,
          split: z.enum(['train', 'validation', 'test']),
        }),
      )
      .max(100000),
  })
  .strict()

export const LeakageSnapshotV1 = z
  .object({
    schemaVersion: z.literal('LeakageSnapshotV1'),
    dataset: z
      .object({
        id: Alias,
        manifestVersion: z.literal('flir_video_samples_v1'),
        origin: z.literal('artifact'),
        occurrences: z.number().int().positive(),
        uniqueContents: z.number().int().positive(),
        duplicateGroups: z.number().int().nonnegative(),
      })
      .strict(),
    contents: z
      .array(
        z
          .object({
            contentId: Alias,
            occurrences: z
              .array(
                z
                  .object({
                    frameId: Alias,
                    sourceVideo: Alias,
                    sampleIndex: z.number().int().nonnegative(),
                    gridSeconds: z.number().nonnegative(),
                    sampleFps: z.number().positive(),
                    sequenceId: z.null(),
                    captureTimestamp: z.null(),
                  })
                  .strict(),
              )
              .min(1),
          })
          .strict(),
      )
      .min(1)
      .max(20000),
    runs: z.array(LeakageRun).max(256),
    exportPolicy: z.literal(
      'allowlist-v1; all contents; declared candidates; consumed checksums; no scientific recomputation',
    ),
  })
  .strict()
  .superRefine((s, ctx) => {
    const issue = (message: string) => ctx.addIssue({ code: 'custom', message })
    const contents = new Map(s.contents.map((c) => [c.contentId, c]))
    const frames = new Map<string, string>()
    for (const c of s.contents)
      for (const o of c.occurrences) {
        if (frames.has(o.frameId)) issue('Duplicate occurrence')
        frames.set(o.frameId, c.contentId)
      }
    if (
      contents.size !== s.contents.length ||
      contents.size !== s.dataset.uniqueContents ||
      frames.size !== s.dataset.occurrences ||
      s.contents.filter((c) => c.occurrences.length > 1).length !==
        s.dataset.duplicateGroups
    )
      issue('Dataset coverage mismatch')
    const runs = new Map(s.runs.map((r) => [r.id, r]))
    if (runs.size !== s.runs.length) issue('Duplicate run')
    for (const r of s.runs) {
      if (r.datasetId !== s.dataset.id) issue('Dataset identity mismatch')
      if (r.coverage !== contents.size) issue('Incomplete run coverage')
      const methods = {
        embeddings: ['DINOv2', 'CLIP'],
        similarity: ['cosine'],
        reduction: ['t-SNE', 'PaCMAP'],
        clustering: ['DBSCAN', 'OPTICS', 'HDBSCAN'],
        splits: ['historical', 'random_content', 'cluster_aware'],
        detector: ['detector'],
      }
      if (!methods[r.stage].includes(r.method)) issue('Stage method mismatch')
      if (
        r.stage === 'embeddings' &&
        (r.encoder === null ||
          r.method !== r.encoder ||
          r.contentIndex.length !== r.coverage ||
          new Set(r.contentIndex.map((p) => p.contentId)).size !== r.coverage ||
          r.contentIndex.some(
            (p, i) => !contents.has(p.contentId) || p.embeddingRow !== i,
          ))
      )
        issue('Feature index mismatch')
      if (
        (r.stage !== 'embeddings' && r.contentIndex.length) ||
        (r.stage !== 'reduction' && r.coordinates.length) ||
        (r.stage !== 'clustering' && r.labels.length) ||
        (r.stage !== 'similarity' && r.pairs.length) ||
        (r.stage !== 'splits' && r.assignments.length)
      )
        issue('Unexpected stage evidence')
      for (const [ref, stage] of [
        [r.featureSpaceId, 'embeddings'],
        [r.similaritySpaceId, 'similarity'],
        [r.reductionSpaceId, 'reduction'],
        [r.clusteringSpaceId, 'clustering'],
        [r.splitSpaceId, 'splits'],
      ] as const) {
        if (ref === null) continue
        const upstream = runs.get(ref)
        if (
          !upstream ||
          upstream.stage !== stage ||
          (r.encoder !== null &&
            upstream.encoder !== null &&
            upstream.encoder !== r.encoder) ||
          (stage !== 'embeddings' &&
            r.featureSpaceId &&
            upstream.featureSpaceId &&
            r.featureSpaceId !== upstream.featureSpaceId)
        )
          issue('Upstream identity mismatch')
      }
      if (
        r.stage === 'embeddings' &&
        (r.featureSpaceId !== r.id ||
          r.coverage > contents.size ||
          (r.encoder === 'DINOv2'
            ? r.dimensions !== 384 || r.pooling !== 'cls_token'
            : r.dimensions !== 512 || r.pooling !== 'projected_pooler_output'))
      )
        issue('Representation mismatch')
      if (
        ['similarity', 'reduction', 'clustering'].includes(r.stage) &&
        !r.featureSpaceId
      )
        issue('Missing feature identity')
      if (r.stage === 'reduction' && !r.similaritySpaceId)
        issue('Missing similarity identity')
      if (r.stage === 'detector' && !r.splitSpaceId)
        issue('Missing split identity')
      if (
        r.coordinates.length &&
        (r.coordinates.length !== r.coverage ||
          new Set(r.coordinates.map((p) => p.contentId)).size !== r.coverage ||
          new Set(r.coordinates.map((p) => p.embeddingRow)).size !==
            r.coverage ||
          r.coordinates.some(
            (p) => !contents.has(p.contentId) || p.embeddingRow >= r.coverage,
          ))
      )
        issue('Coordinate coverage mismatch')
      const feature = r.featureSpaceId ? runs.get(r.featureSpaceId) : undefined
      if (
        r.coordinates.some(
          (p) =>
            feature?.contentIndex[p.embeddingRow]?.contentId !== p.contentId,
        )
      )
        issue('Coordinate row mapping mismatch')
      if (
        r.pairs.some(
          (p) =>
            !contents.has(p.a) ||
            !contents.has(p.b) ||
            p.a === p.b ||
            (!p.sameSourceVideo &&
              (p.sampleGap !== null || p.gridSecondsGap !== null)),
        )
      )
        issue('Invalid neighbor mapping')
      const labels = new Map(r.labels.map((p) => [p.contentId, p.clusterId]))
      if (
        r.stage === 'clustering' &&
        (labels.size !== r.coverage ||
          labels.size !== r.labels.length ||
          r.labels.some((p) => !contents.has(p.contentId)))
      )
        issue('Cluster coverage mismatch')
      if (r.stage === 'splits') {
        if (
          r.assignments.length !== frames.size ||
          new Set(r.assignments.map((a) => a.frameId)).size !== frames.size ||
          r.assignments.some((a) => frames.get(a.frameId) !== a.contentId)
        )
          issue('Split occurrence coverage mismatch')
        const cluster = r.clusteringSpaceId
          ? runs.get(r.clusteringSpaceId)
          : undefined
        if (r.method === 'cluster_aware' && !cluster)
          issue('Missing clustering reference')
        const groups = new Map(
          cluster?.labels.map((l) => [
            l.contentId,
            l.clusterId === -1 ? l.contentId : `cluster-${l.clusterId}`,
          ]),
        )
        const allocation = new Map<string, string>()
        for (const a of r.assignments) {
          const group = groups.get(a.contentId) ?? a.contentId
          if (
            r.method !== 'historical' &&
            allocation.has(group) &&
            allocation.get(group) !== a.split
          )
            issue('Indivisible group was split')
          allocation.set(group, a.split)
        }
      }
    }
  })
export type LeakageSnapshot =
  | z.infer<typeof LeakageSnapshotV1>
  | z.infer<typeof import('./leakage-v2').LeakageSnapshotV2>
export type ArtifactRun = z.infer<typeof LeakageRun>
