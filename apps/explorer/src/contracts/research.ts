import { z } from 'zod'
// No free-text producer metadata: aliases and controlled vocabulary only.
export const ResearchAlias = z
  .string()
  .regex(
    /^(dataset|video|sequence|zone|evidence|experiment|linkage|review|variant|content|frame|feature|group)-[a-z0-9-]{1,40}$/,
  )
export const EvidenceState = z.enum([
  'implemented',
  'artifact_available',
  'executed',
  'verified',
  'experimental',
  'review_required',
  'pending',
  'unavailable',
  'demo',
])
const Count = z.number().int().nonnegative()
const Score = z.number().finite().nullable()
const Decision = z.enum([
  'candidate',
  'reviewed',
  'accepted',
  'rejected',
  'ambiguous',
  'supported',
  'unsupported',
])
const Zone = z
  .object({
    id: ResearchAlias,
    sourceVideo: ResearchAlias,
    start: Count,
    end: Count,
    kind: z.enum([
      'candidate_zone',
      'boundary_zone',
      'sequence_candidate',
      'sequence_instance',
    ]),
    decision: Decision,
    reviewId: ResearchAlias.nullable(),
  })
  .strict()
  .refine((r) => r.start <= r.end, 'Reversed sampling interval')
const Review = z
  .object({
    id: ResearchAlias,
    targetId: ResearchAlias,
    decision: Decision.exclude(['candidate']),
    groundTruth: z.literal(false),
  })
  .strict()
const Recurrence = z
  .object({
    a: ResearchAlias,
    b: ResearchAlias,
    score: Score,
    kind: z.literal('visual_recurrence'),
  })
  .strict()
const base = {
  datasetId: ResearchAlias,
  origin: z.enum(['artifact', 'reported', 'mock']),
  groundTruth: z.literal(false),
  integrity: z.literal('consumed-files-validated'),
}
const Sequences = z
  .object({
    ...base,
    id: ResearchAlias,
    zones: z.array(Zone).max(20000),
    reviews: z.array(Review).max(20000),
    recurrence: z.array(Recurrence).max(20000),
    instanceReviewBound: z.boolean().default(false),
  })
  .strict()
const Experiment = z
  .object({
    ...base,
    id: ResearchAlias,
    encoder: z.enum(['CLIP', 'DINOv2', 'agreement']).nullable(),
    representation: z.enum(['original_l2', 'PaCMAP', 't-SNE']).nullable(),
    method: z
      .enum(['temporal', 'DBSCAN', 'OPTICS', 'HDBSCAN', 'Agglomerative'])
      .nullable(),
    coverage: Count.nullable(),
    agreement: Score,
    stability: Score,
    recurrenceCandidates: Count.nullable(),
    reviewState: z.enum(['pending', 'partial', 'reviewed']),
    reviewMaskCoverage: Score,
    ablation: z.boolean(),
    immutable: z.boolean(),
    reportedMetrics: z
      .partialRecord(
        z.enum([
          'ari',
          'ami',
          'evaluated_content_coverage',
          'evaluated_n',
          'total_contents',
          'cluster_count',
          'noise_coverage',
          'successful_runs',
          'failed_cells',
        ]),
        Score,
      )
      .default({}),
  })
  .strict()
const Evidence = z
  .object({
    ...base,
    id: ResearchAlias,
    source: z.enum([
      'hypatia_legacy_evidence_v1',
      'native',
      'manual',
      'external',
    ]),
    state: z.enum(['imported', 'verified', 'pending']),
    canonicalBinding: z.boolean().nullable(),
    reviewMode: z.enum(['legacy', 'manual', 'unavailable']),
  })
  .strict()
const Link = z
  .object({
    id: ResearchAlias,
    labeledContentId: ResearchAlias,
    videoContentId: ResearchAlias,
    sequenceIds: z.array(ResearchAlias),
    clipCosine: z.number().min(-1.00001).max(1.00001).nullable(),
    dinov2Cosine: z.number().min(-1.00001).max(1.00001).nullable(),
    state: z.enum([
      'candidate',
      'reviewed',
      'confirmed',
      'rejected',
      'ambiguous',
      'supported',
      'unsupported',
    ]),
    reviewId: ResearchAlias.nullable(),
  })
  .strict()
const Linkage = z
  .object({
    ...base,
    id: ResearchAlias,
    labeledDatasetId: ResearchAlias,
    sequenceSetId: ResearchAlias,
    candidates: z.array(Link).max(100000),
    labeledContents: z.array(ResearchAlias).max(20000),
    labeledOccurrences: z
      .array(
        z.object({ frameId: ResearchAlias, contentId: ResearchAlias }).strict(),
      )
      .max(100000),
  })
  .strict()
const Aggregation = z
  .object({
    ...base,
    id: ResearchAlias,
    linkageId: ResearchAlias,
    reviews: Count,
    reviewedItems: Count,
    confirmed: Count.nullable(),
    rejected: Count.nullable(),
    ambiguous: Count.nullable(),
    supported: Count.nullable(),
    unsupported: Count.nullable(),
    conflicts: Count.nullable(),
  })
  .strict()
function stage<T extends z.ZodType>(artifact: T) {
  return z
    .discriminatedUnion('state', [
      z
        .object({
          state: z.enum(['pending', 'unavailable', 'implemented']),
          artifact: z.null(),
        })
        .strict(),
      z
        .object({
          state: z.enum([
            'artifact_available',
            'executed',
            'verified',
            'experimental',
            'review_required',
            'demo',
          ]),
          artifact,
        })
        .strict(),
    ])
    .prefault({ state: 'pending', artifact: null })
}
export const ResearchSchema = z
  .object({
    sequences: stage(Sequences),
    experiments: stage(z.array(Experiment).min(1).max(256)),
    evidence: stage(z.array(Evidence).min(1).max(256)),
    linkage: stage(Linkage),
    linkageReview: stage(z.array(Review).min(1).max(100000)),
    groupReview: stage(
      z
        .object({
          ...base,
          id: ResearchAlias,
          linkageId: ResearchAlias,
          items: z
            .array(
              z
                .object({
                  id: ResearchAlias,
                  labeledContentId: ResearchAlias,
                  groupId: ResearchAlias,
                  decision: z.enum([
                    'supported',
                    'unsupported',
                    'ambiguous',
                    'pending',
                  ]),
                })
                .strict(),
            )
            .max(100000),
        })
        .strict(),
    ),
    reviewAggregation: stage(Aggregation),
    variants: z
      .array(
        z
          .object({
            id: ResearchAlias,
            datasetId: ResearchAlias,
            name: z.enum(['unspecified', 'original_with_hud', 'no_hud']),
          })
          .strict(),
      )
      .max(32)
      .default([]),
  })
  .strict()
export type Research = z.infer<typeof ResearchSchema>
