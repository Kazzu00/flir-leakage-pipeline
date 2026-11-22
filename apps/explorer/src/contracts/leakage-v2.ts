import { z } from 'zod'
import { LeakageSnapshotV1 } from './leakage'
import { ResearchSchema } from './research'
/** Reuse every V1 invariant without weakening its strict wire contract. */
export const LeakageSnapshotV2 = z
  .object({
    ...LeakageSnapshotV1.shape,
    schemaVersion: z.literal('LeakageSnapshotV2'),
    research: ResearchSchema,
  })
  .strict()
  .superRefine((s, ctx) => {
    const { research, ...base } = s
    const valid = LeakageSnapshotV1.safeParse({
      ...base,
      schemaVersion: 'LeakageSnapshotV1',
    })
    if (!valid.success)
      for (const issue of valid.error.issues)
        ctx.addIssue({
          code: 'custom',
          message: issue.message,
          path: issue.path,
        })
    const issue = (message: string) => ctx.addIssue({ code: 'custom', message })
    const frames = new Map(
      s.contents.flatMap((c) =>
        c.occurrences.map(
          (o) => [`${o.sourceVideo}:${o.sampleIndex}`, o] as const,
        ),
      ),
    )
    const contents = new Set(s.contents.map((c) => c.contentId))
    const seq = research.sequences.artifact
    const zones = new Map(seq?.zones.map((z) => [z.id, z]))
    const reviews = new Map(seq?.reviews.map((r) => [r.id, r]))
    if (
      seq &&
      (zones.size !== seq.zones.length || reviews.size !== seq.reviews.length)
    )
      issue('Duplicate sequence evidence identity')
    for (const zone of zones.values()) {
      if (
        !frames.has(`${zone.sourceVideo}:${zone.start}`) ||
        !frames.has(`${zone.sourceVideo}:${zone.end}`)
      )
        issue('Sequence source/grid mismatch')
      const review = zone.reviewId ? reviews.get(zone.reviewId) : undefined
      if (zone.kind === 'sequence_instance') {
        if (!seq?.instanceReviewBound || zone.decision !== 'reviewed')
          issue('Sequence instance needs bound source-set review')
      } else if (
        zone.decision !== 'candidate' &&
        (!review ||
          review.targetId !== zone.id ||
          review.decision !== zone.decision)
      )
        issue('Sequence decision missing bound review')
    }
    for (const review of reviews.values())
      if (!zones.has(review.targetId)) issue('Unknown reviewed sequence target')
    for (const r of seq?.recurrence ?? [])
      if (!zones.has(r.a) || !zones.has(r.b) || r.a === r.b)
        issue('Unknown recurrence target')
    const link = research.linkage.artifact
    const links = new Map(link?.candidates.map((c) => [c.id, c]))
    const linkReviews = new Map(
      research.linkageReview.artifact?.map((r) => [r.id, r]),
    )
    if (link) {
      if (link.datasetId === link.labeledDatasetId)
        issue('Historical and sampled datasets must remain distinct')
      if (!seq || link.sequenceSetId !== seq.id)
        issue('Linkage sequence set mismatch')
      const labeled = new Set(link.labeledContents)
      if (
        links.size !== link.candidates.length ||
        labeled.size !== link.labeledContents.length
      )
        issue('Duplicate linkage identity')
      if (
        new Set(link.labeledOccurrences.map((o) => o.frameId)).size !==
          link.labeledOccurrences.length ||
        link.labeledOccurrences.some((o) => !labeled.has(o.contentId)) ||
        new Set(link.labeledOccurrences.map((o) => o.contentId)).size !==
          labeled.size
      )
        issue('Labeled occurrence mapping mismatch')
      for (const c of links.values()) {
        if (!contents.has(c.videoContentId) || !labeled.has(c.labeledContentId))
          issue('Linkage content mismatch')
        if (
          c.sequenceIds.some(
            (id) => zones.get(id)?.kind !== 'sequence_instance',
          )
        )
          issue('Linkage sequence mismatch')
        if (c.state !== 'candidate') {
          const review = c.reviewId ? linkReviews.get(c.reviewId) : undefined
          // Current upstream supported/unsupported reviews never confirm exact matches.
          if (
            c.state === 'confirmed' ||
            !review ||
            review.targetId !== c.id ||
            review.decision !== c.state
          )
            issue('Unsupported linkage confirmation or unbound review')
        }
      }
    }
    if (linkReviews.size !== (research.linkageReview.artifact?.length ?? 0))
      issue('Duplicate linkage review')
    for (const review of linkReviews.values())
      if (!links.has(review.targetId)) issue('Unknown linkage review target')
    const aggregate = research.reviewAggregation.artifact
    if (
      aggregate &&
      (!link || aggregate.linkageId !== link.id || aggregate.confirmed !== null)
    )
      issue('Aggregation linkage or confirmation mismatch')
    const groupReview = research.groupReview.artifact
    if (
      groupReview &&
      (!link ||
        groupReview.linkageId !== link.id ||
        groupReview.items.some(
          (r) => !link.labeledContents.includes(r.labeledContentId),
        ) ||
        new Set(groupReview.items.map((r) => r.id)).size !==
          groupReview.items.length)
    )
      issue('Group review linkage mismatch')
    const artifacts = [
      seq,
      link,
      aggregate,
      groupReview,
      ...(research.experiments.artifact ?? []),
      ...(research.evidence.artifact ?? []),
    ]
    for (const artifact of artifacts)
      if (artifact && artifact.datasetId !== s.dataset.id)
        issue('Research dataset mismatch')
    if (
      research.variants.some((v) => v.datasetId !== s.dataset.id) ||
      new Set(research.variants.map((v) => v.id)).size !==
        research.variants.length
    )
      issue('Variant identity mismatch')
  })
export const LeakageSnapshotSchema = z.discriminatedUnion('schemaVersion', [
  LeakageSnapshotV1,
  LeakageSnapshotV2,
])
