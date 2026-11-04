import fixture from './fixtures/leakage-synthetic.json' with { type: 'json' }
export function researchFixture() {
  const base = {
    datasetId: 'dataset-000001',
    origin: 'artifact',
    groundTruth: false,
    integrity: 'consumed-files-validated',
  }
  return {
    ...structuredClone(fixture),
    schemaVersion: 'LeakageSnapshotV2',
    research: {
      sequences: {
        state: 'review_required',
        artifact: {
          ...base,
          id: 'sequence-000001',
          instanceReviewBound: true,
          zones: [
            {
              id: 'zone-000001',
              sourceVideo: 'video-000001',
              start: 0,
              end: 1,
              kind: 'candidate_zone',
              decision: 'candidate',
              reviewId: null,
            },
            {
              id: 'zone-000002',
              sourceVideo: 'video-000001',
              start: 1,
              end: 2,
              kind: 'boundary_zone',
              decision: 'accepted',
              reviewId: 'review-000001',
            },
            {
              id: 'sequence-000002',
              sourceVideo: 'video-000001',
              start: 0,
              end: 2,
              kind: 'sequence_instance',
              decision: 'reviewed',
              reviewId: null,
            },
          ],
          reviews: [
            {
              id: 'review-000001',
              targetId: 'zone-000002',
              decision: 'accepted',
              groundTruth: false,
            },
          ],
          recurrence: [],
        },
      },
      linkage: {
        state: 'review_required',
        artifact: {
          ...base,
          id: 'linkage-000001',
          labeledDatasetId: 'dataset-labeled',
          sequenceSetId: 'sequence-000001',
          labeledContents: ['content-labeled-000001'],
          labeledOccurrences: [
            {
              frameId: 'frame-labeled-000001',
              contentId: 'content-labeled-000001',
            },
          ],
          candidates: [
            {
              id: 'linkage-000002',
              labeledContentId: 'content-labeled-000001',
              videoContentId: 'content-000001',
              sequenceIds: ['sequence-000002'],
              clipCosine: 0.91,
              dinov2Cosine: 0.88,
              state: 'candidate',
              reviewId: null,
            },
          ],
        },
      },
    },
  }
}
