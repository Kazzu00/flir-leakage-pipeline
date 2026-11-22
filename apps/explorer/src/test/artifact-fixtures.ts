import {
  LeakageSnapshotV1,
  type ArtifactRun,
  type LeakageSnapshot,
} from '../contracts/leakage'
import fixture from './fixtures/leakage-synthetic.json' with { type: 'json' }

/** Extra synthetic states only; never exported research results. */
export function withClusteringAndSplit(): LeakageSnapshot {
  const snapshot = LeakageSnapshotV1.parse(fixture)
  const reduction = snapshot.runs[2]
  const cluster: ArtifactRun = {
    ...reduction,
    id: 'clustering-000001',
    stage: 'clustering',
    method: 'DBSCAN',
    clusteringSpaceId: 'clustering-000001',
    coordinates: [],
    contentIndex: [],
    labels: snapshot.contents.map((c, i) => ({
      contentId: c.contentId,
      clusterId: i === 2 ? -1 : 0,
      medoid: i === 0,
    })),
  }
  const split: ArtifactRun = {
    ...cluster,
    id: 'split-000001',
    stage: 'splits',
    method: 'cluster_aware',
    encoder: null,
    featureSpaceId: null,
    similaritySpaceId: null,
    reductionSpaceId: null,
    splitSpaceId: 'split-000001',
    labels: [],
    assignments: snapshot.contents.flatMap((c, i) =>
      c.occurrences.map((o) => ({
        frameId: o.frameId,
        contentId: c.contentId,
        split: i === 2 ? ('test' as const) : ('train' as const),
      })),
    ),
  }
  snapshot.runs.push(cluster, split)
  return snapshot
}

export function largeSyntheticSnapshot(n = 9000): LeakageSnapshot {
  const snapshot = LeakageSnapshotV1.parse(fixture)
  const alias = (prefix: string, i: number) =>
    `${prefix}-${String(i + 1).padStart(6, '0')}`
  snapshot.dataset = {
    ...snapshot.dataset,
    occurrences: n,
    uniqueContents: n,
    duplicateGroups: 0,
  }
  snapshot.contents = Array.from({ length: n }, (_, i) => ({
    contentId: alias('content', i),
    occurrences: [
      {
        frameId: alias('frame', i),
        sourceVideo: 'video-000001',
        sampleIndex: i,
        gridSeconds: i,
        sampleFps: 1,
        sequenceId: null,
        captureTimestamp: null,
      },
    ],
  }))
  snapshot.runs = snapshot.runs.slice(0, 3).map((r) => ({
    ...r,
    coverage: n,
    contentIndex:
      r.stage === 'embeddings'
        ? snapshot.contents.map((c, i) => ({
            contentId: c.contentId,
            embeddingRow: i,
          }))
        : [],
    coordinates:
      r.stage === 'reduction'
        ? snapshot.contents.map((c, i) => ({
            contentId: c.contentId,
            embeddingRow: i,
            x: i % 100,
            y: Math.floor(i / 100),
          }))
        : [],
    pairs: [],
  }))
  return snapshot
}
