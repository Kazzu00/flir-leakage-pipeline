import type { Point, Run, MetricValue } from '@/contracts'
export const metric = (
  name: string,
  value: number | null,
  unit = '',
): MetricValue => ({
  name,
  value,
  unit,
  origin: value === null ? 'unavailable' : 'mock',
  caveat:
    value === null
      ? 'No validated artifact connected.'
      : 'Synthetic demonstration; not a scientific result.',
})
const base = (
  id: string,
  module: Run['module'],
  stage: string,
  method: string,
): Run => ({
  id,
  module,
  stage,
  method,
  datasetId: 'demo-dataset-v1',
  status: 'complete',
  verification: 'mock',
  origin: 'mock',
  seed: 42,
  parameters: { seed: 42 },
  metrics: [],
  artifacts: [
    {
      id: `${id}-metadata`,
      label: 'Run metadata',
      format: 'JSON',
      status: 'mock',
      description:
        'Synthetic normalized metadata; no underlying scientific file is connected.',
    },
  ],
  caveat: 'DEMO · synthetic fixture, not an executed scientific run.',
})
const points: Point[] = Array.from({ length: 144 }, (_, i) => {
  const clusterId = i >= 132 ? -1 : Math.floor(i / 22)
  const group = clusterId === -1 ? i : clusterId
  const angle = i * 2.399963
  const radius = 3 + ((i * 17) % 22)
  const centers = [
    [25, 65],
    [62, 70],
    [42, 33],
    [77, 34],
    [19, 25],
    [85, 83],
  ]
  const center = centers[Math.max(0, clusterId)]
  return {
    contentId: `demo-content-${String(i).padStart(3, '0')}`,
    frameIds: [`demo-frame-${i}`, ...(i < 12 ? [`demo-copy-${i}`] : [])],
    embeddingRow: i,
    clusterId,
    groupId: `demo-group-${group}`,
    sourceVideo: `demo-video-${Math.floor(i / 48) + 1}`,
    sequenceId: null,
    sequenceVerified: false,
    sampleIndex: i % 48,
    x:
      clusterId === -1
        ? (i * 31) % 95
        : center[0] + Math.cos(angle) * radius * 0.55,
    y:
      clusterId === -1
        ? (i * 19) % 95
        : center[1] + Math.sin(angle) * radius * 0.5,
  }
})
const clusteringBase = base(
  'demo-cluster-dbscan',
  'organization',
  'Clustering',
  'DBSCAN',
)
const clusters = [-1, 0, 1, 2, 3, 4, 5].map((id) => {
  const members = points.filter((p) => p.clusterId === id)
  return {
    id,
    size: members.length,
    medoidContentId: id === -1 ? null : members[0].contentId,
    provenance: 'Synthetic source-video sampling grid; sequence unknown.',
    metrics: [metric('Silhouette · original space', id === -1 ? null : 0.42)],
  }
})
const clusterings = [
  {
    ...clusteringBase,
    encoder: 'DINOv2',
    reduction: 'PaCMAP',
    algorithm: 'DBSCAN',
    parameters: { eps: 0.28, min_samples: 5, seed: 42 },
    points,
    clusters,
  },
  {
    ...base('demo-cluster-hdbscan', 'organization', 'Clustering', 'HDBSCAN'),
    encoder: 'CLIP',
    reduction: 't-SNE',
    algorithm: 'HDBSCAN',
    parameters: { min_cluster_size: 10, seed: 42 },
    points: points.map((p) => ({ ...p, x: 100 - p.x })),
    clusters,
  },
  {
    ...base('demo-cluster-optics', 'organization', 'Clustering', 'OPTICS'),
    encoder: 'DINOv2',
    reduction: 'Original L2',
    algorithm: 'OPTICS',
    parameters: { min_samples: 5, xi: 0.05, seed: 42 },
    points,
    clusters,
  },
]
const embeddings = [
  {
    ...base(
      'demo-dinov2',
      'organization',
      'Embeddings',
      'facebook/dinov2-small',
    ),
    encoder: 'DINOv2',
    dimensions: 384,
    pooling: 'CLS token',
    featureSpaceId: 'demo-dino-cls-v1',
    coverage: 144,
  },
  {
    ...base(
      'demo-clip',
      'organization',
      'Embeddings',
      'openai/clip-vit-base-patch32',
    ),
    encoder: 'CLIP',
    dimensions: 512,
    pooling: 'Projected image embedding',
    featureSpaceId: 'demo-clip-projected-v1',
    coverage: 144,
  },
]
const reductions = ['PaCMAP', 't-SNE'].map((method, i) => ({
  ...base(`demo-reduction-${i}`, 'organization', 'Reduction', method),
  encoder: i ? 'CLIP' : 'DINOv2',
  method,
  selectedCandidate: i === 0,
  parameters: i
    ? { perplexity: 30, seed: 42 }
    : { n_neighbors: 10, MN_ratio: 0.5, FP_ratio: 2, seed: 42 },
  metrics: [
    metric('Trustworthiness @15', 0.93 - i * 0.02),
    metric('Continuity @15', 0.9 - i * 0.02),
  ],
}))
const splits = ['Historical', 'Random/content', 'Cluster-aware'].map(
  (strategy, j) => {
    const assignments = points.map((p, i) => ({
      contentId: p.contentId,
      splits:
        j === 0 && i < 12
          ? ['train', 'test']
          : [
              j === 2
                ? p.clusterId === 4
                  ? 'validation'
                  : p.clusterId === 5
                    ? 'test'
                    : 'train'
                : i % 10 < 7
                  ? 'train'
                  : i % 10 === 7
                    ? 'validation'
                    : 'test',
            ],
    }))
    const counts = { train: 0, validation: 0, test: 0 }
    const occurrences = points.flatMap((p, i) =>
      p.frameIds.map((frameId, k) => ({
        frameId,
        contentId: p.contentId,
        split:
          assignments[i].splits.length > 1
            ? assignments[i].splits[k]
            : assignments[i].splits[0],
      })),
    )
    occurrences.forEach((o) => {
      counts[o.split as keyof typeof counts]++
    })
    return {
      ...base(`demo-split-${j}`, 'organization', 'Splits', strategy),
      strategy,
      clusteringRunId: clusteringBase.id,
      assignments,
      occurrences,
      counts,
      metrics: [
        metric('Cross-split exact contents', j === 0 ? 12 : 0),
        metric('Residual cosine · mean NN', [0.92, 0.88, 0.76][j]),
      ],
      caveat:
        j === 0
          ? 'DEMO · historical content may occur in multiple splits.'
          : j === 1
            ? 'DEMO · content integrity only; clusters can cross splits.'
            : 'DEMO · indivisible groups. Zero exact overlap does not establish independence.',
    }
  },
)
const preprocessing = {
  ...base('demo-preprocess-01', 'preprocessing', 'Denoising', 'Noise2Noise'),
  parameters: { mode: 'sin_hud', epochs: 15, learning_rate: 0.0001 },
  metrics: [
    metric('NIQE', 5.8),
    metric('BRISQUE', 31.2),
    metric('PIQE', 28.5),
    metric('Noise sigma', 8.2),
    metric('Laplacian variance', 104.3),
    metric('Sharpness retention', 0.91),
  ],
}
const segmentation = {
  ...base('demo-panoptic-01', 'segmentation', 'Predictions', 'Panoptic FCN'),
  status: 'experimental',
  modelVariant: 'base',
  parameters: {
    backbone: 'placeholder conv trunk',
    stuff_classes: 2,
    thing_classes: 3,
    seed: 42,
  },
  caveat:
    'Source documentation is inconsistent. Dense ground-truth evaluation is not connected. Existing upstream box-based outputs report overlap caveats; no final conclusion is supported.',
  metrics: [
    'PQ',
    'SQ',
    'RQ',
    'IoU',
    'F1',
    'Precision',
    'Recall',
    'Mask AP',
  ].map((name) => metric(name, null)),
  predictions: [0, 1, 2].map((i) => ({
    id: `demo-prediction-${i}`,
    contentId: points[i].contentId,
    origin: 'mock',
    metrics: [metric('IoU', 0.48 + i * 0.02)],
    caveat: 'SAMPLE placeholder overlay, not model output or ground truth.',
  })),
  classes: ['River', 'SDZI', 'Thing classes (mapping pending)'].map((name) => ({
    name,
    metrics: [metric('IoU', null), metric('F1', null), metric('Recall', null)],
  })),
}
const detector = {
  ...base('demo-detector-protocol', 'organization', 'Detector', 'YOLO11n'),
  status: 'pending',
  metrics: ['Precision', 'Recall', 'mAP@50', 'mAP@50–95'].map((name) =>
    metric(name, null),
  ),
  caveat:
    'DEMO protocol placeholder only. The verified historical controlled comparison uses a separate contract under Evaluation.',
}
export const fixture = {
  schemaVersion: 1,
  modules: [
    {
      id: 'preprocessing',
      number: '01',
      name: 'Preprocessing',
      subtitle: 'From aerial video to analysis-ready frames',
      description:
        'Inspect extraction, HUD removal and denoising across source videos.',
      status: 'experimental',
      repository: 'https://github.com/Laura-Martinez-Galindo/proyecto-FAC',
      evidence:
        'Scripts and stage states exist. No executions reverified here.',
      stages: [
        ['frames', 'Frame extraction', 'Original frames'],
        ['hud', 'HUD segmentation', 'Binary masks'],
        ['inpainting', 'ProPainter inpainting', 'Cleaned frames'],
        ['denoising', 'Denoising', 'Restored frames'],
        ['quality', 'Quality evaluation', 'NR-IQA summaries'],
      ].map(([id, name, output]) => ({
        id,
        name,
        output,
        status: 'experimental',
      })),
    },
    {
      id: 'organization',
      number: '02',
      name: 'Representation & organization',
      subtitle: 'Make visual relationships auditable',
      description:
        'Explore representations, related contents and leakage-aware partitions.',
      status: 'experimental',
      repository: 'https://github.com/Kazzu00/flir-leakage-pipeline',
      evidence:
        'DEMO organization fixture. Sampled-video downstream is separate from the historical detector contract under Evaluation.',
      stages: [
        ['dataset', 'Canonical manifest', 'Occurrence ↔ content mapping'],
        ['embeddings', 'DINOv2 / CLIP', 'Raw + L2 embeddings'],
        ['similarity', 'Cosine similarity', 'Neighbors and pairs'],
        ['reduction', 't-SNE / PaCMAP', 'Reduced coordinates'],
        ['clustering', 'Density clustering', 'Clusters + explicit noise'],
        ['splits', 'Group-aware splitting', 'Atomic assignments'],
      ].map(([id, name, output]) => ({
        id,
        name,
        output,
        status: 'experimental',
      })),
    },
    {
      id: 'segmentation',
      number: '03',
      name: 'Panoptic segmentation',
      subtitle: 'Explore semantic and instance predictions',
      description:
        'Inspect pseudo-annotations, model variants and prediction evidence.',
      status: 'inconsistent',
      repository:
        'https://github.com/manugalarza/proyecto-segementacion-panoptica',
      evidence:
        'README, training code and reports disagree. Outputs have evaluation caveats.',
      stages: [
        ['points', 'Box → points → pseudo-mask', 'Pseudo-annotations'],
        ['model', 'Panoptic FCN + context fusion', 'Model variants'],
        ['training', 'Training', 'Checkpoints'],
        ['predictions', 'Predictions', 'Panoptic maps'],
        ['evaluation', 'Evaluation', 'Per-image / per-class metrics'],
      ].map(([id, name, output]) => ({
        id,
        name,
        output,
        status: id === 'training' ? 'inconsistent' : 'experimental',
      })),
    },
  ],
  dataset: {
    id: 'demo-dataset-v1',
    name: 'Aerial sampling · synthetic collection',
    origin: 'mock',
    occurrences: 156,
    uniqueContents: 144,
    duplicateGroups: 12,
    provenance:
      '3 fictional source videos. No validated sequences or capture timestamps.',
    verification: 'mock',
  },
  runs: [
    preprocessing,
    ...embeddings,
    ...reductions,
    ...clusterings,
    ...splits,
    segmentation,
    detector,
  ],
  embeddings,
  reductions,
  clusterings,
  splits,
  similarity: {
    ...base('demo-cosine', 'organization', 'Similarity', 'Cosine · L2'),
    pairs: points.slice(1, 9).map((p, i) => ({
      a: points[0].contentId,
      b: p.contentId,
      cosine: 0.98 - i * 0.025,
      sameSourceVideo: true,
      sampleGap: i + 1,
    })),
    distribution: Array.from({ length: 10 }, (_, i) => ({
      bin: (i / 10).toFixed(1),
      count: [3, 8, 15, 24, 39, 48, 62, 47, 29, 13][i],
    })),
  },
  segmentation,
  quality: Array.from({ length: 24 }, (_, i) => ({
    index: i,
    original: 8 + Math.sin(i * 0.7) * 0.8,
    n2n: 6 + Math.sin(i * 0.6) * 0.6,
    n2v: 6.4 + Math.cos(i * 0.5) * 0.5,
  })),
}
