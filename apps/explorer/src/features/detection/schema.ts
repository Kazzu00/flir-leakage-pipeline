import { z } from 'zod'

// Mirrors detection-export-v1; validation checks integrity, never scientific estimates.
export const strategyIds = [
  'historical',
  'random_content',
  'C10',
  'C12',
] as const
export const associationIds = [
  'primary_dinov2',
  'primary_clip',
  'extreme_dinov2',
  'extreme_clip',
  'temporal_at5',
  'domain_dinov2',
  'domain_clip',
] as const
export const payloadNames = [
  'manifest',
  'summary',
  'strategies',
  'splits',
  'runs',
  'classes',
  'support',
  'variance',
  'associations',
  'bootstrap',
] as const
export const contractFiles = [
  ...payloadNames.map((name) => `${name}.json`),
  'schema/detection-export-v1.schema.json',
]
const strategy = z.enum(strategyIds)
const associationId = z.enum(associationIds)
const id = z.string().min(1)
const count = z.number().int().nonnegative()
const positive = count.min(1)
const metric = z.number().min(0).max(1).nullable()
const correlation = z.number().min(-1).max(1).nullable()
const classId = z.union([
  z.literal(0),
  z.literal(1),
  z.literal(2),
  z.literal(3),
  z.literal(4),
])
const populationId = z.union([z.literal(-1), classId])
const metricId = z.enum(['map50_95', 'map50', 'precision', 'recall'])
const state = z.literal('COMPLETE_CONTROLLED_COMPARISON')
const unit = z.literal('split_mean_across_detector_seeds')
const hash = z.string().regex(/^[a-f0-9]{64}$/)
const means = {
  mean_map50_95: metric,
  mean_map50: metric,
  mean_precision: metric,
  mean_recall: metric,
}
const identity = { strategy, split_seed: z.number().int(), split_space_id: id }
const classIdentity = { class_id: classId, class_name: id }
const correlations = {
  n_splits: count,
  n_valid_splits: count,
  global_pearson: correlation,
  global_spearman: correlation,
  within_strategy_centered_pearson: correlation,
}
export const DetectionManifestSchema = z.strictObject({
  schema_version: z.literal('detection-export-v1'),
  state,
  scientific_result: z.literal(true),
  generated_from_verified_artifacts: z.literal(true),
  plan_id: id,
  model_config_id: id,
  expected_runs: positive,
  completed_runs: positive,
  strategy_count: positive,
  split_count: positive,
  detector_seed_count: positive,
  excluded_small_pilots: count,
  generated_at: z.string().datetime({ offset: true }),
  source_commit: z.string().regex(/^[a-f0-9]{40}$/),
  source_worktree_dirty: z.boolean().nullable(),
  source_receipt_sha256: z.record(z.string(), hash),
  file_sha256: z.record(z.string(), hash),
  limitations: z.array(id),
})
const StrategySchema = z.strictObject({
  ...means,
  strategy,
  display_name: id,
  n_splits: positive,
  n_detector_runs: positive,
  std_between_splits: metric,
  median_map50_95: metric,
  min_map50_95: metric,
  max_map50_95: metric,
  mean_temporal_at5: metric,
  mean_dinov2_nn_mean: correlation,
  mean_clip_nn_mean: correlation,
})
const VarianceSchema = z.strictObject({
  strategy,
  n_splits: positive,
  mean_within_split_detector_seed_std: metric,
  between_split_std: metric,
  between_vs_within_ratio: z.number().nonnegative().nullable(),
})
const ProtocolSchema = z.strictObject({
  plan_id: id,
  model_config_id: id,
  model: id,
  epochs: positive,
  batch: positive,
  detector_seeds: z.array(z.number().int()).min(1),
  bootstrap_method: z.literal('image_percentile_bootstrap'),
  bootstrap_resamples: positive,
  bootstrap_seed: z.number().int(),
  confidence_level: z.number().gt(0).lt(1),
})
export const DetectionSummarySchema = z.strictObject({
  state,
  scientific_result: z.literal(true),
  protocol: ProtocolSchema,
  headline_metrics: z.record(strategy, metric),
  strategies: z.array(StrategySchema),
  variance: z.array(VarianceSchema),
  association_summary: z.record(associationId, z.strictObject(correlations)),
  interpretation: z.strictObject({
    causal: z.literal(false),
    unit_of_analysis: unit,
    comparison: id,
    temporal_at5: id,
    clustering: id,
    missing_values: id,
  }),
  limitations: z.array(id),
})
const BundleSchema = z.strictObject({
  manifest: DetectionManifestSchema,
  summary: DetectionSummarySchema,
  strategies: z.array(StrategySchema),
  variance: z.array(VarianceSchema),
  splits: z.array(
    z.strictObject({
      ...identity,
      ...means,
      n_detector_runs: positive,
      std_map50_95: metric,
      temporal_at5: metric,
      dinov2_nn_mean: correlation,
      clip_nn_mean: correlation,
      dinov2_top001_pairs: count,
      clip_top001_pairs: count,
      exact_duplicate_cross_split_count: count,
      class_deviation_pp: z.number().min(0).max(100),
    }),
  ),
  runs: z.array(
    z.strictObject({
      ...identity,
      detector_run_id: id,
      detector_seed: z.number().int(),
      state: z.literal('COMPLETE'),
      map50_95: metric,
      map50: metric,
      precision: metric,
      recall: metric,
      best_epoch: count.nullable(),
      training_seconds: z.number().nonnegative().nullable(),
      peak_memory_bytes: count.nullable(),
    }),
  ),
  classes: z.array(
    z.strictObject({
      ...means,
      ...classIdentity,
      strategy,
      n_splits: positive,
      std_map50_95: metric,
    }),
  ),
  support: z.array(
    z.strictObject({ ...identity, ...classIdentity, support: count }),
  ),
  associations: z.array(
    z.strictObject({
      ...correlations,
      association_id: associationId,
      category: id,
      class_id: populationId,
      detector_metric: metricId,
      population: z.enum(['overall', 'class']),
      prespecified: z.literal(true),
      residual_metric: id,
      points: z.array(
        z.strictObject({ ...identity, x: z.number().nullable(), y: metric }),
      ),
      per_strategy: z.array(
        z.strictObject({
          strategy,
          n_valid_splits: count,
          pearson: correlation,
          spearman: correlation,
        }),
      ),
      interpretation: z.strictObject({
        causal: z.literal(false),
        unit_of_analysis: unit,
        absolute_r_reduction: z.number().nullable(),
        centering_reduces_absolute_r: z.boolean().nullable(),
        global_vs_within: id,
        undefined_correlation: id,
      }),
    }),
  ),
  bootstrap: z.array(
    z.strictObject({
      detector_run_id: id,
      class_id: populationId,
      metric: metricId,
      lower: metric,
      upper: metric,
      valid_resamples: count,
    }),
  ),
})

export const DetectionExportSchema = BundleSchema.superRefine((data, ctx) => {
  const check = (ok: boolean, message: string) => {
    if (!ok) ctx.addIssue({ code: 'custom', message })
  }
  const unique = (values: (string | number)[], name: string) =>
    check(new Set(values).size === values.length, `Duplicate ${name}`)
  const { manifest: m, summary: s } = data
  check(
    m.completed_runs === m.expected_runs &&
      data.runs.length === m.completed_runs,
    'Run count mismatch',
  )
  check(
    data.splits.length === m.split_count &&
      data.strategies.length === m.strategy_count,
    'Split/strategy count mismatch',
  )
  check(
    m.strategy_count === strategyIds.length &&
      data.associations.length === associationIds.length,
    'Incomplete controlled comparison',
  )
  check(
    s.protocol.plan_id === m.plan_id &&
      s.protocol.model_config_id === m.model_config_id,
    'Protocol identity mismatch',
  )
  check(
    s.protocol.detector_seeds.length === m.detector_seed_count,
    'Detector seed count mismatch',
  )
  unique(s.protocol.detector_seeds, 'detector seed')
  unique(
    data.strategies.map((v) => v.strategy),
    'strategy',
  )
  unique(
    data.splits.map((v) => v.split_space_id),
    'split identity',
  )
  unique(
    data.splits.map((v) => `${v.strategy}/${v.split_seed}`),
    'split seed',
  )
  unique(
    data.runs.map((v) => v.detector_run_id),
    'run identity',
  )
  unique(
    data.runs.map((v) => `${v.split_space_id}/${v.detector_seed}`),
    'run seed',
  )
  unique(
    data.associations.map((v) => v.association_id),
    'association',
  )
  unique(
    data.classes.map((v) => `${v.strategy}/${v.class_id}`),
    'class summary',
  )
  unique(
    data.support.map((v) => `${v.split_space_id}/${v.class_id}`),
    'class support',
  )
  unique(
    data.variance.map((v) => v.strategy),
    'variance strategy',
  )
  unique(
    data.bootstrap.map((v) => `${v.detector_run_id}/${v.class_id}/${v.metric}`),
    'bootstrap interval',
  )
  const splits = new Map(data.splits.map((v) => [v.split_space_id, v]))
  const bound = (
    v:
      | z.infer<typeof BundleSchema>['splits'][number]
      | { split_space_id: string; strategy: string; split_seed: number },
  ) => {
    const split = splits.get(v.split_space_id)
    return (
      !!split &&
      split.strategy === v.strategy &&
      split.split_seed === v.split_seed
    )
  }
  check(
    data.classes.length === data.strategies.length * 5 &&
      data.support.length === data.splits.length * 5,
    'Class/support coverage mismatch',
  )
  check(
    data.variance.length === data.strategies.length,
    'Variance coverage mismatch',
  )
  for (const v of data.strategies) {
    check(
      data.splits.filter((r) => r.strategy === v.strategy).length ===
        v.n_splits &&
        data.runs.filter((r) => r.strategy === v.strategy).length ===
          v.n_detector_runs,
      'Strategy membership mismatch',
    )
    check(
      s.headline_metrics[v.strategy] === v.mean_map50_95,
      'Headline mismatch',
    )
    check(
      JSON.stringify(s.strategies.find((r) => r.strategy === v.strategy)) ===
        JSON.stringify(v),
      'Summary strategy mismatch',
    )
    if (v.n_splits === 1)
      check(v.std_between_splits === null, 'Single-split variance must be null')
  }
  for (const v of data.splits) {
    const runs = data.runs.filter((r) => r.split_space_id === v.split_space_id)
    check(
      runs.length === v.n_detector_runs &&
        runs.length === m.detector_seed_count,
      'Split run count mismatch',
    )
  }
  for (const v of data.runs)
    check(
      bound(v) && s.protocol.detector_seeds.includes(v.detector_seed),
      'Unbound run',
    )
  for (const v of data.classes) {
    check(
      v.class_name ===
        data.classes.find(
          (r) => r.strategy === 'historical' && r.class_id === v.class_id,
        )?.class_name,
      'Class name/identity mismatch',
    )
    check(
      v.n_splits ===
        data.strategies.find((r) => r.strategy === v.strategy)?.n_splits,
      'Class split count mismatch',
    )
    if (v.n_splits === 1)
      check(v.std_map50_95 === null, 'Single-split class variance must be null')
  }
  for (const v of data.support)
    check(
      bound(v) &&
        v.class_name ===
          data.classes.find(
            (r) => r.strategy === v.strategy && r.class_id === v.class_id,
          )?.class_name,
      'Unbound class support',
    )
  for (const v of data.variance) {
    check(
      v.n_splits ===
        data.strategies.find((r) => r.strategy === v.strategy)?.n_splits,
      'Variance split count mismatch',
    )
    check(
      JSON.stringify(s.variance.find((r) => r.strategy === v.strategy)) ===
        JSON.stringify(v),
      'Summary variance mismatch',
    )
    if (v.n_splits === 1)
      check(
        v.between_split_std === null && v.between_vs_within_ratio === null,
        'Single-split variance/ratio must be null',
      )
  }
  for (const a of data.associations) {
    const domain = a.association_id.startsWith('domain_')
    const residual =
      a.association_id === 'temporal_at5'
        ? 'temporal_at5'
        : `${a.association_id.endsWith('clip') ? 'clip' : 'dinov2'}_${a.association_id.startsWith('extreme_') ? 'top001_pairs' : 'nn_mean'}`
    check(
      a.residual_metric === residual &&
        a.detector_metric === (domain ? 'recall' : 'map50_95') &&
        a.class_id === (domain ? 4 : -1),
      'Pre-specified association definition mismatch',
    )
    unique(
      a.points.map((p) => p.split_space_id),
      'association split',
    )
    unique(
      a.per_strategy.map((p) => p.strategy),
      'association strategy',
    )
    check(
      a.n_splits === m.split_count &&
        a.points.length === a.n_splits &&
        a.points.every(bound),
      'Association points must be split-level',
    )
    check(
      a.n_valid_splits ===
        a.points.filter((p) => p.x !== null && p.y !== null).length,
      'Valid point count mismatch',
    )
    check(
      (a.population === 'overall') === (a.class_id === -1),
      'Association population mismatch',
    )
    check(
      a.per_strategy.length === m.strategy_count,
      'Association strategy coverage mismatch',
    )
    for (const key of Object.keys(
      correlations,
    ) as (keyof typeof correlations)[])
      check(
        s.association_summary[a.association_id][key] === a[key],
        'Association summary mismatch',
      )
  }
  const runs = new Set(data.runs.map((r) => r.detector_run_id))
  for (const b of data.bootstrap) {
    check(runs.has(b.detector_run_id), 'Unbound bootstrap run')
    check(
      (b.lower === null) === (b.upper === null) &&
        (b.lower === null || b.upper === null || b.lower <= b.upper),
      'Invalid bootstrap bounds',
    )
    check(
      b.valid_resamples <= s.protocol.bootstrap_resamples,
      'Invalid resample count',
    )
  }
  const expected = contractFiles.filter((f) => f !== 'manifest.json').sort()
  check(
    JSON.stringify(Object.keys(m.file_sha256).sort()) ===
      JSON.stringify(expected),
    'Manifest file allowlist mismatch',
  )
})
export type DetectionExport = z.infer<typeof DetectionExportSchema>
export type DetectionAssociation = DetectionExport['associations'][number]
export type DetectionStrategyId = z.infer<typeof strategy>
