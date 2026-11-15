import { useState } from 'react'
import { Select } from '@/components/feedback/Primitives'
import { DataTable } from '@/components/visualization/DataTable'
import type { ColumnDef } from '@tanstack/react-table'
import type { DetectionExport } from './schema'
import { labels, percent, decimal, strategyOptions } from './presentation'
import provenance from './snapshot-provenance.json'

export function DetectionProvenance({ data }: { data: DetectionExport }) {
  const m = data.manifest
  return (
    <dl className="detection-provenance">
      {Object.entries({
        'Schema version': m.schema_version,
        'Generation state': m.state,
        'Plan ID': m.plan_id,
        'Model config ID': m.model_config_id,
        'Generator source commit': m.source_commit,
        'Publication commit': provenance.publication_commit,
        'Publication branch': provenance.branch,
        'Expected / completed runs': `${m.expected_runs} / ${m.completed_runs}`,
        scientific_result: String(m.scientific_result),
        generated_from_verified_artifacts: String(
          m.generated_from_verified_artifacts,
        ),
        'Generated at': m.generated_at,
        'Source worktree dirty': String(m.source_worktree_dirty),
        'Excluded small pilots': m.excluded_small_pilots,
      }).map(([key, value]) => (
        <div key={key}>
          <dt>{key}</dt>
          <dd>{value}</dd>
        </div>
      ))}
      <div>
        <dt>Publication source</dt>
        <dd>
          <a
            className="row-link"
            href={`${provenance.repository}/tree/${provenance.publication_commit}/exports/frontend/detection`}
            target="_blank"
            rel="noreferrer"
          >
            flir-leakage-pipeline · frontend contract
          </a>
        </dd>
      </div>
      <div>
        <dt>Verification boundary</dt>
        <dd>
          Scientific verification is reported upstream. This frontend checks
          schema, file hashes and cross-file identities; it does not re-run the
          experiment.
        </dd>
      </div>
    </dl>
  )
}

export function DetectionLimitations({ data }: { data: DetectionExport }) {
  return (
    <div className="detection-methodology">
      <p lang="es">{data.summary.interpretation.comparison}</p>
      <dl className="detection-provenance">
        <div>
          <dt>Unit of analysis</dt>
          <dd>{data.summary.interpretation.unit_of_analysis}</dd>
        </div>
        <div>
          <dt>Causal estimate</dt>
          <dd>{String(data.summary.interpretation.causal)}</dd>
        </div>
      </dl>
      <ul lang="es">
        {Array.from(
          new Set([...data.manifest.limitations, ...data.summary.limitations]),
        ).map((text) => (
          <li key={text}>{text}</li>
        ))}
      </ul>
      <p lang="es">{data.summary.interpretation.missing_values}</p>
      <p>
        Bootstrap: {data.summary.protocol.bootstrap_method} · confidence level{' '}
        {percent(data.summary.protocol.confidence_level, 0)} ·{' '}
        {data.summary.protocol.bootstrap_resamples} resamples · seed{' '}
        {data.summary.protocol.bootstrap_seed}. Intervals describe individual
        runs, not a strategy-level confidence interval.
      </p>
    </div>
  )
}

type Run = DetectionExport['runs'][number]
const columns: ColumnDef<Run>[] = [
  {
    accessorKey: 'strategy',
    header: 'Strategy',
    cell: ({ row }) => labels[row.original.strategy],
  },
  { accessorKey: 'split_seed', header: 'Split seed' },
  { accessorKey: 'detector_seed', header: 'Detector seed' },
  ...(['map50_95', 'map50', 'precision', 'recall'] as const).map((key) => ({
    accessorKey: key,
    header: {
      map50_95: 'mAP50–95',
      map50: 'mAP50',
      precision: 'Precision',
      recall: 'Recall',
    }[key],
    cell: ({ row }: { row: { original: Run } }) => (
      <span title={decimal(row.original[key], 6)}>
        {percent(row.original[key], 2)}
      </span>
    ),
  })),
  {
    accessorKey: 'best_epoch',
    header: 'Best epoch',
    cell: ({ row }) => row.original.best_epoch ?? 'Unavailable',
  },
  {
    accessorKey: 'training_seconds',
    header: 'Training (s)',
    cell: ({ row }) => decimal(row.original.training_seconds, 1),
  },
  {
    accessorKey: 'peak_memory_bytes',
    header: 'Peak memory (bytes)',
    cell: ({ row }) =>
      row.original.peak_memory_bytes?.toLocaleString('en-US') ?? 'Unavailable',
  },
  { accessorKey: 'detector_run_id', header: 'Run ID' },
  { accessorKey: 'split_space_id', header: 'Split ID' },
]
export function DetectionRuns({ data }: { data: DetectionExport }) {
  const [strategy, setStrategy] = useState('all')
  const [splitSeed, setSplitSeed] = useState('all')
  const [detectorSeed, setDetectorSeed] = useState('all')
  const runs = data.runs.filter(
    (r) =>
      (strategy === 'all' || r.strategy === strategy) &&
      (splitSeed === 'all' || String(r.split_seed) === splitSeed) &&
      (detectorSeed === 'all' || String(r.detector_seed) === detectorSeed),
  )
  const all = { value: 'all', label: 'All' }
  const seeds = (values: number[]) => [
    all,
    ...Array.from(new Set(values))
      .sort((a, b) => a - b)
      .map((v) => ({ value: String(v), label: String(v) })),
  ]
  return (
    <>
      <div className="toolbar detection-filters">
        <Select
          label="Run strategy"
          value={strategy}
          onChange={setStrategy}
          options={[all, ...strategyOptions]}
        />
        <Select
          label="Split seed"
          value={splitSeed}
          onChange={setSplitSeed}
          options={seeds(data.splits.map((r) => r.split_seed))}
        />
        <Select
          label="Detector seed"
          value={detectorSeed}
          onChange={setDetectorSeed}
          options={seeds(data.summary.protocol.detector_seeds)}
        />
      </div>
      <p>
        {runs.length} of {data.runs.length} exported runs. No new summaries are
        calculated.
      </p>
      <DataTable label="Verified detector runs" data={runs} columns={columns} />
      <BootstrapIntervals data={data} />
    </>
  )
}
function BootstrapIntervals({ data }: { data: DetectionExport }) {
  const [runId, setRunId] = useState(data.runs[0].detector_run_id)
  const intervals = data.bootstrap.filter((b) => b.detector_run_id === runId)
  return (
    <details className="detection-disclosure">
      <summary>Exported bootstrap intervals</summary>
      <Select
        label="Bootstrap run"
        value={runId}
        onChange={setRunId}
        options={data.runs.map((r) => ({
          value: r.detector_run_id,
          label: `${labels[r.strategy]} · split ${r.split_seed} · detector ${r.detector_seed}`,
        }))}
      />
      <p>
        {data.summary.protocol.bootstrap_method} ·{' '}
        {percent(data.summary.protocol.confidence_level, 0)} confidence level.
        Image independence is assumed; residual frame dependence may
        underestimate uncertainty.
      </p>
      <div
        className="table-scroll"
        tabIndex={0}
        role="region"
        aria-label="Bootstrap interval table"
      >
        <table>
          <caption className="sr-only">
            Exported intervals for selected run
          </caption>
          <thead>
            <tr>
              <th>Population</th>
              <th>Metric</th>
              <th>Lower</th>
              <th>Upper</th>
              <th>Valid resamples</th>
            </tr>
          </thead>
          <tbody>
            {intervals.map((b) => (
              <tr key={`${b.class_id}/${b.metric}`}>
                <th scope="row">
                  {b.class_id === -1
                    ? 'Overall'
                    : data.classes.find((c) => c.class_id === b.class_id)
                        ?.class_name}
                </th>
                <td>{b.metric}</td>
                <td>{percent(b.lower, 2)}</td>
                <td>{percent(b.upper, 2)}</td>
                <td>{b.valid_resamples}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {!intervals.length && <p>Intervals unavailable for this run.</p>}
    </details>
  )
}
