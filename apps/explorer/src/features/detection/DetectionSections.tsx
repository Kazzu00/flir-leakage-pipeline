import { useState } from 'react'
import { Panel, Select, Notice } from '@/components/feedback/Primitives'
import { Chart } from '@/components/visualization/Chart'
import type { DetectionExport } from './schema'
import { associationIds, strategyIds } from './schema'
import {
  associationOption,
  comparisonOption,
  labels,
  percent,
  decimal,
} from './presentation'

export function StrategyComparison({ data }: { data: DetectionExport }) {
  return (
    <Panel
      title="How did detector performance differ?"
      meta="mAP50–95 · SPLIT MEANS"
    >
      <Chart
        option={comparisonOption(data)}
        label="Split means and exported strategy means for mAP50–95"
        height={240}
      />
      <div className="detection-chart-caption">
        <span>
          Small symbols: one split · ◆ large diamond: strategy mean · line:
          observed split range
        </span>
        <span>
          Historical has one split; its between-split variability is undefined.
        </span>
      </div>
      <details className="detection-disclosure">
        <summary>Exact strategy and split values</summary>
        <div
          className="table-scroll"
          tabIndex={0}
          role="region"
          aria-label="Strategy values"
        >
          <table>
            <caption className="sr-only">Exported strategy means</caption>
            <thead>
              <tr>
                <th>Strategy</th>
                <th>Splits / runs</th>
                <th>mAP50–95</th>
                <th>mAP50</th>
                <th>Precision</th>
                <th>Recall</th>
              </tr>
            </thead>
            <tbody>
              {strategyIds.map((id) => {
                const s = data.strategies.find((s) => s.strategy === id)!
                return (
                  <tr key={id}>
                    <th scope="row">
                      {labels[id]}
                      <small className="detection-secondary">
                        {s.display_name}
                      </small>
                    </th>
                    <td>
                      {s.n_splits} / {s.n_detector_runs}
                    </td>
                    <td>{decimal(s.mean_map50_95, 6)}</td>
                    <td>{decimal(s.mean_map50, 6)}</td>
                    <td>{decimal(s.mean_precision, 6)}</td>
                    <td>{decimal(s.mean_recall, 6)}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
        <div
          className="table-scroll"
          tabIndex={0}
          role="region"
          aria-label="Split values"
        >
          <table>
            <caption>
              One row = one split; means across detector seeds, exported
              upstream.
            </caption>
            <thead>
              <tr>
                <th>Strategy</th>
                <th>Split seed</th>
                <th>mAP50–95</th>
                <th>Detector seed SD</th>
              </tr>
            </thead>
            <tbody>
              {data.splits.map((s) => (
                <tr key={s.split_space_id}>
                  <th scope="row">{labels[s.strategy]}</th>
                  <td>{s.split_seed}</td>
                  <td>{decimal(s.mean_map50_95, 6)}</td>
                  <td>{decimal(s.std_map50_95, 6)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </Panel>
  )
}
export function Variability({ data }: { data: DetectionExport }) {
  return (
    <Panel
      title="Splits versus detector seeds"
      meta="EXPORTED STANDARD DEVIATIONS"
    >
      <div className="panel-body">
        <p>
          Compare variation of split means with the mean detector-seed variation
          within a split. Ratios describe this protocol and these partitions.
        </p>
      </div>
      <div
        className="table-scroll"
        tabIndex={0}
        role="region"
        aria-label="Variability comparison"
      >
        <table>
          <caption className="sr-only">
            Exported variability, mAP50–95 percentage points
          </caption>
          <thead>
            <tr>
              <th>Strategy</th>
              <th>Within-split SD</th>
              <th>Between-split SD</th>
              <th>Between / within</th>
            </tr>
          </thead>
          <tbody>
            {strategyIds.map((id) => {
              const v = data.variance.find((v) => v.strategy === id)!
              return (
                <tr key={id}>
                  <th scope="row">{labels[id]}</th>
                  <td>
                    {v.mean_within_split_detector_seed_std === null
                      ? 'Unavailable'
                      : `${(v.mean_within_split_detector_seed_std * 100).toFixed(2)} pp`}
                  </td>
                  <td>
                    {v.n_splits === 1
                      ? 'Not defined — only one split'
                      : v.between_split_std === null
                        ? 'Unavailable'
                        : `${(v.between_split_std * 100).toFixed(2)} pp`}
                  </td>
                  <td>
                    {v.between_vs_within_ratio === null
                      ? v.n_splits === 1
                        ? 'Not defined'
                        : 'Unavailable'
                      : `${v.between_vs_within_ratio.toFixed(2)}×`}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </Panel>
  )
}
export function ClassPerformance({
  data,
  onSupport,
}: {
  data: DetectionExport
  onSupport: () => void
}) {
  const [metric, setMetric] = useState<'mean_map50_95' | 'mean_recall'>(
    'mean_map50_95',
  )
  const classes = data.classes.filter((c) => c.strategy === 'historical')
  return (
    <Panel title="Performance by class" meta="EXPORTED SPLIT-LEVEL MEANS">
      <div className="panel-body">
        <div className="toolbar">
          <Select
            label="Class metric"
            value={metric}
            onChange={(v) =>
              setMetric(v === 'mean_recall' ? 'mean_recall' : 'mean_map50_95')
            }
            options={[
              { value: 'mean_map50_95', label: 'mAP50–95' },
              { value: 'mean_recall', label: 'Recall' },
            ]}
          />
          <button className="details-trigger" onClick={onSupport}>
            View test composition
          </button>
        </div>
        <p>
          Each strategy evaluates different test membership and class support.
          These values are descriptive, not matched-test causal estimates.
        </p>
      </div>
      <div
        className="table-scroll"
        tabIndex={0}
        role="region"
        aria-label="Class performance"
      >
        <table>
          <caption className="sr-only">Class performance matrix</caption>
          <thead>
            <tr>
              <th>Class</th>
              {strategyIds.map((id) => (
                <th key={id}>{labels[id]}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {classes.map((c) => (
              <tr key={c.class_id}>
                <th scope="row">{c.class_name}</th>
                {strategyIds.map((id) => {
                  const value = data.classes.find(
                    (v) => v.strategy === id && v.class_id === c.class_id,
                  )![metric]
                  return (
                    <td key={id}>
                      <span title={decimal(value, 6)}>{percent(value)}</span>
                      {value !== null && (
                        <span
                          className="detection-cell-track"
                          aria-hidden="true"
                        >
                          <span style={{ width: `${value * 100}%` }} />
                        </span>
                      )}
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Panel>
  )
}
export function TestComposition({ data }: { data: DetectionExport }) {
  const classes = data.classes.filter((c) => c.strategy === 'historical')
  // Select an exported maximum only to set a common visual axis, not a new statistic.
  const axisMaximum =
    [...data.support].sort((a, b) => b.support - a.support)[0]?.support || 1
  return (
    <Panel
      title="Different test sets, different support"
      meta="EXPORTED CLASS SUPPORT"
    >
      <div className="panel-body">
        <Notice>
          Test composition changes across strategies; detector differences are
          therefore descriptive rather than isolated causal estimates of
          leakage.
        </Notice>
      </div>
      <div
        className="table-scroll"
        tabIndex={0}
        role="region"
        aria-label="Test composition"
      >
        <table>
          <caption>
            One row per split · class support as exported, without aggregation
          </caption>
          <thead>
            <tr>
              <th>Strategy / split seed</th>
              {classes.map((c) => (
                <th key={c.class_id}>{c.class_name}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.splits.map((s) => (
              <tr key={s.split_space_id}>
                <th scope="row">
                  {labels[s.strategy]} / {s.split_seed}
                </th>
                {classes.map((c) => {
                  const value = data.support.find(
                    (v) =>
                      v.split_space_id === s.split_space_id &&
                      v.class_id === c.class_id,
                  )?.support
                  return (
                    <td key={c.class_id}>
                      {value ?? 'Unavailable'}
                      {value !== undefined && (
                        <meter
                          className="detection-support-meter"
                          aria-label={`${c.class_name} support, ${labels[s.strategy]} split ${s.split_seed}`}
                          min={0}
                          max={axisMaximum}
                          value={value}
                        />
                      )}
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Panel>
  )
}
export function AssociationExplorer({ data }: { data: DetectionExport }) {
  const [selected, setSelected] = useState('temporal_at5')
  const a = data.associations.find((a) => a.association_id === selected)!
  return (
    <Panel
      title="Residual dependence and detector performance"
      meta="DESCRIPTIVE ASSOCIATION · NOT A CAUSAL ESTIMATE"
    >
      <div className="panel-body">
        <Select
          label="Pre-specified association"
          value={selected}
          onChange={setSelected}
          options={associationIds.map((id) => ({ value: id, label: id }))}
        />
        <div className="detection-correlation">
          <dl>
            <dt>GLOBAL ASSOCIATION</dt>
            <dd>
              {decimal(a.global_pearson)} <small>Pearson r</small>
            </dd>
            <dt>Spearman ρ: {decimal(a.global_spearman)}</dt>
          </dl>
          <span aria-hidden="true">→</span>
          <dl>
            <dt>WITHIN-STRATEGY ASSOCIATION</dt>
            <dd>
              {decimal(a.within_strategy_centered_pearson)}{' '}
              <small>centered Pearson r</small>
            </dd>
            <dt>Strategy-level offsets removed upstream</dt>
          </dl>
        </div>
        <p>
          {a.n_splits} splits · {a.n_valid_splits} valid pairs ·{' '}
          {a.prespecified ? 'Pre-specified' : 'Not pre-specified'} ·{' '}
          {a.population === 'overall'
            ? 'Overall detector performance'
            : data.classes.find((c) => c.class_id === a.class_id)?.class_name}
          . One point = one split, averaged across detector seeds upstream.
        </p>
        {selected === 'temporal_at5' && (
          <p>
            Frame/name index-distance proxy Δ≤5; not seconds or verified
            temporal continuity.
          </p>
        )}
      </div>
      <Chart
        option={associationOption(a)}
        label={`${a.association_id}: one point per split, ${a.n_valid_splits} valid pairs`}
        height={330}
      />
      <details className="detection-disclosure">
        <summary>Interpretation and exact split points</summary>
        <p lang="es">{a.interpretation.global_vs_within}</p>
        <p lang="es">{a.interpretation.undefined_correlation}</p>
        <p>Unit of analysis: {a.interpretation.unit_of_analysis}</p>
        <div
          className="table-scroll"
          tabIndex={0}
          role="region"
          aria-label="Association points"
        >
          <table>
            <caption>
              Exported points · {a.residual_metric} versus {a.detector_metric}
            </caption>
            <thead>
              <tr>
                <th>Strategy</th>
                <th>Split seed</th>
                <th>Residual metric (x)</th>
                <th>Detector metric (y)</th>
              </tr>
            </thead>
            <tbody>
              {a.points.map((p) => (
                <tr key={p.split_space_id}>
                  <th scope="row">{labels[p.strategy]}</th>
                  <td>{p.split_seed}</td>
                  <td>{decimal(p.x, 6)}</td>
                  <td>{decimal(p.y, 6)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </Panel>
  )
}
