import { useState } from 'react'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs'
import { TechnicalDetailsDrawer } from '@/components/research/TechnicalDetailsDrawer'
import { Notice } from '@/components/feedback/Primitives'
import { useDetectionSnapshot } from './query'
import { DetectionContractError } from './adapter'
import type { DetectionExport } from './schema'
import { labels, decimal, percent } from './presentation'
import {
  DetectionProvenance,
  DetectionLimitations,
  DetectionRuns,
} from './DetectionDetails'
import {
  StrategyComparison,
  Variability,
  ClassPerformance,
  TestComposition,
  AssociationExplorer,
} from './DetectionSections'

export function DetectionEvaluation() {
  const { data, isPending, error, refetch } = useDetectionSnapshot()
  if (isPending)
    return (
      <p className="empty" role="status">
        Loading verified detector results…
      </p>
    )
  if (error)
    return (
      <section className="empty" role="alert">
        <h2>
          {error instanceof DetectionContractError &&
          error.kind === 'validation'
            ? 'Detector result contract failed validation.'
            : 'Verified detector results are not available.'}
        </h2>
        <p>
          No demo detector results are substituted. Incomplete contracts are not
          final scientific evidence.
        </p>
        <button className="details-trigger" onClick={() => void refetch()}>
          Retry detector contract
        </button>
        <details className="detection-disclosure">
          <summary>Data-integrity details</summary>
          <pre>{error.message}</pre>
        </details>
      </section>
    )
  return <DetectionReady data={data} />
}
export function DetectionReady({ data }: { data: DetectionExport }) {
  const [view, setView] = useState('comparison')
  const m = data.manifest
  const protocol = data.summary.protocol
  const temporal = data.associations.find(
    (a) => a.association_id === 'temporal_at5',
  )!
  // Sorting selects an already-exported value, not a new scientific aggregate.
  const highest = [...data.strategies]
    .filter((s) => s.mean_map50_95 !== null)
    .sort((a, b) => b.mean_map50_95! - a.mean_map50_95!)[0]
  return (
    <div className="detection-evaluation">
      <section
        className="detection-status"
        aria-label="Detector experiment status"
      >
        <div>
          <span className="status status-verified">REAL / VERIFIED</span>
          <h2>Complete controlled comparison</h2>
          <p>
            {m.completed_runs} / {m.expected_runs} verified detector runs ·{' '}
            {m.split_count} splits · {m.strategy_count} strategies ·{' '}
            {m.detector_seed_count} detector seeds
          </p>
          <p className="muted">
            {protocol.model.replace(/\.pt$/, '').replace(/^yolo/i, 'YOLO')} ·{' '}
            {protocol.epochs} epochs · verification reported upstream
          </p>
        </div>
        <div className="detection-actions">
          <TechnicalDetailsDrawer title="Experiment provenance">
            <DetectionProvenance data={data} />
          </TechnicalDetailsDrawer>
          <TechnicalDetailsDrawer title="Interpretation & limitations">
            <DetectionLimitations data={data} />
          </TechnicalDetailsDrawer>
          <TechnicalDetailsDrawer title="View runs">
            <DetectionRuns data={data} />
          </TechnicalDetailsDrawer>
        </div>
      </section>
      <Notice>
        Controlled descriptive comparison. Test composition, class distribution,
        visual difficulty and partition membership can change together;
        performance differences do not isolate a causal leakage effect.
      </Notice>
      <Tabs value={view} onValueChange={setView} className="detection-tabs">
        <TabsList aria-label="Detector evidence views" variant="line">
          <TabsTrigger value="comparison">Comparison</TabsTrigger>
          <TabsTrigger value="variability">Variability</TabsTrigger>
          <TabsTrigger value="classes">Classes</TabsTrigger>
          <TabsTrigger value="support">Test composition</TabsTrigger>
          <TabsTrigger value="associations">Associations</TabsTrigger>
        </TabsList>
        <TabsContent value="comparison">
          <StrategyComparison data={data} />
          <div className="detection-takeaways">
            <div>
              <span className="micro muted">HIGHEST REPORTED mAP50–95</span>
              <strong>
                {highest
                  ? `${labels[highest.strategy]} · ${percent(highest.mean_map50_95)}`
                  : 'Unavailable'}
              </strong>
              <small>
                Across these evaluated splits; not evidence of a better
                scientific partition.
              </small>
            </div>
            <button onClick={() => setView('variability')}>
              <span className="micro muted">SPLIT / DETECTOR-SEED SD</span>
              <strong>
                {data.variance
                  .filter((v) => v.n_splits > 1)
                  .map(
                    (v) =>
                      `${labels[v.strategy]} ${v.between_vs_within_ratio === null ? 'Unavailable' : `${decimal(v.between_vs_within_ratio, 2)}×`}`,
                  )
                  .join(' · ')}
              </strong>
              <small>Exported variability ratios. Explore variability →</small>
            </button>
            <button onClick={() => setView('associations')}>
              <span className="micro muted">TEMPORAL PROXY · PEARSON r</span>
              <strong>
                {decimal(temporal.global_pearson)} global →{' '}
                {decimal(temporal.within_strategy_centered_pearson)} within
                strategy
              </strong>
              <small>
                Strategy-level offsets removed upstream. Compare associations →
              </small>
            </button>
          </div>
        </TabsContent>
        <TabsContent value="variability">
          <Variability data={data} />
        </TabsContent>
        <TabsContent value="classes">
          <ClassPerformance data={data} onSupport={() => setView('support')} />
        </TabsContent>
        <TabsContent value="support">
          <TestComposition data={data} />
        </TabsContent>
        <TabsContent value="associations">
          <AssociationExplorer data={data} />
        </TabsContent>
      </Tabs>
    </div>
  )
}
