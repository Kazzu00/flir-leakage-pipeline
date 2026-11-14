import { useState } from 'react'
import type { ArtifactRun, LeakageSnapshot } from '@/contracts/leakage'
import {
  Panel,
  Notice,
  Select,
  Metrics,
} from '@/components/feedback/Primitives'
import { TechnicalDetailsDrawer } from '@/components/research/TechnicalDetailsDrawer'
import {
  EmptyArtifactState,
  MetricSummary,
} from '@/components/research/Evidence'
import { ArtifactExplorer } from './evidence-views/ArtifactExplorer'
import { SimilarityEvidence } from './evidence-views/SimilarityEvidence'
import { SplitEvidence } from './evidence-views/SplitEvidence'
import { SummaryTables } from './evidence-views/SummaryTables'

export function RunDetails({ run }: { run: ArtifactRun }) {
  const {
    contentIndex,
    coordinates,
    pairs,
    labels,
    assignments,
    summaries,
    ...metadata
  } = run
  return (
    <TechnicalDetailsDrawer>
      <pre className="audit-json">{JSON.stringify(metadata, null, 2)}</pre>
      <p>
        Index rows: {contentIndex.length}; coordinates: {coordinates.length};
        pairs: {pairs.length}; labels: {labels.length}; assignments:{' '}
        {assignments.length}.
      </p>
      <Metrics
        metrics={Object.entries(run.metrics).map(([name, value]) => ({
          name,
          value,
          unit: '',
          origin: run.metricOrigin,
          caveat: 'Reported upstream.',
        }))}
      />
      {summaries.length > 0 && <SummaryTables run={run} />}
    </TechnicalDetailsDrawer>
  )
}
/** Compatibility entrypoint. Domain renderers own their interactions. */
export function ArtifactStage({
  snapshot,
  stage,
  encoder: controlledEncoder,
}: {
  snapshot: LeakageSnapshot
  stage: string
  encoder?: string
}) {
  const [encoder, setEncoder] = useState('DINOv2')
  const [selected, setSelected] = useState('')
  const [method, setMethod] = useState('all')
  const actualStage = stage === 'groups' ? 'clustering' : stage
  const available = snapshot.runs.filter(
    (r) =>
      r.stage === actualStage &&
      (!r.encoder || r.encoder === (controlledEncoder ?? encoder)),
  )
  const methods = [...new Set(available.map((r) => r.method))]
  const runs = available.filter((r) => method === 'all' || r.method === method)
  const run = runs.find((r) => r.id === selected) ?? runs[0]
  return (
    <>
      <div className="toolbar">
        {!controlledEncoder && !['splits', 'detector'].includes(stage) && (
          <Select
            label="Encoder"
            value={encoder}
            onChange={(v) => {
              setEncoder(v)
              setSelected('')
              setMethod('all')
            }}
            options={['DINOv2', 'CLIP'].map((value) => ({
              value,
              label: value,
            }))}
          />
        )}
        {methods.length > 1 && (
          <Select
            label={stage === 'reduction' ? 'Representation' : 'Method'}
            value={method}
            onChange={(v) => {
              setMethod(v)
              setSelected('')
            }}
            options={[
              { value: 'all', label: 'All methods' },
              ...methods.map((value) => ({ value, label: value })),
            ]}
          />
        )}
        {runs.length > 1 && (
          <Select
            label="Artifact run"
            value={run.id}
            onChange={setSelected}
            options={runs.map((r, i) => ({
              value: r.id,
              label: `${r.method} · Run ${i + 1}${r.candidate ? ' · candidate' : ''}`,
            }))}
          />
        )}
        {run && <RunDetails run={run} />}
      </div>
      {!run ? (
        <EmptyArtifactState stage={actualStage} />
      ) : (
        <>
          <div className="run-summary">
            <h2>
              {run.method} · {run.id}
            </h2>
            <span className="micro muted">ARTIFACT · metrics REPORTED</span>
          </div>
          <MetricSummary
            items={[
              { label: 'Content coverage', value: run.coverage },
              ...Object.entries(run.metrics)
                .slice(0, 3)
                .map(([label, value]) => ({ label, value })),
            ]}
          />
          {stage === 'embeddings' && (
            <Panel title={`${run.encoder} representation`} meta="ARTIFACT">
              <MetricSummary
                items={[
                  { label: 'Dimensions', value: run.dimensions },
                  { label: 'Unique contents', value: run.coverage },
                ]}
              />
              <Notice>
                Embeddings represent unique contents. Full vectors stay outside
                the browser. Select Reduction to inspect an exported projection.
              </Notice>
            </Panel>
          )}
          {actualStage === 'similarity' && (
            <SimilarityEvidence key={run.id} run={run} />
          )}
          {['reduction', 'clustering'].includes(actualStage) && (
            <ArtifactExplorer key={run.id} snapshot={snapshot} run={run} />
          )}
          {actualStage === 'splits' && <SplitEvidence key={run.id} run={run} />}
          {actualStage === 'detector' && (
            <Notice>
              Reported detector evidence for this dataset and its compatible
              split only. Pilot ≠ final scientific result.
            </Notice>
          )}
        </>
      )}
    </>
  )
}
