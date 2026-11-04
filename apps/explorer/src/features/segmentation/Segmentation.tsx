import { useState } from 'react'
import { NavLink, useParams, Link } from 'react-router-dom'
import { useSnapshot } from '@/data/provider'
import {
  PageTitle,
  Panel,
  Notice,
  Metrics,
  Select,
} from '@/components/feedback/Primitives'
import { ModuleOverview } from '@/features/overview/ModuleOverview'
import { SampleImage } from '@/components/visualization/Gallery'
import { Status } from '@/components/feedback/Status'
const stages = [
  ['', 'Overview'],
  ['points', 'Pseudo-annotations'],
  ['model', 'Configuration'],
  ['training', 'Training / runs'],
  ['predictions', 'Prediction explorer'],
  ['evaluation', 'Evaluation'],
]
export function Segmentation() {
  const { stage = '' } = useParams()
  const { data } = useSnapshot()
  const [prediction, setPrediction] = useState('demo-prediction-0')
  const [view, setView] = useState('Side by side')
  const [opacity, setOpacity] = useState(45)
  if (!data) return null
  const run = data.segmentation
  const pred = run.predictions.find((p) => p.id === prediction)!
  if (!stages.some(([id]) => id === stage))
    return (
      <>
        <h1>Stage unavailable</h1>
        <Link to="/segmentation">Module overview</Link>
      </>
    )
  return (
    <>
      <PageTitle
        eyebrow="MODULE 03 / PANOPTIC SEGMENTATION"
        title={
          stage
            ? stages.find(([id]) => id === stage)![1]
            : 'Panoptic segmentation'
        }
        description="Inspect model configuration, prediction layers and the limits of evaluation evidence."
      />
      <nav className="tabs-nav" aria-label="Segmentation stages">
        {stages.map(([id, name]) => (
          <NavLink end key={id} to={`/segmentation${id ? `/${id}` : ''}`}>
            {name}
          </NavLink>
        ))}
      </nav>
      <Notice>
        <strong>Inconsistent source status.</strong> {run.caveat} A box-based
        comparison is not a dense-mask PQ evaluation.
      </Notice>
      {!stage && <ModuleOverview id="segmentation" />}
      {(stage === 'model' || stage === 'training' || !stage) && (
        <Panel
          title="Model and run configuration"
          meta="DEMO · upstream status inconsistent"
        >
          <dl className="details">
            <dt>Model / variant</dt>
            <dd>
              {run.method} / {run.modelVariant}
            </dd>
            <dt>Configuration</dt>
            <dd className="mono">{JSON.stringify(run.parameters)}</dd>
            <dt>Training state</dt>
            <dd>
              <Status state="inconsistent" /> README, code and reports disagree;
              no live job connected.
            </dd>
            <dt>Context-fusion ablation</dt>
            <dd>
              Comparison unavailable; not a validated architectural improvement.
            </dd>
          </dl>
          <Link
            className="row-link panel-body"
            style={{ display: 'block' }}
            to="/experiments?module=segmentation"
          >
            Inspect segmentation run →
          </Link>
        </Panel>
      )}
      {stage === 'points' && (
        <Panel title="Supervision provenance">
          <div className="method-flow">
            <div>Bounding boxes</div>
            <span>→</span>
            <div>Interior points</div>
            <span>→</span>
            <div>Pseudo-masks</div>
          </div>
          <Notice>
            Pseudo-masks are training signals, not independent evaluation ground
            truth. The actual mask-supervision path requires upstream
            validation.
          </Notice>
        </Panel>
      )}
      {(stage === 'predictions' || !stage) && (
        <>
          <div className="toolbar">
            <Select
              label="Prediction · DEMO"
              value={prediction}
              onChange={setPrediction}
              options={run.predictions.map((p) => ({
                value: p.id,
                label: p.contentId,
              }))}
            />
            <Select
              label="Comparison layout"
              value={view}
              onChange={setView}
              options={['Side by side', 'Overlay'].map((v) => ({
                value: v,
                label: v,
              }))}
            />
            {view === 'Overlay' && (
              <label className="select-field">
                <span>Overlay opacity · {opacity}%</span>
                <input
                  aria-label="Overlay opacity"
                  type="range"
                  min={0}
                  max={100}
                  value={opacity}
                  onChange={(e) => setOpacity(Number(e.target.value))}
                />
              </label>
            )}
          </div>
          <Panel
            title={`Prediction inspection · ${pred.contentId}`}
            meta="SAMPLE · placeholder only"
          >
            <div className="comparison-grid">
              {view === 'Side by side' ? (
                <>
                  <figure>
                    <SampleImage label="Original image unavailable" />
                    <figcaption>Original · {pred.contentId}</figcaption>
                  </figure>
                  <figure>
                    <SampleImage
                      label="Panoptic prediction unavailable"
                      overlay
                      opacity={0.45}
                    />
                    <figcaption>
                      Illustrative layer · not a model prediction
                    </figcaption>
                  </figure>
                </>
              ) : (
                <figure style={{ gridColumn: '1 / -1' }}>
                  <SampleImage
                    label="Original image unavailable"
                    overlay
                    opacity={opacity / 100}
                  />
                  <figcaption>
                    Illustrative overlay · opacity {opacity}%
                  </figcaption>
                </figure>
              )}
            </div>
            <Notice>{pred.caveat}</Notice>
            <Metrics metrics={pred.metrics} />
          </Panel>
        </>
      )}
      {(stage === 'evaluation' || !stage) && (
        <>
          <Panel
            title="Panoptic evaluation"
            meta="VALIDATED ARTIFACTS NOT CONNECTED"
          >
            <Metrics metrics={run.metrics} />
          </Panel>
          <Panel title="Per-class evaluation">
            <div className="table-scroll">
              <table>
                <thead>
                  <tr>
                    <th>Class</th>
                    <th>IoU</th>
                    <th>F1</th>
                    <th>Recall</th>
                  </tr>
                </thead>
                <tbody>
                  {run.classes.map((c) => (
                    <tr key={c.name}>
                      <td>{c.name}</td>
                      {c.metrics.map((m) => (
                        <td key={m.name}>{m.value ?? 'Unavailable'}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Panel>
          <Panel title="Baseline comparison">
            <div className="panel-body">
              <Status state="pending" />
              <p style={{ marginTop: 12 }}>
                A controlled comparison requires compatible splits, class
                mapping, independent ground truth and an explicit mask-to-box
                protocol. No comparable validated results are connected.
              </p>
            </div>
          </Panel>
        </>
      )}
    </>
  )
}
