import { useState } from 'react'
import { NavLink, useParams, Link } from 'react-router-dom'
import { useSnapshot } from '@/data/provider'
import {
  PageTitle,
  Panel,
  Select,
  Metrics,
  Notice,
} from '@/components/feedback/Primitives'
import { ModuleOverview } from '@/features/overview/ModuleOverview'
import { SampleImage } from '@/components/visualization/Gallery'
import { Chart } from '@/components/visualization/Chart'
import { Status } from '@/components/feedback/Status'
const stages = [
  ['', 'Overview'],
  ['frames', 'Videos / frames'],
  ['hud', 'HUD masks'],
  ['inpainting', 'HUD removal'],
  ['denoising', 'Denoising'],
  ['quality', 'Quality evaluation'],
]
export function Preprocessing() {
  const { stage = '' } = useParams()
  const { data } = useSnapshot()
  const [video, setVideo] = useState('demo-video-1')
  const [method, setMethod] = useState('Noise2Noise')
  const [index, setIndex] = useState(0)
  if (!data) return null
  const run = data.runs.find((r) => r.module === 'preprocessing')!
  if (!stages.some(([s]) => s === stage))
    return (
      <>
        <h1>Stage not found</h1>
        <Link to="/preprocessing">Module overview</Link>
      </>
    )
  return (
    <>
      <PageTitle
        eyebrow="MODULE 01 / PREPROCESSING"
        title={
          stages.find(([s]) => s === stage)?.[1] === 'Overview'
            ? 'Preprocessing'
            : (stages.find(([s]) => s === stage)?.[1] ?? 'Preprocessing')
        }
        description="Inspect video preparation, restoration methods and quality evidence."
      />
      <nav className="tabs-nav" aria-label="Preprocessing stages">
        {stages.map(([s, n]) => (
          <NavLink end key={s} to={`/preprocessing${s ? `/${s}` : ''}`}>
            {n}
          </NavLink>
        ))}
      </nav>
      {!stage && <ModuleOverview id="preprocessing" />}
      <div className="toolbar">
        <Select
          label="Source video · DEMO"
          value={video}
          onChange={setVideo}
          options={[1, 2, 3].map((i) => ({
            value: `demo-video-${i}`,
            label: `Demo source video ${i}`,
          }))}
        />
        <Select
          label="Comparison method"
          value={method}
          onChange={setMethod}
          options={[
            'Noise2Noise',
            'Noise2Void',
            'Neighbor2Neighbor',
            'Blind2Unblind',
          ].map((m) => ({ value: m, label: m }))}
        />
        <label className="select-field">
          <span>Sampling index · {index}</span>
          <input
            aria-label="Sampling index"
            type="range"
            min={0}
            max={47}
            value={index}
            onChange={(e) => setIndex(Number(e.target.value))}
          />
        </label>
      </div>
      <Panel
        title={`${video} · sample ${index}`}
        meta="DEMO · visual assets unavailable"
      >
        <div className="comparison-grid">
          <figure>
            <SampleImage label="Original frame unavailable" />
            <figcaption>
              Original · {video} / sample {index}
            </figcaption>
          </figure>
          <figure>
            <SampleImage
              label={
                stage === 'hud'
                  ? 'HUD mask unavailable'
                  : stage === 'inpainting'
                    ? 'Cleaned frame unavailable'
                    : `${method} output unavailable`
              }
            />
            <figcaption>
              {stage === 'hud'
                ? 'HUD segmentation'
                : stage === 'inpainting'
                  ? 'ProPainter inpainting'
                  : method}{' '}
              · synthetic comparison slot
            </figcaption>
          </figure>
        </div>
      </Panel>
      <div className="two-columns">
        <Panel title="Processing record" meta="DEMO">
          <dl className="details">
            <dt>Run</dt>
            <dd className="mono">{run.id}</dd>
            <dt>Configuration</dt>
            <dd>Mode: sin_hud · learning rate 0.0001 · 15 epochs</dd>
            <dt>Fixture method</dt>
            <dd>Noise2Noise (metrics below only describe this demo fixture)</dd>
            <dt>Execution / verification</dt>
            <dd>
              <Status state={run.status} /> <Status state={run.verification} />
            </dd>
          </dl>
        </Panel>
        <Panel title="Quality summary · Noise2Noise" meta="ALL VALUES DEMO">
          <Metrics metrics={run.metrics} />
        </Panel>
      </div>
      <Panel
        title="Quality across the sampling grid"
        meta="NIQE · synthetic per-sample series"
      >
        <Chart
          label="Demo NIQE curves for original, Noise2Noise and Noise2Void over 24 sample indices"
          option={{
            tooltip: { trigger: 'axis' },
            legend: {
              data: ['Original', 'Noise2Noise', 'Noise2Void'],
              textStyle: { color: '#94a9af' },
              bottom: 0,
            },
            grid: { left: 50, right: 25, top: 25, bottom: 60 },
            xAxis: {
              type: 'category',
              name: 'Sample index',
              nameLocation: 'middle',
              nameGap: 28,
              data: data.quality.map((p) => p.index),
            },
            yAxis: { type: 'value', name: 'NIQE' },
            series: [
              {
                name: 'Original',
                type: 'line',
                data: data.quality.map((p) => p.original),
                lineStyle: { type: 'dashed' },
                itemStyle: { color: '#edc583' },
                symbol: 'diamond',
              },
              {
                name: 'Noise2Noise',
                type: 'line',
                data: data.quality.map((p) => p.n2n),
                itemStyle: { color: '#73d9be' },
                symbol: 'circle',
              },
              {
                name: 'Noise2Void',
                type: 'line',
                data: data.quality.map((p) => p.n2v),
                itemStyle: { color: '#8cb7ee' },
                symbol: 'rect',
              },
            ],
          }}
        />
        <details className="panel-body">
          <summary>Accessible curve data · DEMO</summary>
          <table>
            <thead>
              <tr>
                <th>Index</th>
                <th>Original</th>
                <th>N2N</th>
                <th>N2V</th>
              </tr>
            </thead>
            <tbody>
              {data.quality.map((p) => (
                <tr key={p.index}>
                  <td>{p.index}</td>
                  <td>{p.original.toFixed(2)}</td>
                  <td>{p.n2n.toFixed(2)}</td>
                  <td>{p.n2v.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </details>
      </Panel>
      <Notice>
        These curves demonstrate the interface; source XLSX summaries do not
        establish per-frame trajectories. Lower NIQE alone does not identify a
        best method. Source video is not a validated sequence.
      </Notice>
      <Link className="row-link" to="/experiments?module=preprocessing">
        Inspect preprocessing experiments →
      </Link>
    </>
  )
}
