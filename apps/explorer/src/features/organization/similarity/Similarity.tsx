import { useState } from 'react'
import { useSnapshot } from '@/data/provider'
import { Panel, Notice, Select } from '@/components/feedback/Primitives'
import { Chart } from '@/components/visualization/Chart'
import { SampleImage } from '@/components/visualization/Gallery'
export function Similarity() {
  const { data } = useSnapshot()
  const [index, setIndex] = useState('0')
  if (!data?.similarity) return null
  const pair = data.similarity.pairs[Number(index)]
  return (
    <>
      <Panel
        title="Cosine similarity distribution"
        meta="DEMO · L2-normalized content embeddings"
      >
        <Chart
          height={250}
          label="Demo similarity histogram; accessible counts below"
          option={{
            grid: { left: 50, right: 25, top: 30, bottom: 40 },
            xAxis: {
              type: 'category',
              data: data.similarity.distribution.map((d) => d.bin),
              name: 'Cosine',
            },
            yAxis: { type: 'value', name: 'Pairs' },
            series: [
              {
                type: 'bar',
                data: data.similarity.distribution.map((d) => d.count),
                itemStyle: { color: '#73d9be' },
                barMaxWidth: 30,
              },
            ],
          }}
        />
        <details className="panel-body">
          <summary>Histogram values · DEMO</summary>
          <p>
            {data.similarity.distribution
              .map((d) => `${d.bin}: ${d.count} pairs`)
              .join(' · ')}
          </p>
        </details>
      </Panel>
      <div className="toolbar">
        <Select
          label="Nearest neighbor pair · query demo-content-000"
          value={index}
          onChange={setIndex}
          options={data.similarity.pairs.map((p, i) => ({
            value: String(i),
            label: `${p.b} · cosine ${p.cosine.toFixed(3)}`,
          }))}
        />
      </div>
      <Panel
        title="Pair inspection"
        meta={`DEMO · cosine ${pair.cosine.toFixed(3)}`}
      >
        <div className="comparison-grid">
          {[pair.a, pair.b].map((id) => (
            <figure key={id}>
              <SampleImage />
              <figcaption className="mono">{id}</figcaption>
            </figure>
          ))}
        </div>
        <dl className="details">
          <dt>Provenance</dt>
          <dd>
            Same source video: {String(pair.sameSourceVideo)} · sampling index
            gap: {pair.sampleGap}
          </dd>
          <dt>Capture timestamp / validated sequence</dt>
          <dd>Unavailable / unknown</dd>
        </dl>
      </Panel>
      <Notice>
        Exact duplicate ≠ visual neighbor ≠ temporal neighbor. Source video ≠
        sequence. Sampling-grid timestamp ≠ verified capture timestamp. High
        cosine alone does not confirm leakage.
      </Notice>
    </>
  )
}
