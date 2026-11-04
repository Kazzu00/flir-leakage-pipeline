import { useState } from 'react'
import { useSnapshot } from '@/data/provider'
import {
  Panel,
  Metrics,
  Notice,
  Select,
} from '@/components/feedback/Primitives'
import { TechnicalDetailsDrawer } from '@/components/research/TechnicalDetailsDrawer'
import { Embeddings } from '../embeddings/Embeddings'
import { Status } from '@/components/feedback/Status'
export function Reduction() {
  const { data } = useSnapshot()
  const [left, setLeft] = useState('demo-reduction-0')
  const [right, setRight] = useState('demo-reduction-1')
  if (!data) return null
  const options = data.reductions.map((r) => ({
    value: r.id,
    label: `${r.encoder} / ${r.method} · seed ${r.seed}`,
  }))
  return (
    <>
      <Embeddings />
      <TechnicalDetailsDrawer title="Compare demo reduction metadata">
        <div className="toolbar">
          <Select
            label="Candidate A"
            value={left}
            onChange={setLeft}
            options={options}
          />
          <Select
            label="Candidate B"
            value={right}
            onChange={setRight}
            options={options}
          />
        </div>
        <div className="two-columns">
          {[left, right].map((id, i) => {
            const r = data.reductions.find((r) => r.id === id)!
            return (
              <Panel key={i} title={`${r.encoder} / ${r.method}`} meta="DEMO">
                <dl className="details">
                  <dt>Run</dt>
                  <dd className="mono">{r.id}</dd>
                  <dt>Parameters</dt>
                  <dd className="mono">{JSON.stringify(r.parameters)}</dd>
                  <dt>Seed</dt>
                  <dd>{r.seed}</dd>
                  <dt>Selected candidate</dt>
                  <dd>
                    {r.selectedCandidate
                      ? 'Demo selection (not a scientific recommendation)'
                      : 'No'}
                  </dd>
                  <dt>Verification</dt>
                  <dd>
                    <Status state={r.verification} />
                  </dd>
                </dl>
                <Metrics metrics={r.metrics} />
              </Panel>
            )
          })}
        </div>
      </TechnicalDetailsDrawer>
      <Notice>
        Preservation metrics are synthetic examples. A separated 2D cloud is not
        evidence of a meaningful cluster. The committed methods are t-SNE and
        PaCMAP; no UMAP is implied.
      </Notice>
    </>
  )
}
