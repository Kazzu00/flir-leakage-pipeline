import { useState } from 'react'
import { useSnapshot } from '@/data/provider'
import { Panel, Select, Notice } from '@/components/feedback/Primitives'
import { Scatter } from '../clustering/Scatter'
import { TechnicalDetailsDrawer } from '@/components/research/TechnicalDetailsDrawer'
import { Gallery } from '@/components/visualization/Gallery'
export function Embeddings() {
  const { data } = useSnapshot()
  const [encoder, setEncoder] = useState('DINOv2')
  const [selected, setSelected] = useState('demo-content-000')
  const [cluster, setCluster] = useState<number | null>(null)
  if (!data) return null
  const embedding = data.embeddings.find((e) => e.encoder === encoder)!
  const run = data.clusterings.find((r) => r.encoder === encoder)!
  const contents = run.points.filter(
    (p) => cluster === null || p.clusterId === cluster,
  )
  const point = contents.find((p) => p.contentId === selected) ?? contents[0]
  return (
    <>
      <div className="toolbar">
        <Select
          label="Image encoder"
          value={encoder}
          onChange={setEncoder}
          options={data.embeddings.map((e) => ({
            value: e.encoder,
            label: e.encoder,
          }))}
        />
        <Select
          label="Selected content"
          value={point.contentId}
          onChange={setSelected}
          options={contents.map((p) => ({
            value: p.contentId,
            label: p.contentId,
          }))}
        />
      </div>
      <div className="cluster-workspace">
        <Panel
          title="Reduced representation preview"
          meta={`DEMO · ${run.reduction}`}
        >
          <Scatter
            run={run}
            selected={cluster}
            after={false}
            splitMap={new Map()}
            onSelect={(id) => {
              setCluster(id)
              setSelected('')
            }}
            onContentSelect={setSelected}
          />
          <div className="cluster-buttons">
            <button onClick={() => setCluster(null)}>Show all contents</button>
            {run.clusters.map((c) => (
              <button
                key={c.id}
                onClick={() => {
                  setCluster(c.id)
                  setSelected('')
                }}
              >
                Cluster {c.id}
              </button>
            ))}
          </div>
        </Panel>
        <Panel title="Selected content" meta="DEMO">
          <p className="panel-body">
            {point.contentId} · {embedding.encoder} · {embedding.dimensions}{' '}
            dimensions
          </p>
          <TechnicalDetailsDrawer title="Feature-space identity">
            <dl className="details">
              <dt>Encoder</dt>
              <dd>{embedding.encoder}</dd>
              <dt>Representation</dt>
              <dd>{embedding.pooling}</dd>
              <dt>Dimensions</dt>
              <dd>{embedding.dimensions}</dd>
              <dt>Coverage</dt>
              <dd>{embedding.coverage} / 144 · DEMO</dd>
              <dt>Feature space</dt>
              <dd className="mono">{embedding.featureSpaceId}</dd>
              <dt>Selected content</dt>
              <dd className="mono">{point.contentId}</dd>
              <dt>Occurrence mapping</dt>
              <dd className="mono">{point.frameIds.join(', ')}</dd>
            </dl>
          </TechnicalDetailsDrawer>
        </Panel>
      </div>
      <Panel title="Selection gallery" meta="DEMO · images not connected">
        <Gallery
          points={contents}
          selected={point.contentId}
          onSelect={setSelected}
        />
      </Panel>
      <Notice>
        Both encoders use images only. Raw and L2 embeddings remain distinct.
        Coordinates are demo fixtures, not embeddings computed in this browser.
        Model revisions, processor metadata and array validation must come from
        a future verified adapter.
      </Notice>
    </>
  )
}
