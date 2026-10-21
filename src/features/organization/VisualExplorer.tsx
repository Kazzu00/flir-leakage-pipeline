import { useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { useRepresentationSnapshot } from '@/data/representation'
import { Select, Notice } from '@/components/feedback/Primitives'
import { ArtifactStage } from './ArtifactStage'
import { Clustering } from './clustering/Clustering'
const views = {
  embeddings: true,
  similarity: true,
  reduction: true,
  clustering: true,
}
export function VisualExplorer() {
  const [params, setParams] = useSearchParams()
  const [encoder, setEncoder] = useState('DINOv2')
  const requested = params.get('view') ?? 'reduction'
  const view =
    requested in views ? (requested as keyof typeof views) : 'reduction'
  const evidence = useRepresentationSnapshot(view !== 'clustering')
  const data = evidence.data
  return (
    <>
      <div className="toolbar explorer-toolbar">
        {view !== 'clustering' && data && (
          <Select
            label="Encoder"
            value={encoder}
            onChange={setEncoder}
            options={['DINOv2', 'CLIP'].map((value) => ({
              value,
              label: value,
            }))}
          />
        )}
        <Select
          label="View"
          value={view}
          onChange={(value) => setParams({ view: value })}
          options={Object.keys(views).map((value) => ({
            value,
            label: value[0].toUpperCase() + value.slice(1),
          }))}
        />
        <span className="muted">Visual similarity ≠ temporal continuity</span>
      </div>
      {view === 'clustering' ? (
        <Clustering />
      ) : data ? (
        <ArtifactStage
          key={`${view}-${encoder}`}
          snapshot={data}
          stage={view}
          encoder={encoder}
        />
      ) : (
        <Notice>
          {evidence.isPending
            ? 'Loading real representation evidence...'
            : 'UNAVAILABLE: real representation snapshot could not be loaded or validated. No synthetic substitute.'}
        </Notice>
      )}
    </>
  )
}
