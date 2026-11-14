import type { ReactNode } from 'react'
import { Panel, Notice } from '@/components/feedback/Primitives'
export function ArtifactStatus({ state }: { state: string }) {
  return (
    <span className={`evidence-state evidence-${state}`}>
      <span aria-hidden="true">
        {state === 'pending' || state === 'unavailable' ? '○' : '◇'}
      </span>{' '}
      <span>{state.replaceAll('_', ' ')}</span>
    </span>
  )
}
export function EmptyArtifactState({ stage }: { stage: string }) {
  return (
    <Panel title={`${stage}: pending`} meta="UNAVAILABLE">
      <Notice>
        No compatible {stage} artifact was exported for this dataset. Supply an
        explicit artifact to inspect its evidence. No synthetic results are
        substituted.
      </Notice>
    </Panel>
  )
}
export function MetricSummary({
  items,
}: {
  items: { label: string; value: ReactNode }[]
}) {
  return (
    <dl className="metric-summary">
      {items.slice(0, 5).map(({ label, value }) => (
        <div key={label}>
          <dt>{label}</dt>
          <dd>{value ?? 'Unavailable'}</dd>
        </div>
      ))}
    </dl>
  )
}
