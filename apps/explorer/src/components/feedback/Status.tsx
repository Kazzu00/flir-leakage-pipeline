import type { State } from '@/contracts'
export function Status({ state }: { state: State }) {
  return (
    <span className={`status status-${state}`}>
      <span aria-hidden="true">
        {state === 'verified'
          ? '✓'
          : state === 'inconsistent' || state === 'invalid'
            ? '!'
            : '●'}
      </span>
      {state}
    </span>
  )
}
