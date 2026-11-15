import type { Snapshot } from '@/contracts'
/** Adapters normalize external artifacts and validate before crossing into the UI. */
export interface DataAdapter {
  readonly id: string
  getSnapshot(signal?: AbortSignal): Promise<Snapshot>
}
