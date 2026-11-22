import { SnapshotSchema } from '@/contracts'
import { fixture } from '@/data/mock/fixture'
import type { DataAdapter } from './types'
export class MockDataAdapter implements DataAdapter {
  readonly id = 'mock-v1'
  async getSnapshot(signal?: AbortSignal) {
    signal?.throwIfAborted()
    return SnapshotSchema.parse(structuredClone(fixture))
  }
}
