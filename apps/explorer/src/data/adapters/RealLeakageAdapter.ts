import type { LeakageSnapshot } from '@/contracts/leakage'
import { LeakageSnapshotSchema } from '@/contracts/leakage-v2'

export type DataFailure =
  | 'snapshot-missing'
  | 'snapshot-malformed'
  | 'schema-mismatch'
  | 'identity-mismatch'
  | 'snapshot-unavailable'
export class SnapshotError extends Error {
  constructor(readonly kind: DataFailure) {
    super(kind)
    this.name = 'SnapshotError'
  }
}
export function parseLeakageSnapshot(input: unknown): LeakageSnapshot {
  const result = LeakageSnapshotSchema.safeParse(input)
  if (!result.success)
    throw new SnapshotError(
      result.error.issues.every((i) => i.code === 'custom')
        ? 'identity-mismatch'
        : 'schema-mismatch',
    )
  return result.data
}
export class RealLeakageAdapter {
  readonly id = 'artifact-leakage-v1'
  constructor(private readonly url = '/runtime/leakage-snapshot.json') {}
  async getLeakage(signal?: AbortSignal): Promise<LeakageSnapshot> {
    // Runtime snapshots may contain restricted research metadata: same origin only.
    if (
      !this.url.startsWith('/') ||
      this.url.startsWith('//') ||
      this.url.includes('\\')
    )
      throw new SnapshotError('snapshot-unavailable')
    let response: Response
    try {
      response = await fetch(this.url, {
        signal,
        cache: 'no-store',
        credentials: 'same-origin',
      })
    } catch {
      throw new SnapshotError('snapshot-unavailable')
    }
    if (response.status === 404) throw new SnapshotError('snapshot-missing')
    if (!response.ok) throw new SnapshotError('snapshot-unavailable')
    const raw = await response.text()
    if (raw.length > 80 * 1024 * 1024)
      throw new SnapshotError('schema-mismatch')
    let parsed: unknown
    try {
      parsed = JSON.parse(raw)
    } catch {
      throw new SnapshotError(
        raw.trimStart().startsWith('<')
          ? 'snapshot-missing'
          : 'snapshot-malformed',
      )
    }
    return parseLeakageSnapshot(parsed)
  }
}
