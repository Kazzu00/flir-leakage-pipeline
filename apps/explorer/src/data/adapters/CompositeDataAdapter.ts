import { SnapshotSchema, type Run } from '@/contracts'
import type { LeakageSnapshot } from '@/contracts/leakage'
import type { DataAdapter } from './types'
import { MockDataAdapter } from './MockDataAdapter'
import { RealLeakageAdapter } from './RealLeakageAdapter'

export class CompositeDataAdapter implements DataAdapter {
  readonly id = 'mixed-leakage-v1'
  constructor(
    private readonly leakage: Pick<
      RealLeakageAdapter,
      'getLeakage'
    > = new RealLeakageAdapter(),
    private readonly demo: DataAdapter = new MockDataAdapter(),
  ) {}
  async getSnapshot(signal?: AbortSignal) {
    const [base, leakage] = await Promise.all([
      this.demo.getSnapshot(signal),
      this.leakage.getLeakage(signal),
    ])
    const runs: Run[] = leakage.runs.map((r) => ({
      id: r.id,
      module: 'organization',
      stage: r.stage,
      datasetId: r.datasetId,
      method: r.method,
      origin: 'artifact',
      status: r.status,
      verification: 'experimental',
      seed: r.seed,
      parameters: Object.fromEntries(
        Object.entries(r.parameters).filter(
          (v): v is [string, number] => v[1] !== null,
        ),
      ),
      metrics: Object.entries(r.metrics).map(([name, value]) => ({
        name,
        value,
        unit: '',
        origin: r.metricOrigin,
        caveat: 'Reported upstream; not recomputed by the explorer.',
      })),
      artifacts: [],
      caveat:
        'Consumed files validated during export. Scientific metrics are reported upstream; this is not an independent scientific verification.',
    }))
    return SnapshotSchema.parse({
      ...base,
      leakage,
      dataset: {
        id: leakage.dataset.id,
        name: 'Sampled-video dataset',
        origin: 'artifact',
        occurrences: leakage.dataset.occurrences,
        uniqueContents: leakage.dataset.uniqueContents,
        duplicateGroups: leakage.dataset.duplicateGroups,
        provenance:
          'Source-video membership and sampling grid; sequences and capture timestamps unknown.',
        verification: 'experimental',
      },
      runs: [...base.runs.filter((r) => r.module !== 'organization'), ...runs],
      embeddings: [],
      reductions: [],
      clusterings: [],
      splits: [],
      similarity: null,
      modules: base.modules.map((m) =>
        m.id !== 'organization'
          ? m
          : {
              ...m,
              status: 'experimental',
              evidence:
                'Local sanitized artifacts. Coverage is shown per stage.',
              stages: m.stages.map((s) => ({
                ...s,
                status:
                  s.id === 'dataset' ||
                  leakage.runs.some((r) => r.stage === stageKey(s.id))
                    ? 'complete'
                    : 'pending',
              })),
            },
      ),
    })
  }
}
function stageKey(
  id: string,
): LeakageSnapshot['runs'][number]['stage'] | string {
  return id === 'splitting' ? 'splits' : id
}
