import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { MemoryRouter } from 'react-router-dom'
import fixture from './fixtures/leakage-synthetic.json'
import { LeakageSnapshotV1 } from '@/contracts/leakage'
import {
  RealLeakageAdapter,
  parseLeakageSnapshot,
} from '@/data/adapters/RealLeakageAdapter'
import { CompositeDataAdapter } from '@/data/adapters/CompositeDataAdapter'
import { MockDataAdapter } from '@/data/adapters/MockDataAdapter'
import { ArtifactStage } from '@/features/organization/ArtifactStage'
import { DataAdapterContext } from '@/data/provider'
import { AppShell } from '@/components/layout/AppShell'
import {
  withClusteringAndSplit,
  largeSyntheticSnapshot,
} from './artifact-fixtures'

afterEach(() => vi.unstubAllGlobals())
const input = () => structuredClone(fixture)
const validated = () => parseLeakageSnapshot(input())
describe('Real artifact boundary', () => {
  it('validates 9000 contents without sampling or discarding row mappings', () => {
    const snapshot = parseLeakageSnapshot(largeSyntheticSnapshot())
    expect(snapshot.runs[2].coordinates).toHaveLength(9000)
    expect(snapshot.runs[0].contentIndex[8999].embeddingRow).toBe(8999)
  })
  it('rejects fractured groups and retains singleton noise membership', () => {
    const good = withClusteringAndSplit()
    expect(parseLeakageSnapshot(good).runs.at(-2)?.labels[2].clusterId).toBe(-1)
    good.runs
      .at(-1)!
      .assignments.find((a) => a.contentId === 'content-000002')!.split =
      'validation'
    expect(() => parseLeakageSnapshot(good)).toThrow('identity-mismatch')
  })
  it('retains deterministic DEMO independently', async () => {
    const demo = await new MockDataAdapter().getSnapshot()
    expect(demo.dataset.origin).toBe('mock')
    expect(demo.leakage).toBeUndefined()
    expect(demo.clusterings.length).toBeGreaterThan(0)
  })
  it('accepts the exact synthetic Python-exported contract and both encoders', () => {
    const data = validated()
    expect(
      data.runs
        .filter((r) => r.stage === 'embeddings')
        .map((r) => r.dimensions),
    ).toEqual([384, 512])
    expect(data.contents[0].occurrences).toHaveLength(2)
  })
  it('composes artifact M02 with demo M01 and M03 without mock organization records', async () => {
    const adapter = new CompositeDataAdapter({
      getLeakage: async () => validated(),
    })
    const data = await adapter.getSnapshot()
    expect(data.dataset.origin).toBe('artifact')
    expect(
      data.runs
        .filter((r) => r.module === 'organization')
        .every((r) => r.origin === 'artifact'),
    ).toBe(true)
    expect(
      data.runs
        .filter((r) => r.module !== 'organization')
        .every((r) => r.origin === 'mock'),
    ).toBe(true)
    expect(data.clusterings).toEqual([])
    expect(data.splits).toEqual([])
    expect(data.similarity).toBeNull()
  })
  it('fetches a valid real snapshot', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(new Response(JSON.stringify(input()))),
    )
    expect((await new RealLeakageAdapter().getLeakage()).dataset.origin).toBe(
      'artifact',
    )
  })
  it.each([
    [404, '', 'snapshot-missing'],
    [200, '{bad', 'snapshot-malformed'],
    [200, '<html>Vite fallback</html>', 'snapshot-missing'],
    [200, '{"schemaVersion":99}', 'schema-mismatch'],
    [500, '', 'snapshot-unavailable'],
  ])('controls fetch errors (%s)', async (status, body, kind) => {
    vi.stubGlobal(
      'fetch',
      vi
        .fn()
        .mockResolvedValue(
          new Response(body as string, { status: status as number }),
        ),
    )
    await expect(new RealLeakageAdapter().getLeakage()).rejects.toMatchObject({
      kind,
      message: kind,
    })
  })
  it('rejects cross-origin snapshot URLs without making requests', async () => {
    const fetcher = vi.fn()
    vi.stubGlobal('fetch', fetcher)
    await expect(
      new RealLeakageAdapter('https://example.com/private').getLeakage(),
    ).rejects.toMatchObject({ kind: 'snapshot-unavailable' })
    expect(fetcher).not.toHaveBeenCalled()
  })
  it('rejects dataset and feature identity mismatches', () => {
    const data = input()
    data.runs[2].datasetId = 'dataset-another'
    expect(() => parseLeakageSnapshot(data)).toThrow('identity-mismatch')
    const other = input()
    other.runs[2].featureSpaceId = 'feature-000002'
    expect(() => parseLeakageSnapshot(other)).toThrow('identity-mismatch')
  })
  it('does not infer sequences or collapse multisource content', () => {
    const data = validated()
    expect(
      new Set(data.contents[0].occurrences.map((o) => o.sourceVideo)).size,
    ).toBe(2)
    expect(
      data.contents[0].occurrences.every(
        (o) => o.sequenceId === null && o.captureTimestamp === null,
      ),
    ).toBe(true)
    const fabricated = input() as unknown as {
      contents: { occurrences: { sequenceId: string | null }[] }[]
    }
    fabricated.contents[0].occurrences[0].sequenceId = 'video-000001'
    expect(LeakageSnapshotV1.safeParse(fabricated).success).toBe(false)
  })
  it('renders pending clustering and no unavailable split assignments', () => {
    const { rerender } = render(
      <ArtifactStage snapshot={validated()} stage="clustering" />,
    )
    expect(
      screen.getByRole('heading', { name: 'clustering: pending' }),
    ).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'AFTER SPLIT' })).toBeNull()
    rerender(<ArtifactStage snapshot={validated()} stage="splits" />)
    expect(
      screen.getByRole('heading', { name: 'splits: pending' }),
    ).toBeInTheDocument()
    expect(
      screen.queryByRole('heading', { name: 'Persisted split assignments' }),
    ).toBeNull()
  })
  it('renders actual persisted coordinates and unknown sequence state', () => {
    render(<ArtifactStage snapshot={validated()} stage="reduction" />)
    expect(
      screen.getByRole('img', {
        name: 'Artifact scatter: 3 persisted coordinates',
      }),
    ).toBeInTheDocument()
    expect(screen.getByText(/coordinates: -4.5, 12/)).toBeInTheDocument()
    expect(
      screen.getByText('Sequence: unknown. Capture timestamp: unavailable.'),
    ).toBeInTheDocument()
    expect(document.body.textContent).not.toMatch(
      /private-drone|private_person|C:\/|\.jpg|demo-content/,
    )
  })
  it('rejects private paths and unexpected raw metadata before rendering', () => {
    const data = { ...input(), privatePath: 'C:\\Users\\someone\\artifacts' }
    expect(() => parseLeakageSnapshot(data)).toThrow('schema-mismatch')
    const other = input()
    other.contents[0].contentId = '/home/research/private.png'
    expect(() => parseLeakageSnapshot(other)).toThrow('schema-mismatch')
  })
  it('shows MIXED in the workspace', async () => {
    const adapter = new CompositeDataAdapter({
      getLeakage: async () => validated(),
    })
    render(
      <QueryClientProvider
        client={
          new QueryClient({ defaultOptions: { queries: { retry: false } } })
        }
      >
        <DataAdapterContext.Provider value={adapter}>
          <MemoryRouter>
            <AppShell />
          </MemoryRouter>
        </DataAdapterContext.Provider>
      </QueryClientProvider>,
    )
    expect(await screen.findByText('MIXED WORKSPACE')).toBeInTheDocument()
    expect(screen.getByText('MIXED', { exact: true })).toBeInTheDocument()
  })
})
