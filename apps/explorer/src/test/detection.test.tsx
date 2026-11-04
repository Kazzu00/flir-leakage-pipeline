import { describe, it, expect, vi, afterEach } from 'vitest'
import { webcrypto } from 'node:crypto'
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import {
  DetectionExportSchema,
  contractFiles,
} from '@/features/detection/schema'
import {
  DetectionContractError,
  loadDetectionExport,
  validateDetectionFiles,
} from '@/features/detection/adapter'
import {
  DetectionEvaluation,
  DetectionReady,
} from '@/features/detection/DetectionEvaluation'
import {
  AssociationExplorer,
  Variability,
  TestComposition,
} from '@/features/detection/DetectionSections'
import { associationOption } from '@/features/detection/presentation'
import { isDetectionRoute } from '@/features/detection/route'
import { syncDetection } from '../../tools/sync-detection'
import {
  detectionDirectory,
  detectionFiles,
  detectionFixture,
} from './detection-fixture'

afterEach(() => vi.unstubAllGlobals())
const urls = Object.fromEntries(
  contractFiles.map((file) => [file, `/detection/${file}`]),
)
function fixtureFetch() {
  const files = detectionFiles()
  return vi.fn<typeof fetch>(async (url) => {
    const name =
      String(url).split('/snapshot/')[1]?.split('?')[0] ??
      String(url).replace('/detection/', '')
    return new Response(new Uint8Array(files[name]))
  })
}
function renderQuery() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false, gcTime: 0 } },
  })
  return render(
    <QueryClientProvider client={client}>
      <DetectionEvaluation />
    </QueryClientProvider>,
  )
}
describe('detection-export-v1 integrity', () => {
  it('loads manifest and summary with byte checksums and complete counts', async () => {
    vi.stubGlobal('crypto', webcrypto)
    const fetcher = fixtureFetch()
    const d = await loadDetectionExport(urls, undefined, fetcher)
    expect(fetcher).toHaveBeenCalledTimes(11)
    expect(d.manifest.state).toBe('COMPLETE_CONTROLLED_COMPARISON')
    expect(d.manifest.source_commit).not.toBe('0a4ea1f')
    expect(d.summary.protocol.plan_id).toBe(d.manifest.plan_id)
    expect(d.runs).toHaveLength(48)
    expect(d.splits).toHaveLength(16)
    expect(d.strategies).toHaveLength(4)
    expect(d.associations).toHaveLength(7)
    expect(
      d.strategies.find((s) => s.strategy === 'historical')?.std_between_splits,
    ).toBeNull()
  })
  it('rejects changed bytes, unsupported versions and unfinished experiments', async () => {
    vi.stubGlobal('crypto', webcrypto)
    const files = detectionFiles()
    files['runs.json'][10] ^= 1
    await expect(validateDetectionFiles(files)).rejects.toThrow(
      'Checksum mismatch',
    )
    for (const change of [
      { schema_version: 'v2' },
      { state: 'PARTIAL' },
      { scientific_result: false },
      { generated_from_verified_artifacts: false },
    ]) {
      const d = detectionFixture()
      expect(
        DetectionExportSchema.safeParse({
          ...d,
          manifest: { ...d.manifest, ...change },
        }).success,
      ).toBe(false)
    }
  })
  it('rejects missing runs, duplicated splits and mismatched references', () => {
    const d = detectionFixture()
    d.runs.pop()
    expect(DetectionExportSchema.safeParse(d).success).toBe(false)
    const duplicate = detectionFixture()
    duplicate.splits[1] = duplicate.splits[0]
    expect(DetectionExportSchema.safeParse(duplicate).success).toBe(false)
    const foreign = detectionFixture()
    foreign.support[0].split_space_id = 'foreign'
    expect(DetectionExportSchema.safeParse(foreign).success).toBe(false)
    const classId = detectionFixture()
    expect(
      DetectionExportSchema.safeParse({
        ...classId,
        classes: [
          { ...classId.classes[0], class_id: 99 },
          ...classId.classes.slice(1),
        ],
      }).success,
    ).toBe(false)
  })
  it('rejects zero replacing undefined historical variance', () => {
    const d = detectionFixture()
    d.strategies.find((s) => s.strategy === 'historical')!.std_between_splits =
      0
    expect(DetectionExportSchema.safeParse(d).success).toBe(false)
  })
  it('rejects repeated detector observations masquerading as association splits', () => {
    const d = detectionFixture()
    d.associations[0].points[1] = d.associations[0].points[0]
    expect(DetectionExportSchema.safeParse(d).success).toBe(false)
  })
  it('rejects relabeling a pre-specified association', () => {
    const d = detectionFixture()
    d.associations.find(
      (a) => a.association_id === 'temporal_at5',
    )!.residual_metric = 'clip_nn_mean'
    expect(DetectionExportSchema.safeParse(d).success).toBe(false)
  })
  it('returns missing data without demo fallback', async () => {
    await expect(
      loadDetectionExport(
        urls,
        undefined,
        vi.fn(async () => new Response('', { status: 404 })),
      ),
    ).rejects.toMatchObject({ kind: 'missing' })
  })
  it('sync is deterministic, copies only the allowlist, and validates before writes', async () => {
    vi.stubGlobal('crypto', webcrypto)
    const root = await mkdtemp(join(tmpdir(), 'flir-detection-test-'))
    const destination = join(root, 'snapshot')
    try {
      const publication = {
        branch: 'feat/hypatia-detector-final-report',
        commit: '0a4ea1f',
      }
      await syncDetection(detectionDirectory, destination, publication)
      const before = await readFile(join(root, 'snapshot-provenance.json'))
      await syncDetection(detectionDirectory, destination, publication)
      expect(await readFile(join(root, 'snapshot-provenance.json'))).toEqual(
        before,
      )
      const corruptSource = join(root, 'source')
      await syncDetection(detectionDirectory, corruptSource, publication)
      await writeFile(join(corruptSource, 'runs.json'), '[]')
      await expect(
        syncDetection(corruptSource, destination, publication),
      ).rejects.toThrow('Checksum mismatch')
      expect(await readFile(join(destination, 'runs.json'))).toEqual(
        await readFile(join(detectionDirectory, 'runs.json')),
      )
      for (const file of contractFiles)
        expect(await readFile(join(destination, file))).toEqual(
          await readFile(join(detectionDirectory, file)),
        )
      await expect(
        syncDetection(detectionDirectory, destination, {
          ...publication,
          commit: '../unsafe',
        }),
      ).rejects.toThrow()
      expect(await readFile(join(root, 'snapshot-provenance.json'))).toEqual(
        before,
      )
    } finally {
      await rm(root, { recursive: true, force: true })
    }
  })
})
describe('detector presentation', () => {
  it('shows loading state', () => {
    vi.stubGlobal('fetch', () => new Promise(() => {}))
    renderQuery()
    expect(screen.getByRole('status')).toHaveTextContent(
      'Loading verified detector results',
    )
  })
  it('shows unavailable instead of demo on missing contract', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => new Response('', { status: 404 })),
    )
    renderQuery()
    expect(
      await screen.findByRole('heading', {
        name: 'Verified detector results are not available.',
      }),
    ).toBeVisible()
    expect(screen.queryByText('REAL / VERIFIED')).toBeNull()
  })
  it('shows a validation error with disclosed details', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => new Response('{}')),
    )
    renderQuery()
    expect(
      await screen.findByRole('heading', {
        name: 'Detector result contract failed validation.',
      }),
    ).toBeVisible()
    expect(screen.getByText('Data-integrity details')).toBeVisible()
    expect(screen.queryByText('REAL / VERIFIED')).toBeNull()
  })
  it('loads a verified ready evaluation through the canonical query', async () => {
    vi.stubGlobal('crypto', webcrypto)
    vi.stubGlobal('fetch', fixtureFetch())
    renderQuery()
    expect(
      await screen.findByText('48 / 48 verified detector runs', {
        exact: false,
      }),
    ).toBeVisible()
  })
  it('exposes complete state, provenance, filtered runs and support', async () => {
    render(
      <DetectionReady data={DetectionExportSchema.parse(detectionFixture())} />,
    )
    expect(screen.getByText('REAL / VERIFIED')).toBeVisible()
    expect(
      screen.getByRole('heading', { name: 'Complete controlled comparison' }),
    ).toBeVisible()
    await userEvent.click(
      screen.getByRole('button', { name: 'Experiment provenance' }),
    )
    expect(
      within(screen.getByRole('dialog')).getByText('0a4ea1f'),
    ).toBeVisible()
    await userEvent.keyboard('{Escape}')
    await userEvent.click(screen.getByRole('button', { name: 'View runs' }))
    await userEvent.selectOptions(
      screen.getByLabelText('Run strategy'),
      'historical',
    )
    expect(
      screen.getByText('3 of 48 exported runs.', { exact: false }),
    ).toBeVisible()
    await userEvent.keyboard('{Escape}')
    await userEvent.click(screen.getByRole('tab', { name: 'Test composition' }))
    expect(
      screen.getByRole('table', { name: /One row per split/ }),
    ).toBeVisible()
  })
  it('renders undefined variance as undefined, never zero', () => {
    render(<Variability data={detectionFixture()} />)
    const row = screen.getByRole('row', { name: /Historical/ })
    expect(row).toHaveTextContent('Not defined — only one split')
    expect(row).not.toHaveTextContent('0.00')
  })
  it('uses exported global and centered correlations, without estimating from points', () => {
    const d = detectionFixture()
    const a = d.associations.find((a) => a.association_id === 'temporal_at5')!
    a.global_pearson = 0.123456
    a.within_strategy_centered_pearson = -0.654321
    render(<AssociationExplorer data={d} />)
    expect(screen.getByText('0.123', { exact: false })).toBeVisible()
    expect(screen.getByText('-0.654', { exact: false })).toBeVisible()
    const option = associationOption(a)
    const series = option.series as { data: { value: number[] }[] }[]
    const values = series.flatMap((s) => s.data.map((p) => p.value))
    expect(values).toHaveLength(16)
    for (const p of a.points) expect(values).toContainEqual([p.x, p.y])
  })
  it('renders support exactly as exported', () => {
    const d = detectionFixture()
    render(<TestComposition data={d} />)
    const row = screen.getByRole('row', { name: /^C10 \/ 0 / })
    expect(
      within(row)
        .getAllByRole('cell')
        .map((c) => c.textContent),
    ).toEqual(
      d.support
        .filter((s) => s.strategy === 'C10' && s.split_seed === 0)
        .map((s) => String(s.support)),
    )
  })
  it('keeps legacy detector and split routes separate', () => {
    expect(isDetectionRoute('/organization/detector', '')).toBe(true)
    expect(isDetectionRoute('/organization/evaluation', '')).toBe(true)
    expect(isDetectionRoute('/organization/evaluation', '?view=splits')).toBe(
      false,
    )
    expect(new DetectionContractError('validation', 'test').kind).toBe(
      'validation',
    )
  })
})
