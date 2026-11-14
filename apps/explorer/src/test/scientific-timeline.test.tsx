import { fireEvent, render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { webcrypto } from 'node:crypto'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { ScientificTimelineView } from '@/features/organization/ScientificTimeline'
import {
  ScientificTimelineSchema,
  framesForRun,
  temporalOrder,
  timelineCsv,
  clusterColor,
  loadScientificTimeline,
  summarizeFrames,
  admitVideoTimeline,
} from '@/data/scientific-timeline'
import {
  scientificTimelineFixture,
  verifiedVideoFixture,
} from './scientific-timeline-fixture'

afterEach(() => vi.unstubAllGlobals())

describe('scientific timeline: synthetic presentation invariants', () => {
  it('preserves duplicate content occurrences, index zero, recurrence, noise and missing cluster', () => {
    const data = ScientificTimelineSchema.parse(scientificTimelineFixture())
    const frames = framesForRun(data, data.runs[0], 'synthetic-cluster-a').sort(
      temporalOrder,
    )
    expect(frames).toHaveLength(8)
    expect(frames.filter((f) => f.content_id === 'c-first')).toHaveLength(2)
    expect(
      frames.filter((f) => f.position === 0).map((f) => f.frame_id),
    ).toEqual(['f-copy', 'f-first'])
    expect(frames.find((f) => f.frame_id === 'f-noise')?.cluster_id).toBe(-1)
    expect(
      frames.find((f) => f.frame_id === 'f-no-cluster')?.cluster_id,
    ).toBeNull()
    expect(
      frames.filter((f) => f.cluster_id === 7).map((f) => f.position),
    ).toEqual([30, 42])
    expect(summarizeFrames(frames).reduce((n, s) => n + s.count, 0)).toBe(8)
  })

  it.each([
    'duplicate-record',
    'duplicate-membership',
    'wrong-run',
    'unknown-frame',
    'changed-test',
    'index-conflict',
    'sequence-promotion',
  ])('rejects %s', (kind) => {
    const data = scientificTimelineFixture()
    if (kind === 'duplicate-record') data.records.push({ ...data.records[0] })
    if (kind === 'duplicate-membership')
      data.cluster_memberships.push({ ...data.cluster_memberships[0] })
    if (kind === 'wrong-run') data.runs[0].seed = 4
    if (kind === 'unknown-frame') data.runs[0].members[0].frame_id = 'unknown'
    if (kind === 'changed-test') data.runs[1].members[1].new_split = 'train'
    if (kind === 'index-conflict') data.records[0].position = 0
    if (kind === 'sequence-promotion')
      data.intervals[0].sequence_id = 'invented-confirmed'
    expect(ScientificTimelineSchema.safeParse(data).success).toBe(false)
  })

  it('exports one selected run and clustering reference, quotes filenames, keeps split identities separate', () => {
    const data = scientificTimelineFixture()
    data.records[0].filename = '=formula,"quoted".jpg'
    const frames = framesForRun(data, data.runs[1], 'synthetic-cluster-b')
    const csv = timelineCsv(frames, data.runs[1], 'synthetic-cluster-b')
    expect(csv.split('\r\n')).toHaveLength(9)
    expect(csv).toContain('historical-S1-1')
    expect(csv).not.toContain('historical-S1-0')
    expect(csv).toContain('synthetic-cluster-b')
    expect(csv).toContain(`"'=formula,""quoted"".jpg"`)
    expect(csv).toContain('membership-historical-S1-1')
    expect(clusterColor('synthetic-cluster-a', 7)).toBe(
      clusterColor('synthetic-cluster-a', 7),
    )
    expect(clusterColor('synthetic-cluster-a', 0)).not.toBe(
      clusterColor('synthetic-cluster-a', null),
    )
  })

  it('renders both lanes per occurrence, uses candidates without splitting at color changes, supports keyboard alternatives', async () => {
    render(<ScientificTimelineView data={scientificTimelineFixture()} />)
    expect(screen.getByText(/DEMO \/ SYNTHETIC/)).toBeInTheDocument()
    expect(document.querySelectorAll('[data-lane="cluster"]')).toHaveLength(7)
    expect(document.querySelectorAll('[data-lane="split"]')).toHaveLength(7)
    expect(document.querySelectorAll('[data-interval-id]')).toHaveLength(1)
    const table = screen.getByRole('table', {
      name: 'Tabla accesible de ocurrencias temporales',
    })
    const rows = within(table).getAllByRole('row').slice(1)
    expect(rows.map((r) => r.children[0].textContent)).toEqual([
      '0',
      '0',
      '12',
      '20',
      '30',
      '40',
      '42',
    ])
    await userEvent.click(
      screen.getByRole('button', { name: 'Ver frame f-noise' }),
    )
    expect(
      screen.getByRole('region', { name: 'Detalle del frame' }),
    ).toHaveTextContent('f-noise')
    await userEvent.selectOptions(
      screen.getByLabelText('Secuencia / tramo exportado'),
      'candidate-synthetic',
    )
    expect(
      screen.getByRole('heading', { name: 'Tramo candidato seleccionado' }),
    ).toBeInTheDocument()
    await userEvent.selectOptions(
      screen.getByLabelText('Clúster (resaltar)'),
      '7',
    )
    expect(within(table).getAllByRole('row')).toHaveLength(3)
    expect(
      screen.getByLabelText('Contabilidad de ocurrencias'),
    ).toHaveTextContent('8')
    await userEvent.selectOptions(
      screen.getByLabelText('Ejecución de clustering'),
      'synthetic-cluster-b',
    )
    expect(screen.getByLabelText('Clúster (resaltar)')).toHaveValue('')
    expect(within(table).getAllByRole('row')).toHaveLength(8)
    fireEvent.change(screen.getByLabelText('Desde índice'), {
      target: { value: '20' },
    })
    expect(within(table).getAllByRole('row')).toHaveLength(5)
    await userEvent.selectOptions(
      screen.getByLabelText('Población'),
      'expanded',
    )
    expect(screen.getByLabelText('Política')).toHaveValue('S3')
    expect(screen.getByLabelText('Semilla de particionamiento')).toHaveValue(
      '4',
    )
  })

  it('shows missing positions as exclusions without making a fake axis', () => {
    const data = scientificTimelineFixture()
    data.records.forEach((r) => {
      r.reason = 'unknown_order'
      r.position = null
    })
    render(<ScientificTimelineView data={data} />)
    expect(
      screen.getByText(/No hay ocurrencias temporalmente representables/),
    ).toBeInTheDocument()
    expect(document.querySelectorAll('[data-lane]')).toHaveLength(0)
    expect(
      screen.getByRole('button', { name: 'Exportar excluded_frames.csv' }),
    ).toBeEnabled()
  })

  it('verifies runtime bytes against receipt and rejects tampering without a demo fallback', async () => {
    const { data, admission } = verifiedVideoFixture()
    const json = JSON.stringify(data)
    const admissionJson = JSON.stringify(admission)
    const admissionSha = [
      ...new Uint8Array(
        await webcrypto.subtle.digest(
          'SHA-256',
          new TextEncoder().encode(admissionJson),
        ),
      ),
    ]
      .map((v) => v.toString(16).padStart(2, '0'))
      .join('')
    const raw = new TextEncoder().encode(json)
    const sha = [
      ...new Uint8Array(await webcrypto.subtle.digest('SHA-256', raw)),
    ]
      .map((b) => b.toString(16).padStart(2, '0'))
      .join('')
    const receipt = {
      schema_version: 'scientific-timeline-export-receipt-v1',
      files: { 'index.json': sha, 'admission.json': admissionSha },
      source_checksums: scientificTimelineFixture().source_checksums,
    }
    vi.stubGlobal('crypto', webcrypto)
    vi.stubGlobal(
      'fetch',
      vi.fn(
        async (url) =>
          new Response(
            String(url).endsWith('receipt.json')
              ? JSON.stringify(receipt)
              : String(url).endsWith('admission.json')
                ? admissionJson
                : json,
          ),
      ),
    )
    expect((await loadScientificTimeline()).origin).toBe('synthetic_fixture')
    receipt.files['index.json'] = '0'.repeat(64)
    await expect(loadScientificTimeline()).rejects.toThrow('checksum')
  })

  it.each([
    'inferred-axis',
    'unknown-video',
    'wrong-content',
    'wrong-source-hash',
    'unbound-position',
    'wrong-package-run',
    'ambiguous-content',
  ])('rejects unverified admission: %s', (kind) => {
    const { data, admission } = verifiedVideoFixture()
    if (kind === 'inferred-axis')
      data.axes[0].association = 'filename_inference'
    if (kind === 'unknown-video')
      admission.accepted_records[0].source_video_id = 'unauthorized'
    if (kind === 'wrong-content')
      admission.accepted_records[0].content_id = 'wrong'
    if (kind === 'wrong-source-hash')
      admission.accepted_records[0].source_video_sha256 = '0'.repeat(64)
    if (kind === 'unbound-position') admission.accepted_records.pop()
    if (kind === 'wrong-package-run') admission.membership_id = 'different'
    if (kind === 'ambiguous-content') {
      data.records[0].content_id = 'c-recurrent'
      admission.accepted_records[0].content_id = 'c-recurrent'
    }
    expect(() => admitVideoTimeline(data, admission)).toThrow()
  })

  it('keeps all three authorized videos visible with zero verified positions and no fake marks', () => {
    const { data, admission } = verifiedVideoFixture()
    admission.accepted_records = []
    data.records.forEach((r) => {
      r.reason = 'unknown_video'
      r.axis_id = null
      r.position = null
    })
    render(
      <ScientificTimelineView data={admitVideoTimeline(data, admission)} />,
    )
    expect(document.querySelectorAll('[data-lane]')).toHaveLength(0)
    expect(screen.getByText(/Entrega B bloqueada/)).toBeInTheDocument()
    expect(
      screen
        .getByLabelText('Vídeo original / asociación temporal')
        .querySelectorAll('option'),
    ).toHaveLength(3)
  })
})
