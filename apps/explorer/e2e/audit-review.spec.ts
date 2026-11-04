import { test, expect } from '@playwright/test'
import AxeBuilder from '@axe-core/playwright'
import { spawn, execFileSync, type ChildProcess } from 'node:child_process'
import { mkdtempSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { resolve, join } from 'node:path'

test.describe.configure({ mode: 'serial' })

// A real backend and SQLite ledger, not intercepted/mock HTTP responses.
// Run after `uv sync --extra dev` in the sibling backend checkout.
const backend = resolve(
  process.env.FLIR_AUDIT_BACKEND ?? '../flir-leakage-pipeline',
)
const python = join(
  backend,
  '.venv',
  process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python',
)
const workspace = mkdtempSync(join(tmpdir(), 'flir-audit-SYNTHETIC-e2e-'))
const db = join(workspace, 'audit.sqlite3')
const entry = [
  '-c',
  'from flir_pipeline.cli import app; app()',
  'sequences',
  'audit',
]
let service: ChildProcess | undefined
let serviceError = ''
async function start() {
  serviceError = ''
  service = spawn(python, [...entry, 'serve', '--db', db], {
    cwd: backend,
    windowsHide: true,
    stdio: 'pipe',
  })
  service.stderr?.on('data', (chunk) => {
    serviceError += String(chunk)
  })
  await expect
    .poll(
      async () => {
        if (service?.exitCode !== null) throw new Error(serviceError)
        try {
          return (
            await fetch('http://127.0.0.1:8765/api/audit', {
              signal: AbortSignal.timeout(2000),
            })
          ).status
        } catch {
          return 0
        }
      },
      { timeout: 20000 },
    )
    .toBe(200)
}
async function stop() {
  if (service && service.exitCode === null) {
    const current = service
    if (process.platform === 'win32' && current.pid) {
      execFileSync('taskkill', ['/PID', String(current.pid), '/T', '/F'], {
        windowsHide: true,
      })
      return
    }
    await new Promise<void>((done) => {
      current.once('exit', () => done())
      current.kill()
    })
  }
}
test.beforeAll(async () => {
  execFileSync(python, [...entry, 'demo', workspace], {
    cwd: backend,
    windowsHide: true,
  })
  writeFileSync(
    join(workspace, 'inputs', 'synthetic-missing.json'),
    JSON.stringify({
      synthetic: true,
      source_video: 'ZZ-SYNTHETIC-MISSING',
      results: [
        { start: 4, end: 5, length: 2 },
        { start: 8, end: 9, length: 2 },
      ],
    }),
    'utf8',
  )
  execFileSync(
    python,
    [...entry, 'import', join(workspace, 'inputs'), '--db', db],
    { cwd: backend, windowsHide: true },
  )
  execFileSync(
    python,
    [
      '-c',
      `
from pathlib import Path
import sys
from flir_pipeline.audit.store import Store
s=Store(Path(sys.argv[1])); data=s.snapshot()
visual=next(e for e in data['events'] if e['evidence'])
target=next(e for e in data['events'] if e['source_video']=='ZZ-SYNTHETIC-MISSING' and e['start_index']==4)
p=Path(sys.argv[1]).parent/'SYNTHETIC-external.png'
p.write_bytes(s.media(visual['event_id'],0)[0])
s.attach_evidence(target['event_id'],p,locator='SYNTHETIC explicit record 4',basis='explicit_record_reference',rationale='Fixture association by record ID, not temporal proximity',reviewer='SYNTHETIC registrar')
`,
      db,
    ],
    { cwd: backend, windowsHide: true },
  )
  await start()
})
test.afterAll(stop)

test('synthetic evidence → review → correction → service restart → exact exports', async ({
  page,
  browser,
  request,
}) => {
  await page.goto('/organization/sequences?mode=audit')
  await expect(
    page.getByText('DEMO · Evidencia completamente sintética'),
  ).toBeVisible()
  const image = page.getByRole('img', { name: /Evidencia de contexto/ })
  await expect(image).toBeVisible()
  await expect
    .poll(() => image.evaluate((node: HTMLImageElement) => node.naturalWidth))
    .toBe(640)
  await page
    .getByLabel('Revisor', { exact: true })
    .fill('SYNTHETIC-browser-reviewer')
  await page
    .getByLabel('Justificación', { exact: true })
    .fill('SYNTHETIC pending review')
  await page
    .getByLabel('Estado de revisión', { exact: true })
    .selectOption('reviewed')
  await page.getByLabel('Confianza', { exact: true }).selectOption('high')
  await expect(
    page.getByRole('button', { name: 'Guardar revisión' }),
  ).toBeDisabled()
  await expect(page.getByRole('alert')).toContainText('evidencia consultada')
  await page
    .getByLabel('Estado de revisión', { exact: true })
    .selectOption('pending')
  await page.getByRole('button', { name: 'Guardar revisión' }).click()
  await expect(
    page.getByText('Historial persistido (1)', { exact: true }),
  ).toBeVisible()
  await page
    .getByLabel('Estado de revisión', { exact: true })
    .selectOption('reviewed')
  await page
    .getByLabel('Continuidad audiovisual', { exact: true })
    .selectOption('shot_change')
  await page
    .getByLabel('Dependencia visual o de región', { exact: true })
    .selectOption('same_region')
  await page.getByLabel('Confianza', { exact: true }).selectOption('high')
  await page
    .getByLabel('Justificación', { exact: true })
    .fill('SYNTHETIC correction: cut, same region')
  await page.getByLabel('Consulté esta evidencia').check()
  await page.getByRole('button', { name: 'Guardar revisión' }).click()
  await expect(
    page.getByText('Historial persistido (2)', { exact: true }),
  ).toBeVisible()
  const accessibility = await new AxeBuilder({ page })
    .include('.audit-review')
    .withTags(['wcag2a', 'wcag2aa', 'wcag21aa', 'wcag22aa'])
    .analyze()
  expect(accessibility.violations).toEqual([])
  const before = await (
    await request.get('http://127.0.0.1:8765/api/export/json')
  ).json()
  expect(before.decisions).toHaveLength(2)
  const downloadReady = page.waitForEvent('download')
  await page.getByRole('link', { name: 'Exportar JSON', exact: true }).click()
  expect((await downloadReady).suggestedFilename()).toBe('audit-decisions.json')
  expect(before.decisions[1]).toMatchObject({
    continuity: 'shot_change',
    dependence: 'same_region',
    status: 'reviewed',
  })
  await page.screenshot({
    path: 'test-results/audit-review-synthetic.png',
    fullPage: true,
  })
  await page.close()
  await stop()
  await start()
  const reopened = await browser.newPage()
  await reopened.goto('http://127.0.0.1:5173/organization/sequences?mode=audit')
  await expect(
    reopened.getByText('Historial persistido (2)', { exact: true }),
  ).toBeVisible()
  await expect(
    reopened.getByLabel('Justificación', { exact: true }),
  ).toHaveValue('SYNTHETIC correction: cut, same region')
  const after = await (
    await request.get('http://127.0.0.1:8765/api/export/json')
  ).json()
  expect(after).toEqual(before)
  const csv = await request.get('http://127.0.0.1:8765/api/export/csv')
  expect(await csv.text()).toContain('decision_json')
  const denied = await request.get('http://127.0.0.1:8765/api/audit', {
    headers: { Origin: 'https://external.invalid' },
  })
  expect(denied.status()).toBe(403)
  await reopened.close()
})

test('API evidence guards, externally registered visual evidence and pending navigation', async ({
  page,
  request,
}) => {
  const initial = await (
    await request.get('http://127.0.0.1:8765/api/audit')
  ).json()
  const missing = initial.events.find(
    (e: { start_index: number; source_video: string }) =>
      e.source_video === 'ZZ-SYNTHETIC-MISSING' && e.start_index === 8,
  )
  const external = initial.external_evidence[0]
  const input = {
    event_id: missing.event_id,
    expected_version: 0,
    status: 'reviewed',
    confidence: 'high',
    continuity: 'shot_change',
    dependence: 'same_region',
    reviewer: 'SYNTHETIC',
    justification: 'Síntesis: región, cámara y revisión',
    evidence_consulted: [],
    external_evidence_consulted: [],
  }
  const post = (data: unknown) =>
    request.post('http://127.0.0.1:8765/api/decisions', {
      headers: { 'Content-Type': 'application/json', 'X-Audit-Review': '1' },
      data,
    })
  expect((await post(input)).status()).toBe(400)
  expect(
    (await post({ ...input, evidence_consulted: ['invented.jpg'] })).status(),
  ).toBe(400)
  expect(
    (
      await post({
        ...input,
        external_evidence_consulted: [external.evidence_id],
      })
    ).status(),
  ).toBe(400)
  const pending = await post({ ...input, status: 'pending' })
  expect(pending.status()).toBe(201)
  await page.goto('/organization/sequences?mode=audit')
  await page
    .getByLabel('Evidencia visual', { exact: true })
    .selectOption('missing')
  await expect(
    page.getByLabel('Evento candidato', { exact: true }),
  ).toHaveValue(missing.event_id)
  await expect(
    page.getByText(
      'Evidencia visual no disponible para este evento; no se infiere a partir de tiempos cercanos.',
    ),
  ).toBeVisible()
  await expect(
    page.getByRole('button', { name: 'Siguiente pendiente' }),
  ).toBeEnabled()
  await page.getByRole('button', { name: 'Siguiente pendiente' }).click()
  await page
    .getByLabel('Evidencia visual', { exact: true })
    .selectOption('registered')
  await page
    .getByLabel('Evento candidato', { exact: true })
    .selectOption(external.event_id)
  await expect(
    page.getByRole('img', { name: /Evidencia externa registrada/ }),
  ).toBeVisible()
  await page
    .getByLabel('Revisor', { exact: true })
    .fill('SYNTHETIC external reviewer')
  await page
    .getByLabel('Justificación', { exact: true })
    .fill('Correspondencia explícita; región y cámara, no proximidad temporal.')
  await page
    .getByLabel('Estado de revisión', { exact: true })
    .selectOption('reviewed')
  await page
    .getByLabel('Continuidad audiovisual', { exact: true })
    .selectOption('shot_change')
  await page
    .getByLabel('Dependencia visual o de región', { exact: true })
    .selectOption('same_region')
  await page.getByLabel('Confianza', { exact: true }).selectOption('high')
  await expect(
    page.getByRole('button', { name: 'Guardar revisión' }),
  ).toBeDisabled()
  await page.getByLabel(/Consulté la evidencia externa/).check()
  await page.getByRole('button', { name: 'Guardar revisión' }).click()
  await expect(
    page.getByText('Historial persistido (1)', { exact: true }),
  ).toBeVisible()
  const after = await (
    await request.get('http://127.0.0.1:8765/api/audit')
  ).json()
  const decision = after.decisions.find(
    (d: { event_id: string }) => d.event_id === external.event_id,
  )
  expect(decision.external_evidence_consulted).toEqual([external.evidence_id])
  expect(decision.justification).toContain('explícita; región y cámara')
  await page.getByLabel('Orden temporal', { exact: true }).selectOption('desc')
  await expect(page.locator('.audit-review')).not.toContainText('Ã')
})
