import { test, expect } from '@playwright/test'
import AxeBuilder from '@axe-core/playwright'
import { createHash } from 'node:crypto'
import { readFile } from 'node:fs/promises'
import { verifiedVideoFixture } from '../src/test/scientific-timeline-fixture'

test('synthetic timeline: synchronized lanes, keyboard, recurrence, interval, CSV and both themes', async ({
  page,
}) => {
  const { data, admission } = verifiedVideoFixture()
  const raw = JSON.stringify(data)
  const admissionRaw = JSON.stringify(admission)
  const receipt = {
    schema_version: 'scientific-timeline-export-receipt-v1',
    files: {
      'index.json': createHash('sha256').update(raw).digest('hex'),
      'admission.json': createHash('sha256').update(admissionRaw).digest('hex'),
    },
    source_checksums: data.source_checksums,
  }
  await page.route(
    '**/runtime/verified-video-timeline-v1/admission.json',
    (r) => r.fulfill({ body: admissionRaw, contentType: 'application/json' }),
  )
  await page.route('**/runtime/verified-video-timeline-v1/index.json', (r) =>
    r.fulfill({ body: raw, contentType: 'application/json' }),
  )
  await page.route('**/runtime/verified-video-timeline-v1/receipt.json', (r) =>
    r.fulfill({ json: receipt }),
  )
  await page.goto('/organization/sequences')
  await expect(
    page.getByRole('heading', {
      name: 'Línea temporal científica',
      exact: true,
    }),
  ).toBeVisible()
  await expect(page.locator('.scientific-timeline')).toContainText(
    'DEMO / SYNTHETIC',
  )
  await expect(page.locator('[data-lane="cluster"]')).toHaveCount(7)
  await expect(page.locator('[data-lane="split"]')).toHaveCount(7)
  await page
    .getByRole('button', { name: 'Ver frame f-copy', exact: true })
    .click()
  await expect(
    page.getByRole('region', { name: 'Detalle del frame' }),
  ).toContainText('quarantine')
  await page.getByLabel('Zoom horizontal', { exact: true }).focus()
  await page.keyboard.press('ArrowRight')
  await expect(page.getByLabel('Factor de zoom')).toHaveText('2×')
  const scroll = page.getByRole('region', {
    name: 'Desplazamiento horizontal de las tres pistas',
  })
  expect(await scroll.evaluate((e) => e.scrollWidth > e.clientWidth)).toBe(true)
  await scroll.focus()
  await page.keyboard.press('ArrowRight')
  await page.getByLabel('Clúster (resaltar)').selectOption('7')
  await expect(
    page
      .getByRole('table', { name: 'Tabla accesible de ocurrencias temporales' })
      .getByRole('row'),
  ).toHaveCount(3)
  await page
    .getByLabel('Secuencia / tramo exportado')
    .selectOption('candidate-synthetic')
  await expect(
    page.getByRole('heading', { name: 'Tramo candidato seleccionado' }),
  ).toBeVisible()
  const download = page.waitForEvent('download')
  await page
    .getByRole('button', { name: 'Exportar frames asignados (run completo)' })
    .click()
  const file = await download
  expect(file.suggestedFilename()).toBe('timeline_assigned.csv')
  const csv = await readFile((await file.path())!, 'utf8')
  expect(csv.trim().split(/\r?\n/)).toHaveLength(8)
  expect(csv).toContain('historical-S1-0')
  expect(csv).not.toContain('expanded-S3-4')
  await page.getByLabel('Clúster (resaltar)').selectOption('')
  await page.getByRole('button', { name: 'Restablecer intervalo' }).click()
  await page.getByLabel('Zoom horizontal', { exact: true }).focus()
  await page.keyboard.press('ArrowLeft')
  const diagram = page.getByRole('img', { name: /Tres pistas sincronizadas/ })
  await diagram.scrollIntoViewIfNeeded()
  const box = (await diagram.boundingBox())!
  await page.mouse.move(box.x + box.width * 0.25, box.y + 60)
  await page.mouse.down()
  await page.mouse.move(box.x + box.width * 0.75, box.y + 60)
  await page.mouse.up()
  expect(
    Number(await page.getByLabel('Desde índice', { exact: true }).inputValue()),
  ).toBeGreaterThan(0)
  await page.getByRole('button', { name: 'Restablecer intervalo' }).click()
  for (const theme of ['light', 'dark']) {
    await page.evaluate(
      (v) => document.documentElement.setAttribute('data-theme', v),
      theme,
    )
    const contrasts = await page.evaluate(() => {
      const context = document
        .createElement('canvas')
        .getContext('2d', { willReadFrequently: true })!
      const luminance = (color: string) => {
        context.fillStyle = color
        context.fillRect(0, 0, 1, 1)
        const rgb = [...context.getImageData(0, 0, 1, 1).data]
          .slice(0, 3)
          .map((c) => {
            const value = c / 255
            return value <= 0.04045
              ? value / 12.92
              : ((value + 0.055) / 1.055) ** 2.4
          })
        return rgb[0] * 0.2126 + rgb[1] * 0.7152 + rgb[2] * 0.0722
      }
      const background = luminance(
        getComputedStyle(document.querySelector('.scientific-timeline-scroll')!)
          .backgroundColor,
      )
      return [
        ...new Set(
          [...document.querySelectorAll('[data-lane]')].map(
            (e) => getComputedStyle(e).fill,
          ),
        ),
      ].map((color) => {
        const foreground = luminance(color)
        return {
          color,
          ratio:
            (Math.max(foreground, background) + 0.05) /
            (Math.min(foreground, background) + 0.05),
        }
      })
    })
    expect(
      Math.min(...contrasts.map((c) => c.ratio)),
      `${theme}: ${JSON.stringify(contrasts)}`,
    ).toBeGreaterThanOrEqual(3)
    await test.info().attach(`mark-contrast-${theme}.json`, {
      body: JSON.stringify(contrasts),
      contentType: 'application/json',
    })
    const result = await new AxeBuilder({ page })
      .include('.scientific-timeline')
      .withTags(['wcag2a', 'wcag2aa', 'wcag22aa'])
      .analyze()
    expect(
      result.violations,
      `${theme}: ${JSON.stringify(result.violations.map((v) => ({ id: v.id, targets: v.nodes.map((n) => n.target) })))}`,
    ).toEqual([])
    await page.screenshot({
      path: `test-results/scientific-timeline-synthetic-${theme}.png`,
      fullPage: true,
    })
  }
  await page.getByLabel('Población').selectOption('expanded')
  await expect(page.getByLabel('Política', { exact: true })).toHaveValue('S3')
  await expect(page.getByLabel('Semilla de particionamiento')).toHaveValue('4')
})

test('missing temporal publication remains unavailable, independent of legacy temporal evidence', async ({
  page,
}) => {
  await page.route('**/runtime/verified-video-timeline-v1/**', (r) =>
    r.fulfill({ status: 404 }),
  )
  await page.goto('/organization/sequences')
  await expect(
    page.getByRole('heading', {
      name: 'Línea temporal científica no disponible',
    }),
  ).toBeVisible()
  await expect(page.locator('[data-lane]')).toHaveCount(0)
  await expect(page.locator('main')).not.toContainText(
    'DEMO / SYNTHETIC — fixture',
  )
  await page
    .getByRole('button', { name: 'Temporal evidence', exact: true })
    .click()
  await expect(page).toHaveURL(/mode=sequences/)
})

test('real strict video scope: S2-3 delivery identity and complete exclusions without inferred frames', async ({
  page,
  request,
}) => {
  test.skip(
    process.env.FLIR_SCIENTIFIC_TIMELINE_REAL !== '1',
    'Requires explicitly mounted private local evidence',
  )
  const base = '/runtime/verified-video-timeline-v1'
  const response = await request.get(base + '/coverage.json')
  expect(response.ok()).toBe(true)
  const coverage = await response.json()
  expect(coverage.source_run).toBe('historical-S2-3')
  expect(coverage.total).toBe(1657)
  expect(coverage.accepted).toBe(0)
  expect(coverage.excluded).toBe(1657)
  expect(Object.keys(coverage.accepted_by_video)).toHaveLength(3)
  expect(coverage.absence_claimed).toBe(false)
  const admission = await (await request.get(base + '/admission.json')).json()
  const source = await (
    await request.get('/runtime/scientific-timeline-v1/index.json')
  ).json()
  const frozen = source.runs.find(
    (r: { run_id: string }) => r.run_id === 'historical-S2-3',
  )
  expect(admission.membership_id).toBe(frozen.membership_id)
  await page.goto('/organization/sequences')
  await expect(page.getByLabel('Contabilidad de ocurrencias')).toContainText(
    '1657',
  )
  await expect(page.getByLabel('Política', { exact: true })).toHaveValue('S2')
  await expect(page.getByLabel('Semilla de particionamiento')).toHaveValue('3')
  await expect(
    page.getByLabel('Vídeo original / asociación temporal').locator('option'),
  ).toHaveCount(3)
  await expect(page.locator('[data-lane]')).toHaveCount(0)
  await expect(page.locator('.scientific-timeline')).toContainText(
    'Entrega B bloqueada',
  )
  await expect(page.locator('.scientific-timeline')).toContainText(
    'Clúster no disponible',
  )
  await expect(page.locator('.scientific-timeline')).not.toContainText(
    'DEMO / SYNTHETIC',
  )
  const pending = page.waitForEvent('download')
  await page
    .getByRole('button', { name: 'Exportar excluded_frames.csv' })
    .click()
  const file = await pending
  expect(file.suggestedFilename()).toBe('excluded_frames.csv')
  const csv = await readFile((await file.path())!, 'utf8')
  expect(csv.trim().split(/\r?\n/)).toHaveLength(1658)
  expect(csv).toContain('historical-S2-3')
  expect(csv).not.toContain('historical-S1-0')
  for (const theme of ['light', 'dark']) {
    await page.evaluate(
      (v) => document.documentElement.setAttribute('data-theme', v),
      theme,
    )
    const a11y = await new AxeBuilder({ page })
      .include('.scientific-timeline')
      .withTags(['wcag2a', 'wcag2aa', 'wcag21aa', 'wcag22aa'])
      .analyze()
    expect(a11y.violations).toEqual([])
    await page.screenshot({
      path: `test-results/verified-video-timeline-real-${theme}.png`,
      fullPage: true,
    })
  }
})
