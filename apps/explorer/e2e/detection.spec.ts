import { test, expect } from '@playwright/test'
import AxeBuilder from '@axe-core/playwright'

test('verified detector story, disclosure, associations and responsive layouts', async ({
  page,
}) => {
  const errors: string[] = []
  page.on('pageerror', (error) => errors.push(error.message))
  // A missing workspace artifact must not block the independent detector contract.
  await page.addInitScript(() => {
    window.__FLIR_CONFIG__ = {
      dataMode: 'real',
      snapshotUrl: '/runtime/missing-workspace.json',
    }
  })
  await page.goto('/organization/evaluation')
  await expect(
    page.getByRole('heading', { name: 'Complete controlled comparison' }),
  ).toBeVisible()
  await expect(
    page.getByText('48 / 48 verified detector runs', { exact: false }),
  ).toBeVisible()
  await expect(page.getByLabel('Data mode', { exact: true })).toHaveCount(0)
  await expect(page.locator('.demo-bar')).toContainText('REAL / VERIFIED')
  await expect(page.locator('.demo-bar')).not.toContainText('MIXED')
  await expect(
    page.getByRole('img', { name: /Split means/ }).locator('svg'),
  ).toBeVisible()
  await page.screenshot({
    path: 'test-results/detection-desktop.png',
    fullPage: true,
  })
  for (const width of [768, 1024, 1440]) {
    await page.setViewportSize({ width, height: 1000 })
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= window.innerWidth,
      ),
      `width ${width}`,
    ).toBe(true)
  }
  await page.setViewportSize({ width: 768, height: 1024 })
  await page.screenshot({
    path: 'test-results/detection-tablet.png',
    fullPage: true,
  })
  await page.setViewportSize({ width: 1440, height: 1000 })
  for (const name of [
    'Comparison',
    'Variability',
    'Classes',
    'Test composition',
    'Associations',
  ]) {
    await page.getByRole('tab', { name, exact: true }).click()
    expect(
      (
        await new AxeBuilder({ page })
          .withTags(['wcag2a', 'wcag2aa', 'wcag22aa'])
          .analyze()
      ).violations,
      name,
    ).toEqual([])
  }
  await expect(page.getByText('0.827', { exact: false })).toBeVisible()
  await expect(page.getByText('0.047 centered Pearson r')).toBeVisible()
  await expect(
    page.getByRole('img', {
      name: /temporal_at5: one point per split, 16 valid pairs/,
    }),
  ).toBeVisible()
  await page.getByLabel('Pre-specified association').selectOption('domain_clip')
  await expect(
    page.getByText(/16 splits · 16 valid pairs · Pre-specified · Heavy/),
  ).toBeVisible()
  await page
    .getByLabel('Pre-specified association')
    .selectOption('temporal_at5')
  await page.screenshot({
    path: 'test-results/detection-associations.png',
    fullPage: true,
  })
  await page.getByRole('button', { name: 'Switch to light theme' }).click()
  expect(
    (
      await new AxeBuilder({ page })
        .withTags(['wcag2a', 'wcag2aa', 'wcag22aa'])
        .analyze()
    ).violations,
  ).toEqual([])
  const trigger = page.getByRole('button', { name: 'View runs', exact: true })
  await trigger.focus()
  await page.keyboard.press('Enter')
  await expect(page.getByRole('dialog')).toBeVisible()
  await page.getByLabel('Run strategy').selectOption('C10')
  await page.getByLabel('Split seed', { exact: true }).selectOption('0')
  await page.getByLabel('Detector seed', { exact: true }).selectOption('42')
  await expect(
    page.getByText('1 of 48 exported runs.', { exact: false }),
  ).toBeVisible()
  expect(
    (
      await new AxeBuilder({ page })
        .withTags(['wcag2a', 'wcag2aa', 'wcag22aa'])
        .analyze()
    ).violations,
  ).toEqual([])
  await page.keyboard.press('Escape')
  await expect(trigger).toBeFocused()
  expect(errors).toEqual([])
})

test('missing, invalid and incomplete contracts never fall back to demo', async ({
  page,
}) => {
  await page.route('**/snapshot/manifest.json*', (route) =>
    route.request().resourceType() === 'fetch'
      ? route.fulfill({ status: 404, body: '' })
      : route.continue(),
  )
  await page.goto('/organization/evaluation')
  await expect(
    page.getByRole('heading', {
      name: 'Verified detector results are not available.',
    }),
  ).toBeVisible()
  await expect(page.locator('.demo-bar')).toContainText('UNAVAILABLE')
  await expect(page.getByText('Random content-level · 25.2%')).toHaveCount(0)
  await page.unroute('**/snapshot/manifest.json*')
  await page.route('**/snapshot/manifest.json*', (route) =>
    route.request().resourceType() === 'fetch'
      ? route.fulfill({
          contentType: 'application/json',
          body: JSON.stringify({ state: 'PARTIAL' }),
        })
      : route.continue(),
  )
  await page.reload()
  await expect(
    page.getByRole('heading', {
      name: 'Detector result contract failed validation.',
    }),
  ).toBeVisible()
  await expect(page.getByText('REAL / VERIFIED', { exact: true })).toHaveCount(
    0,
  )
})

test('detector contract loading is explicit', async ({ page }) => {
  let release!: () => void
  const wait = new Promise<void>((resolve) => {
    release = resolve
  })
  await page.route('**/snapshot/manifest.json*', async (route) => {
    if (route.request().resourceType() !== 'fetch') return route.continue()
    await wait
    await route.continue()
  })
  await page.goto('/organization/evaluation', { waitUntil: 'domcontentloaded' })
  await expect(
    page
      .getByRole('status', { name: '' })
      .filter({ hasText: 'Loading verified detector results' }),
  ).toBeVisible()
  release()
  await expect(
    page.getByRole('heading', { name: 'Complete controlled comparison' }),
  ).toBeVisible()
})
