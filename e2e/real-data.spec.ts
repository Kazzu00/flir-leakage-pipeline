import { test, expect } from '@playwright/test'
import AxeBuilder from '@axe-core/playwright'
import fixture from '../src/test/fixtures/leakage-synthetic.json' with { type: 'json' }
import { largeSyntheticSnapshot } from '../src/test/artifact-fixtures'

test('synthetic artifact contract flow: persisted coordinates, both encoders and missing stages', async ({
  page,
}) => {
  const errors: string[] = []
  page.on('pageerror', (e) => errors.push(e.message))
  await page.route('**/runtime/leakage-snapshot.json', (route) =>
    route.fulfill({ json: fixture }),
  )
  await page.goto('/organization/reduction')
  await expect(
    page.getByText('EXPORTED EVIDENCE', { exact: true }),
  ).toBeVisible()
  await expect(
    page.getByRole('img', {
      name: 'Artifact scatter: 3 persisted coordinates',
    }),
  ).toBeVisible()
  await expect(page.locator('canvas').first()).toBeVisible()
  await expect(page.getByText(/coordinates: -4.5, 12/)).toBeVisible()
  await page.getByLabel('Encoder', { exact: true }).selectOption('CLIP')
  await expect(
    page.getByRole('heading', { name: 't-SNE · reduction-000002' }),
  ).toBeVisible()
  await page.screenshot({
    path: 'test-results/artifact-reduction.png',
    fullPage: true,
  })
  await expect(
    page.getByText('Sequence: unknown. Capture timestamp: unavailable.'),
  ).toBeVisible()
  expect(
    (await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa']).analyze())
      .violations,
  ).toEqual([])
  const nav = page.getByRole('navigation', { name: 'Organization stages' })
  await page.getByLabel('View', { exact: true }).selectOption('clustering')
  await expect(
    page.getByRole('heading', { name: 'Cluster visual explorer' }),
  ).toBeVisible()
  await expect(
    page.getByLabel('Clustering configuration', { exact: true }),
  ).toBeVisible()
  await expect(page.getByLabel('Encoder', { exact: true })).toHaveCount(0)
  await nav.getByRole('link', { name: 'Evaluation', exact: true }).click()
  // Evaluation and split membership use independent real contracts.
  await page.getByLabel('Evaluation evidence').selectOption('splits')
  await expect(
    page.getByRole('heading', {
      name: 'Visual inspection of real split memberships',
    }),
  ).toBeVisible()
  await expect(page.getByRole('button', { name: 'AFTER SPLIT' })).toHaveCount(0)
  await nav
    .getByRole('link', { name: 'Visual exploration', exact: true })
    .click()
  await page.getByLabel('View', { exact: true }).selectOption('similarity')
  await expect(page.getByText(/Histogram counts are unavailable/)).toBeVisible()
  expect(await page.locator('body').innerText()).not.toMatch(
    /private_person|private-drone|C:\\Users|demo-content/,
  )
  await expect(page.locator('.demo-bar')).toContainText('EXPORTED EVIDENCE')
  expect(errors).toEqual([])
})

test('malformed private snapshot produces a safe recoverable error', async ({
  page,
}) => {
  await page.route('**/runtime/leakage-snapshot.json', (route) =>
    route.fulfill({
      body: '{private C:\\Users\\secret',
      contentType: 'application/json',
    }),
  )
  await page.goto('/')
  await page.getByLabel('Data mode', { exact: true }).selectOption('real')
  await expect(page.getByRole('alert')).toContainText('snapshot-malformed')
  await expect(page.locator('body')).not.toContainText('secret')
  await page.getByLabel('Data mode', { exact: true }).selectOption('demo')
  await expect(
    page.getByRole('heading', { name: 'One pipeline. Three perspectives.' }),
  ).toBeVisible()
})

test('9000 persisted points use canvas and a bounded content table', async ({
  page,
}) => {
  await page.route('**/runtime/leakage-snapshot.json', (route) =>
    route.fulfill({ json: largeSyntheticSnapshot() }),
  )
  await page.goto('/organization/reduction')
  await expect(
    page.getByRole('img', {
      name: 'Artifact scatter: 9000 persisted coordinates',
    }),
  ).toBeVisible()
  await expect(page.locator('canvas').first()).toBeVisible()
  await page.getByText('Browse all contents · keyboard alternative').click()
  await page.getByLabel('Find content alias').fill('content-009000')
  await page
    .getByRole('button', { name: 'content-009000', exact: true })
    .click()
  await expect(
    page.getByRole('heading', { name: 'Content detail · content-009000' }),
  ).toBeVisible()
  expect(await page.locator('tbody tr').count()).toBeLessThan(30)
})

test('organization clustering is independent of legacy snapshots and workspace mode', async ({
  page,
}) => {
  await page.route('**/runtime/leakage-snapshot.json', (route) =>
    route.fulfill({ json: fixture }),
  )
  await page.goto('/organization/clustering')
  const configuration = page.getByLabel('Clustering configuration', {
    exact: true,
  })
  await expect(configuration).toBeVisible()
  const runId = await configuration.inputValue()
  await expect(page.getByLabel('Data mode', { exact: true })).toHaveCount(0)
  await page.getByLabel('View', { exact: true }).selectOption('reduction')
  await page.getByLabel('View', { exact: true }).selectOption('clustering')
  await expect(configuration).toHaveValue(runId)
  await expect(page.getByLabel('Encoder', { exact: true })).toHaveCount(0)
  await expect(
    page.getByRole('button', { name: 'AFTER SPLIT', exact: true }),
  ).toHaveCount(0)
  const evidence = page.getByRole('region', { name: 'Clustering evidence' })
  await expect(evidence).not.toContainText(
    /DEMO|SYNTHETIC IDENTITIES|Status mock/,
  )
  await expect(evidence.locator('canvas')).toHaveCount(0)
  const grid = page.getByRole('list', { name: 'Cluster contents' })
  await expect(grid.getByRole('button').first()).toBeVisible()
  await grid.getByRole('button').first().click()
  const contentId = (await grid
    .getByRole('button')
    .first()
    .getAttribute('aria-label'))!.replace('Inspect content ', '')
  const selected = page.locator('section.panel').filter({
    has: page.getByRole('heading', { name: 'Selected content', exact: true }),
  })
  await expect(selected.getByText(contentId, { exact: true })).toBeVisible()

  const options = await configuration
    .locator('option')
    .evaluateAll((elements) =>
      elements.map((option) => (option as HTMLOptionElement).value),
    )
  const otherRun = options.find((value) => value !== runId)
  if (otherRun) {
    await configuration.selectOption(otherRun)
    await expect(page.getByLabel('Cluster', { exact: true })).toBeVisible()
    await expect(
      page.getByText(
        'Select a content thumbnail to inspect its stored evidence.',
      ),
    ).toBeVisible()
    await configuration.selectOption(runId)
    await expect(grid.getByRole('button').first()).toHaveAttribute(
      'aria-pressed',
      'false',
    )
  }
  await page.getByLabel('View', { exact: true }).selectOption('reduction')
  await page.getByLabel('View', { exact: true }).selectOption('clustering')
  await expect(configuration).toHaveValue(runId)
  await expect(evidence).not.toContainText(
    /DEMO|SYNTHETIC IDENTITIES|Status mock/,
  )
})
