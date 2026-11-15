import { test, expect } from './synthetic-fixture'

// Local evidence is large; await observable readiness, never a fixed sleep.
test.use({ actionTimeout: 20000, organizationScenario: 'with_candidates' })
test.setTimeout(120000)
const evidenceExpect = expect.configure({ timeout: 20000 })
import AxeBuilder from '@axe-core/playwright'
import { researchFixture } from '../src/test/research-fixture'
test('runtime V2 overview, exported temporal evidence, linkage and keyboard disclosure', async ({
  page,
}) => {
  await page.route('**/runtime-config.js', (route) =>
    route.fulfill({
      contentType: 'application/javascript',
      body: 'window.__FLIR_CONFIG__={dataMode:"real",snapshotUrl:"/runtime/synthetic-v2.json"}',
    }),
  )
  await page.route('**/runtime/synthetic-v2.json', (route) =>
    route.fulfill({ json: researchFixture() }),
  )
  await page.goto('/organization')
  await evidenceExpect(
    page.getByText('EXPORTED EVIDENCE', { exact: true }),
  ).toBeVisible()
  await evidenceExpect(
    page
      .getByRole('navigation', { name: 'Organization stages' })
      .getByRole('link'),
  ).toHaveCount(4)
  await evidenceExpect(page.getByRole('dialog')).toHaveCount(0)
  const details = page.getByRole('button', {
    name: 'Organization export provenance',
  })
  await details.focus()
  await page.keyboard.press('Enter')
  await evidenceExpect(page.getByRole('dialog')).toBeVisible()
  expect(
    (
      await new AxeBuilder({ page })
        .withTags(['wcag2a', 'wcag2aa', 'wcag22aa'])
        .analyze()
    ).violations,
  ).toEqual([])
  await page.keyboard.press('Escape')
  await evidenceExpect(details).toBeFocused()
  await page
    .getByRole('link', { name: 'Sequences & linkage', exact: true })
    .click()
  await page
    .getByRole('button', { name: 'Temporal evidence', exact: true })
    .click()
  await evidenceExpect(
    page.getByRole('heading', {
      name: 'Temporal structure and boundary evidence',
      exact: true,
    }),
  ).toBeVisible()
  const boundaryZones = page.getByRole('button', {
    name: /^boundary zone .+: \d+–\d+, .+$/,
  })
  // Find an exported zone without assuming a timeline ID or a candidate decision.
  const timeline = page.getByLabel('Timeline', { exact: true })
  const timelineIds = await timeline
    .locator('option')
    .evaluateAll((options) =>
      options.map((option) => (option as HTMLOptionElement).value),
    )
  for (const id of timelineIds) {
    if (await boundaryZones.count()) break
    await timeline.selectOption(id)
  }
  const boundary = boundaryZones.last()
  await evidenceExpect(boundary).toBeVisible()
  const label = await boundary.getAttribute('aria-label')
  const match = /^boundary zone (.+): (\d+)–(\d+), (.+)$/.exec(label ?? '')
  expect(match).not.toBeNull()
  const [, evidenceId, start, end, decision] = match!
  await boundary.click()
  await evidenceExpect(boundary).toHaveAttribute('aria-pressed', 'true')
  const selectedBoundary = page.locator('section.panel').filter({
    has: page.getByRole('heading', {
      name: 'Selected boundary evidence',
      exact: true,
    }),
  })
  await evidenceExpect(
    selectedBoundary.getByText(evidenceId, { exact: true }),
  ).toBeVisible()
  await evidenceExpect(
    selectedBoundary.getByText(`${start}–${end}`, { exact: true }),
  ).toBeVisible()
  await evidenceExpect(
    selectedBoundary.getByText(decision, { exact: true }),
  ).toBeVisible()
  await evidenceExpect(
    page.getByText(
      /Boundary zones are inclusive uncertainty intervals, not exact cuts\./,
    ),
  ).toBeVisible()
  const boundaryDetails = selectedBoundary.getByRole('button', {
    name: 'Technical details',
    exact: true,
  })
  await boundaryDetails.focus()
  await page.keyboard.press('Enter')
  const boundaryDialog = page.getByRole('dialog')
  await evidenceExpect(boundaryDialog).toBeVisible()
  const boundaryEvidence = JSON.parse(
    await boundaryDialog.locator('pre').innerText(),
  )
  expect(boundaryEvidence.boundary_zone).toMatchObject({
    element_id: evidenceId,
    start: Number(start),
    end: Number(end),
    decision,
  })
  expect(boundaryEvidence.timeline.timeline_id).toBe(
    await timeline.inputValue(),
  )
  await page.keyboard.press('Escape')
  await evidenceExpect(boundaryDetails).toBeFocused()
  await evidenceExpect(
    selectedBoundary
      .locator('.metric-summary > div')
      .filter({
        has: page.locator('dt', { hasText: /^Exact cut$/ }),
      })
      .locator('dd'),
  ).toHaveText(boundaryEvidence.boundary_zone.exact_cut ? 'Yes' : 'No')
  expect(
    (
      await new AxeBuilder({ page })
        .withTags(['wcag2a', 'wcag2aa', 'wcag22aa'])
        .analyze()
    ).violations,
  ).toEqual([])
  await page.screenshot({
    path: 'test-results/sequences-v2.png',
    fullPage: true,
  })
  await page
    .getByRole('button', { name: 'Evidence and provenance', exact: true })
    .click()
  await evidenceExpect(
    page
      .getByRole('dialog')
      .getByText(/Timeline ≠ sequence\. Boundary zone ≠ exact cut\./),
  ).toBeVisible()
  await page.keyboard.press('Escape')
  await page
    .getByRole('button', { name: 'Cross-dataset linkage', exact: true })
    .click()
  await evidenceExpect(
    page.getByText(
      /A candidate is not a confirmed match, sequence identity, ground truth, dependency or split constraint\./,
    ),
  ).toBeVisible()
  const candidate = page
    .getByRole('button', { name: /LABELED CONTENT .*candidate.*VIDEO CONTENT/ })
    .first()
  await evidenceExpect(candidate).toBeVisible()
  await candidate.click()
  await evidenceExpect(candidate).toHaveAttribute('aria-pressed', 'true')
  const selectedCandidate = page.locator('section.panel').filter({
    has: page.getByRole('heading', { name: 'Selected candidate', exact: true }),
  })
  const candidateDetails = selectedCandidate.getByRole('button', {
    name: 'Technical details',
    exact: true,
  })
  await candidateDetails.focus()
  await page.keyboard.press('Enter')
  const candidateDialog = page.getByRole('dialog')
  await evidenceExpect(candidateDialog).toBeVisible()
  const candidateEvidence = JSON.parse(
    await candidateDialog.locator('pre').innerText(),
  )
  await page.keyboard.press('Escape')
  await evidenceExpect(candidateDetails).toBeFocused()
  await evidenceExpect(
    selectedCandidate
      .locator('code')
      .filter({ hasText: candidateEvidence.labeled_content_id }),
  ).toHaveText(candidateEvidence.labeled_content_id)
  await evidenceExpect(
    selectedCandidate
      .locator('code')
      .filter({ hasText: candidateEvidence.video_content_id }),
  ).toHaveText(candidateEvidence.video_content_id)
  for (const [metric, score] of [
    ['CLIP cosine', candidateEvidence.clip_cosine],
    ['DINOv2 cosine', candidateEvidence.dinov2_cosine],
  ] as const) {
    const row = selectedCandidate.locator('.metric-summary > div').filter({
      has: page.locator('dt', { hasText: new RegExp(`^${metric}$`) }),
    })
    await evidenceExpect(row.locator('dd')).toHaveText(
      score === null ? 'unavailable' : score.toFixed(4),
    )
  }
  expect(
    (
      await new AxeBuilder({ page })
        .withTags(['wcag2a', 'wcag2aa', 'wcag22aa'])
        .analyze()
    ).violations,
  ).toEqual([])
})
test('legacy redirects and active exported temporal evidence', async ({
  page,
}) => {
  for (const [old, destination] of Object.entries({
    dataset: '/organization',
    embeddings: '/organization/explore?view=embeddings',
    similarity: '/organization/explore?view=similarity',
    reduction: '/organization/explore?view=reduction',
    clustering: '/organization/explore?view=clustering',
    groups: '/organization/sequences',
    splits: '/organization/evaluation?view=splits',
    detector: '/organization/evaluation?view=detector',
  })) {
    await page.goto(`/organization/${old}`)
    await evidenceExpect(page).toHaveURL(
      new RegExp(destination.replace('?', '\\?') + '$'),
    )
  }
  await page.goto('/organization/sequences')
  await page
    .getByRole('button', { name: 'Temporal evidence', exact: true })
    .click()
  await evidenceExpect(
    page.getByRole('heading', {
      name: 'Temporal structure and boundary evidence',
      exact: true,
    }),
  ).toBeVisible()
  await evidenceExpect(
    page.getByRole('button', { name: 'Temporal evidence', exact: true }),
  ).toHaveAttribute('aria-pressed', 'true')
  await evidenceExpect(
    page.getByLabel('Timeline', { exact: true }),
  ).toBeVisible()
  await evidenceExpect(
    page.getByRole('heading', {
      name: 'Temporal evidence timeline',
      exact: true,
    }),
  ).toBeVisible()
})
