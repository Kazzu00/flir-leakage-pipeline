import { test, expect } from '@playwright/test'

// Local evidence is large; await observable readiness, never a fixed sleep.
test.use({ actionTimeout: 20000 })
test.setTimeout(120000)
const evidenceExpect = expect.configure({ timeout: 20000 })
import { createHash } from 'node:crypto'
import assert from 'node:assert/strict'
import { OrganizationEvidenceBundleSchema } from '../src/contracts/organization-evidence-v2'

// Explicit local acceptance only: CI must never require private runtime files.
test.skip(
  process.env.FLIR_M02_REAL_ACCEPTANCE !== '1',
  'Requires existing local real exports',
)

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => {
    window.__FLIR_CONFIG__ = {
      dataMode: 'demo',
      snapshotUrl: '/runtime/leakage-snapshot.json',
    }
  })
})

test('real bundle transport, schema, counts and presentation joins', async ({
  request,
}) => {
  const base = '/runtime/organization-evidence-v2/'
  const manifest = await (await request.get(base + 'manifest.json')).json()
  const bundle: Record<string, unknown> = { manifest }
  for (const [file, digest] of Object.entries(manifest.file_sha256)) {
    const response = await request.get(base + file)
    expect(response.ok(), file).toBe(true)
    const bytes = await response.body()
    expect(createHash('sha256').update(bytes).digest('hex'), file).toBe(digest)
    if (!file.startsWith('schema/'))
      bundle[file.replace('.json', '')] = JSON.parse(bytes.toString('utf8'))
  }
  const data = OrganizationEvidenceBundleSchema.parse(bundle)
  expect(data.records).toHaveLength(manifest.record_count)
  expect(data.contents).toHaveLength(manifest.unique_content_count)
  expect(data.candidate_pairs).toHaveLength(manifest.candidate_pair_count)
  expect(data.splits).toHaveLength(manifest.split_count)
  const contents = new Map(data.contents.map((row) => [row.content_id, row]))
  const records = new Map(data.records.map((row) => [row.record_id, row]))
  const previews = new Map(data.media.map((row) => [row.preview_key, row]))
  expect(contents.size).toBe(data.contents.length)
  expect(records.size).toBe(data.records.length)
  // Native assertions avoid creating tens of thousands of Playwright report steps.
  for (const content of data.contents) {
    assert.equal(
      previews.get(content.preview_key)?.content_id,
      content.content_id,
    )
    for (const id of content.record_ids)
      assert.equal(records.get(id)?.content_id, content.content_id)
  }
  for (const pair of data.candidate_pairs) {
    assert.ok(
      contents.has(pair.labeled_content_id) &&
        contents.has(pair.video_content_id),
    )
  }
  const splitIds = new Set(data.splits.map((row) => row.split_space_id))
  for (const row of data.split_memberships) {
    assert.ok(splitIds.has(row.split_space_id))
    assert.equal(records.get(row.record_id)?.content_id, row.content_id)
  }
  for (const timeline of data.timelines) {
    for (const point of timeline.points) {
      assert.ok(contents.has(point.content_id))
      for (const id of point.record_ids)
        assert.equal(records.get(id)?.content_id, point.content_id)
    }
  }
})

test('real overview, visual selection, temporal playback, linkage and detector navigation', async ({
  page,
  request,
}) => {
  const errors: string[] = []
  page.on('pageerror', (error) => errors.push(error.message))
  const manifest = await (
    await request.get('/runtime/organization-evidence-v2/manifest.json')
  ).json()
  await page.goto('/organization')
  const summary = page.getByRole('region', {
    name: 'Organization export summary',
  })
  await evidenceExpect(summary).toContainText(String(manifest.record_count))
  await evidenceExpect(summary).toContainText(
    String(manifest.unique_content_count),
  )
  await summary
    .getByRole('button', { name: 'Organization export provenance' })
    .click()
  await evidenceExpect(page.getByRole('dialog')).toContainText(
    manifest.source_commit,
  )
  await page.keyboard.press('Escape')
  await evidenceExpect(
    page.getByLabel('Data mode', { exact: true }),
  ).toHaveCount(0)
  await evidenceExpect(page.locator('main')).not.toContainText(
    'DEMO / SYNTHETIC',
  )
  const nav = page.getByRole('navigation', { name: 'Organization stages' })
  await nav
    .getByRole('link', { name: 'Visual exploration', exact: true })
    .click()
  await evidenceExpect(
    page.getByLabel('Artifact run', { exact: true }),
  ).toBeVisible()
  await page.getByLabel('Encoder', { exact: true }).selectOption('CLIP')
  await evidenceExpect(
    page.getByText('Coordinates were not exported for this run.', {
      exact: false,
    }),
  ).toBeVisible()
  await page.getByLabel('View', { exact: true }).selectOption('clustering')
  await evidenceExpect(
    page.getByText('Preserved membership from frozen splits.', {
      exact: false,
    }),
  ).toBeVisible()
  const grid = page.getByRole('list', { name: 'Cluster contents' })
  await grid.getByRole('button').first().click()
  await evidenceExpect(
    page.getByRole('heading', { name: 'Selected content', exact: true }),
  ).toBeVisible()
  await page.getByLabel('Cluster', { exact: true }).selectOption('-1')
  await evidenceExpect(
    page.getByText('contains unassigned contents.', { exact: false }),
  ).toBeVisible()
  await nav
    .getByRole('link', { name: 'Sequences & linkage', exact: true })
    .click()
  const player = page.getByRole('region', { name: 'Timeline frame playback' })
  await page
    .getByRole('button', { name: 'Temporal evidence', exact: true })
    .click()
  await evidenceExpect(player.getByRole('img')).toBeVisible()
  await page.getByRole('button', { name: 'Next frame', exact: true }).click()
  await evidenceExpect(player).toContainText('Point 2 /')
  await page.getByRole('button', { name: 'PLAY', exact: true }).click()
  await evidenceExpect(player).not.toContainText('Point 2 /')
  await page.getByRole('button', { name: 'PAUSE', exact: true }).click()
  await page
    .getByRole('button', { name: 'Cross-dataset linkage', exact: true })
    .click()
  await evidenceExpect(
    page.getByText(String(manifest.candidate_pair_count), { exact: true }),
  ).toBeVisible()
  await evidenceExpect(
    page.getByText('A candidate is not a confirmed match,', { exact: false }),
  ).toBeVisible()
  await page.locator('button.linkage-pair').first().click()
  await evidenceExpect(
    page.getByRole('heading', { name: 'Selected candidate', exact: true }),
  ).toBeVisible()
  await evidenceExpect(
    page.getByRole('img', { name: /candidate preview/ }).first(),
  ).toBeVisible()
  await nav.getByRole('link', { name: 'Evaluation', exact: true }).click()
  await evidenceExpect(
    page.getByText('48 / 48 verified detector runs', { exact: false }),
  ).toBeVisible()
  await page
    .getByRole('button', { name: 'Experiment provenance', exact: true })
    .click()
  await evidenceExpect(page.getByRole('dialog')).toContainText(
    '18d61b35e4c82f41b0c9092e997fb9088f7a10f5',
  )
  await page.keyboard.press('Escape')
  await page
    .getByRole('button', { name: 'Interpretation & limitations', exact: true })
    .click()
  await evidenceExpect(page.getByRole('dialog')).toContainText(
    'Causal estimate',
  )
  await evidenceExpect(page.getByRole('dialog')).toContainText('false')
  expect(errors).toEqual([])
})

test('real sequences do not require the legacy workspace snapshot', async ({
  page,
}) => {
  await page.addInitScript(() => {
    window.__FLIR_CONFIG__ = {
      dataMode: 'real',
      snapshotUrl: '/runtime/missing-workspace.json',
    }
  })
  await page.goto('/organization/sequences')
  await page
    .getByRole('button', { name: 'Temporal evidence', exact: true })
    .click()
  await evidenceExpect(
    page.getByLabel('Timeline', { exact: true }),
  ).toBeVisible()
  await evidenceExpect(
    page.getByRole('heading', { name: 'Data could not be validated' }),
  ).toHaveCount(0)
  await evidenceExpect(page.locator('.demo-bar')).not.toContainText('DEMO')
})

for (const [resource, url, message] of [
  [
    'timelines',
    '/organization/sequences?mode=sequences',
    'temporal evidence could not be loaded',
  ],
  [
    'candidate_pairs',
    '/organization/sequences?mode=linkage',
    'candidate evidence could not be loaded',
  ],
  [
    'clustering_configurations',
    '/organization/explore?view=clustering',
    'could not be loaded',
  ],
]) {
  test(`missing ${resource} is explicit, never verified or replaced with demo`, async ({
    page,
  }) => {
    await page.route(`**/organization-evidence-v2/${resource}.json`, (route) =>
      route.fulfill({ status: 404, body: '' }),
    )
    await page.goto(url)
    await evidenceExpect(
      page.getByText(message, { exact: false }),
    ).toBeVisible()
    await evidenceExpect(page.locator('.demo-bar')).not.toContainText(
      'VERIFIED',
    )
    await evidenceExpect(page.locator('main')).not.toContainText(
      'SYNTHETIC IDENTITIES',
    )
  })
}

test('real candidate splits, filters, related members, side-by-side and reference context', async ({
  page,
  request,
}) => {
  const errors: string[] = []
  page.on('pageerror', (error) => errors.push(error.message))
  const data = await (
    await request.get('/runtime/candidate-splits-view-v1.json')
  ).json()
  const chosen = data.runs.find(
    (r: { population: string; policy: string; seed: number }) =>
      r.population === 'expanded' && r.policy === 'S3' && r.seed === 0,
  )
  const other = data.runs.find(
    (r: { population: string; policy: string; seed: number }) =>
      r.population === 'expanded' && r.policy === 'S1' && r.seed === 0,
  )
  await page.goto('/organization/evaluation?view=splits')
  await evidenceExpect(
    page.getByText('16 available configurations;', { exact: false }),
  ).toBeVisible()
  await evidenceExpect(
    page.getByLabel('Data mode', { exact: true }),
  ).toHaveCount(0)
  await page
    .getByLabel('Experimental family')
    .selectOption('common_test_candidates')
  await evidenceExpect(
    page.getByText('30 available configurations;', { exact: false }),
  ).toBeVisible()
  await page.getByLabel('Population', { exact: true }).selectOption('expanded')
  await page.getByLabel('Policy', { exact: true }).selectOption('S3')
  await page.getByLabel('Split seed', { exact: true }).selectOption('0')
  await evidenceExpect(
    page.getByLabel('Left split', { exact: true }),
  ).toHaveValue(chosen.id)
  const left = page.getByRole('region', { name: 'Left split details' })
  await evidenceExpect(left).toContainText('Test status: PROPOSED')
  await evidenceExpect(
    left
      .getByRole('row')
      .filter({ has: page.getByRole('cell', { name: 'test', exact: true }) })
      .first(),
  ).toContainText('372')
  await page.getByLabel('Left partition', { exact: true }).selectOption('train')
  await page.getByLabel('Left class', { exact: true }).selectOption('0')
  await page
    .getByLabel('Left source', { exact: true })
    .selectOption('Imagenes.zip')
  await left.locator('.split-member').first().click()
  await evidenceExpect(
    left.getByRole('heading', { name: 'Selected split member' }),
  ).toBeVisible()
  await evidenceExpect(left).toContainText(
    'does not confirm a temporal sequence',
  )
  const member = chosen.members.find(
    (m: { partition: string }) => m.partition === 'train',
  )
  await page.getByLabel('Left class', { exact: true }).selectOption('all')
  await page.getByLabel('Left source', { exact: true }).selectOption('all')
  await page
    .getByLabel('Left group filter', { exact: true })
    .fill(member.group_id)
  await evidenceExpect(
    left
      .getByRole('button', {
        name: `Inspect occurrence ${member.record_id}`,
        exact: true,
      })
      .first(),
  ).toBeVisible()
  await page.getByLabel('Compare side by side').check()
  await page.getByLabel('Right split', { exact: true }).selectOption(other.id)
  await evidenceExpect(
    page.getByRole('region', { name: 'Right split details' }),
  ).toContainText('S1 / expanded / seed 0')
  await page.screenshot({
    path: 'test-results/m02-real-splits.png',
    fullPage: true,
  })
  await left.getByRole('button', { name: 'Left split provenance' }).click()
  await evidenceExpect(page.getByRole('dialog')).toContainText(
    chosen.membership_id,
  )
  await evidenceExpect(page.getByRole('dialog')).toContainText(
    'ready_for_training',
  )
  await page.keyboard.press('Escape')
  await page.getByLabel('Evaluation evidence').selectOption('reference')
  await evidenceExpect(
    page.getByRole('heading', {
      name: 'Eyes in the Sky and FLIR experimental context',
    }),
  ).toBeVisible()
  await evidenceExpect(
    page.getByRole('cell', { name: '0.721', exact: true }),
  ).toBeVisible()
  await evidenceExpect(
    page.getByRole('cell', { name: '0.663', exact: true }),
  ).toBeVisible()
  await evidenceExpect(
    page.getByRole('cell', { name: 'YOLO11n · C10', exact: true }),
  ).toBeVisible()
  await evidenceExpect(
    page.getByText('PROPOSAL-REPORTED', { exact: false }),
  ).toBeVisible()
  await evidenceExpect(
    page.getByRole('region', { name: 'Experimental variants' }),
  ).toBeVisible()
  expect(errors).toEqual([])
})

test('missing representation and candidate contracts remain unavailable even in demo workspace mode', async ({
  page,
}) => {
  await page.route('**/runtime/leakage-snapshot.json', (route) =>
    route.fulfill({ status: 404, body: '' }),
  )
  await page.goto('/organization/explore')
  await evidenceExpect(
    page.getByText('UNAVAILABLE: real representation snapshot', {
      exact: false,
    }),
  ).toBeVisible()
  await evidenceExpect(page.locator('main')).not.toContainText('DEMO')
  await page.route('**/runtime/candidate-splits-view-v1.json', (route) =>
    route.fulfill({ status: 404, body: '' }),
  )
  await page.goto('/organization/evaluation?view=splits')
  await page
    .getByLabel('Experimental family')
    .selectOption('common_test_candidates')
  await evidenceExpect(
    page.getByText('UNAVAILABLE: this split family', { exact: false }),
  ).toBeVisible()
  await evidenceExpect(page.locator('main')).not.toContainText('SYNTHETIC')
})
