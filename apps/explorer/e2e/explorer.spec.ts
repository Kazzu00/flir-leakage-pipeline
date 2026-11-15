import { test, expect } from './synthetic-fixture'
import AxeBuilder from '@axe-core/playwright'
test('home, module navigation and core clustering interaction', async ({
  page,
}) => {
  const errors: string[] = []
  page.on('pageerror', (e) => errors.push(e.message))
  await page.goto('/')
  await expect(
    page.getByRole('heading', { name: 'One pipeline. Three perspectives.' }),
  ).toBeVisible()
  await page.screenshot({
    path: 'test-results/home-desktop.png',
    fullPage: true,
  })
  await page.getByRole('link', { name: 'Explore module' }).nth(1).click()
  await page
    .getByRole('navigation', { name: 'Organization stages' })
    .getByRole('link', { name: 'Visual exploration', exact: true })
    .click()
  await page.getByLabel('View', { exact: true }).selectOption('clustering')
  await expect(
    page.getByLabel('Clustering configuration', { exact: true }),
  ).toBeVisible()
  await expect(page.getByLabel('Encoder', { exact: true })).toHaveCount(0)
  const cluster = page.getByLabel('Cluster', { exact: true })
  await expect(cluster).toBeVisible()
  const options = await cluster
    .locator('option')
    .evaluateAll((elements) =>
      elements.map((option) => (option as HTMLOptionElement).value),
    )
  await cluster.selectOption(options.at(-1)!)
  const grid = page.getByRole('list', { name: 'Cluster contents' })
  const content = grid.getByRole('button').first()
  await expect(content).toBeVisible()
  await content.scrollIntoViewIfNeeded()
  const preview = content.getByRole('img')
  if (await preview.count()) {
    await expect
      .poll(
        () =>
          preview.evaluate(
            (image: HTMLImageElement) =>
              image.complete && image.naturalWidth > 0,
          ),
        { timeout: 20000 },
      )
      .toBe(true)
  } else {
    await expect(
      content.getByText('Preview unavailable', { exact: true }),
    ).toBeVisible()
  }
  expect(await grid.getByRole('button').count()).toBeLessThanOrEqual(60)
  await content.focus()
  await page.keyboard.press('Enter')
  await expect(content).toHaveAttribute('aria-pressed', 'true')
  const contentId = (await content.getAttribute('aria-label'))!.replace(
    'Inspect content ',
    '',
  )
  const selected = page.locator('section.panel').filter({
    has: page.getByRole('heading', { name: 'Selected content', exact: true }),
  })
  await expect(selected.getByText(contentId, { exact: true })).toBeVisible()
  await expect(
    selected
      .getByRole('region', { name: 'Content occurrences' })
      .getByRole('listitem')
      .first(),
  ).toBeVisible()
  await page
    .locator('.cluster-inspection-layout')
    .evaluate((element) =>
      window.scrollTo(
        0,
        window.scrollY + element.getBoundingClientRect().top + 40,
      ),
    )
  await expect
    .poll(() =>
      page
        .locator('#cluster-selected-content')
        .evaluate((element) => Math.round(element.getBoundingClientRect().top)),
    )
    .toBe(16)
  await page.screenshot({
    path: 'test-results/clustering-desktop.png',
    fullPage: false,
  })
  await expect(
    page.getByRole('button', { name: 'BEFORE SPLIT', exact: true }),
  ).toHaveCount(0)
  await expect(
    page.getByRole('region', { name: 'Clustering evidence' }),
  ).not.toContainText(/DEMO|SYNTHETIC IDENTITIES|Status mock/)
  expect(errors).toEqual([])
})
test('split comparison and prediction overlay', async ({ page }) => {
  await page.goto('/organization/splits')
  await expect(
    page.getByRole('heading', {
      name: 'Visual inspection of real split memberships',
    }),
  ).toBeVisible()
  await expect(page.getByLabel('Left split', { exact: true })).toBeVisible()
  await page.goto('/segmentation/predictions')
  await page.getByLabel('Comparison layout').selectOption('Overlay')
  await page.getByRole('slider', { name: 'Overlay opacity' }).fill('70')
  await expect(
    page.getByText('Illustrative overlay · opacity 70%'),
  ).toBeVisible()
})
test('mobile navigation, keyboard and responsive layout', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await page.goto('/')
  await page.getByRole('button', { name: 'Open navigation' }).click()
  await page
    .getByRole('navigation', { name: 'Main navigation' })
    .getByRole('link', { name: 'Experiments' })
    .click()
  await expect(page.getByRole('heading', { name: 'Experiments' })).toBeVisible()
  await expect(
    page.getByRole('button', { name: 'Open navigation' }),
  ).toBeVisible()
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBe(true)
  await page.screenshot({ path: 'test-results/mobile.png', fullPage: true })
  await page.keyboard.press('Tab')
  await expect(page.locator(':focus')).toBeVisible()
})
test('all primary and stage routes render without runtime errors', async ({
  page,
}) => {
  const errors: string[] = []
  page.on('pageerror', (e) => errors.push(e.message))
  for (const route of [
    '/',
    '/preprocessing',
    '/preprocessing/frames',
    '/preprocessing/hud',
    '/preprocessing/inpainting',
    '/preprocessing/denoising',
    '/preprocessing/quality',
    '/organization',
    '/organization/dataset',
    '/organization/embeddings',
    '/organization/similarity',
    '/organization/reduction',
    '/organization/clustering',
    '/organization/sequences',
    '/organization/evaluation',
    '/organization/explore',
    '/organization/groups',
    '/organization/splits',
    '/organization/detector',
    '/segmentation',
    '/segmentation/points',
    '/segmentation/model',
    '/segmentation/training',
    '/segmentation/predictions',
    '/segmentation/evaluation',
    '/experiments',
    '/evaluation',
  ]) {
    await page.goto(route)
    await expect(page.locator('main h1')).toBeVisible()
    await expect(page.getByText('Data could not be validated')).toHaveCount(0)
  }
  expect(errors).toEqual([])
})
test('axe smoke: primary views in both themes', async ({ page }) => {
  for (const route of [
    '/',
    '/organization/clustering',
    '/organization/sequences',
    '/organization/evaluation',
    '/organization/explore',
    '/organization/splits',
    '/segmentation/predictions',
    '/experiments',
  ]) {
    await page.goto(route)
    await expect(page.locator('main h1')).toBeVisible()
    for (const theme of ['dark', 'light']) {
      if (theme === 'light')
        await page
          .getByRole('button', { name: 'Switch to light theme' })
          .click()
      const result = await new AxeBuilder({ page })
        .withTags(['wcag2a', 'wcag2aa', 'wcag21aa', 'wcag22aa'])
        .analyze()
      expect(
        result.violations,
        `${route} ${theme}: ${JSON.stringify(result.violations.map((v) => ({ id: v.id, nodes: v.nodes.map((n) => n.target) })))}`,
      ).toEqual([])
      if (theme === 'light')
        await page.getByRole('button', { name: 'Switch to dark theme' }).click()
    }
  }
})
test('clustering previews and selected evidence remain contained on desktop and mobile', async ({
  page,
}) => {
  for (const width of [390, 1024, 1360, 1440, 1920]) {
    await page.setViewportSize({ width, height: 1080 })
    await page.goto('/organization/clustering')
    await expect(
      page.getByRole('heading', {
        name: 'Cluster visual explorer',
        exact: true,
      }),
    ).toBeVisible()
    const density = page.getByLabel('Density', { exact: true })
    await expect(density).toHaveValue('compact')
    const content = page
      .getByRole('list', { name: 'Cluster contents' })
      .getByRole('button')
      .first()
    await expect(content).toBeVisible()
    await content.click()
    await expect(content).toHaveAttribute('aria-pressed', 'true')
    const grid = page.getByRole('list', { name: 'Cluster contents' })
    const geometry = await grid.evaluate((element) => {
      const detail = document.querySelector('#cluster-selected-content')!
      const availableWidth = document
        .querySelector('.clustering-evidence')!
        .getBoundingClientRect().width
      return {
        availableWidth,
        columns:
          getComputedStyle(element).gridTemplateColumns.split(' ').length,
        sticky: getComputedStyle(detail).position,
        gridRight: element.getBoundingClientRect().right,
        detailLeft: detail.getBoundingClientRect().left,
        detailWidth: detail.getBoundingClientRect().width,
        previewWidth: detail
          .querySelector('.cluster-preview')!
          .getBoundingClientRect().width,
      }
    })
    const sideBySide = geometry.availableWidth > 900
    expect(geometry.sticky).toBe(sideBySide ? 'sticky' : 'static')
    expect(geometry.previewWidth).toBeLessThanOrEqual(220)
    expect((await density.boundingBox())!.width).toBeLessThanOrEqual(150)
    if (sideBySide) {
      expect(geometry.columns).toBeGreaterThanOrEqual(6)
      expect(geometry.columns).toBeLessThanOrEqual(8)
      expect(geometry.detailLeft).toBeGreaterThan(geometry.gridRight)
      expect(geometry.detailWidth).toBeGreaterThanOrEqual(320)
      expect(geometry.detailWidth).toBeLessThanOrEqual(380)
    }
    const count = await grid.getByRole('button').count()
    await density.selectOption('detailed')
    await expect(content.locator('.cluster-tile-text')).toBeVisible()
    await expect(content).toHaveAttribute('aria-pressed', 'true')
    await expect(grid.getByRole('button')).toHaveCount(count)
    await density.selectOption('compact')
    await expect(content.locator('.cluster-tile-text')).toHaveCount(0)
    await expect(content).toHaveAttribute('aria-pressed', 'true')
    if (width === 1440) {
      await page.screenshot({
        path: 'test-results/clustering-compact-desktop.png',
      })
    }
    const contentId = (await content.getAttribute('aria-label'))!.replace(
      'Inspect content ',
      '',
    )
    const detailLink = page.getByRole('link', {
      name: 'View selected content',
      exact: true,
    })
    if (sideBySide) {
      await expect(detailLink).toBeHidden()
      const canScrollGrid = await page
        .locator('.cluster-inspection-layout')
        .evaluate(
          (element) =>
            element.getBoundingClientRect().height -
              element
                .querySelector('#cluster-selected-content')!
                .getBoundingClientRect().height >
            80,
        )
      if (canScrollGrid) {
        await page
          .locator('.cluster-inspection-layout')
          .evaluate((element) =>
            window.scrollTo(
              0,
              window.scrollY + element.getBoundingClientRect().top + 40,
            ),
          )
        const detailTop = await page
          .locator('#cluster-selected-content')
          .evaluate((element) => element.getBoundingClientRect().top)
        expect(detailTop).toBeCloseTo(16, 0)
      }
    } else {
      await detailLink.click()
      await expect(page.locator('#cluster-selected-content')).toBeFocused()
    }
    const selected = page.locator('section.panel').filter({
      has: page.getByRole('heading', {
        name: 'Selected content',
        exact: true,
      }),
    })
    await expect(selected.getByText(contentId, { exact: true })).toBeVisible()
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= window.innerWidth,
      ),
      `width ${width}`,
    ).toBe(true)
    if (width === 390) {
      await page.screenshot({ path: 'test-results/clustering-mobile.png' })
    }
  }
})
