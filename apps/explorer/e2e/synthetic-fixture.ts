import { test as base, expect } from '@playwright/test'
import { readFileSync } from 'node:fs'
import { resolve, sep } from 'node:path'

/** Real contract transport with synthetic backend fixtures only. No FAC mounts. */
export const test = base.extend<{
  organizationEvidence: void
  organizationScenario: 'core' | 'with_candidates'
}>({
  organizationScenario: ['core', { option: true }],
  organizationEvidence: [
    async ({ page, organizationScenario }, use) => {
      const root = process.env.FLIR_E2E_SYNTHETIC_BUNDLE
      if (!root)
        throw new Error('Synthetic organization fixture was not prepared')
      const directory = resolve(
        root,
        organizationScenario === 'with_candidates'
          ? 'export-with-candidates'
          : 'export',
      )
      const manifest = JSON.parse(
        readFileSync(resolve(directory, 'manifest.json'), 'utf8'),
      ) as { file_sha256: Record<string, string> }
      const names = new Set([
        'manifest.json',
        ...Object.keys(manifest.file_sha256),
      ])
      await page.route(
        '**/runtime/organization-evidence-v2/*',
        async (route) => {
          const name = new URL(route.request().url()).pathname
            .split('/')
            .at(-1)!
          if (!names.has(name)) return route.fulfill({ status: 404 })
          await route.fulfill({ path: resolve(directory, name) })
        },
      )
      await page.route('**/runtime/organization-media/**', async (route) => {
        const relative = decodeURIComponent(
          new URL(route.request().url()).pathname.split(
            '/organization-media/',
          )[1],
        )
        const media = resolve(root, 'media')
        const path = resolve(media, relative)
        if (!path.startsWith(media + sep))
          throw new Error('Synthetic media path escaped its fixture')
        await route.fulfill({ path })
      })
      await use()
    },
    { auto: true },
  ],
})

export { expect }
