import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { afterEach, describe, expect, it, vi } from 'vitest'
import {
  OrganizationEvidenceAdapter,
  OrganizationEvidenceError,
} from './OrganizationEvidenceAdapter'

const manifestPath = resolve(
  process.cwd(),
  'public/runtime/organization-evidence-v2/manifest.json',
)

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('OrganizationEvidenceAdapter', () => {
  it('loads and validates only the requested manifest resource', async () => {
    const requested: string[] = []

    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input)
        requested.push(url)

        if (url === '/runtime/organization-evidence-v2/manifest.json') {
          return new Response(readFileSync(manifestPath, 'utf8'), {
            status: 200,
            headers: { 'content-type': 'application/json' },
          })
        }

        return new Response('', { status: 404 })
      }),
    )

    const adapter = new OrganizationEvidenceAdapter()
    const manifest = await adapter.getManifest()

    expect(manifest.schema_version).toBe('organization-evidence-v2')
    expect(manifest.record_count).toBeGreaterThanOrEqual(0)

    expect(requested).toEqual([
      '/runtime/organization-evidence-v2/manifest.json',
    ])
  })

  it('supports a configurable bundle base URL', () => {
    const adapter = new OrganizationEvidenceAdapter(
      '/runtime/another-organization-bundle',
    )

    expect(adapter.getResourceUrl('timelines')).toBe(
      '/runtime/another-organization-bundle/timelines.json',
    )
  })

  it('rejects unsafe bundle base URLs', () => {
    expect(
      () => new OrganizationEvidenceAdapter('/runtime/../private'),
    ).toThrow('invalid-organization-evidence-base-url')

    expect(
      () => new OrganizationEvidenceAdapter('//external.example/data'),
    ).toThrow('invalid-organization-evidence-base-url')
  })

  it('reports missing resources explicitly', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => new Response('', { status: 404 })),
    )

    const adapter = new OrganizationEvidenceAdapter()

    await expect(adapter.getResource('timelines')).rejects.toEqual(
      expect.objectContaining<Partial<OrganizationEvidenceError>>({
        kind: 'resource-missing',
        resource: 'timelines',
      }),
    )
  })
})
