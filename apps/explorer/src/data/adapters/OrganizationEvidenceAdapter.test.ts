import { afterEach, describe, expect, it, vi } from 'vitest'
import {
  OrganizationEvidenceAdapter,
  OrganizationEvidenceError,
} from './OrganizationEvidenceAdapter'

// Synthetic fixture: unit tests must not depend on private runtime exports.
const syntheticManifest = {
  schema_version: 'organization-evidence-v2',
  generated_at: '2026-01-01T00:00:00Z',
  source_commit: null,
  source_dirty: null,
  dataset_id: 'SYNTHETIC',
  manifest_sha256: '0'.repeat(64),
  plan_id: 'SYNTHETIC',
  plan_sha256: '0'.repeat(64),
  runtime_freeze_sha256: '0'.repeat(64),
  record_count: 0,
  unique_content_count: 0,
  labeled_record_count: 0,
  labeled_unique_content_count: 0,
  source_video_count: 0,
  timeline_count: 0,
  split_count: 0,
  strategy_count: 0,
  candidate_pair_count: 0,
  clustering_configuration_ids: [],
  evidence_sources: [],
  limitations: ['SYNTHETIC TEST ONLY'],
  terminology: {},
  file_sha256: {},
}

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
          return new Response(JSON.stringify(syntheticManifest), {
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
