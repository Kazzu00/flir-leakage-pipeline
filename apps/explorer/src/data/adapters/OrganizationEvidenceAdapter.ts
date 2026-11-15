import { z } from 'zod'
import {
  OrganizationEvidenceResourceSchemas,
  type OrganizationEvidenceResource,
} from '@/contracts/organization-evidence-v2'

export type OrganizationEvidenceFailure =
  | 'resource-missing'
  | 'resource-malformed'
  | 'schema-mismatch'
  | 'resource-unavailable'

export class OrganizationEvidenceError extends Error {
  constructor(
    readonly kind: OrganizationEvidenceFailure,
    readonly resource: OrganizationEvidenceResource,
  ) {
    super(`${kind}: ${resource}`)
    this.name = 'OrganizationEvidenceError'
  }
}

const resourceFiles: Record<OrganizationEvidenceResource, string> = {
  manifest: 'manifest.json',
  contents: 'contents.json',
  records: 'records.json',
  splits: 'splits.json',
  split_memberships: 'split_memberships.json',
  clustering_configurations: 'clustering_configurations.json',
  clusters: 'clusters.json',
  cluster_memberships: 'cluster_memberships.json',
  candidate_pairs: 'candidate_pairs.json',
  linkage_groups: 'linkage_groups.json',
  linkage_memberships: 'linkage_memberships.json',
  boundary_zones: 'boundary_zones.json',
  reviews: 'reviews.json',
  timelines: 'timelines.json',
  media: 'media.json',
}

type ResourceSchemaMap = typeof OrganizationEvidenceResourceSchemas

export type OrganizationEvidenceResourceData<
  K extends OrganizationEvidenceResource,
> = z.infer<ResourceSchemaMap[K]>

function normalizeBaseUrl(baseUrl: string) {
  const normalized = baseUrl.replace(/\/+$/, '')

  if (
    !normalized.startsWith('/') ||
    normalized.startsWith('//') ||
    normalized.includes('\\') ||
    normalized.split('/').includes('..')
  ) {
    throw new Error('invalid-organization-evidence-base-url')
  }

  return normalized
}

export class OrganizationEvidenceAdapter {
  readonly id: string
  readonly baseUrl: string

  constructor(baseUrl = '/runtime/organization-evidence-v2') {
    this.baseUrl = normalizeBaseUrl(baseUrl)
    this.id = `organization-evidence-v2:${this.baseUrl}`
  }

  getResourceUrl(resource: OrganizationEvidenceResource) {
    return `${this.baseUrl}/${resourceFiles[resource]}`
  }

  async getResource<K extends OrganizationEvidenceResource>(
    resource: K,
    signal?: AbortSignal,
  ): Promise<OrganizationEvidenceResourceData<K>> {
    const url = this.getResourceUrl(resource)

    let response: Response
    try {
      response = await fetch(url, {
        signal,
        cache: 'no-store',
        credentials: 'same-origin',
      })
    } catch {
      throw new OrganizationEvidenceError('resource-unavailable', resource)
    }

    if (response.status === 404) {
      throw new OrganizationEvidenceError('resource-missing', resource)
    }

    if (!response.ok) {
      throw new OrganizationEvidenceError('resource-unavailable', resource)
    }

    let raw: unknown
    try {
      raw = await response.json()
    } catch {
      throw new OrganizationEvidenceError('resource-malformed', resource)
    }

    const schema = OrganizationEvidenceResourceSchemas[resource]
    const result = schema.safeParse(raw)

    if (!result.success) {
      throw new OrganizationEvidenceError('schema-mismatch', resource)
    }

    return result.data as OrganizationEvidenceResourceData<K>
  }

  getManifest(signal?: AbortSignal) {
    return this.getResource('manifest', signal)
  }
}
