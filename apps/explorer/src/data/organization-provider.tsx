import {
  createContext,
  useContext,
  useMemo,
  type ReactNode,
} from 'react'
import { useQuery } from '@tanstack/react-query'
import type { OrganizationEvidenceResource } from '@/contracts/organization-evidence-v2'
import {
  OrganizationEvidenceAdapter,
  type OrganizationEvidenceResourceData,
} from './adapters/OrganizationEvidenceAdapter'

const OrganizationEvidenceContext =
  createContext<OrganizationEvidenceAdapter | null>(null)

export function OrganizationEvidenceProvider({
  children,
  baseUrl = '/runtime/organization-evidence-v2',
}: {
  children: ReactNode
  baseUrl?: string
}) {
  const adapter = useMemo(
    () => new OrganizationEvidenceAdapter(baseUrl),
    [baseUrl],
  )

  return (
    <OrganizationEvidenceContext.Provider value={adapter}>
      {children}
    </OrganizationEvidenceContext.Provider>
  )
}

export function useOrganizationEvidenceAdapter() {
  const adapter = useContext(OrganizationEvidenceContext)

  if (!adapter) {
    throw new Error('OrganizationEvidenceProvider is missing')
  }

  return adapter
}

export function useOrganizationResource<
  K extends OrganizationEvidenceResource,
>(resource: K, enabled = true) {
  const adapter = useOrganizationEvidenceAdapter()

  return useQuery<OrganizationEvidenceResourceData<K>>({
    queryKey: ['organization-evidence-v2', adapter.id, resource],
    queryFn: ({ signal }) => adapter.getResource(resource, signal),
    staleTime: Infinity,
    enabled,
  })
}

export function useOrganizationManifest(enabled = true) {
  return useOrganizationResource('manifest', enabled)
}

export function useOrganizationContents(enabled = true) {
  return useOrganizationResource('contents', enabled)
}

export function useOrganizationRecords(enabled = true) {
  return useOrganizationResource('records', enabled)
}

export function useOrganizationTimelines(enabled = true) {
  return useOrganizationResource('timelines', enabled)
}

export function useOrganizationMedia(enabled = true) {
  return useOrganizationResource('media', enabled)
}

export function useOrganizationCandidatePairs(enabled = true) {
  return useOrganizationResource('candidate_pairs', enabled)
}

export function useOrganizationClusters(enabled = true) {
  return useOrganizationResource('clusters', enabled)
}

export function useOrganizationClusterMemberships(enabled = true) {
  return useOrganizationResource('cluster_memberships', enabled)
}

export function useOrganizationClusteringConfigurations(enabled = true) {
  return useOrganizationResource('clustering_configurations', enabled)
}

export function useOrganizationSplits(enabled = true) {
  return useOrganizationResource('splits', enabled)
}

export function useOrganizationSplitMemberships(enabled = true) {
  return useOrganizationResource('split_memberships', enabled)
}

export function useOrganizationBoundaryZones(enabled = true) {
  return useOrganizationResource('boundary_zones', enabled)
}

export function useOrganizationLinkageGroups(enabled = true) {
  return useOrganizationResource('linkage_groups', enabled)
}

export function useOrganizationLinkageMemberships(enabled = true) {
  return useOrganizationResource('linkage_memberships', enabled)
}

export function useOrganizationReviews(enabled = true) {
  return useOrganizationResource('reviews', enabled)
}
