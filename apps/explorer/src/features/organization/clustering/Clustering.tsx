import { useMemo, useState } from 'react'
import {
  useOrganizationClusteringConfigurations,
  useOrganizationClusters,
  useOrganizationClusterMemberships,
  useOrganizationContents,
  useOrganizationMedia,
  useOrganizationRecords,
} from '@/data/organization-provider'
import type {
  OrganizationClusteringConfiguration,
  OrganizationCluster,
  OrganizationClusterMembership,
  OrganizationContent,
  OrganizationMedia,
  OrganizationRecord,
} from '@/contracts/organization-evidence-v2'
import { Panel, Select, Notice } from '@/components/feedback/Primitives'
import { TechnicalDetailsDrawer } from '@/components/research/TechnicalDetailsDrawer'
import { ClusterVisualGrid, type ClusterDensity } from './ClusterVisualGrid'
import { ClusterContentDetail, EvidenceFacts } from './ClusterContentDetail'

function configurationLabel(
  configuration: OrganizationClusteringConfiguration,
) {
  return [
    configuration.strategy_labels.length
      ? configuration.strategy_labels.join(', ')
      : (configuration.extractor ??
        configuration.model_id ??
        configuration.representation ??
        configurationSourceLabel(configuration.source_kind)),
    configuration.source_kind === 'frozen_split_membership' &&
    configuration.strategy_labels.length
      ? 'preserved membership'
      : null,
    configuration.algorithm?.toUpperCase(),
    `${configuration.n_clusters} ${configuration.n_clusters === 1 ? 'cluster' : 'clusters'}`,
    `${(configuration.noise_fraction * 100).toFixed(2)}% noise`,
  ]
    .filter(Boolean)
    .join(' · ')
}

function configurationSourceLabel(sourceKind: string) {
  if (sourceKind === 'frozen_split_membership')
    return 'Preserved split membership'
  if (sourceKind === 'full_clustering_artifact')
    return 'Full clustering artifact'
  return sourceKind.replaceAll('_', ' ')
}

function clusterLabel(cluster: OrganizationCluster) {
  return cluster.is_noise ? 'Noise −1' : `Cluster ${cluster.cluster_id}`
}

export function Clustering() {
  const configurations = useOrganizationClusteringConfigurations()
  const [runId, setRunId] = useState('')
  const active =
    configurations.data?.find((run) => run.cluster_run_id === runId) ??
    configurations.data?.[0]

  return (
    <section className="clustering-evidence" aria-label="Clustering evidence">
      <Notice>
        Algorithmic visual grouping · not sequence identity or ground truth.
        Cluster membership does not establish confirmed leakage or temporal
        identity.
      </Notice>
      {configurations.isPending ? (
        <p role="status">Loading clustering configurations…</p>
      ) : configurations.isError ? (
        <Notice>
          Clustering configurations could not be loaded or validated from
          organization-evidence-v2.
        </Notice>
      ) : !active ? (
        <Notice>
          No clustering configurations are available in this organization
          evidence export.
        </Notice>
      ) : (
        <>
          <div className="toolbar">
            <Select
              label="Clustering configuration"
              value={active.cluster_run_id}
              onChange={setRunId}
              options={configurations.data.map((configuration) => ({
                value: configuration.cluster_run_id,
                label: configurationLabel(configuration),
              }))}
            />
          </div>
          <Panel
            title="Configuration overview"
            meta="ORGANIZATION EVIDENCE V2 · EXPORTED"
          >
            <EvidenceFacts
              items={[
                ['Strategy label', active.strategy_labels.join(', ')],
                [
                  'Representation',
                  active.representation ?? 'Not preserved in export',
                ],
                ['Extractor', active.extractor ?? 'Not preserved in export'],
                [
                  'Algorithm',
                  active.algorithm?.toUpperCase() ?? 'Not preserved in export',
                ],
                ['Clusters', active.n_clusters],
                ['Noise contents', active.n_noise],
                [
                  'Noise fraction',
                  `${(active.noise_fraction * 100).toFixed(2)}%`,
                ],
                ['Source kind', configurationSourceLabel(active.source_kind)],
                [
                  'Original clustering artifact',
                  active.full_clustering_artifact_available
                    ? 'Included in export'
                    : 'Not included in export',
                ],
                [
                  'Stored membership consistency',
                  active.membership_consistency_verified
                    ? 'Verified in export'
                    : 'Not verified in export',
                ],
              ]}
            />
            {active.source_kind === 'frozen_split_membership' && (
              <Notice>
                Preserved membership from frozen splits. Membership consistency
                verification concerns stored relationships; it does not
                establish ground truth.
              </Notice>
            )}
            {!active.full_clustering_artifact_available && (
              <Notice>
                The original clustering artifact is not included in this export.
                Missing algorithm, representation or parameters describe what
                this export preserves, not whether clustering used them.
              </Notice>
            )}
            {(active.parameters || active.effective_parameters) && (
              <details className="cluster-disclosure">
                <summary>Parameters and effective parameters</summary>
                {active.parameters && (
                  <>
                    <h3>Parameters</h3>
                    <pre>{JSON.stringify(active.parameters, null, 2)}</pre>
                  </>
                )}
                {active.effective_parameters && (
                  <>
                    <h3>Effective parameters</h3>
                    <pre>
                      {JSON.stringify(active.effective_parameters, null, 2)}
                    </pre>
                  </>
                )}
              </details>
            )}
            <TechnicalDetailsDrawer title="Configuration provenance">
              <pre className="audit-json">
                {JSON.stringify(active, null, 2)}
              </pre>
            </TechnicalDetailsDrawer>
          </Panel>
          {/* Cluster/content identities and rendering limits are scoped to the active run. */}
          <RunExplorer
            key={active.cluster_run_id}
            runId={active.cluster_run_id}
          />
        </>
      )}
    </section>
  )
}

function RunExplorer({ runId }: { runId: string }) {
  const clustersQuery = useOrganizationClusters()
  const membershipsQuery = useOrganizationClusterMemberships()
  const contentsQuery = useOrganizationContents()
  const mediaQuery = useOrganizationMedia()
  const recordsQuery = useOrganizationRecords()
  const [clusterId, setClusterId] = useState<number | null>(null)
  const recordsByContentId = useMemo(() => {
    const index = new Map<string, OrganizationRecord[]>()
    for (const record of recordsQuery.data ?? []) {
      const records = index.get(record.content_id) ?? []
      records.push(record)
      index.set(record.content_id, records)
    }
    return index
  }, [recordsQuery.data])

  const clusters = useMemo(
    () =>
      (clustersQuery.data ?? [])
        .filter((cluster) => cluster.cluster_run_id === runId)
        .sort(
          (a, b) =>
            Number(a.is_noise) - Number(b.is_noise) ||
            a.cluster_id - b.cluster_id,
        ),
    [clustersQuery.data, runId],
  )
  const contentsById = useMemo(
    () =>
      new Map(
        (contentsQuery.data ?? []).map((content) => [
          content.content_id,
          content,
        ]),
      ),
    [contentsQuery.data],
  )
  const mediaByPreviewKey = useMemo(
    () =>
      new Map(
        (mediaQuery.data ?? []).map((media) => [media.preview_key, media]),
      ),
    [mediaQuery.data],
  )
  const runMemberships = useMemo(
    () =>
      (membershipsQuery.data ?? []).filter(
        (member) => member.cluster_run_id === runId,
      ),
    [membershipsQuery.data, runId],
  )
  const membershipsByCluster = useMemo(() => {
    const groups = new Map<number, OrganizationClusterMembership[]>()
    for (const member of runMemberships) {
      const group = groups.get(member.cluster_id) ?? []
      group.push(member)
      groups.set(member.cluster_id, group)
    }
    return groups
  }, [runMemberships])

  const integrityError = useMemo(() => {
    // These are identity joins, not a re-execution or scientific validation of clustering.
    if (contentsQuery.data && contentsById.size !== contentsQuery.data.length)
      return 'Duplicate content identities.'
    const clustersById = new Map(
      clusters.map((cluster) => [cluster.cluster_id, cluster]),
    )
    if (clustersById.size !== clusters.length)
      return 'Duplicate cluster identities in the selected run.'
    if (
      clusters.some(
        (cluster) => cluster.is_noise !== (cluster.cluster_id === -1),
      )
    )
      return 'Conflicting noise identities.'
    const seen = new Set<string>()
    for (const member of runMemberships) {
      const cluster = clustersById.get(member.cluster_id)
      if (seen.has(member.content_id))
        return 'More than one membership row for a content in the selected run.'
      if (!cluster || cluster.is_noise !== member.is_noise)
        return 'Membership references an unavailable or inconsistent cluster.'
      if (!contentsById.has(member.content_id))
        return 'Membership references an unavailable content.'
      seen.add(member.content_id)
    }
    return null
  }, [clusters, contentsById, contentsQuery.data, runMemberships])

  const active =
    clusters.find((cluster) => cluster.cluster_id === clusterId) ?? clusters[0]
  const memberships = active
    ? (membershipsByCluster.get(active.cluster_id) ?? [])
    : []

  if (
    [clustersQuery, membershipsQuery, contentsQuery].some(
      (query) => query.isPending,
    )
  ) {
    return <p role="status">Loading cluster memberships and contents…</p>
  }
  if (
    [clustersQuery, membershipsQuery, contentsQuery].some(
      (query) => query.isError,
    )
  ) {
    return (
      <Notice>
        Cluster evidence could not be loaded or validated from
        organization-evidence-v2. Configuration provenance remains available.
      </Notice>
    )
  }
  if (integrityError)
    return (
      <Notice>
        Cluster evidence has inconsistent identities: {integrityError}
      </Notice>
    )
  if (!active)
    return <Notice>No clusters are exported for this configuration.</Notice>

  return (
    <>
      <div className="toolbar">
        <Select
          label="Cluster"
          value={String(active.cluster_id)}
          onChange={(value) => setClusterId(Number(value))}
          options={clusters.map((cluster) => ({
            value: String(cluster.cluster_id),
            label: `${clusterLabel(cluster)} · ${cluster.size_unique_contents} contents`,
          }))}
        />
      </div>
      {active.is_noise && (
        <Notice>
          Noise −1 contains unassigned contents. It is not a single cluster or
          an indivisible allocation group.
        </Notice>
      )}
      {active.size_unique_contents !== memberships.length && (
        <Notice>
          The exported cluster size ({active.size_unique_contents}) differs from
          the available membership rows ({memberships.length}). This view shows
          the available rows.
        </Notice>
      )}
      {mediaQuery.isPending && <p role="status">Loading preview metadata…</p>}
      {mediaQuery.isError && (
        <Notice>
          Preview metadata could not be loaded or validated. Cluster membership
          and content evidence remain available.
        </Notice>
      )}
      <ClusterInspection
        key={active.cluster_id}
        cluster={active}
        memberships={memberships}
        contentsById={contentsById}
        mediaByPreviewKey={mediaByPreviewKey}
        recordsByContentId={recordsByContentId}
        recordsState={recordsQuery.status}
      />
    </>
  )
}

function ClusterInspection({
  cluster,
  memberships,
  contentsById,
  mediaByPreviewKey,
  recordsByContentId,
  recordsState,
}: {
  cluster: OrganizationCluster
  memberships: OrganizationClusterMembership[]
  contentsById: Map<string, OrganizationContent>
  mediaByPreviewKey: Map<string, OrganizationMedia>
  recordsByContentId: Map<string, OrganizationRecord[]>
  recordsState: 'pending' | 'error' | 'success'
}) {
  const [selectedContentId, setSelectedContentId] = useState('')
  const [density, setDensity] = useState<ClusterDensity>('compact')
  const membersById = useMemo(
    () => new Map(memberships.map((member) => [member.content_id, member])),
    [memberships],
  )
  const selectedMembership = membersById.get(selectedContentId)
  const selectedContent = selectedMembership
    ? contentsById.get(selectedContentId)
    : undefined

  return (
    <>
      <details className="cluster-disclosure">
        <summary>Cluster provenance</summary>
        <p>
          Reported timeline spans describe temporal context, not sequence
          identities.
        </p>
        <pre>{JSON.stringify(cluster, null, 2)}</pre>
      </details>
      <div className="cluster-inspection-layout">
        <Panel
          title="Cluster visual explorer"
          meta={`${clusterLabel(cluster)} · ${memberships.length} available contents`}
        >
          <div className="cluster-inspection-toolbar">
            <Select
              label="Density"
              value={density}
              onChange={(value) => setDensity(value as ClusterDensity)}
              options={[
                { value: 'compact', label: 'Compact' },
                { value: 'detailed', label: 'Detailed' },
              ]}
            />
            <a className="cluster-detail-link" href="#cluster-selected-content">
              View selected content
            </a>
          </div>
          <ClusterVisualGrid
            memberships={memberships}
            contentsById={contentsById}
            mediaByPreviewKey={mediaByPreviewKey}
            selectedContentId={selectedContentId}
            onSelect={setSelectedContentId}
            density={density}
          />
        </Panel>
        <div
          id="cluster-selected-content"
          className="cluster-selected-content"
          tabIndex={-1}
        >
          {selectedContent && selectedMembership ? (
            <ClusterContentDetail
              key={selectedContentId}
              content={selectedContent}
              membership={selectedMembership}
              media={mediaByPreviewKey.get(selectedContent.preview_key)}
              records={recordsByContentId.get(selectedContentId) ?? []}
              recordsState={recordsState}
            />
          ) : (
            <Panel title="Selected content">
              <p className="panel-body muted">
                Select a content thumbnail to inspect its stored evidence.
              </p>
            </Panel>
          )}
        </div>
      </div>
    </>
  )
}
