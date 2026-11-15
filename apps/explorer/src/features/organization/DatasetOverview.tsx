import { Link } from 'react-router-dom'
import { useOrganizationManifest } from '@/data/organization-provider'
import { useRepresentationSnapshot } from '@/data/representation'
import { MetricSummary } from '@/components/research/Evidence'
import { TechnicalDetailsDrawer } from '@/components/research/TechnicalDetailsDrawer'
import { Notice } from '@/components/feedback/Primitives'
import { VariantRegister } from './VariantRegister'

export function DatasetOverview() {
  const organization = useOrganizationManifest()
  const representation = useRepresentationSnapshot()
  return (
    <>
      <h2>What data and analysis are available?</h2>
      <section aria-label="Organization export summary">
        <h3>REAL / EXPORTED - organization-evidence-v2</h3>
        {organization.isPending ? (
          <p role="status">Loading organization manifest...</p>
        ) : organization.isError ? (
          <Notice>
            UNAVAILABLE: organization export could not be loaded or validated.
            No synthetic substitute.
          </Notice>
        ) : (
          <>
            <MetricSummary
              items={[
                {
                  label: 'Export occurrences',
                  value: organization.data.record_count,
                },
                {
                  label: 'Export unique contents',
                  value: organization.data.unique_content_count,
                },
                {
                  label: 'Candidate pairs',
                  value: organization.data.candidate_pair_count,
                },
                {
                  label: 'Historical splits',
                  value: organization.data.split_count,
                },
              ]}
            />
            <TechnicalDetailsDrawer title="Organization export provenance">
              <pre className="audit-json">
                {JSON.stringify(organization.data, null, 2)}
              </pre>
            </TechnicalDetailsDrawer>
          </>
        )}
      </section>
      <section aria-label="Representation availability">
        <h3>Separate sampled-video representation snapshot</h3>
        {representation.isPending ? (
          <p role="status">Loading real representation evidence...</p>
        ) : representation.isError ? (
          <Notice>
            UNAVAILABLE: representation snapshot. The organization bundle
            remains independent.
          </Notice>
        ) : (
          <>
            <p>
              {representation.data.dataset.occurrences} occurrences /{' '}
              {representation.data.dataset.uniqueContents} contents. This
              population is not the labeled historical population.
            </p>
            <p>
              {['embeddings', 'similarity', 'reduction']
                .map(
                  (stage) =>
                    `${stage}: ${representation.data.runs.some((r) => r.stage === stage) ? 'EXPORTED' : 'PENDING'}`,
                )
                .join(' / ')}
            </p>
          </>
        )}
      </section>
      <nav className="toolbar" aria-label="Real evidence shortcuts">
        <Link to="/organization/explore">Representation evidence</Link>
        <Link to="/organization/explore?view=clustering">
          Stored cluster memberships
        </Link>
        <Link to="/organization/sequences">Temporal evidence</Link>
        <Link to="/organization/sequences?mode=linkage">Candidate linkage</Link>
        <Link to="/organization/evaluation?view=splits">
          Visual split explorer
        </Link>
        <Link to="/organization/evaluation">Historical detector results</Link>
      </nav>
      <Notice>
        Candidate dependency is not confirmed dependency. Cluster is not
        sequence. Frozen membership does not reconstruct the original clustering
        experiment. Missing coordinates remain unavailable. Export integrity is
        not scientific validation.
      </Notice>
      <VariantRegister />
    </>
  )
}
