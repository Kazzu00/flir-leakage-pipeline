import { lazy, Suspense } from 'react'
import { NavLink, useParams, Navigate, useLocation } from 'react-router-dom'
import { organizationTabs, legacyOrganizationRoutes } from '@/app/navigation'
import { PageTitle } from '@/components/feedback/Primitives'
import { OrganizationEvidenceProvider } from '@/data/organization-provider'
import { DatasetOverview } from './DatasetOverview'
const VisualExplorer = lazy(() =>
  import('./VisualExplorer').then((m) => ({ default: m.VisualExplorer })),
)
const SequenceExplorer = lazy(() =>
  import('./SequenceExplorer').then((m) => ({ default: m.SequenceExplorer })),
)
const EvaluationOverview = lazy(() =>
  import('./EvaluationOverview').then((m) => ({
    default: m.EvaluationOverview,
  })),
)
export function Organization() {
  const { stage = '' } = useParams()
  const location = useLocation()
  const legacy = legacyOrganizationRoutes[stage]
  if (legacy)
    return (
      <Navigate
        replace
        to={legacy + (legacy.includes('?') ? '' : location.search)}
      />
    )
  const title =
    organizationTabs.find(([id]) => id === stage)?.[1] ?? 'Stage unavailable'
  return (
    <OrganizationEvidenceProvider>
      <PageTitle
        eyebrow="MODULE 02 / RESEARCH EXPLORATION"
        title={stage ? title : 'Representation & organization'}
        description="Explore visual structure, temporal evidence and downstream consequences."
      />
      <nav className="tabs-nav" aria-label="Organization stages">
        {organizationTabs.map(([id, name]) => (
          <NavLink end key={id} to={`/organization${id ? `/${id}` : ''}`}>
            {name}
          </NavLink>
        ))}
      </nav>
      <Suspense fallback={<p role="status">Loading explorer…</p>}>
        {stage === 'explore' ? (
          <VisualExplorer />
        ) : stage === 'sequences' ? (
          <SequenceExplorer />
        ) : stage === 'evaluation' ? (
          <EvaluationOverview />
        ) : stage ? (
          <Navigate to="/organization" replace />
        ) : (
          <DatasetOverview />
        )}
      </Suspense>
    </OrganizationEvidenceProvider>
  )
}


