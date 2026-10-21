import {
  BrowserRouter,
  Routes,
  Route,
  Link,
  useLocation,
} from 'react-router-dom'
import { isDetectionRoute } from '@/features/detection/route'
import { lazy, Suspense } from 'react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { AppShell } from '@/components/layout/AppShell'
import { Overview } from '@/features/overview/Overview'
const Preprocessing = lazy(() =>
  import('@/features/preprocessing/Preprocessing').then((m) => ({
    default: m.Preprocessing,
  })),
)
const Organization = lazy(() =>
  import('@/features/organization/Organization').then((m) => ({
    default: m.Organization,
  })),
)
const Segmentation = lazy(() =>
  import('@/features/segmentation/Segmentation').then((m) => ({
    default: m.Segmentation,
  })),
)
const Experiments = lazy(() =>
  import('@/features/experiments/Experiments').then((m) => ({
    default: m.Experiments,
  })),
)
import { Evaluation } from '@/features/overview/Evaluation'
import {
  useSnapshot,
  RuntimeDataProvider,
  DataModeSwitch,
} from '@/data/provider'
import { SnapshotError } from '@/data/adapters/RealLeakageAdapter'
const client = new QueryClient({ defaultOptions: { queries: { retry: 1 } } })
function DataGate() {
  const location = useLocation()
  const detector = isDetectionRoute(location.pathname, location.search)
  const independentEvidence =
    detector || /^\/organization(?:\/|$)/.test(location.pathname)
  const { isPending, error, refetch } = useSnapshot(!independentEvidence)
  if (isPending && !independentEvidence)
    return (
      <p className="empty" role="status">
        Loading research workspace…
      </p>
    )
  if (error && !independentEvidence)
    return (
      <div className="empty" role="alert">
        <h1>Data could not be validated</h1>
        <p>
          {error instanceof SnapshotError
            ? error.kind
            : error.message === 'invalid-runtime-configuration'
              ? 'invalid-runtime-configuration'
              : 'snapshot-unavailable'}
          . No artifact data has been substituted with demo results.
        </p>
        <button onClick={() => void refetch()}>Retry</button>
        <DataModeSwitch />
      </div>
    )
  return (
    <Suspense
      fallback={
        <p className="empty" role="status">
          Loading module…
        </p>
      }
    >
      <Routes>
        <Route element={<AppShell />}>
          <Route index element={<Overview />} />
          <Route path="preprocessing/:stage?" element={<Preprocessing />} />
          <Route path="organization/:stage?" element={<Organization />} />
          <Route path="segmentation/:stage?" element={<Segmentation />} />
          <Route path="experiments" element={<Experiments />} />
          <Route path="evaluation" element={<Evaluation />} />
          <Route
            path="*"
            element={
              <>
                <h1>Page not found</h1>
                <Link to="/">Return to pipeline</Link>
              </>
            }
          />
        </Route>
      </Routes>
    </Suspense>
  )
}
export function App() {
  return (
    <QueryClientProvider client={client}>
      <BrowserRouter>
        <RuntimeDataProvider>
          <DataGate />
        </RuntimeDataProvider>
      </BrowserRouter>
    </QueryClientProvider>
  )
}
