import { useQuery } from '@tanstack/react-query'
import { loadDetectionExport } from './adapter'

// Only the explicitly approved lightweight contract enters the build. Public runtime
// snapshots remain excluded. Hash-named assets also prevent stale mixed-file releases.
const assets = import.meta.glob<string>(
  [
    './snapshot/manifest.json',
    './snapshot/summary.json',
    './snapshot/strategies.json',
    './snapshot/splits.json',
    './snapshot/runs.json',
    './snapshot/classes.json',
    './snapshot/support.json',
    './snapshot/variance.json',
    './snapshot/associations.json',
    './snapshot/bootstrap.json',
    './snapshot/schema/detection-export-v1.schema.json',
  ],
  {
    eager: true,
    query: '?url&no-inline',
    import: 'default',
  },
)
export const detectionUrls = Object.fromEntries(
  Object.entries(assets).map(([path, url]) => [
    path.replace('./snapshot/', ''),
    url,
  ]),
)
export function useDetectionSnapshot(enabled = true) {
  return useQuery({
    queryKey: ['detection-export-v1', detectionUrls['manifest.json']],
    queryFn: ({ signal }) => loadDetectionExport(detectionUrls, signal),
    staleTime: Infinity,
    retry: false,
    enabled,
  })
}
