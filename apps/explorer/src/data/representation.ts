import { useQuery } from '@tanstack/react-query'
import { RealLeakageAdapter } from './adapters/RealLeakageAdapter'
import { resolveRuntimeConfig } from './runtime-config'

/** Representation evidence is real-only, independently of M01/M03 workspace mode. */
export function useRepresentationSnapshot(enabled = true) {
  const config = resolveRuntimeConfig(window.__FLIR_CONFIG__, import.meta.env)
  const url = config.success ? config.data.snapshotUrl : null
  return useQuery({
    queryKey: ['m02-representation', url],
    enabled,
    staleTime: Infinity,
    queryFn: ({ signal }) => {
      if (!url) throw new Error('invalid-runtime-configuration')
      return new RealLeakageAdapter(url).getLeakage(signal)
    },
  })
}
