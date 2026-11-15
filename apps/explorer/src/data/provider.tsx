import {
  createContext,
  useContext,
  useMemo,
  useState,
  type ReactNode,
} from 'react'
import { useQuery } from '@tanstack/react-query'
import type { DataAdapter } from './adapters/types'
import { MockDataAdapter } from './adapters/MockDataAdapter'
import { CompositeDataAdapter } from './adapters/CompositeDataAdapter'
import { RealLeakageAdapter } from './adapters/RealLeakageAdapter'
import { resolveRuntimeConfig } from './runtime-config'
export const DataModeContext = createContext<{
  mode: 'demo' | 'real'
  setMode: (mode: 'demo' | 'real') => void
}>({ mode: 'demo', setMode: () => {} })
export function RuntimeDataProvider({ children }: { children: ReactNode }) {
  const [config] = useState(() =>
    resolveRuntimeConfig(window.__FLIR_CONFIG__, import.meta.env),
  )
  const [mode, setMode] = useState<'demo' | 'real'>(
    config.success ? config.data.dataMode : 'demo',
  )
  const adapter = useMemo(
    () =>
      !config.success
        ? {
            id: 'invalid-runtime-configuration',
            getSnapshot: async () => {
              throw new Error('invalid-runtime-configuration')
            },
          }
        : mode === 'real'
          ? new CompositeDataAdapter(
              new RealLeakageAdapter(config.data.snapshotUrl),
            )
          : new MockDataAdapter(),
    [mode, config],
  )
  return (
    <DataModeContext.Provider value={{ mode, setMode }}>
      <DataAdapterContext.Provider value={adapter}>
        {children}
      </DataAdapterContext.Provider>
    </DataModeContext.Provider>
  )
}
export function DataModeSwitch() {
  const { mode, setMode } = useContext(DataModeContext)
  return (
    <label className="select-field">
      <span>Data mode</span>
      <select
        aria-label="Data mode"
        value={mode}
        onChange={(e) => setMode(e.target.value === 'real' ? 'real' : 'demo')}
      >
        <option value="demo">DEMO</option>
        <option value="real">REAL / MIXED</option>
      </select>
    </label>
  )
}
export const DataAdapterContext = createContext<DataAdapter>(
  new MockDataAdapter(),
)
export function useSnapshot(enabled = true) {
  const adapter = useContext(DataAdapterContext)
  return useQuery({
    queryKey: ['snapshot', adapter.id],
    queryFn: ({ signal }) => adapter.getSnapshot(signal),
    staleTime: Infinity,
    enabled,
  })
}
