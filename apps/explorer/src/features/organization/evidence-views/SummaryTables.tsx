import { useState } from 'react'
import type { ArtifactRun } from '@/contracts/leakage'
import { Select } from '@/components/feedback/Primitives'
import { Pager } from './Pager'
export function SummaryTables({ run }: { run: ArtifactRun }) {
  const [scope, setScope] = useState(run.summaries[0]?.scope ?? 'source')
  const [page, setPage] = useState(0)
  const rows = run.summaries.filter((s) => s.scope === scope)
  return (
    <div className="panel-body">
      <Select
        label="Summary scope"
        value={scope}
        onChange={(v) => {
          setScope(v as typeof scope)
          setPage(0)
        }}
        options={[...new Set(run.summaries.map((s) => s.scope))].map((v) => ({
          value: v,
          label: v,
        }))}
      />
      <p>
        Persisted rows; numeric fields are allowlisted. Sampling gaps do not
        establish sequence identity.
      </p>
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              <th>Row</th>
              <th>Reported values</th>
            </tr>
          </thead>
          <tbody>
            {rows.slice(page * 20, page * 20 + 20).map((r) => (
              <tr key={r.index}>
                <td>{r.label ?? r.configurationId ?? r.index + 1}</td>
                <td>
                  {Object.entries(r.metrics)
                    .map(([k, v]) => `${k}: ${v ?? 'unavailable'}`)
                    .join(' · ') || 'No supported numeric fields'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <Pager page={page} count={rows.length} size={20} setPage={setPage} />
    </div>
  )
}
