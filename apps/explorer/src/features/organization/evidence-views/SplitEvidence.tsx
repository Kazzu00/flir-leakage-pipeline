import { useState } from 'react'
import type { ArtifactRun } from '@/contracts/leakage'
import { Panel, Metrics } from '@/components/feedback/Primitives'
import { Pager } from './Pager'
export function SplitEvidence({ run }: { run: ArtifactRun }) {
  const [page, setPage] = useState(0)
  return (
    <Panel
      title="Persisted split assignments"
      meta="ARTIFACT · occurrence counts"
    >
      <Metrics
        metrics={['train', 'validation', 'test'].map((name) => ({
          name,
          value: run.assignments.filter((a) => a.split === name).length,
          unit: '',
          origin: 'artifact',
          caveat: 'Count of persisted assignment rows.',
        }))}
      />
      <div className="panel-body">
        <p>
          Strategy: {run.method}. Clustering reference:{' '}
          {run.clusteringSpaceId ?? 'not applicable'}.
        </p>
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Occurrence</th>
                <th>Content</th>
                <th>Split</th>
              </tr>
            </thead>
            <tbody>
              {run.assignments.slice(page * 25, page * 25 + 25).map((a) => (
                <tr key={a.frameId}>
                  <td>{a.frameId}</td>
                  <td>{a.contentId}</td>
                  <td>{a.split}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <Pager
          page={page}
          count={run.assignments.length}
          size={25}
          setPage={setPage}
        />
      </div>
    </Panel>
  )
}
