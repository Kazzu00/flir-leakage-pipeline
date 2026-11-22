import { useState, useMemo } from 'react'
import type { ArtifactRun } from '@/contracts/leakage'
import { Panel, Notice, Select } from '@/components/feedback/Primitives'
import { Chart } from '@/components/visualization/Chart'
import { TechnicalDetailsDrawer } from '@/components/research/TechnicalDetailsDrawer'
export function SimilarityEvidence({ run }: { run: ArtifactRun }) {
  const queries = useMemo(() => [...new Set(run.pairs.map((p) => p.a))], [run])
  const [search, setSearch] = useState('')
  const [query, setQuery] = useState(queries[0] ?? '')
  const [selectedRank, setSelectedRank] = useState(1)
  const options = queries.filter((q) => q.includes(search)).slice(0, 100)
  const byQuery = useMemo(() => {
    const index = new Map<string, ArtifactRun['pairs']>()
    for (const pair of run.pairs) {
      const rows = index.get(pair.a) ?? []
      rows.push(pair)
      index.set(pair.a, rows)
    }
    return index
  }, [run])
  const neighbors = byQuery.get(query) ?? []
  const pair = neighbors.find((p) => p.rank === selectedRank) ?? neighbors[0]
  const quantiles = ['min', 'Q1', 'median', 'Q3', 'p95', 'p99', 'max'].filter(
    (k) => run.metrics[k] != null,
  )
  return (
    <>
      <Notice>
        The global values above are persisted summary statistics and quantiles.
        Histogram counts are unavailable; quantiles have not been converted into
        bins.
      </Notice>
      {quantiles.length > 0 && (
        <Panel
          title="Reported similarity distribution"
          meta="Persisted quantiles · not histogram bins"
        >
          <Chart
            height={220}
            renderer="canvas"
            label="Reported cosine quantiles; exact numeric values in technical details"
            option={{
              grid: { left: 55, right: 30, top: 25, bottom: 35 },
              xAxis: { type: 'category', data: quantiles },
              yAxis: { type: 'value', min: -1, max: 1 },
              series: [
                {
                  type: 'line',
                  data: quantiles.map((k) => run.metrics[k]),
                  itemStyle: { color: '#73d9be' },
                },
              ],
            }}
          />
        </Panel>
      )}
      <Panel
        title="Nearest-neighbor inspection"
        meta="Persisted top-k · cosine on L2 embeddings"
      >
        <div className="panel-body">
          <label className="select-field">
            <span>Search content alias</span>
            <input
              type="search"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="content-000001"
            />
          </label>
          <p className="micro">
            Selector shows up to 100 matching aliases. Search an exact alias to
            inspect any content.
          </p>
          <Select
            label="Query content"
            value={options.includes(query) ? query : ''}
            onChange={setQuery}
            options={[
              { value: '', label: 'Select a content' },
              ...options.map((v) => ({ value: v, label: v })),
            ]}
          />
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Rank</th>
                  <th>Neighbor</th>
                  <th>Cosine</th>
                </tr>
              </thead>
              <tbody>
                {neighbors.slice(0, 25).map((p) => (
                  <tr key={p.rank}>
                    <td>{p.rank}</td>
                    <td>
                      <button
                        className="row-link"
                        onClick={() => setSelectedRank(p.rank)}
                      >
                        {p.b}
                      </button>
                    </td>
                    <td>{p.cosine.toFixed(6)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="micro">
            First 25 persisted neighbors. Full pair evidence remains accessible
            below.
          </p>
          {pair && (
            <div className="pair-detail">
              <h3>Selected pair</h3>
              <p>
                {pair.a} ↔ {pair.b} · cosine {pair.cosine.toFixed(4)}
              </p>
              <p>
                {pair.sameSourceVideo
                  ? 'Shared source-video membership'
                  : 'Disjoint source-video membership'}{' '}
                · visual similarity does not prove temporal continuity.
              </p>
              <TechnicalDetailsDrawer title="Pair technical details">
                <pre className="audit-json">
                  {JSON.stringify({ pair, neighbors }, null, 2)}
                </pre>
              </TechnicalDetailsDrawer>
            </div>
          )}
          <p>
            Thumbnails unavailable. No FLIR images are included in the snapshot.
          </p>
        </div>
      </Panel>
    </>
  )
}
