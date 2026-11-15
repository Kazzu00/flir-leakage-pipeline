import { useState, useMemo } from 'react'
import type { ArtifactRun, LeakageSnapshot } from '@/contracts/leakage'
import { Panel, Notice, Select } from '@/components/feedback/Primitives'
import { Chart } from '@/components/visualization/Chart'
import { Pager } from './Pager'
import { TechnicalDetailsDrawer } from '@/components/research/TechnicalDetailsDrawer'
const colors = [
  '#73d9be',
  '#a4a0f2',
  '#edbd79',
  '#80b8ea',
  '#ed99b1',
  '#bcda87',
]
const splitColors: Record<string, string> = {
  train: '#73d9be',
  validation: '#edbd79',
  test: '#a4a0f2',
}
export function ArtifactExplorer({
  snapshot,
  run,
}: {
  snapshot: LeakageSnapshot
  run: ArtifactRun
}) {
  const [cluster, setCluster] = useState('all')
  const [source, setSource] = useState('all')
  const [color, setColor] = useState('Source membership')
  const [after, setAfter] = useState(false)
  const [selectedId, setSelectedId] = useState('')
  const [search, setSearch] = useState('')
  const [page, setPage] = useState(0)
  const coordinates = useMemo(
    () =>
      run.stage === 'reduction'
        ? run.coordinates
        : (snapshot.runs.find((r) => r.id === run.reductionSpaceId)
            ?.coordinates ?? []),
    [run, snapshot.runs],
  )
  const contents = useMemo(
    () => new Map(snapshot.contents.map((c) => [c.contentId, c])),
    [snapshot],
  )
  const labels = useMemo(
    () => new Map(run.labels.map((l) => [l.contentId, l])),
    [run],
  )
  const split = snapshot.runs.find(
    (r) =>
      r.stage === 'splits' &&
      r.method === 'cluster_aware' &&
      r.clusteringSpaceId === run.id,
  )
  const allocations = useMemo(
    () => new Map(split?.assignments.map((a) => [a.contentId, a.split])),
    [split],
  )
  const clusters = useMemo(
    () =>
      [...new Set(run.labels.map((l) => l.clusterId))].sort((a, b) => a - b),
    [run],
  )
  const sources = useMemo(
    () =>
      [
        ...new Set(
          snapshot.contents.flatMap((c) =>
            c.occurrences.map((o) => o.sourceVideo),
          ),
        ),
      ].sort(),
    [snapshot],
  )
  const points = useMemo(
    () =>
      coordinates.filter(
        (p) =>
          (cluster === 'all' ||
            labels.get(p.contentId)?.clusterId === Number(cluster)) &&
          (source === 'all' ||
            contents
              .get(p.contentId)
              ?.occurrences.some((o) => o.sourceVideo === source)),
      ),
    [coordinates, cluster, source, labels, contents],
  )
  const selected =
    contents.get(selectedId) ??
    contents.get(points[0]?.contentId) ??
    contents.get(run.labels[0]?.contentId)
  const listed = (
    run.stage === 'clustering'
      ? run.labels
          .map((l) => l.contentId)
          .filter(
            (id) =>
              cluster === 'all' ||
              labels.get(id)?.clusterId === Number(cluster),
          )
      : coordinates.map((p) => p.contentId)
  ).filter(
    (id) =>
      id.includes(search) &&
      (source === 'all' ||
        contents.get(id)?.occurrences.some((o) => o.sourceVideo === source)),
  )
  const getGroup = (id: string) =>
    after
      ? (allocations.get(id) ?? 'unavailable')
      : color === 'Cluster'
        ? String(labels.get(id)?.clusterId ?? 'unavailable')
        : [...new Set(contents.get(id)?.occurrences.map((o) => o.sourceVideo))]
            .sort()
            .join(' + ')
  const groups = [...new Set(points.map((p) => getGroup(p.contentId)))].sort()
  const groupColors = new Map(
    groups.map((group, i) => [group, colors[i % colors.length]]),
  )
  const pointColor = (id: string) =>
    after
      ? (splitColors[getGroup(id)] ?? '#8c9da3')
      : getGroup(id) === '-1'
        ? '#8c9da3'
        : (groupColors.get(getGroup(id)) ?? '#8c9da3')
  const selectedRows = selected?.occurrences ?? []
  const sequenceIds =
    snapshot.schemaVersion === 'LeakageSnapshotV2'
      ? (snapshot.research.sequences.artifact?.zones
          .filter(
            (z) =>
              z.kind === 'sequence_instance' &&
              selectedRows.some(
                (o) =>
                  o.sourceVideo === z.sourceVideo &&
                  o.sampleIndex >= z.start &&
                  o.sampleIndex <= z.end,
              ),
          )
          .map((z) => z.id) ?? [])
      : []
  const featureRows = useMemo(
    () =>
      new Map(
        snapshot.runs
          .find((r) => r.id === run.featureSpaceId)
          ?.contentIndex.map((p) => [p.contentId, p.embeddingRow]),
      ),
    [snapshot, run.featureSpaceId],
  )
  return (
    <div className="artifact-workspace">
      <div className="toolbar artifact-controls">
        {run.stage === 'clustering' && (
          <Select
            label="Cluster filter"
            value={cluster}
            onChange={(v) => {
              setCluster(v)
              setPage(0)
              setSelectedId('')
            }}
            options={[
              { value: 'all', label: 'All clusters' },
              ...clusters.map((c) => ({
                value: String(c),
                label: c === -1 ? 'Noise (-1)' : `Cluster ${c}`,
              })),
            ]}
          />
        )}
        <Select
          label="Source video filter"
          value={source}
          onChange={(v) => {
            setSource(v)
            setPage(0)
            setSelectedId('')
          }}
          options={[
            { value: 'all', label: 'All sources' },
            ...sources.map((s) => ({ value: s, label: s })),
          ]}
        />
        <Select
          label="Color by"
          value={color}
          onChange={setColor}
          options={[
            'Source membership',
            ...(run.stage === 'clustering' ? ['Cluster'] : []),
          ].map((v) => ({ value: v, label: v }))}
        />
        {run.stage === 'clustering' && (
          <div className="split-toggle">
            <button aria-pressed={!after} onClick={() => setAfter(false)}>
              BEFORE SPLIT
            </button>
            <button
              aria-pressed={after}
              disabled={!split}
              title={!split ? 'No compatible split artifact' : undefined}
              onClick={() => setAfter(true)}
            >
              AFTER SPLIT
            </button>
          </div>
        )}
      </div>
      <Panel
        className="artifact-scatter"
        title={
          run.stage === 'reduction'
            ? 'Persisted reduction coordinates'
            : after
              ? 'Clusters after allocation'
              : 'Clusters before allocation'
        }
        meta={`${points.length} / ${coordinates.length} coordinates · no visual subsampling`}
      >
        {coordinates.length ? (
          <Chart
            renderer="canvas"
            height={460}
            label={`Artifact scatter: ${points.length} persisted coordinates`}
            onSelect={(i) => setSelectedId(points[i]?.contentId ?? '')}
            option={{
              grid: { left: 55, right: 25, top: 30, bottom: 45 },
              tooltip: {
                trigger: 'item',
                renderMode: 'richText',
                formatter: (p: unknown) => {
                  const idx = (p as { dataIndex: number }).dataIndex
                  return points[idx]?.contentId ?? ''
                },
              },
              xAxis: { type: 'value', scale: true, name: 'Dimension 1' },
              yAxis: { type: 'value', scale: true, name: 'Dimension 2' },
              series: [
                {
                  type: 'scatter',
                  symbolSize: 6,
                  progressive: 2000,
                  data: points.map((p) => ({
                    value: [p.x, p.y],
                    symbolSize: p.contentId === selected?.contentId ? 12 : 6,
                    itemStyle: { color: pointColor(p.contentId) },
                  })),
                },
              ],
            }}
          />
        ) : (
          <Notice>
            Coordinates were not exported for this run. The explorer does not
            compute or substitute a projection.
          </Notice>
        )}
        <p className="panel-body micro">
          {after ? 'Color: assigned split' : `Color: ${color}`}. Multi-source
          contents retain their full membership set. Axes use persisted values
          without 0–100 normalization. Select a point or use the content table
          below.
        </p>
      </Panel>
      {run.stage === 'clustering' && !split && (
        <Notice>
          AFTER SPLIT unavailable: no compatible cluster-aware split artifact.
        </Notice>
      )}
      {selected && (
        <Panel
          className="artifact-detail"
          title={`Content detail · ${selected.contentId}`}
          meta="No private filenames or thumbnails"
        >
          <div className="panel-body">
            <p>
              {sequenceIds.length
                ? `Reviewed sequence memberships: ${sequenceIds.join(', ')}. Capture timestamp: unavailable.`
                : 'Sequence: unknown. Capture timestamp: unavailable.'}
            </p>
            <p>
              Cluster:{' '}
              {labels.get(selected.contentId)?.clusterId ?? 'unavailable'} ·
              split: {allocations.get(selected.contentId) ?? 'unavailable'} ·
              coordinates:{' '}
              {coordinates.find((p) => p.contentId === selected.contentId)?.x ??
                'unavailable'}
              ,{' '}
              {coordinates.find((p) => p.contentId === selected.contentId)?.y ??
                'unavailable'}
            </p>
            <TechnicalDetailsDrawer title="Content identities">
              {' '}
              <div className="table-scroll">
                <table>
                  <thead>
                    <tr>
                      <th>frame_id</th>
                      <th>content_id</th>
                      <th>embedding_row</th>
                      <th>Source video</th>
                      <th>Sample index</th>
                      <th>Grid seconds</th>
                    </tr>
                  </thead>
                  <tbody>
                    {selectedRows.map((o) => (
                      <tr key={o.frameId}>
                        <td>{o.frameId}</td>
                        <td>{selected.contentId}</td>
                        <td>
                          {featureRows.get(selected.contentId) ?? 'unavailable'}
                        </td>
                        <td>{o.sourceVideo}</td>
                        <td>{o.sampleIndex}</td>
                        <td>{o.gridSeconds}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </TechnicalDetailsDrawer>
          </div>
        </Panel>
      )}
      {selected && (
        <Panel
          className="artifact-sampling"
          title="Source-video sampling grid"
          meta="Selected content occurrences · not a sequence timeline"
        >
          {' '}
          <Chart
            height={220}
            label="Sampling-grid occurrences by source; not a capture timeline"
            option={{
              grid: { left: 120, right: 30, top: 30, bottom: 40 },
              xAxis: { type: 'value', name: 'Sample index', minInterval: 1 },
              yAxis: {
                type: 'category',
                data: [...new Set(selectedRows.map((o) => o.sourceVideo))],
              },
              series: [
                {
                  type: 'scatter',
                  symbolSize: 10,
                  data: selectedRows.map((o) => [o.sampleIndex, o.sourceVideo]),
                  itemStyle: { color: '#73d9be' },
                },
              ],
            }}
          />
        </Panel>
      )}
      <Panel
        className="artifact-gallery"
        title="Representative contents"
        meta="First 6 matching contents; full coverage in the table"
      >
        <div className="gallery">
          {listed.slice(0, 6).map((id) => (
            <button
              key={id}
              className={`gallery-item ${selected?.contentId === id ? 'selected' : ''}`}
              aria-label={`Inspect ${id}`}
              onClick={() => setSelectedId(id)}
            >
              <div className="sample-image">
                <span>Thumbnail unavailable</span>
                <small>ARTIFACT IDENTITY</small>
              </div>
              <span className="mono">{id}</span>
            </button>
          ))}
        </div>
      </Panel>
      <details className="artifact-browse">
        <summary>Browse all contents · keyboard alternative</summary>
        <Panel
          title="Content inspection"
          meta="ARTIFACT · thumbnails unavailable"
        >
          <div className="panel-body">
            <label className="select-field">
              <span>Find content alias</span>
              <input
                type="search"
                value={search}
                onChange={(e) => {
                  setSearch(e.target.value)
                  setPage(0)
                }}
              />
            </label>
            <div className="table-scroll">
              <table>
                <thead>
                  <tr>
                    <th>Content</th>
                    <th>Cluster</th>
                    <th>Medoid</th>
                    <th>Sources</th>
                  </tr>
                </thead>
                <tbody>
                  {listed.slice(page * 25, page * 25 + 25).map((id) => (
                    <tr key={id}>
                      <td>
                        <button
                          className="row-link"
                          onClick={() => setSelectedId(id)}
                        >
                          {id}
                        </button>
                      </td>
                      <td>{labels.get(id)?.clusterId ?? 'unavailable'}</td>
                      <td>
                        {labels.has(id)
                          ? String(labels.get(id)?.medoid)
                          : 'unavailable'}
                      </td>
                      <td>
                        {[
                          ...new Set(
                            contents
                              .get(id)
                              ?.occurrences.map((o) => o.sourceVideo),
                          ),
                        ].join(', ')}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <Pager
              page={page}
              count={listed.length}
              size={25}
              setPage={setPage}
            />
          </div>
        </Panel>
      </details>
    </div>
  )
}
