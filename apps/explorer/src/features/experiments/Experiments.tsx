import { useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import type { ColumnDef } from '@tanstack/react-table'
import type { Run } from '@/contracts'
import { useSnapshot } from '@/data/provider'
import {
  PageTitle,
  Panel,
  Select,
  Notice,
  Metrics,
} from '@/components/feedback/Primitives'
import { Status } from '@/components/feedback/Status'
import { DataTable } from '@/components/visualization/DataTable'
export function Experiments() {
  const { data } = useSnapshot()
  const [params, setParams] = useSearchParams()
  const module = params.get('module') ?? 'all'
  const [status, setStatus] = useState('all')
  const [search, setSearch] = useState('')
  const [selected, setSelected] = useState<string | null>(null)
  if (!data) return null
  const runs = data.runs.filter(
    (r) =>
      (module === 'all' || r.module === module) &&
      (status === 'all' || r.status === status) &&
      `${r.id} ${r.method} ${r.datasetId}`
        .toLowerCase()
        .includes(search.toLowerCase()),
  )
  const run = runs.find((r) => r.id === selected)
  const columns: ColumnDef<Run>[] = [
    {
      accessorKey: 'id',
      header: 'Run ID',
      cell: ({ row }) => (
        <button
          className="row-link mono"
          onClick={() => setSelected(row.original.id)}
        >
          {row.original.id}
        </button>
      ),
    },
    { accessorKey: 'module', header: 'Module' },
    { accessorKey: 'stage', header: 'Stage' },
    { accessorKey: 'origin', header: 'Origin' },
    { accessorKey: 'method', header: 'Method / model' },
    {
      accessorKey: 'status',
      header: 'Lifecycle',
      cell: ({ row }) => <Status state={row.original.status} />,
    },
    {
      accessorKey: 'verification',
      header: 'Verification',
      cell: ({ row }) => <Status state={row.original.verification} />,
    },
  ]
  return (
    <>
      <PageTitle
        eyebrow="CROSS-MODULE / EXPERIMENT REGISTRY"
        title="Experiments"
        description="Browse normalized run records from all three research modules."
      />
      <div className="toolbar">
        <Select
          label="Module"
          value={module}
          onChange={(v) => {
            setParams(v === 'all' ? {} : { module: v })
            setSelected(null)
          }}
          options={['all', 'preprocessing', 'organization', 'segmentation'].map(
            (v) => ({ value: v, label: v }),
          )}
        />
        <Select
          label="Lifecycle status"
          value={status}
          onChange={setStatus}
          options={[
            'all',
            'verified',
            'complete',
            'running',
            'experimental',
            'pending',
            'invalid',
            'unavailable',
            'mock',
            'inconsistent',
          ].map((v) => ({ value: v, label: v }))}
        />
        <label className="select-field">
          <span>Search runs / model / dataset</span>
          <input
            type="search"
            placeholder="Search demo runs…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </label>
      </div>
      <Panel
        title="Run registry"
        meta={`${runs.length} runs · ${data.leakage ? 'MIXED origins' : 'ALL DEMO'}`}
      >
        <DataTable
          label={
            data.leakage
              ? 'Mixed experiment registry'
              : 'Demo experiment registry'
          }
          data={runs}
          columns={columns}
        />
      </Panel>
      {run ? (
        <Panel title={`Run detail · ${run.id}`} meta={run.origin.toUpperCase()}>
          <div className="run-detail">
            <dl>
              <dt className="muted">Dataset</dt>
              <dd className="mono">{run.datasetId}</dd>
              <dt className="muted">Parameters</dt>
              <dd>
                <pre>{JSON.stringify(run.parameters, null, 2)}</pre>
              </dd>
              <dt className="muted">Execution / verification</dt>
              <dd>
                <Status state={run.status} />{' '}
                <Status state={run.verification} />
              </dd>
            </dl>
            <div>
              <h3>Artifact references</h3>
              {run.artifacts.map((a) => (
                <div key={a.id} style={{ marginTop: 12 }}>
                  <strong>
                    {a.label} · {a.format}
                  </strong>{' '}
                  <Status state={a.status} />
                  <p className="muted">{a.description}</p>
                </div>
              ))}
            </div>
          </div>
          {run.metrics.length > 0 && <Metrics metrics={run.metrics} />}
          <Notice>{run.caveat}</Notice>
        </Panel>
      ) : (
        <Notice>
          Select a run ID to inspect its parameters, dataset, metrics, artifact
          references and verification state. Lifecycle “complete” in a demo
          fixture does not mean scientifically verified.
        </Notice>
      )}
    </>
  )
}
