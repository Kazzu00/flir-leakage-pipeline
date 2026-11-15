import { useMemo, useState } from 'react'
import { Panel, Notice, Select } from '@/components/feedback/Primitives'
import { TechnicalDetailsDrawer } from '@/components/research/TechnicalDetailsDrawer'
import { organizationMediaUrl } from '@/data/organization-media'
import {
  useScientificTimeline,
  framesForRun,
  temporalOrder,
  timelineCsv,
  summarizeFrames,
  clusterLabel,
  clusterColor,
  partitionLabels,
  reasonLabels,
  type ScientificTimelineData,
  type TimelineFrame,
} from '@/data/scientific-timeline'
import { ScientificTimelineLanes, splitColors } from './ScientificTimelineLanes'
import './ScientificTimeline.css'

export function ScientificTimeline() {
  const query = useScientificTimeline()
  if (query.isPending)
    return <p role="status">Cargando línea temporal científica…</p>
  if (query.isError)
    return (
      <Panel title="Línea temporal científica no disponible">
        <Notice>
          No se pudo verificar el export temporal. La implementación requiere
          una publicación local con receipt y membresías verificables; no se
          sustituye por datos sintéticos.
        </Notice>
        <p className="panel-body">
          El export se prepara con{' '}
          <code>tools/export_verified_video_timeline.py</code>. La evidencia
          temporal anterior sigue disponible en la pestaña Temporal evidence.
        </p>
        <button onClick={() => void query.refetch()}>Reintentar carga</button>
      </Panel>
    )
  return <ScientificTimelineView data={query.data} />
}

function download(name: string, text: string, type = 'text/csv;charset=utf-8') {
  const url = URL.createObjectURL(new Blob([text], { type }))
  const link = document.createElement('a')
  link.href = url
  link.download = name
  link.click()
  setTimeout(() => URL.revokeObjectURL(url), 0)
}

export function ScientificTimelineView({
  data,
}: {
  data: ScientificTimelineData
}) {
  const initial =
    data.runs.find((r) => r.run_id === data.admission?.delivery_run_id) ??
    data.runs.find((r) => r.run_id === 'historical-S2-3') ??
    data.runs.find((r) => r.run_id === 'historical-S1-0') ??
    data.runs[0]
  const [runId, setRunId] = useState(initial.run_id)
  const [clusterRun, setClusterRun] = useState(
    data.clustering_runs[0]?.cluster_run_id ?? '',
  )
  const [clusterFilter, setClusterFilter] = useState('')
  const [axisId, setAxisId] = useState('')
  const [frameId, setFrameId] = useState('')
  const [intervalId, setIntervalId] = useState('')
  const [rangeSelection, setRangeSelection] = useState<[number, number] | null>(
    null,
  )
  const [zoom, setZoom] = useState(1)
  const [page, setPage] = useState(0)
  const run = data.runs.find((r) => r.run_id === runId) ?? initial
  const frames = useMemo(
    () => framesForRun(data, run, clusterRun),
    [data, run, clusterRun],
  )
  const assigned = useMemo(
    () => frames.filter((r) => r.reason === null),
    [frames],
  )
  const excluded = useMemo(
    () => frames.filter((r) => r.reason !== null),
    [frames],
  )
  const axes = data.admission
    ? data.axes
    : data.axes.filter((a) => assigned.some((r) => r.axis_id === a.axis_id))
  const axis = axes.find((a) => a.axis_id === axisId) ?? axes[0]
  const activeAxisId = axis?.axis_id
  const axisFrames = assigned
    .filter((r) => r.axis_id === activeAxisId)
    .sort(temporalOrder)
  const min = axisFrames[0]?.position ?? 0
  const max = axisFrames.at(-1)?.position ?? min
  const range: [number, number] = rangeSelection ?? [min, max]
  const rangeStart = range[0]
  const rangeEnd = range[1]
  const rangeValid = range[0] <= range[1]
  const intervals = data.intervals.filter((i) => i.axis_id === axis?.axis_id)
  const selectedInterval = intervals.find((i) => i.interval_id === intervalId)
  const focusFrames = axisFrames.filter(
    (r) => r.position! >= rangeStart && r.position! <= rangeEnd,
  )
  const tableFrames = focusFrames.filter(
    (r) => clusterFilter === '' || String(r.cluster_id) === clusterFilter,
  )
  const active = frames.find((f) => f.frame_id === frameId) ?? axisFrames[0]
  const activeIndex = axisFrames.findIndex(
    (r) => r.frame_id === active?.frame_id,
  )
  const counts = summarizeFrames(focusFrames)
  const clusters = [...new Set(frames.map((r) => r.cluster_id))].sort(
    (a, b) => (a ?? Infinity) - (b ?? Infinity),
  )
  const clusterCounts = new Map<number | null, number>()
  for (const r of focusFrames)
    clusterCounts.set(r.cluster_id, (clusterCounts.get(r.cluster_id) ?? 0) + 1)
  const changeRun = (population: string, policy: string, seed: string) => {
    const next =
      data.runs.find(
        (r) =>
          r.population === population &&
          r.policy === policy &&
          r.seed === Number(seed),
      ) ??
      data.runs.find(
        (r) => r.population === population && r.policy === policy,
      ) ??
      data.runs.find((r) => r.population === population)
    if (next) {
      setRunId(next.run_id)
      setFrameId('')
      setRangeSelection(null)
      setIntervalId('')
      setPage(0)
    }
  }
  const selectInterval = (id: string) => {
    const value = intervals.find((i) => i.interval_id === id)
    setIntervalId(id)
    setRangeSelection(value ? [value.start, value.end] : null)
    setPage(0)
  }
  const coverage = {
    run_id: run.run_id,
    total: frames.length,
    displayed: assigned.length,
    excluded: excluded.length,
    by_axis: axes.map((a) => ({
      axis_id: a.axis_id,
      label: a.label,
      association: a.association,
      occurrences: assigned.filter((r) => r.axis_id === a.axis_id).length,
    })),
    exclusion_reasons: Object.keys(reasonLabels).map((reason) => ({
      reason,
      count: excluded.filter((r) => r.reason === reason).length,
    })),
  }

  return (
    <div className="scientific-timeline">
      <h2>Línea temporal científica</h2>
      <Notice>
        {data.origin === 'synthetic_fixture'
          ? 'DEMO / SYNTHETIC — fixture de infraestructura.'
          : 'Evidencia real local · publicación provisional.'}{' '}
        Estado del test en esta fuente: {data.test_status}; esta visualización
        no autoriza entrenamiento ni selección de modelos. Clúster ≠ secuencia.
        {data.clustering_runs.length
          ? ' La ejecución de clustering seleccionada es una referencia independiente de S1/S2/S3.'
          : ' Clúster no disponible: no hay membresías reales compatibles en la fuente seleccionada.'}
      </Notice>
      {data.admission && (
        <Notice>
          Solo correspondencias verificadas con los tres vídeos autorizados.
          Entrega A: histórico S2-3, candidato de mitigación parcial, sin
          restricción por vídeo. Entrega B bloqueada: faltan correspondencias
          canónicas y unidades de secuencia verificadas. No encontrar
          correspondencia en el alcance previo no demuestra ausencia en los
          vídeos.
          {run.run_id !== data.admission.delivery_run_id &&
            ' Comparación exploratoria: este run no es el paquete entregado.'}
        </Notice>
      )}
      <div className="toolbar timeline-filters">
        <Select
          label="Población"
          value={run.population}
          onChange={(v) => changeRun(v, run.policy, String(run.seed))}
          options={[...new Set(data.runs.map((r) => r.population))].map(
            (v) => ({
              value: v,
              label: v === 'historical' ? 'Historical' : 'Expanded',
            }),
          )}
        />
        <Select
          label="Política"
          value={run.policy}
          onChange={(v) => changeRun(run.population, v, String(run.seed))}
          options={[
            ...new Set(
              data.runs
                .filter((r) => r.population === run.population)
                .map((r) => r.policy),
            ),
          ].map((v) => ({ value: v, label: v }))}
        />
        <Select
          label="Semilla de particionamiento"
          value={String(run.seed)}
          onChange={(v) => changeRun(run.population, run.policy, v)}
          options={data.runs
            .filter(
              (r) => r.population === run.population && r.policy === run.policy,
            )
            .sort((a, b) => a.seed - b.seed)
            .map((r) => ({ value: String(r.seed), label: String(r.seed) }))}
        />
        <Select
          label="Ejecución de clustering"
          value={clusterRun}
          onChange={(v) => {
            setClusterRun(v)
            setClusterFilter('')
            setPage(0)
          }}
          options={
            data.clustering_runs.length
              ? data.clustering_runs.map((r) => ({
                  value: r.cluster_run_id,
                  label: `${r.cluster_run_id} · ${r.source_kind === 'frozen_split_membership' ? 'membresía congelada' : 'artefacto de clustering'}`,
                }))
              : [{ value: '', label: 'No disponible' }]
          }
        />
        <Select
          label="Clúster (resaltar)"
          value={clusterFilter}
          onChange={(v) => {
            setClusterFilter(v)
            setPage(0)
          }}
          options={[
            { value: '', label: 'Todos los clústeres' },
            ...clusters.map((id) => ({
              value: String(id),
              label: clusterLabel(id),
            })),
          ]}
        />
        <Select
          label="Vídeo original / asociación temporal"
          value={axis?.axis_id ?? ''}
          onChange={(v) => {
            setAxisId(v)
            setFrameId('')
            setRangeSelection(null)
            setIntervalId('')
            setPage(0)
          }}
          options={
            axes.length
              ? axes.map((a) => ({
                  value: a.axis_id,
                  label: `${a.label}${a.association === 'filename_inference' ? ' · familia inferida, vídeo no verificado' : ''}`,
                }))
              : [{ value: '', label: 'Sin asociación temporal utilizable' }]
          }
        />
      </div>
      <Panel title="Cobertura y conservación de ocurrencias" meta={run.run_id}>
        <dl
          className="timeline-metrics"
          aria-label="Contabilidad de ocurrencias"
        >
          <div>
            <dt>Total de la partición</dt>
            <dd>{frames.length}</dd>
          </div>
          <div>
            <dt>Representables en ejes temporales</dt>
            <dd>{assigned.length}</dd>
          </div>
          <div>
            <dt>Excluidas solo de la vista temporal</dt>
            <dd>{excluded.length}</dd>
          </div>
          <div>
            <dt>Orden inferido de nombres</dt>
            <dd>
              {
                assigned.filter((f) => f.order_basis === 'filename_inference')
                  .length
              }
            </dd>
          </div>
        </dl>
        <div className="toolbar">
          <button
            onClick={() =>
              download(
                'timeline_assigned.csv',
                timelineCsv(assigned, run, clusterRun),
              )
            }
          >
            Exportar frames asignados (run completo)
          </button>
          <button
            onClick={() =>
              download(
                'excluded_frames.csv',
                timelineCsv(excluded, run, clusterRun),
              )
            }
          >
            Exportar excluded_frames.csv
          </button>
          <button
            onClick={() =>
              download(
                'timeline_coverage.json',
                JSON.stringify(coverage, null, 2),
                'application/json',
              )
            }
          >
            Exportar cobertura
          </button>
        </div>
        <p className="panel-body muted">
          Los filtros de vídeo, clúster e intervalo no cambian estos totales ni
          las pertenencias. Los CSV conservan el run y clúster de referencia
          seleccionados; nunca mezclan semillas.
        </p>
      </Panel>
      {axis && axisFrames.length > 0 ? (
        <>
          <Notice>
            {axis.association === 'filename_inference'
              ? 'Asociación a familia de nombres inferida: no identifica un vídeo original verificado. El índice es una inferencia de aparición, no un timestamp ni un frame de captura confirmado.'
              : 'Identidad de vídeo reportada por la fuente. Una rejilla de muestreo no equivale a tiempo de captura verificado.'}{' '}
            Las ocurrencias con igual índice se apilan; su desempate por
            frame_id no implica un orden temporal entre ellas.
          </Notice>
          <Panel
            title={`Distribución temporal · ${axis.label}`}
            meta={`${axisFrames.length} ocurrencias · índices ${min}–${max}`}
          >
            <div className="toolbar">
              <label className="select-field">
                Zoom horizontal
                <input
                  aria-label="Zoom horizontal"
                  type="range"
                  min={1}
                  max={12}
                  step={1}
                  value={zoom}
                  onChange={(e) => setZoom(Number(e.target.value))}
                />
              </label>
              <output aria-label="Factor de zoom">{zoom}×</output>
              <label className="select-field">
                Desde índice
                <input
                  aria-label="Desde índice"
                  type="number"
                  min={min}
                  max={max}
                  value={range[0]}
                  onChange={(e) => {
                    setRangeSelection([
                      Math.max(min, Math.min(max, Number(e.target.value))),
                      range[1],
                    ])
                    setIntervalId('')
                    setPage(0)
                  }}
                />
              </label>
              <label className="select-field">
                Hasta índice
                <input
                  aria-label="Hasta índice"
                  type="number"
                  min={min}
                  max={max}
                  value={range[1]}
                  onChange={(e) => {
                    setRangeSelection([
                      range[0],
                      Math.max(min, Math.min(max, Number(e.target.value))),
                    ])
                    setIntervalId('')
                    setPage(0)
                  }}
                />
              </label>
              <button
                onClick={() => {
                  setRangeSelection(null)
                  setIntervalId('')
                  setPage(0)
                }}
              >
                Restablecer intervalo
              </button>
            </div>
            {!rangeValid && (
              <p role="alert">El inicio debe ser menor o igual al final.</p>
            )}
            <p className="panel-body micro">
              Haz clic en una marca para ver el frame o arrastra para
              seleccionar índices. Usa la tabla o Anterior/Siguiente con
              teclado. Zoom y desplazamiento son comunes a las tres pistas.
            </p>
            <ScientificTimelineLanes
              key={axis.axis_id}
              frames={axisFrames}
              intervals={intervals}
              clusterRunId={clusterRun}
              clusterFilter={clusterFilter}
              selectedId={active?.frame_id ?? ''}
              zoom={zoom}
              range={range}
              onFrame={setFrameId}
              onInterval={selectInterval}
              onRange={(start, end) => {
                setRangeSelection([start, end])
                setIntervalId('')
                setPage(0)
              }}
            />
            <div
              className="timeline-legend"
              aria-label="Leyenda de particiones"
            >
              {Object.entries(partitionLabels).map(([key, label]) => (
                <span key={key}>
                  <i
                    style={{
                      background: splitColors[key as keyof typeof splitColors],
                    }}
                  />
                  {label}
                </span>
              ))}
            </div>
            <details className="timeline-cluster-legend">
              <summary>Leyenda de clústeres y recurrencias</summary>
              <div className="timeline-legend">
                {clusters.map((id) => (
                  <button
                    key={String(id)}
                    aria-pressed={clusterFilter === String(id)}
                    onClick={() => {
                      setClusterFilter(String(id))
                      setPage(0)
                    }}
                  >
                    <i style={{ background: clusterColor(clusterRun, id) }} />
                    {clusterLabel(id)}
                  </button>
                ))}
              </div>
              <p>
                El color depende de cluster_run_id y cluster_id, nunca del
                split. Las reapariciones se resaltan sin asumir continuidad
                entre ellas.
              </p>
            </details>
          </Panel>
          <Panel
            title={
              selectedInterval
                ? selectedInterval.kind === 'candidate_span'
                  ? 'Tramo candidato seleccionado'
                  : 'Secuencia seleccionada'
                : 'Selección exploratoria de índices'
            }
            meta={`${range[0]}–${range[1]} · ${focusFrames.length} ocurrencias`}
          >
            <div className="toolbar">
              <Select
                label="Secuencia / tramo exportado"
                value={selectedInterval?.interval_id ?? ''}
                onChange={selectInterval}
                options={[
                  {
                    value: '',
                    label: 'Selección exploratoria (no crea una secuencia)',
                  },
                  ...intervals.map((i) => ({
                    value: i.interval_id,
                    label: `${i.kind === 'candidate_span' ? 'Tramo candidato' : 'Secuencia'} · ${i.sequence_id ?? i.interval_id} · ${i.start}–${i.end}`,
                  })),
                ]}
              />
            </div>
            {selectedInterval && (
              <p className="panel-body">
                ID:{' '}
                {selectedInterval.sequence_id ?? selectedInterval.interval_id} ·{' '}
                {selectedInterval.provenance} · Revisión:{' '}
                {selectedInterval.review} · Fuente:{' '}
                {selectedInterval.source_reference}
              </p>
            )}
            <p className="panel-body">
              Clústeres presentes (distribución agregada, no orden temporal):{' '}
              {[...clusterCounts]
                .map(([id, n]) => `${clusterLabel(id)}: ${n}`)
                .join(' · ') || 'Sin frames en el intervalo'}
            </p>
            <table className="timeline-table">
              <caption>
                Ocurrencias y clústeres por partición del intervalo
              </caption>
              <thead>
                <tr>
                  <th>Partición</th>
                  <th>Frames</th>
                  <th>Clústeres presentes</th>
                </tr>
              </thead>
              <tbody>
                {counts.map((c) => (
                  <tr key={c.partition}>
                    <th scope="row">
                      {
                        partitionLabels[
                          c.partition as keyof typeof partitionLabels
                        ]
                      }
                    </th>
                    <td>{c.count}</td>
                    <td>{c.clusters}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Panel>
          <Panel
            title="Frames en orden de aparición"
            meta={`${tableFrames.length} coincidencias · 25 por página`}
          >
            <p className="panel-body">
              La tabla aplica el clúster resaltado y el intervalo. Un mismo
              contenido puede tener varios frame_id y permanece en varias filas.
            </p>
            <div className="toolbar">
              <button
                disabled={activeIndex <= 0}
                onClick={() => setFrameId(axisFrames[activeIndex - 1].frame_id)}
              >
                Frame anterior
              </button>
              <button
                disabled={
                  activeIndex < 0 || activeIndex >= axisFrames.length - 1
                }
                onClick={() => setFrameId(axisFrames[activeIndex + 1].frame_id)}
              >
                Frame siguiente
              </button>
              <button
                disabled={page === 0}
                onClick={() => setPage((p) => p - 1)}
              >
                Página anterior
              </button>
              <span>
                Página {page + 1} /{' '}
                {Math.max(1, Math.ceil(tableFrames.length / 25))}
              </span>
              <button
                disabled={(page + 1) * 25 >= tableFrames.length}
                onClick={() => setPage((p) => p + 1)}
              >
                Página siguiente
              </button>
            </div>
            <div className="timeline-table-scroll">
              <table className="timeline-table">
                <caption>Tabla accesible de ocurrencias temporales</caption>
                <thead>
                  <tr>
                    <th>Índice</th>
                    <th>Archivo / ocurrencia</th>
                    <th>Clúster</th>
                    <th>Partición</th>
                    <th>Acción</th>
                  </tr>
                </thead>
                <tbody>
                  {tableFrames.slice(page * 25, page * 25 + 25).map((f) => (
                    <tr key={f.frame_id}>
                      <td>{f.position}</td>
                      <td>
                        {f.filename}
                        <small>{f.frame_id}</small>
                      </td>
                      <td>{clusterLabel(f.cluster_id)}</td>
                      <td>{partitionLabels[f.new_split]}</td>
                      <td>
                        <button
                          aria-label={`Ver frame ${f.frame_id}`}
                          onClick={() => setFrameId(f.frame_id)}
                        >
                          Ver detalle
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Panel>
        </>
      ) : (
        <Notice>
          No hay ocurrencias temporalmente representables en este run. Todas
          permanecen contabilizadas en el reporte de exclusiones.
        </Notice>
      )}
      {active && (
        <FrameDetails
          key={`${run.run_id}:${active.frame_id}:${clusterRun}`}
          frame={active}
          data={data}
          runId={run.run_id}
          clusterRun={clusterRun}
        />
      )}
      <Panel title="Cobertura por vídeo / asociación y motivos de exclusión">
        <table className="timeline-table">
          <caption>
            Cobertura completa del run, independiente de los filtros
          </caption>
          <thead>
            <tr>
              <th>Eje o motivo</th>
              <th>Ocurrencias</th>
              <th>Procedencia</th>
            </tr>
          </thead>
          <tbody>
            {coverage.by_axis.map((a) => (
              <tr key={a.axis_id}>
                <th scope="row">{a.label}</th>
                <td>{a.occurrences}</td>
                <td>
                  {a.association === 'filename_inference'
                    ? 'Familia inferida; vídeo original no verificado'
                    : data.admission
                      ? 'Vídeo autorizado; solo se cuentan apariciones verificadas'
                      : 'Vídeo reportado'}
                </td>
              </tr>
            ))}
            {coverage.exclusion_reasons.map((r) => (
              <tr key={r.reason}>
                <th scope="row">
                  {reasonLabels[r.reason as keyof typeof reasonLabels]}
                </th>
                <td>{r.count}</td>
                <td>Excluida solo de la visualización temporal</td>
              </tr>
            ))}
          </tbody>
        </table>
        <details>
          <summary>
            Ocurrencias excluidas ({excluded.length}); primeras 25
          </summary>
          <ul>
            {excluded.slice(0, 25).map((f) => (
              <li key={f.frame_id}>
                <button onClick={() => setFrameId(f.frame_id)}>
                  {f.filename} · {reasonLabels[f.reason!]}
                </button>
              </li>
            ))}
          </ul>
          <p>
            El CSV contiene todas las exclusiones de este run, con frame_id,
            content_id, filename, reason, source_reference y
            temporal_provenance.
          </p>
        </details>
      </Panel>
      <TechnicalDetailsDrawer title="Procedencia y limitaciones científicas">
        <p>
          Los tramos candidatos resumen el alcance observado de agrupaciones
          existentes. No se deducen secuencias de cambios de color. Las zonas de
          frontera del contrato v2 no se convierten en secuencias.
        </p>
        <ul>
          {data.limitations.map((text) => (
            <li key={text}>{text}</li>
          ))}
        </ul>
        <dl className="timeline-facts">
          <div>
            <dt>Commit de la publicación</dt>
            <dd>{data.source_commit}</dd>
          </div>
          <div>
            <dt>Fuente con cambios locales</dt>
            <dd>{String(data.source_dirty)}</dd>
          </div>
          <div>
            <dt>Receipt de fuente SHA-256</dt>
            <dd>{data.source_receipt_sha256}</dd>
          </div>
          <div>
            <dt>Manifest organización SHA-256</dt>
            <dd>{data.organization_manifest_sha256}</dd>
          </div>
        </dl>
      </TechnicalDetailsDrawer>
    </div>
  )
}

function FrameDetails({
  frame,
  data,
  runId,
  clusterRun,
}: {
  frame: TimelineFrame
  data: ScientificTimelineData
  runId: string
  clusterRun: string
}) {
  const [imageFailed, setImageFailed] = useState(false)
  const run = data.runs.find((r) => r.run_id === runId)!
  const axis = data.axes.find((a) => a.axis_id === frame.axis_id)
  const facts = [
    ['frame_id', frame.frame_id],
    ['content_id', frame.content_id],
    ['Nombre de archivo original', frame.filename],
    [
      'Vídeo de origen',
      axis?.source_video_id ??
        (axis ? `${axis.label} — familia inferida; vídeo no verificado` : null),
    ],
    ['Posición / índice de aparición', frame.position],
    ['Tipo de orden temporal', frame.order_basis],
    ['Procedencia temporal', frame.temporal_provenance],
    ['Evidencia de correspondencia', frame.source_reference],
    ['Fuente de la asignación', run.source_reference],
    ['cluster_run_id', clusterRun || null],
    [
      'cluster_id',
      frame.cluster_id === null ? 'Clúster no disponible' : frame.cluster_id,
    ],
    [
      'split_space_id',
      run.split_space_id ??
        'No exportado; membership_id se conserva por separado',
    ],
    ['membership_id', run.membership_id],
    ['new_split', frame.new_split],
    ['Secuencia asociada', frame.sequence_id ?? 'Sin secuencia confirmada'],
    [
      'Exclusión temporal',
      frame.reason ? reasonLabels[frame.reason] : 'No excluida',
    ],
  ] as const
  return (
    <Panel title="Detalle del frame" meta={run.run_id}>
      <section aria-label="Detalle del frame" className="timeline-frame-detail">
        <div>
          {frame.preview_path && !imageFailed ? (
            <img
              className="timeline-preview"
              src={organizationMediaUrl(frame.preview_path)}
              alt={`Miniatura sin anotaciones: ${frame.filename}`}
              loading="lazy"
              onError={() => setImageFailed(true)}
            />
          ) : (
            <p>Miniatura no disponible. La membresía permanece intacta.</p>
          )}
          <p className="micro">
            Sin anotaciones protegidas, predicciones ni métricas del detector.
          </p>
        </div>
        <dl className="timeline-facts">
          {facts.map(([label, value]) => (
            <div key={label}>
              <dt>{label}</dt>
              <dd>{value ?? 'No disponible'}</dd>
            </div>
          ))}
        </dl>
      </section>
    </Panel>
  )
}
