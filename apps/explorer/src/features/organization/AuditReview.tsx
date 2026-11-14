import { useState } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import {
  auditBase,
  auditMedia,
  loadAudit,
  saveAuditDecision,
  needsEvidence,
  temporalOrder,
  externalMedia,
} from '@/data/audit-client'
import type {
  AuditDecision,
  AuditEvent,
  DecisionInput,
  AuditSnapshot,
} from '@/data/audit-client'

export function AuditReview() {
  const query = useQuery({
    queryKey: ['local-audit'],
    queryFn: loadAudit,
    retry: false,
  })
  const [selected, setSelected] = useState('')
  const [filters, setFilters] = useState({
    source: '',
    method: '',
    status: '',
    alert: '',
    evidence: '',
  })
  const [descending, setDescending] = useState(false)
  if (query.isPending) return <p role="status">Abriendo auditoría local…</p>
  if (query.isError)
    return (
      <section>
        <h2>Auditoría local no disponible</h2>
        <p role="alert">{query.error.message}</p>
        <p>
          Inicie el servicio con{' '}
          <code>
            uv run flir-pipeline sequences audit serve --db RUTA_LOCAL
          </code>
          . No se sustituye por datos de demostración.
        </p>
        <button onClick={() => void query.refetch()}>Reintentar</button>
      </section>
    )
  const data = query.data
  const latest = new Map(data.decisions.map((d) => [d.event_id, d]))
  const available = (id: string) =>
    !!data.events.find((e) => e.event_id === id)?.evidence.length ||
    data.external_evidence.some((e) => e.event_id === id)
  const state = (id: string) => {
    const d = latest.get(id)
    return d && data.legacy_review_warnings.includes(d.decision_id)
      ? 'needs_evidence'
      : (d?.status ?? 'pending')
  }
  const events = temporalOrder(
    data.events.filter(
      (e) =>
        (!filters.source || (e.source_video ?? 'unknown') === filters.source) &&
        (!filters.method || e.method === filters.method) &&
        (!filters.evidence ||
          (filters.evidence === 'registered') === available(e.event_id)) &&
        (!filters.alert || e.alert_type === filters.alert) &&
        (!filters.status || state(e.event_id) === filters.status),
    ),
    descending,
  )
  const active = events.find((e) => e.event_id === selected) ?? events[0]
  const synthetic =
    data.assets.length > 0 &&
    data.assets.every((a) => a.metadata.synthetic === true)
  const options = {
    source: [...new Set(data.events.map((e) => e.source_video ?? 'unknown'))],
    method: [...new Set(data.events.map((e) => e.method))],
    alert: [...new Set(data.events.map((e) => e.alert_type))],
    status: ['pending', 'reviewed', 'needs_evidence'],
    evidence: ['registered', 'missing'],
  }
  return (
    <section className="audit-review">
      <h2>FLIR Audit &amp; Human Review v1.1</h2>
      <p>
        <strong>
          {synthetic
            ? 'DEMO · Evidencia completamente sintética'
            : 'Evidencia local exploratoria'}
        </strong>
      </p>
      <p>
        Los tiempos son nominales, no alineaciones verificadas. Un corte
        audiovisual no demuestra independencia visual. Un vídeo completo tampoco
        define una secuencia indivisible. La revisión no crea secuencias ni
        particiones.
      </p>
      <p>
        {data.events.length} eventos · {data.observation_count} observaciones ·{' '}
        {data.decisions.length} revisiones guardadas.
      </p>
      <div className="toolbar">
        {(Object.keys(options) as Array<keyof typeof options>).map((key) => (
          <label key={key}>
            {
              {
                source: 'Origen',
                method: 'Método',
                alert: 'Tipo de alerta',
                status: 'Estado',
                evidence: 'Evidencia visual',
              }[key]
            }
            <select
              aria-label={
                key === 'evidence'
                  ? 'Evidencia visual'
                  : key === 'status'
                    ? 'Filtrar estado'
                    : key
              }
              value={filters[key]}
              onChange={(e) =>
                setFilters({ ...filters, [key]: e.target.value })
              }
            >
              <option value="">Todos</option>
              {options[key].map((value) => (
                <option key={value} value={value}>
                  {value === 'registered'
                    ? 'Con evidencia registrada'
                    : value === 'missing'
                      ? 'Sin evidencia visual'
                      : value}
                </option>
              ))}
            </select>
          </label>
        ))}
      </div>
      <p>
        Con evidencia registrada:{' '}
        {data.events.filter((e) => available(e.event_id)).length} · Sin
        evidencia visual:{' '}
        {data.events.filter((e) => !available(e.event_id)).length}. La
        disponibilidad no acredita que la imagen haya sido consultada.
      </p>
      <div className="toolbar">
        <label>
          Orden temporal
          <select
            aria-label="Orden temporal"
            value={descending ? 'desc' : 'asc'}
            onChange={(e) => setDescending(e.target.value === 'desc')}
          >
            <option value="asc">Ascendente por fuente</option>
            <option value="desc">Descendente por fuente</option>
          </select>
        </label>
        <button
          type="button"
          disabled={!events.some((e) => state(e.event_id) !== 'reviewed')}
          onClick={() => {
            const index = events.findIndex(
              (e) => e.event_id === active?.event_id,
            )
            const next = [
              ...events.slice(index + 1),
              ...events.slice(0, index + 1),
            ].find((e) => state(e.event_id) !== 'reviewed')
            if (next) setSelected(next.event_id)
          }}
        >
          Siguiente pendiente
        </button>
      </div>
      {!!data.legacy_review_warnings.length && (
        <p role="alert">
          Hay revisiones v1 sin evidencia suficiente. Se conservan; requieren
          una nueva revisión explícita.
        </p>
      )}
      {!!data.anomalies.length && (
        <details>
          <summary>Anomalías históricas ({data.anomalies.length})</summary>
          <p>Compartir un índice no confirma leakage semántico.</p>
          <pre>{JSON.stringify(data.anomalies, null, 2)}</pre>
        </details>
      )}
      <label>
        Evento candidato
        <select
          aria-label="Evento candidato"
          value={active?.event_id ?? ''}
          onChange={(e) => setSelected(e.target.value)}
        >
          {events.map((e) => (
            <option key={e.event_id} value={e.event_id}>
              {e.record_locator} · {e.start_second ?? '?'} s ·{' '}
              {state(e.event_id)} ?{' '}
              {available(e.event_id)
                ? 'evidencia registrada'
                : 'sin evidencia visual'}{' '}
              · {e.event_id.slice(0, 8)}
            </option>
          ))}
        </select>
      </label>
      {!active ? (
        <p>No hay eventos para estos filtros.</p>
      ) : (
        <ReviewCase
          key={`${active.event_id}:${latest.get(active.event_id)?.version ?? 0}`}
          event={active}
          external={data.external_evidence.filter(
            (e) => e.event_id === active.event_id,
          )}
          companionHistory={data.decisions.filter((d) =>
            data.companions.some((c) =>
              c.event_links.some(
                (l) =>
                  l.canonical_event_id === active.event_id &&
                  l.companion_event_id === d.event_id,
              ),
            ),
          )}
          history={data.decisions.filter((d) => d.event_id === active.event_id)}
          filename={
            data.assets.find((a) => a.asset_id === active.asset_id)?.filename ??
            'unknown'
          }
        />
      )}
      <p>
        <a
          href={`${auditBase()}/api/export/json`}
          download="audit-decisions.json"
        >
          Exportar JSON
        </a>{' '}
        ·{' '}
        <a
          href={`${auditBase()}/api/export/csv`}
          download="audit-decisions.csv"
        >
          Exportar CSV
        </a>
      </p>
      <details>
        <summary>Informe de importación y archivos ausentes</summary>
        <pre style={{ whiteSpace: 'pre-wrap' }}>
          {JSON.stringify(data.last_import, null, 2)}
        </pre>
      </details>
    </section>
  )
}

function ReviewCase({
  event,
  history,
  filename,
  external,
  companionHistory,
}: {
  event: AuditEvent
  history: AuditDecision[]
  filename: string
  external: AuditSnapshot['external_evidence']
  companionHistory: AuditDecision[]
}) {
  const client = useQueryClient()
  const previous = history.at(-1)
  const [form, setForm] = useState<DecisionInput>({
    event_id: event.event_id,
    expected_version: previous?.version ?? 0,
    status: previous?.status ?? 'pending',
    continuity: previous?.continuity ?? 'uncertain',
    dependence: previous?.dependence ?? 'uncertain',
    confidence: previous?.confidence ?? 'undetermined',
    reviewer: previous?.reviewer ?? '',
    justification: previous?.justification ?? '',
    phenomena: previous?.phenomena ?? [],
    evidence_consulted: [],
    external_evidence_consulted: [],
  })
  const blocked = needsEvidence(form)
  const [mediaIndex, setMediaIndex] = useState(0)
  const [mediaError, setMediaError] = useState(false)
  const [message, setMessage] = useState('')
  const [saving, setSaving] = useState(false)
  const selections = {
    status: ['pending', 'reviewed'],
    continuity: ['continuous', 'shot_change', 'uncertain'],
    dependence: ['same_region', 'different_region', 'uncertain'],
    confidence: ['undetermined', 'low', 'medium', 'high'],
  } as const
  const labels = {
    status: 'Estado de revisión',
    continuity: 'Continuidad audiovisual',
    dependence: 'Dependencia visual o de región',
    confidence: 'Confianza',
  }
  return (
    <article>
      <h3>Inspección del candidato</h3>
      <p>
        Archivo: {filename} · Vídeo: {event.source_video ?? 'No declarado'} ·
        Método: {event.method} ({event.method_version})
      </p>
      <p>
        Índices {event.start_index ?? '?'}–{event.end_index ?? '?'} · Tiempo
        estimado {event.start_second ?? '?'}–{event.end_second ?? '?'} s
      </p>
      <p>
        Muestras {event.event_start_sample ?? '?'}–
        {event.event_end_sample ?? '?'} · Posición representativa{' '}
        {event.representative_second ?? '?'} s · Precisión:{' '}
        {event.precision_seconds ?? 'no declarada'}. {event.temporal_basis}
      </p>
      <details>
        <summary>Identidad y métricas automáticas</summary>
        <p>{event.event_id}</p>
        <p>{event.record_locator}</p>
        <pre>{JSON.stringify(event.metrics, null, 2)}</pre>
      </details>
      {event.evidence.length ? (
        <figure>
          <label>
            Contexto disponible
            <select
              value={mediaIndex}
              onChange={(e) => {
                setMediaIndex(Number(e.target.value))
                setMediaError(false)
              }}
            >
              {event.evidence.map((name, i) => (
                <option key={name} value={i}>
                  {name}
                </option>
              ))}
            </select>
          </label>
          {mediaError ? (
            <p role="alert">Imagen ausente o no legible.</p>
          ) : (
            <a
              href={auditMedia(event.event_id, mediaIndex)}
              target="_blank"
              rel="noreferrer"
            >
              <img
                src={auditMedia(event.event_id, mediaIndex)}
                alt="Evidencia de contexto antes/después; tiempos nominales rotulados en la lámina"
                style={{
                  maxWidth: '100%',
                  maxHeight: 600,
                  objectFit: 'contain',
                }}
                onError={() => setMediaError(true)}
              />
            </a>
          )}
          <figcaption>
            Lámina de contexto original. Abrir imagen para inspeccionar a
            resolución completa. No se reconstruyen fotogramas ausentes.
          </figcaption>
          <label>
            <input
              type="checkbox"
              checked={form.evidence_consulted.includes(
                event.evidence[mediaIndex],
              )}
              onChange={(e) =>
                setForm({
                  ...form,
                  evidence_consulted: e.target.checked
                    ? [
                        ...new Set([
                          ...form.evidence_consulted,
                          event.evidence[mediaIndex],
                        ]),
                      ]
                    : form.evidence_consulted.filter(
                        (x) => x !== event.evidence[mediaIndex],
                      ),
                })
              }
            />
            Consulté esta evidencia
          </label>
        </figure>
      ) : (
        <p>
          Evidencia visual no disponible para este evento; no se infiere a
          partir de tiempos cercanos.
        </p>
      )}
      {external.map((e) => (
        <figure key={e.evidence_id}>
          <img
            src={externalMedia(e.evidence_id)}
            alt={`Evidencia externa registrada: ${e.filename}`}
            style={{ maxWidth: '100%', maxHeight: 450, objectFit: 'contain' }}
          />
          <figcaption>
            {e.filename} · {e.locator} · Asociación: {e.basis} · {e.reviewer}.{' '}
            {e.rationale}
          </figcaption>
          <details>
            <summary>Procedencia y hash externo</summary>
            <pre style={{ whiteSpace: 'pre-wrap' }}>
              {JSON.stringify(e, null, 2)}
            </pre>
          </details>
          <label>
            <input
              type="checkbox"
              checked={form.external_evidence_consulted.includes(e.evidence_id)}
              onChange={(event) =>
                setForm({
                  ...form,
                  external_evidence_consulted: event.target.checked
                    ? [...form.external_evidence_consulted, e.evidence_id]
                    : form.external_evidence_consulted.filter(
                        (id) => id !== e.evidence_id,
                      ),
                })
              }
            />
            Consulté la evidencia externa {e.filename}
          </label>
        </figure>
      ))}
      {!event.evidence.length && !external.length && (
        <details>
          <summary>Registrar evidencia externa verificable</summary>
          <p>
            El servicio no busca archivos automáticamente. Registre una imagen
            existente mediante <code>sequences audit attach-evidence</code>,
            indicando este event_id, localizador, base de correspondencia,
            justificación y revisor. Después recargue la auditoría. La cercanía
            temporal por sí sola no acredita la asociación.
          </p>
          <code>{event.event_id}</code>
        </details>
      )}
      <form
        onSubmit={(e) => {
          e.preventDefault()
          if (blocked) {
            setMessage(
              'Una revisión conclusiva requiere evidencia consultada y trazable.',
            )
            return
          }
          setSaving(true)
          setMessage('')
          void saveAuditDecision(form)
            .then(async () => {
              await client.invalidateQueries({ queryKey: ['local-audit'] })
            })
            .catch((error: Error) => setMessage(error.message))
            .finally(() => setSaving(false))
        }}
      >
        <h4>Juicio humano independiente</h4>
        <div className="toolbar">
          {(Object.keys(selections) as Array<keyof typeof selections>).map(
            (key) => (
              <label key={key}>
                {labels[key]}
                <select
                  aria-label={labels[key]}
                  value={form[key]}
                  onChange={(e) => setForm({ ...form, [key]: e.target.value })}
                >
                  {selections[key].map((value) => (
                    <option key={value}>{value}</option>
                  ))}
                </select>
              </label>
            ),
          )}
        </div>
        <label>
          Revisor
          <input
            required
            maxLength={200}
            value={form.reviewer}
            onChange={(e) => setForm({ ...form, reviewer: e.target.value })}
          />
        </label>
        <label>
          Justificación
          <textarea
            aria-label="Justificación"
            required
            maxLength={10000}
            value={form.justification}
            onChange={(e) =>
              setForm({ ...form, justification: e.target.value })
            }
          />
        </label>
        <fieldset>
          <legend>Fenómenos observados</legend>
          {(
            [
              'zoom',
              'camera_motion',
              'gain_change',
              'polarity_change',
              'signal_loss',
              'unknown',
            ] as const
          ).map((tag) => (
            <label key={tag}>
              <input
                type="checkbox"
                checked={form.phenomena.includes(tag)}
                onChange={(e) =>
                  setForm({
                    ...form,
                    phenomena: e.target.checked
                      ? [...form.phenomena, tag]
                      : form.phenomena.filter((t) => t !== tag),
                  })
                }
              />
              {tag}{' '}
            </label>
          ))}
        </fieldset>
        {blocked && (
          <p role="alert">
            Seleccione evidencia consultada para guardar una revisión
            conclusiva. Puede mantener el caso pendiente o indeterminado.
          </p>
        )}
        <button disabled={saving || blocked} type="submit">
          {saving ? 'Guardando…' : 'Guardar revisión'}
        </button>
        {message && <p role="alert">{message}</p>}
      </form>
      {!!companionHistory.length && (
        <details>
          <summary>
            Historial conservado de fuente compañera ({companionHistory.length})
          </summary>
          <p>
            Estas revisiones conservan sus event_id originales. No se
            trasladaron al evento canónico ni se resolvieron discrepancias
            automáticamente.
          </p>
          <pre style={{ whiteSpace: 'pre-wrap' }}>
            {JSON.stringify(companionHistory, null, 2)}
          </pre>
        </details>
      )}
      <h4>Historial persistido ({history.length})</h4>
      {history.length === 0 ? (
        <p>Pendiente: sin decisión humana.</p>
      ) : (
        <ol>
          {history.map((d) => (
            <li key={d.decision_id}>
              v{d.version} · {d.created_at} · {d.reviewer} · {d.status} ·{' '}
              {d.continuity} + {d.dependence} · {d.confidence}
              <p>{d.justification}</p>
              <details>
                <summary>Procedencia de la revisión</summary>
                <pre style={{ whiteSpace: 'pre-wrap' }}>
                  {JSON.stringify(d, null, 2)}
                </pre>
              </details>
            </li>
          ))}
        </ol>
      )}
    </article>
  )
}
