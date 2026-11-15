import { useRef, type PointerEvent } from 'react'
import {
  clusterColor,
  clusterLabel,
  partitionLabels,
  type TimelineFrame,
  type TimelineInterval,
} from '@/data/scientific-timeline'

export const splitColors = {
  train: '#237c69',
  val: '#986700',
  test: '#435cc4',
  quarantine: '#b94d73',
  unassigned: '#758195',
}

/** The three lanes share ONE coordinate system and horizontal scroll container.
 * Equal-index occurrences are stacked, never offset to invent a temporal order.
 * Keyboard access uses the bounded occurrence table and previous/next controls.
 */
export function ScientificTimelineLanes({
  frames,
  intervals,
  clusterRunId,
  clusterFilter,
  selectedId,
  zoom,
  onFrame,
  onInterval,
  onRange,
  range,
}: {
  frames: TimelineFrame[]
  intervals: TimelineInterval[]
  clusterRunId: string
  clusterFilter: string
  selectedId: string
  zoom: number
  range: [number, number]
  onFrame: (id: string) => void
  onInterval: (id: string) => void
  onRange: (start: number, end: number) => void
}) {
  const down = useRef<number | null>(null)
  const min = frames[0]?.position ?? 0
  const max = frames.at(-1)?.position ?? min
  const span = Math.max(1, max - min)
  const plotWidth = 1000 * zoom
  const trackWidth = plotWidth - 48
  const x = (position: number) => 24 + ((position - min) / span) * trackWidth
  const slots = new Map<number, number>()
  const marks = frames.map((frame) => {
    const position = frame.position!
    const slot = slots.get(position) ?? 0
    slots.set(position, slot + 1)
    return { frame, slot }
  })
  const stacks = Math.max(1, ...slots.values())
  const frameY = 54
  const intervalY = frameY + stacks * 23 + 42
  const splitY = intervalY + Math.max(1, intervals.length) * 43 + 38
  const height = splitY + stacks * 23 + 34
  const coordinate = (event: PointerEvent<SVGSVGElement>) => {
    const rect = event.currentTarget.getBoundingClientRect()
    return (
      min +
      Math.max(
        0,
        Math.min(
          1,
          (((event.clientX - rect.left) / rect.width) * plotWidth - 24) /
            trackWidth,
        ),
      ) *
        span
    )
  }
  const up = (event: PointerEvent<SVGSVGElement>) => {
    if (down.current === null) return
    const start = down.current
    down.current = null
    const end = coordinate(event)
    if (Math.abs(x(start) - x(end)) > 6) {
      onRange(
        Math.round(Math.min(start, end)),
        Math.round(Math.max(start, end)),
      )
      return
    }
    const target = event.target as Element
    const interval = target
      .closest('[data-interval-id]')
      ?.getAttribute('data-interval-id')
    if (interval) {
      onInterval(interval)
      return
    }
    const id = target.closest('[data-frame-id]')?.getAttribute('data-frame-id')
    if (id) {
      onFrame(id)
      return
    }
    const nearest = frames.reduce<TimelineFrame | undefined>(
      (best, frame) =>
        !best ||
        Math.abs(frame.position! - end) < Math.abs(best.position! - end)
          ? frame
          : best,
      undefined,
    )
    if (nearest) onFrame(nearest.frame_id)
  }
  const emphasized = (f: TimelineFrame) =>
    (clusterFilter === '' || String(f.cluster_id) === clusterFilter) &&
    f.position! >= range[0] &&
    f.position! <= range[1]

  return (
    <div
      className="scientific-timeline-scroll"
      tabIndex={0}
      role="region"
      aria-label="Desplazamiento horizontal de las tres pistas"
    >
      <div style={{ minWidth: `${Math.max(100, zoom * 100)}%` }}>
        <svg
          className="scientific-timeline-svg"
          viewBox={`0 0 ${plotWidth} ${height}`}
          role="img"
          aria-label={`Tres pistas sincronizadas: ${frames.length} ocurrencias en orden de aparición`}
          onPointerDown={(event) => {
            down.current = coordinate(event)
          }}
          onPointerUp={up}
          onPointerCancel={() => {
            down.current = null
          }}
        >
          <title>
            Clústeres, tramos temporales y particiones por ocurrencia
          </title>
          <desc>
            Selecciona una marca o arrastra un intervalo. La tabla y los
            controles permiten las mismas selecciones con teclado.
          </desc>
          {(max === min ? [0] : [0, 0.25, 0.5, 0.75, 1]).map((ratio) => (
            <g key={ratio}>
              <line
                x1={24 + ratio * trackWidth}
                x2={24 + ratio * trackWidth}
                y1={32}
                y2={height - 16}
                className="timeline-guide"
              />
              <text
                x={24 + ratio * trackWidth}
                y={17}
                textAnchor={
                  ratio === 0 ? 'start' : ratio === 1 ? 'end' : 'middle'
                }
              >
                {Math.round(min + ratio * (max - min))}
              </text>
            </g>
          ))}
          <rect
            x={x(Math.max(min, range[0]))}
            width={Math.max(
              1,
              x(Math.min(max, range[1])) - x(Math.max(min, range[0])),
            )}
            y={33}
            height={height - 47}
            className="timeline-range"
          />
          <text x={24} y={frameY - 10}>
            1 · Frames originales — color = clúster
          </text>
          {marks.map(({ frame: f, slot }) => (
            <rect
              key={f.frame_id}
              data-frame-id={f.frame_id}
              data-lane="cluster"
              x={x(f.position!) - 1.25}
              y={frameY + slot * 23}
              width={2.5}
              height={17}
              fill={clusterColor(clusterRunId, f.cluster_id)}
              opacity={emphasized(f) ? 1 : 0.2}
              className={
                selectedId === f.frame_id ? 'timeline-selected-mark' : ''
              }
            >
              <title>{`${f.filename} · índice ${f.position} · ${clusterLabel(f.cluster_id)} · ${f.frame_id}`}</title>
            </rect>
          ))}
          <text x={24} y={intervalY - 10}>
            2 · Secuencias / agrupaciones temporales de la fuente
          </text>
          {!intervals.length && (
            <text x={24} y={intervalY + 18}>
              Sin intervalos exportados; no se crean fronteras a partir de
              clústeres.
            </text>
          )}
          {intervals.map((interval, index) => {
            const members = frames.filter(
              (f) =>
                f.position! >= interval.start && f.position! <= interval.end,
            )
            const counts = new Map<number | null, number>()
            for (const f of members)
              counts.set(f.cluster_id, (counts.get(f.cluster_id) ?? 0) + 1)
            const distribution = [...counts]
              .map(([id, n]) => `${clusterLabel(id)}: ${n}`)
              .join(' · ')
            return (
              <g
                key={interval.interval_id}
                data-interval-id={interval.interval_id}
                className="timeline-interval"
              >
                <rect
                  x={x(interval.start)}
                  y={intervalY + index * 43}
                  width={Math.max(3, x(interval.end) - x(interval.start))}
                  height={31}
                  rx={4}
                  className={
                    interval.kind === 'candidate_span'
                      ? 'timeline-candidate-block'
                      : 'timeline-reviewed-block'
                  }
                />
                <text x={x(interval.start) + 8} y={intervalY + index * 43 + 21}>
                  {interval.kind === 'candidate_span'
                    ? 'Tramo candidato'
                    : 'Secuencia'}{' '}
                  · {interval.start}–{interval.end} · {members.length} frames ·{' '}
                  {interval.review}
                </text>
                <title>{`${interval.interval_id} · ${members.length} frames · ${distribution} · ${interval.provenance} · ${interval.source_reference}`}</title>
              </g>
            )
          })}
          <text x={24} y={splitY - 10}>
            3 · Particiones — una marca por ocurrencia, sin split dominante
          </text>
          {marks.map(({ frame: f, slot }) => (
            <rect
              key={f.frame_id}
              data-frame-id={f.frame_id}
              data-lane="split"
              data-partition={f.new_split}
              x={x(f.position!) - 1.25}
              y={splitY + slot * 23}
              width={2.5}
              height={17}
              fill={splitColors[f.new_split]}
              opacity={emphasized(f) ? 1 : 0.2}
              className={
                selectedId === f.frame_id ? 'timeline-selected-mark' : ''
              }
            >
              <title>{`${f.filename} · índice ${f.position} · ${partitionLabels[f.new_split]} · ${clusterLabel(f.cluster_id)}`}</title>
            </rect>
          ))}
        </svg>
      </div>
    </div>
  )
}
