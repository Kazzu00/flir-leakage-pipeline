import type { Point } from '@/contracts'
import { ScanLine } from 'lucide-react'
/** No private imagery: this is an explicit unavailable-media state. */
export function SampleImage({
  label = 'Image not connected',
  overlay = false,
  opacity = 0.5,
}: {
  label?: string
  overlay?: boolean
  opacity?: number
}) {
  return (
    <div className="sample-image">
      <ScanLine size={32} strokeWidth={1} aria-hidden="true" />
      <span>{label}</span>
      <small>SAMPLE · NO SOURCE IMAGE</small>
      {overlay && (
        <div
          aria-hidden="true"
          className="sample-overlay"
          style={{ opacity }}
        />
      )}
    </div>
  )
}
export function Gallery({
  points,
  selected,
  onSelect,
}: {
  points: Point[]
  selected?: string
  onSelect?: (id: string) => void
}) {
  return (
    <div className="gallery">
      {points.slice(0, 6).map((p) => (
        <button
          key={p.contentId}
          className={`gallery-item ${selected === p.contentId ? 'selected' : ''}`}
          onClick={() => onSelect?.(p.contentId)}
          aria-label={`Inspect ${p.contentId}`}
        >
          <SampleImage />
          <span className="mono">{p.contentId}</span>
          <small>
            {p.sourceVideo} · index {p.sampleIndex}
          </small>
        </button>
      ))}
      {points.length === 0 && (
        <p className="empty">No contents in this selection.</p>
      )}
    </div>
  )
}
