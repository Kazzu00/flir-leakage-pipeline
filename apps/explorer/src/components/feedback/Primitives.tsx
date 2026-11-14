import type { ReactNode } from 'react'
import type { MetricValue } from '@/contracts'
import { Info } from 'lucide-react'
export function PageTitle({
  eyebrow,
  title,
  description,
  children,
}: {
  eyebrow: string
  title: string
  description: string
  children?: ReactNode
}) {
  return (
    <div className="page-title">
      <div>
        <p className="eyebrow">{eyebrow}</p>
        <h1>{title}</h1>
        <p className="muted">{description}</p>
      </div>
      {children}
    </div>
  )
}
export function Notice({ children }: { children: ReactNode }) {
  return (
    <div className="notice">
      <Info size={16} aria-hidden="true" />
      <div>{children}</div>
    </div>
  )
}
export function Panel({
  title,
  meta,
  children,
  className = '',
}: {
  title: string
  meta?: string
  children: ReactNode
  className?: string
}) {
  return (
    <section className={`panel ${className}`}>
      <header className="panel-header">
        <h2>{title}</h2>
        {meta && <span className="micro muted">{meta}</span>}
      </header>
      {children}
    </section>
  )
}
export function Metrics({ metrics }: { metrics: MetricValue[] }) {
  return (
    <dl className="metrics">
      {metrics.map((m) => (
        <div key={m.name}>
          <dt>{m.name}</dt>
          <dd>
            {m.value === null ? 'Unavailable' : `${m.value}${m.unit}`}
            <small style={{ display: 'block', marginTop: 4 }}>
              {m.origin === 'mock' ? 'DEMO' : m.origin}
            </small>
          </dd>
        </div>
      ))}
    </dl>
  )
}
export function Select({
  label,
  value,
  onChange,
  options,
}: {
  label: string
  value: string
  onChange: (value: string) => void
  options: { value: string; label: string }[]
}) {
  return (
    <label className="select-field">
      <span>{label}</span>
      <select
        aria-label={label}
        value={value}
        onChange={(e) => onChange(e.target.value)}
      >
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    </label>
  )
}
