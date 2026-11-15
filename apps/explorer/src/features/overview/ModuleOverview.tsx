import { Link } from 'react-router-dom'
import { ArrowRight, ExternalLink } from 'lucide-react'
import { useSnapshot } from '@/data/provider'
import { Notice, Panel } from '@/components/feedback/Primitives'
import { Status } from '@/components/feedback/Status'
export function ModuleOverview({
  id,
}: {
  id: 'preprocessing' | 'organization' | 'segmentation'
}) {
  const { data } = useSnapshot()
  const module = data?.modules.find((m) => m.id === id)
  if (!module) return null
  return (
    <>
      <Notice>
        <strong>Repository evidence:</strong> {module.evidence}{' '}
        <a href={module.repository} target="_blank" rel="noreferrer">
          Inspect source <ExternalLink size={12} />
        </a>
      </Notice>
      <Panel
        title="Stages & artifact handoffs"
        meta="SOURCE STRUCTURE · execution not reverified"
      >
        <div className="stage-flow">
          {module.stages.map((s) => (
            <Link key={s.id} to={`/${id}/${s.id}`}>
              <Status state={s.status} />
              <h3>{s.name}</h3>
              <p>{s.output}</p>
              <ArrowRight size={17} />
            </Link>
          ))}
        </div>
      </Panel>
    </>
  )
}
