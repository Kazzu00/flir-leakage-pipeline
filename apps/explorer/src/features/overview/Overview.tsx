import { Link } from 'react-router-dom'
import {
  ArrowDown,
  ArrowRight,
  ArrowUpRight,
  Box,
  GitBranch,
  Layers,
  Video,
} from 'lucide-react'
import { useSnapshot } from '@/data/provider'
import { PageTitle, Notice } from '@/components/feedback/Primitives'
import { Status } from '@/components/feedback/Status'
const icons = [Layers, GitBranch, Box]
export function Overview() {
  const { data } = useSnapshot()
  if (!data) return null
  return (
    <>
      <PageTitle
        eyebrow="RESEARCH / END-TO-END WORKFLOW"
        title="One pipeline. Three perspectives."
        description="Trace aerial imagery from source video to auditable data and panoptic predictions."
      />
      <div className="overview-meta">
        <span>
          <span className="tiny-dot" /> 3 independent research modules
        </span>
        <span>CRISP-ML(Q) · Understanding → evaluation</span>
        <Link to="/experiments">
          {data.leakage
            ? 'Browse mixed experiment registry'
            : 'Browse demo experiments'}{' '}
          <ArrowRight size={14} />
        </Link>
      </div>
      <div className="pipeline-input">
        <span className="input-icon">
          <Video size={20} />
        </span>
        <div>
          <strong>FAC aerial videos</strong>
          <span>Source imagery · Colombian Amazon</span>
        </div>
        <span className="pipeline-note">
          SOURCE INPUT <ArrowDown size={15} />
        </span>
      </div>
      <div className="pipeline-modules">
        {data.modules.map((m, i) => {
          const Icon = icons[i]
          return (
            <article key={m.id} className={`module-zone module-${i}`}>
              <div className="module-number" aria-hidden="true">
                {m.number}
                <span />
              </div>
              <div className="module-heading">
                <Icon size={23} />
                <Status state={m.status} />
              </div>
              <Link to={`/${m.id}`} className="module-title">
                <h2>{m.name}</h2>
                <ArrowUpRight size={23} />
              </Link>
              <p className="module-subtitle">{m.subtitle}</p>
              {m.id === 'organization' && (
                <div className="home-fork">
                  <span>Visual structure</span>
                  <span>↘ Representation & organization ↗</span>
                  <span>Temporal structure</span>
                </div>
              )}
              <p className="module-evidence">
                {m.id === 'organization' && data.leakage
                  ? 'ARTIFACT · missing stages pending'
                  : 'DEMO · synthetic exploration'}
              </p>
              <div className="module-foot">
                <span className="micro">SOURCE EVIDENCE · NOT REVERIFIED</span>
                <Link to={`/${m.id}`}>
                  Explore module <ArrowRight size={16} />
                </Link>
              </div>
            </article>
          )
        })}
      </div>
      <Link to="/evaluation" className="evaluation-lane">
        <span className="evaluation-symbol">↳</span>
        <div>
          <span className="eyebrow">GLOBAL EVALUATION</span>
          <h2>Compare evidence across the pipeline</h2>
          <p>
            Quality · residual correlation · controlled detection · panoptic
            evaluation
          </p>
        </div>
        <span className="status status-pending">○ pending integration</span>
        <ArrowRight size={23} />
      </Link>
      <Notice>
        <strong>Evidence has boundaries.</strong> A visual cluster is not a
        validated sequence. An infrastructure pilot is not a scientific result.
        The three scientific repositories remain independent.
      </Notice>
      <div className="overview-bottom">
        <div>
          <span className="eyebrow">EXPLORATION PATH</span>
          <p>
            Pipeline <span>→</span> Module <span>→</span> Stage <span>→</span>{' '}
            Run / artifact
          </p>
        </div>
        <div>
          <span className="eyebrow">INTEGRATION STATUS</span>
          <p>
            {data.leakage
              ? 'M02 local artifacts · M01/M03 DEMO'
              : 'Mock adapter active · APIs not connected'}
          </p>
        </div>
      </div>
    </>
  )
}
