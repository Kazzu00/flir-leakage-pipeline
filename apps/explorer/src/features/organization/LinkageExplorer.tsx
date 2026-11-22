import { useState } from 'react'
import type { Research } from '@/contracts/research'
import { Panel, Notice } from '@/components/feedback/Primitives'
import {
  ArtifactStatus,
  EmptyArtifactState,
  MetricSummary,
} from '@/components/research/Evidence'
import { TechnicalDetailsDrawer } from '@/components/research/TechnicalDetailsDrawer'
import { Pager } from './evidence-views/Pager'
export function LinkageExplorer({ research }: { research?: Research }) {
  const [page, setPage] = useState(0)
  const [selected, setSelected] = useState('')
  const artifact = research?.linkage.artifact
  const candidate = artifact?.candidates.find((c) => c.id === selected)
  const aggregate = research?.reviewAggregation.artifact
  const groupReview = research?.groupReview.artifact
  return (
    <>
      <h2>Which labeled contents may relate to sampled video?</h2>
      <Notice>
        A candidate line is not a confirmed link. CLIP and DINOv2 scores are
        independent; supported group-level review does not confirm an exact
        frame or sequence.
      </Notice>
      {!artifact ? (
        <EmptyArtifactState stage="Linkage" />
      ) : (
        <Panel
          title="Cross-dataset candidates"
          meta="ARTIFACT · no automatic confirmation"
        >
          <div className="linkage-list">
            {artifact.candidates.slice(page * 12, page * 12 + 12).map((c) => (
              <button
                className="linkage-pair"
                key={c.id}
                aria-pressed={selected === c.id}
                onClick={() => setSelected(c.id)}
              >
                <span>
                  <span className="eyebrow">LABELED DATA</span>
                  <strong>{c.labeledContentId}</strong>
                  <small>Image unavailable</small>
                </span>
                <span className="linkage-connection">
                  <ArtifactStatus state={c.state} />
                  <span>
                    CLIP {c.clipCosine?.toFixed(3) ?? 'unavailable'} · DINOv2{' '}
                    {c.dinov2Cosine?.toFixed(3) ?? 'unavailable'}
                  </span>
                  <span aria-hidden="true">┄┄┄ ↔ ┄┄┄</span>
                </span>
                <span>
                  <span className="eyebrow">VIDEO DATA</span>
                  <strong>{c.videoContentId}</strong>
                  <small>{c.sequenceIds.length} sequence memberships</small>
                </span>
              </button>
            ))}
          </div>
          <Pager
            page={page}
            size={12}
            count={artifact.candidates.length}
            setPage={setPage}
          />
          {candidate && (
            <div className="panel-body">
              <h3>Selected candidate · {candidate.state}</h3>
              <p>
                Sequence memberships:{' '}
                {candidate.sequenceIds.join(', ') || 'unavailable'}. Images
                unavailable in this export.
              </p>
              <TechnicalDetailsDrawer>
                <pre className="audit-json">
                  {JSON.stringify(
                    {
                      candidate,
                      review:
                        research?.linkageReview.artifact?.find(
                          (r) => r.id === candidate.reviewId,
                        ) ?? null,
                    },
                    null,
                    2,
                  )}
                </pre>
              </TechnicalDetailsDrawer>
            </div>
          )}
        </Panel>
      )}
      {groupReview && (
        <Panel
          title="Group-level manual review"
          meta="Supported does not mean confirmed link"
        >
          <MetricSummary
            items={['supported', 'unsupported', 'ambiguous', 'pending'].map(
              (decision) => ({
                label: decision,
                value: groupReview.items.filter(
                  (item) => item.decision === decision,
                ).length,
              }),
            )}
          />
          <TechnicalDetailsDrawer title="View review details">
            <p>
              Review unit: labeled content ↔ proposed visual dependency group.
              No exact frame match is confirmed.
            </p>
            <pre className="audit-json">
              {JSON.stringify(groupReview, null, 2)}
            </pre>
          </TechnicalDetailsDrawer>
        </Panel>
      )}
      {aggregate ? (
        <Panel title="Review aggregation" meta="Descriptive review counts">
          <MetricSummary
            items={[
              { label: 'Reviews', value: aggregate.reviews },
              { label: 'Reviewed items', value: aggregate.reviewedItems },
              { label: 'Supported', value: aggregate.supported },
              { label: 'Ambiguous', value: aggregate.ambiguous },
              { label: 'Conflicts', value: aggregate.conflicts },
            ]}
          />
          <TechnicalDetailsDrawer>
            <pre className="audit-json">
              {JSON.stringify(aggregate, null, 2)}
            </pre>
            <p>
              Supported ≠ confirmed. Descriptive rates are not representative
              accuracy.
            </p>
          </TechnicalDetailsDrawer>
        </Panel>
      ) : (
        <p className="panel-body muted">Review aggregation: pending.</p>
      )}
    </>
  )
}
