import { useMemo, useState } from 'react'
import type { OrganizationCandidatePair } from '@/contracts/organization-evidence-v2'
import {
  useOrganizationCandidatePairs,
  useOrganizationContents,
  useOrganizationMedia,
} from '@/data/organization-provider'
import { Notice, Panel } from '@/components/feedback/Primitives'
import {
  EmptyArtifactState,
  MetricSummary,
} from '@/components/research/Evidence'
import { TechnicalDetailsDrawer } from '@/components/research/TechnicalDetailsDrawer'
import { Pager } from './evidence-views/Pager'
import { organizationMediaUrl } from '@/data/organization-media'

const PAGE_SIZE = 12

export function OrganizationLinkageExplorer() {
  const candidatesQuery = useOrganizationCandidatePairs()
  const candidates = candidatesQuery.data ?? []

  const [page, setPage] = useState(0)
  const [selectedId, setSelectedId] = useState('')

  const selected =
    candidates.find((candidate) => candidate.candidate_id === selectedId) ??
    null

  const contentsQuery = useOrganizationContents(selected !== null)
  const mediaQuery = useOrganizationMedia(selected !== null)

  const selectedPreviews = useMemo(() => {
    if (!selected || !contentsQuery.data || !mediaQuery.data) return null

    const labeledContent = contentsQuery.data.find(
      (content) => content.content_id === selected.labeled_content_id,
    )
    const videoContent = contentsQuery.data.find(
      (content) => content.content_id === selected.video_content_id,
    )

    const mediaByPreview = new Map(
      mediaQuery.data.map((item) => [item.preview_key, item]),
    )

    const resolve = (
      content:
        | (typeof contentsQuery.data)[number]
        | undefined,
    ) => {
      if (!content) return null

      const media = mediaByPreview.get(content.preview_key)

      return {
        contentId: content.content_id,
        filename: content.canonical_filename,
        previewKey: content.preview_key,
        media: media ?? null,
      }
    }

    return {
      labeled: resolve(labeledContent),
      video: resolve(videoContent),
    }
  }, [selected, contentsQuery.data, mediaQuery.data])

  const summary = useMemo(
    () => ({
      total: candidates.length,
      bothTopK: candidates.filter((candidate) => candidate.both_topk).length,
      clipTopK: candidates.filter((candidate) => candidate.clip_topk).length,
      dinov2TopK: candidates.filter((candidate) => candidate.dinov2_topk)
        .length,
    }),
    [candidates],
  )

  if (candidatesQuery.isPending) {
    return <p role="status">Loading cross-dataset candidate evidence…</p>
  }

  if (candidatesQuery.isError) {
    return (
      <Notice>
        Cross-dataset candidate evidence could not be loaded from
        organization-evidence-v2.
      </Notice>
    )
  }

  if (!candidates.length) {
    return <EmptyArtifactState stage="Cross-dataset linkage" />
  }

  const visible = candidates.slice(
    page * PAGE_SIZE,
    page * PAGE_SIZE + PAGE_SIZE,
  )

  return (
    <>
      <h2>Which labeled contents may relate to sampled video?</h2>

      <Notice>
        Each row is a stored candidate edge. CLIP and DINOv2 evidence are
        preserved independently. A candidate is not a confirmed match,
        sequence identity, ground truth, dependency or split constraint.
      </Notice>

      <Panel
        title="Cross-dataset candidate evidence"
        meta="CANDIDATE ONLY · no automatic confirmation"
      >
        <MetricSummary
          items={[
            {
              label: 'Candidate pairs',
              value: summary.total,
            },
            {
              label: 'CLIP top-k',
              value: summary.clipTopK,
            },
            {
              label: 'DINOv2 top-k',
              value: summary.dinov2TopK,
            },
            {
              label: 'Both top-k',
              value: summary.bothTopK,
            },
          ]}
        />

        <div className="linkage-list">
          {visible.map((candidate) => {
            const isSelected =
              selected?.candidate_id === candidate.candidate_id

            return (
              <div className="linkage-item" key={candidate.candidate_id}>
                <CandidateRow
                  candidate={candidate}
                  selected={isSelected}
                  onSelect={() =>
                    setSelectedId(isSelected ? '' : candidate.candidate_id)
                  }
                />

                {isSelected && selected && (
                  <SelectedCandidateDetail
                    candidate={selected}
                    previews={selectedPreviews}
                    contentsPending={contentsQuery.isPending}
                    mediaPending={mediaQuery.isPending}
                    contentsError={contentsQuery.isError}
                    mediaError={mediaQuery.isError}
                  />
                )}
              </div>
            )
          })}
        </div>

        <Pager
          page={page}
          size={PAGE_SIZE}
          count={candidates.length}
          setPage={setPage}
        />
      </Panel>

    </>
  )
}

function SelectedCandidateDetail({
  candidate,
  previews,
  contentsPending,
  mediaPending,
  contentsError,
  mediaError,
}: {
  candidate: OrganizationCandidatePair
  previews: {
    labeled: PreviewEvidence
    video: PreviewEvidence
  } | null
  contentsPending: boolean
  mediaPending: boolean
  contentsError: boolean
  mediaError: boolean
}) {
  return (
    <Panel
      title="Selected candidate"
      meta={`ID · ${candidate.candidate_id}`}
    >
      <MetricSummary
        items={[
          {
            label: 'CLIP cosine',
            value: formatScore(candidate.clip_cosine),
          },
          {
            label: 'DINOv2 cosine',
            value: formatScore(candidate.dinov2_cosine),
          },
          {
            label: 'Mean reciprocal rank',
            value: formatScore(candidate.mean_reciprocal_rank),
          },
          {
            label: 'Evidence agreement',
            value: candidate.both_topk
              ? 'Both top-k'
              : candidate.clip_topk
                ? 'CLIP top-k only'
                : candidate.dinov2_topk
                  ? 'DINOv2 top-k only'
                  : 'Outside both top-k flags',
          },
        ]}
      />

      {contentsPending || mediaPending ? (
        <p className="panel-body" role="status">
          Loading selected previews…
        </p>
      ) : contentsError || mediaError ? (
        <Notice>
          Preview metadata could not be loaded. Candidate scores and IDs remain
          available.
        </Notice>
      ) : previews ? (
        <div
          className="panel-body"
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))',
            gap: '1rem',
          }}
        >
          <PreviewFigure
            label="Labeled content"
            evidence={previews.labeled}
          />
          <PreviewFigure
            label="Video content"
            evidence={previews.video}
          />
        </div>
      ) : null}

      <div className="panel-body">
        <p>
          <strong>Labeled content:</strong>{' '}
          <code>{candidate.labeled_content_id}</code>
        </p>
        <p>
          <strong>Video content:</strong>{' '}
          <code>{candidate.video_content_id}</code>
        </p>
        <p>
          Upstream evidence reports {candidate.video_occurrence_count} video
          occurrence{candidate.video_occurrence_count === 1 ? '' : 's'} and{' '}
          {candidate.video_sequence_count} sequence count
          {candidate.video_sequence_count === 1 ? '' : 's'} for this video
          content. These counts do not identify or confirm a particular
          sequence.
        </p>
      </div>

      <TechnicalDetailsDrawer>
        <pre className="audit-json">
          {JSON.stringify(candidate, null, 2)}
        </pre>
      </TechnicalDetailsDrawer>
    </Panel>
  )
}

function CandidateRow({
  candidate,
  selected,
  onSelect,
}: {
  candidate: OrganizationCandidatePair
  selected: boolean
  onSelect: () => void
}) {
  return (
    <button
      className="linkage-pair"
      aria-pressed={selected}
      onClick={onSelect}
    >
      <span>
        <span className="eyebrow">LABELED CONTENT</span>
        <strong>{shortId(candidate.labeled_content_id)}</strong>
        <small>{candidate.labeled_content_id}</small>
      </span>

      <span className="linkage-connection">
        <strong>candidate</strong>
        <span>
          CLIP {formatScore(candidate.clip_cosine)} · DINOv2{' '}
          {formatScore(candidate.dinov2_cosine)}
        </span>
        <span>
          ranks {formatRank(candidate.clip_rank)} /{' '}
          {formatRank(candidate.dinov2_rank)}
        </span>
        <span aria-hidden="true">┄┄┄ ↔ ┄┄┄</span>
      </span>

      <span>
        <span className="eyebrow">VIDEO CONTENT</span>
        <strong>{shortId(candidate.video_content_id)}</strong>
        <small>
          {candidate.video_occurrence_count} occurrence
          {candidate.video_occurrence_count === 1 ? '' : 's'} · upstream
          sequence count {candidate.video_sequence_count}
        </small>
      </span>
    </button>
  )
}

type PreviewEvidence = {
  contentId: string
  filename: string | null
  previewKey: string
  media: {
    media_available: boolean
    relative_path: string | null
    unavailable_reason: string | null
    width: number | null
    height: number | null
  } | null
} | null

function PreviewFigure({
  label,
  evidence,
}: {
  label: string
  evidence: PreviewEvidence
}) {
  const media = evidence?.media

  return (
    <figure style={{ margin: 0 }}>
      <figcaption>
        <span className="eyebrow">{label.toUpperCase()}</span>
        <br />
        <strong>{evidence?.filename ?? 'Preview unavailable'}</strong>
      </figcaption>

      {media?.media_available && media.relative_path ? (
        <img
          src={organizationMediaUrl(media.relative_path)}
          alt={`${label} candidate preview`}
          width={media.width ?? undefined}
          height={media.height ?? undefined}
          loading="lazy"
          decoding="async"
          style={{
            width: '100%',
            height: 'auto',
            marginTop: '0.5rem',
          }}
        />
      ) : (
        <p className="muted">
          {media?.unavailable_reason ?? 'No exported preview is available.'}
        </p>
      )}

      {evidence && (
        <small>
          content {shortId(evidence.contentId)}
        </small>
      )}
    </figure>
  )
}

function formatScore(value: number | null) {
  return value === null ? 'unavailable' : value.toFixed(4)
}

function formatRank(value: number | null) {
  return value === null ? '—' : String(value)
}

function shortId(value: string) {
  return value.length <= 16 ? value : `${value.slice(0, 12)}…`
}


