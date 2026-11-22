import { useState } from 'react'
import type {
  OrganizationClusterMembership,
  OrganizationContent,
  OrganizationMedia,
} from '@/contracts/organization-evidence-v2'
import { organizationMediaUrl } from '@/data/organization-media'

const BATCH_SIZE = 60
export type ClusterDensity = 'compact' | 'detailed'

/** One tile per unique content; occurrences belong in the evidence panel. */
export function ClusterVisualGrid({
  memberships,
  contentsById,
  mediaByPreviewKey,
  selectedContentId,
  onSelect,
  density,
}: {
  memberships: OrganizationClusterMembership[]
  contentsById: Map<string, OrganizationContent>
  mediaByPreviewKey: Map<string, OrganizationMedia>
  selectedContentId: string
  onSelect: (contentId: string) => void
  density: ClusterDensity
}) {
  const [limit, setLimit] = useState(BATCH_SIZE)
  if (!memberships.length)
    return (
      <p className="panel-body muted">
        No content memberships are exported for this cluster.
      </p>
    )

  return (
    <>
      <ul
        className="cluster-visual-grid"
        data-density={density}
        aria-label="Cluster contents"
      >
        {memberships.slice(0, limit).map((member) => {
          const content = contentsById.get(member.content_id)!
          return (
            <li key={member.content_id}>
              <button
                type="button"
                className="cluster-content-tile"
                aria-label={`Inspect content ${content.content_id}`}
                aria-description={
                  [
                    content.canonical_filename,
                    content.frame_index !== null
                      ? `Frame ${content.frame_index}`
                      : null,
                  ]
                    .filter(Boolean)
                    .join(', ') || undefined
                }
                title={[
                  content.canonical_filename ?? content.content_id,
                  content.frame_index !== null
                    ? `Frame ${content.frame_index}`
                    : null,
                ]
                  .filter(Boolean)
                  .join(', ')}
                aria-pressed={selectedContentId === content.content_id}
                onClick={() => onSelect(content.content_id)}
              >
                <ContentPreview
                  content={content}
                  media={mediaByPreviewKey.get(content.preview_key)}
                />
                {density === 'detailed' && (
                  <span className="cluster-tile-text">
                    <strong className="mono">
                      {shortContentId(content.content_id)}
                    </strong>
                    {content.canonical_filename && (
                      <span>{content.canonical_filename}</span>
                    )}
                    {content.frame_index !== null && (
                      <span>Frame {content.frame_index}</span>
                    )}
                    {content.source_video_id && (
                      <span>Source video: {content.source_video_id}</span>
                    )}
                  </span>
                )}
              </button>
            </li>
          )
        })}
      </ul>
      <div className="cluster-grid-footer">
        <p role="status">
          Showing {Math.min(limit, memberships.length)} of {memberships.length}{' '}
          contents
        </p>
        {limit < memberships.length && (
          <button
            type="button"
            onClick={() => setLimit((value) => value + BATCH_SIZE)}
          >
            Load more
          </button>
        )}
      </div>
    </>
  )
}

function shortContentId(id: string) {
  return id.length > 20 ? `${id.slice(0, 12)}…${id.slice(-6)}` : id
}

export function ContentPreview({
  content,
  media,
}: {
  content: OrganizationContent
  media?: OrganizationMedia
}) {
  const [failedUrl, setFailedUrl] = useState<string | null>(null)
  let url: string | null = null
  let reason = 'No exported preview is available.'
  if (
    media &&
    (media.content_id !== content.content_id ||
      media.preview_key !== content.preview_key)
  ) {
    reason = 'Preview identity does not match this content.'
  } else if (media?.media_available && media.relative_path) {
    try {
      url = organizationMediaUrl(media.relative_path)
    } catch {
      reason = 'The exported preview path is invalid.'
    }
  } else if (media?.unavailable_reason) {
    reason = media.unavailable_reason.replaceAll('_', ' ')
  }
  return (
    <span className="cluster-preview">
      {url && failedUrl !== url ? (
        <img
          src={url}
          alt={`Preview of ${content.canonical_filename ?? content.content_id}`}
          width={media?.width ?? content.width ?? undefined}
          height={media?.height ?? content.height ?? undefined}
          loading="lazy"
          decoding="async"
          onError={() => setFailedUrl(url)}
        />
      ) : (
        <span className="cluster-preview-missing">
          <strong>Preview unavailable</strong>
          <span>{url ? 'The image could not be loaded.' : reason}</span>
        </span>
      )}
    </span>
  )
}
