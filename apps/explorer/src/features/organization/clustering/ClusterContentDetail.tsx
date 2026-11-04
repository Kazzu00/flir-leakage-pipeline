import { useState } from 'react'
import type {
  OrganizationContent,
  OrganizationClusterMembership,
  OrganizationMedia,
  OrganizationRecord,
} from '@/contracts/organization-evidence-v2'
import { Panel, Notice } from '@/components/feedback/Primitives'
import { ContentPreview } from './ClusterVisualGrid'

type Fact = readonly [string, string | number | null | undefined]

export function EvidenceFacts({ items }: { items: readonly Fact[] }) {
  return (
    <dl className="cluster-evidence-facts">
      {items
        .filter(
          ([, value]) => value !== null && value !== undefined && value !== '',
        )
        .map(([label, value]) => (
          <div key={label}>
            <dt>{label}</dt>
            <dd>{value}</dd>
          </div>
        ))}
    </dl>
  )
}

export function ClusterContentDetail({
  content,
  membership,
  media,
  records,
  recordsState,
}: {
  content: OrganizationContent
  membership: OrganizationClusterMembership
  media?: OrganizationMedia
  records: OrganizationRecord[]
  recordsState: 'pending' | 'error' | 'success'
}) {
  const [recordLimit, setRecordLimit] = useState(20)
  const recordIds = new Set(content.record_ids)
  const consistentRecords =
    records.length === recordIds.size &&
    records.every((record) => recordIds.has(record.record_id)) &&
    new Set(records.map((record) => record.record_id)).size === records.length

  return (
    <Panel
      title="Selected content"
      meta={
        membership.is_noise ? 'Noise −1' : `Cluster ${membership.cluster_id}`
      }
    >
      <div className="cluster-selected-preview">
        <ContentPreview content={content} media={media} />
      </div>
      <div className="cluster-selected-summary">
        <EvidenceFacts
          items={[
            ['Canonical filename', content.canonical_filename],
            ['Frame index', content.frame_index],
            ['Source video', content.source_video_id],
            ['Cluster ID', membership.cluster_id],
            ['Is noise', membership.is_noise ? 'Yes' : 'No'],
            ['Record count', content.record_ids.length],
            [
              'Timelines (occurrences)',
              recordsState === 'success' && consistentRecords
                ? [
                    ...new Set(
                      records.flatMap((record) =>
                        record.timeline_id ? [record.timeline_id] : [],
                      ),
                    ),
                  ].join(', ')
                : null,
            ],
          ]}
        />
      </div>
      <EvidenceFacts
        items={[
          ['Content ID', content.content_id],
          ['Source frame index', content.source_frame_index],
          ['Source frame index estimate', content.source_frame_index_estimate],
          ['Timestamp (seconds)', content.timestamp_seconds],
          ['Temporal source', content.temporal_source.replaceAll('_', ' ')],
          ['Class names', content.class_names?.join(', ')],
          ['Class IDs', content.class_ids?.join(', ')],
          [
            'Annotation consensus',
            content.annotation_consensus.replaceAll('_', ' '),
          ],
          ['Probability', membership.probability],
          [
            'Core distance',
            membership.core_distance_infinite
              ? '∞ (reported)'
              : membership.core_distance,
          ],
          [
            'Reachability',
            membership.reachability_infinite
              ? '∞ (reported)'
              : membership.reachability,
          ],
          ['Ordering position', membership.ordering_position],
        ]}
      />
      <p className="panel-body muted">
        Temporal indices and timestamps are reported evidence. Sampling-grid
        time and filename heuristics do not verify capture time.
      </p>
      <section className="cluster-occurrences" aria-label="Content occurrences">
        <h3>Occurrences</h3>
        <p>
          Records of the same content are occurrences, not independent images.
          Timeline identity is not sequence identity.
        </p>
        {recordsState === 'pending' ? (
          <p role="status">Loading content occurrences…</p>
        ) : recordsState === 'error' ? (
          <Notice>
            Occurrence records could not be loaded or validated. Selected
            content evidence remains available.
          </Notice>
        ) : (
          <>
            {!consistentRecords && (
              <Notice>
                The available occurrence records do not match the content's
                exported record IDs. No occurrences are inferred.
              </Notice>
            )}
            {!records.length && (
              <p>No occurrence records are available for this content.</p>
            )}
            <ol className="cluster-occurrence-list">
              {records.slice(0, recordLimit).map((record, index) => (
                <li key={`${record.record_id}-${index}`}>
                  <EvidenceFacts
                    items={[
                      ['Record ID', record.record_id],
                      ['Cohort', record.cohort],
                      ['Filename', record.filename],
                      ['Source archive', record.source_archive],
                      ['Source video', record.source_video_id],
                      ['Timeline', record.timeline_id],
                      ['Frame index', record.frame_index],
                    ]}
                  />
                </li>
              ))}
            </ol>
            {records.length > recordLimit && (
              <button
                className="cluster-records-more"
                type="button"
                onClick={() => setRecordLimit((limit) => limit + 20)}
              >
                Load more occurrences
              </button>
            )}
          </>
        )}
      </section>
    </Panel>
  )
}
