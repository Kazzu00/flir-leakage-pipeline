import { lazy, Suspense, useMemo, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import {
  useOrganizationBoundaryZones,
  useOrganizationMedia,
  useOrganizationTimelines,
} from '@/data/organization-provider'
import type {
  OrganizationBoundaryZone,
  OrganizationTimeline,
} from '@/contracts/organization-evidence-v2'
import { Panel, Notice, Select } from '@/components/feedback/Primitives'
import {
  EmptyArtifactState,
  MetricSummary,
} from '@/components/research/Evidence'
import { TechnicalDetailsDrawer } from '@/components/research/TechnicalDetailsDrawer'
import { OrganizationLinkageExplorer } from './OrganizationLinkageExplorer'
import { TimelinePlayer } from './TimelinePlayer'
import { Pager } from './evidence-views/Pager'
import { AuditReview } from './AuditReview'
const ScientificTimeline = lazy(() =>
  import('./ScientificTimeline').then((module) => ({
    default: module.ScientificTimeline,
  })),
)

export function SequenceExplorer() {
  const [params, setParams] = useSearchParams()
  const linkage = params.get('mode') === 'linkage'
  const audit = params.get('mode') === 'audit'
  const temporal = params.get('mode') === 'sequences'

  return (
    <>
      <div className="toolbar">
        <div className="split-toggle" aria-label="Sequence and linkage views">
          <button
            aria-pressed={!linkage && !audit && !temporal}
            onClick={() => setParams({ mode: 'timeline' })}
          >
            Línea temporal científica
          </button>
          <button
            aria-pressed={temporal}
            onClick={() => setParams({ mode: 'sequences' })}
          >
            Temporal evidence
          </button>
          <button
            aria-pressed={linkage}
            onClick={() => setParams({ mode: 'linkage' })}
          >
            Cross-dataset linkage
          </button>
          <button
            aria-pressed={audit}
            onClick={() => setParams({ mode: 'audit' })}
          >
            Human audit review
          </button>
        </div>
      </div>

      {audit ? (
        <AuditReview />
      ) : linkage ? (
        <OrganizationLinkageExplorer />
      ) : temporal ? (
        <TemporalEvidence />
      ) : (
        <Suspense
          fallback={<p role="status">Cargando línea temporal científica…</p>}
        >
          <ScientificTimeline />
        </Suspense>
      )}
    </>
  )
}

function TemporalEvidence() {
  const timelinesQuery = useOrganizationTimelines()
  const zonesQuery = useOrganizationBoundaryZones()
  const mediaQuery = useOrganizationMedia()

  const timelines = timelinesQuery.data ?? []
  const allZones = zonesQuery.data ?? []

  const [timelineId, setTimelineId] = useState('')
  const [selectedZoneId, setSelectedZoneId] = useState('')
  const [page, setPage] = useState(0)

  const active =
    timelines.find((timeline) => timeline.timeline_id === timelineId) ??
    timelines[0]

  const zones = useMemo(
    () =>
      active
        ? allZones.filter((zone) =>
            zone.timeline_ids.includes(active.timeline_id),
          )
        : [],
    [active, allZones],
  )

  const selectedZone =
    zones.find((zone) => zone.element_id === selectedZoneId) ?? zones[0]

  const frameIndexes = useMemo(
    () =>
      active?.points
        .map((point) => point.frame_index)
        .filter((value): value is number => value !== null) ?? [],
    [active],
  )

  const frameMin = frameIndexes.length ? Math.min(...frameIndexes) : null
  const frameMax = frameIndexes.length ? Math.max(...frameIndexes) : null

  if (timelinesQuery.isPending || zonesQuery.isPending) {
    return <p role="status">Loading temporal evidence…</p>
  }

  if (timelinesQuery.isError || zonesQuery.isError) {
    return (
      <Notice>
        Verified temporal evidence could not be loaded from the organization
        bundle.
      </Notice>
    )
  }

  if (!timelines.length) {
    return <EmptyArtifactState stage="Temporal evidence" />
  }

  return (
    <>
      <h2>Temporal structure and boundary evidence</h2>

      <Notice>
        Timelines are presentation namespaces derived from stored evidence.
        Filename families are not authoritative source-video identities.
        Boundary zones are inclusive uncertainty intervals, not exact cuts.
      </Notice>

      <div className="toolbar">
        <Select
          label="Timeline"
          value={active?.timeline_id ?? ''}
          onChange={(value) => {
            setTimelineId(value)
            setSelectedZoneId('')
            setPage(0)
          }}
          options={timelines.map((timeline) => ({
            value: timeline.timeline_id,
            label: timelineLabel(timeline),
          }))}
        />
      </div>

      {active && (
        <>
          <Panel
            title="Temporal evidence timeline"
            meta={active.temporal_source.replaceAll('_', ' ')}
          >
            <MetricSummary
              items={[
                {
                  label: 'Observed points',
                  value: active.points.length,
                },
                {
                  label: 'Frame range',
                  value:
                    frameMin === null || frameMax === null
                      ? 'Unavailable'
                      : `${frameMin}–${frameMax}`,
                },
                {
                  label: 'Boundary zones',
                  value: zones.length,
                },
                {
                  label: 'Source',
                  value:
                    active.source_video_id ??
                    active.inferred_family ??
                    active.source_archive ??
                    'Unavailable',
                },
              ]}
            />

            {mediaQuery.isPending ? (
              <p className="panel-body" role="status">
                Loading frame previews…
              </p>
            ) : mediaQuery.isError ? (
              <Notice>
                Frame preview metadata could not be loaded. Temporal and
                boundary evidence remain available.
              </Notice>
            ) : (
              <TimelinePlayer timeline={active} media={mediaQuery.data ?? []} />
            )}

            <div
              className="sequence-grid"
              aria-label="Temporal boundary evidence"
            >
              <div className="grid-axis">
                <span>{frameMin ?? '—'}</span>
                <span>{frameMax ?? '—'}</span>
              </div>

              {zones.slice(page * 20, page * 20 + 20).map((zone) => (
                <BoundaryLane
                  key={zone.element_id}
                  zone={zone}
                  frameMin={frameMin}
                  frameMax={frameMax}
                  selected={selectedZone?.element_id === zone.element_id}
                  onSelect={() => setSelectedZoneId(zone.element_id)}
                />
              ))}

              {!zones.length && (
                <p className="panel-body muted">
                  No boundary zones are exported for this timeline. This does
                  not establish that no temporal boundaries exist.
                </p>
              )}

              <Pager
                page={page}
                size={20}
                count={zones.length}
                setPage={setPage}
              />
            </div>
          </Panel>

          {selectedZone && (
            <Panel title="Selected boundary evidence">
              <MetricSummary
                items={[
                  {
                    label: 'Evidence ID',
                    value: selectedZone.element_id,
                  },
                  {
                    label: 'Decision',
                    value: selectedZone.decision,
                  },
                  {
                    label: 'Inclusive interval',
                    value: `${selectedZone.start}–${selectedZone.end}`,
                  },
                  {
                    label: 'Exact cut',
                    value: selectedZone.exact_cut ? 'Yes' : 'No',
                  },
                ]}
              />

              <TechnicalDetailsDrawer>
                <pre className="audit-json">
                  {JSON.stringify(
                    {
                      timeline: {
                        timeline_id: active.timeline_id,
                        source_video_id: active.source_video_id,
                        source_archive: active.source_archive,
                        inferred_family: active.inferred_family,
                        temporal_source: active.temporal_source,
                        point_count: active.points.length,
                      },
                      boundary_zone: selectedZone,
                    },
                    null,
                    2,
                  )}
                </pre>
              </TechnicalDetailsDrawer>
            </Panel>
          )}

          <TechnicalDetailsDrawer title="Evidence and provenance">
            <p>
              Source video ≠ timeline. Timeline ≠ sequence. Boundary zone ≠
              exact cut. Stored evidence is presented without automatic
              confirmation.
            </p>
            <pre className="audit-json">
              {JSON.stringify(
                {
                  timeline_id: active.timeline_id,
                  temporal_source: active.temporal_source,
                  source_video_id: active.source_video_id,
                  source_archive: active.source_archive,
                  inferred_family: active.inferred_family,
                  observed_points: active.points.length,
                  exported_boundary_zones: zones.length,
                },
                null,
                2,
              )}
            </pre>
          </TechnicalDetailsDrawer>
        </>
      )}
    </>
  )
}

function BoundaryLane({
  zone,
  frameMin,
  frameMax,
  selected,
  onSelect,
}: {
  zone: OrganizationBoundaryZone
  frameMin: number | null
  frameMax: number | null
  selected: boolean
  onSelect: () => void
}) {
  const span =
    frameMin === null || frameMax === null
      ? 1
      : Math.max(1, frameMax - frameMin + 1)

  const offset =
    frameMin === null ? 0 : Math.max(0, (zone.start - frameMin) / span)

  const width = Math.min(
    1 - offset,
    Math.max(0.01, (zone.end - zone.start + 1) / span),
  )

  return (
    <div className="sequence-lane">
      <button
        className={`sequence-interval decision-${zone.decision}`}
        style={{
          marginLeft: `${offset * 85}%`,
          width: `${Math.max(4, width * 85)}%`,
        }}
        aria-label={`boundary zone ${zone.element_id}: ${zone.start}–${zone.end}, ${zone.decision}`}
        aria-pressed={selected}
        onClick={onSelect}
      >
        <span aria-hidden="true">
          {zone.decision === 'candidate'
            ? '?'
            : zone.decision === 'unsupported'
              ? '×'
              : '◇'}
        </span>
      </button>

      <span>
        {zone.element_id} · {zone.decision} · {zone.start}–{zone.end}
      </span>
    </div>
  )
}

function timelineLabel(timeline: OrganizationTimeline) {
  const identity =
    timeline.source_video_id ??
    timeline.inferred_family ??
    timeline.source_archive ??
    timeline.timeline_id

  return `${identity} · ${timeline.temporal_source.replaceAll('_', ' ')}`
}
