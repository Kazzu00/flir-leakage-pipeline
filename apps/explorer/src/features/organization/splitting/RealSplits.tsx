import { useMemo, useState } from 'react'
import {
  candidateRuns,
  useSplitEvidence,
  type SplitRun,
  type SplitMember,
  type SplitEdge,
} from '@/data/split-evidence'
import { organizationMediaUrl } from '@/data/organization-media'
import { Notice, Panel, Select } from '@/components/feedback/Primitives'
import { TechnicalDetailsDrawer } from '@/components/research/TechnicalDetailsDrawer'
import { Pager } from '../evidence-views/Pager'

const options = (values: string[]) =>
  values.map((value) => ({ value, label: value }))
const runLabel = (run: SplitRun) =>
  `${run.policy} / ${run.population} / seed ${run.seed}`

export function RealSplits() {
  const { historical, candidates } = useSplitEvidence()
  const [family, setFamily] = useState('historical_yolo11n')
  const [policy, setPolicy] = useState('all')
  const [population, setPopulation] = useState('all')
  const [seed, setSeed] = useState('all')
  const [leftId, setLeftId] = useState('')
  const [rightId, setRightId] = useState('')
  const [compare, setCompare] = useState(false)
  const runs = useMemo(() => {
    if (family === 'historical_yolo11n') return historical.data?.runs ?? []
    return candidates.data
      ? candidateRuns(candidates.data, historical.data?.previews ?? new Map())
      : []
  }, [family, historical.data, candidates.data])
  const edges: SplitEdge[] = useMemo(
    () =>
      family === 'common_test_candidates'
        ? (candidates.data?.relations ?? []).map((r) => ({
            id: r.relation_id,
            left: r.left_frame_id,
            right: r.right_frame_id,
            state: r.state,
          }))
        : [],
    [family, candidates.data],
  )
  const filtered = runs.filter(
    (r) =>
      (policy === 'all' || r.policy === policy) &&
      (population === 'all' || r.population === population) &&
      (seed === 'all' || String(r.seed) === seed),
  )
  const left = filtered.find((r) => r.id === leftId) ?? filtered[0]
  const right =
    runs.find((r) => r.id === rightId) ??
    runs.find((r) => r.id !== left?.id) ??
    left
  const query = family === 'historical_yolo11n' ? historical : candidates
  return (
    <section aria-label="Real split explorer">
      <h2>Visual inspection of real split memberships</h2>
      <Notice>
        Historical YOLO11n: 16 exported splits, with results in the detector
        contract. Common-test S1/S2/S3: provisional memberships, test PROPOSED,
        training blocked. Counts below describe stored memberships, not
        detection performance. Quarantine is not an independent test.
      </Notice>
      <div className="toolbar">
        <Select
          label="Experimental family"
          value={family}
          onChange={(v) => {
            setFamily(v)
            setPolicy('all')
            setPopulation('all')
            setSeed('all')
            setLeftId('')
            setRightId('')
          }}
          options={[
            { value: 'historical_yolo11n', label: 'Historical YOLO11n' },
            {
              value: 'common_test_candidates',
              label: 'Common-test candidates S1/S2/S3',
            },
          ]}
        />
        <Select
          label="Policy"
          value={policy}
          onChange={setPolicy}
          options={options(['all', ...new Set(runs.map((r) => r.policy))])}
        />
        <Select
          label="Population"
          value={population}
          onChange={setPopulation}
          options={options(['all', ...new Set(runs.map((r) => r.population))])}
        />
        <Select
          label="Split seed"
          value={seed}
          onChange={setSeed}
          options={options([
            'all',
            ...new Set(runs.map((r) => String(r.seed))),
          ])}
        />
        <label>
          <input
            type="checkbox"
            checked={compare}
            onChange={(e) => setCompare(e.target.checked)}
          />{' '}
          Compare side by side
        </label>
      </div>
      {query.isPending ? (
        <p role="status">Loading real split evidence…</p>
      ) : query.isError ? (
        <Notice>
          UNAVAILABLE: this split family could not be loaded or validated. No
          synthetic memberships substituted.
        </Notice>
      ) : (
        <>
          <p>
            {runs.length} available configurations; {filtered.length} match
            selection. Family: {family}. Different test protocols must not be
            ranked together.
          </p>
          {family === 'common_test_candidates' && historical.isError && (
            <Notice>
              Historical media index unavailable; candidate memberships remain
              readable without previews.
            </Notice>
          )}
          <div className={compare ? 'split-comparison-grid' : ''}>
            <div>
              <Select
                label="Left split"
                value={left?.id ?? ''}
                onChange={setLeftId}
                options={filtered.map((r) => ({
                  value: r.id,
                  label: runLabel(r),
                }))}
              />
              {left ? (
                <SplitPanel
                  key={left.id}
                  run={left}
                  edges={edges}
                  links={
                    family === 'historical_yolo11n'
                      ? (historical.data?.links ?? [])
                      : []
                  }
                  side="Left"
                />
              ) : (
                <Notice>No memberships match these filters.</Notice>
              )}
            </div>
            {compare && (
              <div>
                <Select
                  label="Right split"
                  value={right?.id ?? ''}
                  onChange={setRightId}
                  options={runs.map((r) => ({
                    value: r.id,
                    label: runLabel(r),
                  }))}
                />
                {right && (
                  <SplitPanel
                    key={right.id}
                    run={right}
                    edges={edges}
                    links={
                      family === 'historical_yolo11n'
                        ? (historical.data?.links ?? [])
                        : []
                    }
                    side="Right"
                  />
                )}
              </div>
            )}
          </div>
        </>
      )}
    </section>
  )
}

type Link = {
  linkage_group_id: string
  evidence_artifact_id: string
  record_ids: string[]
}
function SplitPanel({
  run,
  edges,
  links,
  side,
}: {
  run: SplitRun
  edges: SplitEdge[]
  links: Link[]
  side: string
}) {
  const [partition, setPartition] = useState('all')
  const [classId, setClassId] = useState('all')
  const [source, setSource] = useState('all')
  const [group, setGroup] = useState('')
  const [selectedId, setSelectedId] = useState('')
  const [page, setPage] = useState(0)
  const byId = useMemo(
    () => new Map(run.members.map((m) => [m.recordId, m])),
    [run],
  )
  const partitions = [...new Set(run.members.map((m) => m.partition))]
  const filtered = run.members.filter(
    (m) =>
      (partition === 'all' || m.partition === partition) &&
      (classId === 'all' ||
        (classId === 'unknown'
          ? m.classes === null
          : m.classes?.includes(Number(classId)))) &&
      (source === 'all' || m.source === source) &&
      (!group || m.group?.includes(group)),
  )
  const selected = byId.get(selectedId)
  const relatedEdges = selected
    ? edges.filter(
        (e) => e.left === selected.recordId || e.right === selected.recordId,
      )
    : []
  const directLinks = selected
    ? links.filter((l) => l.record_ids.includes(selected.recordId))
    : []
  const selectedGroupIds = new Set(
    directLinks.map((l) => `${l.evidence_artifact_id}:${l.linkage_group_id}`),
  )
  const selectedLinks = links.filter((l) =>
    selectedGroupIds.has(`${l.evidence_artifact_id}:${l.linkage_group_id}`),
  )
  const relatedIds = new Set(selectedLinks.flatMap((l) => l.record_ids))
  for (const edge of relatedEdges) {
    relatedIds.add(edge.left)
    relatedIds.add(edge.right)
  }
  const related = selected
    ? run.members.filter(
        (m) =>
          m.recordId !== selected.recordId &&
          ((selected.group !== null && m.group === selected.group) ||
            relatedIds.has(m.recordId) ||
            m.contentId === selected.contentId),
      )
    : []
  const crossing = edges.filter(
    (e) =>
      e.state !== 'rejected' &&
      byId.has(e.left) &&
      byId.has(e.right) &&
      byId.get(e.left)!.partition !== byId.get(e.right)!.partition,
  )
  const groups = new Map<string, Set<string>>()
  for (const m of run.members)
    if (m.group) {
      const parts = groups.get(m.group) ?? new Set()
      parts.add(m.partition)
      groups.set(m.group, parts)
    }
  const crossingGroups = [...groups].filter(([, parts]) => parts.size > 1)
  const diagnosticGroups = new Map<string, Set<string>>()
  for (const link of links) {
    const key = `${link.evidence_artifact_id}:${link.linkage_group_id}`
    const parts = diagnosticGroups.get(key) ?? new Set<string>()
    for (const id of link.record_ids) {
      const m = byId.get(id)
      if (m) parts.add(m.partition)
    }
    diagnosticGroups.set(key, parts)
  }
  const fragmentedDiagnostics = [...diagnosticGroups].filter(
    ([, parts]) => parts.size > 1,
  )
  return (
    <section aria-label={`${side} split details`}>
      <Panel title={runLabel(run)} meta={run.family}>
        <p>Test status: {run.testStatus}</p>
        <div className="table-scroll">
          <table>
            <caption>
              Stored membership counts (occurrences and exact contents)
            </caption>
            <thead>
              <tr>
                <th>Partition</th>
                <th>Occurrences</th>
                <th>Contents</th>
              </tr>
            </thead>
            <tbody>
              {partitions.map((p) => {
                const rows = run.members.filter((m) => m.partition === p)
                return (
                  <tr key={p}>
                    <td>{p === 'val' ? 'validation' : p}</td>
                    <td>{rows.length}</td>
                    <td>{new Set(rows.map((m) => m.contentId)).size}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
        <details>
          <summary>Reported class support by partition</summary>
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Partition</th>
                  <th>Class ID</th>
                  <th>Images</th>
                  <th>Contents</th>
                  <th>Instances</th>
                </tr>
              </thead>
              <tbody>
                {run.support.map((s) => (
                  <tr key={`${s.partition}:${s.class_id}`}>
                    <td>{s.partition}</td>
                    <td>{s.class_id}</td>
                    <td>{s.images}</td>
                    <td>{s.contents}</td>
                    <td>{s.instances}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p>
            Class ID 4 nomenclature remains disputed (SDZI / Heavy Machinery);
            no annotation adjudication is implied.
          </p>
        </details>
        <details>
          <summary>Cross-partition stored relations and groups</summary>
          <p>
            {crossingGroups.length} allocation/content groups span partitions;{' '}
            {fragmentedDiagnostics.length} stored candidate diagnostic groups
            span partitions; {crossing.length} non-rejected relation records
            span partitions. These are presentation joins, not independent pairs
            or confirmed leakage. Test/quarantine crossings are shown, not
            classified as development exposure.
          </p>
          <ul>
            {crossingGroups.map(([id, parts]) => (
              <li key={id}>
                <button
                  onClick={() => {
                    setGroup(id)
                    setPage(0)
                  }}
                >
                  {id}
                </button>
                : {[...parts].join(', ')}
              </li>
            ))}
          </ul>
          <ul>
            {fragmentedDiagnostics.map(([id, parts]) => (
              <li key={id}>
                {id}: {[...parts].join(', ')}
                <button
                  onClick={() => {
                    const link = links.find(
                      (l) =>
                        `${l.evidence_artifact_id}:${l.linkage_group_id}` ===
                        id,
                    )
                    const record = link?.record_ids.find((r) => byId.has(r))
                    setSelectedId(record ?? '')
                  }}
                >
                  Inspect candidate group
                </button>
              </li>
            ))}
          </ul>
          <ul>
            {crossing.map((e) => (
              <li key={e.id}>
                <button onClick={() => setSelectedId(e.left)}>
                  Inspect {e.state} relation {e.id.slice(0, 12)}
                </button>
                : {byId.get(e.left)?.partition} / {byId.get(e.right)?.partition}
              </li>
            ))}
          </ul>
          {!edges.length && (
            <p>
              Occurrence-level relation records are unavailable for this family.
              Candidate labeled-to-video edges do not assign video contents to
              historical splits.
            </p>
          )}
        </details>
        <TechnicalDetailsDrawer title={`${side} split provenance`}>
          <pre className="audit-json">
            {JSON.stringify(run.metadata, null, 2)}
          </pre>
        </TechnicalDetailsDrawer>
      </Panel>
      <div className="toolbar">
        <Select
          label={`${side} partition`}
          value={partition}
          onChange={(v) => {
            setPartition(v)
            setPage(0)
          }}
          options={options(['all', ...partitions])}
        />
        <Select
          label={`${side} class`}
          value={classId}
          onChange={(v) => {
            setClassId(v)
            setPage(0)
          }}
          options={options(['all', '0', '1', '2', '3', '4', 'unknown'])}
        />
        <Select
          label={`${side} source`}
          value={source}
          onChange={(v) => {
            setSource(v)
            setPage(0)
          }}
          options={options([
            'all',
            ...new Set(run.members.map((m) => m.source)),
          ])}
        />
        <label>
          {side} group filter
          <input
            aria-label={`${side} group filter`}
            value={group}
            onChange={(e) => {
              setGroup(e.target.value)
              setPage(0)
            }}
          />
        </label>
      </div>
      <p>
        {filtered.length} matching occurrences. Preview joins use exact content
        IDs only.
      </p>
      <div className="split-member-grid" aria-label={`${side} members`}>
        {filtered.slice(page * 12, page * 12 + 12).map((m) => (
          <MemberTile
            key={m.recordId}
            member={m}
            select={() => setSelectedId(m.recordId)}
          />
        ))}
      </div>
      <Pager page={page} size={12} count={filtered.length} setPage={setPage} />
      {selected && (
        <Panel title="Selected split member">
          <pre className="audit-json">
            {JSON.stringify(
              {
                ...selected,
                preview: selected.preview
                  ? 'Available in separate local media root'
                  : 'UNAVAILABLE',
              },
              null,
              2,
            )}
          </pre>
          <p>
            Related by stored group, exact content or candidate relation; this
            does not confirm a temporal sequence. {related.length} related
            occurrences.
          </p>
          <div className="split-member-grid">
            {related.slice(0, 12).map((m) => (
              <MemberTile
                key={m.recordId}
                member={m}
                select={() => setSelectedId(m.recordId)}
              />
            ))}
          </div>
          {related.length > 12 && (
            <p>
              First 12 related occurrences shown; use the group filter for
              complete membership.
            </p>
          )}
          <pre className="audit-json">
            {JSON.stringify(
              { relations: relatedEdges, candidate_groups: selectedLinks },
              null,
              2,
            )}
          </pre>
        </Panel>
      )}
    </section>
  )
}

function MemberTile({
  member,
  select,
}: {
  member: SplitMember
  select: () => void
}) {
  return (
    <button
      className="split-member"
      onClick={select}
      aria-label={`Inspect occurrence ${member.recordId}`}
    >
      {member.preview ? (
        <img
          src={organizationMediaUrl(member.preview)}
          alt={member.filename}
          loading="lazy"
        />
      ) : (
        <span>Preview UNAVAILABLE</span>
      )}
      <span>{member.filename}</span>
      <small>
        {member.partition} · {member.recordId.slice(0, 12)}
      </small>
    </button>
  )
}
