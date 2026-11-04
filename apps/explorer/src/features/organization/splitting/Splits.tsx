import { useMemo, useState } from 'react'
import {
  Notice,
  Panel,
  Select,
} from '@/components/feedback/Primitives'
import {
  useOrganizationSplits,
  useOrganizationSplitMemberships,
} from '@/data/organization-provider'

const ASSIGNMENT_PREVIEW_SIZE = 24

function formatStrategy(value: string) {
  return value.replaceAll('_', ' ')
}

function formatNumber(value: number) {
  return new Intl.NumberFormat('en-US').format(value)
}

export function Splits() {
  const [selectedSplitSpaceId, setSelectedSplitSpaceId] = useState('')
  const [after, setAfter] = useState(true)

  const splitsQuery = useOrganizationSplits()

  const splits = splitsQuery.data ?? []

  const selected =
    splits.find((split) => split.split_space_id === selectedSplitSpaceId) ??
    splits[0]

  const membershipsQuery = useOrganizationSplitMemberships(
    after && Boolean(selected),
  )

  const selectedMemberships = useMemo(() => {
    if (!selected || !membershipsQuery.data) return []

    return membershipsQuery.data.filter(
      (membership) =>
        membership.split_space_id === selected.split_space_id,
    )
  }, [membershipsQuery.data, selected])

  if (splitsQuery.isPending) {
    return (
      <Panel title="Grouping / split" meta="ORGANIZATION EVIDENCE V2">
        <p className="panel-body" role="status">
          Loading split artifacts…
        </p>
      </Panel>
    )
  }

  if (splitsQuery.isError) {
    return (
      <Notice>
        The verified split artifact could not be loaded from
        organization-evidence-v2.
      </Notice>
    )
  }

  if (!selected) {
    return (
      <Notice>
        No split artifacts are available in this organization evidence export.
      </Notice>
    )
  }

  const classSupport = selected.class_support ?? []

  const partitionEntries = Object.entries(selected.partitions)

  const totalRecords = partitionEntries.reduce(
    (sum, [, partition]) => sum + partition.n_records,
    0,
  )

  const totalContents = partitionEntries.reduce(
    (sum, [, partition]) => sum + partition.n_unique_contents,
    0,
  )

  return (
    <>
      <div className="toolbar">
        <Select
          label="Partition strategy / seed"
          value={selected.split_space_id}
          onChange={setSelectedSplitSpaceId}
          options={splits.map((split) => ({
            value: split.split_space_id,
            label: `${split.strategy} · seed ${split.split_seed} · ${formatStrategy(
              split.source_strategy,
            )}`,
          }))}
        />

        <div className="split-toggle">
          <button
            aria-pressed={!after}
            onClick={() => setAfter(false)}
          >
            BEFORE SPLIT
          </button>

          <button
            aria-pressed={after}
            onClick={() => setAfter(true)}
          >
            AFTER SPLIT
          </button>
        </div>
      </div>

      <Notice>
        This view reads stored split evidence from
        organization-evidence-v2. Counts, class support and memberships are
        presented from the exported artifacts; the frontend does not recompute
        a partition or rank strategies.
      </Notice>

      <Panel
        title={`${selected.strategy} · seed ${selected.split_seed}`}
        meta="REAL / VERIFIED EXPORT"
      >
        <div className="panel-body">
          <table>
            <tbody>
              <tr>
                <th>Split space</th>
                <td className="mono">{selected.split_space_id}</td>
              </tr>
              <tr>
                <th>Artifact</th>
                <td className="mono">{selected.artifact_id}</td>
              </tr>
              <tr>
                <th>Source strategy</th>
                <td>{formatStrategy(selected.source_strategy)}</td>
              </tr>
              <tr>
                <th>Cluster run</th>
                <td className="mono">
                  {selected.cluster_run_id ?? '—'}
                </td>
              </tr>
              <tr>
                <th>Total records</th>
                <td>{formatNumber(totalRecords)}</td>
              </tr>
              <tr>
                <th>Summed unique-content counts</th>
                <td>{formatNumber(totalContents)}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </Panel>

      {after ? (
        <>
          <Panel
            title={`${selected.strategy} · split distribution`}
            meta={`seed ${selected.split_seed}`}
          >
            <div className="panel-body">
              <div className="split-bar">
                {partitionEntries.map(([partitionName, partition]) => (
                  <span
                    key={partitionName}
                    style={{
                      width:
                        totalRecords > 0
                          ? `${(partition.n_records / totalRecords) * 100}%`
                          : '0%',
                    }}
                  >
                    {partitionName} · {formatNumber(partition.n_records)}
                  </span>
                ))}
              </div>

              <table>
                <thead>
                  <tr>
                    <th>Partition</th>
                    <th>Records</th>
                    <th>Unique contents</th>
                    <th>Records</th>
                  </tr>
                </thead>
                <tbody>
                  {partitionEntries.map(([partitionName, partition]) => (
                    <tr key={partitionName}>
                      <td>{partitionName}</td>
                      <td>{formatNumber(partition.n_records)}</td>
                      <td>{formatNumber(partition.n_unique_contents)}</td>
                      <td>
                        {totalRecords > 0
                          ? `${(
                              (partition.n_records / totalRecords) *
                              100
                            ).toFixed(1)}%`
                          : '—'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Panel>

          <Panel
            title="Class support"
            meta={`${classSupport.length} exported rows`}
          >
            <div className="table-scroll">
              <table>
                <thead>
                  <tr>
                    <th>Partition</th>
                    <th>Class</th>
                    <th>Images</th>
                    <th>Contents</th>
                    <th>Instances</th>
                    <th>Instance %</th>
                    <th>Global %</th>
                    <th>|Δ| pp</th>
                  </tr>
                </thead>
                <tbody>
                  {classSupport.map((row) => (
                    <tr
                      key={`${row.split}-${row.class_id}`}
                    >
                      <td>{row.split}</td>
                      <td>
                        {row.class_name}{' '}
                        <span className="mono">
                          ({row.class_id})
                        </span>
                      </td>
                      <td>{formatNumber(row.images)}</td>
                      <td>{formatNumber(row.contents)}</td>
                      <td>{formatNumber(row.instances)}</td>
                      <td>
                        {row.instance_percentage.toFixed(2)}%
                      </td>
                      <td>
                        {row.global_instance_percentage.toFixed(2)}%
                      </td>
                      <td>
                        {row.absolute_percentage_point_deviation.toFixed(3)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Panel>

          <Panel
            title="Split memberships"
            meta={
              membershipsQuery.isPending
                ? 'Loading assignments'
                : `First ${Math.min(
                    ASSIGNMENT_PREVIEW_SIZE,
                    selectedMemberships.length,
                  )} of ${formatNumber(selectedMemberships.length)}`
            }
          >
            {membershipsQuery.isPending ? (
              <p className="panel-body" role="status">
                Loading split memberships…
              </p>
            ) : membershipsQuery.isError ? (
              <Notice>
                Split membership rows could not be loaded. The split-level
                artifact above remains available.
              </Notice>
            ) : (
              <div className="table-scroll">
                <table>
                  <thead>
                    <tr>
                      <th>Partition</th>
                      <th>Content ID</th>
                      <th>Record ID</th>
                    </tr>
                  </thead>
                  <tbody>
                    {selectedMemberships
                      .slice(0, ASSIGNMENT_PREVIEW_SIZE)
                      .map((membership) => (
                        <tr
                          key={`${membership.record_id}-${membership.content_id}-${membership.partition}`}
                        >
                          <td>{membership.partition}</td>
                          <td className="mono">
                            {membership.content_id}
                          </td>
                          <td className="mono">
                            {membership.record_id}
                          </td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
            )}
          </Panel>
        </>
      ) : (
        <Panel
          title="Grouping context before assignment"
          meta="NO SYNTHETIC PRE-SPLIT GROUPS"
        >
          <div className="panel-body">
            {selected.cluster_run_id ? (
              <p>
                This split references clustering run{' '}
                <code>{selected.cluster_run_id}</code>.
              </p>
            ) : (
              <p>
                This split does not reference a clustering run in the exported
                split artifact.
              </p>
            )}

            <p>
              The split resources expose the resulting split definition and
              final membership assignments. They do not provide a standalone
              pre-allocation group-membership table for this view.
            </p>
          </div>

          <Notice>
            No DEMO groups are substituted here. A BEFORE SPLIT grouping view
            should only be shown when the corresponding grouping artifact is
            joined explicitly.
          </Notice>
        </Panel>
      )}

      <Panel
        title="Available partition artifacts"
        meta={`${splits.length} exported splits · no scientific ranking`}
      >
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Strategy</th>
                <th>Source</th>
                <th>Seed</th>
                <th>Train records</th>
                <th>Val records</th>
                <th>Test records</th>
                <th>Split space</th>
              </tr>
            </thead>
            <tbody>
              {splits.map((split) => (
                <tr key={split.split_space_id}>
                  <td>
                    <button
                      className="row-link"
                      onClick={() =>
                        setSelectedSplitSpaceId(split.split_space_id)
                      }
                    >
                      {split.strategy}
                    </button>
                  </td>
                  <td>{formatStrategy(split.source_strategy)}</td>
                  <td>{split.split_seed}</td>
                  <td>
                    {formatNumber(split.partitions.train.n_records)}
                  </td>
                  <td>
                    {formatNumber(split.partitions.val.n_records)}
                  </td>
                  <td>
                    {formatNumber(split.partitions.test.n_records)}
                  </td>
                  <td className="mono">{split.split_space_id}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>

      <Notice>
        A split assignment does not by itself prove that visual or temporal
        dependence has been eliminated. The split artifact is evidence of the
        stored partition, not a claim of leakage-free ground truth.
      </Notice>
    </>
  )
}

