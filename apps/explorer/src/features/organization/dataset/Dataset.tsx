import { useSnapshot } from '@/data/provider'
import { Panel, Notice } from '@/components/feedback/Primitives'
import { Status } from '@/components/feedback/Status'
export function Dataset() {
  const { data } = useSnapshot()
  if (!data) return null
  return (
    <>
      <Panel title={data.dataset.name} meta="DEMO">
        <div className="stat-row">
          <div>
            <strong>{data.dataset.occurrences}</strong>
            <span>Historical occurrences / frame_id</span>
          </div>
          <div>
            <strong>{data.dataset.uniqueContents}</strong>
            <span>Unique contents / content_id</span>
          </div>
          <div>
            <strong>{data.dataset.duplicateGroups}</strong>
            <span>Exact duplicate groups</span>
          </div>
        </div>
        <dl className="details">
          <dt>Dataset identity</dt>
          <dd className="mono">{data.dataset.id}</dd>
          <dt>Provenance</dt>
          <dd>{data.dataset.provenance}</dd>
          <dt>Verification</dt>
          <dd>
            <Status state={data.dataset.verification} />
          </dd>
        </dl>
      </Panel>
      <Panel title="Identity and traceability">
        <p className="identity-chain">
          frame_id → content_id → embedding_row → cluster_id → group_id → split
        </p>
        <table>
          <thead>
            <tr>
              <th>Identity</th>
              <th>Meaning</th>
            </tr>
          </thead>
          <tbody>
            {[
              [
                'frame_id',
                'An occurrence; exact copies retain separate identities',
              ],
              [
                'content_id',
                'Exact bytes; one embedding / clustering row per unique content',
              ],
              ['cluster_id', 'Run-local assignment; noise is −1'],
              [
                'group_id',
                'Indivisible allocation unit; noise uses singleton groups',
              ],
              [
                'original_split',
                'Historical metadata, never representation input',
              ],
            ].map(([id, meaning]) => (
              <tr key={id}>
                <td className="mono">{id}</td>
                <td>{meaning}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Panel>
      <Panel title="Encoder coverage" meta="DEMO">
        <table>
          <thead>
            <tr>
              <th>Encoder</th>
              <th>Contents</th>
              <th>Dimension</th>
              <th>Representation</th>
            </tr>
          </thead>
          <tbody>
            {data.embeddings.map((e) => (
              <tr key={e.id}>
                <td>{e.encoder}</td>
                <td>
                  {e.coverage} / {data.dataset.uniqueContents}
                </td>
                <td>{e.dimensions}</td>
                <td>{e.pooling}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Panel>
      <Notice>
        No real manifest is loaded. Annotation conflicts must be preserved by
        future adapters, not silently resolved. Labels and historical split are
        not visual encoder inputs.
      </Notice>
    </>
  )
}
