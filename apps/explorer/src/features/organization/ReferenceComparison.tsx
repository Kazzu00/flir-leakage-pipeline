import { useDetectionSnapshot } from '@/features/detection/query'
import { Notice } from '@/components/feedback/Primitives'
import { VariantRegister } from './VariantRegister'

export function ReferenceComparison() {
  const detector = useDetectionSnapshot(true)
  return (
    <section aria-label="Scientific comparison context">
      <h2>Eyes in the Sky and FLIR experimental context</h2>
      <Notice>
        Acosta-Bernal et al. (2025): values reported in the thesis proposal
        according to the project brief supplied by the researcher. The complete
        article and its protocol have NOT been verified here. These values are
        not imported experimental results.
      </Notice>
      <div className="table-scroll">
        <table>
          <caption>
            Reported values only; no ranking, improvement percentage or protocol
            equivalence
          </caption>
          <thead>
            <tr>
              <th>Source</th>
              <th>mAP@50</th>
              <th>Recall</th>
              <th>Evidence / test</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Eyes in the Sky · Acosta-Bernal et al. (2025)</td>
              <td>0.721</td>
              <td>0.663</td>
              <td>
                PROPOSAL-REPORTED / full article unverified; test comparability
                UNAVAILABLE
              </td>
            </tr>
            {detector.data?.strategies.map((s) => (
              <tr key={s.strategy}>
                <td>YOLO11n · {s.strategy}</td>
                <td>{s.mean_map50}</td>
                <td>{s.mean_recall}</td>
                <td>
                  Historical exported comparison; test membership varies by
                  split
                </td>
              </tr>
            ))}
            <tr>
              <td>YOLO11m · common-test protocol</td>
              <td>UNAVAILABLE</td>
              <td>UNAVAILABLE</td>
              <td>Test PROPOSED; no results admitted to this frontend</td>
            </tr>
            <tr>
              <td>YOLO11m · HUD-free variant</td>
              <td>UNAVAILABLE</td>
              <td>UNAVAILABLE</td>
              <td>Protocol, labels and results not connected</td>
            </tr>
          </tbody>
        </table>
      </div>
      {detector.isPending && (
        <p role="status">Loading historical detector contract…</p>
      )}
      {detector.isError && (
        <Notice>
          UNAVAILABLE: historical detector contract could not be validated. No
          values substituted.
        </Notice>
      )}
      <p>
        Compare descriptively only. Models, populations, training conditions,
        metrics and tests have not been established as equivalent. YOLO11n
        values are copied from the existing strategy summaries, not recomputed.
      </p>
      <VariantRegister />
    </section>
  )
}
