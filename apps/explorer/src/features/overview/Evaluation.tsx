import { Link } from 'react-router-dom'
import { PageTitle, Panel, Notice } from '@/components/feedback/Primitives'
import { Status } from '@/components/feedback/Status'
import { useSnapshot } from '@/data/provider'
import { useDetectionSnapshot } from '@/features/detection/query'
export function Evaluation() {
  const { data } = useSnapshot()
  const detector = useDetectionSnapshot()
  return (
    <>
      <PageTitle
        eyebrow="CROSS-MODULE / EVIDENCE"
        title="Global evaluation"
        description="Keep measurement families and experimental protocols distinct."
      />
      <Notice>
        Integration pending. This frontend has no common validated end-to-end
        experiment; metrics from different datasets or protocols cannot be
        pooled into one score.
      </Notice>
      <Panel title="Evaluation boundaries">
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Module</th>
                <th>Supported metric family</th>
                <th>Evidence state</th>
                <th>Explore</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Preprocessing</td>
                <td>NIQE / BRISQUE / PIQE / sigma / sharpness</td>
                <td>
                  <Status state="mock" />
                </td>
                <td>
                  <Link className="row-link" to="/preprocessing/quality">
                    Quality →
                  </Link>
                </td>
              </tr>
              <tr>
                <td>Organization</td>
                <td>Preservation / ARI / AMI / residual correlation</td>
                <td>
                  <Status state={data?.leakage ? 'experimental' : 'mock'} />
                  {data?.leakage && ' Artifact metrics reported upstream'}
                </td>
                <td>
                  <Link className="row-link" to="/organization/splits">
                    Partitions →
                  </Link>
                </td>
              </tr>
              <tr>
                <td>Downstream detector</td>
                <td>Precision / Recall / mAP@50 / mAP@50–95</td>
                <td>
                  <span>
                    {detector.data
                      ? 'REAL / VERIFIED · upstream contract'
                      : detector.isPending
                        ? 'Loading detector contract'
                        : 'Verified results unavailable'}
                  </span>
                </td>
                <td>
                  <Link className="row-link" to="/organization/detector">
                    Controlled protocol →
                  </Link>
                </td>
              </tr>
              <tr>
                <td>Panoptic segmentation</td>
                <td>PQ / SQ / RQ / IoU / F1 / mask AP</td>
                <td>
                  <Status state="inconsistent" />
                </td>
                <td>
                  <Link className="row-link" to="/segmentation/evaluation">
                    Evaluation →
                  </Link>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </Panel>
      <Panel title="Before scientific comparison">
        <div className="panel-body">
          <ol
            style={{ listStyle: 'decimal', paddingLeft: 18, lineHeight: 2.2 }}
          >
            <li>
              Verify dataset identity, content mapping and split provenance.
            </li>
            <li>
              Preserve source caveats and distinguish pilots from controlled
              experiments.
            </li>
            <li>
              Match class definitions, annotation type and evaluation protocol.
            </li>
            <li>
              Record budgets, seeds, model revision and artifact verification.
            </li>
            <li>
              Report uncertainty and residual relationships without claiming
              independence.
            </li>
          </ol>
        </div>
      </Panel>
    </>
  )
}
