import { Notice } from '@/components/feedback/Primitives'

/** Availability of connected contracts, not a claim about all upstream artifacts. */
export function VariantRegister() {
  return (
    <section aria-label="Experimental variants">
      <h2>Experimental variants and evidence boundaries</h2>
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              <th>Variant / source</th>
              <th>Version / state</th>
              <th>Annotations</th>
              <th>Split policy / results</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>A. Original labeled: Imagenes + Adendo + Etiquetas</td>
              <td>
                Historical organization-evidence-v2; expanded
                fixed_test_train_val_v1 when the local candidate contract is
                available
              </td>
              <td>
                Existing labels; unresolved conflicts and class-4 nomenclature
                remain
              </td>
              <td>
                Historical / random-content / C10 / C12: YOLO11n results.
                S1/S2/S3 historical and expanded: PROVISIONAL, test PROPOSED,
                YOLO11m results UNAVAILABLE. Adendo is not part of the
                historical detector experiment.
              </td>
            </tr>
            <tr>
              <td>B. Frames without HUD</td>
              <td>
                UNAVAILABLE in connected scientific contracts; no version
                admitted here
              </td>
              <td>
                Not established by a connected annotation contract; do not
                inherit original labels automatically
              </td>
              <td>
                Split policy and detector results UNAVAILABLE. No completed
                training claimed.
              </td>
            </tr>
            <tr>
              <td>C. Unprocessed source videos / sampled frames</td>
              <td>
                Separate real representation snapshot and organization timeline
                evidence, when available
              </td>
              <td>
                Video contents are not automatically labeled by candidate
                linkage
              </td>
              <td>
                Presentation timelines are not confirmed sequences. No
                sampled-video detector results in the historical contract.
              </td>
            </tr>
          </tbody>
        </table>
      </div>
      <Notice>
        Availability refers to imported frontend contracts. It does not imply
        that other upstream artifacts do not exist. Variant differences do not
        identify causal effects. No new training is scheduled by this viewer.
      </Notice>
    </section>
  )
}
