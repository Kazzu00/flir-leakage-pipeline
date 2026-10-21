import { useSearchParams } from 'react-router-dom'
import { Select } from '@/components/feedback/Primitives'
import { DetectionEvaluation } from '@/features/detection/DetectionEvaluation'
import { RealSplits } from './splitting/RealSplits'
import { ReferenceComparison } from './ReferenceComparison'
export function EvaluationOverview() {
  const [params, setParams] = useSearchParams()
  const requested = params.get('view')
  const view =
    requested === 'splits' || requested === 'reference' ? requested : 'detector'
  return (
    <>
      <div className="toolbar detection-view-selector">
        <Select
          label="Evaluation evidence"
          value={view}
          onChange={(v) => setParams({ view: v })}
          options={[
            { value: 'splits', label: 'Grouping / split' },
            { value: 'detector', label: 'Detector evaluation' },
            {
              value: 'reference',
              label: 'Eyes in the Sky / experimental context',
            },
          ]}
        />
      </div>
      {view === 'splits' ? (
        <RealSplits />
      ) : view === 'reference' ? (
        <ReferenceComparison />
      ) : (
        <DetectionEvaluation />
      )}
    </>
  )
}
