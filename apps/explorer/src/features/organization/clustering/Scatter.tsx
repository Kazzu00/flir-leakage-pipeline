import type { ClusterRun } from '@/contracts'
import { Chart } from '@/components/visualization/Chart'
const colors = [
  '#73d9be',
  '#8cb7ee',
  '#edc583',
  '#bca9ee',
  '#e49c7b',
  '#c4d28c',
]
export function Scatter({
  run,
  selected,
  after,
  splitMap,
  onSelect,
  onContentSelect,
  groupBy = 'Cluster',
}: {
  run: ClusterRun
  selected: number | null
  after: boolean
  splitMap: Map<string, string>
  onSelect: (id: number) => void
  onContentSelect?: (id: string) => void
  groupBy?: string
}) {
  const data = run.points.map((p) => ({
    value: [p.x, p.y],
    name: p.contentId,
    symbol:
      p.clusterId === -1
        ? 'triangle'
        : after
          ? splitMap.get(p.contentId) === 'train'
            ? 'circle'
            : splitMap.get(p.contentId) === 'validation'
              ? 'rect'
              : 'diamond'
          : 'circle',
    symbolSize: selected === null || selected === p.clusterId ? 7 : 4,
    itemStyle: {
      color:
        groupBy === 'Source video'
          ? colors[Number(p.sourceVideo.slice(-1)) - 1]
          : groupBy === 'Split' && after
            ? ({ train: colors[0], validation: colors[1], test: colors[2] }[
                splitMap.get(p.contentId) ?? ''
              ] ?? '#94a9af')
            : p.clusterId === -1
              ? '#94a9af'
              : colors[p.clusterId % colors.length],
      opacity: selected === null || selected === p.clusterId ? 0.9 : 0.18,
    },
  }))
  return (
    <Chart
      height={355}
      label={`DEMO scatter, ${run.points.length} unique contents; ${after ? 'split shapes added' : 'before split'}. Use cluster buttons and content selector for keyboard inspection.`}
      onSelect={(index) => {
        const p = run.points[index]
        if (p) {
          onSelect(p.clusterId)
          onContentSelect?.(p.contentId)
        }
      }}
      option={{
        grid: { left: 45, right: 25, top: 25, bottom: 38 },
        tooltip: { trigger: 'item', formatter: '{b}' },
        xAxis: {
          type: 'value',
          min: 0,
          max: 100,
          name: 'Dimension 1',
          nameLocation: 'middle',
          nameGap: 24,
          splitLine: { lineStyle: { color: '#2b3a3f', type: 'dashed' } },
          axisLabel: { show: false },
        },
        yAxis: {
          type: 'value',
          min: 0,
          max: 100,
          name: 'Dimension 2',
          splitLine: { lineStyle: { color: '#2b3a3f', type: 'dashed' } },
          axisLabel: { show: false },
        },
        series: [{ type: 'scatter', data }],
      }}
    />
  )
}
