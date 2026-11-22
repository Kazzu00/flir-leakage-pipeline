import ReactECharts from 'echarts-for-react/esm/core'
import * as echarts from 'echarts/core'
import { ScatterChart, LineChart, BarChart } from 'echarts/charts'
import {
  GridComponent,
  TooltipComponent,
  LegendComponent,
} from 'echarts/components'
import { SVGRenderer, CanvasRenderer } from 'echarts/renderers'
import type { EChartsOption } from 'echarts'
echarts.use([
  ScatterChart,
  LineChart,
  BarChart,
  GridComponent,
  TooltipComponent,
  LegendComponent,
  SVGRenderer,
  CanvasRenderer,
])
export function Chart({
  option,
  label,
  onSelect,
  height = 340,
  renderer = 'svg',
}: {
  option: EChartsOption
  label: string
  onSelect?: (index: number) => void
  height?: number
  renderer?: 'svg' | 'canvas'
}) {
  return (
    <div role="img" aria-label={label} className="chart">
      <ReactECharts
        echarts={echarts}
        option={{
          animation: false,
          // Keep plot contrast stable across shell themes, including SVG axis labels.
          backgroundColor: '#151f22',
          textStyle: {
            fontFamily: 'Inter, system-ui, sans-serif',
            color: '#8c9da3',
          },
          ...option,
        }}
        style={{ height }}
        opts={{ renderer }}
        onEvents={
          onSelect
            ? {
                click: (event: { dataIndex: number }) =>
                  onSelect(event.dataIndex),
              }
            : undefined
        }
      />
    </div>
  )
}
