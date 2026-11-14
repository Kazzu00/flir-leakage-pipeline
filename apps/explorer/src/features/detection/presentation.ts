import type { EChartsOption } from 'echarts'
import type {
  DetectionAssociation,
  DetectionExport,
  DetectionStrategyId,
} from './schema'
import { strategyIds } from './schema'

export const labels: Record<DetectionStrategyId, string> = {
  historical: 'Historical',
  random_content: 'Random content-level',
  C10: 'C10',
  C12: 'C12',
}
export const colors = ['#b4c5cc', '#72c9ba', '#d9bb83', '#9faee0']
export const symbols = ['circle', 'rect', 'triangle', 'diamond']
export const percent = (v: number | null, digits = 1) =>
  v === null ? 'Unavailable' : `${(v * 100).toFixed(digits)}%`
export const decimal = (v: number | null, digits = 3) =>
  v === null ? 'Unavailable' : v.toFixed(digits)
export const strategyOptions = strategyIds.map((value) => ({
  value,
  label: labels[value],
}))
export function comparisonOption(data: DetectionExport): EChartsOption {
  return {
    grid: { left: 158, right: 64, top: 16, bottom: 38 },
    tooltip: { trigger: 'item', renderMode: 'richText' },
    xAxis: {
      type: 'value',
      name: 'mAP50–95',
      min: 0,
      axisLabel: { color: '#ccd6da', formatter: (v: number) => percent(v, 0) },
      splitLine: { lineStyle: { color: '#2c3a40' } },
    },
    yAxis: {
      type: 'category',
      inverse: true,
      data: strategyIds.map((s) => labels[s]),
      axisLabel: { color: '#ccd6da', fontSize: 12 },
    },
    series: strategyIds.flatMap((strategy, i) => {
      const summary = data.strategies.find((s) => s.strategy === strategy)!
      return [
        {
          type: 'line' as const,
          name: `${labels[strategy]} · exported split range`,
          data:
            summary.min_map50_95 === null || summary.max_map50_95 === null
              ? []
              : [
                  [summary.min_map50_95, i],
                  [summary.max_map50_95, i],
                ],
          symbol: 'none',
          lineStyle: { color: colors[i], opacity: 0.5, width: 2 },
          tooltip: { show: false },
        },
        {
          type: 'scatter' as const,
          name: `${labels[strategy]} · split mean`,
          symbol: symbols[i],
          symbolSize: 10,
          itemStyle: { color: colors[i] },
          data: data.splits
            .filter((s) => s.strategy === strategy && s.mean_map50_95 !== null)
            .map((s) => ({
              name: `Split seed ${s.split_seed} · ${decimal(s.mean_map50_95, 6)}`,
              value: [s.mean_map50_95!, i],
            })),
        },
        {
          type: 'scatter' as const,
          name: `${labels[strategy]} · strategy mean`,
          symbol: 'diamond',
          symbolSize: 19,
          itemStyle: {
            color: colors[i],
            borderColor: '#151f22',
            borderWidth: 2,
          },
          label: {
            show: true,
            position: 'top' as const,
            color: '#edf3f5',
            formatter: percent(summary.mean_map50_95),
            fontSize: 15,
          },
          data:
            summary.mean_map50_95 === null
              ? []
              : [
                  {
                    name: `Strategy mean · ${decimal(summary.mean_map50_95, 6)}`,
                    value: [summary.mean_map50_95, i],
                  },
                ],
        },
      ]
    }),
  }
}
/** Arrange only exported split points. No seed replication, centering or trend fitting. */
export function associationOption(a: DetectionAssociation): EChartsOption {
  return {
    grid: { left: 62, right: 25, bottom: 75, top: 48 },
    tooltip: { trigger: 'item', renderMode: 'richText' },
    legend: {
      top: 8,
      left: 'center',
      textStyle: { color: '#ccd6da' },
      itemWidth: 13,
    },
    xAxis: {
      type: 'value',
      name: a.residual_metric,
      nameLocation: 'middle',
      nameGap: 35,
      scale: true,
      axisLabel: { color: '#ccd6da' },
      splitLine: { lineStyle: { color: '#2c3a40' } },
    },
    yAxis: {
      type: 'value',
      name: a.detector_metric === 'map50_95' ? 'mAP50–95' : a.detector_metric,
      axisLabel: { color: '#ccd6da', formatter: (v: number) => percent(v, 0) },
      splitLine: { lineStyle: { color: '#2c3a40' } },
    },
    series: strategyIds.map((strategy, i) => ({
      type: 'scatter',
      name: labels[strategy],
      symbol: symbols[i],
      symbolSize: 12,
      itemStyle: { color: colors[i] },
      data: a.points
        .filter((p) => p.strategy === strategy && p.x !== null && p.y !== null)
        .map((p) => ({
          name: `Split seed ${p.split_seed}\nx: ${decimal(p.x, 6)} · y: ${decimal(p.y, 6)}`,
          value: [p.x!, p.y!],
        })),
    })),
  }
}
