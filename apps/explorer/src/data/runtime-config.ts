import { z } from 'zod'
export const RuntimeConfigSchema = z
  .object({
    dataMode: z.enum(['demo', 'real']),
    snapshotUrl: z
      .string()
      .regex(/^\/(?!\/)[a-zA-Z0-9_./-]+\.json$/)
      .refine(
        (url) => !url.split('/').includes('..'),
        'Parent paths are not runtime URLs',
      ),
  })
  .strict()
declare global {
  interface Window {
    __FLIR_CONFIG__?: unknown
  }
}
export function resolveRuntimeConfig(
  runtime: unknown,
  env: Record<string, unknown> = {},
) {
  return RuntimeConfigSchema.safeParse(
    runtime ?? {
      dataMode: env.VITE_DATA_MODE ?? 'demo',
      snapshotUrl:
        env.VITE_LEAKAGE_SNAPSHOT_URL || '/runtime/leakage-snapshot.json',
    },
  )
}
