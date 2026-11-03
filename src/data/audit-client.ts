import { z } from 'zod'

const event = z.object({
  event_id: z.string(),
  asset_id: z.string(),
  record_locator: z.string(),
  source_video: z.string().nullable(),
  start_second: z.number().nullable(),
  end_second: z.number().nullable(),
  start_index: z.number().nullable(),
  end_index: z.number().nullable(),
  method: z.string(),
  method_version: z.string(),
  alert_type: z.string(),
  metrics: z.record(z.string(), z.number().nullable()),
  evidence: z.array(z.string()),
  verification: z.literal('candidate_only'),
})
const decision = z.object({
  event_id: z.string(),
  decision_id: z.string(),
  version: z.number().int().positive(),
  expected_version: z.number().int().nonnegative(),
  status: z.enum(['pending', 'reviewed']),
  continuity: z.enum(['continuous', 'shot_change', 'uncertain']),
  dependence: z.enum(['same_region', 'different_region', 'uncertain']),
  confidence: z.enum(['low', 'medium', 'high', 'undetermined']),
  reviewer: z.string(),
  justification: z.string(),
  created_at: z.string(),
  phenomena: z.array(
    z.enum([
      'zoom',
      'camera_motion',
      'gain_change',
      'polarity_change',
      'signal_loss',
      'unknown',
    ]),
  ),
  evidence_consulted: z.array(z.string()),
  evidence_sha256: z.string(),
})
const snapshot = z.object({
  schema: z.literal('audit-evidence-v1'),
  events: z.array(event),
  decisions: z.array(decision),
  assets: z.array(
    z.object({
      asset_id: z.string(),
      filename: z.string(),
      sha256: z.string(),
      metadata: z.record(z.string(), z.unknown()),
    }),
  ),
  observation_count: z.number(),
  last_import: z.unknown(),
})
export type AuditSnapshot = z.infer<typeof snapshot>
export type AuditEvent = z.infer<typeof event>
export type AuditDecision = z.infer<typeof decision>
export type DecisionInput = Omit<
  AuditDecision,
  'decision_id' | 'version' | 'created_at' | 'evidence_sha256'
>

// Local reviewer is deliberately separate from immutable research snapshots.
// Never send private evidence or decisions to a configurable remote endpoint.
export function auditBase() {
  if (!['127.0.0.1', 'localhost'].includes(window.location.hostname))
    throw new Error(
      'La revisión solo está disponible desde un navegador local.',
    )
  return 'http://127.0.0.1:8765'
}
async function response(path: string, init?: RequestInit) {
  const result = await fetch(`${auditBase()}${path}`, init)
  const body: unknown = await result.json()
  if (!result.ok) {
    const error = z.object({ error: z.string() }).safeParse(body)
    throw new Error(
      error.success
        ? error.data.error
        : `Servicio local: HTTP ${result.status}`,
    )
  }
  return body
}
export async function loadAudit() {
  return snapshot.parse(await response('/api/audit'))
}
export async function saveAuditDecision(input: DecisionInput) {
  return decision.parse(
    await response('/api/decisions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-Audit-Review': '1' },
      body: JSON.stringify(input),
    }),
  )
}
export function auditMedia(eventId: string, index: number) {
  return `${auditBase()}/api/media/${encodeURIComponent(eventId)}/${index}`
}
