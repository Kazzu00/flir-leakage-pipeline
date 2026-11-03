import { describe, it, expect } from 'vitest'
import { readFileSync } from 'node:fs'
import {
  needsEvidence,
  temporalOrder,
  type AuditEvent,
  type DecisionInput,
} from '@/data/audit-client'

function input(values: Partial<DecisionInput> = {}): DecisionInput {
  return {
    event_id: 'SYNTHETIC',
    expected_version: 0,
    status: 'reviewed',
    confidence: 'undetermined',
    continuity: 'uncertain',
    dependence: 'uncertain',
    reviewer: 'SYNTHETIC',
    justification: 'Revisión sintética',
    phenomena: [],
    evidence_consulted: [],
    external_evidence_consulted: [],
    ...values,
  }
}
function event(
  id: string,
  time: number | null,
  source = 'SYNTHETIC',
): AuditEvent {
  return {
    event_id: id,
    asset_id: 'SYNTHETIC-asset',
    record_locator: id,
    source_video: source,
    start_second: time,
    end_second: time,
    start_index: null,
    end_index: null,
    event_start_sample: null,
    event_end_sample: null,
    representative_second: null,
    sample_rate_fps: null,
    temporal_basis: 'synthetic',
    precision_seconds: null,
    method: 'synthetic',
    method_version: '1',
    alert_type: 'synthetic',
    metrics: {},
    evidence: [],
    verification: 'candidate_only',
  }
}
describe('audit v1.1 guards and temporal navigation', () => {
  it('retains UTF-8 Spanish labels without lossy PowerShell replacements', () => {
    const text = readFileSync(
      'src/features/organization/AuditReview.tsx',
      'utf8',
    )
    for (const label of [
      'Consulté la evidencia externa',
      'Anomalías históricas',
      'revisión explícita',
      'canónico',
      'Precisión',
    ])
      expect(text).toContain(label)
    expect(text).not.toMatch(/[A-Za-z]\?[A-Za-z]|\uFFFD|\u00C3[\u0080-\u00BF]/)
  })
  it('permits uncertainty and pending judgments but requires evidence for conclusions', () => {
    expect(needsEvidence(input())).toBe(false)
    expect(needsEvidence(input({ confidence: 'high' }))).toBe(true)
    expect(
      needsEvidence(input({ status: 'pending', confidence: 'high' })),
    ).toBe(false)
    expect(needsEvidence(input({ continuity: 'shot_change' }))).toBe(true)
    expect(needsEvidence(input({ dependence: 'different_region' }))).toBe(true)
    expect(
      needsEvidence(
        input({
          continuity: 'shot_change',
          dependence: 'same_region',
          evidence_consulted: ['SYNTHETIC.png'],
        }),
      ),
    ).toBe(false)
    expect(
      needsEvidence(
        input({
          confidence: 'high',
          external_evidence_consulted: ['registered-id'],
        }),
      ),
    ).toBe(false)
  })
  it('sorts within source and preserves unknown times and the input array', () => {
    const original = [
      event('b', 2),
      event('unknown', null),
      event('a', 1),
      event('other', 0, 'ZZ-SYNTHETIC'),
    ]
    expect(temporalOrder(original).map((e) => e.event_id)).toEqual([
      'a',
      'b',
      'unknown',
      'other',
    ])
    expect(temporalOrder(original, true).map((e) => e.event_id)).toEqual([
      'b',
      'a',
      'unknown',
      'other',
    ])
    expect(original[0].event_id).toBe('b')
  })
})
