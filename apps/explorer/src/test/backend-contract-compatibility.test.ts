import { beforeAll, describe, expect, it } from 'vitest'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import prepareSyntheticEvidence from '../../e2e/global-setup'
import {
  ClusteringConfigurationSchema,
  OrganizationEvidenceResourceSchemas,
} from '@/contracts/organization-evidence-v2'

describe('Python producer to TypeScript consumer compatibility', () => {
  beforeAll(() => prepareSyntheticEvidence(), 60000)

  for (const [resource, schema] of Object.entries(
    OrganizationEvidenceResourceSchemas,
  )) {
    it(`accepts the actual synthetic backend ${resource} resource`, () => {
      const root = process.env.FLIR_E2E_SYNTHETIC_BUNDLE!
      const value = JSON.parse(
        readFileSync(resolve(root, 'export', `${resource}.json`), 'utf8'),
      )
      const parsed = schema.safeParse(value)
      expect(
        parsed.success,
        parsed.success ? resource : JSON.stringify(parsed.error.issues),
      ).toBe(true)
    })
  }

  it('rejects malformed recovered provenance without promoting it to a full artifact', () => {
    const root = process.env.FLIR_E2E_SYNTHETIC_BUNDLE!
    const [configuration] = JSON.parse(
      readFileSync(
        resolve(root, 'export/clustering_configurations.json'),
        'utf8',
      ),
    )
    const provenance = {
      source_kind: 'clustering_run_table',
      source_sha256: 'a'.repeat(64),
      configuration_identity_verified: true,
    }
    expect(
      ClusteringConfigurationSchema.safeParse({
        ...configuration,
        configuration_provenance: provenance,
      }).success,
    ).toBe(false)
    expect(
      ClusteringConfigurationSchema.safeParse({
        ...configuration,
        source_kind: 'frozen_split_membership',
        full_clustering_artifact_available: false,
        feature_space_id: null,
        model_id: null,
        reduction_seed: 42,
        configuration_provenance: provenance,
      }).success,
    ).toBe(true)
    for (const provenance of [
      {
        source_kind: 'clustering_run_table',
        source_sha256: 'not-a-digest',
        configuration_identity_verified: true,
      },
      {
        source_kind: 'clustering_run_table',
        source_sha256: 'a'.repeat(64),
        configuration_identity_verified: false,
      },
    ]) {
      expect(
        ClusteringConfigurationSchema.safeParse({
          ...configuration,
          configuration_provenance: provenance,
        }).success,
      ).toBe(false)
    }
  })
})
