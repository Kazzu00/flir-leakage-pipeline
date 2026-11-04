import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { organizationTabs, legacyOrganizationRoutes } from '@/app/navigation'
import { TechnicalDetailsDrawer } from '@/components/research/TechnicalDetailsDrawer'
import { resolveRuntimeConfig } from '@/data/runtime-config'
import { parseLeakageSnapshot } from '@/data/adapters/RealLeakageAdapter'
import { LinkageExplorer } from '@/features/organization/LinkageExplorer'
import { researchFixture } from './research-fixture'
import exportedV2 from './fixtures/leakage-v2-synthetic.json'
describe('UX v2 and scientific evidence', () => {
  it('accepts Python-exported native synthetic sequences and linkage', () => {
    const snapshot = parseLeakageSnapshot(exportedV2)
    expect(snapshot.schemaVersion).toBe('LeakageSnapshotV2')
    if (snapshot.schemaVersion === 'LeakageSnapshotV2')
      expect(
        snapshot.research.linkage.artifact?.candidates[0].sequenceIds,
      ).toHaveLength(2)
  })
  it('has four primary sections and preserves all eight old routes', () => {
    expect(organizationTabs.map((t) => t[0])).toEqual([
      '',
      'explore',
      'sequences',
      'evaluation',
    ])
    expect(Object.keys(legacyOrganizationRoutes)).toHaveLength(8)
    expect(legacyOrganizationRoutes.reduction).toBe(
      '/organization/explore?view=reduction',
    )
  })
  it('hides technical details until keyboard activation and restores focus', async () => {
    render(
      <TechnicalDetailsDrawer>
        <p>Private audit panel</p>
      </TechnicalDetailsDrawer>,
    )
    expect(screen.queryByText('Private audit panel')).toBeNull()
    const trigger = screen.getByRole('button', { name: 'Technical details' })
    trigger.focus()
    await userEvent.keyboard('{Enter}')
    expect(screen.getByRole('dialog')).toBeVisible()
    expect(screen.getByText('Private audit panel')).toBeVisible()
    await userEvent.keyboard('{Escape}')
    expect(screen.queryByRole('dialog')).toBeNull()
    expect(trigger).toHaveFocus()
  })
  it('accepts candidates and bound reviews without conflating them', () => {
    const s = parseLeakageSnapshot(researchFixture())
    expect(s.schemaVersion).toBe('LeakageSnapshotV2')
    if (s.schemaVersion !== 'LeakageSnapshotV2') throw Error('Wrong version')
    expect(s.research.sequences.artifact?.zones.map((z) => z.decision)).toEqual(
      ['candidate', 'accepted', 'reviewed'],
    )
    expect(s.research.experiments).toEqual({ state: 'pending', artifact: null })
    render(<LinkageExplorer research={s.research} />)
    expect(
      screen.getByText(/A candidate line is not a confirmed link/),
    ).toBeVisible()
    expect(screen.getByText('candidate', { exact: true })).toBeVisible()
  })
  it('rejects unbound reviews, foreign sources and unsupported confirmation', () => {
    const s = researchFixture()
    s.research.sequences.artifact.zones[1].reviewId = null
    expect(() => parseLeakageSnapshot(s)).toThrow('identity-mismatch')
    const foreign = researchFixture()
    foreign.research.sequences.artifact.zones[0].sourceVideo = 'video-foreign'
    expect(() => parseLeakageSnapshot(foreign)).toThrow('identity-mismatch')
    const link = researchFixture()
    link.research.linkage.artifact.candidates[0].state = 'confirmed'
    expect(() => parseLeakageSnapshot(link)).toThrow('identity-mismatch')
  })
  it('uses runtime configuration before local Vite settings', () => {
    expect(resolveRuntimeConfig(undefined, {}).data?.dataMode).toBe('demo')
    expect(
      resolveRuntimeConfig(undefined, { VITE_DATA_MODE: 'real' }).data
        ?.dataMode,
    ).toBe('real')
    expect(
      resolveRuntimeConfig(
        { dataMode: 'demo', snapshotUrl: '/runtime/demo.json' },
        { VITE_DATA_MODE: 'real' },
      ).data?.dataMode,
    ).toBe('demo')
    expect(
      resolveRuntimeConfig({
        dataMode: 'real',
        snapshotUrl: '/runtime/leakage-snapshot.json',
      }).success,
    ).toBe(true)
  })
  it.each([
    'https://external.test/snapshot.json',
    '//external.test/a.json',
    '/runtime/../private.json',
    '/runtime/a.json?secret=1',
    '/runtime/evil\\file.json',
  ])('rejects unsafe URL %s', (url) => {
    expect(
      resolveRuntimeConfig({ dataMode: 'real', snapshotUrl: url }).success,
    ).toBe(false)
  })
  it('rejects invalid mode without a silent demo fallback', () => {
    expect(
      resolveRuntimeConfig({ dataMode: 'reel', snapshotUrl: '/runtime/a.json' })
        .success,
    ).toBe(false)
  })
})
