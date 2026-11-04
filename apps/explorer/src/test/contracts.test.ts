import { describe, it, expect } from 'vitest'
import { SnapshotSchema } from '@/contracts'
import { fixture } from '@/data/mock/fixture'
import { MockDataAdapter } from '@/data/adapters/MockDataAdapter'
describe('Scientific identity boundaries', () => {
  it('rejects fabricated split counts and merged noise groups', () => {
    const data = structuredClone(fixture)
    data.splits[2].counts.train++
    expect(SnapshotSchema.safeParse(data).success).toBe(false)
    const other = structuredClone(fixture)
    const noise = other.clusterings[0].points.filter((p) => p.clusterId === -1)
    noise[1].groupId = noise[0].groupId
    expect(SnapshotSchema.safeParse(other).success).toBe(false)
  })
  it('validates the deterministic mock adapter', async () => {
    const data = await new MockDataAdapter().getSnapshot()
    expect(data.dataset.uniqueContents).toBe(144)
    expect(
      data.clusterings[0].points.reduce((n, p) => n + p.frameIds.length, 0),
    ).toBe(data.dataset.occurrences)
    for (const split of data.splits)
      expect(Object.values(split.counts).reduce((a, b) => a + b, 0)).toBe(
        data.dataset.occurrences,
      )
  })
  it('rejects duplicate clustering contents', () => {
    const data = structuredClone(fixture)
    data.clusterings[0].points[1].contentId =
      data.clusterings[0].points[0].contentId
    expect(SnapshotSchema.safeParse(data).success).toBe(false)
  })
  it('rejects fractured cluster-aware groups', () => {
    const data = structuredClone(fixture)
    data.splits[2].assignments[1].splits = ['test']
    expect(SnapshotSchema.safeParse(data).success).toBe(false)
  })
  it('rejects unverified sequence identities', () => {
    const data = structuredClone(fixture)
    data.clusterings[0].points[0].sequenceId = 'invented-sequence'
    expect(SnapshotSchema.safeParse(data).success).toBe(false)
  })
  it('retains historical overlaps and separates noise allocation groups', async () => {
    const data = await new MockDataAdapter().getSnapshot()
    expect(
      data.splits[0].assignments.filter((a) => a.splits.length > 1),
    ).toHaveLength(12)
    const noise = data.clusterings[0].points.filter((p) => p.clusterId === -1)
    expect(new Set(noise.map((p) => p.groupId)).size).toBe(noise.length)
  })
  it('rejects missing split coverage and invalid nonhistorical assignments', () => {
    const data = structuredClone(fixture)
    data.splits[1].assignments.pop()
    expect(SnapshotSchema.safeParse(data).success).toBe(false)
    const other = structuredClone(fixture)
    other.splits[1].assignments[0].splits = ['train', 'test']
    expect(SnapshotSchema.safeParse(other).success).toBe(false)
  })
})
