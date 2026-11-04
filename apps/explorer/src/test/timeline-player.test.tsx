import { act, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type {
  OrganizationMedia,
  OrganizationTimeline,
} from '@/contracts/organization-evidence-v2'
import { TimelinePlayer } from '@/features/organization/TimelinePlayer'
import { organizationMediaUrl } from '@/data/organization-media'

const requests: ControlledImage[] = []

class ControlledImage {
  src = ''
  onload: (() => void) | null = null
  onerror: (() => void) | null = null
  resolve!: () => void
  reject!: () => void
  decoded = new Promise<void>((resolve, reject) => {
    this.resolve = resolve
    this.reject = () => reject(new Error('synthetic decode failure'))
  })
  constructor() {
    requests.push(this)
  }
  decode() {
    return this.decoded
  }
  removeAttribute() {
    this.src = ''
  }
}

function fixture(count = 3, id = 'synthetic-timeline') {
  const timeline: OrganizationTimeline = {
    timeline_id: id,
    source_video_id: null,
    source_archive: null,
    inferred_family: null,
    temporal_source: 'unknown',
    points: Array.from({ length: count }, (_, index) => ({
      content_id: `${id}-content-${index}`,
      record_ids: [],
      frame_index: index,
      timestamp_seconds: null,
      preview_key: `${id}-${index}`,
    })),
  }
  const media: OrganizationMedia[] = timeline.points.map((point) => ({
    preview_key: point.preview_key,
    content_id: point.content_id,
    relative_path: `${point.preview_key}.jpg`,
    width: 640,
    height: 480,
    checksum: null,
    media_available: true,
    unavailable_reason: null,
  }))
  return { timeline, media }
}

async function decode(index: number, success = true) {
  await act(async () => {
    if (success) requests[index].resolve()
    else requests[index].reject()
  })
}

async function tick(ms = 250) {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(ms)
  })
}

describe('TimelinePlayer presentation buffer', () => {
  beforeEach(() => {
    requests.length = 0
    vi.useFakeTimers()
    vi.stubGlobal('Image', ControlledImage)
  })
  afterEach(() => {
    vi.useRealTimers()
    vi.unstubAllGlobals()
  })

  it('waits for decode without skipping observed points or retaining the old image', async () => {
    const data = fixture()
    render(<TimelinePlayer {...data} />)
    await decode(0)
    const first = screen.getByRole('img')
    fireEvent.click(screen.getByRole('button', { name: 'PLAY' }))
    await tick(2000)
    expect(screen.getByText('1 / 3')).toBeVisible()
    expect(screen.getByRole('img')).toBe(first)
    await decode(1)
    expect(screen.getByText('2 / 3')).toBeVisible()
    expect(screen.getByRole('img')).not.toBe(first)
    expect(screen.getByRole('img')).toHaveAttribute(
      'src',
      organizationMediaUrl(data.media[1].relative_path!),
    )
    await tick(2000)
    expect(screen.getByText('2 / 3')).toBeVisible()
    await decode(2)
    expect(screen.getByText('3 / 3')).toBeVisible()
  })

  it('scrubs immediately, pauses, and hides the old bitmap during preparation', async () => {
    render(<TimelinePlayer {...fixture()} />)
    await decode(0)
    fireEvent.click(screen.getByRole('button', { name: 'PLAY' }))
    fireEvent.change(screen.getByRole('slider'), { target: { value: '2' } })
    expect(screen.getByText('3 / 3')).toBeVisible()
    expect(screen.getByRole('button', { name: 'PLAY' })).toHaveAttribute(
      'aria-pressed',
      'false',
    )
    expect(screen.queryByRole('img')).toBeNull()
    expect(screen.getByText('Loading preview…')).toBeVisible()
    await decode(2)
    expect(screen.getByRole('img')).toHaveAttribute('alt', 'Timeline frame 2')
    await tick(1000)
    expect(screen.getByText('3 / 3')).toBeVisible()
  })

  it('resets on timeline change and ignores completion from the old timeline', async () => {
    const { rerender } = render(<TimelinePlayer {...fixture()} />)
    await decode(0)
    fireEvent.click(screen.getByRole('button', { name: 'Next frame' }))
    fireEvent.click(screen.getByRole('button', { name: 'PLAY' }))
    rerender(<TimelinePlayer {...fixture(2, 'other-timeline')} />)
    expect(screen.getByText('1 / 2')).toBeVisible()
    expect(screen.getByRole('button', { name: 'PLAY' })).toHaveAttribute(
      'aria-pressed',
      'false',
    )
    await decode(1)
    await tick(1000)
    expect(screen.getByText('1 / 2')).toBeVisible()
    expect(screen.queryByRole('img')).toBeNull()
    await decode(3)
    expect(screen.getByRole('img')).toHaveAttribute('alt', 'Timeline frame 0')
  })

  it('preserves evidence with absent media and after a failed decode', async () => {
    const data = fixture()
    data.media[0].media_available = false
    data.media[0].relative_path = null
    render(<TimelinePlayer {...data} />)
    expect(screen.getByText('Preview unavailable')).toBeVisible()
    fireEvent.click(screen.getByRole('button', { name: 'PLAY' }))
    await tick()
    await decode(0, false)
    expect(screen.getByText('2 / 3')).toBeVisible()
    expect(screen.getByText('Preview unavailable')).toBeVisible()
    await decode(1)
    await tick()
    expect(screen.getByText('3 / 3')).toBeVisible()
  })

  it('limits preload to the selected point plus 24 points and prioritizes a manual jump', async () => {
    render(<TimelinePlayer {...fixture(700)} />)
    expect(requests).toHaveLength(25)
    fireEvent.change(screen.getByRole('slider'), { target: { value: '400' } })
    expect(requests[25].src).toContain('synthetic-timeline-400.jpg')
    expect(requests.filter((image) => image.src)).toHaveLength(25)
    expect(screen.getByText('401 / 700')).toBeVisible()
  })

  it('cancels a pending playback advance when paused or unmounted', async () => {
    const { unmount } = render(<TimelinePlayer {...fixture()} />)
    await decode(0)
    fireEvent.click(screen.getByRole('button', { name: 'PLAY' }))
    await tick()
    fireEvent.click(screen.getByRole('button', { name: 'PAUSE' }))
    await decode(1)
    expect(screen.getByText('1 / 3')).toBeVisible()
    fireEvent.click(screen.getByRole('button', { name: 'PLAY' }))
    unmount()
    await tick(1000)
    expect(requests.every((image) => image.src === '')).toBe(true)
    expect(vi.getTimerCount()).toBe(0)
  })
})
