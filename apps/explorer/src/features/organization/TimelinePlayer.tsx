import { useEffect, useMemo, useState } from 'react'
import type {
  OrganizationMedia,
  OrganizationTimeline,
} from '@/contracts/organization-evidence-v2'
import { organizationMediaUrl } from '@/data/organization-media'

const PLAYBACK_RATES = [1, 2, 4, 8, 12]
const PRELOAD_AHEAD = 24

function createPreviewBuffer() {
  const entries = new Map<
    string,
    {
      ready: Promise<boolean>
      cancel: () => void
    }
  >()

  function prepare(url: string | null): Promise<boolean> {
    if (!url) return Promise.resolve(false)
    const cached = entries.get(url)
    if (cached) return cached.ready

    const image = new Image()
    let cancel = () => {}
    const ready = new Promise<boolean>((resolve) => {
      let settled = false
      const finish = (success: boolean) => {
        if (settled) return
        settled = true
        image.onload = null
        image.onerror = null
        resolve(success)
      }
      cancel = () => {
        finish(false)
        image.removeAttribute('src')
      }
      image.onerror = () => finish(false)
      image.onload = () => {
        if (typeof image.decode !== 'function') finish(true)
      }
      image.src = url
      if (typeof image.decode === 'function') {
        image.decode().then(
          () => finish(true),
          () => finish(false),
        )
      }
    })
    entries.set(url, { ready, cancel })
    return ready
  }

  return {
    prepare,
    retain(urls: Set<string>) {
      for (const [url, entry] of entries) {
        if (!urls.has(url)) {
          entry.cancel()
          entries.delete(url)
        }
      }
    },
    clear() {
      for (const entry of entries.values()) entry.cancel()
      entries.clear()
    },
  }
}

type TimelinePlayerProps = {
  timeline: OrganizationTimeline
  media: OrganizationMedia[]
}

export function TimelinePlayer({ timeline, media }: TimelinePlayerProps) {
  return (
    <TimelinePlayback
      key={timeline.timeline_id}
      timeline={timeline}
      media={media}
    />
  )
}

function TimelinePlayback({ timeline, media }: TimelinePlayerProps) {
  const [index, setIndex] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [fps, setFps] = useState(4)
  const [loop, setLoop] = useState(true)

  const mediaByPreview = useMemo(
    () => new Map(media.map((item) => [item.preview_key, item])),
    [media],
  )

  const frames = useMemo(() => {
    return timeline.points
      .map((point, sourceOrder) => {
        const preview = mediaByPreview.get(point.preview_key) ?? null
        return {
          point,
          sourceOrder,
          media: preview,
          url:
            preview?.media_available && preview.relative_path
              ? organizationMediaUrl(preview.relative_path)
              : null,
        }
      })
      .sort((a, b) => {
        const aFrame = a.point.frame_index
        const bFrame = b.point.frame_index

        if (aFrame !== null && bFrame !== null) {
          return aFrame - bFrame
        }

        if (aFrame !== null) return -1
        if (bFrame !== null) return 1

        const aTime = a.point.timestamp_seconds
        const bTime = b.point.timestamp_seconds

        if (aTime !== null && bTime !== null) {
          return aTime - bTime
        }

        return a.sourceOrder - b.sourceOrder
      })
  }, [mediaByPreview, timeline.points])

  const buffer = useMemo(() => createPreviewBuffer(), [])
  const [previewState, setPreviewState] = useState<{
    url: string | null
    buffer: typeof buffer
    ready: boolean
  } | null>(null)
  const currentIndex = Math.min(index, Math.max(0, frames.length - 1))
  const currentUrl = frames[currentIndex]?.url ?? null

  useEffect(() => () => buffer.clear(), [buffer, frames])

  useEffect(() => {
    let cancelled = false
    // Prioritize the selected point before filling the bounded forward window.
    void buffer.prepare(currentUrl).then((ready) => {
      if (!cancelled) setPreviewState({ url: currentUrl, buffer, ready })
    })
    const urls = new Set<string>()
    const count = Math.min(frames.length, PRELOAD_AHEAD + 1)
    for (let offset = 0; offset < count; offset++) {
      const position = currentIndex + offset
      if (!loop && position >= frames.length) break
      const url = frames[position % frames.length]?.url
      if (url) urls.add(url)
    }
    buffer.retain(urls)
    for (const url of urls) void buffer.prepare(url)
    return () => {
      cancelled = true
    }
  }, [buffer, currentIndex, currentUrl, frames, loop])

  useEffect(() => {
    if (!playing || frames.length <= 1) return

    let cancelled = false
    let timer: number | undefined
    void buffer.prepare(currentUrl).then(() => {
      if (cancelled) return
      timer = window.setTimeout(() => {
        const next = currentIndex + 1
        if (next >= frames.length && !loop) {
          setPlaying(false)
          return
        }
        const target = next % frames.length
        void buffer.prepare(frames[target].url).then((ready) => {
          if (cancelled) return
          // Advance evidence and readiness together; never catch up by skipping points.
          setPreviewState({ url: frames[target].url, buffer, ready })
          setIndex(target)
        })
      }, 1000 / fps)
    })

    return () => {
      cancelled = true
      window.clearTimeout(timer)
    }
  }, [playing, fps, loop, frames, buffer, currentIndex, currentUrl])

  if (!frames.length) {
    return (
      <div className="timeline-player">
        <p className="panel-body muted">
          No observed timeline points are available for playback.
        </p>
      </div>
    )
  }

  const current = frames[Math.min(index, frames.length - 1)]
  const currentMedia = current.media

  const prepared =
    previewState?.url === currentUrl && previewState.buffer === buffer
  const hasPreview = Boolean(currentUrl && prepared && previewState?.ready)
  const loadingPreview = Boolean(currentUrl && !prepared)

  function previousFrame() {
    setPlaying(false)
    setIndex((currentIndex) => Math.max(0, currentIndex - 1))
  }

  function nextFrame() {
    setPlaying(false)
    setIndex((currentIndex) => Math.min(frames.length - 1, currentIndex + 1))
  }

  return (
    <section className="timeline-player" aria-label="Timeline frame playback">
      <div className="timeline-player-stage">
        {hasPreview && currentUrl ? (
          <img
            key={currentUrl}
            src={currentUrl}
            alt={`Timeline frame ${current.point.frame_index ?? index}`}
            width={currentMedia?.width ?? undefined}
            height={currentMedia?.height ?? undefined}
            decoding="sync"
            onError={() =>
              setPreviewState({ url: currentUrl, buffer, ready: false })
            }
          />
        ) : (
          <div className="timeline-player-missing">
            <strong>
              {loadingPreview ? 'Loading preview…' : 'Preview unavailable'}
            </strong>
            <span>
              {loadingPreview
                ? 'Preparing the selected observed point.'
                : currentUrl
                  ? 'The exported preview could not be loaded or decoded.'
                  : (currentMedia?.unavailable_reason ??
                    'No exported preview is available for this point.')}
            </span>
          </div>
        )}
      </div>

      <div className="timeline-player-meta">
        <span>
          Frame <strong>{current.point.frame_index ?? 'unknown'}</strong>
        </span>

        <span>
          Point{' '}
          <strong>
            {index + 1} / {frames.length}
          </strong>
        </span>

        <span>
          Timestamp{' '}
          <strong>
            {current.point.timestamp_seconds === null
              ? 'unavailable'
              : `${current.point.timestamp_seconds.toFixed(3)} s`}
          </strong>
        </span>

        <span>
          Content <strong>{current.point.content_id}</strong>
        </span>
      </div>

      <input
        className="timeline-player-scrubber"
        type="range"
        min={0}
        max={Math.max(0, frames.length - 1)}
        value={index}
        aria-label="Timeline point"
        onChange={(event) => {
          setPlaying(false)
          setIndex(Number(event.target.value))
        }}
      />

      <div className="timeline-player-controls">
        <button
          type="button"
          onClick={previousFrame}
          disabled={index === 0}
          aria-label="Previous frame"
        >
          ◀
        </button>

        <button
          type="button"
          onClick={() => setPlaying((value) => !value)}
          aria-pressed={playing}
        >
          {playing ? 'PAUSE' : 'PLAY'}
        </button>

        <button
          type="button"
          onClick={nextFrame}
          disabled={index >= frames.length - 1}
          aria-label="Next frame"
        >
          ▶
        </button>

        <label className="timeline-player-rate">
          Display rate
          <select
            value={fps}
            onChange={(event) => setFps(Number(event.target.value))}
          >
            {PLAYBACK_RATES.map((rate) => (
              <option key={rate} value={rate}>
                {rate} {rate === 1 ? 'frame/s' : 'frames/s'}
              </option>
            ))}
          </select>
        </label>

        <label className="timeline-player-loop">
          <input
            type="checkbox"
            checked={loop}
            onChange={(event) => setLoop(event.target.checked)}
          />
          Loop
        </label>
      </div>

      <p className="timeline-player-caption">
        Playback reconstructs the exported timeline from observed frame
        evidence. Display rate is a presentation setting and does not imply the
        original capture frame rate. Timeline membership is not sequence ground
        truth.
      </p>
    </section>
  )
}
