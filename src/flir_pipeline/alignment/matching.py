"""CPU retrieval with independent photometric and geometric support, media PTS."""

import time
from fractions import Fraction

import numpy as np

from .inventory import sha
from .models import Match


class BudgetExceeded(ValueError):
    pass


class Decoder:
    """Backward keyframe seek then sequential decode, never FPS-derived timestamps."""

    def __init__(self, path, max_frames=90000, max_seconds=1800):
        import av

        self.container = av.open(str(path))
        self.stream = self.container.streams.video[0]
        self.stream.thread_count = 1
        self.decoded = 0
        self.sampled = 0
        self.seek_log = []
        self.max_frames = max_frames
        self.max_seconds = max_seconds
        self.started = time.monotonic()
        self.metadata = {
            "codec": self.stream.codec_context.name,
            "time_base": str(self.stream.time_base),
            "average_rate_reported": str(self.stream.average_rate),
            "width": self.stream.width,
            "height": self.stream.height,
            "duration_seconds": float(self.stream.duration * self.stream.time_base)
            if self.stream.duration
            else None,
            "timestamp_basis": "decoded_frame.pts * decoded_frame.time_base",
            "pixel_hash_basis": "RGB24 decoded and resized to 384x216 with PyAV bilinear; not original JPEG bytes",
        }

    def close(self):
        self.container.close()

    def frames(self, start, end, step):
        start = max(0, start)
        duration = self.metadata["duration_seconds"]
        if duration is not None:
            end = min(end, duration)
        if start > end:
            return []
        self.container.seek(
            int(Fraction(str(start)) / self.stream.time_base),
            stream=self.stream,
            backward=True,
            any_frame=False,
        )
        result, first, target = [], None, start
        for frame in self.container.decode(self.stream):
            if (
                self.decoded >= self.max_frames
                or time.monotonic() - self.started > self.max_seconds
            ):
                raise BudgetExceeded("Decoded-frame or elapsed-time budget exhausted")
            self.decoded += 1
            if frame.pts is None:
                raise ValueError("Decoded frame has no verifiable PTS")
            second = float(frame.pts * frame.time_base)
            if first is None:
                first = second
            if second > end + 1e-9:
                break
            if second + 1e-9 < target:
                continue
            rgb = frame.reformat(width=384, height=216, format="rgb24").to_ndarray()
            result.append(
                {
                    "rgb": rgb,
                    "pts": frame.pts,
                    "time_base": str(frame.time_base),
                    "second": second,
                    "duration_seconds": float(frame.duration * frame.time_base)
                    if frame.duration
                    else None,
                    "decoded_rgb_sha256": sha(rgb.tobytes()),
                    "requested_seek_second": start,
                    "first_decoded_second": first,
                }
            )
            self.sampled += 1
            target = second + step if step else second
        self.seek_log.append(
            {
                "requested_start": start,
                "requested_end": end,
                "requested_step": step,
                "first_decoded_second": first,
                "returned_pts": [f["pts"] for f in result],
                "effective_steps_seconds": sorted(
                    {
                        round(b["second"] - a["second"], 9)
                        for a, b in zip(result, result[1:], strict=False)
                    }
                ),
            }
        )
        return result


def features(rgb):
    import cv2

    cv2.setNumThreads(1)
    cv2.setRNGSeed(20261103)
    image = cv2.resize(np.asarray(rgb), (384, 216), interpolation=cv2.INTER_AREA)
    # Fixed interior mask reduces HUD support. Not fitted using splits or labels.
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)[22:194, 38:346]
    norm = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8)).apply(gray)
    points, descriptors = cv2.SIFT_create(nfeatures=500).detectAndCompute(norm, None)
    tiny = cv2.resize(norm, (9, 8))
    return {
        "norm": norm,
        "points": points,
        "descriptors": descriptors,
        "hash": tiny[:, 1:] > tiny[:, :-1],
    }


def support(query, frame, config, stage):
    import cv2

    candidate = features(frame["rgb"])
    a, b = query["norm"].astype(float).ravel(), candidate["norm"].astype(float).ravel()
    correlation = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 0.0
    good, inliers, coverage, residual = [], 0, 0.0, None
    if (
        query["descriptors"] is not None
        and candidate["descriptors"] is not None
        and len(candidate["descriptors"]) >= 2
    ):
        pairs = cv2.BFMatcher().knnMatch(
            query["descriptors"], candidate["descriptors"], k=2
        )
        good = [m for m, n in pairs if m.distance < 0.75 * n.distance]
        if len(good) >= 4:
            source = np.float32([query["points"][m.queryIdx].pt for m in good])
            target = np.float32([candidate["points"][m.trainIdx].pt for m in good])
            cv2.setRNGSeed(config.seed)
            homography, mask = cv2.findHomography(source, target, cv2.RANSAC, 3.0)
            if homography is not None and mask is not None:
                keep = mask.ravel().astype(bool)
                inliers = int(keep.sum())
                if inliers >= 4:
                    coverage = (
                        min(
                            cv2.contourArea(cv2.convexHull(source[keep])),
                            cv2.contourArea(cv2.convexHull(target[keep])),
                        )
                        / query["norm"].size
                    )
                    projected = cv2.perspectiveTransform(
                        source.reshape(-1, 1, 2), homography
                    ).reshape(-1, 2)
                    residual = float(
                        np.median(
                            np.linalg.norm(projected[keep] - target[keep], axis=1)
                        )
                    )
    ratio = inliers / len(good) if good else 0.0
    return Match(
        **{k: v for k, v in frame.items() if k != "rgb"},
        correlation=correlation,
        dhash_distance=int(np.count_nonzero(query["hash"] != candidate["hash"])),
        matches=len(good),
        inliers=inliers,
        inlier_ratio=ratio,
        coverage=coverage,
        reprojection_median_px=residual,
        supported=inliers >= config.min_inliers
        and ratio >= config.min_inlier_ratio
        and coverage >= config.min_coverage,
        search_stage=stage,
    ).model_dump()


def ranking(row):
    return (
        row["supported"],
        row["inliers"],
        row["inlier_ratio"],
        row["correlation"],
        -row["pts"],
    )


def classify(matches, controls):
    supported = sorted(
        [m for m in matches if m["supported"]], key=ranking, reverse=True
    )
    if not supported:
        return (
            "not_found",
            "No candidate passed the uncalibrated geometric support gates",
            None,
        )
    best = supported[0]
    alternatives = [
        m
        for m in supported[1:] + controls
        if m["supported"]
        and abs(m["second"] - best["second"]) > 2
        and m["inliers"] >= best["inliers"] * 0.8
    ]
    near = [
        m["second"]
        for m in supported
        if abs(m["second"] - best["second"]) <= 2
        and m["inliers"] >= best["inliers"] * 0.8
    ]
    interval = [min(near), max(near)]
    if alternatives:
        return (
            "ambiguous",
            "Competitive support at distinct times; no unique correspondence established",
            interval,
        )
    return (
        "candidate",
        "Geometric retrieval support only; human identity review required",
        interval,
    )


def search(image, index, decoder, config):
    query = features(np.asarray(image))
    center = config.offset_hypothesis + index
    scored, controls, frames, intervals = {}, [], {}, []

    def scan(start, end, step, stage, control=False):
        intervals.append([max(0, start), end])
        for frame in decoder.frames(start, end, step):
            row = support(query, frame, config, stage)
            frames[row["pts"]] = frame
            if control:
                controls.append(row)
            else:
                scored[row["pts"]] = row

    scan(
        center - config.local_radius,
        center + config.local_radius,
        config.coarse_step,
        "local",
    )
    # Temporal controls are NOT ground-truth negatives of region identity.
    for second in (max(0, center - 120), center + 120, 0):
        scan(second, second + 0.1, 0.1, "outside_window_control", True)
    if not any(row["supported"] for row in scored.values()):
        scan(
            center - config.expanded_radius,
            center + config.expanded_radius,
            config.expanded_step,
            "expanded",
        )
    for control in controls:
        if control["supported"]:
            scored[control["pts"]] = control
    coarse = sorted(scored.values(), key=ranking, reverse=True)
    # Refine two distinct temporal modes, not just one convenient maximum.
    centers = []
    for row in coarse:
        if all(abs(row["second"] - t) > 1 for t in centers):
            centers.append(row["second"])
        if len(centers) == 2:
            break
    for second in centers:
        scan(
            second - config.refine_radius,
            second + config.refine_radius,
            0,
            "native_pts_refinement",
        )
    all_matches = sorted(scored.values(), key=ranking, reverse=True)
    state, reason, interval = classify(all_matches, controls)
    # Preserve distinct temporal alternatives, then fill with native-rate neighbors.
    top = []
    for row in all_matches:
        if all(abs(row["second"] - m["second"]) > 1 for m in top):
            top.append(row)
        if len(top) == config.top_k:
            break
    top += [row for row in all_matches if row not in top][: config.top_k - len(top)]
    return dict(
        status=state,
        reason=reason,
        top_k=top,
        controls=controls,
        searched_intervals=intervals,
        candidate_interval=interval,
    ), frames
