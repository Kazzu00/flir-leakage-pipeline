"""Content-only retrieval primitives; nominal filenames/times are never inputs.

Sparse pixel digests are retrieval keys only. Acceptance requires equality of
the complete decoded RGB raster; all matching occurrences must be retained.
Approximate scores never confer VIDEO_VERIFIED or sequence membership.
"""

from __future__ import annotations

import hashlib

import numpy as np


def pixel_key(rgb: np.ndarray) -> str:
    """Fast candidate lookup invariant to packaging, not to raster transforms."""
    shape = np.asarray(rgb.shape, dtype="<i8").tobytes()
    return hashlib.sha256(shape + rgb[::16, ::16].tobytes()).hexdigest()


def raster_digest(rgb: np.ndarray) -> str:
    """Hash full RGB raster and shape, distinguishing it from encoded file bytes."""
    shape = np.asarray(rgb.shape, dtype="<i8").tobytes()
    return hashlib.sha256(shape + rgb.tobytes()).hexdigest()


def descriptor(rgb: np.ndarray) -> np.ndarray:
    """Interior low-resolution luminance, normalized; retrieval only.

    Fixed interior crop reduces HUD influence without learning from labels,
    splits or source filenames. It does not prove absence of a visual match.
    """
    import cv2

    h, w = rgb.shape[:2]
    crop = rgb[h // 10 : h * 9 // 10, w // 10 : w * 9 // 10]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    vector = cv2.resize(gray, (32, 18), interpolation=cv2.INTER_AREA)
    vector = vector.astype(np.float32).ravel()
    vector -= vector.mean()
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else np.zeros_like(vector)


def temporal_state(exact_occurrences: list, *, search_complete: bool) -> str:
    """Only a completed full scan can establish uniqueness of an exact raster."""
    if len(exact_occurrences) > 1:
        return "VIDEO_AMBIGUOUS"
    if len(exact_occurrences) == 1 and search_complete:
        return "VIDEO_VERIFIED"
    # No exact equality does not mean an approximate match is absent.
    return "NOT_VERIFIED"


def diverse_candidates(scores, rows, *, limit=8, separation_seconds=3.0):
    """Retain different videos/time modes; no nominal-index restriction."""
    selected = []
    for index in np.argsort(-scores, kind="stable"):
        row = rows[int(index)]
        if any(
            row["video_sha256"] == rows[j]["video_sha256"]
            and abs(row["second"] - rows[j]["second"]) < separation_seconds
            for j in selected
        ):
            continue
        selected.append(int(index))
        if len(selected) == limit:
            break
    return selected
