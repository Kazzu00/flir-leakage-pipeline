"""Geometry proposals are not annotation approval or temporal identity."""

import numpy as np

from flir_pipeline.data.no_hud_alignment import SIZE


def transform_boxes(boxes, inverse_sampling_parameters):
    """Invert the existing registration's target->source sampling transform.

    Use pixel-centre resize coordinates. Return axis-aligned target envelopes,
    never clipped boxes. Rotation envelopes still require object-level review.
    """
    p = np.asarray(inverse_sampling_parameters, dtype=float)
    if p.shape != (4,) or not np.isfinite(p).all():
        raise ValueError("Non-finite or malformed transformation")
    dx, dy, scale, angle = p
    if not 0.94 <= scale <= 1.06 or abs(angle) > np.pi / 60 + 1e-9:
        raise ValueError("Degenerate or unsupported transformation")
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    size = np.array(SIZE)
    center = (size - 1) / 2
    transformed = []
    for c, x, y, w, h in boxes:
        values = np.asarray([x, y, w, h], dtype=float)
        if not np.isfinite(values).all() or min(w, h) <= 0:
            raise ValueError("Invalid original box")
        corners = np.array(
            [
                [x - w / 2, y - h / 2],
                [x + w / 2, y - h / 2],
                [x + w / 2, y + h / 2],
                [x - w / 2, y + h / 2],
            ]
        )
        if (corners < 0).any() or (corners > 1).any():
            raise ValueError("Original box out of bounds")
        source = corners * size - 0.5
        target = ((source - center - [dx, dy]) @ rotation / scale + center + 0.5) / size
        low, high = target.min(axis=0), target.max(axis=0)
        if (low < -1e-12).any() or (high > 1 + 1e-12).any() or (high - low <= 0).any():
            raise ValueError("Transformed box out of bounds; clipping forbidden")
        cx, cy = (low + high) / 2
        width, height = high - low
        transformed.append([int(c), float(cx), float(cy), float(width), float(height)])
    return transformed
