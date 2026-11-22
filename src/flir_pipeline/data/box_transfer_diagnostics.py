"""Object-region diagnostics, not automatic identity or annotation approval."""

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates

from flir_pipeline.data.no_hud_alignment import SIZE, metrics


def box_diagnostics(source, target, boxes, parameters):
    """Compare original object regions after the declared inverse sampling map.

    No residual-dependent masking: report the entire object rectangle, including
    modified HUD pixels. Photometric agreement cannot certify temporal identity.
    """
    a = (
        np.asarray(
            source.convert("L").resize(target.size, Image.Resampling.BILINEAR),
            dtype=float,
        )
        / 255
    )
    b = np.asarray(target.convert("L"), dtype=float) / 255
    dx, dy, scale, angle = parameters
    h, w = b.shape
    y, x = np.indices((h, w), dtype=float)
    # Pixel-center conversion is consistent with registration at SIZE.
    u, v = (x + 0.5) * SIZE[0] / w - SIZE[0] / 2, (y + 0.5) * SIZE[1] / h - SIZE[1] / 2
    sx = (
        scale * (np.cos(angle) * u - np.sin(angle) * v) + SIZE[0] / 2 + dx
    ) * w / SIZE[0] - 0.5
    sy = (
        scale * (np.sin(angle) * u + np.cos(angle) * v) + SIZE[1] / 2 + dy
    ) * h / SIZE[1] - 0.5
    warped = map_coordinates(a, [sy, sx], order=1, mode="constant", cval=np.nan)
    result = []
    for c, cx, cy, bw, bh in boxes:
        mask = (
            (x >= (cx - bw / 2) * w)
            & (x < (cx + bw / 2) * w)
            & (y >= (cy - bh / 2) * h)
            & (y < (cy + bh / 2) * h)
        )
        good = mask & np.isfinite(warped)
        m = metrics(warped, b, mask)
        m.update(
            class_id=int(c),
            valid_fraction=float(good.sum() / max(mask.sum(), 1)),
            source_texture_std_255=float(warped[good].std() * 255)
            if good.any()
            else None,
            changed_over_12_fraction=float(
                (abs(warped[good] - b[good]) * 255 > 12).mean()
            )
            if good.any()
            else None,
            object_equivalence_verified=False,
        )
        result.append(m)
    return result
