"""Diagnostic registration and temporal models; never certify native identity.

All spatial fits use fixed disjoint fitting/evaluation tiles outside a conservative
HUD exclusion. This exclusion is diagnostic, not a validated removal mask. Neither
registration nor agreement with a temporal model creates a same-frame relation.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, map_coordinates
from scipy.optimize import least_squares

SIZE = (320, 180)


def gray_and_support(image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.asarray(
        image.convert("RGB").resize(SIZE, Image.Resampling.BILINEAR), dtype=float
    )
    gray = (
        np.asarray(Image.fromarray(rgb.astype(np.uint8)).convert("L"), dtype=float)
        / 255
    )
    y, x = np.indices(gray.shape)
    # Fixed border and crosshair exclusion; no pair-dependent error masking.
    support = (x >= 32) & (x < 288) & (y >= 32) & (y < 144)
    support &= ~((abs(x - 160) < 24) & (abs(y - 90) < 24))
    green = (rgb[..., 1] > rgb[..., 0] * 1.25 + 10) & (
        rgb[..., 1] > rgb[..., 2] * 1.25 + 10
    )
    support &= ~binary_dilation(green, iterations=3)
    return gray, support


def warp(array: np.ndarray, parameters) -> np.ndarray:
    """Inverse sampling: dx/dy at 320x180, uniform scale, angle in radians.

    This is source sampling coordinate = scale*rotation*(target-center)+center+
    translation. It is deliberately not a YOLO box transform or a crop estimate.
    """
    dx, dy, scale, angle = parameters
    y, x = np.indices(array.shape, dtype=float)
    cy, cx = (np.asarray(array.shape) - 1) / 2
    u, v = x - cx, y - cy
    sx = scale * (np.cos(angle) * u - np.sin(angle) * v) + cx + dx
    sy = scale * (np.sin(angle) * u + np.cos(angle) * v) + cy + dy
    return map_coordinates(array, [sy, sx], order=1, mode="constant", cval=np.nan)


def metrics(a, b, mask) -> dict:
    good = mask & np.isfinite(a) & np.isfinite(b)
    x, y = a[good], b[good]
    if len(x) < 100:
        return dict(pixels=int(len(x)), ncc=None, mae_255=None, rmse_255=None)
    centered_x, centered_y = x - x.mean(), y - y.mean()
    norm = float(np.linalg.norm(centered_x) * np.linalg.norm(centered_y))
    return dict(
        pixels=int(len(x)),
        ncc=float(centered_x @ centered_y / norm) if norm > 1e-12 else None,
        mae_255=float(np.mean(abs(x - y)) * 255),
        rmse_255=float(np.sqrt(np.mean((x - y) ** 2)) * 255),
    )


def register_pair(source: Image.Image, target: Image.Image) -> dict:
    """Compare declared-size baseline, translation and similarity on held-out tiles.

    Bounds are +/-12 analysis pixels, scale .94..1.06 and angle +/-3 degrees.
    Diagnostics are invalid for a different aspect ratio or EXIF orientation.
    Pilot thresholds (texture >=5 gray levels, NCC >=.95, MAE <=12) screen
    geometric compatibility only, not calibrated accuracy or native identity.
    """
    if source.getexif().get(274, 1) != 1 or target.getexif().get(274, 1) != 1:
        return dict(
            estimated=False,
            reason="unsupported_orientation",
            geometry_screen_pass=False,
        )
    if source.width * target.height != source.height * target.width:
        return dict(
            estimated=False,
            reason="aspect_ratio_requires_separate_crop_model",
            geometry_screen_pass=False,
        )
    a, ma = gray_and_support(source)
    b, mb = gray_and_support(target)
    mask = ma & mb
    y, x = np.indices(a.shape)
    fit = mask & (((x // 16 + y // 16) % 2) == 0)
    evaluation = mask & ~fit
    if mask.sum() < 3000:
        return dict(
            estimated=False,
            reason="insufficient_diagnostic_support",
            geometry_screen_pass=False,
        )
    texture = [float(a[mask].std() * 255), float(b[mask].std() * 255)]
    base = np.array([0.0, 0.0, 1.0, 0.0])
    transforms = []
    for kind, indices, low, high in (
        ("full_frame_resize", [], [], []),
        ("translation", [0, 1], [-12, -12], [12, 12]),
        (
            "similarity",
            [0, 1, 2, 3],
            [-12, -12, 0.94, -np.pi / 60],
            [12, 12, 1.06, np.pi / 60],
        ),
    ):
        params = base.copy()
        success, boundary = True, False
        if indices and min(texture) >= 5:

            def residual(values, active_indices=indices):
                p = base.copy()
                p[active_indices] = values
                delta = warp(a, p)[fit] - b[fit]
                return np.nan_to_num(delta, nan=1.0)

            result = least_squares(
                residual,
                params[indices],
                bounds=(low, high),
                diff_step=1e-3,
                max_nfev=50,
                loss="soft_l1",
                f_scale=0.03,
            )
            params[indices] = result.x
            success = bool(result.success)
            boundary = bool(
                np.any(np.isclose(result.x, low, atol=0.001))
                or np.any(np.isclose(result.x, high, atol=0.001))
            )
        warped = warp(a, params)
        transforms.append(
            dict(
                model=kind,
                inverse_sampling_parameters=params.tolist(),
                fit=metrics(warped, b, fit),
                held_out=metrics(warped, b, evaluation),
                converged=success,
                at_bound=boundary,
            )
        )
    # Compare every fitted transform on the SAME surviving source/target support.
    # Otherwise a transform could improve its score by moving difficult pixels
    # into an excluded region. Support never depends on pixel residual magnitude.
    common = mask.copy()
    for transform in transforms:
        projected = warp(ma.astype(float), transform["inverse_sampling_parameters"])
        common &= np.isfinite(projected) & (projected > 0.999)
    fit &= common
    evaluation &= common
    if evaluation.sum() < 1000:
        return dict(
            estimated=False,
            reason="insufficient_common_support",
            geometry_screen_pass=False,
        )
    for transform in transforms:
        projected = warp(a, transform["inverse_sampling_parameters"])
        transform["fit"] = metrics(projected, b, fit)
        transform["held_out"] = metrics(projected, b, evaluation)
    eligible = [
        r
        for r in transforms
        if r["converged"] and not r["at_bound"] and r["held_out"]["mae_255"] is not None
    ]
    chosen = min(eligible, key=lambda r: r["held_out"]["mae_255"])
    m = chosen["held_out"]
    passes = bool(
        min(texture) >= 5
        and m["ncc"] is not None
        and m["ncc"] >= 0.95
        and m["mae_255"] <= 12
    )
    return dict(
        estimated=True,
        analysis_size=list(SIZE),
        source_size=list(source.size),
        target_size=list(target.size),
        texture_std_255=texture,
        support_fraction=float(mask.mean()),
        mask_validated=False,
        common_support_fraction=float(common.mean()),
        evaluation_scope="validation_tiles_for_model_comparison_not_independent_test",
        fit_pixels=int(fit.sum()),
        evaluation_pixels=int(evaluation.sum()),
        transforms=transforms,
        selected_model=chosen["model"],
        geometry_screen_pass=passes,
        reason="geometric_compatibility_only"
        if passes
        else "fails_pilot_geometry_screen",
        frame_identity_verified=False,
        human_validated=False,
    )


def temporal_fit(x, y) -> dict:
    """Median-pair-slope model on alternating anchors; never infer identity."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 4 or not np.isfinite(x).all() or not np.isfinite(y).all():
        return dict(estimable=False, reason="fewer_than_four_finite_anchors")
    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]
    train = np.arange(len(x)) % 2 == 0
    tx, ty = x[train], y[train]
    slopes = [
        (ty[j] - ty[i]) / (tx[j] - tx[i])
        for i in range(len(tx))
        for j in range(i + 1, len(tx))
        if tx[j] != tx[i]
    ]
    if not slopes:
        return dict(estimable=False, reason="no_distinct_training_indices")
    slope = float(np.median(slopes))
    offset = float(np.median(ty - slope * tx))
    residual = y - (slope * x + offset)
    return dict(
        estimable=True,
        slope=slope,
        offset=offset,
        anchors=len(x),
        fit_indices=order[train].tolist(),
        held_out_indices=order[~train].tolist(),
        residuals=residual.tolist(),
        held_out_median_abs_error=float(np.median(abs(residual[~train]))),
        held_out_max_abs_error=float(max(abs(residual[~train]))),
        interpretation="candidate_model_not_verified_alignment",
    )
