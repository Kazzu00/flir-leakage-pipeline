"""Offline spatial/temporal diagnostics do not adjudicate native identity."""

import numpy as np
import pytest
from PIL import Image
from scipy.ndimage import gaussian_filter

from flir_pipeline.data.no_hud_alignment import (
    gray_and_support,
    metrics,
    register_pair,
    temporal_fit,
    warp,
)


def texture(seed=2):
    a = gaussian_filter(np.random.default_rng(seed).normal(size=(180, 320)), 2)
    a = (a - a.min()) / (a.max() - a.min())
    return Image.fromarray(np.round(a * 255).astype(np.uint8)).convert("RGB")


def test_identity_is_geometric_only():
    image = texture()
    result = register_pair(image, image)
    assert result["geometry_screen_pass"]
    assert not result["frame_identity_verified"] and not result["human_validated"]
    assert len({r["held_out"]["pixels"] for r in result["transforms"]}) == 1
    assert result["transforms"][0]["held_out"]["mae_255"] == 0


@pytest.mark.parametrize("params", [[3.0, -2.0, 1.0, 0.0], [2.0, -1.0, 1.015, 0.01]])
def test_recovers_synthetic_geometry_without_identity_claim(params):
    image = texture()
    a, _ = gray_and_support(image)
    b = np.nan_to_num(warp(a, params), nan=0.5)
    target = Image.fromarray(np.round(b * 255).astype(np.uint8)).convert("RGB")
    result = register_pair(image, target)
    best = min(result["transforms"], key=lambda r: r["held_out"]["mae_255"])
    assert best["held_out"]["mae_255"] < 2
    assert result["geometry_screen_pass"]
    assert not result["frame_identity_verified"]


def test_constant_and_unrelated_images_do_not_pass():
    flat = Image.new("RGB", (320, 180), (128, 128, 128))
    assert not register_pair(flat, flat)["geometry_screen_pass"]
    assert not register_pair(texture(1), texture(7))["geometry_screen_pass"]


def test_crop_and_exif_are_not_silently_corrected():
    image = texture()
    assert not register_pair(image, image.crop((1, 0, 320, 180)))["estimated"]
    image.getexif()[274] = 6
    assert not register_pair(image, texture())["estimated"]


def test_temporal_model_learns_rate_and_offset_with_held_out_anchors():
    x = np.arange(20) * 11 + 3
    result = temporal_fit(x, 17 * x + 9)
    assert result["slope"] == 17 and result["offset"] == 9
    assert result["held_out_max_abs_error"] == 0
    assert not set(result["fit_indices"]) & set(result["held_out_indices"])
    assert result["interpretation"] == "candidate_model_not_verified_alignment"


def test_temporal_outlier_is_preserved_and_insufficient_anchors_refused():
    x = np.arange(10.0)
    y = x * 2 + 7
    y[3] += 100
    result = temporal_fit(x, y)
    assert result["held_out_max_abs_error"] == 100
    assert not temporal_fit([1, 2, 3], [1, 2, 3])["estimable"]
    assert not temporal_fit([1, 1, 1, 1], [1, 2, 3, 4])["estimable"]


def test_metrics_do_not_fabricate_correlation_for_flat_or_missing_support():
    a = np.zeros((20, 20))
    assert metrics(a, a, np.ones_like(a, dtype=bool))["ncc"] is None
    assert metrics(a, a, np.zeros_like(a, dtype=bool))["mae_255"] is None


def test_pilot_output_cannot_modify_prior_publications(tmp_path):
    import importlib.util
    from pathlib import Path

    script = (
        Path(__file__).resolve().parents[1] / "scripts/build_no_hud_alignment_pilot.py"
    )
    spec = importlib.util.spec_from_file_location("alignment_script", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = tmp_path / "original"
    original.mkdir()
    with pytest.raises(ValueError, match="overlaps"):
        module.safe_output(original / "new", original)
    with pytest.raises(ValueError, match="fresh"):
        module.safe_output(original)
    module.safe_output(tmp_path / "fresh", original)
