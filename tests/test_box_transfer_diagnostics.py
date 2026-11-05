"""Synthetic object-region evidence must not become automatic approval."""

import numpy as np
from PIL import Image

from flir_pipeline.data.box_transfer import transform_boxes
from flir_pipeline.data.box_transfer_diagnostics import box_diagnostics


def test_identity_has_zero_error_but_never_approves():
    rng = np.random.default_rng(42)
    image = Image.fromarray(rng.integers(0, 255, (180, 320), dtype=np.uint8))
    result = box_diagnostics(image, image, [(0, 0.5, 0.5, 0.4, 0.4)], [0, 0, 1, 0])[0]
    assert result["mae_255"] < 1e-10
    assert result["ncc"] > 0.999999
    assert result["valid_fraction"] == 1
    assert not result["object_equivalence_verified"]


def test_changed_object_is_not_hidden_by_mask():
    a = np.full((180, 320), 50, dtype=np.uint8)
    b = a.copy()
    b[54:126, 96:224] = 200
    result = box_diagnostics(
        Image.fromarray(a), Image.fromarray(b), [(0, 0.5, 0.5, 0.4, 0.4)], [0, 0, 1, 0]
    )[0]
    assert result["mae_255"] > 149
    assert result["changed_over_12_fraction"] == 1
    assert not result["object_equivalence_verified"]


def test_inverse_translation_and_box_coordinates_agree():
    rng = np.random.default_rng(7)
    source = rng.integers(0, 255, (180, 320), dtype=np.uint8)
    target = np.zeros_like(source)
    target[:, :-3] = source[:, 3:]
    params = [3, 0, 1, 0]
    boxes = transform_boxes([(1, 0.5, 0.5, 0.3, 0.3)], params)
    result = box_diagnostics(
        Image.fromarray(source), Image.fromarray(target), boxes, params
    )[0]
    assert result["mae_255"] < 1e-9
    assert result["valid_fraction"] == 1
