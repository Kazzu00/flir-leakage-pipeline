"""Synthetic geometry invariants: inverse direction, resize, no clipping."""

import numpy as np
import pytest

from flir_pipeline.data.box_transfer import transform_boxes


def test_identity_and_full_frame_resize_preserve_normalized_boxes():
    boxes = [[1, 0.5, 0.4, 0.2, 0.1]]
    np.testing.assert_allclose(transform_boxes(boxes, [0, 0, 1, 0]), boxes)


def test_inverse_translation_direction():
    result = transform_boxes([[1, 0.5, 0.5, 0.2, 0.2]], [3.2, 1.8, 1, 0])[0]
    np.testing.assert_allclose(result, [1, 0.49, 0.49, 0.2, 0.2])


@pytest.mark.parametrize(
    "params", [[0, 0, 0, 0], [0, 0, float("nan"), 0], [0, 0, 1, 2], [0, 0, -1, 0]]
)
def test_degenerate_transforms_rejected(params):
    with pytest.raises(ValueError):
        transform_boxes([[0, 0.5, 0.5, 0.1, 0.1]], params)


def test_outside_boxes_rejected_never_clipped():
    with pytest.raises(ValueError, match="clipping"):
        transform_boxes([[0, 0.05, 0.5, 0.1, 0.1]], [12, 0, 1, 0])
