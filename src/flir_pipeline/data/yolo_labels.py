"""Strict, non-mutating validation for classic YOLO label files."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class LabelValidationResult:
    """Non-mutating YOLO QA result, including errors and canonical box tuples.

    Empty labels are valid backgrounds. ``canonical`` sorts boxes and rounds to
    ten decimals for duplicate-annotation comparison; it never rewrites sources.
    Geometry validity is separate from syntax and normalized-coordinate validity.
    """

    label_empty: bool
    syntax_valid: bool
    normalized_values_valid: bool
    geometry_valid: bool
    label_valid: bool
    num_objects: int
    classes_present: list[int] = field(default_factory=list)
    bbox_area_mean: float | None = None
    bbox_area_min: float | None = None
    bbox_area_max: float | None = None
    bbox_width_mean: float | None = None
    bbox_height_mean: float | None = None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    canonical: tuple[tuple[int, float, float, float, float], ...] = ()


def _number(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("value is not finite")
    return parsed


def validate_label_text(text: str) -> LabelValidationResult:
    """Validate classic class/x-center/y-center/width/height lines without edits.

    Require integer nonnegative classes, finite normalized coordinates, positive
    size and corners inside the image. Slight corner overflow is still invalid,
    with a warning. Return QA/statistics, not a corrected label or a class ontology
    check; the five observed classes are a dataset finding, not a parser constant.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return LabelValidationResult(
            label_empty=True,
            syntax_valid=True,
            normalized_values_valid=True,
            geometry_valid=True,
            label_valid=True,
            num_objects=0,
        )

    boxes: list[tuple[int, float, float, float, float]] = []
    errors: list[str] = []
    warnings: list[str] = []
    syntax_valid = True
    normalized_valid = True
    geometry_valid = True
    for line_number, line in enumerate(lines, start=1):
        columns = line.split()
        if len(columns) != 5:
            syntax_valid = False
            errors.append(f"line {line_number}: expected 5 columns, got {len(columns)}")
            continue
        try:
            class_id = int(columns[0])
        except ValueError:
            syntax_valid = False
            errors.append(f"line {line_number}: class_id is not an integer")
            continue
        if class_id < 0:
            syntax_valid = False
            errors.append(f"line {line_number}: class_id is negative")
            continue
        try:
            values = tuple(_number(value) for value in columns[1:])
        except ValueError as error:
            syntax_valid = False
            errors.append(f"line {line_number}: {error}")
            continue
        x_center, y_center, width, height = values
        if not (
            0 <= x_center <= 1
            and 0 <= y_center <= 1
            and 0 < width <= 1
            and 0 < height <= 1
        ):
            normalized_valid = False
            errors.append(f"line {line_number}: normalized value outside allowed range")
        x_min = x_center - width / 2
        x_max = x_center + width / 2
        y_min = y_center - height / 2
        y_max = y_center + height / 2
        if not (0 <= x_min <= 1 and 0 <= x_max <= 1 and 0 <= y_min <= 1 and 0 <= y_max <= 1):
            geometry_valid = False
            if all(-0.01 <= value <= 1.01 for value in (x_min, x_max, y_min, y_max)):
                warnings.append(f"line {line_number}: slight geometry overflow")
            else:
                errors.append(f"line {line_number}: geometry outside image bounds")
        boxes.append((class_id, x_center, y_center, width, height))

    areas = [box[3] * box[4] for box in boxes]
    widths = [box[3] for box in boxes]
    heights = [box[4] for box in boxes]
    canonical = tuple(
        sorted(
            (class_id, round(x, 10), round(y, 10), round(width, 10), round(height, 10))
            for class_id, x, y, width, height in boxes
        )
    )
    return LabelValidationResult(
        label_empty=False,
        syntax_valid=syntax_valid,
        normalized_values_valid=normalized_valid,
        geometry_valid=geometry_valid,
        label_valid=syntax_valid and normalized_valid and geometry_valid,
        num_objects=len(boxes),
        classes_present=sorted({box[0] for box in boxes}),
        bbox_area_mean=sum(areas) / len(areas) if areas else None,
        bbox_area_min=min(areas) if areas else None,
        bbox_area_max=max(areas) if areas else None,
        bbox_width_mean=sum(widths) / len(widths) if widths else None,
        bbox_height_mean=sum(heights) / len(heights) if heights else None,
        warnings=warnings,
        errors=errors,
        canonical=canonical if syntax_valid else (),
    )


def validate_label_bytes(content: bytes) -> LabelValidationResult:
    """Decode UTF-8 label bytes and validate them."""
    try:
        return validate_label_text(content.decode("utf-8"))
    except UnicodeDecodeError as error:
        return LabelValidationResult(
            label_empty=False,
            syntax_valid=False,
            normalized_values_valid=False,
            geometry_valid=False,
            label_valid=False,
            num_objects=0,
            errors=[f"UTF-8 decode error: {error}"],
        )
