"""Portable GIF bytes; playback speed is never stored as dataset metadata."""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import asdict

from PIL import Image, ImageDraw

from flir_pipeline.explorer.frames import FrameReader
from flir_pipeline.explorer.models import PreviewPlan

PLAYBACK_NOTICE = "Playback FPS is for visualization only; source timing is unknown."


def cache_key(plan: PreviewPlan) -> str:
    """All ordered occurrences, content identities and display settings are bound."""
    payload = {"renderer": "pillow_gif_v1", **asdict(plan)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def render_preview(plan: PreviewPlan, reader: FrameReader) -> bytes:
    """No extraction, writes, timestamps or original FPS inference."""
    if not plan.frame_ids or len(plan.frame_ids) != len(plan.content_ids) or len(plan.indices) != len(plan.frame_ids):
        raise ValueError("Incomplete playback plan")
    frames = []
    previous = None
    for frame_id, content_id, index in zip(plan.frame_ids, plan.content_ids, plan.indices, strict=True):
        if reader.rows.loc[frame_id, "content_id"] != content_id:
            raise ValueError("Preview occurrence does not map to expected content")
        image = reader.image(frame_id, width=plan.width)
        canvas = Image.new("RGB", (plan.width, plan.width + 42), "#162b39")
        canvas.paste(image, ((plan.width - image.width) // 2, (plan.width - image.height) // 2))
        gap = f" | gap {index - previous}" if previous is not None else ""
        ImageDraw.Draw(canvas).text((10, plan.width + 12), f"Inferred index {index}{gap} | display {plan.fps} FPS", fill="white")
        frames.append(canvas)
        previous = index
    output = io.BytesIO()
    # GIF timing is quantized to 10 ms; label the requested visualization speed.
    duration = max(10, round(100 / plan.fps) * 10)
    frames[0].save(output, format="GIF", save_all=True, append_images=frames[1:],
                   duration=duration, loop=0, optimize=False, disposal=2)
    return output.getvalue()
