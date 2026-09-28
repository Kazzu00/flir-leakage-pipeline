"""Deterministic all-occurrence temporal sheets over read-only declared images."""

import math
import re
from html import escape
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageOps

from flir_pipeline.data.local_images import relative_posix_path
from flir_pipeline.features.image_source import ImageSource
from flir_pipeline.linkage.review_model import KEYS
from flir_pipeline.similarity.storage import stable_id


def folder_name(identity):
    # Readable organization plus an identity suffix; never use arbitrary CSV
    # values directly as paths (traversal, Windows reserved names, collisions).
    label = re.sub(r"[^a-zA-Z0-9_-]+", "_", identity).strip("_")[:48] or "group"
    return f"{label}-{stable_id({'value': identity})}"


def context_plan(review, candidates, candidate_occurrences, sequences, config):
    """One center for EVERY candidate occurrence, even across different groups.

    Neighbor windows stay within a video and may cross sequence/group boundaries;
    those memberships are displayed, never silently clipped to the proposed group.
    No top candidate, occurrence or encoder threshold is used to discard evidence.
    """
    rows = []
    timelines = {
        video: frame.sort_values(["timestamp_seconds", "sample_index", "frame_id"])
        for video, frame in sequences.groupby("video_id", sort=True)
    }
    for query in review.sort_values(KEYS).itertuples():
        contents = candidates.loc[
            candidates.labeled_content_id.eq(query.labeled_content_id),
            "video_content_id",
        ]
        centers = candidate_occurrences.loc[
            candidate_occurrences.video_content_id.isin(contents)
        ].sort_values(["video_id", "sample_index", "video_frame_id"])
        folder = f"contact_sheets/{folder_name(query.review_stratum)}/{folder_name(query.proposed_visual_dependency_group_id)}/{query.review_query_id}"
        for center in centers.itertuples():
            path = f"{folder}/{stable_id({'frame_id': center.video_frame_id})}.png"
            timeline = timelines[center.video_id]
            context = timeline.loc[
                timeline.timestamp_seconds.sub(center.timestamp_seconds)
                .abs()
                .le(config.context_seconds + 1e-9)
            ]
            for frame in context.itertuples():
                rows.append(
                    {
                        "review_query_id": query.review_query_id,
                        "labeled_content_id": query.labeled_content_id,
                        "proposed_visual_dependency_group_id": query.proposed_visual_dependency_group_id,
                        "review_stratum": query.review_stratum,
                        "center_video_frame_id": center.video_frame_id,
                        "center_video_content_id": center.video_content_id,
                        "contact_sheet": path,
                        "video_frame_id": frame.frame_id,
                        "video_content_id": frame.content_id,
                        "video_id": frame.video_id,
                        "sample_index": int(frame.sample_index),
                        "timestamp_seconds": float(frame.timestamp_seconds),
                        "relative_seconds": float(
                            frame.timestamp_seconds - center.timestamp_seconds
                        ),
                        "sequence_id": frame.sequence_id,
                        "visual_dependency_group_id": frame.visual_dependency_group_id,
                        "image_path": frame.image_path,
                        "image_sha256": frame.image_sha256,
                        "is_center": frame.frame_id == center.video_frame_id,
                    }
                )
    if not rows:
        raise ValueError("No candidate occurrence context is available for the review")
    return pd.DataFrame(rows)


def _label(value):
    # The portable Pillow default font is sufficient for identifiers; exact full
    # text survives in CSV/JSON/HTML even if a font cannot render a Unicode glyph.
    return str(value).encode("ascii", "replace").decode()


def _sheet(query, context, labeled_source, video_source, labeled_path_column):
    tiles = [
        (
            labeled_source.decode(query[labeled_path_column], query.image_sha256),
            [
                "LABELED QUERY (display occurrence)",
                str(query.frame_id),
                str(query.content_id)[:32],
            ],
            False,
        )
    ]
    for row in context.itertuples():
        tiles.append(
            (
                video_source.decode(row.image_path, row.image_sha256),
                [
                    f"{'CENTER' if row.is_center else 'context'} {row.relative_seconds:+g}s / t={row.timestamp_seconds:g}",
                    f"video {row.video_id}",
                    f"seq {row.sequence_id}",
                    f"group {row.visual_dependency_group_id}",
                ],
                row.is_center,
            )
        )
    width, height, columns = 240, 246, min(4, len(tiles))
    result = Image.new(
        "RGB", (width * columns, height * math.ceil(len(tiles) / columns)), "white"
    )
    draw = ImageDraw.Draw(result)
    for i, (image, lines, center) in enumerate(tiles):
        x, y = (i % columns) * width, (i // columns) * height
        thumbnail = ImageOps.contain(
            image.convert("RGB"), (width - 8, 176), Image.Resampling.LANCZOS
        )
        result.paste(
            thumbnail,
            (x + (width - thumbnail.width) // 2, y + (180 - thumbnail.height) // 2),
        )
        draw.rectangle(
            (x, y, x + width - 1, y + height - 1),
            outline="#b44b00" if center else "#cccccc",
            width=3 if center else 1,
        )
        for line, text in enumerate(lines):
            draw.text((x + 5, y + 182 + line * 14), _label(text)[:38], fill="black")
    return result


def render_sheets(
    output: Path,
    review,
    context,
    labeled,
    *,
    labeled_images_archive=None,
    labeled_images_root=None,
    video_images_root,
):
    """Hash-check every displayed image. No extraction or source writes occur."""
    with (
        ImageSource(labeled_images_archive, labeled_images_root) as labeled_source,
        ImageSource(None, video_images_root) as video_source,
    ):
        path_column = labeled_source.path_column(labeled)
        display = (
            labeled.sort_values("frame_id")
            .drop_duplicates("content_id")
            .set_index("content_id", drop=False)
        )
        for path, frames in context.groupby("contact_sheet", sort=True):
            relative_posix_path(path)
            target = output / path
            target.parent.mkdir(parents=True, exist_ok=True)
            sheet = _sheet(
                display.loc[frames.labeled_content_id.iloc[0]],
                frames,
                labeled_source,
                video_source,
                path_column,
            )
            sheet.save(target, format="PNG", optimize=False, compress_level=6)
    links = []
    for query in review.sort_values(["review_stratum", *KEYS]).itertuples():
        paths = sorted(
            context.loc[
                context.review_query_id.eq(query.review_query_id), "contact_sheet"
            ].unique()
        )
        links.append(
            f"<section><h2>{escape(query.review_stratum)} / {escape(query.proposed_visual_dependency_group_id)}</h2><p>{escape(query.labeled_content_id)}</p>"
            + "".join(
                f'<p><a href="{escape(path, quote=True)}">{escape(Path(path).stem)}</a></p><img style="max-width:100%" src="{escape(path, quote=True)}" loading="lazy" alt="Temporal candidate evidence">'
                for path in paths
            )
            + "</section>"
        )
    (output / "index.html").write_text(
        '<!doctype html><html lang="es"><meta charset="utf-8"><title>Calibracion manual de linkage</title>'
        "<body><h1>Calibracion manual de evidencia por grupo</h1><p>Muestra estratificada: no estima accuracy. "
        "No confirma matches, frames exactos, secuencias exactas ni splits. Revisar todas las ocurrencias. "
        "Los vecinos pueden cruzar limites de secuencia/grupo.</p>"
        + "".join(links)
        + "</body></html>",
        encoding="utf-8",
    )
