"""Static native-raster review evidence; drawing never changes original labels."""

from __future__ import annotations

import html
import io
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from flir_pipeline.data.classes import class_display_name
from flir_pipeline.detection.annotation_review import project_box

COLORS = ("#ffcf00", "#ff5b5b", "#00e5ff", "#7dff65", "#ff85ed")
CAUTION = (
    "Test PROPOSED; entrenamiento bloqueado. Revisar posibles omisiones, clases, "
    "objetos agrupados, cajas parciales y tamaño de cajas. Una imagen visible no "
    "demuestra una omisión. Las cajas se dibujan sin corregir ni redondear el YOLO."
)


def page(title, body):
    return (
        "<!doctype html><html lang='es'><meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title><style>"
        "body{font:16px system-ui;background:#151a22;color:#eee;margin:24px}"
        "a{color:#82d8ff}img{max-width:100%}.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:18px}"
        "article{padding:12px;background:#242c38}code{overflow-wrap:anywhere}"
        "table{border-collapse:collapse}td,th{padding:7px;border:1px solid #666}"
        "</style><body>"
        f"<h1>{html.escape(title)}</h1><p>{CAUTION}</p>{body}</body></html>"
    )


def render_case(root: Path, raw: bytes, label_raw: bytes, record, facts, conflict):
    frame = record["frame_id"]
    target = root / "cases" / frame
    target.mkdir(parents=True)
    with Image.open(io.BytesIO(raw)) as opened:
        suffix = {"JPEG": ".jpg", "PNG": ".png"}.get(opened.format, ".image")
        raster = opened.convert("RGB")
    if raster.size != (record["width"], record["height"]):
        raise ValueError("Original raster dimensions disagree with manifest")
    (target / f"original{suffix}").write_bytes(raw)
    (target / "original.txt").write_bytes(label_raw)
    # Both displayed rasters use the same decoded coordinates. Browsers must not
    # apply EXIF orientation to only the original JPEG panel.
    raster.info.clear()
    raster.save(target / "unannotated.png")
    annotated = raster.copy()
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default(size=16)
    for box in facts["boxes"]:
        c = box["class_id"]
        coords = project_box([c, *[box[k] for k in ("x", "y", "w", "h")]], *raster.size)
        draw.rectangle(coords, outline=COLORS[c], width=2)
        text = f"[{box['box_index']}] {class_display_name(c)}"
        x, y = max(0, coords[0]), max(0, coords[1] - 20)
        draw.rectangle(draw.textbbox((x, y), text, font=font), fill="black")
        draw.text((x, y), text, font=font, fill=COLORS[c])
    annotated.save(target / "annotated.png")
    raster.thumbnail((320, 220))
    raster.save(target / "thumbnail.png")
    source = html.escape(f"{record['source_archive']}::{record['zip_member_name']}")
    details = (
        f"<p><a href='../../index.html'>Índice</a> · <a href='original.txt'>YOLO original</a> · "
        f"<a href='original{suffix}'>Archivo de imagen original sin modificar</a></p>"
        f"<p><code>{frame}</code><br>{source}<br>{record['width']} × {record['height']}; "
        f"{facts['box_count']} cajas. Etiqueta vacía: {facts['empty']}.</p>"
        f"<p>SHA-256 imagen: <code>{record['image_sha256']}</code><br>"
        f"SHA-256 etiqueta: <code>{record['label_sha256']}</code></p>"
    )
    if conflict:
        details += (
            f"<p><a href='../../prior_review/cases/{conflict}/comparison.png'>"
            "Conflicto previo A/B, pendiente de adjudicación</a></p>"
        )
    details += (
        "<div class='pair'><figure><figcaption>Original sin cajas</figcaption>"
        "<a href='unannotated.png'><img src='unannotated.png'></a></figure>"
        "<figure><figcaption>Anotación actual del test</figcaption>"
        "<a href='annotated.png'><img src='annotated.png'></a></figure></div>"
        "<p>Ampliar las imágenes a resolución nativa para inspección fina. "
        "Registrar criterios, evidencia de vídeo y justificación en una copia externa de human_review.template.json.</p>"
    )
    (target / "index.html").write_text(
        page(record["basename"], details), encoding="utf-8"
    )


def gallery_indexes(root, inventory, conflicts, exposure):
    subsets = {
        "all": inventory,
        "vehicles": [r for r in inventory if 0 in r["classes"]],
        "heavy-machinery": [r for r in inventory if 4 in r["classes"]],
        "empty": [r for r in inventory if r["empty"]],
        "conflicts": [r for r in inventory if r["conflict_id"]],
    }
    links = []
    for name, rows in subsets.items():
        pages = max(1, (len(rows) + 23) // 24)
        links.append(
            f"<li><a href='{name}-1.html'>{name}</a>: {len(rows)} imágenes</li>"
        )
        for n in range(pages):
            nav = " · ".join(
                f"<a href='{name}-{j + 1}.html'>{j + 1}</a>" for j in range(pages)
            )
            cards = []
            for r in rows[n * 24 : (n + 1) * 24]:
                f = r["frame_id"]
                cards.append(
                    f"<article><a href='cases/{f}/index.html'><img loading='lazy' "
                    f"src='cases/{f}/thumbnail.png'><br>{html.escape(r['basename'])}</a>"
                    f"<p>{r['box_count']} cajas; clases {r['classes']}</p></article>"
                )
            (root / f"{name}-{n + 1}.html").write_text(
                page(
                    name,
                    f"<a href='index.html'>Índice</a><p>{nav}</p><div class='grid'>{''.join(cards)}</div>",
                ),
                encoding="utf-8",
            )
    body = (
        f"<p>{len(inventory)} imágenes; {exposure} contenidos expuestos a entrenamiento histórico. "
        "Test retrospectivo, no independiente.</p><ul>" + "".join(links) + "</ul>"
        "<p><a href='report.md'>Informe crítico</a> · <a href='support.csv'>Soporte por clase</a> · "
        "<a href='human_review.template.csv'>Matriz humana</a> · "
        "<a href='prior_review/index.html'>Revisión previa sin modificar</a></p>"
        f"<p>{len(conflicts)} conflictos permanecen pendientes. Las observaciones aportadas "
        "por el solicitante están separadas en observations.json y no son decisiones humanas completadas.</p>"
    )
    (root / "index.html").write_text(
        page("Ground Truth Quality — revisión humana", body), encoding="utf-8"
    )
