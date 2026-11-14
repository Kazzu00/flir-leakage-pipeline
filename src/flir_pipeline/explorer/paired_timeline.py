"""Offline paired timeline and reproducible, descriptive visual QA.

Pixel changes identify processed regions, not the unavailable producer mask.
Sample selection is stratified, deliberately diagnostic, not a population estimate.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import tarfile
from collections import Counter
from pathlib import PurePosixPath

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from flir_pipeline.explorer.paired_inputs import (
    digest,
    safe_relative,
    write_csv,
    write_json,
)
from flir_pipeline.explorer.timeline import comparison_figure


def pixel_metrics(a, b, boxes):
    if a.shape != b.shape or a.ndim != 3:
        raise ValueError("Pair dimensions or image channels differ")
    changed = np.any(a != b, axis=2)
    delta = np.max(np.abs(a.astype(np.int16) - b.astype(np.int16)), axis=2)
    h, w = changed.shape
    fractions = []
    for _, x, y, bw, bh in boxes:
        x0, x1 = (
            max(0, int(np.floor((x - bw / 2) * w))),
            min(w, int(np.ceil((x + bw / 2) * w))),
        )
        y0, y1 = (
            max(0, int(np.floor((y - bh / 2) * h))),
            min(h, int(np.ceil((y + bh / 2) * h))),
        )
        area = changed[y0:y1, x0:x1]
        fractions.append(float(area.mean()) if area.size else 0.0)
    return {
        "changed_pixel_fraction": float(changed.mean()),
        "changed_gt8_fraction": float((delta > 8).mean()),
        "max_changed_box_fraction": max(fractions, default=0.0),
        "boxes_intersecting_changed_pixels": sum(x > 0 for x in fractions),
        "box_changed_fractions": fractions,
    }


def sample_pairs(pairs, seed, per_video):
    """Seeded hash order covers video/resolution, each class, empty and changed boxes."""
    selected = {}
    for video in sorted({p["source_video"] for p in pairs}):
        population = [p for p in pairs if p["source_video"] == video]
        ordered = sorted(population, key=lambda p: digest([seed, p["pair_id"]]))
        strata = {
            "all": ordered,
            "empty": [p for p in ordered if p["label_empty"]],
            "changed_boxes": [
                p
                for p in ordered
                if p["pixel_metrics"]["boxes_intersecting_changed_pixels"]
            ],
        }
        strata.update(
            {f"class_{c}": [p for p in ordered if c in p["classes"]] for c in range(5)}
        )
        strata.update(
            {
                f"resolution_{w}x{h}": [
                    p for p in ordered if (p["width"], p["height"]) == (w, h)
                ]
                for w, h in sorted({(p["width"], p["height"]) for p in ordered})
            }
        )
        for label, candidates in strata.items():
            if candidates:
                selected.setdefault(candidates[0]["pair_id"], []).append(
                    f"{video}:{label}"
                )
        for p in ordered:
            if sum(q["pair_id"] in selected for q in population) >= per_video:
                break
            selected.setdefault(p["pair_id"], []).append(f"{video}:seeded_fill")
    return selected


class TarImages:
    def __init__(self, root):
        self.root = root
        self.archives = {}

    def read(self, record):
        source, member = record["source"], record["member"]
        if PurePosixPath(member).is_absolute() or ".." in PurePosixPath(member).parts:
            raise ValueError("Unsafe member")
        if source not in self.archives:
            archive = tarfile.open(safe_relative(self.root, source))
            members = archive.getmembers()
            if len({m.name for m in members}) != len(members):
                archive.close()
                raise ValueError("Duplicate TAR member names")
            self.archives[source] = archive
        archive = self.archives[source]
        info = archive.getmember(member)
        if not info.isfile() or info.size > 128 * 1024 * 1024:
            raise ValueError("Unsupported archive entry")
        with archive.extractfile(info) as stream:
            data = stream.read()
        if hashlib.sha256(data).hexdigest() != record["sha256"]:
            raise ValueError("Image differs from audited checksum")
        with Image.open(io.BytesIO(data)) as image:
            return image.convert("RGB")

    def close(self):
        for archive in self.archives.values():
            archive.close()


def thumbnail(image, boxes, width):
    thumb = image.copy()
    thumb.thumbnail((width, width))
    draw = ImageDraw.Draw(thumb)
    w, h = thumb.size
    for cls, x, y, bw, bh in boxes:
        draw.rectangle(
            ((x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h),
            outline="#ffca3a",
            width=1,
        )
        draw.text(
            (max(0, (x - bw / 2) * w), max(0, (y - bh / 2) * h)),
            str(cls),
            fill="#ffca3a",
        )
    data = io.BytesIO()
    thumb.save(data, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(data.getvalue()).decode()


def public_row(p):
    row = {
        k: v
        for k, v in p.items()
        if k not in {"original", "no_hud", "boxes", "thumbnails"}
    }
    for side in ("original", "no_hud"):
        for key in (
            "inventory_occurrence_id",
            "source",
            "member",
            "sha256",
            "label_sha256",
            "label_members",
        ):
            row[f"{side}_{key}"] = p[side][key]
    return row


def render_html(pairs, output):
    charts = {}
    for video in sorted({p["source_video"] for p in pairs}):
        selected = [p for p in pairs if p["source_video"] == video]
        charts[video] = []
        for left, label in (
            ("historical_split", "Histórico declarado"),
            ("random_split", "Aleatorio provisional"),
        ):
            table = pd.DataFrame(
                {
                    "possible_frame_index": [p["frame_index"] for p in selected],
                    "left": [p.get(left, "") for p in selected],
                    "right": [p.get("grouped_split", "") for p in selected],
                }
            )
            figure = comparison_figure(table, label, "Agrupado provisional")
            figure.axes[0].set_xlabel(
                "Índice nativo declarado por proveedor; no verificado contra vídeo"
            )
            figure.axes[0].set_title(video)
            buffer = io.BytesIO()
            figure.savefig(buffer, format="png", dpi=110)
            figure.clear()
            charts[video].append(
                "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
            )
    # Data are escaped for a script element; UI uses textContent, never supplier HTML.
    payload = json.dumps(
        [{**public_row(p), "thumbnails": p["thumbnails"]} for p in pairs],
        ensure_ascii=False,
    ).replace("<", "\\u003c")
    template = """<!doctype html><html lang="es"><meta charset="utf-8"><title>FLIR · Comparación temporal provisional</title>
<style>body{font:16px system-ui;margin:24px;background:#101820;color:#eef3f7}header,nav{position:sticky;top:0;background:#101820;padding:10px;z-index:2}select,input,button{font:inherit;padding:8px;margin:5px}article{background:#1b2935;padding:18px;margin:18px 0;border-radius:10px}.pair{display:flex;gap:12px;flex-wrap:wrap}.pair img{width:min(46vw,640px);height:auto}pre{white-space:pre-wrap;font:14px system-ui}.warn{color:#ffd27d}a{color:#8fd3ff}</style>
<header><h1>FLIR · Original / NO-HUD</h1><p class="warn">PROPUESTA PROVISIONAL · No autoriza entrenamiento. Tiempo declarado por proveedor, no verificado contra vídeo. Secuencias pendientes.</p>
<label>Vídeo <select id="video"></select></label><label>Secuencia <select id="sequence"></select></label><label>Estado <select id="state"><option value="">Todos</option><option>eligible</option><option>quarantine</option><option>unresolved</option></select></label>
<label>Buscar <input id="query" placeholder="Nombre original o entregado"></label><label><input id="sample" type="checkbox">Solo muestra visual</label>
<button id="prev">Anterior</button><button id="next">Siguiente</button><span id="count" role="status"></span></header><section id="timeline" aria-label="Ejes temporales por vídeo"></section><p>Los ejes muestran el vídeo seleccionado completo. Las filas sin asignación corresponden a casos excluidos o pendientes; no prueban independencia. Los filtros afectan las tarjetas.</p><main id="cards"></main>
<script id="charts" type="application/json">__CHARTS__</script><script id="records" type="application/json">__DATA__</script><script>
const rows=JSON.parse(document.getElementById('records').textContent),video=document.getElementById('video'),sequence=document.getElementById('sequence'),state=document.getElementById('state'),query=document.getElementById('query'),sample=document.getElementById('sample'),cards=document.getElementById('cards');let page=0;
for(const v of [...new Set(rows.map(x=>x.source_video))])video.add(new Option(v,v));
function sequences(){sequence.replaceChildren(new Option('Todas',''));for(const s of [...new Set(rows.filter(r=>r.source_video===video.value).map(r=>r.sequence_id))])sequence.add(new Option(s==='unresolved'?'Sin secuencia confirmada':s,s));const target=document.getElementById('timeline');target.replaceChildren();for(const src of JSON.parse(document.getElementById('charts').textContent)[video.value]){const img=document.createElement('img');img.src=src;img.alt='Comparación temporal de particiones para '+video.value;img.style.width='min(100%,1210px)';target.append(img);}}
function render(){let list=rows.filter(r=>r.source_video===video.value&&(!sequence.value||r.sequence_id===sequence.value)&&(!state.value||r.status===state.value)&&(!sample.checked||r.visual_sample)&&((r.filename+' '+r.original_name).toLowerCase().includes(query.value.toLowerCase())));page=Math.max(0,Math.min(page,Math.ceil(list.length/12)-1));cards.replaceChildren();document.getElementById('count').textContent=`${list.length} pares · página ${page+1}/${Math.max(1,Math.ceil(list.length/12))}`;
for(const r of list.slice(page*12,page*12+12)){const card=document.createElement('article'),title=document.createElement('h2'),details=document.createElement('pre'),pair=document.createElement('div');title.textContent=`${r.filename} · ${r.status}`;pair.className='pair';
for(const side of ['original','no_hud']){const f=document.createElement('figure'),i=document.createElement('img'),c=document.createElement('figcaption');i.src=r.thumbnails[side];i.alt=side+' '+r.filename;i.loading='lazy';c.textContent=side==='original'?'Original · cajas fuente':'NO-HUD · mismas cajas fuente';f.append(i,c);pair.append(f);}
details.textContent=JSON.stringify({nombre_original:r.original_name,frame_declarado:r.frame_index,segundo_declarado:r.timestamp_seconds,temporalidad:r.temporal_status,secuencia:r.sequence_id,split_historico:r.historical_split,split_aleatorio:r.random_split||'pendiente',split_agrupado:r.grouped_split||'pendiente',clases:r.classes,vacio:r.label_empty,cajas:r.num_objects,motivos:r.reasons,dependencias_historicas:r.historical_decisions,potenciales_cruces:r.candidate_cross_split||[],muestra:r.visual_sample||false,cambios_pixeles:r.pixel_metrics},null,2);card.append(title,pair,details);cards.append(card);}}
video.onchange=()=>{page=0;sequences();render()};for(const el of [sequence,state,query,sample])el.oninput=()=>{page=0;render()};document.getElementById('prev').onclick=()=>{page--;render()};document.getElementById('next').onclick=()=>{page++;render()};sequences();render();
</script></html>"""
    output.write_text(
        template.replace("__DATA__", payload).replace(
            "__CHARTS__", json.dumps(charts).replace("<", "\\u003c")
        ),
        encoding="utf-8",
    )


def build_timeline(pairs, data_root, output, cfg, reconciliation):
    reader = TarImages(data_root)
    try:
        for i, p in enumerate(pairs):
            images = {side: reader.read(p[side]) for side in ("original", "no_hud")}
            p["pixel_metrics"] = pixel_metrics(
                np.asarray(images["original"]), np.asarray(images["no_hud"]), p["boxes"]
            )
            p["thumbnails"] = {
                side: thumbnail(image, p["boxes"], cfg["thumbnail_width"])
                for side, image in images.items()
            }
            if i % 100 == 0:
                print(f"Timeline media {i}/{len(pairs)}", flush=True)
        selected = sample_pairs(pairs, cfg["seed"], cfg["sample_per_video"])
        for p in pairs:
            p["visual_sample"] = p["pair_id"] in selected
            p["sample_strata"] = selected.get(p["pair_id"], [])
        # Local contact sheets make the actual sample inspectable without a browser.
        sample = [p for p in pairs if p["visual_sample"]]
        for page in range(0, len(sample), 4):
            sheet = Image.new("RGB", (1200, 4 * 380), "white")
            draw = ImageDraw.Draw(sheet)
            for j, p in enumerate(sample[page : page + 4]):
                draw.text(
                    (10, j * 380),
                    p["filename"] + " | " + str(p["classes"]),
                    fill="black",
                )
                for side, x in (("original", 0), ("no_hud", 600)):
                    image = reader.read(p[side])
                    encoded = thumbnail(image, p["boxes"], 590).split(",", 1)[1]
                    with Image.open(io.BytesIO(base64.b64decode(encoded))) as thumb:
                        sheet.paste(thumb, (x, j * 380 + 30))
            sheet.save(output / f"visual_sample_{page // 4 + 1:02d}.jpg", quality=90)
    finally:
        reader.close()
    write_json(output / "timeline_records.json", pairs)
    write_csv(output / "timeline_manifest.csv", [public_row(p) for p in pairs])
    render_html(pairs, output / "timeline_comparison.html")
    report = {
        "status": "provisional",
        "pairs": len(pairs),
        "videos": dict(Counter(p["source_video"] for p in pairs)),
        "unique_pair_ids": len({p["pair_id"] for p in pairs}),
        "unique_occurrence_ids": len(
            {
                p[s]["inventory_occurrence_id"]
                for p in pairs
                for s in ("original", "no_hud")
            }
        ),
        "sample_size": len(selected),
        "sample_seed": cfg["seed"],
        "sample_design": "video/resolution/class/empty/changed-box strata; hash-seeded selection; not population accuracy estimate",
        "source_member_hashes_verified": 2 * len(pairs),
        "confirmed_temporal_pairs": 0,
        "confirmed_sequence_assignments": 0,
        "pixel_comparison": "diagnostic_only; changed pixels are not producer mask",
        "object_semantics_review": "pending_human; source labels unchanged",
        "reconciliation": reconciliation,
    }
    write_json(output / "timeline_verification.json", report)
    return report
