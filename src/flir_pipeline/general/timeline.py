"""Offline comparative view of evidence-backed positions, never filename axes."""

from __future__ import annotations

from html import escape

from flir_pipeline.general.contract import Manifest


def render_timeline(manifest: Manifest, result: dict) -> str | None:
    positions = [
        (r, p)
        for r in manifest.records
        for p in r.positions
        if p.kind == "decoded_pts" and p.status == "confirmed"
    ]
    if not positions:
        return None
    colors = {"train": "#235789", "val": "#d58900", "test": "#a23b72"}
    parts = [
        '<!doctype html><html lang="es"><meta charset="utf-8"><title>Dependencias y posiciones observadas</title>',
        "<style>body{font:16px system-ui;margin:2rem;max-width:1400px}svg{width:100%;height:auto}table{width:100%;table-layout:fixed;border-collapse:collapse;font-size:13px}td,th{border:1px solid #ccc;padding:5px;overflow-wrap:anywhere}code{word-break:break-all}</style>",
        "<h1>Posiciones observadas y comparación de políticas</h1><p>Azul: train; ámbar: val; magenta: test; gris: excluido/desconocido. Se muestran todas las apariciones verificadas. Varias apariciones no determinan cuál originó la imagen. Un vídeo no es un grupo de dependencia.</p>",
    ]
    runs = [("historical", None, result["historical_assignments"])] + [
        (r["policy"], r["seed"], r["assignments"]) for r in result["runs"]
    ]
    for video in manifest.videos:
        subset = [(r, p) for r, p in positions if p.video_id == video.video_id]
        if not subset:
            continue
        low, high = min(p.seconds for _, p in subset), max(p.seconds for _, p in subset)
        parts.append(
            f"<h2>{escape(video.video_id)}</h2><p>SHA256: <code>{video.sha256}</code>. Intervalo observado: {low:.6f}–{high:.6f} s.</p>"
        )
        parts.append(
            '<svg viewBox="0 0 1300 410" role="img" aria-label="Comparación temporal de particiones">'
        )
        for i, (policy, seed, mapping) in enumerate(runs):
            y = 25 + i * 35
            parts.append(
                f'<text x="0" y="{y}" font-size="12">{escape(policy)} {seed if seed is not None else ""}</text><line x1="300" x2="1280" y1="{y}" y2="{y}" stroke="#ddd"/>'
            )
            for r, p in subset:
                x = 300 + 980 * (p.seconds - low) / max(high - low, 1e-9)
                split = mapping.get(r.frame_id, "excluded/unknown")
                title = escape(
                    f"{r.frame_id} | {p.seconds:.6f}s | {split} | PTS={p.pts}"
                )
                parts.append(
                    f'<circle cx="{x:.3f}" cy="{y}" r="2.5" fill="{colors.get(split, "#777")}"><title>{title}</title></circle>'
                )
        for tick in range(6):
            x = 300 + 980 * tick / 5
            seconds = low + (high - low) * tick / 5
            anchor = "end" if tick == 5 else "start" if tick == 0 else "middle"
            parts.append(
                f'<text x="{x:.1f}" y="405" text-anchor="{anchor}" font-size="12">{seconds:.1f} s</text>'
            )
        parts.append("</svg>")
    parts.append(
        "<h2>Registro de evidencia</h2><table><tr><th>Ocurrencia</th><th>Vídeo</th><th>Ordinal decodificado</th><th>PTS / base</th><th>Segundos</th><th>Evidencia</th></tr>"
    )
    for r, p in sorted(
        positions,
        key=lambda item: (item[1].video_id, item[1].seconds, item[0].frame_id),
    ):
        parts.append(
            "<tr>"
            + "".join(
                f"<td>{escape(str(x))}</td>"
                for x in (
                    r.frame_id,
                    p.video_id,
                    p.decoded_ordinal,
                    f"{p.pts} / {p.time_base}",
                    p.seconds,
                    p.evidence.path,
                )
            )
            + "</tr>"
        )
    parts.append(
        "</table><p>Los tiempos nominales, externos y candidatos se conservan en el manifiesto; no se sitúan como PTS verificado. Cero cruces aceptados no demuestra ausencia universal de leakage.</p></html>"
    )
    return "\n".join(parts)
