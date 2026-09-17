"""Consolidate existing evidence only; never run an experimental stage.

The public notebook is narrative source. This reader uses aggregate tables,
existing verification receipts and figures. Manifest access checks identity and
coverage only. Missing, inconsistent or changed evidence is visible and causes
a nonzero builder exit; it is never replaced with remembered results.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import re
import subprocess
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path("reports/progress/review")
SOURCE = Path("notebooks/progress_review.ipynb")

SOURCES = {
    "manifest": "data/manifests/flir_canonical_candidate_v1.parquet",
    "f_meta": "reports/feature_engineering/metadata.json",
    "f_audit": "reports/feature_engineering_closure/final_audit.json",
    "c_summary": "reports/clustering/summary.json",
    "s_summary": "reports/splitting/summary.json",
    "d_meta": "reports/detection/report_metadata.json",
    "d_env": "artifacts/detection/environment.json",
    "d_plan": "artifacts/detection/protocol/plan.json",
    "d_budget": "artifacts/detection/compute_budget.json",
    "d_verify": "reports/detection/verification_receipt.json",
    "status_doc": "docs/current_status.md",
    "detector_doc": "docs/detector_comparison_analysis.md",
    "split_protocol": "docs/splitting_protocol.md",
}
for prefix, stage, names in [
    ("f", "feature_engineering", ["dataset_summary", "historical_split_summary", "duplicate_summary", "class_catalog", "object_instances_by_class", "annotation_quality", "bbox_geometry_by_class", "embedding_health_dinov2", "embedding_health_clip"]),
    ("v", "similarity", ["feature_spaces", "global_similarity", "topk_global_summary", "temporal_neighbors", "sequence_similarity", "frame_delta_similarity", "temporal_coverage", "neighbor_agreement_summary"]),
    ("r", "reduction", ["runs", "candidate_reference_metrics", "input_spaces"]),
    ("c", "clustering", ["all_runs", "references", "candidates"]),
    ("s", "splitting", ["runs", "metric_variation", "clustering_candidates"]),
    ("d", "detection", ["pilot_validation", "split_context"]),
]:
    for name in names:
        SOURCES[f"{prefix}_{name}"] = f"reports/{stage}/tables/{name}.csv"
for prefix, stage in [("v", "similarity"), ("r", "reduction"), ("c", "clustering"), ("s", "splitting")]:
    SOURCES[f"{prefix}_meta"] = f"reports/{stage}/report_metadata.json"

# Fourteen existing PNGs plus one small HTML pipeline diagram: fifteen figures.
# No galleries are needed; this report never opens source images or label ZIPs.
FIGURES = {
    3: [("feature_engineering/01_dataset_overview.png", "Universo de registros históricos y contenidos únicos."), ("feature_engineering/13_cross_split_duplicate_matrix.png", "Contenidos exactos compartidos entre particiones históricas; la diagonal no representa overlap entre splits.")],
    4: [("feature_engineering/15_class_instances.png", "Instancias anotadas por clase; una imagen puede contener varios objetos.")],
    5: [("feature_engineering/06_bbox_normalized_area_by_class.png", "Área normalizada por instancia y clase; se conservan extremos."), ("feature_engineering/17_bbox_aspect_ratio_by_class.png", "Ratio de ancho/alto en coordenadas YOLO normalizadas, no ratio en píxeles.")],
    7: [("similarity/03_rank1_similarity_comparison.png", "Distribución por consulta del coseno rank-1 y de la media de 5/10/20 vecinos; escalas propias de cada encoder.")],
    8: [("similarity/05_similarity_vs_frame_delta_dinov2.png", "DINOv2: similitud frente a log(1 + Δ inferido) dentro de una secuencia; el color cuenta pares por celda en escala logarítmica."), ("similarity/06_similarity_vs_frame_delta_clip.png", "CLIP: similitud frente a log(1 + Δ inferido), no segundos. El color cuenta pares por celda en escala logarítmica; ese rótulo está parcialmente recortado en el PNG original.")],
    10: [("reduction/05_tsne_dinov2_reference.png", "Referencia DINOv2 t-SNE, semilla 0; la apariencia no determina la selección."), ("reduction/03_pacmap_dinov2_reference.png", "Referencia DINOv2 PaCMAP, semilla 0; densidad 2D no equivale a densidad original.")],
    12: [("clustering/R6_dinov2_tsne_clusters.png", "R6: ejemplo exploratorio DINOv2/t-SNE/OPTICS, con ruido visible; no es C10 ni un ganador global.")],
    15: [("splitting/06_high_similarity_cross_split_pairs.png", "Fracción porcentual cross-split en seis cohortes de similitud: medias entre cinco semillas, para random y los 12 candidatos. La tabla principal muestra conteos top 0.1% y C10 seed 0; las magnitudes no son intercambiables."), ("splitting/04_cross_split_nn_similarity_dinov2.png", "Distribución residual DINOv2 en seed 0 para random y los 12 candidatos. La tabla principal usa la media random de cinco semillas. C01 es una referencia de balance; correlación mínima por sí sola no determina elegibilidad."), ("splitting/07_temporal_cross_split_rates.png", "Fracciones temporales inferidas: medias entre cinco semillas, para random y los 12 candidatos. La tabla principal muestra C10 seed 0; se conservan los valores y leyendas originales.")],
}
for items in FIGURES.values():
    for name, _ in items:
        stage, filename = name.split("/")
        SOURCES[f"figure:{name}"] = f"reports/{stage}/figures/{filename}"

DEPS = {
    1: [], 2: ["d_meta"],
    3: ["manifest", "f_dataset_summary", "f_historical_split_summary", "f_duplicate_summary"],
    4: ["f_class_catalog", "f_object_instances_by_class", "f_annotation_quality"],
    5: ["f_bbox_geometry_by_class", "f_meta"],
    6: ["manifest", "f_audit", "f_embedding_health_dinov2", "f_embedding_health_clip"],
    7: ["v_feature_spaces", "v_global_similarity", "v_topk_global_summary"],
    8: ["v_temporal_neighbors", "v_sequence_similarity", "v_frame_delta_similarity", "v_temporal_coverage"],
    9: ["v_neighbor_agreement_summary"],
    10: ["r_runs", "r_candidate_reference_metrics", "r_input_spaces"],
    11: ["c_summary", "c_all_runs", "c_candidates"],
    12: ["c_references"],
    13: ["s_summary", "s_runs", "split_protocol"],
    14: ["s_runs", "s_clustering_candidates"],
    15: ["s_runs", "s_metric_variation"],
    16: ["s_runs", "s_metric_variation", "d_plan", "d_split_context"],
    17: ["d_meta", "d_env", "d_plan", "d_verify", "d_pilot_validation", "detector_doc"],
    18: ["d_meta", "d_budget", "d_plan"],
    19: ["f_meta", "f_audit", "v_feature_spaces", "v_temporal_coverage", "r_runs", "c_summary", "s_summary", "d_meta", "status_doc"],
    20: ["f_duplicate_summary", "v_temporal_neighbors", "s_runs", "s_metric_variation", "d_meta"],
    21: ["v_temporal_coverage", "f_annotation_quality", "f_object_instances_by_class", "s_runs", "d_meta", "d_env"],
    22: ["d_plan"],
}
for section, names in DEPS.items():
    names.extend(f"figure:{name}" for name, _ in FIGURES.get(section, []))
    for prefix in ("v", "r", "c", "s"):
        if any(name.startswith(prefix + "_") for name in names):
            names.append(prefix + "_meta")

CSS = """<style>
:root { color-scheme: light; }
body { background: #f4f6f8 !important; color: #172b3a; }
.jp-Notebook { max-width: 1150px; margin: auto; background: white; padding: 36px !important; }
.jp-RenderedHTMLCommon { font-size: 16px; line-height: 1.65; }
h1 { color: #11394b; font-size: 2.3em !important; }
h2 { color: #11394b; border-top: 2px solid #dce8ec; padding-top: 30px; margin-top: 28px !important; }
.progress-table { border-collapse: collapse; width: 100%; margin: 18px 0 !important; font-size: 14px !important; }
.progress-table th { background: #eaf1f5; text-align: left !important; }
.progress-table td, .progress-table th { border-bottom: 1px solid #dce3e8; padding: 9px 12px !important; text-align: left; overflow-wrap: anywhere; }
.progress-table tr:nth-child(even) { background: #f7f9fa; }
.badge { display: inline-block; padding: 2px 9px; border-radius: 4px; font-weight: 700; font-size: 12px; margin-right: 8px; }
.observado { color: #184b83; background: #e5effd; }
.validado { color: #176047; background: #e4f4eb; }
.limitacion { color: #855000; background: #fff0cf; }
.pendiente { color: #6b467e; background: #f1eafa; }
.finding { background: #edf4f6; border-left: 4px solid #347285; padding: 14px 18px; margin: 22px 0; }
.source-note { color: #526573; font-size: 12px; overflow-wrap: anywhere; }
.progress-figure { margin: 24px 0; break-inside: avoid; }
.progress-figure img { display: block; width: 100%; height: auto; }
.progress-figure figcaption { color: #526573; font-size: 13px; padding: 8px 6px; }
.pipeline { display: flex; gap: 8px; flex-wrap: wrap; padding: 20px; background: #edf4f6; }
.pipeline span { padding: 9px 12px; border: 1px solid #b9cfd7; border-radius: 5px; background: white; }
.missing { border-left: 4px solid #b06e1d; background: #fff5e1; padding: 15px; }
@media (max-width: 700px) { .jp-Notebook { padding: 12px !important; } .progress-table { font-size: 12px !important; } }
@media print { body { background: white !important; } h2 { break-after: avoid; } .jp-Notebook { padding: 0 !important; } }
</style>"""


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(columns, rows) -> str:
    """Display explicitly selected aggregate values, never arbitrary source columns."""
    def cell(value):
        if isinstance(value, float):
            if pd.isna(value):
                return "no disponible"
            return str(int(value)) if value.is_integer() else f"{value:.6f}"
        return html.escape(str(value))
    head = "".join(f"<th>{cell(c)}</th>" for c in columns)
    body = "".join("<tr>" + "".join(f"<td>{cell(v)}</td>" for v in row) + "</tr>" for row in rows)
    return f'<table class="progress-table"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def pct(value, digits=2):
    return f"{100 * value:.{digits}f}%"


def one(frame, **filters):
    for key, value in filters.items():
        frame = frame.loc[frame[key].eq(value)]
    if len(frame) != 1:
        raise ValueError("La selección de evidencia no identifica una fila única")
    return frame.iloc[0]


class Review:
    """Read-only source snapshot with separate availability and scientific checks."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.data = {}
        self.inventory = {}
        self.issues = {}
        self.sections = {}
        self.sensitive_ids = set()
        for key, relative in SOURCES.items():
            path = self.root / relative
            entry = {"path": relative, "state": "missing"}
            self.inventory[key] = entry
            if not path.is_file():
                self.issues[key] = "missing: artefacto requerido ausente"
                continue
            if not path.resolve().is_relative_to(self.root):
                self.issues[key] = "invalid: fuente fuera del repositorio"
                entry["state"] = "invalid"
                continue
            entry.update(state="available", sha256=digest(path))
            try:
                if path.suffix == ".csv":
                    self.data[key] = pd.read_csv(path)
                elif path.suffix == ".json":
                    self.data[key] = json.loads(path.read_text(encoding="utf-8"))
                elif key == "manifest":
                    frame = pd.read_parquet(path, columns=["frame_id", "content_id", "original_split"])
                    self.data[key] = frame
                    self.sensitive_ids.update(frame.content_id.astype(str))
                    self.sensitive_ids.update(frame.frame_id.astype(str))
                elif path.suffix == ".md":
                    self.data[key] = path.read_text(encoding="utf-8")
            except (ValueError, KeyError, OSError):
                self.issues[key] = "invalid: no se pudo leer el esquema esperado"
                entry["state"] = "invalid"
        self._check_receipts()
        self._check_consistency()

    def _check_receipts(self):
        """Compare only reused files to stored hashes; no numerical verification."""
        for key in ("v_meta", "r_meta", "c_meta", "s_meta", "d_meta"):
            if key in self.issues:
                continue
            parent = Path(SOURCES[key]).parent
            expected = self.data[key].get("output_sha256", {})
            for name, entry in self.inventory.items():
                relative = Path(entry["path"])
                if relative.is_relative_to(parent) and name not in self.issues:
                    recorded = expected.get(relative.relative_to(parent).as_posix())
                    if recorded is not None:
                        entry["matches_previous_receipt"] = recorded == entry["sha256"]
                        if not entry["matches_previous_receipt"]:
                            self.issues[name] = "invalid: cambió respecto al recibo de la etapa"
                            entry["state"] = "invalid"

    def _check_consistency(self):
        def check(keys, condition, message):
            if any(k in self.issues for k in keys):
                return
            try:
                valid = bool(condition())
            except (KeyError, ValueError, TypeError, AttributeError, IndexError):
                valid = False
            if not valid:
                for key in keys:
                    self.issues[key] = "invalid: " + message
                    self.inventory[key]["state"] = "invalid"
        d = self.data
        check(["manifest", "f_dataset_summary", "f_historical_split_summary"], lambda: (
            len(d["manifest"]) == one(d["f_dataset_summary"]).total_records
            and d["manifest"].frame_id.is_unique
            and d["manifest"].content_id.nunique() == one(d["f_dataset_summary"]).unique_content_ids
            and d["manifest"].original_split.value_counts().to_dict() == d["f_historical_split_summary"].set_index("original_split").records.to_dict()
        ), "manifest y resumen no concuerdan")
        check(["f_object_instances_by_class", "f_annotation_quality", "f_class_catalog"], lambda: (
            d["f_object_instances_by_class"].instance_count.sum() == one(d["f_annotation_quality"]).candidate_objects
            and d["f_class_catalog"].academic_mapping_validated.eq(True).all()
            and d["f_class_catalog"].class_name.tolist() == ["Vehicles", "Buildings", "Roads", "Rivers", "Heavy Machinery"]
        ), "catálogo o total de anotaciones inconsistente")
        for encoder, dimension in [("dinov2", 384), ("clip", 512)]:
            def valid_features(encoder=encoder, dimension=dimension):
                row = one(d[f"f_embedding_health_{encoder}"])
                audit = d["f_audit"]["full_features"][encoder]
                return (row.reproducible_full_dataset_valid and audit["reproducible_full_dataset_valid"]
                        and row.content_embedding_count == d["manifest"].content_id.nunique()
                        and row.record_count == len(d["manifest"])
                        and row.embedding_dimension == dimension
                        and row.feature_space_id == audit["feature_space_id"])
            check([f"f_embedding_health_{encoder}", "f_audit", "manifest"], valid_features, "evidencia de extracción completa no válida")
        check(["v_feature_spaces", "v_global_similarity", "manifest"], lambda: (
            d["v_feature_spaces"].quality_valid.eq(True).all()
            and d["v_feature_spaces"].content_count.eq(d["manifest"].content_id.nunique()).all()
            and d["v_global_similarity"]["count"].eq(d["manifest"].content_id.nunique() * (d["manifest"].content_id.nunique() - 1) // 2).all()
        ), "resumen de similitud sin cobertura completa")
        check(["r_runs", "r_meta"], lambda: len(d["r_runs"]) == d["r_meta"]["run_count"] and d["r_runs"].quality_valid.eq(True).all(), "reducciones incompletas o no válidas")
        check(["c_all_runs", "c_summary", "c_candidates"], lambda: len(d["c_all_runs"]) == d["c_summary"]["total_runs"] and len(d["c_candidates"]) == d["c_summary"]["candidate_count"], "resumen de clustering inconsistente")
        check(["s_runs", "s_summary"], lambda: (
            len(d["s_runs"]) == d["s_summary"]["valid_runs"] == d["s_summary"]["run_count"]
            and d["s_runs"].quality_valid.eq(True).all()
            and d["s_runs"].loc[d["s_runs"].strategy.ne("historical"), "exact_duplicate_cross_split_count"].eq(0).all()
            and d["s_runs"].loc[d["s_runs"].strategy.eq("cluster_aware"), "cluster_fracture_count"].eq(0).all()
        ), "invariantes de particiones no acreditados")
        check(["d_meta", "d_budget", "d_verify", "d_pilot_validation"], lambda: (
            d["d_meta"]["scientific_state"] == "PENDING_FINAL_EXPERIMENT"
            and d["d_meta"]["fair_comparison"]["completed_runs"] == 0
            and d["d_verify"]["scientific_runs_completed"] == 0
            and len(d["d_pilot_validation"]) == d["d_meta"]["small_pilot_count"]
            and d["d_pilot_validation"].scientific_result.eq(False).all()
            and d["d_meta"]["compute_budget"] == d["d_budget"]
        ), "el estado del detector cambió; revisar la narrativa antes de consolidar")

    def unchanged(self):
        return all((self.root / entry["path"]).is_file() and digest(self.root / entry["path"]) == entry["sha256"] for entry in self.inventory.values() if "sha256" in entry)

    def section(self, number):
        keys = DEPS[number]
        bad = {key: self.issues[key] for key in keys if key in self.issues}
        if bad:
            self.sections[number] = {"state": "missing_or_invalid", "sources": keys}
            details = table(["Fuente requerida", "Estado"], [(SOURCES[k], v) for k, v in bad.items()])
            return '<div class="missing"><b>LIMITACIÓN · missing / invalid</b>' + details + '<p><b>Hallazgo principal:</b> esta sección no puede acreditar resultados con las fuentes disponibles. No se ejecutó ningún cálculo experimental para reemplazarlas.</p></div>'
        try:
            body, finding, state = self._section(number)
        except (ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
            # Do not expose exceptions containing private source values or paths.
            self.sections[number] = {"state": "invalid", "sources": keys, "error_type": type(exc).__name__}
            return '<div class="missing"><b>LIMITACIÓN · invalid</b><p>El esquema o la selección de filas no coincide con esta revisión.</p><p><b>Hallazgo principal:</b> evidencia insuficiente; no se sustituyen resultados.</p></div>'
        for name, caption in FIGURES.get(number, []):
            relative = SOURCES[f"figure:{name}"]
            encoded = base64.b64encode((self.root / relative).read_bytes()).decode("ascii")
            body += f'<figure class="progress-figure"><img alt="{html.escape(caption)}" src="data:image/png;base64,{encoded}"><figcaption>{html.escape(caption)}<br>Fuente: {relative}</figcaption></figure>'
        citations = "<br>".join(html.escape(SOURCES[k]) for k in keys if not k.startswith("figure:"))
        body += f'<details class="source-note"><summary>Fuentes de esta sección · artefactos existentes</summary>{citations or "Contexto metodológico de la revisión."}</details>'
        badge = state.lower().replace("ó", "o")
        body += f'<div class="finding"><span class="badge {badge}">{state}</span><b>Hallazgo principal:</b> {finding}</div>'
        self.sections[number] = {"state": "available", "scientific_label": state, "sources": keys}
        return body

    def _section(self, number):
        d = self.data
        body = ""
        state = "OBSERVADO"
        if number == 1:
            finding = "La unidad de agrupación debe preservar relaciones entre contenidos y la trazabilidad de cada ocurrencia histórica."
            state = "PENDIENTE"
        elif number == 2:
            steps = ["Data audit", "Canonical content", "DINOv2 / CLIP", "Cosine similarity", "Temporal / visual analysis", "t-SNE / PaCMAP", "DBSCAN / OPTICS / HDBSCAN", "Cluster-aware splitting", "Detector comparison · PENDIENTE"]
            body = '<figure class="progress-figure"><div class="pipeline">' + '<b aria-hidden="true">→</b>'.join(f"<span>{s}</span>" for s in steps) + '</div><figcaption>Ruta metodológica; diagrama de síntesis, sin resultados nuevos.</figcaption></figure>'
            finding = "La evidencia llega hasta la evaluación de particiones; el efecto sobre el detector sigue pendiente."
        elif number == 3:
            r, dup = one(d["f_dataset_summary"]), one(d["f_duplicate_summary"])
            split = d["f_historical_split_summary"].set_index("original_split").records
            body = table(["Unidad / indicador", "Resultado"], [
                ["Registros históricos (frame_id)", r.total_records], ["Contenidos únicos (content_id)", r.unique_content_ids],
                ["Train / val / test históricos", " / ".join(str(split[s]) for s in ["train", "val", "test"])],
                ["Grupos duplicados exactos", dup.duplicate_groups], ["Overlap exacto train–val", dup.train_val_duplicate_content_ids],
                ["Overlap exacto train–test", dup.train_test_duplicate_content_ids], ["Overlap exacto val–test", dup.val_test_duplicate_content_ids]])
            finding = "El histórico contiene duplicados exactos entre particiones: evidencia directa de contenido repetido cross-split."
        elif number == 4:
            q = one(d["f_annotation_quality"])
            body = table(["ID", "Clase canónica", "Bounding boxes"], d["f_object_instances_by_class"][["class_id", "class_name", "instance_count"]].values.tolist())
            body += table(["Universo", "Cantidad"], [["Bounding boxes del candidato canónico", q.candidate_objects], ["Anotaciones vacías", q.candidate_empty_labels], ["Labels huérfanos excluidos", q.orphan_labels], ["Objetos en huérfanos excluidos", q.orphan_objects]])
            finding = "Buildings domina el conteo de instancias; Vehicles y Heavy Machinery tienen soporte limitado. Los vacíos representan background, no una sexta clase."
        elif number == 5:
            rows = d["f_bbox_geometry_by_class"].pivot(index="class_id", columns="metric", values="median")
            names = d["f_bbox_geometry_by_class"].drop_duplicates("class_id").set_index("class_id").class_name
            body = table(["Clase", "Mediana ancho norm.", "Mediana alto norm.", "Mediana área norm.", "Mediana ratio norm."], [[names[i], r.normalized_width, r.normalized_height, r.normalized_area, r.aspect_ratio] for i, r in rows.iterrows()])
            finding = "La caracterización existente distingue ancho, alto, área y ratio por instancia/clase y conserva los valores extremos."
            state = "VALIDADO"
        elif number == 6:
            rows = []
            for encoder in ["dinov2", "clip"]:
                r = one(d[f"f_embedding_health_{encoder}"])
                rows.append(["DINOv2" if encoder == "dinov2" else "CLIP", r.model_id, f"{r.content_embedding_count} × {r.embedding_dimension}", r.pooling_strategy, r.record_count, "Completo y validado en recibo existente"])
            body = table(["Encoder", "Modelo", "Shape", "Representación", "Mapping de registros", "Estado"], rows)
            finding = "Ambos espacios completos cuentan con evidencia previa de cobertura, finitud, índices y L2 válidos; no se establece superioridad."
            state = "VALIDADO"
        elif number == 7:
            rows = []
            for _, r in d["v_global_similarity"].iterrows():
                nn = one(d["v_topk_global_summary"], extractor=r.extractor, metric="rank1_similarity")
                rows.append([r.extractor, r["count"], r["median"], r["mean"], nn["mean"], nn["median"]])
            body = table(["Encoder", "Pares únicos i<j", "Mediana global", "Media global", "Media rank-1", "Mediana rank-1"], rows)
            finding = "Los vecinos cercanos presentan mayor coseno que el par global típico dentro de cada encoder; las escalas no son directamente comparables entre encoders."
        elif number == 8:
            rows = []
            for encoder in ["DINOv2", "CLIP"]:
                nn = d["v_temporal_neighbors"]
                seq = d["v_sequence_similarity"]
                delta = d["v_frame_delta_similarity"]
                rows.append([encoder, f'{one(nn, extractor=encoder, neighborhood="rank1").same_sequence_percentage_all:.4f}%', f'{one(nn, extractor=encoder, neighborhood="topk").same_sequence_percentage_all:.4f}%', one(seq, extractor=encoder, relation="same_sequence")["median"], one(seq, extractor=encoder, relation="different_sequence")["median"], one(delta, extractor=encoder, frame_delta_bin="1")["median"], one(delta, extractor=encoder, frame_delta_bin=">100")["median"]])
            body = table(["Encoder", "Rank-1 misma secuencia", "Top-20 misma secuencia"], [r[:3] for r in rows])
            body += table(["Encoder", "Mediana misma sec.", "Mediana distinta sec.", "Mediana Δ=1", "Mediana Δ>100"], [[r[0]] + r[3:] for r in rows])
            finding = "Ambos encoders recuperan fuertemente la estructura secuencial inferida. La asociación visual/temporal es descriptiva: no valida timestamps ni confirma leakage temporal."
        elif number == 9:
            a = d["v_neighbor_agreement_summary"]
            body = table(["k", "Jaccard medio"], a[["k", "mean_jaccard"]].values.tolist())
            body += f'<p>Acuerdo exacto del vecino rank-1: <b>{one(a, k=1).exact_neighbor_match_percentage:.4f}%</b>.</p>'
            finding = "Los encoders recuperan estructura secuencial similar, pero no necesariamente los mismos vecinos; el acuerdo no determina cuál es mejor."
        elif number == 10:
            runs = d["r_runs"]
            body = table(["Método", "Runs", "Grid registrado", "Seeds"], [[method, len(g), ", ".join(sorted(g.label.unique())), "/".join(str(int(s)) for s in sorted(g.seed.unique()))] for method, g in runs.groupby("method")])
            refs = d["r_candidate_reference_metrics"].sort_values(["encoder", "method"])
            body += table(["Encoder", "Método", "Candidato", "T@20", "C@20", "Jaccard@20", "Estabilidad@20"], refs[["encoder", "method", "label", "trustworthiness@20", "continuity@20", "jaccard@20", "stability@20"]].values.tolist())
            body += f'<p><b>{len(runs)} ejecuciones.</b> T/C/J corresponden a seed 0; estabilidad es el Jaccard de vecinos entre los tres pares de seeds. T mide intrusiones de vecinos, C pérdidas y J el solapamiento local. Selección previa por frente no dominado y rangos de métricas, incluido Spearman.</p>'
            finding = "Las cuatro referencias seleccionadas permiten comparar preservación y estabilidad; no validan por sí solas clústeres ni justifican selección por apariencia."
        elif number == 11:
            s = d["c_summary"]
            body = table(["Algoritmo", "Runs"], [[k.upper(), v] for k, v in s["runs_by_algorithm"].items()])
            body += table(["Indicador", "Resultado"], [["Total runs", s["total_runs"]], ["Rango de clústeres", f'{s["clusters_min"]}–{s["clusters_max"]}'], ["Rango de noise", f'{pct(s["noise_fraction_min"], 4)}–{pct(s["noise_fraction_max"], 4)}'], ["Runs con un solo clúster", int(d["c_all_runs"].single_cluster.sum())], ["Candidatos Pareto", s["candidate_count"]], ["Comparaciones ARI / AMI", s["comparison_count"]]])
            body += table(["Métrica", "Qué pregunta responde / límite"], [
                ["Noise", "Fracción sin asignación densa; leer junto con cobertura."],
                ["Silhouette original", "Separación/cohesión en el encoder original L2; excluye ruido."],
                ["Coherencia visual", "Coseno intraclúster y retención de vecinos del encoder original."],
                ["Coherencia temporal", "Secuencia dominante y retención de pares con Δ inferido."],
                ["Neighbor retention", "Qué vecinos permanecen agrupados; reportar el universo elegible."],
                ["ARI / AMI", "Acuerdo ajustado por azar entre perturbaciones; all-points y common-clustered con cobertura, sin ground truth de escenas."]])
            finding = "El grid produjo alternativas con compromisos entre ruido, coherencia y estabilidad; no hay un algoritmo ganador global."
        elif number == 12:
            r = one(d["c_references"], reference="R6", encoder="dinov2", representation="tsne")
            body = table(["R6 · DINOv2 → t-SNE → OPTICS", "Resultado"], [["Clústeres", r.n_clusters_excluding_noise], ["Noise", pct(r.noise_fraction)], ["Contenidos agrupados / total", f"{r.clustered_points} / {r.total_points}"], ["Coseno intraclúster (ponderado por pares)", r.weighted_mean_intra_cluster_similarity], ["Fracción de secuencia dominante (ponderada)", r.weighted_dominant_sequence_fraction], ["Visual@10 (consultas agrupadas)", r["visual_neighbor_coherence@10"]]])
            finding = "El ejemplo muestra alta coherencia en la población agrupada, junto con una fracción elevada de ruido; es una ilustración, no el ganador ni el split C10."
        elif number == 13:
            runs, s = d["s_runs"], d["s_summary"]
            body = table(["Estrategia", "Runs"], [[name, len(g)] for name, g in runs.groupby("strategy")])
            new = runs.loc[runs.strategy.ne("historical")]
            aware = runs.loc[runs.strategy.eq("cluster_aware")]
            body += table(["Invariante", "Resultado acreditado"], [["Duplicados exactos cross-split, nuevas particiones", f"0 en {len(new)} / {len(new)}"], ["Fracturas del clúster fuente", f"0 en {len(aware)} / {len(aware)} cluster-aware"], ["Candidatos × semillas", f'{s["clustering_candidate_count"]} × {len(s["cluster_seeds"])}'], ["Noise policy", s["noise_policy"]], ["Asignación", "SciPy MILP; balance posterior de registros, clases y vacíos"]])
            body += f'<p><b>{s["pareto_count"]} configuraciones elegibles Pareto</b> cumplen las restricciones registradas de cobertura, tamaño y balance entre semillas. Minimizar correlación sin cumplir esas restricciones no basta para seleccionar una partición.</p>'
            finding = "La evidencia existente acredita indivisibilidad de contenido en las nuevas particiones y del grupo fuente en las cluster-aware; eso no equivale a eliminar toda correlación residual."
            state = "VALIDADO"
        elif number == 14:
            r = one(d["s_runs"], candidate_label="C10", seed=0)
            c = one(d["s_clustering_candidates"], candidate_label="C10")
            body = f'<p><b>{c.encoder.upper()} → {c.representation} → {c.algorithm.upper()}</b> · representante fijo seed 0.<br>split_space_id: <b>{r.split_space_id}</b> (identificador experimental, no hash de imagen).</p>'
            body += table(["Conteo", "Train", "Validation", "Test"], [[label] + [r[f"{s}_{suffix}"] for s in ["train", "val", "test"]] for label, suffix in [("Registros", "records"), ("Heavy Machinery", "heavy_machinery"), ("Anotaciones vacías", "empty")]])
            body += f'<p>Desviación media de clases: <b>{r.class_deviation_pp:.4f} pp</b>.</p>'
            finding = "C10 es la referencia principal de correlación visual extrema; conserva los tamaños históricos objetivo, con un compromiso medible de balance."
        elif number == 15:
            runs = d["s_runs"]
            hist, c10 = one(runs, strategy="historical"), one(runs, candidate_label="C10", seed=0)
            variations = d["s_metric_variation"]
            rows = []
            for label, key, is_percentage in [("NN cross-split medio · DINOv2", "dinov2_nn_mean", False), ("NN cross-split medio · CLIP", "clip_nn_mean", False), ("Pares top 0.1% · DINOv2", "dinov2_top001_pairs", False), ("Pares top 0.1% · CLIP", "clip_top001_pairs", False), ("Pares temporales cross-split Δ≤5", "temporal_at5", True)]:
                avg = one(variations, strategy="random_content", metric=key)
                if avg["n"] != 5:
                    raise ValueError("Random requiere las cinco semillas registradas")
                values = [hist[key], avg["mean"], c10[key]]
                rows.append([label] + ([pct(v) for v in values] if is_percentage else values))
            body = table(["Métrica residual", "Histórico", "Random: media seeds 0–4", "C10: seed 0"], rows)
            body += '<p>NN se calcula entre <b>contenidos distintos</b>, excluyendo la diagonal incluso en el histórico. Los duplicados exactos se auditan por separado. Top 0.1% usa la cohorte fija de pares de cada encoder; no es un umbral semántico de leakage. La tabla compara una referencia fija con la media random, no una estimación de significancia.</p>'
            finding = "C10 reduce fuertemente los pares visuales extremos y la similitud NN media. Su fracción temporal Δ≤5 supera la histórica: no hay mejora uniforme ni eliminación total de leakage."
        elif number == 16:
            rows = []
            for candidate in ["C10", "C12"]:
                r = one(d["s_runs"], candidate_label=candidate, seed=0)
                v = one(d["s_metric_variation"], candidate_label=candidate, metric="temporal_at5")
                one(d["d_split_context"], strategy=candidate, split_seed=0)
                rows.append([candidate, f"{r.train_records}/{r.val_records}/{r.test_records}", r.class_deviation_pp, r.dinov2_nn_mean, r.clip_nn_mean, pct(r.temporal_at5), f'{pct(v["min"])}–{pct(v["max"])}', f"{r.dinov2_top001_pairs}/{r.clip_top001_pairs}"])
            body = table(["Seed 0", "NN DINOv2", "NN CLIP", "Temporal Δ≤5", "Pares extremos DINO/CLIP"], [[r[0]] + r[3:6] + [r[7]] for r in rows])
            body += table(["Candidato", "Registros train/val/test · seed 0", "Desv. clases pp · seed 0", "Rango temporal Δ≤5 · 5 seeds"], [r[:3] + [r[6]] for r in rows])
            finding = "C12 (DINOv2/t-SNE/HDBSCAN) tiene menores NN medios y fracción temporal que C10, con peor balance y más pares extremos CLIP. Es candidato secundario, no una mejora en todos los indicadores."
        elif number == 17:
            env, plan, pilots = d["d_env"], d["d_plan"]["identity"]["config"], d["d_pilot_validation"]
            if "Ryzen 7 7730U" not in d["detector_doc"]:
                raise ValueError("Nombre del hardware sin evidencia documental")
            body = table(["Hardware registrado", "Evidencia existente"], [["CPU (registro documental)", "AMD Ryzen 7 7730U"], ["Cores / threads (environment.json)", f'{env["physical_cores"]} / {env["logical_cores"]}'], ["RAM utilizable", f'{env["ram_total_bytes"] / 2**30:.2f} GiB'], ["CUDA disponible en PyTorch", env["cuda_available"]]])
            body += table(["Pilotos", "Modelo", "Batch", "Resolución", "Epochs", "Train / val / test por piloto"], [[len(pilots), plan["model"], "/".join(map(str, sorted(pilots.batch.unique()))), "/".join(map(str, sorted(pilots.image_size.unique()))) + " px", "/".join(map(str, sorted(pilots.epochs_completed.unique()))), " / ".join(str(plan["compute"][f"small_pilot_{s}_images"]) for s in ["train", "val", "test"])]])
            body += '<p><b>PIPELINE VALIDATION ONLY · NO SCIENTIFIC RESULTS.</b> El protocolo está preparado; no se muestran Precision, Recall o mAP de pilotos como comparación científica.</p>'
            finding = "Cuatro pilotos pequeños acreditan funcionamiento operativo. Stage A completo, Stage B y la comparación controlada multi-seed no se han ejecutado."
            state = "PENDIENTE"
        elif number == 18:
            b = d["d_budget"]
            body = table(["Presupuesto previsto / estimación existente", "Valor"], [["Entrenamientos finales planeados", b["expected_full_runs"]], ["Extrapolación con última epoch", f'{b["estimated_last_epoch_full_training_hours"]:.2f} h CPU'], ["Extrapolación con wall time total del piloto", f'{b["estimated_full_training_hours"]:.2f} h CPU'], ["Equivalencia aproximada", f'{b["estimated_last_epoch_full_training_hours"] / 24:.0f}–{b["estimated_full_training_hours"] / 24:.0f} días continuos'], ["Seeds del detector previstas", " / ".join(map(str, d["d_plan"]["identity"]["config"]["training_seeds"]))]])
            finding = "El coste estimado motivó dejar Stage B pendiente. Son dos extrapolaciones, no un intervalo de confianza ni tiempo medido del experimento final; no incluyen evaluación/bootstrap final. No se sustituyó el protocolo por una comparación científica de una sola seed."
            state = "LIMITACIÓN"
        elif number == 19:
            if not d["f_meta"]["feature_engineering_completed"]:
                raise ValueError("Preparación sin evidencia completa")
            body = table(["Etapa", "Estado", "Alcance de la evidencia"], [
                ["Data understanding", "DONE", "Inventario, manifest, labels y duplicados auditados"],
                ["Feature engineering", "DONE", "Diagnósticos y caracterización; separados de embeddings"],
                ["DINOv2 / CLIP", "DONE", "Extracciones completas; recibos previos validados"],
                ["Similarity", "DONE", "Coseno y vecinos completos de ambos encoders"],
                ["Temporal validation", "PARTIAL", "Secuencia/índice inferidos; sin timestamps verificados"],
                ["Reduction", "DONE", "Grid y referencias evaluadas"],
                ["Clustering", "DONE WITH LIMITS", "Grid exploratorio; cobertura y estabilidad explícitas"],
                ["Cluster-aware splitting", "DONE WITH LIMITS", "Invariantes verificadas; correlación residual medida"],
                ["Detector controlled comparison", "PENDING", "Infraestructura + pilotos; cero resultados finales"]])
            finding = "El avance experimental llega a particiones evaluadas. Las limitaciones temporales y la comparación final del detector permanecen abiertas."
        elif number == 20:
            body = table(["Evidencia observada", "Interpretación permitida"], [
                ["Duplicados exactos cross-split históricos", "Contenidos repetidos entre train y evaluación"],
                ["Vecinos DINOv2/CLIP mayoritariamente de la misma secuencia", "Fuerte asociación con estructura secuencial inferida"],
                ["Random content-level: cero exact overlap", "La correlación visual alta no disminuye automáticamente"],
                ["C10: fuerte reducción de pares extremos", "Mejor separación visual en los indicadores medidos"],
                ["Temporal Δ≤5 no mejora uniformemente frente al histórico", "El beneficio visual no implica beneficio temporal universal"],
                ["Comparación final del detector pendiente", "No se conoce el efecto sobre Precision, Recall o mAP"]])
            finding = "Se ha medido la dependencia residual entre particiones; aún no se ha medido su efecto en la generalización del detector."
        elif number == 21:
            q = one(d["f_annotation_quality"])
            runs = d["s_runs"].loc[d["s_runs"].strategy.eq("cluster_aware")]
            no_proof = int(runs.solver_optimal.eq(False).sum())
            body = table(["Limitación", "Consecuencia"], [
                ["Sin timestamps verificados; procedencia temporal inferida", "No convertir índices a tiempo real ni confirmar leakage solo por proximidad"],
                [f"{q.conflicting_duplicate_groups} grupos duplicados con conflicto de anotación", "Se preservan sin corrección silenciosa"],
                ["Soporte limitado de Vehicles / Heavy Machinery", "Interpretar cobertura y métricas por clase con cautela"],
                ["Ruido de clustering por densidad", "Noise singleton puede dejar correlación residual entre splits"],
                [f"{no_proof}/{len(runs)} MILP: incumbentes factibles sin optimalidad probada", "Factibilidad no garantiza el mejor balance; gaps registrados"],
                ["Experimento final del detector pendiente", "No hay conclusión sobre cambio de mAP"],
                ["Capacidad de cómputo del portátil", "El protocolo multi-seed requiere un presupuesto viable"]])
            finding = "Los límites están documentados y restringen las conclusiones; un split válido por invariantes no garantiza independencia estadística total."
            state = "LIMITACIÓN"
        elif number == 22:
            body = "<ol><li>Conseguir acceso a GPU / VM si está disponible y verificar capacidad para el protocolo común.</li><li>Ejecutar el experimento controlado del detector con todas las seeds previstas.</li><li>Comparar Precision, Recall, mAP@50 y mAP@50–95.</li><li>Interpretar las métricas junto con la correlación residual y el soporte por clase.</li><li>Preparar resultados y conclusiones finales con sus límites.</li></ol>"
            finding = "El siguiente paso metodológico es ejecutar y evaluar la comparación controlada del detector; este reporte solo consolida evidencia existente."
            state = "PENDIENTE"
        else:
            raise ValueError("Sección desconocida")
        return body, finding, state


class VisibleHTML(HTMLParser):
    def __init__(self):
        super().__init__()
        self.skipped = 0
        self.text = []
        self.inputs = 0
        self.figures = 0
        self.headings = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in {"style", "script"}:
            self.skipped += 1
        classes = attrs.get("class", "").split()
        # Lab wraps rendered Markdown in jp-InputArea too. Code editors are the
        # actual input boundary; classic exports use input_area/highlight.
        self.inputs += int(any(c in classes for c in ["input_area", "jp-CodeMirrorEditor", "highlight"]))
        self.figures += int(tag == "figure" and "progress-figure" in classes)
        self.headings += int(tag == "h2")
        for key in ["alt", "title"]:
            if key in attrs:
                self.text.append(attrs[key])

    def handle_endtag(self, tag):
        if tag in {"style", "script"}:
            self.skipped -= 1

    def handle_data(self, data):
        if not self.skipped:
            self.text.append(data)


def audit_html(body: str, sensitive_ids=()):
    parser = VisibleHTML()
    parser.feed(body)
    visible = " ".join(parser.text)
    patterns = [r"[A-Za-z]:[/\\]", r"/(?:home|Users|mnt)/", r"\b[0-9a-fA-F]{40,64}\b"]
    if any(re.search(pattern, visible) for pattern in patterns) or any(value and value in visible for value in sensitive_ids):
        raise ValueError("Privacy check failed: private identifier/path/hash in visible report")
    if parser.inputs:
        raise ValueError("HTML contains visible code input containers")
    return {"code_hidden": True, "visible_privacy_passed": True, "sections": parser.headings, "figures": parser.figures}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check existing inputs and selections without building outputs")
    args = parser.parse_args()
    review = Review(ROOT)
    for number in DEPS:
        review.section(number)
    errors = bool(review.issues or any(s["state"] != "available" for s in review.sections.values()))
    if args.check:
        print(f"Sources: {len(review.inventory)}; missing/invalid: {len(review.issues)}; sections available: {sum(s['state'] == 'available' for s in review.sections.values())}/22")
        for key, issue in review.issues.items():
            print(f"{SOURCES[key]}: {issue}")
        for number, section in review.sections.items():
            if section["state"] != "available":
                print(f"Section {number}: {section['state']}")
        raise SystemExit(int(errors))

    # Reporting libraries remain optional and never enter ordinary offline tests.
    import os

    import nbformat
    from check_notebook_source import check_notebook_source
    from nbclient import NotebookClient
    from nbconvert import HTMLExporter

    check_notebook_source(ROOT / SOURCE)
    for key, directory in [("IPYTHONDIR", "ipython"), ("JUPYTER_RUNTIME_DIR", "jupyter-runtime")]:
        path = ROOT / ".cache" / directory
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    notebook = nbformat.read(ROOT / SOURCE, as_version=4)
    NotebookClient(notebook, timeout=180, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
    nbformat.validate(notebook)
    exporter = HTMLExporter(template_name="lab", exclude_input=True, exclude_input_prompt=True, exclude_output_prompt=True)
    body, _ = exporter.from_notebook_node(notebook, resources={"metadata": {"name": "FLIR Leakage-Aware Pipeline — Progress Review"}})
    qa = audit_html(body, review.sensitive_ids)
    if qa["sections"] != 22 or (not errors and qa["figures"] != 15):
        raise ValueError("Unexpected report structure")
    if not review.unchanged():
        raise ValueError("An existing source changed during the build; output not published")
    output = ROOT / OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    executed = output / "progress_review.executed.ipynb"
    target = output / "progress_review.html"
    nbformat.write(notebook, executed)
    target.write_text(body, encoding="utf-8")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    receipt = {
        "created_at": datetime.now(UTC).isoformat(), "source_git_commit": commit,
        "policy": "existing evidence only; no experimental stages or numerical metrics recomputed",
        "validation_scope": "availability, recorded checksums, aggregate consistency, manifest identity/counts, HTML privacy and structure; previous numerical validations reused",
        "complete": not errors, "source_notebook_sha256": digest(ROOT / SOURCE),
        "sources": review.inventory, "source_issues": review.issues, "sections": review.sections,
        "qa": qa, "existing_sources_unchanged": True,
        "output_sha256": {p.name: digest(p) for p in [executed, target]},
    }
    (output / "build_receipt.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Built {OUTPUT / target.name}; 22 sections; {qa['figures']} figures; complete={not errors}")
    raise SystemExit(int(errors))


if __name__ == "__main__":
    main()
