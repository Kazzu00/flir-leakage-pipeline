"""Immutable external ground-truth evidence, bound to proposed source publications."""

from __future__ import annotations

import base64
import io
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import PIL
import scipy
from PIL import Image

from flir_pipeline.detection.annotation_review import render_comparison, source_bytes
from flir_pipeline.detection.common_test_storage import signatures
from flir_pipeline.detection.ground_truth_gallery import gallery_indexes, render_case
from flir_pipeline.detection.ground_truth_quality import (
    VERSION,
    annotation_facts,
    annotation_sensitivity,
    area_summary,
    descriptive_support,
    review_template,
    validate_human_review,
)
from flir_pipeline.features.diagnostics import _phash
from flir_pipeline.similarity.storage import (
    file_sha256,
    read_json,
    stable_id,
    write_json,
)
from flir_pipeline.splitting.fixed_test_storage import verify_fixed_test
from flir_pipeline.splitting.labeled import load_labeled_audit, verify_original_sources


def implementation():
    root = Path(__file__).parents[1]
    names = (
        "detection/ground_truth_quality.py",
        "detection/ground_truth_gallery.py",
        "detection/ground_truth_quality_storage.py",
        "detection/annotation_review.py",
        "detection/common_test.py",
        "detection/common_test_storage.py",
        "splitting/labeled.py",
        "splitting/fixed_test_storage.py",
        "splitting/fixed_test.py",
        "splitting/construction.py",
        "data/yolo_labels.py",
        "data/classes.py",
        "features/diagnostics.py",
    )
    return {
        "source_sha256": {n: file_sha256(root / n) for n in names},
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pillow": PIL.__version__,
        "scipy": scipy.__version__,
    }


def _csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False, lineterminator="\n")


def _observations(path, test_ids):
    payload = read_json(path) if path else {"observations": []}
    if set(payload) != {"observations"} or not isinstance(
        payload["observations"], list
    ):
        raise ValueError("Invalid preliminary observations")
    for note in payload["observations"]:
        if (
            set(note) != {"frame_ids", "source", "observation", "status"}
            or note["status"] != "preliminary_not_adjudication"
            or not isinstance(note["frame_ids"], list)
            or not note["frame_ids"]
            or not set(note["frame_ids"]) <= test_ids
            or not isinstance(note["source"], str)
            or not note["source"].strip()
            or not isinstance(note["observation"], str)
            or not note["observation"].strip()
        ):
            raise ValueError("Preliminary observation needs attribution and test scope")
    return payload


def build_quality(
    audit,
    proposal,
    review,
    partitions,
    source_root,
    output,
    *,
    observations=None,
    evidence=None,
    expected_test=372,
    expected_arms=30,
):
    """Recover original test bytes and summarize existing assignments, never split."""
    roots = {
        "audit": audit,
        "proposal": proposal,
        "review": review,
        "partitions": partitions,
    }
    target = output.resolve()
    protected = [
        *roots.values(),
        source_root,
        *([observations] if observations else []),
        *([evidence] if evidence else []),
    ]
    if (
        output.exists()
        or target.is_relative_to(Path(__file__).resolve().parents[3])
        or any(
            target.is_relative_to(p.resolve()) or p.resolve().is_relative_to(target)
            for p in protected
        )
    ):
        raise ValueError("Fresh external output disjoint from sources required")
    before = {n: signatures(p) for n, p in roots.items()}
    suite = verify_fixed_test(partitions)
    if suite["configurations"] != expected_arms or suite["ready_for_training"]:
        raise ValueError("Expected complete blocked provisional suite")
    for name in ("audit", "proposal", "review"):
        if before[name] != signatures(partitions / "sources" / name):
            raise ValueError("Partition/source binding differs")
    source = load_labeled_audit(audit)
    verify_original_sources(source, source_root, evidence)
    archives = source.summary["configuration"]["sources"]
    plan = read_json(proposal / "proposal.json")
    test = (
        pd.read_parquet(proposal / "test_records.parquet")
        .sort_values("frame_id")
        .reset_index(drop=True)
    )
    if len(test) != expected_test or plan["readiness"]["state"] != "PROPOSED":
        raise ValueError("Unexpected proposed test")
    labels = {
        r["label_id"]: r for r in read_json(audit / "annotation_audit.json")["records"]
    }
    frames = {
        r["frame_id"]: r
        for r in read_json(audit / "candidate_manifest.json")["records"]
    }
    prior = read_json(review / "manifest.json")
    for case in prior["cases"]:
        if case["conflict_id"] != "conflict-" + case["content_id"]:
            raise ValueError("Conflict identity mismatch")
        for key in ("A", "B"):
            record = frames[case[key]["occurrence"]["frame_id"]]
            label = labels[record["label_ids"][0]]
            if case[key]["occurrence"] != record or case[key]["annotation"] != label:
                raise ValueError("Original conflict provenance differs")
            raw = source_bytes(source_root, record, record["image_sha256"])
            raw_label = source_bytes(source_root, label, label["label_sha256"])
            facts = annotation_facts(raw_label, record["width"], record["height"])
            boxes = [
                [b["class_id"], *[b[k] for k in ("x", "y", "w", "h")]]
                for b in facts["boxes"]
            ]
            if boxes != case[key]["boxes"]:
                raise ValueError("Original conflict boxes differ")
        expected = review / "cases" / case["conflict_id"] / "comparison.png"
        if expected.read_bytes() != render_comparison(raw, case):
            raise ValueError("Prior conflict visual reproduction mismatch")
    conflicts = {
        c["A"]["occurrence"]["frame_id"]: c["conflict_id"] for c in prior["cases"]
    }
    notes = _observations(observations, set(test.frame_id))
    identity = {
        "schema_version": VERSION,
        "sources": before,
        "archives": archives,
        "plan_id": plan["plan_id"],
        "test_id": plan["test_id"],
        "expected_test": expected_test,
        "expected_arms": expected_arms,
        "implementation": implementation(),
        "observations": notes,
    }
    audit_id = stable_id(identity)
    exposure = set(
        source.manifest.loc[
            (source.manifest.cohort == "historical")
            & (source.manifest.original_split == "train"),
            "content_id",
        ]
    )
    exposed = set(test.content_id) & exposure
    output.mkdir(parents=True)
    # Preserve the entire prior evidence package byte-for-byte, including pending forms.
    shutil.copytree(review, output / "prior_review")
    write_json(output / "observations.json", notes)
    support, concentration, sensitivity, mapping, selection = descriptive_support(
        test,
        source.statistics,
        source.relations,
        "common-test",
        phash_threshold=source.summary["configuration"]["phash_threshold"],
    )
    selections = {"common-test": selection}
    maps = [mapping]
    groups = mapping.set_index("frame_id").candidate_group.to_dict()
    screened_groups = mapping.set_index("frame_id").perceptual_group.to_dict()
    multiplicity = source.manifest.content_id.value_counts().to_dict()
    inventory, instances = [], []
    for frame in test.frame_id:
        record = frames[frame]
        if len(record["label_ids"]) != 1:
            raise ValueError("Ambiguous test annotation")
        label = labels[record["label_ids"][0]]
        raw = source_bytes(source_root, record, record["image_sha256"])
        label_raw = source_bytes(source_root, label, label["label_sha256"])
        with Image.open(io.BytesIO(raw)) as image:
            if _phash(image) != record["phash"]:
                raise ValueError(
                    "Original test pHash differs from recorded screening representation"
                )
        if label_raw != base64.b64decode(label["original_bytes_base64"], validate=True):
            raise ValueError("Original annotation differs from source audit")
        facts = annotation_facts(label_raw, record["width"], record["height"])
        render_case(output, raw, label_raw, record, facts, conflicts.get(frame))
        inventory.append(
            {
                **record,
                **{k: v for k, v in facts.items() if k != "boxes"},
                "candidate_group": groups[frame],
                "perceptual_group": screened_groups[frame],
                "original_phash_verified": True,
                "exact_occurrences_all_sources": multiplicity[record["content_id"]],
                "historical_training_exposure": record["content_id"] in exposed,
                "conflict_id": conflicts.get(frame, ""),
                "human_state": "pendiente",
            }
        )
        instances.extend(
            {"frame_id": frame, "content_id": record["content_id"], **b}
            for b in facts["boxes"]
        )
    for run in suite["runs"]:
        arm = run["arm"]
        membership_path = partitions / "runs" / arm / "occurrence_memberships.parquet"
        if not membership_path.is_file():
            raise ValueError(
                "Complete assignments required; solver-failed arms cannot be silently omitted"
            )
        memberships = pd.read_parquet(membership_path)
        actual_test = (
            memberships.loc[memberships.new_split == "test"]
            .drop(columns="new_split")
            .sort_values("frame_id")
            .reset_index(drop=True)
        )
        pd.testing.assert_frame_equal(actual_test, test)
        for role in ("train", "val"):
            scope = f"{arm}/{role}"
            result = descriptive_support(
                memberships.loc[memberships.new_split == role],
                source.statistics,
                source.relations,
                scope,
                phash_threshold=source.summary["configuration"]["phash_threshold"],
            )
            rows, counts, scenarios, members, selected = result
            support.extend(rows)
            concentration.extend(counts)
            sensitivity.extend(scenarios)
            maps.append(members)
            selections[scope] = selected
    forms = review_template(audit_id, test.frame_id.tolist())
    validate_human_review(forms, audit_id, test.frame_id.tolist())
    write_json(output / "human_review.template.json", forms)
    _csv(output / "human_review.template.csv", forms["cases"])
    write_json(output / "test_inventory.json", {"records": inventory})
    _csv(output / "test_inventory.csv", inventory)
    write_json(output / "test_instances.json", {"records": instances})
    _csv(output / "test_instances.csv", instances)
    _csv(output / "support.csv", support)
    _csv(output / "group_concentration.csv", concentration)
    _csv(output / "rare_class_sensitivity.csv", sensitivity)
    pd.concat(maps, ignore_index=True).to_csv(
        output / "descriptive_groups.csv", index=False, lineterminator="\n"
    )
    write_json(output / "group_selection.json", selections)
    write_json(
        output / "test_relations.json",
        {
            "relations": [
                e
                for e in source.relations
                if e["left_frame_id"] in set(test.frame_id)
                or e["right_frame_id"] in set(test.frame_id)
            ]
        },
    )
    areas = area_summary(instances)
    _csv(output / "test_area_distribution.csv", areas)
    alternatives = annotation_sensitivity(support[:5], prior["cases"])
    _csv(output / "annotation_sensitivity.csv", alternatives)
    summary = {
        "schema_version": VERSION,
        "audit_id": audit_id,
        "identity": identity,
        "verdict": "APROBADO PARA REVISIÓN HUMANA",
        "test_state": "PROPOSED",
        "ready_for_training": False,
        "adjudicated_labels_applied": False,
        "test_images": len(test),
        "test_contents": test.content_id.nunique(),
        "test_empty_annotations": sum(r["empty"] for r in inventory),
        "historically_exposed_test_contents": len(exposed),
        "retrospective_test": True,
        "independent_test": False,
        "partitions": len(suite["runs"]),
        "source_failed_configurations": suite["failed_configurations"],
        "pending_conflicts": len(conflicts),
        "human_reviews_completed": 0,
        "test_support": support[:5],
        "test_area_distribution": areas,
        "single_positive_image_validation": [
            r
            for r in support
            if r["scope"].endswith("/val") and r["single_positive_image"]
        ],
        "statistical_sufficiency": "NOT_ESTABLISHED",
        "checkpoint_selection_eligible": False,
        "actual_object_omissions_detected": False,
        "perceptual_screening_threshold": source.summary["configuration"][
            "phash_threshold"
        ],
        "test_new_perceptual_pairs": len(
            selections["common-test"]["new_within_subset_screening"]["relations"]
        ),
        "hypothetical_annotation_sensitivity": [
            r for r in alternatives if r["scenario"] == "all_stored_B_annotations"
        ],
    }
    write_json(output / "manifest.json", summary)
    gallery_indexes(output, inventory, conflicts, len(exposed))
    _report(output, summary, support)
    verify_original_sources(source, source_root, evidence)
    if before != {n: signatures(p) for n, p in roots.items()} or signatures(
        review
    ) != signatures(output / "prior_review"):
        raise ValueError("Source publications changed during audit")
    write_json(
        output / "receipt.json",
        {
            "schema_version": VERSION,
            "audit_id": audit_id,
            "artifacts": signatures(output),
        },
    )
    return summary


def _report(output, summary, support):
    header = "| Clase | Instancias | Imágenes | Contenidos | Grupos con relaciones previas | Grupos pHash internos | Mayor fracción de instancias en grupo pHash |\n|---|---:|---:|---:|---:|---:|---:|\n"
    rows = "\n".join(
        f"| {r['class_name']} | {r['instances']} | {r['images']} | {r['contents']} | {r['candidate_positive_groups']} | {r['perceptual_positive_groups']} | {r['perceptual_max_instance_share']} |"
        for r in summary["test_support"]
    )
    val = [r for r in support if r["scope"].endswith("/val")]
    wide = "| Validation | Vehicles imágenes | Buildings | Roads | Rivers | Heavy Machinery |\n|---|---:|---:|---:|---:|---:|\n"
    for scope in dict.fromkeys(r["scope"] for r in val):
        wide += (
            "| "
            + scope
            + " | "
            + " | ".join(str(r["images"]) for r in val if r["scope"] == scope)
            + " |\n"
        )
    text = f"""# Calidad y suficiencia de ground truth

**APROBADO PARA REVISIÓN HUMANA.** Auditoría `{summary["audit_id"]}`. No aprueba el test ni habilita entrenamiento.

## Evidencia automática

Se recuperaron y verificaron SHA-256 de {summary["test_images"]} imágenes y sus etiquetas originales; {summary["test_empty_annotations"]} anotaciones vacías. ZIP completos y publicaciones conservan sus checksums. Inventario e instancias mantienen localizadores, clases y coordenadas sin corrección. Las galerías muestran original y anotación actual a resolución nativa. El parser verifica sintaxis, geometría y vocabulario; no puede determinar si falta un objeto real.

{header}{rows}

Áreas normalizadas y en píxeles por caja en `test_instances.csv`; cuantiles por clase en `test_area_distribution.csv`. Contacto con bordes y cajas idénticas son hechos geométricos, no diagnósticos de error. No se usó un umbral arbitrario de caja grande ni de suficiencia. Ordenar áreas y revisar las galerías permite investigar cajas grandes, parciales o que agrupen objetos; no se decide automáticamente su validez. Las etiquetas vacías requieren revisión de imagen y contexto, no corrección automática.

## Test retrospectivo y concentración

La consolidación previa contiene principalmente relaciones cruzadas Adendo/histórico: no es un cribado exhaustivo dentro del test histórico. Por ello se calculan adicionalmente pares pHash64 dentro de cada conjunto, al umbral de Hamming previamente documentado ({summary["perceptual_screening_threshold"]}). En test se obtienen {summary["test_new_perceptual_pairs"]} pares candidatos nuevos. Se verifica el pHash desde los 372 originales; en train/validation se utilizan los pHash registrados y los checksums de las fuentes. `candidate` identifica las relaciones previas; `perceptual`, este cribado diagnóstico nuevo. Los pares nuevos, umbral y componentes constan en `group_selection.json` y `descriptive_groups.csv`. No cambian S1/S2/S3 ni se incorporan al cierre protegido publicado. Las revisiones negativas se respetan al formar los componentes diagnósticos. Un componente puede ser una cadena de similitudes: no garantiza similitud de todos los pares, identidad de objetos ni una secuencia común.

**{summary["historically_exposed_test_contents"]} contenidos del test estuvieron expuestos a entrenamiento histórico. El test es retrospectivo, no independiente.** No permite una afirmación prospectiva de generalización no vista.

`support.csv` contiene test común y train/validation de las {summary["partitions"]} configuraciones. Imágenes, contenidos exactos y componentes candidatos son unidades distintas. Los componentes candidatos se inducen dentro de cada subconjunto, sin puentes a través de nodos excluidos, reutilizando las relaciones auditadas. Son hipótesis de dependencia; ni el número de componentes ni el número de contenidos equivale a tamaño muestral independiente. No se establecieron vídeos/secuencias independientes mediante nombres de archivo.

`group_concentration.csv` identifica cada grupo y su contribución por clase. `rare_class_sensitivity.csv` retira aritméticamente un grupo positivo exacto o candidato cada vez para Vehicles y Heavy Machinery: muestra soporte restante y fracción perdida. No produce nuevos splits, intervalos de confianza, precisión, recall ni AP. Sin predicciones verificadas no hay análisis de error del detector.

`annotation_sensitivity.csv` calcula por separado el cambio de conteos si se utilizara B en cada conflicto o en todos. Se verifican las cajas y las comparaciones previas desde originales antes del cálculo. Son escenarios hipotéticos de etiquetas existentes, no adjudicaciones ni estimaciones de objetos verdaderos. Ningún escenario se aplica al test.

## Validation y selección de checkpoints

{wide}

Todos los conteos, también train, están en `support.csv`; suficiencia y elegibilidad permanecen bloqueadas en cada fila. Una sola imagen positiva no permite evaluar variación entre imágenes para esa clase. Varias cajas de una imagen no subsanan esa limitación. Las semillas históricas S1 3 y 4 deben examinarse especialmente; no se descartan ni se reparan. Contener las cinco clases acredita cobertura, no suficiencia estadística.

Seleccionar checkpoints exclusivamente con validation es el contrato correcto para evitar usar test en la selección, pero estos datos no acreditan que validation tenga soporte y diversidad suficientes para una selección estable por clase. No se ha evaluado estabilidad de checkpoints ni se elige un brazo ganador.

Alternativas para decisión metodológica humana: preespecificar regla y presupuesto de selección usando solo validation; presentar métricas por clase junto con soporte y concentración; registrar sensibilidad entre semillas sin elegir retrospectivamente la más favorable; recopilar y adjudicar ejemplos adicionales de secuencias independientes; estudiar validación agrupada solo cuando haya unidades de procedencia justificadas. La ausencia de soporte suficiente puede obligar a declarar no evaluable una comparación por clase. No se calculan intervalos ni tamaños efectivos suponiendo independencia de imágenes correlacionadas. Ninguna alternativa se ejecuta aquí ni altera las particiones existentes.

## Observaciones visuales y adjudicaciones

La revisión previa de {summary["pending_conflicts"]} conflictos se conserva byte por byte en `prior_review/`. `observations.json` distingue observaciones preliminares aportadas y atribuidas de los hallazgos automáticos. La preferencia preliminar por B, si fue aportada, no se convierte en adjudicación. Esta auditoría no inventa inspecciones visuales exhaustivas ni revisores humanos.

`human_review.template.json` y CSV dejan todos los casos pendientes: revisor, fecha, criterios, observaciones, posibles omisiones/confusiones, objetos agrupados, cajas parciales, evidencia de vídeo, justificación y propuesta. Editar una copia externa conserva el paquete original. La plantilla previa de decisiones A/B permanece intacta. Se necesita contexto de vídeo y criterios consistentes para decidir presencia, separación e identidad de los objetos.

Una corrección futura requiere capa adjudicada nueva con vínculo a imagen/etiqueta original, nuevo checksum, revisor, fecha, criterio, justificación y evidencia. Debe producir nueva identidad y verificación del protocolo antes de cualquier uso; no se publica esa capa aquí. Siguen pendientes adjudicación, suficiencia por clase, dependencias, aprobación metodológica y freeze. `ready_for_training=false`; test `PROPOSED`. Sin entrenamiento, cambios de etiquetas, selección de ganador ni modificación del frontend o exports.

## Reproducción

`receipt.json` vincula el inventario completo del paquete. La CLI de verificación comprueba recibos y reproduce tablas, plantillas y píxeles renderizados desde fuentes verificadas en un directorio temporal externo. La identidad incluye fuentes, implementación, versiones y observaciones atribuidas. Un recibo de esta auditoría no es un freeze del protocolo.
"""
    (output / "report.md").write_text(text, encoding="utf-8")


def verify_quality(
    directory, audit, proposal, review, partitions, source_root, *, evidence=None
):
    receipt = read_json(directory / "receipt.json")
    actual = signatures(directory)
    actual.pop("receipt.json", None)
    if receipt.get("schema_version") != VERSION or actual != receipt.get("artifacts"):
        raise ValueError("Quality package checksum/file inventory mismatch")
    manifest = read_json(directory / "manifest.json")
    identity = manifest["identity"]
    with tempfile.TemporaryDirectory(prefix="flir-quality-replay-") as temp:
        target = Path(temp) / "audit"
        rebuilt = build_quality(
            audit,
            proposal,
            review,
            partitions,
            source_root,
            target,
            observations=directory / "observations.json",
            evidence=evidence,
            expected_test=identity["expected_test"],
            expected_arms=identity["expected_arms"],
        )
        if (
            signatures(target) != signatures(directory)
            or rebuilt["audit_id"] != receipt["audit_id"]
        ):
            raise ValueError("Quality reproduction mismatch or altered blocked states")
    return {
        "audit_id": rebuilt["audit_id"],
        "reproduced": True,
        "test_images": rebuilt["test_images"],
        "partitions": rebuilt["partitions"],
        "ready_for_training": False,
        "test_state": "PROPOSED",
    }
