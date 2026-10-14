"""Read-only evidence for protected conflicts; never adjudicate or mutate a test."""

from __future__ import annotations

import base64
import csv
import hashlib
import html
import io
import json
import tempfile
import textwrap
import zipfile
from collections import Counter
from datetime import date
from pathlib import Path, PureWindowsPath

import pandas as pd
import PIL
from PIL import Image, ImageDraw, ImageFont

from flir_pipeline.data.classes import class_display_name
from flir_pipeline.data.yolo_labels import validate_label_bytes
from flir_pipeline.detection.common_test_storage import signatures, verify_common_test
from flir_pipeline.similarity.storage import (
    file_sha256,
    read_json,
    stable_id,
    write_json,
)
from flir_pipeline.splitting.labeled import load_labeled_audit

VERSION = "protected_annotation_review_v1"
DECISIONS = {
    "pendiente",
    "conservar_anotacion_A",
    "conservar_anotacion_B",
    "proponer_correccion_manual",
    "excluir_de_evaluacion",
}
CAUTION = (
    "PENDIENTE de revision humana. A=test propuesto; B=cuarentena. "
    "Una caja exclusiva no demuestra ausencia real de objeto. Inspeccionar contexto "
    "de video para presencia, identidad y separacion de objetos; nombres de frames "
    "solo aportan una heuristica temporal. No hay adjudicacion ni permiso de entrenamiento."
)


def source_bytes(root: Path, locator: dict, digest: str) -> bytes:
    """Read a unique raw ZIP locator, including its recorded header offset."""
    name = locator["source_archive"]
    member = locator["zip_member_name"]
    normalized = member.replace("\\", "/")
    path = (root / name).resolve()
    if (
        not name
        or any(c in name for c in "/\\:")
        or not path.is_relative_to(root.resolve())
        or normalized.startswith("/")
        or PureWindowsPath(normalized).drive
        or ".." in normalized.split("/")
        or "\x00" in normalized
    ):
        raise ValueError("Unsafe original locator")
    with zipfile.ZipFile(path) as archive:
        matches = [i for i in archive.infolist() if i.filename == member]
        if (
            len(matches) != 1
            or matches[0].header_offset != locator["zip_header_offset"]
            or (matches[0].external_attr >> 16) & 0o170000 == 0o120000
        ):
            raise ValueError("Missing or ambiguous original locator")
        raw = archive.read(matches[0])
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("Original member checksum mismatch")
    return raw


def project_box(box, width: int, height: int) -> tuple[float, ...]:
    """Project source precision onto the raw raster, without clipping or EXIF rotation."""
    _, x, y, w, h = box
    return (
        (x - w / 2) * width,
        (y - h / 2) * height,
        (x + w / 2) * width,
        (y + h / 2) * height,
    )


def differences(a, b) -> dict:
    """Multiset differences; no IoU matching that invents object correspondence.

    Equal geometry can establish a label-class difference, not physical identity.
    A sole same-class residual pair permits a geometric comparison only. Multiple
    residual candidates remain explicitly ambiguous, with all original indices.
    """
    a, b = list(map(tuple, a)), list(map(tuple, b))
    left, right, common = list(range(len(a))), list(range(len(b))), []
    for i in list(left):
        match = next((j for j in right if a[i] == b[j]), None)
        if match is not None:
            common.append([i, match])
            left.remove(i)
            right.remove(match)
    only_a, only_b = left.copy(), right.copy()
    changes = []
    for i in list(left):
        candidates = [j for j in right if a[i][1:] == b[j][1:]]
        if len(candidates) == 1:
            j = candidates[0]
            if sum(a[k][1:] == b[j][1:] for k in left) == 1:
                changes.append({"kind": "class_change_same_geometry", "a": i, "b": j})
                left.remove(i)
                right.remove(j)
    if len(left) == len(right) == 1 and a[left[0]][0] == b[right[0]][0]:
        i, j = left.pop(), right.pop()
        changes.append(
            {
                "kind": "geometry_difference_single_residual_pair",
                "a": i,
                "b": j,
                "delta_xywh_B_minus_A": [b[j][k] - a[i][k] for k in range(1, 5)],
                "physical_object_correspondence": "not_established",
            }
        )
    return {
        "common_index_pairs": common,
        "only_A_indices": only_a,
        "only_B_indices": only_b,
        "unambiguous_comparisons": changes,
        "unmatched_A_indices": left,
        "unmatched_B_indices": right,
        "ambiguous_correspondence": bool(left and right),
        "annotation_presence_difference": bool(left) != bool(right),
        "box_count_A": len(a),
        "box_count_B": len(b),
        "class_counts_A": dict(Counter(str(x[0]) for x in a)),
        "class_counts_B": dict(Counter(str(x[0]) for x in b)),
        "real_object_absence": "not_established",
        "video_context_required": bool(only_a or only_b),
    }


def validate_decisions(document: dict, manifest: dict) -> dict:
    """Validate attributed proposals only; completeness never approves a protocol."""
    if document.get("review_id") != manifest["review_id"]:
        raise ValueError("Decisions bound to another review")
    expected = {c["conflict_id"] for c in manifest["cases"]}
    rows = document["cases"]
    if len(rows) != len(expected) or {r["conflict_id"] for r in rows} != expected:
        raise ValueError("Missing, duplicate or foreign decision cases")
    pending = 0
    fields = {
        "conflict_id",
        "reviewer",
        "review_date",
        "decision",
        "justification",
        "additional_evidence",
    }
    for row in rows:
        if set(row) != fields or row["decision"] not in DECISIONS:
            raise ValueError("Invalid decision fields or state")
        if any(not isinstance(row[k], str) for k in fields):
            raise ValueError("Decision fields must be strings")
        if row["decision"] == "pendiente":
            pending += 1
            continue
        if any(
            not row[k].strip()
            for k in ("reviewer", "review_date", "justification", "additional_evidence")
        ):
            raise ValueError("Incomplete human decision")
        date.fromisoformat(row["review_date"])
    return {
        "pending": pending,
        "complete_proposals": len(rows) - pending,
        "approval_granted": False,
        "ready_for_training": False,
        "adjudicated_labels_applied": False,
    }


def render_comparison(raw: bytes, case: dict) -> bytes:
    """Native-sized panels share identical decoded image pixels and source coordinates."""
    with Image.open(io.BytesIO(raw)) as opened:
        image = opened.convert("RGB")
    w, h = image.size
    panel_w = max(w, 640)
    font = ImageFont.load_default(size=16)
    headers = []
    for key in ("A", "B"):
        occurrence = case[key]["occurrence"]
        label = case[key]["annotation"]
        headers.append(
            f"{key}: {'TEST PROPUESTO' if key == 'A' else 'CUARENTENA'}\n"
            f"Imagen: {occurrence['source_archive']}::{occurrence['zip_member_name']}\n"
            f"YOLO: {label['source_archive']}::{label['zip_member_name']}"
        )
    headers.append(
        "Diferencias: A naranja / B cian\nIndices base 0; cajas comunes omitidas.\n"
        "Superposicion descriptiva; no asigna identidad de objetos."
    )
    wrapped = [
        "\n".join(textwrap.fill(line, max(35, panel_w // 9)) for line in s.splitlines())
        for s in headers
    ]
    top = max(s.count("\n") + 1 for s in wrapped) * 22 + 25
    bottom_text = textwrap.fill(CAUTION, panel_w * 3 // 9)
    bottom = (bottom_text.count("\n") + 1) * 22 + 85
    canvas = Image.new("RGB", (panel_w * 3, top + h + bottom), "#171c24")
    draw = ImageDraw.Draw(canvas)
    for panel in range(3):
        xoffset = panel * panel_w
        canvas.paste(image, (xoffset, top))
        draw.multiline_text(
            (xoffset + 8, 8), wrapped[panel], font=font, fill="white", spacing=4
        )
        for key, color in (("A", "#ff9d28"), ("B", "#00efff")):
            if panel < 2 and key != ("A", "B")[panel]:
                continue
            boxes = case[key]["boxes"]
            indices = (
                range(len(boxes))
                if panel < 2
                else case["differences"][f"only_{key}_indices"]
            )
            for index in indices:
                box = boxes[index]
                x1, y1, x2, y2 = project_box(box, w, h)
                draw.rectangle(
                    (xoffset + x1, top + y1, xoffset + x2, top + y2),
                    outline=color,
                    width=2,
                )
                label = f"{key}[{index}] {class_display_name(box[0])}"
                ty = (
                    max(top, top + y1 - 20)
                    if key == "A"
                    else min(top + h - 20, top + y2)
                )
                tx = xoffset + min(max(0, x1), max(0, panel_w - len(label) * 9))
                draw.text(
                    (tx, ty),
                    label,
                    font=font,
                    fill=color,
                    stroke_width=1,
                    stroke_fill="black",
                )
    d = case["differences"]
    footer = (
        f"{case['conflict_id']} | cajas A={d['box_count_A']}, B={d['box_count_B']} | "
    )
    footer += f"correspondencia ambigua={d['ambiguous_correspondence']}\n" + bottom_text
    draw.multiline_text((10, top + h + 15), footer, font=font, fill="white", spacing=4)
    stream = io.BytesIO()
    canvas.save(stream, format="PNG")
    return stream.getvalue()


def _bound_sources(audit: Path, proposal: Path, acceptance: Path):
    verify_common_test(proposal)
    source = load_labeled_audit(audit)
    if signatures(audit) != signatures(proposal / "sources" / "audit"):
        raise ValueError("Audit differs from proposal snapshot")
    report = read_json(acceptance / "independent_acceptance_audit.json")
    if report["source"]["audit_summary_sha256"] != file_sha256(
        audit / "summary.json"
    ) or report["source"]["internal_summary_sha256"] != file_sha256(
        proposal / "summary.json"
    ):
        raise ValueError("Acceptance evidence belongs to different sources")
    plan = read_json(proposal / "proposal.json")
    if (
        plan["readiness"]["state"] != "PROPOSED"
        or plan["readiness"]["ready_for_training"]
    ):
        raise ValueError("Expected an unapproved PROPOSED protocol")
    return source, plan


def build_review(
    audit: Path,
    proposal: Path,
    acceptance: Path,
    source_root: Path,
    output: Path,
    *,
    expected_cases: int = 8,
) -> dict:
    """Publish a fresh external evidence package. Sources are never written."""
    repo = Path(__file__).resolve().parents[3]
    target = output.resolve()
    roots = (audit, proposal, acceptance, source_root)
    if (
        output.exists()
        or target.is_relative_to(repo)
        or any(
            target.is_relative_to(p.resolve()) or p.resolve().is_relative_to(target)
            for p in roots
        )
    ):
        raise ValueError("Fresh external output disjoint from sources required")
    before = {
        "audit": signatures(audit),
        "proposal": signatures(proposal),
        "acceptance": signatures(acceptance),
    }
    source, plan = _bound_sources(audit, proposal, acceptance)
    labels = {
        r["label_id"]: r for r in read_json(audit / "annotation_audit.json")["records"]
    }
    frames = {
        r["frame_id"]: r
        for r in read_json(audit / "candidate_manifest.json")["records"]
    }
    test = set(pd.read_parquet(proposal / "test_records.parquet").frame_id)
    quarantine = set(pd.read_parquet(proposal / "quarantine_records.parquet").frame_id)
    ids = set(plan["protected_annotation_conflict_relation_ids"])
    edges = [e for e in source.relations if e["relation_id"] in ids]
    contents = sorted({e["left_content_id"] for e in edges})
    if (
        len(contents) != expected_cases
        or len(contents) != plan["protected_annotation_conflict_groups"]
    ):
        raise ValueError("Unexpected protected conflict count")
    with (acceptance / "protected_annotation_conflicts.csv").open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        accepted = list(csv.DictReader(stream))
    if len(accepted) != len(contents) or {r["content_id"] for r in accepted} != set(
        contents
    ):
        raise ValueError("Acceptance conflict inventory differs")
    archive_checks = {}
    cases, payloads = [], {}
    for content in contents:
        members = [r for r in frames.values() if r["content_id"] == content]
        a = [r for r in members if r["frame_id"] in test]
        b = [r for r in members if r["frame_id"] in quarantine]
        if len(members) != 2 or len(a) != 1 or len(b) != 1:
            raise ValueError(
                "Review v1 requires exactly one test and one quarantine occurrence"
            )
        conflict_id = "conflict-" + content
        case = {
            "conflict_id": conflict_id,
            "content_id": content,
            "relation_ids": sorted(
                e["relation_id"] for e in edges if e["left_content_id"] == content
            ),
        }
        raws = []
        for key, occurrence in (("A", a[0]), ("B", b[0])):
            if len(occurrence["label_ids"]) != 1:
                raise ValueError("Ambiguous annotation association")
            label = labels[occurrence["label_ids"][0]]
            for locator in (occurrence, label):
                name = locator["source_archive"]
                if name not in archive_checks:
                    expected = [
                        s["sha256"]
                        for s in source.summary["configuration"]["sources"].values()
                        if s["name"] == name
                    ]
                    if (
                        len(expected) != 1
                        or file_sha256(source_root / name) != expected[0]
                    ):
                        raise ValueError("Original archive checksum mismatch")
                    archive_checks[name] = expected[0]
            raw = source_bytes(source_root, occurrence, content)
            label_raw = source_bytes(source_root, label, label["label_sha256"])
            if label_raw != base64.b64decode(
                label["original_bytes_base64"], validate=True
            ):
                raise ValueError("Original annotation differs from audit bytes")
            parsed = validate_label_bytes(label_raw)
            if not parsed.label_valid:
                raise ValueError("Invalid source YOLO annotation")
            case[key] = {
                "role": "proposed_test" if key == "A" else "quarantine",
                "occurrence": occurrence,
                "annotation": label,
                "boxes": list(map(list, parsed.boxes)),
            }
            payloads[f"cases/{conflict_id}/{key}.txt"] = label_raw
            raws.append(raw)
        if raws[0] != raws[1]:
            raise ValueError("Exact-content pair has unequal original bytes")
        case["differences"] = differences(case["A"]["boxes"], case["B"]["boxes"])
        accepted_case = next(r for r in accepted if r["content_id"] == content)
        for prefix, key in (("test", "A"), ("quarantine", "B")):
            if (
                accepted_case[f"{prefix}_frame_id"]
                != case[key]["occurrence"]["frame_id"]
                or accepted_case[f"{prefix}_label_sha256"]
                != case[key]["annotation"]["label_sha256"]
            ):
                raise ValueError("Acceptance annotation binding differs")
        with Image.open(io.BytesIO(raws[0])) as image:
            case["image_size"] = list(image.size)
            image_format = image.format
        suffix = {"JPEG": ".jpg", "PNG": ".png"}.get(image_format, ".image")
        case["original_image"] = f"cases/{conflict_id}/original{suffix}"
        case["comparison"] = f"cases/{conflict_id}/comparison.png"
        payloads[case["original_image"]] = raws[0]
        payloads[case["comparison"]] = render_comparison(raws[0], case)
        cases.append(case)
    identity = {
        "schema_version": VERSION,
        "plan_id": plan["plan_id"],
        "test_id": plan["test_id"],
        "sources": before,
        "archives": archive_checks,
        "expected_cases": expected_cases,
        "implementation_sha256": file_sha256(Path(__file__)),
        "pillow_version": PIL.__version__,
    }
    manifest = {
        "schema_version": VERSION,
        "review_id": stable_id(identity),
        "identity": identity,
        "state": "PENDING_HUMAN_REVIEW",
        "protocol_state": "PROPOSED",
        "ready_for_training": False,
        "adjudicated_labels_applied": False,
        "cases": cases,
        "caution": CAUTION,
    }
    decisions = {
        "review_id": manifest["review_id"],
        "cases": [
            {
                "conflict_id": c["conflict_id"],
                "reviewer": "",
                "review_date": "",
                "decision": "pendiente",
                "justification": "",
                "additional_evidence": "",
            }
            for c in cases
        ],
    }
    validate_decisions(decisions, manifest)
    output.mkdir(parents=True)
    for name, raw in payloads.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    write_json(output / "manifest.json", manifest)
    write_json(output / "decisions.template.json", decisions)
    with (output / "differences.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = (
            [
                "conflict_id",
                "test_source",
                "quarantine_source",
                *cases[0]["differences"],
            ]
            if cases
            else []
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for c in cases:
            writer.writerow(
                {
                    "conflict_id": c["conflict_id"],
                    "test_source": c["A"]["occurrence"]["zip_member_name"],
                    "quarantine_source": c["B"]["occurrence"]["zip_member_name"],
                    **{
                        k: json.dumps(v, ensure_ascii=False)
                        for k, v in c["differences"].items()
                    },
                }
            )
    sections = []
    for c in cases:
        sections.append(
            f"<section><h2>{html.escape(c['A']['occurrence']['basename'])}</h2>"
            f'<p>{html.escape(c["conflict_id"])}</p><a href="{c["comparison"]}">'
            f'<img src="{c["comparison"]}" alt="Comparación A, B y diferencias"></a>'
            f'<p><a href="{c["original_image"]}">Imagen original sin cajas</a> | '
            f'<a href="cases/{c["conflict_id"]}/A.txt">YOLO A</a> | '
            f'<a href="cases/{c["conflict_id"]}/B.txt">YOLO B</a></p>'
            f"<pre>{html.escape(json.dumps(c['differences'], indent=2))}</pre></section>"
        )
    (output / "index.html").write_text(
        '<!doctype html><html lang="es"><meta charset="utf-8"><title>Revisión protegida FLIR</title>'
        "<style>body{font:17px sans-serif;margin:30px;background:#f4f5f7}img{width:100%}"
        "section{background:white;padding:20px;margin:25px 0}pre{white-space:pre-wrap}</style>"
        "<h1>Revisión humana de conflictos protegidos</h1><p>"
        + html.escape(CAUTION)
        + "</p>"
        "<p>Abra las imágenes a tamaño completo. Las coordenadas y bytes originales están en manifest.json. "
        "Copie decisions.template.json fuera del paquete antes de editarlo. "
        "Decisiones: pendiente, conservar_anotacion_A, conservar_anotacion_B, "
        "proponer_correccion_manual, excluir_de_evaluacion. Fecha ISO YYYY-MM-DD. "
        "Registre evidencia adicional con localizador y checksum; si no existe, declárelo explícitamente. "
        "Una propuesta de corrección manual debe referenciar un nuevo archivo y su checksum. "
        "Esta herramienta no aplica decisiones. Una futura capa adjudicada debe conservar este review_id, "
        "ambos hashes de origen, decisión atribuida y nuevo hash; requiere nueva identidad del test "
        "y reverificación del protocolo. Suficiencia por clase, dependencias y aprobación siguen pendientes.</p>"
        + "".join(sections)
        + "</html>",
        encoding="utf-8",
    )
    if before != {
        "audit": signatures(audit),
        "proposal": signatures(proposal),
        "acceptance": signatures(acceptance),
    }:
        raise ValueError("Sources changed during review generation")
    for name, digest in archive_checks.items():
        if file_sha256(source_root / name) != digest:
            raise ValueError("Original archive changed during generation")
    write_json(
        output / "receipt.json",
        {"review_id": manifest["review_id"], "artifacts": signatures(output)},
    )
    return manifest


def verify_review(
    directory: Path, audit: Path, proposal: Path, acceptance: Path, source_root: Path
) -> dict:
    """Replay from verified original sources, detecting repaired receipts or altered states."""
    receipt = read_json(directory / "receipt.json")
    actual = signatures(directory)
    del actual["receipt.json"]
    if actual != receipt["artifacts"]:
        raise ValueError("Review package checksum or file inventory mismatch")
    manifest = read_json(directory / "manifest.json")
    with tempfile.TemporaryDirectory(prefix="flir-review-replay-") as temporary:
        replay = Path(temporary) / "review"
        rebuilt = build_review(
            audit,
            proposal,
            acceptance,
            source_root,
            replay,
            expected_cases=manifest["identity"]["expected_cases"],
        )
        if (
            signatures(replay) != signatures(directory)
            or rebuilt["review_id"] != receipt["review_id"]
        ):
            raise ValueError("Review reproduction mismatch")
    return {
        "review_id": receipt["review_id"],
        "reproduced": True,
        "cases": len(manifest["cases"]),
        **validate_decisions(
            read_json(directory / "decisions.template.json"), manifest
        ),
    }
