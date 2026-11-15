"""Conservative no-HUD preparation: retrieval is never temporal identity.

Only explicit, checksum-bound native-frame provenance plus exact unmasked raster
agreement can enable candidate labels. Lossy/unknown processing stays pending.
Original annotations and protected memberships are never written by this module.
"""

from __future__ import annotations

import hashlib
import html
import io
import json
import re
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from flir_pipeline.data.classes import DETECTION_CLASSES
from flir_pipeline.data.yolo_labels import validate_label_bytes
from flir_pipeline.detection.annotation_review import project_box, source_bytes
from flir_pipeline.detection.common_test_storage import signatures, verify_common_test
from flir_pipeline.similarity.storage import execution_provenance, read_json, write_json
from flir_pipeline.splitting.labeled import load_labeled_audit, verify_original_sources
from flir_pipeline.utils.hashing import sha256_file

VERSION = "no_hud_label_transfer_v1"


def fingerprint(image: Image.Image) -> np.ndarray:
    """Central grayscale thumbnail NCC, solely for candidate retrieval.

    The crop reduces border HUD influence; it is not a verified HUD mask. A
    threshold or a nearest neighbor cannot establish native-frame identity.
    """
    w, h = image.size
    a = np.asarray(
        image.convert("L")
        .crop((w * 0.15, h * 0.15, w * 0.85, h * 0.85))
        .resize((32, 24), Image.Resampling.BILINEAR),
        dtype=np.float64,
    ).ravel()
    a -= a.mean()
    return a / max(float(np.linalg.norm(a)), 1e-12)


def inventory(archives: list[Path]) -> tuple[list[dict], np.ndarray, list[dict]]:
    """Decode every physical entry read-only, retaining duplicates and errors."""
    rows, vectors, labels = [], [], []
    for path in archives:
        with zipfile.ZipFile(path) as z:
            infos = z.infolist()
            names = Counter(i.filename for i in infos)
            for ordinal, info in enumerate(infos):
                name = info.filename.replace("\\", "/")
                if (
                    name.startswith("/")
                    or PureWindowsPath(name).drive
                    or ".." in name.split("/")
                    or "\x00" in name
                    or (info.external_attr >> 16) & 0o170000 == 0o120000
                ):
                    raise ValueError("Unsafe ZIP entry")
                suffix = Path(name).suffix.lower()
                if info.is_dir() or suffix not in {
                    ".png",
                    ".jpg",
                    ".jpeg",
                    ".bmp",
                    ".tif",
                    ".tiff",
                    ".txt",
                }:
                    continue
                if (
                    info.file_size > 64 * 1024**2
                    or info.file_size / max(info.compress_size, 1) > 1000
                ):
                    raise ValueError("ZIP entry exceeds read budget")
                raw = z.read(info)
                digest = hashlib.sha256(raw).hexdigest()
                row = dict(
                    source_archive=path.name,
                    zip_member_name=info.filename,
                    zip_header_offset=info.header_offset,
                    member_ordinal=ordinal,
                    sha256=digest,
                    ambiguous_locator=names[info.filename] != 1,
                )
                if suffix == ".txt":
                    labels.append(row)
                    continue
                row["target_id"] = hashlib.sha256(
                    json.dumps([path.name, ordinal, info.filename, digest]).encode()
                ).hexdigest()
                match = re.fullmatch(r"frame_(\d+)", Path(name).stem)
                row["observed_index"] = int(match[1]) if match else None
                row["index_provenance"] = "filename_heuristic_not_native_frame_identity"
                vector = np.zeros(768, dtype=np.float64)
                try:
                    with Image.open(io.BytesIO(raw)) as im:
                        if im.width * im.height > 40_000_000:
                            raise ValueError("Image exceeds pixel budget")
                        im.load()
                        row.update(
                            width=im.width,
                            height=im.height,
                            format=im.format,
                            orientation=im.getexif().get(274, 1),
                            decode_error=None,
                        )
                        vector = fingerprint(im)
                except (OSError, ValueError) as error:
                    row.update(
                        width=None,
                        height=None,
                        format=None,
                        orientation=None,
                        decode_error=str(error),
                    )
                rows.append(row)
                vectors.append(vector)
        print(f"Inventoried {path.name}: {len(rows)} images", flush=True)
    label_keys = Counter(
        (r["source_archive"], str(Path(r["zip_member_name"]).with_suffix("")))
        for r in labels
    )
    for row in rows:
        row["existing_label_candidates"] = label_keys[
            (row["source_archive"], str(Path(row["zip_member_name"]).with_suffix("")))
        ]
    return rows, np.asarray(vectors), labels


def geometry(source: Image.Image, target: Image.Image, masks: list) -> dict:
    """Exact unmasked evidence under identity or full-frame uniform bilinear scale.

    No registration fit, cropping, clipping or orientation correction is inferred.
    Masks must be independently supplied normalized modified-region rectangles;
    excluding over 25% is unsupported, not an invitation to fit a larger mask.
    """
    if source.getexif().get(274, 1) != 1 or target.getexif().get(274, 1) != 1:
        return {"valid": False, "reason": "unsupported_orientation"}
    w, h = source.size
    tw, th = target.size
    if w * th != h * tw:
        return {"valid": False, "reason": "unsupported_crop_or_aspect_ratio"}
    keep = np.ones((th, tw), dtype=bool)
    for rect in masks:
        if len(rect) != 4 or not all(np.isfinite(rect)):
            raise ValueError("Invalid modified-region rectangle")
        x0, y0, x1, y1 = rect
        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
            raise ValueError("Invalid normalized mask")
        keep[
            int(np.floor(y0 * th)) : int(np.ceil(y1 * th)),
            int(np.floor(x0 * tw)) : int(np.ceil(x1 * tw)),
        ] = False
    if keep.mean() < 0.75:
        return {"valid": False, "reason": "insufficient_unmasked_support"}
    a = np.asarray(source.convert("RGB").resize(target.size, Image.Resampling.BILINEAR))
    b = np.asarray(target.convert("RGB"))
    residual = np.abs(a.astype(np.int16) - b.astype(np.int16))
    maximum = int(residual[keep].max())
    return dict(
        valid=maximum == 0,
        reason="exact_unmasked_raster" if maximum == 0 else "unexplained_pixel_changes",
        transform="identity"
        if source.size == target.size
        else "uniform_full_frame_bilinear",
        scale_x=tw / w,
        scale_y=th / h,
        translation=[0, 0],
        crop=None,
        unmasked_fraction=float(keep.mean()),
        unmasked_max_error=maximum,
        unmasked_mae=float(residual[keep].mean()),
        masks=masks,
    )


def transfer(raw: bytes, expected_sha: str, geom: dict) -> tuple[bytes, list[int]]:
    """Preserve YOLO bytes for supported full-frame scale; never invent boxes."""
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError("Original annotation checksum mismatch")
    if not geom.get("valid") or geom.get("transform") not in {
        "identity",
        "uniform_full_frame_bilinear",
    }:
        raise ValueError("Unverified geometry")
    parsed = validate_label_bytes(raw)
    if not parsed.label_valid or set(parsed.classes_present) - DETECTION_CLASSES.keys():
        raise ValueError("Invalid class or YOLO geometry")
    warnings = []
    for i, box in enumerate(parsed.boxes):
        x0, y0, x1, y1 = project_box(box, 1, 1)
        if any(
            x0 < mx1 and x1 > mx0 and y0 < my1 and y1 > my0
            for mx0, my0, mx1, my1 in geom["masks"]
        ):
            warnings.append(i)
    return raw, warnings


def provenance_valid(evidence: dict, source: dict, target: dict) -> bool:
    """Require attributed per-image native provenance, not a filename or score.

    External provenance remains attributed evidence, not independently decoded
    video proof. Its bytes are bound in the publication and shown in the gallery.
    """
    a, b = evidence.get("source", {}), evidence.get("target", {})
    return bool(
        evidence.get("reviewer")
        and evidence.get("evidence_reference")
        and a.get("image_sha256") == source["content_id"]
        and b.get("image_sha256") == target["sha256"]
        and re.fullmatch(r"[0-9a-f]{64}", str(a.get("video_sha256", "")))
        and a.get("video_sha256") == b.get("video_sha256")
        and type(a.get("native_frame_index")) is int
        and a["native_frame_index"] >= 0
        and a.get("native_frame_index") == b.get("native_frame_index")
        and evidence.get("modified_regions_complete") is True
    )


def enforce_identity_splits(relations: list[dict], assignments: dict[str, str]) -> None:
    """Mandatory gate for future consumers, including transitive identity chains.

    No current splitter is modified. Consumers must supply every endpoint and may
    not admit protected versions to development. This does not approve training.
    """
    for edge in relations:
        a, b = edge["source_frame_id"], edge["target_id"]
        if a not in assignments or b not in assignments:
            raise ValueError("Missing identity endpoint")
        if assignments[a] != assignments[b]:
            raise ValueError("Same native frame crosses partitions")
        if edge["protected"] and assignments[b] in {"train", "val", "validation"}:
            raise ValueError("Protected frame admitted to development")


def _rgb(raw: bytes) -> Image.Image:
    with Image.open(io.BytesIO(raw)) as im:
        im.load()
        copy = im.copy()
        return copy


def _panel(im: Image.Image, boxes=()) -> Image.Image:
    im = im.convert("RGB")
    draw = ImageDraw.Draw(im)
    for box in boxes:
        draw.rectangle(project_box(box, *im.size), outline="red", width=2)
        xy = project_box(box, *im.size)
        draw.text(xy[:2], str(box[0]), fill="yellow")
    im.thumbnail((720, 480))
    return im


def prepare(
    root: Path,
    audit: Path,
    proposal: Path,
    archives: list[str],
    output: Path,
    *,
    evidence: Path | None = None,
    audit_evidence: Path | None = None,
    gallery_limit: int = 60,
) -> dict:
    """Publish external evidence; lack of per-frame provenance blocks all transfer."""
    root, output = root.resolve(), output.resolve()
    repo = Path(__file__).resolve().parents[3]
    if (
        output.exists()
        or output.is_relative_to(root)
        or root.is_relative_to(output)
        or output.is_relative_to(repo)
        or any(
            output.is_relative_to(p.resolve()) or p.resolve().is_relative_to(output)
            for p in (audit, proposal)
        )
    ):
        raise ValueError(
            "Use a fresh external publication, disjoint from original data"
        )
    if gallery_limit < 0 or len(set(archives)) != len(archives) or not archives:
        raise ValueError("Invalid archive selection or gallery limit")
    paths = []
    for name in sorted(archives):
        if Path(name).name != name or any(c in name for c in "/\\:"):
            raise ValueError("Archives must be direct source filenames")
        paths.append(root / name)
    source = load_labeled_audit(audit)
    verify_original_sources(source, root, audit_evidence)
    verify_common_test(proposal)
    if signatures(audit) != signatures(proposal / "sources" / "audit"):
        raise ValueError("Audit differs from protected proposal source")
    protected = pd.read_parquet(proposal / "protected_records.parquet")
    if not set(protected.frame_id) <= set(source.manifest.frame_id):
        raise ValueError("Protection population not covered by source audit")
    protected_contents = set(protected.content_id)
    inputs = dict(
        root=str(root),
        audit=str(audit.resolve()),
        proposal=str(proposal.resolve()),
        archives=sorted(archives),
        evidence=str(evidence.resolve()) if evidence else None,
        gallery_limit=gallery_limit,
        audit_evidence=str(audit_evidence.resolve()) if audit_evidence else None,
    )
    bindings = dict(
        archives={p.name: sha256_file(p) for p in paths},
        audit=signatures(audit),
        proposal=signatures(proposal),
        evidence=sha256_file(evidence) if evidence else None,
    )
    supplied = read_json(evidence) if evidence else {"pairs": []}
    pairs = supplied["pairs"]
    pair_by_target = {}
    for pair in pairs:
        if pair["target_id"] in pair_by_target:
            raise ValueError("Ambiguous provenance target")
        pair_by_target[pair["target_id"]] = pair
    targets, vectors, existing_labels = inventory(paths)
    if not targets:
        raise ValueError("No no-HUD images: transfer stopped")
    source_rows = read_json(audit / "candidate_manifest.json")["records"]
    label_rows = {
        r["label_id"]: r for r in read_json(audit / "annotation_audit.json")["records"]
    }
    by_content = defaultdict(list)
    by_frame = {r["frame_id"]: r for r in source_rows}
    for row in source_rows:
        by_content[row["content_id"]].append(row)
    unique = [by_content[key][0] for key in sorted(by_content)]
    source_vectors = []
    for row in unique:
        im = _rgb(source_bytes(root, row, row["content_id"]))
        source_vectors.append(fingerprint(im))
    if set(pair_by_target) - {r["target_id"] for r in targets}:
        raise ValueError("Unknown provenance target")
    if any(p["source_frame_id"] not in by_frame for p in pairs):
        raise ValueError("Unknown provenance source")
    # Fixed-size blocks bound memory. Stable sorting preserves ties explicitly.
    nearest = []
    matrix = np.asarray(source_vectors)
    for start in range(0, len(vectors), 128):
        scores = vectors[start : start + 128] @ matrix.T
        for score in scores:
            order = np.argsort(-score, kind="stable")[:3]
            nearest.append([(int(i), round(float(score[i]), 12)) for i in order])
    output.mkdir(parents=True)
    (output / "candidate_labels").mkdir()
    (output / "gallery").mkdir()
    correspondence, relations, pages = [], [], []
    stats = Counter()
    # Gallery prioritizes strongest diagnostic candidates, plus all supplied pairs.
    ranked = sorted(
        range(len(targets)), key=lambda i: (-nearest[i][0][1], targets[i]["target_id"])
    )
    gallery_ids = {targets[i]["target_id"] for i in ranked[:gallery_limit]} | set(
        pair_by_target
    )
    for target, matches in zip(targets, nearest, strict=True):
        pair = pair_by_target.get(target["target_id"])
        best, score = matches[0]
        src = by_frame[pair["source_frame_id"]] if pair else unique[best]
        aliases = by_content[src["content_id"]]
        conflict = len({r["label_sha256"] for r in aliases}) > 1
        row = dict(
            target_id=target["target_id"],
            source_frame_id=src["frame_id"],
            source_content_id=src["content_id"],
            source_occurrences=[r["frame_id"] for r in aliases],
            status="pending_review",
            identity_verified=False,
            candidate_label=None,
            protected=src["content_id"] in protected_contents,
            development_blocked=True,
            annotation_conflict=conflict,
            retrieval_score=score,
            retrieval_source_content_id=unique[best]["content_id"],
            top_candidates=[
                dict(content_id=unique[i]["content_id"], score=s) for i, s in matches
            ],
            score_margin=round(matches[0][1] - matches[1][1], 12)
            if len(matches) > 1
            else None,
            reason="native_frame_provenance_missing",
            geometry=None,
            modified_box_indices=[],
            temporal_identity_scope="none",
            provenance=pair,
            source_resolution=[src["width"], src["height"]],
            target_resolution=[target["width"], target["height"]],
            target_orientation=target["orientation"],
            spatial_verification="not_evaluated_without_native_provenance",
        )
        sim = tim = None
        if (
            (pair or target["target_id"] in gallery_ids)
            and not target["decode_error"]
            and not target["ambiguous_locator"]
        ):
            sim = _rgb(source_bytes(root, src, src["content_id"]))
            tim = _rgb(source_bytes(root, target, target["sha256"]))
            row["diagnostic_full_frame_geometry"] = geometry(sim, tim, [])
            row["source_orientation"] = sim.getexif().get(274, 1)
        if target["decode_error"] or target["ambiguous_locator"]:
            row.update(status="rejected", reason="unreadable_or_ambiguous_target")
        elif pair:
            if not provenance_valid(pair, src, target):
                row["reason"] = "incomplete_or_inconsistent_native_provenance"
            else:
                geom = geometry(sim, tim, pair.get("modified_regions", []))
                row["geometry"] = geom
                row["spatial_verification"] = (
                    "evaluated_against_declared_modified_regions"
                )
                if geom["valid"]:
                    row.update(
                        status="verified",
                        identity_verified=True,
                        reason="attributed_native_provenance_and_exact_unmasked_geometry",
                        temporal_identity_scope="external_provenance_plus_raster_verification",
                    )
                    for alias in aliases:
                        relations.append(
                            dict(
                                source_frame_id=alias["frame_id"],
                                target_id=target["target_id"],
                                relation="same_native_frame"
                                if alias["frame_id"] == src["frame_id"]
                                else "exact_content_dependency",
                                must_link=True,
                                protected=row["protected"],
                                provenance=pair,
                                development_blocked=True,
                            )
                        )
                    if conflict or target["existing_label_candidates"]:
                        row["reason"] = "annotation_conflict_or_existing_target_label"
                    else:
                        label = label_rows[src["label_ids"][0]]
                        raw = source_bytes(root, label, label["label_sha256"])
                        try:
                            candidate, warnings = transfer(
                                raw, label["label_sha256"], geom
                            )
                        except ValueError as error:
                            row["reason"] = str(error)
                        else:
                            name = f"candidate_labels/{target['target_id']}.txt"
                            (output / name).write_bytes(candidate)
                            row.update(
                                candidate_label=name,
                                label_sha256=label["label_sha256"],
                                modified_box_indices=warnings,
                            )
                            stats.update(
                                box[0] for box in validate_label_bytes(candidate).boxes
                            )
                else:
                    row["reason"] = geom["reason"]
        correspondence.append(row)
        if (
            target["target_id"] in gallery_ids
            and not target["decode_error"]
            and not target["ambiguous_locator"]
        ):
            label = label_rows[src["label_ids"][0]]
            parsed = validate_label_bytes(
                source_bytes(root, label, label["label_sha256"])
            )
            left = _panel(sim, parsed.boxes)
            right = _panel(tim, parsed.boxes if row["candidate_label"] else ())
            canvas = Image.new(
                "RGB",
                (left.width + right.width, max(left.height, right.height)),
                "white",
            )
            canvas.paste(left, (0, 0))
            canvas.paste(right, (left.width, 0))
            name = target["target_id"] + ".png"
            canvas.save(output / "gallery" / name)
            text = dict(
                **row,
                source_locator=src["zip_member_name"],
                target_locator=target["zip_member_name"],
                warning="CANDIDATE ONLY. No target boxes unless transfer gates pass. Not approved.",
            )
            pages.append(
                f'<article><img loading="lazy" src="gallery/{name}" alt="HUD original / no-HUD candidate"><pre>{html.escape(json.dumps(text, indent=2))}</pre></article>'
            )
    write_json(output / "inputs.json", inputs)
    write_json(output / "source_bindings.json", bindings)
    write_json(
        output / "inventory.json",
        dict(no_hud=targets, hud=source_rows, target_label_entries=existing_labels),
    )
    write_json(output / "correspondences.json", correspondence)
    write_json(
        output / "frame_identity_relations.json",
        dict(
            relations=relations,
            mandatory_for_future_splits=True,
            consumer_gate="flir_pipeline.data.no_hud.enforce_identity_splits",
            ready_for_training=False,
            existing_partitions_modified=False,
            training_admission="BLOCKED_UNTIL_CONSUMER_INTEGRATION_AND_REVIEW",
        ),
    )
    write_json(
        output / "class_statistics.json",
        dict(
            candidate_instances={str(c): stats[c] for c in DETECTION_CLASSES},
            source_instances=source.summary["counts"],
            candidate_annotations_approved=False,
        ),
    )
    (output / "gallery.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>No-HUD transfer review</title><style>img{max-width:100%}pre{white-space:pre-wrap}article{border-bottom:2px solid}</style><h1>Revision pendiente / ready_for_training=false</h1><p>Left: original labels. Right: candidate labels ONLY for verified geometry and provenance. Retrieval is not identity. Gallery is a ranked diagnostic subset, not a representative sample.</p>'
        + "\n".join(pages),
        encoding="utf-8",
    )
    summary = dict(
        schema_version=VERSION,
        ready_for_training=False,
        verdict="REQUIERE CORRECCIONES",
        no_hud_images=len(targets),
        source_hud_images=len(source_rows),
        source_unique_contents=len(unique),
        existing_target_label_candidates=sum(
            bool(t["existing_label_candidates"]) for t in targets
        ),
        verified_images=sum(r["identity_verified"] for r in correspondence),
        candidate_labels=sum(r["candidate_label"] is not None for r in correspondence),
        pending_review_images=len(targets),
        states=dict(Counter(r["status"] for r in correspondence)),
        protected_candidate_images=sum(r["protected"] for r in correspondence),
        verified_protected_images=sum(
            r["protected"] and r["identity_verified"] for r in correspondence
        ),
        gallery_images=len(pages),
        retrieval_method="central_32x24_grayscale_NCC_top3_unique_content",
        unmatched_verified_images=sum(
            not r["identity_verified"] for r in correspondence
        ),
        limitations=[
            "Nearest neighbors are candidates even at low scores; absence is not proven.",
            "No automatic crop, translation, rotation, lossy-processing tolerance or temporal identity inference.",
            "External provenance is attributed, not independent native-video decoding.",
            "All target records remain blocked from training and development admission.",
        ],
    )
    if (
        summary["candidate_labels"] == len(targets)
        and not summary["verified_protected_images"]
    ):
        summary["verdict"] = "APROBADO PARA REVISIÓN DE TRANSFERENCIA"
    # Recheck original inputs before committing the final receipt.
    verify_original_sources(source, root, audit_evidence)
    if bindings != dict(
        archives={p.name: sha256_file(p) for p in paths},
        audit=signatures(audit),
        proposal=signatures(proposal),
        evidence=sha256_file(evidence) if evidence else None,
    ):
        raise ValueError("Inputs changed during preparation")
    write_json(output / "summary.json", summary)
    write_json(
        output / "receipt.json",
        dict(
            schema_version=VERSION,
            outputs=signatures(output),
            implementation_sha256=sha256_file(Path(__file__)),
            execution=execution_provenance(),
            ready_for_training=False,
        ),
    )
    return summary


def verify(directory: Path, *, replay: bool = False) -> dict:
    """Check all bytes, originals, protection and optionally reproduce all outputs."""
    receipt = read_json(directory / "receipt.json")
    actual = signatures(directory)
    del actual["receipt.json"]
    if (
        receipt["schema_version"] != VERSION
        or receipt["ready_for_training"] is not False
        or actual != receipt["outputs"]
    ):
        raise ValueError("Incomplete or changed transfer publication")
    args = read_json(directory / "inputs.json")
    bindings = read_json(directory / "source_bindings.json")
    root, audit, proposal = (Path(args[k]) for k in ("root", "audit", "proposal"))
    current = dict(
        archives={n: sha256_file(root / n) for n in args["archives"]},
        audit=signatures(audit),
        proposal=signatures(proposal),
        evidence=sha256_file(Path(args["evidence"])) if args["evidence"] else None,
    )
    if current != bindings:
        raise ValueError("Original inputs changed")
    verify_original_sources(
        load_labeled_audit(audit),
        root,
        Path(args["audit_evidence"]) if args["audit_evidence"] else None,
    )
    verify_common_test(proposal)
    summary = read_json(directory / "summary.json")
    if summary["ready_for_training"] is not False:
        raise ValueError("Training cannot be authorized")
    if replay:
        if receipt["implementation_sha256"] != sha256_file(Path(__file__)):
            raise ValueError("Replay requires the recorded implementation")
        with tempfile.TemporaryDirectory(prefix="nh-") as tmp:
            out = Path(tmp) / "replay"
            prepare(
                root,
                audit,
                proposal,
                args["archives"],
                out,
                evidence=Path(args["evidence"]) if args["evidence"] else None,
                gallery_limit=args["gallery_limit"],
                audit_evidence=Path(args["audit_evidence"])
                if args["audit_evidence"]
                else None,
            )
            rebuilt = signatures(out)
            del rebuilt["receipt.json"]
            if rebuilt != actual:
                raise ValueError("Semantic replay differs")
    return dict(
        valid=True,
        replayed=replay,
        ready_for_training=False,
        verified_images=summary["verified_images"],
        candidate_labels=summary["candidate_labels"],
    )
