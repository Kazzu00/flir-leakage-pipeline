"""One static review report: pending empty labels and bounded NO-HUD box pilot.

Reuse existing candidate retrieval and temporal evidence; never certify a match
from scene similarity. All proposed target boxes stay JSON-only and outside train.
"""

import argparse
import hashlib
import html
import io
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

from flir_pipeline.data.box_transfer import transform_boxes
from flir_pipeline.data.no_hud_alignment import register_pair
from flir_pipeline.data.yolo_labels import validate_label_bytes
from flir_pipeline.detection.common_executor import external_output
from flir_pipeline.detection.preliminary import read_rows, write_rows
from flir_pipeline.similarity.storage import file_sha256, read_json, write_json


def panel(image, boxes, title):
    canvas = Image.new("RGB", (640, 400), "white")
    im = image.convert("RGB").resize((640, 360))
    draw = ImageDraw.Draw(im)
    for c, x, y, w, h in boxes:
        draw.rectangle(
            (
                (x - w / 2) * 640,
                (y - h / 2) * 360,
                (x + w / 2) * 640,
                (y + h / 2) * 360,
            ),
            outline="orange",
            width=3,
        )
        draw.text(((x - w / 2) * 640, (y - h / 2) * 360), str(c), fill="orange")
    canvas.paste(im, (0, 40))
    ImageDraw.Draw(canvas).text((5, 5), title, fill="black")
    return canvas


def build(prepared, publication, previous_pilot, archives, output):
    external_output(output, prepared, publication, previous_pilot, archives)
    if output.exists():
        raise ValueError("Fresh external report required")
    rows = read_rows(prepared / "manifest.csv")
    # Input evidence is bound before any raster is read. Only development IDs are used.
    receipt = read_json(prepared / "development_receipt.json")
    for name, sha in receipt["artifacts"].items():
        if file_sha256(prepared / name) != sha:
            raise ValueError("Prepared input changed")
    nh_receipt = read_json(publication / "receipt.json")
    for name in ("inventory.json", "correspondences.json"):
        if file_sha256(publication / name) != nh_receipt["outputs"][name]:
            raise ValueError("Candidate metadata changed")
    output.mkdir(parents=True)
    empty = sorted(
        (r for r in rows if r["label_empty"] == "True"),
        key=lambda r: r["source_member_path"],
    )
    source_by_content = {
        r["content_id"]: r for r in rows if r["label_empty"] == "False"
    }

    def source_image(row):
        path = (
            prepared
            / "canonical/images"
            / (row["frame_id"] + Path(row["image_path"]).suffix)
        )
        return Image.open(path).convert("RGB")

    sheets = []
    for page in range((len(empty) + 11) // 12):
        sheet = Image.new("RGB", (1920, 1600), "white")
        for j, row in enumerate(empty[page * 12 : (page + 1) * 12]):
            name = Path(row["source_member_path"]).name
            p = panel(
                source_image(row),
                [],
                f"E{page * 12 + j + 1:03d} {name} S3={row['S3_split']}",
            )
            sheet.paste(p, ((j % 3) * 640, (j // 3) * 400))
        name = f"empty-{page + 1:02d}.jpg"
        sheet.save(output / name, quality=90)
        sheets.append(name)
    write_rows(
        output / "empty_review.csv",
        [
            dict(
                review_id=f"E{i + 1:03d}",
                **{
                    k: r[k]
                    for k in (
                        "frame_id",
                        "content_id",
                        "source_member_path",
                        "S1_split",
                        "S2_split",
                        "S3_split",
                    )
                },
                status="undetermined_pending_visual_review",
                reviewer="",
                rationale="",
            )
            for i, r in enumerate(empty)
        ],
        [
            "review_id",
            "frame_id",
            "content_id",
            "source_member_path",
            "S1_split",
            "S2_split",
            "S3_split",
            "status",
            "reviewer",
            "rationale",
        ],
    )
    inventory = {
        r["target_id"]: r for r in read_json(publication / "inventory.json")["no_hud"]
    }
    candidates = defaultdict(list)
    for r in read_json(publication / "correspondences.json"):
        for choice in r["top_candidates"]:
            if choice["content_id"] in source_by_content:
                candidates[choice["content_id"]].append(
                    (choice["score"], r["target_id"])
                )
    matching = []
    for content, row in sorted(source_by_content.items()):
        ranked = sorted(candidates[content], reverse=True)[:3]
        matching.append(
            dict(
                source_content_id=content,
                source_frame_id=row["frame_id"],
                candidates=[dict(target_id=t, score=s) for s, t in ranked],
                state="candidate"
                if ranked
                else "no_candidate_in_existing_reverse_index",
                scope="reversed_existing_target_top3_not_exhaustive_source_search",
            )
        )
    write_json(output / "matching.json", matching)
    prior_path = previous_pilot / "results/pairs.json"
    prior_receipt = read_json(previous_pilot / "results/receipt.json")
    if file_sha256(prior_path) != prior_receipt["outputs"]["pairs.json"]:
        raise ValueError("Prior temporal evidence changed")
    prior = read_json(prior_path)
    # Existing chronological pilot first; then cover any absent class/split strata.
    selected = [
        (r["source_content_id"], r["target_id"], r)
        for r in prior
        if r["source_content_id"] in source_by_content and not r["protected"]
    ]
    covered = set()

    def boxes(row):
        return validate_label_bytes(
            (prepared / "canonical/labels" / (row["frame_id"] + ".txt")).read_bytes()
        ).boxes

    for c, _, _ in selected:
        covered.update(
            (source_by_content[c]["S3_split"], int(b[0]))
            for b in boxes(source_by_content[c])
        )
    for split in ("train", "val"):
        for class_id in range(5):
            eligible = [
                c
                for c, r in source_by_content.items()
                if r["S3_split"] == split
                and candidates[c]
                and any(b[0] == class_id for b in boxes(r))
            ]
            if (split, class_id) not in covered and eligible:
                c = max(eligible, key=lambda c: max(candidates[c]))
                selected.append((c, max(candidates[c])[1], None))
                covered.update((split, int(b[0])) for b in boxes(source_by_content[c]))
    results, pair_sheets = [], []
    for i, (content, target_id, old) in enumerate(selected):
        row, target = source_by_content[content], inventory[target_id]
        with zipfile.ZipFile(archives / target["source_archive"]) as z:
            info = z.infolist()[target["member_ordinal"]]
            if (
                info.filename != target["zip_member_name"]
                or info.header_offset != target["zip_header_offset"]
            ):
                raise ValueError("Target ZIP locator changed")
            raw = z.read(info)
        if hashlib.sha256(raw).hexdigest() != target["sha256"]:
            raise ValueError("Target image checksum mismatch")
        dst, src = Image.open(io.BytesIO(raw)).convert("RGB"), source_image(row)
        geometry = register_pair(src, dst)
        original, proposed, reasons = boxes(row), [], []
        if geometry.get("geometry_screen_pass"):
            transform = next(
                t
                for t in geometry["transforms"]
                if t["model"] == geometry["selected_model"]
            )
            try:
                proposed = transform_boxes(
                    original, transform["inverse_sampling_parameters"]
                )
            except ValueError as exc:
                reasons.append(str(exc))
        else:
            reasons.append("geometry_screen_failed")
        reasons.append("native_frame_identity_and_object_level_box_review_missing")
        item = dict(
            source_content_id=content,
            target_content_id=target["sha256"],
            source_frame_id=row["frame_id"],
            target_id=target_id,
            split=row["S3_split"],
            source_member=row["source_member_path"],
            target_archive=target["source_archive"],
            target_member=target["zip_member_name"],
            target_index_heuristic=target["observed_index"],
            source_label_sha256=row["label_sha256"],
            source_dimensions=list(src.size),
            target_dimensions=list(dst.size),
            method="existing_NCC_candidates_then_masked_similarity_registration",
            geometry=geometry,
            original_boxes=original,
            candidate_boxes=proposed,
            temporal_evidence=old,
            confidence="uncalibrated_candidate",
            status="candidate" if proposed else "rejected_geometry",
            reasons=reasons,
            verified_transfer=False,
            training_allowed=False,
        )
        results.append(item)
        sheet = Image.new("RGB", (1280, 400), "white")
        sheet.paste(
            panel(src, original, f"P{i + 1:02d} ORIGINAL / {row['S3_split']}"), (0, 0)
        )
        sheet.paste(
            panel(dst, proposed, "CANDIDATE BOXES - NOT VERIFIED / NO TRAINING"),
            (640, 0),
        )
        name = f"transfer-{i + 1:02d}.jpg"
        sheet.save(output / name, quality=92)
        pair_sheets.append(name)
    write_json(output / "box_candidates.json", results)
    coverage = [
        dict(
            split=s,
            class_id=c,
            pilot_images=sum(
                r["split"] == s and any(b[0] == c for b in r["original_boxes"])
                for r in results
            ),
            geometrically_proposed_boxes=sum(
                r["split"] == s and b[0] == c
                for r in results
                for b in r["candidate_boxes"]
            ),
            verified_transfers=0,
        )
        for s in ("train", "val")
        for c in range(5)
    ]
    write_json(output / "coverage.json", coverage)
    summary = dict(
        empty_labels=len(empty),
        empty_split_distribution={
            p: dict(Counter(r[p + "_split"] for r in empty)) for p in ("S1", "S2", "S3")
        },
        nonempty_sources=len(source_by_content),
        sources_with_cached_candidates=sum(bool(r["candidates"]) for r in matching),
        pilot_pairs=len(results),
        pilot_statuses=dict(Counter(r["status"] for r in results)),
        verified_transfers=0,
        prior_pairs_sha256=file_sha256(prior_path),
        source_receipt_sha256=file_sha256(prepared / "development_receipt.json"),
    )
    write_json(output / "summary.json", summary)
    content = '<meta charset="utf-8"><title>Revisión HUD</title><h1>Revisión de etiquetas vacías y cajas candidatas</h1><p>Vacía no significa errónea. Clasificar: legítima / potencialmente incompleta / indeterminable. No hay aprobaciones automáticas. Ninguna caja NO-HUD está autorizada para entrenamiento.</p>'
    for name in sheets + pair_sheets:
        content += f'<h2>{html.escape(name)}</h2><a href="{name}"><img width="100%" src="{name}"></a>'
    (output / "review.html").write_text(content, encoding="utf-8")
    write_json(
        output / "receipt.json",
        dict(
            artifacts={p.name: file_sha256(p) for p in output.iterdir() if p.is_file()}
        ),
    )
    print(json.dumps(summary))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("prepared", "publication", "previous_pilot", "archives", "output"):
        p.add_argument(arg, type=Path)
    build(**vars(p.parse_args()))
