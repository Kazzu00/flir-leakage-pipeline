"""Bounded development-only search; geometric proposals never authorize training.

Reuses receipt-bound candidates and pilot registration. Missing reverse-index
sources use native descriptor retrieval and a data-fitted target-index hypothesis.
That hypothesis is retrieval only, never a timestamp or identity certificate.
"""

import argparse
import hashlib
import html
import io
import json
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from build_hud_review import panel
from PIL import Image

from flir_pipeline.data.box_transfer import transform_boxes
from flir_pipeline.data.box_transfer_diagnostics import box_diagnostics
from flir_pipeline.data.no_hud import fingerprint
from flir_pipeline.data.no_hud_alignment import register_pair
from flir_pipeline.data.yolo_labels import validate_label_bytes
from flir_pipeline.detection.common_executor import external_output
from flir_pipeline.detection.preliminary import read_rows, verify
from flir_pipeline.similarity.storage import file_sha256, read_json, write_json


def target_image(archives, target):
    with zipfile.ZipFile(archives / target["source_archive"]) as archive:
        entry = archive.infolist()[target["member_ordinal"]]
        if (
            entry.filename != target["zip_member_name"]
            or entry.header_offset != target["zip_header_offset"]
        ):
            raise ValueError("Target locator changed")
        raw = archive.read(entry)
    if hashlib.sha256(raw).hexdigest() != target["sha256"]:
        raise ValueError("Target checksum changed")
    with Image.open(io.BytesIO(raw)) as im:
        return im.copy()


def evaluate(task):
    package, archives, output, row, target, cached, retrieval = task
    key = row["content_id"] + "-" + target["target_id"]
    src = Image.open(package / row["image_path"])
    dst = target_image(archives, target)
    boxes = validate_label_bytes((package / row["label_path"]).read_bytes()).boxes
    geometry = cached if cached is not None else register_pair(src, dst)
    proposed, diagnostics, reasons = [], [], []
    if geometry.get("geometry_screen_pass"):
        chosen = next(
            t
            for t in geometry["transforms"]
            if t["model"] == geometry["selected_model"]
        )
        try:
            proposed = transform_boxes(boxes, chosen["inverse_sampling_parameters"])
            diagnostics = box_diagnostics(
                src, dst, proposed, chosen["inverse_sampling_parameters"]
            )
        except ValueError as exc:
            reasons.append(str(exc))
    else:
        reasons.append("geometry_screen_failed")
    reasons.extend(
        [
            "native_observation_identity_unverified",
            "processing_recipe_and_modified_regions_unknown",
            "object_annotation_review_pending",
        ]
    )
    result = dict(
        source_content_id=row["content_id"],
        source_frame_id=row["frame_id"],
        source_member=row["source_member_path"],
        source_label_sha256=row["label_sha256"],
        target_id=target["target_id"],
        target_content_id=target["sha256"],
        target_locator=target,
        split=row["split"],
        source_dimensions=list(src.size),
        target_dimensions=list(dst.size),
        retrieval=retrieval,
        geometry=geometry,
        reused_geometry=cached is not None,
        original_boxes=boxes,
        candidate_boxes=proposed,
        box_diagnostics=diagnostics,
        reasons=reasons,
        verified_transfer=False,
        status="candidate" if proposed else "rejected_geometry",
        training_allowed=False,
    )
    write_json(output / "pairs" / (key + ".json"), result)
    return result


def build(package, publication, review, video_index, prior, archives, output, workers):
    external_output(output, package, publication, review, video_index, prior, archives)
    if output.exists():
        raise ValueError("Fresh external output required; preserve previous attempts")
    receipt = file_sha256(package / "receipt.json")
    verify(package, receipt)
    for folder, field in (
        (publication, "outputs"),
        (review, "artifacts"),
        (video_index, "outputs"),
        (prior, "outputs"),
    ):
        binding = read_json(folder / "receipt.json")[field]
        required = {
            publication: ["inventory.json", "correspondences.json"],
            review: ["matching.json", "box_candidates.json"],
            video_index: ["fingerprints.npy", "frames.parquet", "metadata.json"],
            prior: ["pairs.json"],
        }[folder]
        for name in required:
            if file_sha256(folder / name) != binding[name]:
                raise ValueError("Changed evidence: " + name)
    rows = read_rows(package / "manifest.csv")
    targets = {
        r["target_id"]: r for r in read_json(publication / "inventory.json")["no_hud"]
    }
    by_index = {r["observed_index"]: r for r in targets.values()}
    if len(by_index) != len(targets):
        raise ValueError("Multiple target series require separate indexing")
    matching = {
        r["source_content_id"]: r["candidates"]
        for r in read_json(review / "matching.json")
    }
    cache = {
        (r["source_content_id"], r["target_id"]): r["geometry"]
        for r in read_json(review / "box_candidates.json")
    }
    old = [r for r in read_json(prior / "pairs.json") if not r["protected"]]
    anchors = sorted(
        (r["video_candidates"]["target"][0]["native_index"], r["target_index"])
        for r in old
    )
    fit, holdout = np.asarray(anchors[::2]), np.asarray(anchors[1::2])
    slope, intercept = np.polyfit(fit[:, 0], fit[:, 1], 1)
    error = float(np.max(abs(holdout[:, 1] - (slope * holdout[:, 0] + intercept))))
    if error > 0.25 or slope <= 0:
        raise ValueError("Native-to-target retrieval hypothesis unsupported")
    vectors = np.load(video_index / "fingerprints.npy", allow_pickle=False)
    frames = pd.read_parquet(video_index / "frames.parquet")
    output.mkdir(parents=True)
    (output / "pairs").mkdir()
    bindings = {
        str(folder): file_sha256(folder / "receipt.json")
        for folder in (package, publication, review, video_index, prior)
    }
    write_json(
        output / "protocol.json",
        dict(
            input_receipts=bindings,
            native_to_target_retrieval_only=dict(
                slope=float(slope),
                intercept=float(intercept),
                heldout_max_index_error=error,
            ),
            training_allowed=False,
            no_hud_origin="externally_processed_recipe_unknown",
            workers=workers,
        ),
    )
    tasks, native_matches = [], []
    for row in rows:
        with Image.open(package / row["image_path"]) as src:
            scores = vectors @ fingerprint(src).astype(np.float32)
        indices = np.argsort(-scores, kind="stable")[:5]
        near = [
            dict(
                native_index=int(frames.iloc[i].native_index),
                pts_seconds=float(frames.iloc[i].pts_seconds),
                score=float(scores[i]),
            )
            for i in indices
        ]
        native_matches.append(
            dict(
                source_content_id=row["content_id"],
                native_candidates=near,
                descriptor_margin=float(scores[indices[0]] - scores[indices[1]]),
                identity_verified=False,
            )
        )
        choices = matching[row["content_id"]]
        method = "existing_reverse_target_top3"
        if not choices:
            method = "native_descriptor_plus_fitted_index_hypothesis"
            choices = []
            for n in near[:3]:
                target = by_index.get(round(slope * n["native_index"] + intercept))
                if target:
                    choices.append(
                        dict(target_id=target["target_id"], score=n["score"])
                    )
        for choice in choices:
            target = targets[choice["target_id"]]
            tasks.append(
                (
                    package,
                    archives,
                    output,
                    row,
                    target,
                    cache.get((row["content_id"], target["target_id"])),
                    dict(
                        method=method,
                        score=choice["score"],
                        source_native_candidates=near,
                    ),
                )
            )
    write_json(output / "native_retrieval.json", native_matches)
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, result in enumerate(pool.map(evaluate, tasks, chunksize=1)):
            results.append(result)
            if (i + 1) % 50 == 0:
                print(f"Evaluated {i + 1}/{len(tasks)} pairs", flush=True)
    # Rank only for review: alternatives within 0.5 gray level remain ambiguous.
    selected = []
    for row in rows:
        candidates = [
            r
            for r in results
            if r["source_content_id"] == row["content_id"] and r["candidate_boxes"]
        ]

        def loss(r):
            return next(
                t["held_out"]["mae_255"]
                for t in r["geometry"]["transforms"]
                if t["model"] == r["geometry"]["selected_model"]
            )

        candidates.sort(key=loss)
        if candidates:
            best = candidates[0]
            best["near_tied_targets"] = [
                r["target_id"] for r in candidates if loss(r) - loss(best) <= 0.5
            ]
            selected.append(best)
    write_json(output / "selected_candidates.json", selected)
    coverage = [
        dict(
            split=s,
            class_id=c,
            candidate_images=sum(
                r["split"] == s and any(b[0] == c for b in r["candidate_boxes"])
                for r in selected
            ),
            candidate_boxes=sum(
                r["split"] == s and b[0] == c
                for r in selected
                for b in r["candidate_boxes"]
            ),
            verified_images=0,
            verified_boxes=0,
        )
        for s in ("train", "val")
        for c in range(5)
    ]
    write_json(output / "coverage.json", coverage)
    summary = dict(
        sources=len(rows),
        evaluated_pairs=len(results),
        sources_evaluated=len({r["source_content_id"] for r in results}),
        pair_statuses=dict(Counter(r["status"] for r in results)),
        selected_candidate_sources=len(selected),
        selected_candidate_boxes=sum(len(r["candidate_boxes"]) for r in selected),
        candidates_by_split=dict(Counter(r["split"] for r in selected)),
        ambiguous_selected_sources=sum(
            len(r["near_tied_targets"]) > 1 for r in selected
        ),
        reused_geometry=sum(r["reused_geometry"] for r in results),
        verified_transfers=0,
        paired_training_population=0,
        training_allowed=False,
    )
    write_json(output / "summary.json", summary)
    # Include every surviving source plus representative rejected class/split cases.
    gallery = selected.copy()
    for s in ("train", "val"):
        for c in range(5):
            if not any(
                r["split"] == s and any(b[0] == c for b in r["original_boxes"])
                for r in gallery
            ):
                choices = [
                    r
                    for r in results
                    if r["split"] == s and any(b[0] == c for b in r["original_boxes"])
                ]
                if choices:
                    gallery.append(choices[0])
    rowmap = {r["content_id"]: r for r in rows}
    markup = '<meta charset="utf-8"><h1>NO-HUD: propuestas, ninguna transferencia verificada</h1><p>Variante procesada de procedencia incompleta. Naranja: cajas originales/propuestas. No autorizadas para entrenamiento.</p>'
    for i, item in enumerate(gallery):
        row = rowmap[item["source_content_id"]]
        sheet = Image.new("RGB", (1280, 400), "white")
        with Image.open(package / row["image_path"]) as src:
            sheet.paste(
                panel(
                    src, item["original_boxes"], f"{i:03d} ORIGINAL / {row['split']}"
                ),
                (0, 0),
            )
        dst = target_image(archives, targets[item["target_id"]])
        sheet.paste(
            panel(dst, item["candidate_boxes"], item["status"] + " / NOT VERIFIED"),
            (640, 0),
        )
        name = f"qc-{i:03d}.jpg"
        sheet.save(output / name, quality=90)
        markup += f'<h2>{i:03d} {html.escape(item["source_member"])}</h2><img loading="lazy" width="100%" src="{name}">'
    write_json(output / "gallery_manifest.json", gallery)
    (output / "gallery.html").write_text(markup, encoding="utf-8")
    verify(package, receipt)
    write_json(
        output / "receipt.json",
        dict(
            artifacts={
                p.relative_to(output).as_posix(): file_sha256(p)
                for p in output.rglob("*")
                if p.is_file()
            }
        ),
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in (
        "package",
        "publication",
        "review",
        "video_index",
        "prior",
        "archives",
        "output",
    ):
        parser.add_argument(arg, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    build(**vars(parser.parse_args()))
