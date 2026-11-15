"""Read-only no-HUD correspondence research. Outputs never authorize label transfer."""

from __future__ import annotations

import argparse
import hashlib
import html
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from flir_pipeline.data.no_hud import fingerprint
from flir_pipeline.data.no_hud_alignment import (
    gray_and_support,
    register_pair,
    temporal_fit,
)
from flir_pipeline.detection.annotation_review import source_bytes
from flir_pipeline.detection.common_test_storage import signatures
from flir_pipeline.similarity.storage import execution_provenance, read_json, write_json
from flir_pipeline.utils.hashing import sha256_file


def safe_output(output, *inputs):
    destination = output.resolve()
    repo = Path(__file__).resolve().parents[1]
    if destination.exists() or destination.is_relative_to(repo):
        raise ValueError("Use a fresh external output")
    for source in inputs:
        source = source.resolve()
        if destination.is_relative_to(source) or source.is_relative_to(destination):
            raise ValueError("Output overlaps read-only inputs")


def index_video(video, output):
    import av

    safe_output(output, video.parent)
    output.mkdir(parents=True, exist_ok=False)
    digest = sha256_file(video)
    rows, vectors = [], []
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        meta = dict(
            video_sha256=digest,
            video=str(video.resolve()),
            av_version=av.__version__,
            libraries=av.library_versions,
            stream_index=stream.index,
            codec=stream.codec_context.name,
            width=stream.width,
            height=stream.height,
            average_rate=str(stream.average_rate),
            base_rate=str(stream.base_rate),
            time_base=str(stream.time_base),
            declared_frames=stream.frames,
            duration=stream.duration,
            container_metadata=dict(container.metadata),
            stream_metadata=dict(stream.metadata),
        )
        for index, frame in enumerate(container.decode(stream)):
            # A bounded descriptor raster; conversion recipe and decoder are recorded.
            small = Image.fromarray(
                frame.reformat(width=160, height=90, format="rgb24").to_ndarray()
            )
            vectors.append(fingerprint(small).astype(np.float32))
            rows.append(
                dict(
                    native_index=index,
                    pts=frame.pts,
                    time_base=str(frame.time_base),
                    pts_seconds=float(frame.pts * frame.time_base)
                    if frame.pts is not None
                    else None,
                    key_frame=frame.key_frame,
                    width=frame.width,
                    height=frame.height,
                )
            )
            if index % 2000 == 0:
                print(f"Decoded {index} native frames", flush=True)
    if sha256_file(video) != digest:
        raise ValueError("Video changed")
    pd.DataFrame(rows).to_parquet(output / "frames.parquet", index=False)
    np.save(output / "fingerprints.npy", np.asarray(vectors))
    deltas = np.diff([r["pts_seconds"] for r in rows if r["pts_seconds"] is not None])
    meta.update(
        decoded_frames=len(rows),
        pts_delta_min=float(deltas.min()),
        pts_delta_max=float(deltas.max()),
        pts_delta_median=float(np.median(deltas)),
        nonpositive_deltas=int((deltas <= 0).sum()),
        descriptor_recipe="PyAV swscale RGB160x90 -> PIL central .15..85 gray32x24 -> centered L2",
        ready_for_training=False,
        extraction_provenance_reconstructed=False,
    )
    write_json(output / "metadata.json", meta)
    write_json(
        output / "receipt.json",
        dict(outputs=signatures(output), execution=execution_provenance()),
    )
    print(json.dumps(meta, indent=2), flush=True)


def image_at(root, row, digest):
    with Image.open(io.BytesIO(source_bytes(root, row, digest))) as im:
        im.load()
        return im.copy()


def check_package(path):
    receipt = read_json(path / "receipt.json")
    actual = signatures(path)
    actual.pop("receipt.json")
    if actual != receipt["outputs"]:
        raise ValueError("Input receipt mismatch")


def run_pilot(publication, video_index, output):
    check_package(publication)
    check_package(video_index)
    inputs = read_json(publication / "inputs.json")
    root = Path(inputs["root"])
    safe_output(
        output,
        publication,
        video_index,
        root,
        Path(inputs["audit"]),
        Path(inputs["proposal"]),
    )
    output.mkdir(parents=True, exist_ok=False)
    protection_path = Path(inputs["proposal"]) / "protected_records.parquet"
    prior_bindings = read_json(publication / "source_bindings.json")
    if (
        sha256_file(protection_path)
        != prior_bindings["proposal"]["protected_records.parquet"]
    ):
        raise ValueError("Protected population changed")
    protected_contents = set(pd.read_parquet(protection_path).content_id)
    inventory = read_json(publication / "inventory.json")
    targets = {r["target_id"]: r for r in inventory["no_hud"]}
    sources = {r["frame_id"]: r for r in inventory["hud"]}
    representatives = {r["content_id"]: r for r in inventory["hud"]}
    all_pairs = read_json(publication / "correspondences.json")
    video_meta = read_json(video_index / "metadata.json")
    video = Path(video_meta["video"])
    if sha256_file(video) != video_meta["video_sha256"]:
        raise ValueError("Original video checksum mismatch")
    frames = pd.read_parquet(video_index / "frames.parquet")
    vf = np.load(video_index / "fingerprints.npy")
    if len(vf) != len(frames):
        raise ValueError("Video descriptor/ledger mismatch")
    cache = {}

    def src_image(row):
        key = row["content_id"]
        if key not in cache:
            cache[key] = image_at(root, row, key)
        return cache[key]

    texture = {}
    # Selection is frozen before registration: 20 equal nominal-index strata,
    # strongest prior candidate with texture >=5 and distinct source content.
    selected = []
    used = set()
    indexes = [r["observed_index"] for r in targets.values()]
    edges = np.linspace(min(indexes), max(indexes) + 1, 21)
    for bin_id in range(20):
        candidates = []
        for row in all_pairs:
            ti = targets[row["target_id"]]["observed_index"]
            if not edges[bin_id] <= ti < edges[bin_id + 1]:
                continue
            for choice in row["top_candidates"]:
                if choice["content_id"] not in used:
                    candidates.append((choice["score"], row, choice["content_id"]))
        candidates.sort(key=lambda t: (-t[0], t[1]["target_id"], t[2]))
        for score, row, content in candidates:
            if content not in texture:
                a, m = gray_and_support(src_image(representatives[content]))
                texture[content] = float(a[m].std() * 255) if m.any() else 0.0
            if texture[content] < 5:
                continue
            selected.append(
                dict(
                    bin=bin_id,
                    target_id=row["target_id"],
                    source=representatives[content],
                    candidate_score=score,
                    source_content_id=content,
                    protected=content in protected_contents,
                )
            )
            used.add(content)
            break
    prior_gallery = [r for r in all_pairs if "diagnostic_full_frame_geometry" in r]
    gallery_rows = []
    for row in prior_gallery:
        src = sources[row["source_frame_id"]]
        target = targets[row["target_id"]]
        im = src_image(src)
        ti = image_at(root, target, target["sha256"])
        diag = register_pair(im, ti)
        gallery_rows.append(
            dict(
                target_id=target["target_id"],
                target_index=target["observed_index"],
                source_member=src["zip_member_name"],
                source_content_id=src["content_id"],
                prior_score=row["retrieval_score"],
                registration=diag,
                human_validated=False,
                frame_identity_verified=False,
            )
        )
    write_json(output / "previous_gallery_review.json", gallery_rows)
    pair_rows = []
    wanted = set()
    for item in selected:
        target = targets[item["target_id"]]
        src = item["source"]
        a = src_image(src)
        b = image_at(root, target, target["sha256"])
        retrieval = {}
        for role, im in (("source", a), ("target", b)):
            v = fingerprint(im.resize((160, 90), Image.Resampling.BILINEAR)).astype(
                np.float32
            )
            scores = vf @ v
            order = np.argsort(-scores, kind="stable")[:5]
            retrieval[role] = [
                dict(
                    native_index=int(j),
                    pts_seconds=float(frames.iloc[j].pts_seconds),
                    score=float(scores[j]),
                )
                for j in order
            ]
            center = int(order[0])
            wanted.update(range(max(0, center - 2), min(len(frames), center + 3)))
        target_order = sorted(targets.values(), key=lambda r: r["observed_index"])
        neg = target_order[
            (target["observed_index"] - 1 + len(target_order) // 2) % len(target_order)
        ]
        negative = image_at(root, neg, neg["sha256"])
        reg = register_pair(a, b)
        neg_reg = register_pair(a, negative)
        row = dict(
            pair_id=f"pilot-{item['bin']:02d}",
            bin=item["bin"],
            target_id=item["target_id"],
            target_member=target["zip_member_name"],
            target_sha256=target["sha256"],
            target_index=target["observed_index"],
            source_frame_id=src["frame_id"],
            source_content_id=src["content_id"],
            source_member=src["zip_member_name"],
            source_possible_sequence=src["possible_sequence"],
            source_nominal_index=src["possible_frame_index"],
            protected=item["protected"],
            visual_candidate=True,
            registration=reg,
            video_candidates=retrieval,
            native_rank1_delta=retrieval["source"][0]["native_index"]
            - retrieval["target"][0]["native_index"],
            native_top5_intersection=sorted(
                {r["native_index"] for r in retrieval["source"]}
                & {r["native_index"] for r in retrieval["target"]}
            ),
            descriptor_margins={
                role: rows[0]["score"] - rows[1]["score"]
                for role, rows in retrieval.items()
            },
            negative_control=dict(
                target_id=neg["target_id"],
                target_index=neg["observed_index"],
                registration=neg_reg,
            ),
            frame_identity_verified=False,
            human_validated=False,
            human_reviewer=None,
            status="pending_review"
            if reg["geometry_screen_pass"]
            else "rejected_pilot_geometry",
            rejection_scope="selected_pair_under_this_protocol_not_absence_of_a_match",
            ready_for_training=False,
            development_blocked=True,
        )
        pair_rows.append(row)
        print(f"Registered {row['pair_id']}: {row['status']}", flush=True)
    # Decode only the selected candidate windows into external review rasters.
    import av

    if av.__version__ != video_meta["av_version"]:
        raise ValueError("Use the decoder version recorded by index-video")
    native_dir = output / "native_windows"
    native_dir.mkdir()
    native_rows = []
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for index, frame in enumerate(container.decode(stream)):
            if index in wanted:
                rgb = frame.to_ndarray(format="rgb24")
                raw_sha = hashlib.sha256(rgb.tobytes()).hexdigest()
                im = Image.fromarray(rgb)
                im.resize((960, 540), Image.Resampling.BILINEAR).save(
                    native_dir / f"{index:06d}.png"
                )
                native_rows.append(
                    dict(
                        native_index=index,
                        pts=frame.pts,
                        pts_seconds=float(frame.pts * frame.time_base),
                        decoded_rgb_sha256=raw_sha,
                        native_size=list(im.size),
                        review_size=[960, 540],
                    )
                )
            if index >= max(wanted):
                break
    write_json(output / "native_windows.json", native_rows)
    pages = []
    for row in pair_rows:
        a = src_image(sources[row["source_frame_id"]])
        target = targets[row["target_id"]]
        b = image_at(root, target, target["sha256"])
        for role, im in (("source", a), ("target", b)):
            center = row["video_candidates"][role][0]["native_index"]
            local = []
            for j in range(max(0, center - 2), min(len(frames), center + 3)):
                with Image.open(native_dir / f"{j:06d}.png") as native:
                    g = register_pair(native, im)
                local.append(
                    dict(
                        native_index=j,
                        pts_seconds=float(frames.iloc[j].pts_seconds),
                        registration=g,
                    )
                )
            row[f"{role}_native_window"] = local
            candidates = []
            for candidate in local:
                reg = candidate["registration"]
                if reg.get("estimated"):
                    chosen = next(
                        t
                        for t in reg["transforms"]
                        if t["model"] == reg["selected_model"]
                    )
                    candidates.append(
                        (chosen["held_out"]["mae_255"], candidate["native_index"])
                    )
            candidates.sort()
            row[f"{role}_local_refinement"] = dict(
                best_native_index=candidates[0][1] if candidates else None,
                best_mae_255=candidates[0][0] if candidates else None,
                within_half_gray_level=[
                    j for error, j in candidates if error <= candidates[0][0] + 0.5
                ],
                uniqueness_scope="only_five_local_candidates_not_global_proof",
            )
        canvas = Image.new("RGB", (1440, 910), "white")
        draw = ImageDraw.Draw(canvas)
        for x, im, title in ((0, a, "HUD source"), (720, b, "No-HUD candidate")):
            canvas.paste(im.convert("RGB").resize((720, 405)), (x, 30))
            draw.text((x + 5, 5), title, fill="black")
        # Signed residual is diagnostic after only full-frame resize, not a HUD mask.
        ar = np.asarray(
            a.convert("RGB").resize(b.size, Image.Resampling.BILINEAR), dtype=np.int16
        )
        br = np.asarray(b.convert("RGB"), dtype=np.int16)
        heat = Image.fromarray(
            np.minimum(abs(ar - br) * 4, 255).astype(np.uint8)
        ).resize((720, 405))
        canvas.paste(heat, (0, 475))
        draw.text(
            (5, 450),
            "Baseline absolute RGB error x4 (includes modified HUD)",
            fill="black",
        )
        center = row["video_candidates"]["target"][0]["native_index"]
        for k, j in enumerate(range(max(0, center - 2), min(len(frames), center + 3))):
            with Image.open(native_dir / f"{j:06d}.png") as native:
                thumb = native.resize((230, 129))
            x = 720 + (k % 3) * 240
            y = 480 + (k // 3) * 170
            canvas.paste(thumb, (x, y))
            draw.text((x, y - 18), f"Native candidate {j}", fill="black")
        filename = f"{row['pair_id']}.jpg"
        canvas.save(output / filename, quality=92)
        pages.append(
            f'<article><h2>{row["pair_id"]}</h2><img src="{filename}"><pre>{html.escape(json.dumps(row, indent=2))}</pre></article>'
        )
    models = dict(
        target_nominal_to_native=temporal_fit(
            [r["target_index"] for r in pair_rows],
            [r["video_candidates"]["target"][0]["native_index"] for r in pair_rows],
        ),
        target_nominal_to_pts=temporal_fit(
            [r["target_index"] for r in pair_rows],
            [r["video_candidates"]["target"][0]["pts_seconds"] for r in pair_rows],
        ),
    )
    for family in sorted({r["source_possible_sequence"] for r in pair_rows}):
        subset = [
            r
            for r in pair_rows
            if r["source_possible_sequence"] == family
            and isinstance(r["source_nominal_index"], int)
        ]
        models[f"source_family_{family}"] = temporal_fit(
            [r["source_nominal_index"] for r in subset],
            [r["video_candidates"]["source"][0]["native_index"] for r in subset],
        )
    write_json(output / "temporal_models.json", models)
    write_json(output / "pairs.json", pair_rows)
    summary = dict(
        pair_count=len(pair_rows),
        verified_frame_identity=0,
        human_validated=0,
        geometry_screen_pass=sum(
            r["registration"]["geometry_screen_pass"] for r in pair_rows
        ),
        pending_review=sum(r["status"] == "pending_review" for r in pair_rows),
        rejected_pilot_pairs=sum(
            r["status"] == "rejected_pilot_geometry" for r in pair_rows
        ),
        previous_gallery_pairs=len(gallery_rows),
        previous_gallery_unique_sources=len(
            {r["source_content_id"] for r in gallery_rows}
        ),
        previous_gallery_geometry_pass=sum(
            r["registration"]["geometry_screen_pass"] for r in gallery_rows
        ),
        low_texture_previous_pairs=sum(
            r["registration"].get("texture_std_255", [0])[0] < 5 for r in gallery_rows
        ),
        protected_pairs=sum(r["protected"] for r in pair_rows),
        negative_controls_pass=sum(
            r["negative_control"]["registration"]["geometry_screen_pass"]
            for r in pair_rows
        ),
        verdict="MAPPING AÚN NO VERIFICABLE",
        ready_for_training=False,
        blockers=[
            "No verified extraction ledger for either image family",
            "HUD processing recipe and validated change mask unavailable",
            "Nearest native descriptors and geometric fits do not prove exact native identity",
            "Human review not performed",
        ],
    )
    write_json(output / "summary.json", summary)
    write_json(
        output / "protocol.json",
        dict(
            selection="20 equal nominal-index strata; strongest prior top3 choice; source diagnostic std>=5; distinct source contents",
            geometry="fixed support, disjoint 16px fit/evaluation tiles; resize/translation/similarity; NCC>=.95 MAE<=12 and std>=5 screen only",
            negative_control="index shifted by half target collection, diagnostic not certified unrelated",
            prior_publication=str(publication.resolve()),
            video_index=str(video_index.resolve()),
            ready_for_training=False,
        ),
    )
    (output / "gallery.html").write_text(
        '<!doctype html><meta charset="utf-8"><style>img{max-width:100%}pre{white-space:pre-wrap}</style><h1>Piloto candidato: sin identidad ni revisión humana confirmadas</h1>'
        + "".join(pages),
        encoding="utf8",
    )
    pd.DataFrame(
        [
            dict(
                pair_id=r["pair_id"],
                source=r["source_member"],
                target=r["target_member"],
                target_index=r["target_index"],
                status=r["status"],
                visual_correspondence_candidate=True,
                geometry_estimated=r["registration"]["estimated"],
                geometry_model=r["registration"].get("selected_model"),
                geometry_screen_pass=r["registration"]["geometry_screen_pass"],
                source_native_candidate=r["video_candidates"]["source"][0][
                    "native_index"
                ],
                target_native_candidate=r["video_candidates"]["target"][0][
                    "native_index"
                ],
                frame_identity_verified=False,
                human_validated=False,
                human_review_status="not_performed",
                protected=r["protected"],
                ready_for_training=False,
            )
            for r in pair_rows
        ]
    ).to_csv(output / "evidence.csv", index=False)
    check_package(publication)
    check_package(video_index)
    if sha256_file(video) != video_meta["video_sha256"]:
        raise ValueError("Video changed during pilot")
    write_json(
        output / "receipt.json",
        dict(
            outputs=signatures(output),
            inputs=dict(
                publication_receipt=sha256_file(publication / "receipt.json"),
                video_index_receipt=sha256_file(video_index / "receipt.json"),
            ),
            implementation={
                "script": sha256_file(Path(__file__)),
                "registration": sha256_file(
                    Path(__file__).parents[1]
                    / "src/flir_pipeline/data/no_hud_alignment.py"
                ),
            },
            execution=execution_provenance(),
            ready_for_training=False,
        ),
    )
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    index = sub.add_parser("index-video")
    index.add_argument("--video", type=Path, required=True)
    index.add_argument("--output", type=Path, required=True)
    pilot = sub.add_parser("pilot")
    pilot.add_argument("--publication", type=Path, required=True)
    pilot.add_argument("--video-index", type=Path, required=True)
    pilot.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    if args.output.resolve().is_relative_to(repo):
        raise ValueError("Scientific outputs must be external")
    if args.command == "index-video":
        index_video(args.video, args.output)
    else:
        run_pilot(args.publication, args.video_index, args.output)


if __name__ == "__main__":
    main()
