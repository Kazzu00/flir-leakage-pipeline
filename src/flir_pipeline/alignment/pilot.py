"""Freeze an explicit human-review sample from existing Phase 2A pairs only."""

import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .inventory import encoded, file_sha, read_image, safe_path, sha
from .matching import Decoder, features
from .pipeline import verify

DIAGNOSTIC = (199, 200, 349, 350, 399, 400, 499, 501)


def support_band(case):
    if case["status"] == "not_found":
        return "automatic_abstention"
    best = case["top_k"][0]
    return "inliers_ge40" if best["inliers"] >= 40 else "inliers_lt40"


def select_pilot(evidence, manifest, seed=20261103):
    rows = {r["image_id"]: r for r in manifest["images"]}
    cases = evidence["cases"]
    selected = {}

    def add(case, reason):
        selected[case["image_id"]] = {
            "alignment_id": case["alignment_id"],
            "image_id": case["image_id"],
            "selection_reason": reason,
        }

    for index in DIAGNOSTIC:
        found = [c for c in cases if rows[c["image_id"]]["dataset_index"] == index]
        if len(found) != 1:
            raise ValueError(
                f"Diagnostic index {index} missing or ambiguous occurrence"
            )
        add(found[0], "diagnostic")
    special = [c for c in cases if rows[c["image_id"]]["dataset_index"] == 630]
    if len(special) != 1:
        raise ValueError("Expected exactly one special case 630")
    rng = random.Random(seed)
    for split in ("train", "val", "test"):
        eligible = sorted(
            [
                c
                for c in cases
                if c["image_id"] not in selected
                and c != special[0]
                and rows[c["image_id"]]["original_split"] == split
            ],
            key=lambda c: c["image_id"],
        )
        if len(eligible) < 4:
            raise ValueError(f"Insufficient distinct additional cases for {split}")
        for case in rng.sample(eligible, 4):
            add(case, f"additional_{split}")
    add(special[0], "difficult_automatic_abstention_630")

    def difficulty(case):
        top = case["top_k"]
        # Small inlier margin = harder discrimination; exploratory, not confidence.
        return (
            top[0]["inliers"] - max((m["inliers"] for m in top[1:]), default=0),
            top[0]["inliers"],
            case["image_id"],
        )

    remaining = sorted(
        [c for c in cases if c["image_id"] not in selected and c["top_k"]],
        key=difficulty,
    )
    for case in remaining[:3]:
        add(case, "difficult_small_inlier_margin")
    if len(selected) != 24:
        raise ValueError("Pilot must contain 24 distinct image occurrences")
    return {
        "seed": seed,
        "selected": list(selected.values()),
        "absent_index_500": not any(r["dataset_index"] == 500 for r in rows.values()),
        "automatic_confidence": "not_available_not_calibrated",
        "support_bands": "ge40/lt40 inliers, plus automatic abstention; descriptive only",
        "excluded": [c["alignment_id"] for c in cases if c["image_id"] not in selected],
    }


def geometry(query_rgb, candidate_rgb, seed):
    """Recompute geometry at an EXISTING PTS; no retrieval or rank modification."""
    import cv2

    query, candidate = features(query_rgb), features(candidate_rgb)
    q, c = query["descriptors"], candidate["descriptors"]
    pairs = (
        cv2.BFMatcher().knnMatch(q, c, k=2)
        if q is not None and c is not None and len(c) >= 2
        else []
    )
    good = [a for a, b in pairs if a.distance < 0.75 * b.distance]
    source = np.float32([query["points"][m.queryIdx].pt for m in good]).reshape(-1, 2)
    target = np.float32([candidate["points"][m.trainIdx].pt for m in good]).reshape(
        -1, 2
    )
    homography, mask = None, np.zeros(len(good), dtype=bool)
    if len(good) >= 4:
        cv2.setRNGSeed(seed)
        homography, found = cv2.findHomography(source, target, cv2.RANSAC, 3.0)
        if found is not None:
            mask = found.ravel().astype(bool)
    valid = (
        homography is not None
        and np.isfinite(homography).all()
        and abs(np.linalg.det(homography)) > 1e-8
        and mask.sum() >= 4
    )
    if valid:
        corners = cv2.perspectiveTransform(
            np.float32([[[0, 0], [307, 0], [307, 171], [0, 171]]]), homography
        )[0]
        valid = bool(
            np.isfinite(corners).all()
            and cv2.isContourConvex(corners)
            and 100 < abs(cv2.contourArea(corners)) < 4 * 308 * 172
        )
    panel = Image.new("RGB", (616, 172))
    panel.paste(Image.fromarray(query["norm"]).convert("RGB"))
    panel.paste(Image.fromarray(candidate["norm"]).convert("RGB"), (308, 0))
    draw = ImageDraw.Draw(panel)
    for a, b, keep in zip(source, target, mask, strict=True):
        draw.line(
            (float(a[0]), float(a[1]), float(b[0] + 308), float(b[1])),
            fill="lime" if keep else "red",
            width=1,
        )
    overlay = None
    if valid:
        warped = cv2.warpPerspective(query["norm"], homography, (308, 172))
        overlay = Image.fromarray(
            cv2.addWeighted(warped, 0.5, candidate["norm"], 0.5, 0)
        )
    return (
        {
            "method": "fixed-pair SIFT/RANSAC reconstruction; interior CLAHE coordinates",
            "source_points": source.tolist(),
            "target_points": target.tolist(),
            "inlier_mask": mask.tolist(),
            "matches": len(good),
            "inliers": int(mask.sum()),
            "homography": homography.tolist()
            if homography is not None and np.isfinite(homography).all()
            else None,
            "overlay_valid": bool(valid),
            "overlay_limitation": "Numerical/geometric plausibility only; not identity verification",
        },
        panel,
        overlay,
    )


def prepare(root, evidence_dir, output, seed=20261103):
    root, evidence_dir, output = [
        Path(p).resolve() for p in (root, evidence_dir, output)
    ]
    if output.is_relative_to(root) or output.is_relative_to(evidence_dir):
        raise ValueError("Pilot must be separate from originals and existing evidence")
    evidence = verify(evidence_dir)
    source_sha = file_sha(evidence_dir / "evidence.json")
    manifest = json.loads((evidence_dir / "manifest.json").read_bytes())
    selection = select_pilot(evidence, manifest, seed)
    pilot_id = sha(encoded([source_sha, selection, "alignment-pilot-v1"]))
    if output.exists():
        data = load_pilot(output)
        if data["pilot_id"] != pilot_id:
            raise ValueError("Existing pilot differs; use a new directory")
        return data
    if (
        file_sha(safe_path(root, evidence["source_video"]))
        != evidence["source_video_sha256"]
    ):
        raise ValueError("Source video changed")
    output.mkdir(parents=True)
    (output / "media").mkdir()
    rows = {r["image_id"]: r for r in manifest["images"]}
    cases = {c["alignment_id"]: c for c in evidence["cases"]}
    media, prepared = {}, []

    def image_file(name, image):
        path = output / "media" / f"{name}.png"
        image.save(path)
        key = f"{name}.png"
        media[key] = {"path": f"media/{key}", "sha256": file_sha(path)}
        return key

    decoder = Decoder(
        safe_path(root, evidence["source_video"]), max_frames=20000, max_seconds=1200
    )
    try:
        for number, chosen in enumerate(selection["selected"], 1):
            case = cases[chosen["alignment_id"]]
            row = rows[case["image_id"]]
            _, original = read_image(root, row)
            original_media = image_file(f"{number}-original", original)
            alternatives = []
            for rank, match in enumerate(case["top_k"], 1):
                frames = decoder.frames(
                    match["second"] - 1e-8, match["second"] + 0.1, 0
                )
                found = next((f for f in frames if f["pts"] == match["pts"]), None)
                if (
                    found is None
                    or found["decoded_rgb_sha256"] != match["decoded_rgb_sha256"]
                ):
                    raise ValueError(
                        "Existing candidate failed exact PTS/RGB reconstruction"
                    )
                detail, lines, overlay = geometry(
                    np.asarray(original), found["rgb"], evidence["config"]["seed"]
                )
                detail["original_metric_counts_agree"] = (
                    detail["inliers"] == match["inliers"]
                    and detail["matches"] == match["matches"]
                )
                alternatives.append(
                    {
                        "rank": rank,
                        "metrics": match,
                        "frame_media": image_file(
                            f"{number}-{rank}-frame", Image.fromarray(found["rgb"])
                        ),
                        "lines_media": image_file(f"{number}-{rank}-lines", lines),
                        "overlay_media": image_file(f"{number}-{rank}-overlay", overlay)
                        if overlay
                        else None,
                        "geometry": detail,
                        "close_to_best_inliers": rank > 1
                        and match["inliers"] >= 0.8 * case["top_k"][0]["inliers"],
                    }
                )
            control = None
            if case["controls"]:
                match = min(case["controls"], key=lambda m: m["second"])
                frames = decoder.frames(
                    match["second"] - 1e-8, match["second"] + 0.1, 0
                )
                found = next((f for f in frames if f["pts"] == match["pts"]), None)
                if (
                    found is None
                    or found["decoded_rgb_sha256"] != match["decoded_rgb_sha256"]
                ):
                    raise ValueError("Existing control changed")
                control = {
                    "metrics": match,
                    "media": image_file(
                        f"{number}-control", Image.fromarray(found["rgb"])
                    ),
                    "reference_status": "negative_candidate_needs_human_reference",
                }
            prepared.append(
                {
                    **chosen,
                    "original": row,
                    "original_media": original_media,
                    "automatic_status": case["status"],
                    "automatic_confidence": "not_available_not_calibrated",
                    "support_band": support_band(case),
                    "candidate_interval": case["candidate_interval"],
                    "time_uncertainty": "Observed support range only; JPEG capture-time precision unknown",
                    "alternatives": alternatives,
                    "control": control,
                    "audit_event_links": case["audit_event_links"],
                }
            )
            print(json.dumps({"prepared": number, "total": 24}), flush=True)
    finally:
        decoder.close()
    if source_sha != file_sha(evidence_dir / "evidence.json"):
        raise ValueError("Source evidence changed")
    data = {
        "schema": "alignment-pilot-v1",
        "pilot_id": pilot_id,
        "source_evidence_sha256": source_sha,
        "source_video": evidence["source_video"],
        "source_video_sha256": evidence["source_video_sha256"],
        "selection": selection,
        "cases": prepared,
        "media": media,
        "decoded_frames": decoder.decoded,
        "summary": dict(Counter(c["original"]["original_split"] for c in prepared)),
        "decisions_initialized": "pending; no human annotations",
    }
    (output / "pilot.json").write_bytes(encoded(data))
    (output / "checksums.json").write_bytes(
        encoded(
            {
                p.relative_to(output).as_posix(): file_sha(p)
                for p in sorted(output.rglob("*"))
                if p.is_file()
            }
        )
    )
    return load_pilot(output)


def load_pilot(output):
    output = Path(output)
    checksums = json.loads((output / "checksums.json").read_bytes())
    for name, digest in checksums.items():
        if file_sha(safe_path(output, name)) != digest:
            raise ValueError("Pilot evidence checksum mismatch")
    data = json.loads((output / "pilot.json").read_bytes())
    if len(data["cases"]) != 24 or len({c["image_id"] for c in data["cases"]}) != 24:
        raise ValueError("Pilot must retain 24 unique images")
    for item in data["media"].values():
        if file_sha(safe_path(output, item["path"])) != item["sha256"]:
            raise ValueError("Media binding mismatch")
    return data
