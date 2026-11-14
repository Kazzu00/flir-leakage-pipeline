"""Directed pair pilot: frozen alignment references, never sequence ground truth."""

import itertools
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .dependency_contract import Bundle
from .inventory import encoded, file_sha, safe_path, sha
from .matching import Decoder
from .pilot import load_pilot
from .pilot_store import Decision

BOUNDARIES = ((199, 200), (349, 350), (399, 400), (499, 501))
QUOTAS = {
    "diagnostic": 4,
    "nearby": 6,
    "nonconsecutive": 6,
    "change_candidate": 4,
    "negative_candidate": 4,
}
METHOD = {
    "version": "dependency-selection-v1",
    "nearby_max_seconds": 60,
    "nonconsecutive_min_seconds": 10,
    "geometry": "SIFT 500; ratio 0.75; RANSAC 3 px; CPU; resized 384x216",
    "mask": "central 70%x60%; green HUD (G-R>25,G-B>20) dilated 7px excluded; descriptors need max(12,3*size)px clearance",
    "limitations": "HUD grayscale may remain inside mask; matches require visual terrain review. No match does not imply different region. Context +/-1s cannot establish continuity across a long gap.",
}


def references(pilot, reviews):
    """Validate every historic evidence binding without rewriting any annotation."""
    data = load_pilot(pilot)
    export = json.loads(Path(reviews).read_bytes())
    if export.get("schema") != "alignment-pilot-export-v1" or export[
        "pilot_sha256"
    ] != file_sha(Path(pilot) / "pilot.json"):
        raise ValueError("Alignment export does not bind to the source pilot")
    cases = {c["alignment_id"]: c for c in data["cases"]}
    latest = {}
    for row in export["history"]:
        Decision.model_validate({k: row[k] for k in Decision.model_fields if k in row})
        cid = row["alignment_id"]
        if (
            cid not in cases
            or row["version"] != latest.get(cid, {}).get("version", 0) + 1
        ):
            raise ValueError("Unknown alignment or non-contiguous history")
        if row["pilot_sha256"] != export["pilot_sha256"]:
            raise ValueError("Historic pilot binding mismatch")
        if row["expected_version"] != row["version"] - 1:
            raise ValueError("Historic expected version mismatch")
        if row["human_verified_top1"] != (
            row["status"] == "reviewed" and row["label"] == "correct_match"
        ):
            raise ValueError("Historic verification flag conflicts with decision")
        allowed = {cases[cid]["original_media"], cases[cid]["control"]["media"]}
        for alt in cases[cid]["alternatives"]:
            allowed.update(
                alt[k]
                for k in ("frame_media", "lines_media", "overlay_media")
                if alt.get(k)
            )
        if (
            set(row["evidence_consulted"]) != set(row["evidence_hashes"])
            or not set(row["evidence_consulted"]) <= allowed
        ):
            raise ValueError("Historic evidence membership mismatch")
        for key, digest in row["evidence_hashes"].items():
            if data["media"][key]["sha256"] != digest:
                raise ValueError("Historic evidence hash mismatch")
        if row["status"] == "reviewed" and row["label"] == "correct_match":
            required = {
                cases[cid]["original_media"],
                cases[cid]["alternatives"][0]["frame_media"],
            }
            if not required <= set(row["evidence_consulted"]):
                raise ValueError("Accepted alignment lacks original/top1 evidence")
        latest[cid] = row
    accepted = [
        c
        for c in data["cases"]
        if latest.get(c["alignment_id"], {}).get("status") == "reviewed"
        and latest[c["alignment_id"]]["label"] == "correct_match"
    ]
    return data, latest, accepted


def describe(rgb):
    """Mask keypoint support, not only feature centres, to reduce HUD shortcuts."""
    import cv2

    rgb = np.asarray(Image.fromarray(rgb).resize((384, 216)))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mask = np.zeros(gray.shape, np.uint8)
    mask[43:173, 58:326] = 255
    signed = rgb.astype(np.int16)
    hud = (
        (signed[:, :, 1] - signed[:, :, 0] > 25)
        & (signed[:, :, 1] - signed[:, :, 2] > 20)
    ).astype(np.uint8)
    hud = cv2.dilate(hud, np.ones((7, 7), np.uint8))
    mask[hud > 0] = 0
    clearance = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    points, descriptors = cv2.SIFT_create(nfeatures=500).detectAndCompute(gray, mask)
    keep = [
        i
        for i, p in enumerate(points)
        if clearance[min(215, int(p.pt[1])), min(383, int(p.pt[0]))]
        >= max(12, p.size * 3)
    ]
    points = [points[i] for i in keep]
    descriptors = descriptors[keep] if descriptors is not None and keep else None
    pixels = gray[43:173, 58:326].astype(float)
    return dict(
        rgb=rgb,
        gray=gray,
        mask=mask,
        points=points,
        descriptors=descriptors,
        mean=float(pixels.mean()),
        std=float(pixels.std()),
    )


def compare(a, b, seed):
    import cv2

    matches = []
    if (
        a["descriptors"] is not None
        and b["descriptors"] is not None
        and len(b["descriptors"]) >= 2
    ):
        matches = [
            m
            for m, n in cv2.BFMatcher().knnMatch(
                a["descriptors"], b["descriptors"], k=2
            )
            if m.distance < 0.75 * n.distance
        ]
    x = np.float32([a["points"][m.queryIdx].pt for m in matches]).reshape(-1, 2)
    y = np.float32([b["points"][m.trainIdx].pt for m in matches]).reshape(-1, 2)
    h, inliers = None, np.zeros(len(matches), bool)
    if len(matches) >= 4:
        cv2.setRNGSeed(seed)
        h, found = cv2.findHomography(x, y, cv2.RANSAC, 3)
        if found is not None:
            inliers = found.ravel().astype(bool)
    error, valid = None, False
    if h is not None and np.isfinite(h).all() and abs(np.linalg.det(h)) > 1e-8:
        projected = cv2.perspectiveTransform(x.reshape(-1, 1, 2), h).reshape(-1, 2)
        error = (
            float(np.median(np.linalg.norm(projected[inliers] - y[inliers], axis=1)))
            if inliers.any()
            else None
        )
        corners = cv2.perspectiveTransform(
            np.float32([[[58, 43], [325, 43], [325, 172], [58, 172]]]), h
        )[0]
        valid = bool(
            inliers.sum() >= 8
            and np.isfinite(corners).all()
            and cv2.isContourConvex(corners)
            and 100 < abs(cv2.contourArea(corners)) < 4 * 268 * 130
        )
    pixels_a = a["gray"][43:173, 58:326].astype(float).ravel()
    pixels_b = b["gray"][43:173, 58:326].astype(float).ravel()
    correlation = (
        float(np.corrcoef(pixels_a, pixels_b)[0, 1])
        if pixels_a.std() > 0 and pixels_b.std() > 0
        else 0.0
    )
    return {
        "matches": len(matches),
        "inliers": int(inliers.sum()),
        "inlier_ratio": float(inliers.mean()) if len(inliers) else 0,
        "reprojection_median_px": error,
        "source_points": x.tolist(),
        "target_points": y.tolist(),
        "inlier_mask": inliers.tolist(),
        "homography": h.tolist() if h is not None and np.isfinite(h).all() else None,
        "overlay_valid": valid,
        "mean_change": abs(a["mean"] - b["mean"]),
        "central_correlation": correlation,
        "contrast_change": abs(a["std"] - b["std"]),
        "interpretation": "geometric/photometric diagnostics only, not dependency labels",
    }


def select(cases, metrics, seed):
    rng = random.Random(seed)
    pool = list(itertools.combinations(range(len(cases)), 2))
    rng.shuffle(pool)
    selected, used, deficits = [], set(), []

    def gap(pair):
        return abs(
            cases[pair[0]]["alternatives"][0]["metrics"]["second"]
            - cases[pair[1]]["alternatives"][0]["metrics"]["second"]
        )

    def add(pair, category):
        if pair in used:
            raise ValueError("Duplicate pair")
        used.add(pair)
        selected.append((pair, category))

    for boundary in BOUNDARIES:
        found = [
            p
            for p in pool
            if tuple(sorted(cases[i]["original"]["dataset_index"] for i in p))
            == boundary
        ]
        if len(found) == 1:
            add(found[0], "diagnostic")
        else:
            deficits.append(f"Diagnostic {boundary}: missing or ambiguous provenance")
    near = sorted(
        [
            p
            for p in pool
            if p not in used and 0 < gap(p) <= METHOD["nearby_max_seconds"]
        ],
        key=lambda p: sum(cases[i]["alternatives"][0]["metrics"]["second"] for i in p),
    )
    # Spread over AVAILABLE midpoint range, never claim full-video coverage.
    if near:

        def midpoint(p):
            return sum(cases[i]["alternatives"][0]["metrics"]["second"] for i in p) / 2

        for target in np.linspace(
            midpoint(near[0]), midpoint(near[-1]), min(6, len(near))
        ):
            choice = min(
                (p for p in near if p not in used),
                key=lambda p: abs(midpoint(p) - target),
            )
            add(choice, "nearby")
    nonconsecutive = [
        p
        for p in pool
        if p not in used and gap(p) >= METHOD["nonconsecutive_min_seconds"]
    ]
    nonconsecutive.sort(
        key=lambda p: (
            metrics[p]["inliers"] == 0,
            cases[p[0]]["original"]["original_split"]
            == cases[p[1]]["original"]["original_split"],
            -metrics[p]["inliers"],
            -metrics[p].get("central_correlation", 0),
        )
    )
    for p in nonconsecutive[:6]:
        add(p, "nonconsecutive")
    changes = sorted(
        [p for p in pool if p not in used and gap(p) <= METHOD["nearby_max_seconds"]],
        key=lambda p: -(metrics[p]["mean_change"] + metrics[p]["contrast_change"]),
    )
    for p in changes[:4]:
        add(p, "change_candidate")
    # Four controls with eight distinct source images, including hard low-support pairs.
    controls = sorted(
        [p for p in pool if p not in used and gap(p) >= 60],
        key=lambda p: (
            metrics[p]["inliers"],
            metrics[p]["mean_change"] + metrics[p]["contrast_change"],
        ),
    )
    endpoints = set()
    for p in controls:
        if not endpoints.intersection(p):
            add(p, "negative_candidate")
            endpoints.update(p)
            if len(endpoints) == 8:
                break
    counts = Counter(category for _, category in selected)
    deficits += [
        f"{k}: {counts[k]}/{v}; insufficient eligible distinct pairs"
        for k, v in QUOTAS.items()
        if counts[k] < v
    ]
    return selected, deficits


def prepare(pilot, reviews, closure, root, output, seed=20261103):
    pilot, output, root = (
        Path(pilot).resolve(),
        Path(output).resolve(),
        Path(root).resolve(),
    )
    if output.exists() or output.is_relative_to(root) or output.is_relative_to(pilot):
        raise ValueError("Use a new output directory outside source data and pilot")
    source, latest, cases = references(pilot, reviews)
    if len(cases) < 2:
        raise ValueError("Insufficient accepted source alignments")
    if (
        file_sha(safe_path(root, source["source_video"]))
        != source["source_video_sha256"]
    ):
        raise ValueError("Source video changed")
    cases = sorted(cases, key=lambda c: c["image_id"])
    features = [
        describe(
            np.asarray(
                Image.open(
                    safe_path(pilot, source["media"][c["original_media"]]["path"])
                ).convert("RGB")
            )
        )
        for c in cases
    ]
    metrics = {
        p: compare(features[p[0]], features[p[1]], seed)
        for p in itertools.combinations(range(len(cases)), 2)
    }
    selected, deficits = select(cases, metrics, seed)
    output.mkdir(parents=True)
    (output / "media").mkdir()
    (output / "reference").mkdir()
    refs = {}
    for name, path in (
        ("alignment-reviews.json", reviews),
        ("closure.md", closure),
        ("alignment-pilot.json", pilot / "pilot.json"),
    ):
        shutil.copyfile(path, output / "reference" / name)
        refs[name] = {
            "path": f"reference/{name}",
            "sha256": file_sha(Path(path)),
            "source_filename": Path(path).name,
        }
    media, sources = {}, {}

    def add_image(key, image):
        path = output / "media" / f"{key}.png"
        image.save(path)
        media[key] = {"path": f"media/{key}.png", "sha256": file_sha(path)}
        return key

    decoder = Decoder(
        safe_path(root, source["source_video"]), max_frames=15000, max_seconds=1200
    )
    try:
        for i in sorted({i for pair, _ in selected for i in pair}):
            c = cases[i]
            keys = []
            for label, original_key in (
                ("original", c["original_media"]),
                ("aligned", c["alternatives"][0]["frame_media"]),
            ):
                keys.append(
                    add_image(
                        f"image-{i}-{label}",
                        Image.open(
                            safe_path(pilot, source["media"][original_key]["path"])
                        ),
                    )
                )
            keys.append(
                add_image(f"image-{i}-mask", Image.fromarray(features[i]["mask"]))
            )
            contexts = []
            second = c["alternatives"][0]["metrics"]["second"]
            for offset in (-1, 1):
                frames = decoder.frames(second + offset, second + offset + 0.05, 0.1)
                if frames:
                    f = frames[0]
                    key = add_image(
                        f"image-{i}-context-{offset}", Image.fromarray(f.pop("rgb"))
                    )
                    keys.append(key)
                    contexts.append(dict(media=key, requested_offset=offset, **f))
            sources[i] = {
                "image_id": c["image_id"],
                "original": c["original"],
                "alignment_id": c["alignment_id"],
                "alignment_review": latest[c["alignment_id"]],
                "alignment_metrics": c["alternatives"][0]["metrics"],
                "automatic_status": c["automatic_status"],
                "time_uncertainty": c["time_uncertainty"],
                "contexts": contexts,
                "media": keys,
                "warning": "Human accepted alignment; exact instant remains uncertain, especially noise/automatic abstention. No continuity/region inference.",
            }
    finally:
        decoder.close()
    pairs = []
    for (a, b), category in selected:
        pair_id = sha(
            encoded([cases[a]["image_id"], cases[b]["image_id"], METHOD["version"]])
        )[:24]
        m = metrics[a, b]
        panel = Image.new("RGB", (768, 216))
        panel.paste(Image.fromarray(features[a]["rgb"]))
        panel.paste(Image.fromarray(features[b]["rgb"]), (384, 0))
        draw = ImageDraw.Draw(panel)
        for x, y, keep in zip(
            m["source_points"], m["target_points"], m["inlier_mask"], strict=True
        ):
            draw.line((x[0], x[1], y[0] + 384, y[1]), fill="lime" if keep else "red")
        keys = [add_image(f"pair-{pair_id}-geometry", panel)]
        if m["overlay_valid"]:
            import cv2

            warped = cv2.warpPerspective(
                features[a]["gray"], np.array(m["homography"]), (384, 216)
            )
            keys.append(
                add_image(
                    f"pair-{pair_id}-overlay",
                    Image.fromarray(
                        cv2.addWeighted(warped, 0.5, features[b]["gray"], 0.5, 0)
                    ),
                )
            )
        pairs.append(
            {
                "pair_id": pair_id,
                "category": category,
                "a": sources[a],
                "b": sources[b],
                "metrics": m,
                "media": sources[a]["media"] + sources[b]["media"] + keys,
                "secondary": {
                    "cross_split": cases[a]["original"]["original_split"]
                    != cases[b]["original"]["original_split"],
                    "time_gap_seconds": abs(
                        cases[a]["alternatives"][0]["metrics"]["second"]
                        - cases[b]["alternatives"][0]["metrics"]["second"]
                    ),
                    "automatic_abstention_source": any(
                        cases[i]["automatic_status"] == "not_found" for i in (a, b)
                    ),
                },
                "status": "pending",
            }
        )
    blind = [
        p["pair_id"]
        for p in pairs
        if p["category"] in {"diagnostic", "change_candidate"}
    ]
    result = {
        "schema": "dependency-pilot-v1",
        "seed": seed,
        "method": METHOD,
        "references": refs,
        "source_video": source["source_video"],
        "source_video_sha256": source["source_video_sha256"],
        "pairs": pairs,
        "media": media,
        "second_reviewer_pairs": blind,
        "deficits": deficits,
        "absent_index_500": source["selection"]["absent_index_500"],
        "comparisons_calculated": len(metrics),
        "decoded_frames": decoder.decoded,
        "seek_log": decoder.seek_log,
        "counts": dict(Counter(p["category"] for p in pairs)),
        "source_review_summary": {
            "latest": len(latest),
            "accepted": len(cases),
            "retrospective_closure": "Author-reported; does not expand original consulted evidence",
        },
    }
    (output / "pilot.json").write_bytes(encoded(result))
    (output / "checksums.json").write_bytes(
        encoded(
            {
                p.relative_to(output).as_posix(): file_sha(p)
                for p in sorted(output.rglob("*"))
                if p.is_file()
            }
        )
    )
    return load(output)


def load(root):
    root = Path(root)
    checksums = json.loads((root / "checksums.json").read_bytes())
    if "pilot.json" not in checksums:
        raise ValueError("Missing manifest checksum")
    for name, digest in checksums.items():
        if file_sha(safe_path(root, name)) != digest:
            raise ValueError("Dependency bundle checksum mismatch")
    data = json.loads((root / "pilot.json").read_bytes())
    Bundle.model_validate(data)
    if data["schema"] != "dependency-pilot-v1":
        raise ValueError("Wrong pair schema")
    ids, endpoints = set(), set()
    for pair in data["pairs"]:
        key = tuple(sorted(pair[s]["image_id"] for s in ("a", "b")))
        if pair["pair_id"] in ids or key in endpoints or key[0] == key[1]:
            raise ValueError("Duplicate pair or self pair")
        ids.add(pair["pair_id"])
        endpoints.add(key)
        if any(pair[side]["original"]["dataset_index"] == 500 for side in ("a", "b")):
            raise ValueError("Historical index 500 is absent")
        for media in pair["media"]:
            entry = data["media"][media]
            if file_sha(safe_path(root, entry["path"])) != entry["sha256"]:
                raise ValueError("Media checksum mismatch")
    if not set(data["second_reviewer_pairs"]) <= ids:
        raise ValueError("Unknown second reviewer pair")
    return data
