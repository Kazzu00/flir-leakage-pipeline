"""Optional exact-raster localization on every actually decoded video frame.

No filename index, declared split or sampling timestamp enters retrieval. Failure
to find exact raster identity remains unknown; compressed/altered images need a
separate candidate retrieval and review workflow, never an automatic acceptance.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from flir_pipeline.alignment.visual_localization import raster_digest
from flir_pipeline.data.local_images import declared_file
from flir_pipeline.general.contract import Evidence, Manifest, Position
from flir_pipeline.general.ingestion import verify_sources
from flir_pipeline.utils.hashing import sha256_file


def decoded_frames(path):
    import av

    with av.open(str(path)) as container:
        if len(container.streams.video) != 1:
            raise ValueError("Declare a single video stream per input")
        for ordinal, frame in enumerate(container.decode(video=0)):
            yield ordinal, frame.pts, frame.time_base, frame.to_ndarray(format="rgb24")


def localize(manifest: Manifest, root: Path, evidence_path: Path) -> Manifest:
    if evidence_path.exists():
        raise ValueError("Localization evidence already exists")
    if not evidence_path.resolve().is_relative_to(root.resolve()):
        raise ValueError(
            "Evidence output must be under declared root, in a new operations location"
        )
    if not manifest.videos:
        raise ValueError("Localization requires explicitly checksummed videos")
    verify_sources(manifest, root)
    queries = defaultdict(list)
    for r in manifest.records:
        if "::" in r.source:
            raise ValueError(
                "Direct localization takes local images; archive adapters may import previously verified evidence"
            )
        with Image.open(declared_file(root, r.source)) as image:
            queries[raster_digest(np.array(image.convert("RGB")))].append(r.frame_id)
    found = defaultdict(list)
    scanned = {}
    for video in manifest.videos:
        count, missing_pts = 0, 0
        for ordinal, pts, base, rgb in decoded_frames(declared_file(root, video.path)):
            count += 1
            key = raster_digest(rgb)
            if pts is None or base is None:
                missing_pts += 1
                continue
            for frame_id in queries.get(key, []):
                found[frame_id].append(
                    {
                        "video_id": video.video_id,
                        "kind": "decoded_pts",
                        "status": "confirmed",
                        "seconds": float(pts * base),
                        "pts": int(pts),
                        "time_base": str(base),
                        "decoded_ordinal": ordinal,
                        "decoded_rgb_sha256": key,
                    }
                )
        scanned[video.video_id] = {
            "decoded_frames": count,
            "frames_without_pts": missing_pts,
            "video_sha256": video.sha256,
        }
    verify_sources(manifest, root)
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": "complete_decoded_rgb_sha256_v1",
        "dataset_id": manifest.dataset_id(),
        "videos": scanned,
        "occurrences": dict(found),
        "unknown_frame_ids": sorted(
            r.frame_id for r in manifest.records if r.frame_id not in found
        ),
        "interpretation": "All exact appearances retained; repeated appearances do not identify source occurrence or dependency.",
    }
    with evidence_path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
    evidence = Evidence(
        path=evidence_path.resolve().relative_to(root.resolve()).as_posix(),
        sha256=sha256_file(evidence_path),
        method="complete_decoded_rgb_sha256",
        version="1",
        provenance="read_only_full_video_decode",
    )
    data = manifest.model_dump()
    for record in data["records"]:
        # Preserve prior candidate/external evidence and add observed positions.
        record["positions"].extend(
            Position(**p, evidence=evidence).model_dump()
            for p in found.get(record["frame_id"], [])
        )
    return Manifest.model_validate(data)
