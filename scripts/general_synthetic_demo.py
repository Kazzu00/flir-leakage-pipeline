"""Produce four independent, private-data-free reproducible demonstrations.

Usage: uv run python scripts/general_synthetic_demo.py NEW_EXTERNAL_DIRECTORY
Video cases use optional PyAV with synthetic lossless FFV1; no model downloads.
"""

from __future__ import annotations

import argparse
import hashlib
from fractions import Fraction
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from flir_pipeline.general.contract import Evidence, Manifest, Relation
from flir_pipeline.general.ingestion import ingest
from flir_pipeline.general.localization import localize
from flir_pipeline.general.publication import build, verify, write_json
from flir_pipeline.splitting.base import SplitConfig


def generate(root: Path, classes: int, videos: int, difficult=False):
    root.mkdir(parents=True, exist_ok=False)
    catalog = [
        {"class_id": 7 + i * 3, "name": f"synthetic category {i}"}
        for i in range(classes)
    ]
    images = []
    for i in range(12):
        name = f"image_{i}.png"
        Image.new("RGB", (16, 16), (i % 11, 100, 170)).save(root / name)
        label = None
        if classes:
            label = f"annotation_{i}.txt"
            text = (
                ""
                if i % 4 == 0
                else f"{catalog[i % classes]['class_id']} 0.5 0.5 0.2 0.2\n"
            )
            (root / label).write_text(text, encoding="utf-8")
        images.append(
            {
                "frame_id": f"observation_{i}",
                "image": name,
                "label": label,
                "original_split": ["train", "val", "test"][i % 3],
            }
        )
    source_videos = []
    if videos:
        import av

        for j in range(videos):
            path = root / f"source_{j}.avi"
            with av.open(str(path), mode="w") as output:
                stream = output.add_stream("ffv1", rate=2)
                stream.width, stream.height, stream.pix_fmt = 16, 16, "bgr0"
                for i in range(3):
                    # Two exact appearances in each video; all alternatives survive.
                    with Image.open(
                        root / f"image_{j if i != 1 else j + 1}.png"
                    ) as image:
                        frame = av.VideoFrame.from_ndarray(
                            np.array(image), format="rgb24"
                        )
                    frame.pts, frame.time_base = i, Fraction(1, 2)
                    for packet in stream.encode(frame):
                        output.mux(packet)
                for packet in stream.encode():
                    output.mux(packet)
            source_videos.append(
                {
                    "video_id": f"source_{j}",
                    "path": path.name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    cfg = {"classes": catalog, "images": images, "videos": source_videos}
    config_path = root / "images.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    manifest = ingest(config_path, root)
    if videos:
        manifest = localize(manifest, root, root / "operations/localization.json")
    if not classes or difficult:
        evidence_path = root / "synthetic_review.txt"
        evidence_path.write_text(
            "Synthetic declared decisions for software invariants only.\n",
            encoding="utf-8",
        )
        evidence = Evidence(
            path=evidence_path.name,
            sha256=hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
            method="synthetic_fixture",
            version="1",
            provenance="independent_demo",
            reviewer="fixture_author" if difficult else None,
            reviewed_at="2026-01-01T00:00:00Z" if difficult else None,
        )
        edges = []
        for i in range(1, 11) if difficult else [1]:
            edges.append(
                Relation(
                    relation_id=f"dependency_{i}",
                    left=manifest.records[0].content_id,
                    right=manifest.records[i].content_id,
                    kind="external",
                    status="confirmed" if difficult else "candidate",
                    accepted=difficult,
                    evidence=evidence,
                )
            )
        manifest = Manifest.model_validate(
            {**manifest.model_dump(), "relations": edges}
        )
    write_json(root / "manifest.json", manifest.canonical())
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    reports = {}
    for name, classes, videos, difficult in (
        ("two_classes_no_video", 2, 0, False),
        ("seven_classes_four_videos", 7, 4, False),
        ("unlabeled_candidates", 0, 0, False),
        ("empty_conflicts_multiple_infeasible", 2, 2, True),
    ):
        root = args.output / name
        manifest = generate(root / "inputs", classes, videos, difficult)
        config = SplitConfig(target_ratios=(0.6, 0.2, 0.2))
        build(manifest, config, root / "inputs", root / "publication")
        reports[name] = verify(root / "publication", root / "inputs")
        write_json(root / "replay.json", reports[name])
    write_json(args.output / "verification.json", reports)
    print(f"Verified {len(reports)} independent synthetic publications")


if __name__ == "__main__":
    main()
