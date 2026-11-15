"""Bounded native-frame check of pending pairs; no global search or admission."""

import argparse
from pathlib import Path

import av
import numpy as np
import pandas as pd
from PIL import Image

from flir_pipeline.data.name_lineage import parse_name_fields
from flir_pipeline.detection.common_executor import external_output
from flir_pipeline.detection.preliminary import read_rows, verify
from flir_pipeline.similarity.storage import file_sha256, read_json, write_json


def check(package, expanded, refined, video_index, video, output):
    external_output(output, package, expanded, refined, video_index, video)
    if output.exists():
        raise ValueError("Fresh external output required")
    verify(package, file_sha256(package / "receipt.json"))
    for folder, field in (
        (expanded, "artifacts"),
        (refined, "artifacts"),
        (video_index, "outputs"),
    ):
        for name, sha in read_json(folder / "receipt.json")[field].items():
            if file_sha256(folder / name) != sha:
                raise ValueError("Evidence changed")
    meta = read_json(video_index / "metadata.json")
    if file_sha256(video) != meta["video_sha256"]:
        raise ValueError("Different native video")
    frames = pd.read_parquet(video_index / "frames.parquet")
    rows = {r["content_id"]: r for r in read_rows(package / "manifest.csv")}
    pairs = {
        r["source_content_id"]: r
        for r in read_json(expanded / "selected_candidates.json")
    }
    pending = [
        r
        for r in read_json(refined / "native_evidence.json")
        if r["admissibility"] == "pending_native_provenance_and_object_review"
    ]
    results = []
    for evidence in pending:
        row = rows[evidence["source_content_id"]]
        parsed = parse_name_fields(row["source_member_path"])
        if parsed["family"] != "scene_named":
            raise ValueError("Scene-index-seconds hypothesis does not apply")
        # The upstream claim is tested against recorded PTS, not promoted to truth.
        nominal_seconds = parsed["nominal_index"]
        center = int(np.argmin(abs(frames.pts_seconds.to_numpy() - nominal_seconds)))
        neighbors = frames.iloc[max(0, center - 1) : center + 2]
        wanted = set(int(x) for x in neighbors.pts)
        with Image.open(package / row["image_path"]) as im:
            source = np.asarray(im.convert("RGB"), dtype=np.int16)
        comparisons = []
        with av.open(str(video)) as container:
            stream = container.streams.video[0]
            container.seek(min(wanted), stream=stream, backward=True)
            for frame in container.decode(stream):
                if frame.pts > max(wanted):
                    break
                if frame.pts not in wanted:
                    continue
                native = frame.to_ndarray(format="rgb24").astype(np.int16)
                if native.shape != source.shape:
                    raise ValueError("Native/source dimensions differ")
                delta = abs(source - native)
                record = neighbors.loc[neighbors.pts == frame.pts].iloc[0]
                comparisons.append(
                    dict(
                        native_index=int(record.native_index),
                        pts=int(frame.pts),
                        pts_seconds=float(record.pts_seconds),
                        full_rgb_mae=float(delta.mean()),
                        max_channel_error=int(delta.max()),
                        exact_pixel_fraction=float(np.all(delta == 0, axis=2).mean()),
                    )
                )
        if len(comparisons) != len(wanted):
            raise ValueError("Native seek did not recover the complete bounded window")
        pair = pairs[row["content_id"]]
        results.append(
            dict(
                source_content_id=row["content_id"],
                source_member=row["source_member_path"],
                parsed_name=parsed,
                nominal_seconds_hypothesis=nominal_seconds,
                native_neighbors=comparisons,
                minimum_error_native_index=min(
                    comparisons, key=lambda r: r["full_rgb_mae"]
                )["native_index"],
                source_descriptor_rank1=evidence["source_native_candidates"][0],
                target_descriptor_rank1=evidence["target_native_candidates"][0],
                target_observed_index=pair["target_locator"]["observed_index"],
                geometry_screen_pass=pair["geometry"]["geometry_screen_pass"],
                box_diagnostics=pair["box_diagnostics"],
                same_frame_verified=False,
                transferable=False,
                limitation="Bounded diagnostic agreement is not extraction provenance; processed pixel masks and box verification remain missing",
            )
        )
    output.mkdir(parents=True)
    write_json(output / "native_checks.json", results)
    write_json(
        output / "receipt.json",
        dict(
            video_sha256=meta["video_sha256"],
            artifacts={
                "native_checks.json": file_sha256(output / "native_checks.json")
            },
            input_receipts={
                str(p): file_sha256(p / "receipt.json")
                for p in (package, expanded, refined, video_index)
            },
        ),
    )
    print(
        {
            "checked_pending_pairs": len(results),
            "verified_transfers": 0,
            "new_global_search": False,
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("package", "expanded", "refined", "video_index", "video", "output"):
        parser.add_argument(name, type=Path)
    check(**vars(parser.parse_args()))
