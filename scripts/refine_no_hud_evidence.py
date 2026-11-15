"""Independently retrieve native neighbors of surviving target candidates.

No target filename is interpreted as a timestamp. Even coincident rank-one
neighbors remain hypotheses when processing provenance and object review are absent.
"""

import argparse
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from expand_no_hud_transfer import target_image

from flir_pipeline.data.no_hud import fingerprint
from flir_pipeline.detection.common_executor import external_output
from flir_pipeline.similarity.storage import file_sha256, read_json, write_json


def refine(expanded, video_index, archives, output):
    external_output(output, expanded, video_index, archives)
    if output.exists():
        raise ValueError("Fresh external output required")
    for name, sha in read_json(expanded / "receipt.json")["artifacts"].items():
        if file_sha256(expanded / name) != sha:
            raise ValueError("Expanded evidence changed")
    for name in ("fingerprints.npy", "frames.parquet", "metadata.json"):
        if (
            file_sha256(video_index / name)
            != read_json(video_index / "receipt.json")["outputs"][name]
        ):
            raise ValueError("Native index changed")
    vectors = np.load(video_index / "fingerprints.npy", allow_pickle=False)
    frames = pd.read_parquet(video_index / "frames.parquet")
    selected = read_json(expanded / "selected_candidates.json")
    evaluated = defaultdict(set)
    for path in (expanded / "pairs").glob("*.json"):
        pair = read_json(path)
        evaluated[pair["source_content_id"]].add(
            pair["target_locator"]["observed_index"]
        )
    mapping = read_json(expanded / "protocol.json")["native_to_target_retrieval_only"]
    strong = [
        r
        for r in read_json(expanded / "native_retrieval.json")
        if r["native_candidates"][0]["score"] >= 0.95
    ]
    missing = [
        r["source_content_id"]
        for r in strong
        if round(
            mapping["slope"] * r["native_candidates"][0]["native_index"]
            + mapping["intercept"]
        )
        not in evaluated[r["source_content_id"]]
    ]
    target_sources = defaultdict(list)
    for pair in selected:
        target_sources[pair["target_content_id"]].append(pair)
    results = []
    for pair in selected:
        image = target_image(archives, pair["target_locator"])
        scores = vectors @ fingerprint(image).astype(np.float32)
        indices = np.argsort(-scores, kind="stable")[:5]
        neighbors = [
            dict(
                native_index=int(frames.iloc[i].native_index),
                pts_seconds=float(frames.iloc[i].pts_seconds),
                score=float(scores[i]),
            )
            for i in indices
        ]
        source = pair["retrieval"]["source_native_candidates"]
        results.append(
            dict(
                source_content_id=pair["source_content_id"],
                target_id=pair["target_id"],
                split=pair["split"],
                source_native_candidates=source,
                target_native_candidates=neighbors,
                rank1_delta=neighbors[0]["native_index"] - source[0]["native_index"],
                target_descriptor_margin=float(scores[indices[0]] - scores[indices[1]]),
                source_descriptor_margin=source[0]["score"] - source[1]["score"],
                near_tied_target_count=len(pair["near_tied_targets"]),
                admissibility="rejected_ambiguous_correspondence"
                if len(pair["near_tied_targets"]) > 1
                or neighbors[0]["native_index"] != source[0]["native_index"]
                else "pending_native_provenance_and_object_review",
                box_diagnostics=pair["box_diagnostics"],
                temporal_identity_verified=False,
                object_equivalence_verified=False,
                training_allowed=False,
            )
        )
    output.mkdir(parents=True)
    write_json(output / "native_evidence.json", results)
    summary = dict(
        evaluated=len(results),
        coincident_native_rank1=sum(r["rank1_delta"] == 0 for r in results),
        differing_native_rank1=sum(r["rank1_delta"] != 0 for r in results),
        admissibility_counts=dict(Counter(r["admissibility"] for r in results)),
        rank1_deltas=dict(Counter(str(r["rank1_delta"]) for r in results)),
        retrieval_probe_threshold_not_acceptance=0.95,
        sources_with_strong_native_neighbor=len(strong),
        unexamined_strong_native_hypotheses=missing,
        reused_target_contents=sum(len(v) > 1 for v in target_sources.values()),
        cross_split_target_contents=sum(
            len({p["split"] for p in v}) > 1 for v in target_sources.values()
        ),
        verified_transfers=0,
        reason="descriptor_neighbors_do_not_certify_native_identity_or_processed_object_integrity",
        expanded_receipt_sha256=file_sha256(expanded / "receipt.json"),
        video_index_receipt_sha256=file_sha256(video_index / "receipt.json"),
    )
    write_json(output / "summary.json", summary)
    write_json(
        output / "receipt.json",
        dict(
            artifacts={p.name: file_sha256(p) for p in output.iterdir() if p.is_file()}
        ),
    )
    print(summary)


if __name__ == "__main__":
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("expanded", "video_index", "archives", "output"):
        parser.add_argument(name, type=Path)
    refine(**vars(parser.parse_args()))
