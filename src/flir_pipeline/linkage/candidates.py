"""Bounded exact cross-cosine ranking; no labels, splits or sequence inputs."""

import numpy as np
import pandas as pd

from flir_pipeline.linkage.base import LinkageConfig
from flir_pipeline.similarity.storage import stable_id

BLOCK_ROWS = 32
CANDIDATE_DTYPES = {
    "candidate_id": "string",
    "labeled_content_id": "string",
    "video_content_id": "string",
    "clip_cosine": "float64",
    "dinov2_cosine": "float64",
    "clip_rank": "Int64",
    "dinov2_rank": "Int64",
    "clip_topk": "bool",
    "dinov2_topk": "bool",
    "both_topk": "bool",
    "mean_reciprocal_rank": "float64",
}


def cosine_block(queries: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """A fixed float64 accumulation kernel, independent of operational block size.

    Keep supplied float32 L2 values intact. No renormalization, score rounding or
    clipping: exact ties use target identity, not a score threshold. Rescoring the
    small union uses this same kernel to preserve each candidate's original score.
    """
    return np.einsum("qd,vd->qv", queries, targets, dtype=np.float64, optimize=False)


def generate_candidates(
    labeled_ids,
    video_ids,
    labeled_vectors,
    video_vectors,
    config: LinkageConfig,
    linkage_id: str,
    *,
    block_rows=BLOCK_ROWS,
):
    """Retain the union, never an encoder-weighted raw score or chosen sequence.

    Arrays are aligned to sorted unique content IDs by the source reader. Only
    one B×N_video score block exists at a time; top-k indexes are retained between
    encoders. Missing-encoder scores are recomputed for the union (at most 2k).
    """
    if type(block_rows) is not int or block_rows < 1:
        raise ValueError("block_rows must be a positive integer")
    if (
        not labeled_ids
        or not video_ids
        or labeled_ids != sorted(set(labeled_ids))
        or video_ids != sorted(set(video_ids))
    ):
        raise ValueError("Expected sorted unique nonempty content IDs")
    k = min(config.top_k, len(video_ids))
    rows = []
    for start in range(0, len(labeled_ids), block_rows):
        stop = min(start + block_rows, len(labeled_ids))
        selected = {}
        for encoder in ("clip", "dinov2"):
            scores = cosine_block(
                labeled_vectors[encoder][start:stop], video_vectors[encoder]
            )
            if not np.isfinite(scores).all():
                raise ValueError("Cross-cosine produced non-finite scores")
            # Columns already follow ascending video_content_id. Stable sorting
            # therefore resolves exact ties lexicographically, without argpartition.
            selected[encoder] = np.argsort(-scores, axis=1, kind="stable")[:, :k].copy()
            del scores
        for offset, query in enumerate(range(start, stop)):
            ranks = {
                encoder: {
                    int(index): rank
                    for rank, index in enumerate(selected[encoder][offset], 1)
                }
                for encoder in ("clip", "dinov2")
            }
            union = sorted(ranks["clip"].keys() | ranks["dinov2"].keys())
            values = {
                encoder: cosine_block(
                    labeled_vectors[encoder][query : query + 1],
                    video_vectors[encoder][union],
                )[0]
                for encoder in ("clip", "dinov2")
            }
            for position, target in enumerate(union):
                clip_rank, dino_rank = (
                    ranks["clip"].get(target),
                    ranks["dinov2"].get(target),
                )
                pair = {
                    "labeled_content_id": labeled_ids[query],
                    "video_content_id": video_ids[target],
                }
                rows.append(
                    {
                        "candidate_id": stable_id(
                            {
                                "artifact_kind": "labeled_video_candidate_pair",
                                "linkage_id": linkage_id,
                                **pair,
                            }
                        ),
                        **pair,
                        "clip_cosine": float(values["clip"][position]),
                        "dinov2_cosine": float(values["dinov2"][position]),
                        "clip_rank": clip_rank,
                        "dinov2_rank": dino_rank,
                        "clip_topk": clip_rank is not None,
                        "dinov2_topk": dino_rank is not None,
                        "both_topk": clip_rank is not None and dino_rank is not None,
                        "mean_reciprocal_rank": (
                            (1 / clip_rank if clip_rank else 0)
                            + (1 / dino_rank if dino_rank else 0)
                        )
                        / 2,
                    }
                )
    result = pd.DataFrame(rows, columns=CANDIDATE_DTYPES).astype(CANDIDATE_DTYPES)
    if not result.candidate_id.is_unique:
        raise ValueError("Candidate ID collision")
    return result
