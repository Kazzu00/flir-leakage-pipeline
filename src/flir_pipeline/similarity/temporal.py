"""Post-hoc temporal/split relations without duplicating contents in similarity."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from flir_pipeline.data.temporal import audit_temporal_lineage
from flir_pipeline.similarity.cosine import distribution_summary

SPLIT_BITS = {"train": 1, "val": 2, "test": 4}


def build_content_provenance(manifest: pd.DataFrame, content_index: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Preserve all occurrences and use strict consensus for content-level time.

    Reuse the existing filename-lineage validator. An inferred archive/sequence
    or frame index is usable only if every occurrence agrees and is known.
    Ambiguity is retained in sets and occurrence rows, never resolved by choosing
    the representative image. Split membership keeps every historical split.
    """
    records = audit_temporal_lineage(manifest, max_frame_gap=0).lineage
    if set(content_index.content_id) != set(records.content_id) or not content_index.content_id.is_unique:
        raise ValueError("Provenance requires complete unique content coverage")
    mapping = content_index.set_index("content_id").embedding_row
    records["embedding_row"] = records.content_id.map(mapping)
    rows = []
    for content_id, group in records.groupby("content_id", sort=True):
        sequence_known = group.source_archive.str.strip().ne("") & group.possible_sequence.str.strip().ne("")
        sequences = sorted(set(zip(group.loc[sequence_known, "source_archive"], group.loc[sequence_known, "possible_sequence"], strict=True)))
        sequence_valid = bool(sequence_known.all() and len(sequences) == 1)
        indices = sorted(int(x) for x in group.possible_frame_index.dropna().unique())
        index_valid = bool(sequence_valid and group.possible_frame_index.notna().all() and len(indices) == 1)
        memberships = sorted(set(group.original_split.dropna()) & set(SPLIT_BITS))
        rows.append({
            "content_id": content_id, "occurrence_count": len(group),
            "sequence_candidates": json.dumps(sequences),
            "sequence_key": json.dumps(sequences[0]) if sequence_valid else "",
            "sequence_id": sequences[0][1] if sequence_valid else "",
            "sequence_provenance_valid": sequence_valid,
            "frame_index_set": json.dumps(indices),
            "frame_index": indices[0] if index_valid else None,
            "frame_index_valid": index_valid,
            "temporal_source": "filename_heuristic", "timestamp_verified": False,
            "temporal_confidence_set": json.dumps(sorted(set(group.temporal_inference_confidence))),
            "split_membership_set": json.dumps(memberships),
            "split_membership_complete": bool(group.original_split.isin(SPLIT_BITS).all()),
            "split_mask": sum(SPLIT_BITS[s] for s in memberships),
        })
    contents = content_index.merge(pd.DataFrame(rows), on="content_id", validate="one_to_one").sort_values("embedding_row").reset_index(drop=True)
    contents["frame_index"] = contents.frame_index.astype("Int64")
    return contents, records.sort_values("frame_id").reset_index(drop=True)


def annotate_pairs(pairs: pd.DataFrame, contents: pd.DataFrame) -> pd.DataFrame:
    """Vectorized posterior joins for either i<j pairs or directed kNN edges.

    cross-split means there exists a known occurrence from each content with
    unequal historical splits. Even two identical multi-split membership sets
    can cross splits. Insufficient/ambiguous provenance remains explicit.
    """
    a, b = pairs.query_row.to_numpy(), pairs.neighbor_row.to_numpy()
    sequences = contents.sequence_key.to_numpy()
    known = contents.sequence_provenance_valid.to_numpy(dtype=bool)
    comparable = known[a] & known[b]
    same = sequences[a] == sequences[b]
    result = pairs.copy()
    result["same_sequence"] = pd.array(np.where(comparable, same, None), dtype="boolean")
    indices = contents.frame_index.to_numpy(dtype=float, na_value=np.nan)
    delta = np.where(comparable & same, np.abs(indices[a]-indices[b]), np.nan)
    result["frame_delta"] = pd.array(delta, dtype="Int64")
    masks = contents.split_mask.to_numpy(dtype=np.int32)
    union = masks[a] | masks[b]
    crosses = (masks[a] != 0) & (masks[b] != 0) & ((union & (union-1)) != 0)
    split_known = contents.split_membership_complete.to_numpy(dtype=bool)
    split_complete = split_known[a] & split_known[b]
    result["historical_cross_split"] = pd.array(np.where(crosses | split_complete, crosses, None), dtype="boolean")
    result["split_provenance_complete"] = split_complete
    result["provenance_complete"] = comparable & np.isfinite(indices[a]) & np.isfinite(indices[b]) & split_complete
    return result


def analyze_temporal_neighbors(neighbors: pd.DataFrame) -> dict:
    """Report denominators for known sequences and all edges, including unknowns."""
    result = {}
    for label, group in (("topk", neighbors), ("rank1", neighbors.loc[neighbors.neighbor_rank == 1])):
        known = group.same_sequence.notna()
        same = int(group.same_sequence.fillna(False).sum())
        result[label] = {"count": len(group), "known_sequence_count": int(known.sum()),
                         "unknown_sequence_count": int((~known).sum()), "same_sequence_count": same,
                         "same_sequence_percentage_all": 100*same/len(group) if len(group) else None,
                         "same_sequence_percentage_known": 100*same/int(known.sum()) if known.any() else None}
    return result


def summarize_pair_relations(pairs: pd.DataFrame, quantiles: tuple[float, ...], delta_bounds: tuple[int, ...]) -> dict[str, pd.DataFrame]:
    """Stratify unordered pairs; inclusive quantile thresholds retain all ties.

    Quantile cohorts are nested, not disjoint. Historical cross-split counts
    overlap same/different-sequence counts and never assert confirmed leakage.
    """
    sequence_rows = []
    for label, mask in (("same_sequence", pairs.same_sequence.fillna(False)),
                        ("different_sequence", (~pairs.same_sequence).fillna(False)),
                        ("unknown_sequence", pairs.same_sequence.isna())):
        sequence_rows.append({"relation": label, **distribution_summary(pairs.loc[mask, "cosine_similarity"].to_numpy())})
    quantile_rows = []
    for quantile in quantiles:
        threshold = float(np.quantile(pairs.cosine_similarity.to_numpy(), quantile, method="linear"))
        selected = pairs.loc[pairs.cosine_similarity >= threshold]
        quantile_rows.append({"quantile": quantile, "top_percentage": 100*(1-quantile), "threshold_cosine": threshold,
                              "pair_count": len(selected), "same_sequence_count": int(selected.same_sequence.fillna(False).sum()),
                              "different_sequence_count": int((~selected.same_sequence).fillna(False).sum()),
                              "unknown_sequence_count": int(selected.same_sequence.isna().sum()),
                              "cross_split_count": int(selected.historical_cross_split.fillna(False).sum()),
                              "unknown_cross_split_count": int(selected.historical_cross_split.isna().sum()),
                              "insufficient_provenance_count": int((~selected.provenance_complete).sum())})
    temporal_rows = []
    previous = -1
    delta = pairs.frame_delta
    for upper in (*delta_bounds, None):
        label = f">{previous}" if upper is None else str(upper) if upper == previous+1 else f"{previous+1}-{upper}"
        mask = delta.gt(previous) & (delta.le(upper) if upper is not None else True)
        temporal_rows.append({"frame_delta_bin": label, **distribution_summary(pairs.loc[mask.fillna(False), "cosine_similarity"].to_numpy())})
        if upper is not None:
            previous = upper
    historical_rows = []
    for label, mask in (("cross_split_candidate", pairs.historical_cross_split.fillna(False)),
                        ("no_cross_split_relation", (~pairs.historical_cross_split).fillna(False)),
                        ("unknown_cross_split_relation", pairs.historical_cross_split.isna())):
        historical_rows.append({"relation": label, **distribution_summary(pairs.loc[mask, "cosine_similarity"].to_numpy())})
    return {"sequence_similarity": pd.DataFrame(sequence_rows), "quantile_candidates": pd.DataFrame(quantile_rows),
            "frame_delta_similarity": pd.DataFrame(temporal_rows), "historical_split_similarity": pd.DataFrame(historical_rows)}
