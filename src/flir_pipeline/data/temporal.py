"""Audit filename-derived temporal lineage without manufacturing timestamps."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class TemporalAudit:
    """Derived occurrence lineage, per-sequence summary and transparent candidates."""

    lineage: pd.DataFrame
    sequences: pd.DataFrame
    neighbors: pd.DataFrame
    summary: dict[str, int | str]


def audit_temporal_lineage(manifest: pd.DataFrame, max_frame_gap: int = 1) -> TemporalAudit:
    """Summarize existing filename guesses and nearby cross-split occurrences.

    Only nonempty inferred sequences with nonnegative integer frame indices are
    orderable. Source archive namespaces a sequence. Compare all cross-split
    pairs within max_frame_gap (including equal indices); flag exact content
    equality separately. An index gap has no unit of seconds or verified FPS.
    No inferred sequence is certified as a source video or real timestamp.
    """
    if max_frame_gap < 0:
        raise ValueError("max_frame_gap must be nonnegative")
    required = {"frame_id", "content_id", "original_split"}
    if not required <= set(manifest) or not manifest["frame_id"].is_unique:
        raise ValueError("Temporal audit requires unique frame_id, content_id and original_split")
    lineage = manifest[["frame_id", "content_id", "original_split"]].copy()
    for column in ("source_archive", "source_member_path", "possible_sequence", "temporal_inference_confidence"):
        lineage[column] = manifest.get(column, pd.Series("", index=manifest.index)).fillna("").astype(str)
    indices = pd.to_numeric(manifest.get("possible_frame_index", pd.Series(index=manifest.index, dtype=float)), errors="coerce")
    valid_index = indices.notna() & indices.ge(0) & indices.mod(1).eq(0)
    lineage["possible_frame_index"] = indices.where(valid_index).astype("Int64")
    lineage["order_reconstructable_from_name"] = valid_index & lineage["possible_sequence"].str.strip().ne("")
    lineage["temporal_source"] = "filename_heuristic"
    lineage["timestamp_available"] = False
    orderable = lineage.loc[lineage["order_reconstructable_from_name"]].copy()
    sequence_columns = ["source_archive", "possible_sequence", "records", "unique_contents", "index_min", "index_max", "unique_indices", "train", "val", "test", "confidence", "temporal_source"]
    neighbor_columns = ["source_archive", "possible_sequence", "frame_id_a", "frame_id_b", "split_a", "split_b", "index_a", "index_b", "index_gap", "exact_content_duplicate", "temporal_source"]
    sequence_rows, neighbor_rows = [], []
    # Sorted-window enumeration makes the selection rule explicit and avoids an
    # unrelated embedding similarity computation or an arbitrary nearest sample.
    for (archive, sequence), group in orderable.groupby(["source_archive", "possible_sequence"], sort=True):
        counts = group["original_split"].value_counts()
        sequence_rows.append({
            "source_archive": archive, "possible_sequence": sequence,
            "records": len(group), "unique_contents": group["content_id"].nunique(),
            "index_min": int(group["possible_frame_index"].min()), "index_max": int(group["possible_frame_index"].max()),
            "unique_indices": group["possible_frame_index"].nunique(),
            **{split: int(counts.get(split, 0)) for split in ("train", "val", "test")},
            "confidence": "|".join(sorted(set(group["temporal_inference_confidence"]))),
            "temporal_source": "filename_heuristic",
        })
        rows = list(group.sort_values(["possible_frame_index", "frame_id"]).itertuples(index=False))
        for index, left in enumerate(rows):
            for right in rows[index + 1:]:
                gap = int(right.possible_frame_index - left.possible_frame_index)
                if gap > max_frame_gap:
                    break
                if left.original_split == right.original_split:
                    continue
                neighbor_rows.append({
                    "source_archive": archive, "possible_sequence": sequence,
                    "frame_id_a": left.frame_id, "frame_id_b": right.frame_id,
                    "split_a": left.original_split, "split_b": right.original_split,
                    "index_a": int(left.possible_frame_index), "index_b": int(right.possible_frame_index),
                    "index_gap": gap, "exact_content_duplicate": left.content_id == right.content_id,
                    "temporal_source": "filename_heuristic",
                })
    neighbors = pd.DataFrame(neighbor_rows, columns=neighbor_columns)
    inferred = int(lineage["order_reconstructable_from_name"].sum())
    return TemporalAudit(
        lineage=lineage,
        sequences=pd.DataFrame(sequence_rows, columns=sequence_columns),
        neighbors=neighbors,
        summary={
            "total_records": len(manifest), "inferred_sequences": len(sequence_rows),
            "records_with_inferred_order": inferred, "records_without_inferred_order": len(manifest) - inferred,
            "records_with_verified_timestamps": 0, "max_frame_gap": max_frame_gap,
            "cross_split_neighbor_pairs": len(neighbors),
            "neighbor_pairs_exact_content": int(neighbors["exact_content_duplicate"].sum()),
            "neighbor_pairs_different_content": int((~neighbors["exact_content_duplicate"].astype(bool)).sum()),
            "source": "filename_heuristic; not verified video timestamps",
        },
    )
