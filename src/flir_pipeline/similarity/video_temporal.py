"""Occurrence-aware source-video relations applied after numerical similarity."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from flir_pipeline.data.video_temporal import audit_video_temporal_lineage

RELATION_COLUMNS = ("same_source_video", "min_sample_index_gap", "min_timestamp_gap_seconds")


def build_video_content_provenance(manifest, content_index):
    records = audit_video_temporal_lineage(manifest)
    if (not content_index.content_id.is_unique
            or set(content_index.content_id) != set(records.content_id)
            or content_index.image_sha256.isna().any()
            or not content_index.image_sha256.eq(content_index.content_id).all()
            or not pd.api.types.is_integer_dtype(content_index.embedding_row.dtype)
            or not np.array_equal(content_index.embedding_row, np.arange(len(content_index)))):
        raise ValueError("Video provenance requires complete ordered unique content coverage")
    records["embedding_row"] = records.content_id.map(content_index.set_index("content_id").embedding_row)
    rows = []
    fields = ["frame_id", "video_id", "source_video", "sample_index", "timestamp_seconds",
              "sample_fps", "source_frame_index_estimate"]
    for content, group in records.groupby("content_id", sort=True):
        rows.append({"content_id": content, "occurrence_count": len(group),
                     "video_id_membership_set": json.dumps(sorted(group.video_id.unique())),
                     "source_video_membership_set": json.dumps(sorted(group.source_video.unique())),
                     # Retain associations, not independent sets that lose video/time pairing.
                     "sampling_occurrences": group[fields].to_json(orient="records", double_precision=15),
                     "temporal_source": "sampled_video_grid",
                     "capture_timestamp_verified": False})
    # Explicit projection prevents arbitrary feature representatives becoming time.
    index = content_index[["content_id", "embedding_row", "image_sha256"]]
    contents = index.merge(pd.DataFrame(rows), on="content_id", validate="one_to_one")
    return contents, records


class VideoRelations:
    """Exact minimum gaps over all occurrences; O(N + M) working space per row.

    For each query occurrence reduce distances to every occurrence in its source
    video by target content. No representative time or video is chosen. The two
    minima may come from different occurrence pairs (e.g. sources with other FPS).
    """

    def __init__(self, records: pd.DataFrame, n: int):
        self.n = n
        self.videos = {
            video: (g.embedding_row.to_numpy(dtype=np.int64),
                    g.sample_index.to_numpy(dtype=np.int64),
                    g.timestamp_seconds.to_numpy(dtype=np.float64))
            for video, g in records.groupby("video_id", sort=False)
        }
        self.occurrences = [[] for _ in range(n)]
        for row in records.itertuples(index=False):
            self.occurrences[row.embedding_row].append((row.video_id, row.sample_index, row.timestamp_seconds))

    def row(self, query: int, targets: np.ndarray):
        missing = np.iinfo(np.int64).max
        samples = np.full(self.n, missing, dtype=np.int64)
        seconds = np.full(self.n, np.inf, dtype=np.float64)
        for video, index, timestamp in self.occurrences[query]:
            rows, indices, times = self.videos[video]
            np.minimum.at(samples, rows, np.abs(indices - index))
            np.minimum.at(seconds, rows, np.abs(times - timestamp))
        # A maximum int64 gap is possible; membership comes from finite time.
        same = np.isfinite(seconds[targets])
        return same, np.where(same, samples[targets], -1), np.where(same, seconds[targets], np.nan)

    def annotate(self, pairs: pd.DataFrame) -> pd.DataFrame:
        same = np.empty(len(pairs), dtype=bool)
        samples = np.empty(len(pairs), dtype=np.int64)
        seconds = np.empty(len(pairs), dtype=np.float64)
        for query, positions in pairs.groupby("query_row", sort=False).indices.items():
            same[positions], samples[positions], seconds[positions] = self.row(
                query, pairs.neighbor_row.to_numpy()[positions])
        result = pairs.copy()
        result["same_source_video"] = same
        result["min_sample_index_gap"] = pd.array(samples, dtype="Int64")
        result.loc[~same, "min_sample_index_gap"] = pd.NA
        result["min_timestamp_gap_seconds"] = seconds
        return result
