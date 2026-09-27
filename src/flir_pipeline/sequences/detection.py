"""Linear-memory occurrence change detection in the two original feature spaces."""

import numpy as np
import pandas as pd

from flir_pipeline.sequences.base import SequenceConfig
from flir_pipeline.similarity.storage import stable_id

EVENT_DTYPES = {
    "event_id": "int64",
    "candidate_id": "string",
    "video_id": "string",
    "first_candidate_sample_index": "int64",
    "last_candidate_sample_index": "int64",
    "candidate_count": "int64",
    "coarse_sample_index": "int64",
    "coarse_score": "float64",
    "coarse_median": "float64",
    "coarse_P1": "float64",
    "high_confidence": "bool",
    "search_start_sample_index": "int64",
    "search_end_sample_index": "int64",
    "localized_sample_index": "int64",
    "localized_F3": "float64",
    "status": "string",
}


def stable_percentiles(values: np.ndarray) -> np.ndarray:
    """Round BEFORE average-tie ranking; NaN windows are outside the denominator.

    Percentile is rank / valid_count (pandas pct=True), not (rank-1)/(N-1).
    Thus a constant video does not make every temporal cut a top-tail candidate.
    """
    values = np.asarray(values, dtype=np.float64)
    if np.isinf(values).any():
        raise ValueError("Change scores must be finite or undefined (NaN)")
    return pd.Series(np.round(values, 12)).rank(method="average", pct=True).to_numpy()


def centroid_changes(
    vectors: np.ndarray, windows=(1, 3, 5, 10, 20)
) -> dict[int, np.ndarray]:
    """At cut t compare [t-w,t) with [t,t+w), normalizing each centroid.

    Float64 prefix sums over stored float32 L2 vectors avoid an NxN matrix and
    minimize cancellation in repeated blocks. Zero centroid norms are undefined,
    never replaced with artificial changes. No truncated windows at video edges.
    """
    values = np.asarray(vectors)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Expected finite occurrence vectors")
    n, dimension = values.shape
    prefix = np.zeros((n + 1, dimension), dtype=np.float64)
    np.cumsum(values, axis=0, dtype=np.float64, out=prefix[1:])
    result = {}
    for w in windows:
        changes = np.full(n - 1, np.nan, dtype=np.float64)
        cuts = np.arange(w, n - w + 1)
        if len(cuts):
            before = prefix[cuts] - prefix[cuts - w]
            after = prefix[cuts + w] - prefix[cuts]
            norms = np.linalg.norm(before, axis=1) * np.linalg.norm(after, axis=1)
            valid = norms > 0
            cosine = np.einsum("ij,ij->i", before[valid], after[valid]) / norms[valid]
            changes[cuts[valid] - 1] = 1 - np.clip(cosine, -1, 1)
        result[w] = changes
    return result


def multiscale_scores(table: pd.DataFrame) -> pd.DataFrame:
    """Consensus requires BOTH encoders and ALL three coarse scales."""
    table = table.copy()
    for w in (1, 3, 5, 10, 20):
        table[f"P{w}"] = np.minimum(
            table[f"clip_percentile_w{w}"], table[f"dinov2_percentile_w{w}"]
        )
    coarse = table[["P5", "P10", "P20"]].to_numpy()
    table["S"] = np.min(coarse, axis=1)
    table["coarse_median"] = np.median(coarse, axis=1)
    table["F3"] = table.P3
    # P1 combines per-encoder percentiles, since raw change scales differ.
    # It only breaks coarse ties; neither S nor the F3 localizer depends on P1.
    return table


def localize_events(
    scores: pd.DataFrame, config: SequenceConfig, detection_id: str
) -> pd.DataFrame:
    """Merge coarse candidates, then allocate disjoint local search intervals."""
    events = []
    for video, timeline in scores.groupby("video_id", sort=True):
        timeline = (
            timeline.sort_values("sample_index")
            .set_index("sample_index", drop=False)
            .rename_axis(None)
        )
        candidates = timeline.loc[timeline.S.ge(config.candidate_threshold)]
        if candidates.empty:
            continue
        groups = candidates.sample_index.diff().gt(config.merge_gap).cumsum()
        coarse_events = []
        for _, group in candidates.groupby(groups, sort=True):
            representative = group.sort_values(
                ["S", "coarse_median", "P1", "sample_index"],
                ascending=[False, False, False, True],
                na_position="last",
            ).iloc[0]
            coarse_events.append(
                {
                    "video_id": video,
                    "first_candidate_sample_index": int(group.sample_index.min()),
                    "last_candidate_sample_index": int(group.sample_index.max()),
                    "candidate_count": len(group),
                    "coarse_sample_index": int(representative.sample_index),
                    "coarse_score": float(representative.S),
                    "coarse_median": float(representative.coarse_median),
                    "coarse_P1": float(representative.P1),
                    "high_confidence": bool(
                        group.S.ge(config.high_confidence_threshold).any()
                    ),
                }
            )
        centers = [e["coarse_sample_index"] for e in coarse_events]
        for i, event in enumerate(coarse_events):
            center = centers[i]
            lower = max(
                int(timeline.sample_index.min()), center - config.localization_radius
            )
            upper = min(
                int(timeline.sample_index.max()), center + config.localization_radius
            )
            if i:
                lower = max(lower, (centers[i - 1] + center) // 2 + 1)
            if i + 1 < len(centers):
                upper = min(upper, (center + centers[i + 1]) // 2)
            search = timeline.loc[lower:upper].dropna(subset=["F3"]).copy()
            if search.empty:
                raise ValueError(
                    "Coarse event has no valid F3 cut inside its search interval"
                )
            search["distance_to_coarse"] = abs(search.sample_index - center)
            best = search.sort_values(
                ["F3", "distance_to_coarse", "sample_index"],
                ascending=[False, True, True],
            ).iloc[0]
            event.update(
                {
                    "event_id": len(events),
                    "candidate_id": stable_id(
                        {
                            "kind": "sequence_candidate",
                            "detection_id": detection_id,
                            **event,
                        }
                    ),
                    "search_start_sample_index": lower,
                    "search_end_sample_index": upper,
                    "localized_sample_index": int(best.sample_index),
                    "localized_F3": float(best.F3),
                    "status": "candidate",
                }
            )
            events.append(event)
    return pd.DataFrame(events, columns=EVENT_DTYPES).astype(EVENT_DTYPES)


def detect_tables(records, embeddings, config, detection_id):
    """Each encoder's record_index mapping supplies its own embedding rows."""
    timelines = []
    for video, occurrences in records.groupby("video_id", sort=True):
        occurrences = occurrences.sort_values("sample_index")
        scores = pd.DataFrame(
            {
                "video_id": video,
                "sample_index": occurrences.sample_index.to_numpy(dtype=np.int64)[1:],
                "timestamp_seconds": occurrences.timestamp_seconds.to_numpy()[1:],
            }
        )
        for encoder in ("clip", "dinov2"):
            rows = occurrences[f"{encoder}_embedding_row"].to_numpy(dtype=np.int64)
            changes = centroid_changes(embeddings[encoder][rows])
            for w, values in changes.items():
                scores[f"{encoder}_change_w{w}"] = values
                scores[f"{encoder}_percentile_w{w}"] = stable_percentiles(values)
        timelines.append(multiscale_scores(scores))
    scores = pd.concat(timelines, ignore_index=True)
    return scores, localize_events(scores, config, detection_id)
