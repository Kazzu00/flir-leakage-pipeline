"""Multiscale original-L2 change evidence, with intervals and local diagnostics."""

import numpy as np
import pandas as pd

from flir_pipeline.sequences.detection import (
    centroid_changes,
    detect_tables,
    stable_percentiles,
)
from flir_pipeline.sequences.experiments.config import BoundaryConfig
from flir_pipeline.similarity.storage import stable_id

ZONE_COLUMNS = [
    "candidate_id",
    "timeline_id",
    "start",
    "end",
    "peak_position",
    "score",
    "rank",
    "signal",
    "encoder_policy",
    "status",
]


def zones_from_scores(scores, config, signal, policy, input_id):
    rows = []
    score_col = f"{policy}_{signal}_rank"
    eligible = scores.loc[
        scores[f"{policy}_{signal}_eligible"]
        & scores[score_col].ge(config.rank_threshold)
    ]
    for (timeline, _segment), group in eligible.groupby(
        ["timeline_id", "segment_id"], sort=True
    ):
        group = group.sort_values("position")
        for _, interval in group.groupby(
            group.position.diff().gt(config.merge_gap).cumsum()
        ):
            best = interval.sort_values(
                [score_col, "position"], ascending=[False, True]
            ).iloc[0]
            # An interval spans the observed neighboring samples around the cuts.
            # peak_position is diagnostic and NEVER an exact committed boundary.
            start, end = (
                int(interval.previous_position.min()),
                int(interval.position.max()),
            )
            rows.append(
                {
                    "candidate_id": stable_id(
                        {
                            "input": input_id,
                            "config": config.model_dump(mode="json"),
                            "timeline": timeline,
                            "start": start,
                            "end": end,
                            "signal": signal,
                            "policy": policy,
                        }
                    ),
                    "timeline_id": timeline,
                    "start": start,
                    "end": end,
                    "peak_position": int(best.position),
                    "score": float(best[score_col]),
                    "signal": signal,
                    "encoder_policy": policy,
                    "status": "candidate",
                }
            )
    result = pd.DataFrame(rows, columns=ZONE_COLUMNS)
    if len(result):
        result = result.sort_values(
            ["score", "timeline_id", "start"], ascending=[False, True, True]
        ).reset_index(drop=True)
        result["rank"] = np.arange(1, len(result) + 1)
    return result


def boundary_tables(source, config: BoundaryConfig, input_id):
    pieces = []
    legacy_scores, legacy_events, legacy_zones = [], [], []
    for timeline, full in source.timelines().groupby("timeline_id", sort=True):
        full = full.sort_values("position")
        # Missing nominal positions interrupt the calculation, never fabricated time.
        for _, group in full.groupby(
            full.position.diff().gt(config.max_index_gap).cumsum()
        ):
            if len(group) < 2:
                continue
            positions = group.position.to_numpy(dtype=np.int64)
            legacy_records = pd.DataFrame(
                {
                    "video_id": timeline,
                    "sample_index": positions,
                    "timestamp_seconds": group.timestamp_seconds.to_numpy()
                    if "timestamp_seconds" in group
                    else np.full(len(group), np.nan),
                    "clip_embedding_row": group.content_row.to_numpy(dtype=int),
                    "dinov2_embedding_row": group.content_row.to_numpy(dtype=int),
                }
            )
            old_scores, old_events = detect_tables(
                legacy_records, source.embeddings, config.current_v1, input_id
            )
            # The reused function calls its grouping key video_id; the experiment
            # contract must not claim filename-inferred families are source videos.
            old_scores = old_scores.rename(columns={"video_id": "timeline_id"})
            old_events = old_events.rename(columns={"video_id": "timeline_id"})
            old_scores["temporal_source"] = group.temporal_source.iloc[0]
            old_events["temporal_source"] = group.temporal_source.iloc[0]
            legacy_scores.append(old_scores)
            legacy_events.append(old_events)
            for event in old_events.itertuples():
                legacy_zones.append(
                    {
                        "candidate_id": event.candidate_id,
                        "timeline_id": timeline,
                        "start": event.search_start_sample_index,
                        "end": event.search_end_sample_index,
                        "peak_position": event.localized_sample_index,
                        "score": event.coarse_score,
                        "rank": 0,
                        "signal": "current_v1",
                        "encoder_policy": "consensus",
                        "status": "candidate",
                    }
                )
            table = pd.DataFrame(
                {
                    "timeline_id": timeline,
                    "segment_id": stable_id(
                        {
                            "timeline": timeline,
                            "start": int(positions[0]),
                            "end": int(positions[-1]),
                        }
                    ),
                    "position": positions[1:],
                    "previous_position": positions[:-1],
                    "temporal_source": group.temporal_source.iloc[0],
                }
            )
            for encoder in ("clip", "dinov2"):
                changes = centroid_changes(
                    source.embeddings[encoder][group.content_row.to_numpy(dtype=int)],
                    config.windows,
                )
                for w, values in changes.items():
                    series = pd.Series(values)
                    baseline, mad, counts = [], [], []
                    for i in range(len(series)):
                        local = values[
                            max(0, i - config.baseline_radius) : i
                            + config.baseline_radius
                            + 1
                        ]
                        local = np.delete(local, i - max(0, i - config.baseline_radius))
                        local = local[np.isfinite(local)]
                        valid = len(local) >= config.baseline_min_count
                        median = float(np.median(local)) if valid else np.nan
                        baseline.append(median)
                        mad.append(
                            float(np.median(np.abs(local - median)))
                            if valid
                            else np.nan
                        )
                        counts.append(len(local))
                    prefix = f"{encoder}_w{w}"
                    table[f"{prefix}_change"] = values
                    table[f"{prefix}_rank"] = stable_percentiles(values)
                    table[f"{prefix}_local_median"] = baseline
                    table[f"{prefix}_local_mad"] = mad
                    table[f"{prefix}_baseline_n"] = counts
                    table[f"{prefix}_excess"] = values - np.asarray(baseline)
                    table[f"{prefix}_robust_z"] = np.divide(
                        values - np.asarray(baseline),
                        1.4826 * np.asarray(mad),
                        out=np.full(len(values), np.nan),
                        where=np.asarray(mad) > 0,
                    )
                table[f"{encoder}_adjacent_cosine"] = 1 - changes[1]
                for signal, windows in (
                    ("adjacent", (1,)),
                    ("multiscale", config.windows),
                ):
                    ranks = table[[f"{encoder}_w{w}_rank" for w in windows]].to_numpy()
                    changes_array = table[
                        [f"{encoder}_w{w}_change" for w in windows]
                    ].to_numpy()
                    excess = table[
                        [f"{encoder}_w{w}_excess" for w in windows]
                    ].to_numpy()
                    table[f"{encoder}_{signal}_rank"] = np.min(ranks, axis=1)
                    table[f"{encoder}_{signal}_eligible"] = np.all(
                        changes_array >= config.minimum_change, axis=1
                    ) & np.all(excess >= config.local_excess, axis=1)
            for signal in ("adjacent", "multiscale"):
                table[f"consensus_{signal}_rank"] = np.minimum(
                    table[f"clip_{signal}_rank"], table[f"dinov2_{signal}_rank"]
                )
                table[f"consensus_{signal}_eligible"] = (
                    table[f"clip_{signal}_eligible"]
                    & table[f"dinov2_{signal}_eligible"]
                )
                table[f"{signal}_encoder_disagreement"] = (
                    table[f"clip_{signal}_eligible"]
                    & table[f"clip_{signal}_rank"].ge(config.rank_threshold)
                ) != (
                    table[f"dinov2_{signal}_eligible"]
                    & table[f"dinov2_{signal}_rank"].ge(config.rank_threshold)
                )
            pieces.append(table)
    scores = (
        pd.concat(pieces, ignore_index=True)
        if pieces
        else pd.DataFrame(columns=["timeline_id", "position", "previous_position"])
    )
    zones = []
    if len(scores):
        for signal in ("adjacent", "multiscale"):
            for policy in ("clip", "dinov2", "consensus"):
                zones.append(
                    zones_from_scores(scores, config, signal, policy, input_id)
                )
    # Empty ablations must not coerce populated numeric/string columns to object
    # (Pandas 3); Parquet would infer them again and change the logical digest.
    zones = [zone for zone in zones if not zone.empty]
    candidates = (
        pd.concat(zones, ignore_index=True)
        if zones
        else pd.DataFrame(columns=ZONE_COLUMNS)
    )
    if legacy_zones:
        old = (
            pd.DataFrame(legacy_zones, columns=ZONE_COLUMNS)
            .sort_values(
                ["score", "timeline_id", "start"], ascending=[False, True, True]
            )
            .reset_index(drop=True)
        )
        old["rank"] = np.arange(1, len(old) + 1)
        candidates = (
            pd.concat([candidates, old], ignore_index=True) if len(candidates) else old
        )
    coverage = [
        {
            "timeline_id": str(t),
            "occurrences": len(g),
            "unique_contents": g.content_id.nunique(),
            "ordered_occurrences": int(g.position.notna().sum()),
            "scored_cuts": int(scores.timeline_id.eq(t).sum()),
            "unknown_temporal_occurrences": int(g.position.isna().sum()),
        }
        for t, g in source.records.groupby("timeline_id", dropna=False, sort=True)
    ]
    return {
        "current_v1_scores": pd.concat(legacy_scores, ignore_index=True)
        if legacy_scores
        else pd.DataFrame(),
        "current_v1_events": pd.concat(legacy_events, ignore_index=True)
        if legacy_events
        else pd.DataFrame(),
        "temporal_scores": scores,
        "candidate_zones": candidates,
        "coverage": pd.DataFrame(coverage),
        "occurrences": source.records.copy(),
    }
