"""Cluster label transitions are experimental discontinuity proposals, not cuts."""

import numpy as np
import pandas as pd

from flir_pipeline.sequences.experiments.evaluation import validate_assignments

TRANSITION_COLUMNS = [
    "run_id",
    "timeline_id",
    "position",
    "previous_position",
    "from_cluster",
    "to_cluster",
    "after_run_length",
    "persistent",
    "isolated_one_frame",
    "scene_boundary_confirmed",
]
RUN_COLUMNS = [
    "run_id",
    "timeline_id",
    "block",
    "cluster_id",
    "start",
    "end",
    "length",
    "noise",
]


def compare_zones(candidates, intervals, tolerance=0, exhaustive=False):
    """Many detections can hit one zone; zone recall counts the zone only once."""
    reviewed = intervals.loc[
        intervals.kind.eq("boundary_zone") & intervals.decision.eq("supported")
    ]
    matched = set()
    supported = 0
    for c in candidates.itertuples():
        zones = reviewed.loc[
            reviewed.timeline_id.eq(c.timeline_id)
            & (reviewed.start - tolerance).le(c.end)
            & (reviewed.end + tolerance).ge(c.start)
        ]
        supported += int(len(zones) > 0)
        matched.update(zones.element_id)
    n = len(candidates)
    return {
        "candidate_count": n,
        "reviewed_zone_count": len(reviewed),
        "hit_zone_count": len(matched),
        "reviewed_zone_recall": len(matched) / len(reviewed) if len(reviewed) else None,
        "boundary_zone_hit_rate": supported / n if n else None,
        "candidate_precision": supported / n if exhaustive and n else None,
        "precision_definition_valid": exhaustive,
        "outside_reviewed_zones": n - supported,
        "false_transitions": n - supported if exhaustive else None,
        "tolerance_index_units": tolerance,
        "manual_evidence_not_ground_truth": True,
    }


def transition_tables(source, assignments, config, intervals=None, exhaustive=False):
    validate_assignments(assignments, source.contents)
    runs, transitions, returns, metrics = [], [], [], []
    for run_id, assignment in assignments.groupby("run_id", sort=True):
        labels = assignment.set_index("content_id").cluster_id
        eligible, all_cuts = 0, 0
        for timeline, group in source.timelines().groupby("timeline_id", sort=True):
            group = group.sort_values("position").copy()
            group["cluster_id"] = group.content_id.map(labels)
            for block, contiguous in group.groupby(
                group.position.diff().gt(config.max_index_gap).cumsum()
            ):
                values = contiguous.cluster_id.to_numpy(dtype=int)
                all_cuts += max(0, len(values) - 1)
                eligible += int(np.sum((values[1:] >= 0) & (values[:-1] >= 0)))
                blocks = contiguous.cluster_id.ne(
                    contiguous.cluster_id.shift()
                ).cumsum()
                local = []
                for _, part in contiguous.groupby(blocks, sort=True):
                    row = {
                        "run_id": run_id,
                        "timeline_id": timeline,
                        "block": int(block),
                        "cluster_id": int(part.cluster_id.iloc[0]),
                        "start": int(part.position.min()),
                        "end": int(part.position.max()),
                        "length": len(part),
                        "noise": bool(part.cluster_id.iloc[0] == -1),
                    }
                    local.append(row)
                    runs.append(row)
                previous_by_cluster = {}
                for i, current in enumerate(local):
                    if i and not current["noise"] and not local[i - 1]["noise"]:
                        previous = local[i - 1]
                        transitions.append(
                            {
                                "run_id": run_id,
                                "timeline_id": timeline,
                                "position": current["start"],
                                "previous_position": previous["end"],
                                "from_cluster": previous["cluster_id"],
                                "to_cluster": current["cluster_id"],
                                "after_run_length": current["length"],
                                "persistent": current["length"] >= config.persistence,
                                "isolated_one_frame": current["length"] == 1,
                                "scene_boundary_confirmed": False,
                            }
                        )
                    label = current["cluster_id"]
                    if label >= 0 and label in previous_by_cluster:
                        previous_index, previous = previous_by_cluster[label]
                        returns.append(
                            {
                                "run_id": run_id,
                                "timeline_id": timeline,
                                "cluster_id": label,
                                "previous_end": previous["end"],
                                "return_start": current["start"],
                                "gap_index_units": current["start"] - previous["end"],
                                "intervening_runs": i - previous_index - 1,
                                "contains_noise": any(
                                    r["noise"] for r in local[previous_index + 1 : i]
                                ),
                                "immediate_A_B_A": i - previous_index == 2,
                            }
                        )
                    if label >= 0:
                        previous_by_cluster[label] = (i, current)
        selected = pd.DataFrame(
            [r for r in transitions if r["run_id"] == run_id and r["persistent"]],
            columns=TRANSITION_COLUMNS,
        )
        metric = {
            "run_id": run_id,
            "eligible_nonnoise_cuts": eligible,
            "total_ordered_cuts": all_cuts,
            "nonnoise_cut_coverage": eligible / all_cuts if all_cuts else None,
            "transition_density": len(selected) / eligible if eligible else None,
        }
        if intervals is not None:
            selected = selected.assign(start=selected.position, end=selected.position)
            metric.update(
                compare_zones(selected, intervals, config.tolerance, exhaustive)
            )
        metrics.append(metric)
    return {
        "cluster_runs": pd.DataFrame(runs, columns=RUN_COLUMNS),
        "transitions": pd.DataFrame(transitions, columns=TRANSITION_COLUMNS),
        "cluster_returns": pd.DataFrame(
            returns,
            columns=[
                "run_id",
                "timeline_id",
                "cluster_id",
                "previous_end",
                "return_start",
                "gap_index_units",
                "intervening_runs",
                "contains_noise",
                "immediate_A_B_A",
            ],
        ),
        "transition_metrics": pd.DataFrame(metrics),
    }


def candidate_positions(zones):
    return {
        (r.timeline_id, p)
        for r in zones.itertuples()
        for p in range(int(r.start), int(r.end) + 1)
    }


def interval_stability(boundaries):
    import itertools

    rows = []
    for (a, left), (b, right) in itertools.combinations(sorted(boundaries.items()), 2):
        x, y = candidate_positions(left), candidate_positions(right)
        rows.append(
            {
                "left_variant": a,
                "right_variant": b,
                "left_candidate_positions": len(x),
                "right_candidate_positions": len(y),
                "interval_position_jaccard": len(x & y) / len(x | y) if x | y else None,
            }
        )
    return pd.DataFrame(rows)


def ablation_tables(boundaries, transitions, intervals, config, exhaustive=False):
    """Explicit candidate unions/intersections; a comparison, never a winner."""
    rows, combined = [], []
    empty = pd.DataFrame(
        columns=["timeline_id", "start", "end", "kind", "decision", "element_id"]
    )
    intervals = empty if intervals is None else intervals
    for variant, zones in sorted(boundaries.items()):
        rows.append(
            {
                "variant": variant,
                "signal": "original_l2_temporal",
                "run_id": "",
                **compare_zones(zones, intervals, config.tolerance, exhaustive),
            }
        )
    for run, group in transitions.groupby("run_id", sort=True):
        group = group.loc[group.persistent].assign(
            start=lambda x: x.position, end=lambda x: x.position
        )
        rows.append(
            {
                "variant": run,
                "signal": "clustering_transition",
                "run_id": run,
                **compare_zones(group, intervals, config.tolerance, exhaustive),
            }
        )
        cluster_set = candidate_positions(group)
        for variant, zones in sorted(boundaries.items()):
            temporal_set = candidate_positions(zones)
            for operation, points in (
                ("union", cluster_set | temporal_set),
                ("intersection", cluster_set & temporal_set),
            ):
                # Coalesce adjacent diagnostic candidate positions for review;
                # they are not claimed to be exact reference boundaries.
                items = pd.DataFrame(
                    sorted(points), columns=["timeline_id", "position"]
                )
                merged = []
                for timeline, g in items.groupby("timeline_id", sort=True):
                    for _, block in g.groupby(g.position.diff().gt(1).cumsum()):
                        merged.append(
                            {
                                "timeline_id": timeline,
                                "start": int(block.position.min()),
                                "end": int(block.position.max()),
                            }
                        )
                table = pd.DataFrame(merged, columns=["timeline_id", "start", "end"])
                combined.extend(
                    {"variant": variant, "run_id": run, "operation": operation, **v}
                    for v in merged
                )
                rows.append(
                    {
                        "variant": variant,
                        "run_id": run,
                        "signal": f"combined_{operation}",
                        **compare_zones(table, intervals, config.tolerance, exhaustive),
                    }
                )
    return {
        "boundary_ablation_comparison": pd.DataFrame(rows),
        "combined_candidate_zones": pd.DataFrame(
            combined,
            columns=["variant", "run_id", "operation", "timeline_id", "start", "end"],
        ),
    }
