"""Post-hoc labels with explicit denominators, noise policies and masks."""

import itertools
import json

import numpy as np
import pandas as pd

from flir_pipeline.clustering.metrics import assignment_agreement
from flir_pipeline.linkage.candidates import cosine_block
from flir_pipeline.sequences.experiments.structure import content_targets


def validate_assignments(assignments, contents):
    if list(assignments) != ["run_id", "content_id", "cluster_id"]:
        raise ValueError("Unexpected exact assignment schema")
    for _, group in assignments.groupby("run_id"):
        if (
            not group.content_id.is_unique
            or set(group.content_id) != set(contents)
            or not pd.api.types.is_integer_dtype(group.cluster_id.dtype)
            or group.cluster_id.lt(-1).any()
        ):
            raise ValueError(
                "Assignments must cover each unique content exactly once; noise=-1"
            )


def label_metrics(labels, target):
    from sklearn.metrics import (
        adjusted_mutual_info_score,
        adjusted_rand_score,
        homogeneity_completeness_v_measure,
    )

    rows = []
    for policy, mask in (
        ("all_evaluated", target.evaluation_mask.to_numpy(dtype=bool)),
        (
            "clustered_evaluated",
            target.evaluation_mask.to_numpy(dtype=bool) & (labels >= 0),
        ),
    ):
        a, b = target.target_label.to_numpy()[mask], labels[mask]
        n = int(mask.sum())
        metrics = dict.fromkeys(
            ["ari", "ami", "homogeneity", "completeness", "v_measure"]
        )
        if n >= 2:
            h, c, v = homogeneity_completeness_v_measure(a, b)
            metrics = {
                "ari": float(adjusted_rand_score(a, b)),
                "ami": float(
                    adjusted_mutual_info_score(a, b, average_method="arithmetic")
                ),
                "homogeneity": float(h),
                "completeness": float(c),
                "v_measure": float(v),
            }
        rows.append(
            {
                "noise_policy": policy,
                "evaluated_n": n,
                "evaluated_content_coverage": n / len(labels),
                "total_contents": len(labels),
                "evaluated_occurrences": int(
                    target.loc[mask, "occurrence_count"].sum()
                ),
                "trivial_partition": len(set(a)) < 2 or len(set(b)) < 2,
                **metrics,
            }
        )
    return rows


def original_neighbors(values, k=20, block_rows=256):
    n = len(values)
    result = np.empty((n, min(k, max(0, n - 1))), dtype=np.int64)
    for start in range(0, n, block_rows):
        stop = min(start + block_rows, n)
        scores = cosine_block(values[start:stop], values)
        scores[np.arange(stop - start), np.arange(start, stop)] = -np.inf
        result[start:stop] = np.argsort(-scores, axis=1, kind="stable")[
            :, : result.shape[1]
        ]
    return result


def temporal_pairs(membership, contents, target, delta):
    rows = membership.loc[
        membership.target.eq(target) & membership.evaluation_mask
    ].copy()
    valid = content_targets(membership, contents, target)
    rows = rows.loc[
        rows.content_id.isin(valid.loc[valid.evaluation_mask, "content_id"])
    ]
    lookup = {c: i for i, c in enumerate(contents)}
    pairs = set()
    for _, group in rows.groupby(["timeline_id", "target_label"], sort=True):
        group = group.drop_duplicates(["position", "content_id"]).sort_values(
            ["position", "content_id"]
        )
        values = list(group.itertuples())
        for i, a in enumerate(values):
            for b in values[i + 1 :]:
                if b.position - a.position > delta:
                    break
                if a.content_id != b.content_id and b.position > a.position:
                    pairs.add(
                        tuple(sorted((lookup[a.content_id], lookup[b.content_id])))
                    )
    return sorted(pairs)


def evaluation_tables(source, assignments, membership):
    validate_assignments(assignments, source.contents)
    metrics, masks, cluster_rows, fragmentation, diagnostics = [], [], [], [], []
    targets = sorted(membership.target.unique())
    target_tables = {
        target: content_targets(membership, source.contents, target)
        for target in targets
    }
    temporal = {
        (target, d): temporal_pairs(membership, source.contents, target, d)
        for target in targets
        for d in (1, 5, 10)
    }
    neighbors = {
        encoder: original_neighbors(values)
        for encoder, values in source.embeddings.items()
    }
    for target in targets:
        masks.extend(target_tables[target].to_dict("records"))
    for run, group in assignments.groupby("run_id", sort=True):
        labels = (
            group.set_index("content_id")
            .loc[source.contents, "cluster_id"]
            .to_numpy(dtype=int)
        )
        common = {
            "run_id": run,
            "total_contents": len(labels),
            "noise_coverage": float(np.mean(labels == -1)),
            "cluster_count": len(set(labels) - {-1}),
            "clustered_content_coverage": float(np.mean(labels >= 0)),
        }
        for encoder, nn in neighbors.items():
            for k in (5, 10, 20):
                supported = nn.shape[1] >= k
                matched = (
                    int(
                        (
                            (labels[:, None] == labels[nn[:, :k]])
                            & (labels[:, None] >= 0)
                        ).sum()
                    )
                    if supported
                    else 0
                )
                denominator = int((labels >= 0).sum()) * k if supported else 0
                diagnostics.append(
                    {
                        **common,
                        "metric": f"visual_neighbor_coherence@{k}",
                        "encoder": encoder,
                        "target": "original_l2",
                        "numerator": matched if supported else None,
                        "denominator": denominator,
                        "defined": supported and denominator > 0,
                        "value": matched / denominator
                        if supported and denominator
                        else None,
                    }
                )
        for target, table in target_tables.items():
            metrics.extend(
                {**common, "target": target, **m} for m in label_metrics(labels, table)
            )
            evaluated = table.loc[table.evaluation_mask].copy()
            evaluated["cluster_id"] = labels[table.evaluation_mask]
            for cluster, items in evaluated.loc[evaluated.cluster_id.ge(0)].groupby(
                "cluster_id"
            ):
                counts = items.target_label.value_counts()
                fractions = counts.to_numpy() / len(items)
                cluster_rows.append(
                    {
                        "run_id": run,
                        "target": target,
                        "cluster_id": int(cluster),
                        "evaluated_n": len(items),
                        "dominant_sequence_fraction": float(fractions.max()),
                        "sequence_entropy_bits": float(
                            -np.sum(fractions * np.log2(fractions))
                        ),
                        "merging_sequence_count": len(counts),
                    }
                )
            for label, items in evaluated.groupby("target_label", sort=True):
                fragmentation.append(
                    {
                        "run_id": run,
                        "target": target,
                        "target_label": label,
                        "evaluated_n": len(items),
                        "clusters_occupied": int(
                            items.loc[items.cluster_id.ge(0), "cluster_id"].nunique()
                        ),
                        "noise_contents": int(items.cluster_id.eq(-1).sum()),
                    }
                )
            relevant = [
                r for r in cluster_rows if r["run_id"] == run and r["target"] == target
            ]
            denominator = sum(r["evaluated_n"] for r in relevant)
            purity = sum(
                r["dominant_sequence_fraction"] * r["evaluated_n"] for r in relevant
            )
            diagnostics.append(
                {
                    **common,
                    "target": target,
                    "encoder": "",
                    "metric": "cluster_temporal_purity",
                    "numerator": purity,
                    "denominator": denominator,
                    "defined": bool(denominator),
                    "value": purity / denominator if denominator else None,
                }
            )
            for delta in (1, 5, 10):
                pairs = temporal[target, delta]
                retained = sum(
                    labels[a] >= 0 and labels[a] == labels[b] for a, b in pairs
                )
                diagnostics.append(
                    {
                        **common,
                        "target": target,
                        "encoder": "",
                        "metric": f"temporal_recall@{delta}",
                        "numerator": int(retained),
                        "denominator": len(pairs),
                        "defined": bool(pairs),
                        "value": retained / len(pairs) if pairs else None,
                    }
                )
    return {
        "metrics": pd.DataFrame(metrics),
        "evaluation_mask": pd.DataFrame(masks),
        "cluster_summary": pd.DataFrame(
            cluster_rows,
            columns=[
                "run_id",
                "target",
                "cluster_id",
                "evaluated_n",
                "dominant_sequence_fraction",
                "sequence_entropy_bits",
                "merging_sequence_count",
            ],
        ),
        "fragmentation": pd.DataFrame(
            fragmentation,
            columns=[
                "run_id",
                "target",
                "target_label",
                "evaluated_n",
                "clusters_occupied",
                "noise_contents",
            ],
        ),
        "diagnostics": pd.DataFrame(diagnostics),
    }


def stability_tables(runs, assignments, transitions=None, recurrence=None):
    labels = {
        r: g.sort_values("content_id").cluster_id.to_numpy(dtype=int)
        for r, g in assignments.groupby("run_id")
    }
    rows = []
    for a, b in itertools.combinations(runs.itertuples(), 2):
        a_reduction, b_reduction = (
            json.loads(a.reduction_json),
            json.loads(b.reduction_json),
        )
        if a_reduction:
            a_reduction.pop("seed")
        if b_reduction:
            b_reduction.pop("seed")
        params_a, params_b = (
            json.loads(a.parameters_json),
            json.loads(b.parameters_json),
        )
        same_method = (
            a.encoder == b.encoder
            and a.algorithm == b.algorithm
            and a.representation == b.representation
        )
        seed_test = (
            same_method
            and a_reduction == b_reduction
            and params_a == params_b
            and a.seed != b.seed
        )
        changed = sorted(
            k
            for k in set(params_a) | set(params_b)
            if params_a.get(k) != params_b.get(k)
        )
        parameter_test = same_method and a.space_id == b.space_id and len(changed) == 1
        representation_test = (
            a.encoder == b.encoder
            and a.algorithm == b.algorithm
            and params_a == params_b
            and a.representation != b.representation
        )
        if not (seed_test or parameter_test or representation_test):
            continue
        x, y = labels[a.run_id], labels[b.run_id]
        noise_union = (x == -1) | (y == -1)
        result = {
            "left_run": a.run_id,
            "right_run": b.run_id,
            "comparison": "reduction_seed"
            if seed_test
            else ("parameter_perturbation" if parameter_test else "representation"),
            "changed_parameters": json.dumps(changed),
            **assignment_agreement(x, y),
            "noise_membership_agreement": float(np.mean((x == -1) == (y == -1))),
            "noise_jaccard": float(np.sum((x == -1) & (y == -1)) / noise_union.sum())
            if noise_union.any()
            else None,
        }
        for name, table, fields, flag in (
            (
                "boundary_candidate",
                transitions,
                ["timeline_id", "position"],
                "persistent",
            ),
            ("recurrence_pair", recurrence, ["pair_id"], "distributed_candidate"),
        ):
            if table is None or table.empty:
                left, right = set(), set()
            else:
                left = set(
                    table.loc[
                        table.run_id.eq(a.run_id) & table[flag], fields
                    ].itertuples(index=False, name=None)
                )
                right = set(
                    table.loc[
                        table.run_id.eq(b.run_id) & table[flag], fields
                    ].itertuples(index=False, name=None)
                )
            result.update(
                {
                    f"{name}_left_n": len(left),
                    f"{name}_right_n": len(right),
                    f"{name}_jaccard": len(left & right) / len(left | right)
                    if left | right
                    else None,
                }
            )
        rows.append(result)
    return pd.DataFrame(rows)
