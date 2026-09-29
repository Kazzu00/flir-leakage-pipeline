"""Directed original-space recurrence, separate encoder scales and diagnostic graphs."""

import itertools

import numpy as np
import pandas as pd

from flir_pipeline.linkage.candidates import cosine_block
from flir_pipeline.similarity.storage import stable_id

PAIR_COLUMNS = [
    "pair_id",
    "left",
    "right",
    "temporally_distinct",
    "exact_shared_contents",
]


def core_contents(source, cores):
    mapping = {}
    for core in cores.itertuples():
        records = source.records.loc[
            source.records.timeline_id.eq(core.timeline_id)
            & source.records.position.between(core.start, core.end)
        ]
        mapping[core.element_id] = sorted(records.content_id.unique())
    return mapping


def directed_matches(a, b, block_rows):
    """Exact NN in bounded row blocks, deterministic lower-content-ID tie break."""
    scores = np.empty(len(a), dtype=np.float64)
    indices = np.empty(len(a), dtype=np.int64)
    for start in range(0, len(a), block_rows):
        matrix = np.clip(
            cosine_block(a[start : start + block_rows], b),
            -1,
            1,
        )
        chosen = np.argmax(matrix, axis=1)
        indices[start : start + len(matrix)] = chosen
        scores[start : start + len(matrix)] = matrix[np.arange(len(matrix)), chosen]
    return scores, indices


def diagnostic_components(pairs, core_ids, selection="refined_candidate"):
    parent = {c: c for c in sorted(core_ids)}

    def root(c):
        while parent[c] != c:
            c = parent[c]
        return c

    for row in pairs.loc[pairs[selection]].itertuples():
        a, b = sorted((root(row.left), root(row.right)))
        parent[b] = a
    groups = {}
    for c in sorted(parent):
        groups.setdefault(root(c), []).append(c)
    return pd.DataFrame(
        [
            {
                "core_id": c,
                "diagnostic_component_id": stable_id({"candidate_cores": group}),
                "diagnostic_only": True,
                "dependency_confirmed": False,
            }
            for group in groups.values()
            for c in group
        ],
        columns=[
            "core_id",
            "diagnostic_component_id",
            "diagnostic_only",
            "dependency_confirmed",
        ],
    )


def recurrence_tables(source, cores, config):
    membership = core_contents(source, cores)
    lookup = {c: i for i, c in enumerate(source.contents)}
    elements = cores.set_index("element_id")
    pairs, scores, distributions, matches, exact_edges = [], [], [], [], []
    for left, right in itertools.combinations(sorted(membership), 2):
        a, b = elements.loc[left], elements.loc[right]
        distinct = a.timeline_id != b.timeline_id or a.end < b.start or b.end < a.start
        if not distinct or not membership[left] or not membership[right]:
            continue
        pair_id = stable_id({"left": left, "right": right})
        shared = sorted(set(membership[left]) & set(membership[right]))
        exact_edges.extend(
            {
                "edge_id": stable_id({"pair_id": pair_id, "content_id": content}),
                "pair_id": pair_id,
                "left": left,
                "right": right,
                "content_id": content,
                "evidence": "shared_exact_content_identity",
                "sequence_instances_merged": False,
                "visual_dependency_group_created": False,
            }
            for content in shared
        )
        pairs.append(
            {
                "pair_id": pair_id,
                "left": left,
                "right": right,
                "temporally_distinct": True,
                "exact_shared_contents": len(shared),
            }
        )
        for encoder in ("clip", "dinov2"):
            x = source.embeddings[encoder][[lookup[c] for c in membership[left]]]
            y = source.embeddings[encoder][[lookup[c] for c in membership[right]]]
            ab, abi = directed_matches(x, y, config.block_rows)
            ba, bai = directed_matches(y, x, config.block_rows)
            cx, cy = x.mean(axis=0, dtype=np.float64), y.mean(axis=0, dtype=np.float64)
            norm = np.linalg.norm(cx) * np.linalg.norm(cy)
            threshold = config.selection_threshold[encoder]
            support = min(
                float(np.mean(ab >= threshold)), float(np.mean(ba >= threshold))
            )
            mutual = np.arange(len(x)) == bai[abi]
            scores.append(
                {
                    "pair_id": pair_id,
                    "encoder": encoder,
                    "centroid_cosine": float(np.clip(cx @ cy / norm, -1, 1))
                    if norm > 0
                    else None,
                    "a_to_b_median": float(np.median(ab)),
                    "a_to_b_p95": float(np.quantile(ab, 0.95)),
                    "b_to_a_median": float(np.median(ba)),
                    "b_to_a_p95": float(np.quantile(ba, 0.95)),
                    "symmetric_median": float(min(np.median(ab), np.median(ba))),
                    "selection_cosine_threshold": threshold,
                    "symmetric_support": support,
                    "mutual_nn_count": int(mutual.sum()),
                    "mutual_support": min(
                        float(mutual.sum() / len(x)), float(mutual.sum() / len(y))
                    ),
                    "broad_candidate": support >= config.broad_support,
                    "encoder_candidate": support >= config.refined_support,
                }
            )
            for direction, values, indices, first, second in (
                ("a_to_b", ab, abi, left, right),
                ("b_to_a", ba, bai, right, left),
            ):
                for t in config.thresholds[encoder]:
                    distributions.append(
                        {
                            "pair_id": pair_id,
                            "encoder": encoder,
                            "direction": direction,
                            "cosine_threshold": t,
                            "fraction_ge": float(np.mean(values >= t)),
                            "query_count": len(values),
                        }
                    )
                matches.extend(
                    {
                        "pair_id": pair_id,
                        "encoder": encoder,
                        "direction": direction,
                        "query_content_id": content,
                        "neighbor_content_id": membership[second][int(j)],
                        "cosine": float(value),
                        "exact_content": content == membership[second][int(j)],
                    }
                    for content, j, value in zip(
                        membership[first], indices, values, strict=True
                    )
                )
    pair_table = pd.DataFrame(pairs, columns=PAIR_COLUMNS)
    score_table = pd.DataFrame(
        scores,
        columns=[
            "pair_id",
            "encoder",
            "centroid_cosine",
            "a_to_b_median",
            "a_to_b_p95",
            "b_to_a_median",
            "b_to_a_p95",
            "symmetric_median",
            "selection_cosine_threshold",
            "symmetric_support",
            "mutual_nn_count",
            "mutual_support",
            "broad_candidate",
            "encoder_candidate",
        ],
    )
    for encoder in ("clip", "dinov2"):
        part = score_table.loc[score_table.encoder.eq(encoder)].copy()
        # Rank the medians within EACH encoder. Cosines are never averaged.
        part["encoder_rank"] = part.symmetric_median.round(12).rank(
            ascending=False, method="average"
        )
        score_table.loc[part.index, "encoder_rank"] = part.encoder_rank
        for column in (
            "encoder_rank",
            "symmetric_support",
            "broad_candidate",
            "encoder_candidate",
        ):
            pair_table[f"{encoder}_{column}"] = pair_table.pair_id.map(
                part.set_index("pair_id")[column]
            )
    pair_table["rank_consensus"] = np.maximum(
        pair_table.clip_encoder_rank, pair_table.dinov2_encoder_rank
    )
    pair_table["encoder_disagreement"] = (
        pair_table.clip_encoder_candidate != pair_table.dinov2_encoder_candidate
    )
    pair_table["broad_candidate"] = pair_table.clip_broad_candidate.astype(
        bool
    ) | pair_table.dinov2_broad_candidate.astype(bool)
    pair_table["refined_candidate"] = (
        pair_table.clip_encoder_candidate.astype(bool)
        & pair_table.dinov2_encoder_candidate.astype(bool)
        & pair_table.rank_consensus.le(max(1, len(pair_table) * config.rank_fraction))
    )
    pair_table["dependency_confirmed"] = False
    pair_table["exact_copy_dependency_observed"] = pair_table.exact_shared_contents.gt(
        0
    )
    rates = []
    for policy in (
        "broad_candidate",
        "refined_candidate",
        "clip_encoder_candidate",
        "dinov2_encoder_candidate",
    ):
        rate = float(pair_table[policy].mean()) if len(pair_table) else None
        rates.append(
            {
                "policy": policy,
                "possible_pairs": len(pair_table),
                "selected_pairs": int(pair_table[policy].sum()),
                "candidate_rate": rate,
                "non_discriminative_warning": rate is not None
                and rate > config.warning_candidate_rate,
            }
        )
    return {
        "pairs": pair_table,
        "exact_copy_edges": pd.DataFrame(
            exact_edges,
            columns=[
                "edge_id",
                "pair_id",
                "left",
                "right",
                "content_id",
                "evidence",
                "sequence_instances_merged",
                "visual_dependency_group_created",
            ],
        ),
        "encoder_scores": score_table,
        "threshold_support": pd.DataFrame(
            distributions,
            columns=[
                "pair_id",
                "encoder",
                "direction",
                "cosine_threshold",
                "fraction_ge",
                "query_count",
            ],
        ),
        "nearest_matches": pd.DataFrame(
            matches,
            columns=[
                "pair_id",
                "encoder",
                "direction",
                "query_content_id",
                "neighbor_content_id",
                "cosine",
                "exact_content",
            ],
        ),
        "candidate_diagnostics": pd.DataFrame(rates),
        "diagnostic_components": diagnostic_components(pair_table, membership),
        "core_content_membership": pd.DataFrame(
            [
                {"core_id": k, "content_id": c}
                for k, ids in membership.items()
                for c in ids
            ],
            columns=["core_id", "content_id"],
        ),
    }


def cluster_recurrence(assignments, cores, source, direct, config):
    """A single noisy/shared frame cannot masquerade as distributed evidence."""
    from scipy.stats import spearmanr

    members = core_contents(source, cores)
    rows, comparisons, correlations = [], [], []
    for run, group in assignments.groupby("run_id", sort=True):
        labels = group.set_index("content_id").cluster_id
        for pair in direct.itertuples():
            a, b = labels.loc[members[pair.left]], labels.loc[members[pair.right]]
            shared = sorted((set(a) & set(b)) - {-1})
            count_a, count_b = int(a.isin(shared).sum()), int(b.isin(shared).sum())
            support_a, support_b = count_a / len(a), count_b / len(b)
            distributed = (
                min(count_a, count_b) >= config.cluster_min_members
                and min(support_a, support_b) >= config.cluster_support
            )
            rows.append(
                {
                    "run_id": run,
                    "pair_id": pair.pair_id,
                    "left": pair.left,
                    "right": pair.right,
                    "shared_clusters": len(shared),
                    "left_members": count_a,
                    "right_members": count_b,
                    "left_support": support_a,
                    "right_support": support_b,
                    "symmetric_support": min(support_a, support_b),
                    "distributed_candidate": distributed,
                    "incidental_only": bool(shared) and not distributed,
                    "noise_shared_is_evidence": False,
                }
            )
            comparisons.append(
                {
                    "run_id": run,
                    "pair_id": pair.pair_id,
                    "cluster_candidate": distributed,
                    "direct_candidate": bool(pair.refined_candidate),
                    "agreement": distributed == bool(pair.refined_candidate),
                    "direct_rank": pair.rank_consensus,
                    "cluster_support": min(support_a, support_b),
                }
            )
        values = pd.DataFrame([r for r in comparisons if r["run_id"] == run])
        defined = (
            len(values) >= 3
            and values.cluster_support.nunique() > 1
            and values.direct_rank.nunique() > 1
        )
        correlations.append(
            {
                "run_id": run,
                "pair_count": len(values),
                "spearman_defined": defined,
                "spearman_support_vs_direct_rank": float(
                    spearmanr(values.cluster_support, -values.direct_rank).statistic
                )
                if defined
                else None,
            }
        )
    evidence = pd.DataFrame(
        rows,
        columns=[
            "run_id",
            "pair_id",
            "left",
            "right",
            "shared_clusters",
            "left_members",
            "right_members",
            "left_support",
            "right_support",
            "symmetric_support",
            "distributed_candidate",
            "incidental_only",
            "noise_shared_is_evidence",
        ],
    )
    comparison = pd.DataFrame(
        comparisons,
        columns=[
            "run_id",
            "pair_id",
            "cluster_candidate",
            "direct_candidate",
            "agreement",
            "direct_rank",
            "cluster_support",
        ],
    )
    return {
        "cluster_recurrence": evidence,
        "agreement": comparison.loc[comparison.agreement.astype(bool)].reset_index(
            drop=True
        ),
        "disagreement": comparison.loc[~comparison.agreement.astype(bool)].reset_index(
            drop=True
        ),
        "rank_correlations": pd.DataFrame(correlations),
    }
