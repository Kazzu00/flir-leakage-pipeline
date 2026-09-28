"""Descriptive aggregation of unchanged, provenance-compatible manual decisions."""

import hashlib

import pandas as pd

from flir_pipeline.linkage.review_model import (
    KEYS,
    SEMANTICS,
    canonical_json,
    decision_counts,
    decision_summary,
)

KIND = "labeled_visual_dependency_manual_calibration_aggregate"
AGGREGATE_SEMANTICS = {
    **SEMANTICS,
    "review_unit": "labeled_content_id + proposed_visual_dependency_group_id",
    "statistics": "pooled descriptive counts/rates across the reviewed calibration set; not representative accuracy or precision estimates",
    "samples_exchangeable": False,
    "duplicate_policy": "identical decisions and compatible evidence counted once per cell; disagreements fail, including blanks",
    "stratum_policy": "preserve every source membership; overlapping stratum cells are not additive",
    "decision_rates_denominator": "unique query/group pairs in each cell, including blanks; source revision cells retain their own sample",
}
FLAGS = (
    "ground_truth",
    "confirmed_matches_created",
    "split_created",
    "automatic_confirmation",
)


def fingerprint(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def group_domain(calibration):
    """Unqualified group IDs may be pooled only within the same frozen domain."""
    sources = calibration["sources"]
    return {
        name: sources[name]
        for name in (
            "sequence_set_id",
            "sequence_files",
            "visual_dependency_files",
            "membership_table",
        )
    }


def evidence_fingerprint(calibration, row):
    """Ignore sample selection/stratum, never ignore the evidence or its sources.

    Different sampling notes, reviewers and manual notes remain in observations.
    Compatibility requires identical underlying sources, protocol/renderer and
    candidate/occurrence evidence. No numerical similarity tolerance is used.
    """
    return fingerprint(
        {
            "calibration_context": {
                **calibration,
                "sources": {
                    key: value
                    for key, value in calibration["sources"].items()
                    if key != "calibration_sample_sha256"
                },
            },
            "candidate_details_json": row["candidate_details_json"],
        }
    )


def aggregate_tables(revisions):
    """Retain all source observations; collapse only explicitly compatible pairs."""
    domains = {
        fingerprint(group_domain(item["metadata"]["identity"]["calibration"]))
        for item in revisions
    }
    if len(domains) != 1:
        raise ValueError(
            "Incompatible visual dependency group domains across revisions"
        )
    frames = []
    for item in revisions:
        meta = item["metadata"]
        frame = item["review"].copy()
        frame["source_revision_id"] = meta["artifact_id"]
        frame["calibration_id"] = meta["calibration_id"]
        frame["evidence_fingerprint"] = [
            evidence_fingerprint(meta["identity"]["calibration"], row)
            for row in frame.to_dict("records")
        ]
        frames.append(frame)
    observations = pd.concat(frames, ignore_index=True).sort_values(
        ["source_revision_id", *KEYS], ignore_index=True
    )
    pooled_rows = []
    for key, rows in observations.groupby(KEYS, sort=True):
        if rows.manual_decision.nunique() != 1:
            details = rows[["source_revision_id", "manual_decision"]].to_dict("records")
            raise ValueError(
                f"Conflicting manual decisions for query/group {key}: {details}"
            )
        if rows.evidence_fingerprint.nunique() != 1:
            raise ValueError(f"Incompatible duplicate evidence for query/group {key}")
        pooled_rows.append(
            {
                **dict(zip(KEYS, key, strict=True)),
                "manual_decision": rows.manual_decision.iloc[0],
                "evidence_fingerprint": rows.evidence_fingerprint.iloc[0],
                "source_observation_count": len(rows),
                "source_revision_ids_json": canonical_json(
                    sorted(rows.source_revision_id)
                ),
                "calibration_ids_json": canonical_json(
                    sorted(set(rows.calibration_id))
                ),
                "review_strata_json": canonical_json(sorted(set(rows.review_stratum))),
            }
        )
    pooled = pd.DataFrame(pooled_rows)
    duplicates = pooled.loc[pooled.source_observation_count.gt(1)].reset_index(
        drop=True
    )

    def grouped(columns):
        results = []
        for key, frame in observations.groupby(columns, sort=True):
            values = key if isinstance(key, tuple) else (key,)
            # A repeated pair can belong to different strata. It occurs once
            # within each original cell; strata therefore need not sum to N.
            result = dict(zip(columns, values, strict=True))
            if KEYS[1] in result:
                result["visual_dependency_group_id"] = result.pop(KEYS[1])
            results.append({**result, **decision_counts(frame.drop_duplicates(KEYS))})
        return results

    summary = {
        "semantics": AGGREGATE_SEMANTICS,
        "source_revision_count": len(revisions),
        "source_calibration_count": int(observations.calibration_id.nunique()),
        "source_observation_count": len(observations),
        "duplicate_pair_count": len(duplicates),
        "duplicate_observations_not_double_counted": len(observations) - len(pooled),
        "overall_pooled_descriptive": decision_counts(pooled),
        "by_source_revision": [
            {
                "source_revision_id": item["metadata"]["artifact_id"],
                "calibration_id": item["metadata"]["calibration_id"],
                **decision_summary(item["review"]),
            }
            for item in revisions
        ],
        "by_stratum": grouped(["review_stratum"]),
        "by_visual_dependency_group": grouped([KEYS[1]]),
        "by_stratum_and_group": grouped(["review_stratum", KEYS[1]]),
    }
    count_rows = []

    def add_count(scope, counts, **labels):
        count_rows.append(
            {
                "scope": scope,
                "source_revision_id": "",
                "calibration_id": "",
                "review_stratum": "",
                "visual_dependency_group_id": "",
                **labels,
                "query_count": counts["query_count"],
                "distinct_group_count": counts["distinct_group_count"],
                "unresolved_count": counts["unresolved_count"],
                **{
                    f"{key}_count": value
                    for key, value in counts["decision_counts"].items()
                },
                **{
                    f"{key}_rate": value
                    for key, value in counts["decision_rates"].items()
                },
            }
        )

    add_count("overall_pooled_descriptive", summary["overall_pooled_descriptive"])
    for row in summary["by_source_revision"]:
        labels = {key: row[key] for key in ("source_revision_id", "calibration_id")}
        add_count("source_revision", row["overall"], **labels)
        for cell in row["by_stratum_and_group"]:
            add_count(
                "source_revision_stratum_and_group",
                cell,
                **labels,
                review_stratum=cell["review_stratum"],
                visual_dependency_group_id=cell["visual_dependency_group_id"],
            )
    for name, labels in (
        ("by_stratum", ["review_stratum"]),
        ("by_visual_dependency_group", ["visual_dependency_group_id"]),
        ("by_stratum_and_group", ["review_stratum", "visual_dependency_group_id"]),
    ):
        for cell in summary[name]:
            add_count(name, cell, **{key: cell[key] for key in labels})
    return {
        "source_observations": observations,
        "pooled_reviews": pooled,
        "duplicate_pairs": duplicates,
        "descriptive_counts": pd.DataFrame(count_rows),
    }, summary
