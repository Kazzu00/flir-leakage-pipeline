"""Assess transferred review tables without promoting them to split authority."""


def assess_base_links(base, candidates, decisions, protected_contents):
    """Keep reported decisions, missing coverage and conditional test conflicts distinct.

    CSV decisions alone do not establish source-bound review provenance. Multiple
    reviewed groups for a query need explicit reconciliation, never rank selection.
    """
    if not base.content_id.is_unique:
        raise ValueError("BASE must contain unique content")
    keys = ["labeled_content_id", "proposed_visual_dependency_group_id"]
    if decisions.duplicated(keys).any():
        raise ValueError("Duplicate or conflicting decision pairs")
    if decisions.labeled_content_id.duplicated().any():
        raise ValueError("Multiple reviewed groups require explicit reconciliation")
    if not set(decisions.manual_decision) <= {"supported", "ambiguous", "unsupported"}:
        raise ValueError("Unsupported manual state")
    supported = decisions.loc[decisions.manual_decision == "supported"]
    protected_groups = set(
        supported.loc[
            supported.labeled_content_id.isin(protected_contents),
            "proposed_visual_dependency_group_id",
        ]
    )
    counts = candidates.groupby(
        "labeled_content_id"
    ).visual_dependency_group_id.nunique()
    joined = base.merge(
        decisions[keys + ["manual_decision"]],
        left_on="content_id",
        right_on="labeled_content_id",
        how="left",
        validate="one_to_one",
    )
    joined["manual_link_status"] = joined.manual_decision.fillna("unreviewed")
    joined["shared_candidate_groups"] = (
        joined.content_id.map(counts).fillna(0).astype(int)
    )
    joined["conditional_protected_dependency"] = (
        joined.manual_link_status == "supported"
    ) & joined.proposed_visual_dependency_group_id.isin(protected_groups)
    joined["sequence_assignment_confirmed"] = False
    joined["training_ready"] = False
    return joined.drop(columns=["labeled_content_id", "manual_decision"])


def check_sequence_membership(assignments, membership):
    """Verify internal coverage/atomicity only; producer signatures remain separate."""
    if not membership.sequence_id.is_unique or not assignments.frame_id.is_unique:
        raise ValueError("Duplicate sequence or occurrence")
    if set(assignments.sequence_id) != set(membership.sequence_id):
        raise ValueError("Sequence coverage mismatch")
    if assignments.sequence_set_id.nunique() != 1:
        raise ValueError("Mixed sequence sets")
    joined = assignments.merge(
        membership,
        on="sequence_id",
        validate="many_to_one",
        suffixes=("_source", "_group"),
    )
    if (
        (joined.sample_index < joined.start_sample_index)
        | (joined.sample_index > joined.end_sample_index)
    ).any():
        raise ValueError("Occurrence outside sequence interval")
    if (joined.video_id_source != joined.video_id_group).any():
        raise ValueError("Video identity mismatch")
    if (joined.groupby("content_id").visual_dependency_group_id.nunique() > 1).any():
        raise ValueError("Exact content dependency split across visual groups")
    if (
        membership.groupby(
            "exact_duplicate_dependency_group_id"
        ).visual_dependency_group_id.nunique()
        > 1
    ).any():
        raise ValueError("Exact sequence dependency split")
    return dict(
        occurrences=len(assignments),
        sequences=len(membership),
        visual_groups=membership.visual_dependency_group_id.nunique(),
        internal_consistency=True,
        producer_provenance_verified=False,
    )
