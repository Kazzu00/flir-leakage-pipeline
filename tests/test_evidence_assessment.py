import pandas as pd
import pytest

from flir_pipeline.linkage.evidence_assessment import (
    assess_base_links,
    check_sequence_membership,
)


@pytest.mark.parametrize("mutation", [None, "interval", "content"])
def test_membership_requires_interval_and_exact_dependency_atomicity(mutation):
    assignments = pd.DataFrame(
        dict(
            frame_id=["f1", "f2"],
            content_id=["a", "b"],
            sequence_id=["s1", "s2"],
            sequence_set_id=["set", "set"],
            video_id=["v", "v"],
            sample_index=[0, 1],
        )
    )
    membership = pd.DataFrame(
        dict(
            sequence_id=["s1", "s2"],
            video_id=["v", "v"],
            start_sample_index=[0, 1],
            end_sample_index=[0, 1],
            visual_dependency_group_id=["g1", "g2"],
            exact_duplicate_dependency_group_id=["d1", "d2"],
        )
    )
    if mutation == "interval":
        assignments.loc[0, "sample_index"] = 2
    if mutation == "content":
        assignments.loc[1, "content_id"] = "a"
    if mutation:
        with pytest.raises(ValueError):
            check_sequence_membership(assignments, membership)
    else:
        result = check_sequence_membership(assignments, membership)
        assert result["internal_consistency"]
        assert not result["producer_provenance_verified"]


def fixtures():
    base = pd.DataFrame(
        dict(content_id=["a", "b", "c", "d"], split=["train", "val", "train", "val"])
    )
    candidates = pd.DataFrame(
        dict(
            labeled_content_id=["a", "b", "c"],
            visual_dependency_group_id=["g", "g", "h"],
        )
    )
    decisions = pd.DataFrame(
        dict(
            labeled_content_id=["a", "b", "test"],
            proposed_visual_dependency_group_id=["g", "g", "g"],
            manual_decision=["supported", "ambiguous", "supported"],
        )
    )
    return base, candidates, decisions


def test_candidates_and_ambiguity_are_never_promoted():
    base, candidates, decisions = fixtures()
    result = assess_base_links(base, candidates, decisions, {"test"}).set_index(
        "content_id"
    )
    assert result.loc["c", "manual_link_status"] == "unreviewed"
    assert result.loc["b", "manual_link_status"] == "ambiguous"
    assert result.loc["d", "shared_candidate_groups"] == 0
    assert not result.training_ready.any()
    assert not result.sequence_assignment_confirmed.any()
    assert result.loc["a", "conditional_protected_dependency"]
    assert not result.loc["b", "conditional_protected_dependency"]


def test_duplicate_decisions_do_not_receive_rank_arbitration():
    base, candidates, decisions = fixtures()
    with pytest.raises(ValueError, match="Duplicate"):
        assess_base_links(
            base, candidates, pd.concat([decisions, decisions.iloc[:1]]), set()
        )


def test_rejected_support_does_not_create_test_link():
    base, candidates, decisions = fixtures()
    decisions.loc[decisions.labeled_content_id == "test", "manual_decision"] = (
        "unsupported"
    )
    assert not assess_base_links(
        base, candidates, decisions, {"test"}
    ).conditional_protected_dependency.any()
