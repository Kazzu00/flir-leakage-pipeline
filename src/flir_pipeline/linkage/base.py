"""Scientific linkage policy, independent of labels, splits and sequence choices."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class LinkageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    top_k: int = Field(default=10, gt=0, strict=True)
    algorithm: Literal["cross_dataset_topk_union_v1"] = "cross_dataset_topk_union_v1"
    representation: Literal["original_float32_l2"] = "original_float32_l2"
    cosine_policy: Literal["float64_dot_no_renormalization_no_clipping"] = (
        "float64_dot_no_renormalization_no_clipping"
    )
    tie_policy: Literal["cosine_desc_video_content_id_asc"] = (
        "cosine_desc_video_content_id_asc"
    )
    small_population_policy: Literal["min_top_k_video_content_count"] = (
        "min_top_k_video_content_count"
    )
    consensus_policy: Literal["mean_reciprocal_rank_missing_zero"] = (
        "mean_reciprocal_rank_missing_zero"
    )
    occurrence_policy: Literal["all_occurrences_normalized_by_video_content_id"] = (
        "all_occurrences_normalized_by_video_content_id"
    )


ARTIFACT_KIND = "labeled_video_link_candidates"
SEMANTICS = {
    "status": "candidates_only",
    "ground_truth": False,
    "links_auto_confirmed": False,
    "split_created": False,
    "visual_dependency_group_created": False,
    "sequence_assignment_created": False,
    "sequence_id": "existing reviewed continuous video segment; not a cluster",
    "candidate_occurrences": "join content_candidates.video_content_id to ALL candidate_occurrences.video_content_id rows",
    "labeled_occurrences": "join labeled_content_id to ALL labeled_occurrences.content_id rows; original_split is provenance only",
    "mean_reciprocal_rank": "(1/clip_rank + 1/dinov2_rank)/2, missing rank contributes zero; not a probability",
    "encoder_cosines": "separate original-space scores; never averaged across encoders",
    "sequence_verification": "source-bound stored partition and review snapshot; external review CSVs are not reread",
}
