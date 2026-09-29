"""Observed legacy report contracts, not inferred from directory names.

Counts are revision acceptance checks supplied by the report owner. They are
never parameters or expected results of a new scientific experiment.
"""

ADAPTER = "hypatia_legacy_evidence_v1"
ROLES = {
    "provenance_decision_v1": "provenance",
    "video11_manual_transition_review_v2": "manual",
    "video11_sequence_structure_v1": "structure",
    "video11_core_recurrence_refined_v1": "recurrence",
}
FALSE_FLAGS = {
    "ground_truth",
    "split_created",
    "confirmed_frame_matches_created",
    "automatic_confirmation",
    "sequence_boundaries_created",
    "sequence_boundary_created",
    "exact_sequence_boundaries_created",
    "sequence_instances_created",
    "visual_dependency_groups_created",
    "exact_boundary_known",
    "candidate_components_are_vdgs",
    "visual_verification_complete",
}
REQUIRED_FLAGS = {
    "provenance": "ground_truth split_created confirmed_frame_matches_created automatic_confirmation",
    "manual": "ground_truth automatic_confirmation exact_sequence_boundaries_created sequence_instances_created visual_dependency_groups_created split_created",
    "structure": "ground_truth exact_boundary_known sequence_instances_created visual_dependency_groups_created split_created",
    "recurrence": "candidate_components_are_vdgs ground_truth visual_dependency_groups_created split_created",
}
PAIR_COLUMNS = """core_a core_b clip_centroid_cosine clip_symmetric_ge_090
clip_symmetric_ge_095 clip_recurrence_score dinov2_centroid_cosine
dinov2_symmetric_ge_090 dinov2_symmetric_ge_095 dinov2_recurrence_score
consensus_score encoder_agreement_score clip_rank dinov2_rank rank_sum
mean_reciprocal_rank joint_symmetric_ge_090 joint_centroid_min""".split()
CSV_COLUMNS = {
    "manual": {
        "supported_boundary_zones.csv": "region_id start_index end_index".split(),
        "transition_regions.csv": """region_id start_index end_index transition_ids
edge_decisions region_status ground_truth sequence_boundary_created
region_semantics exact_boundary_known""".split(),
        "transition_decisions.csv": """review_order transition_id left_index right_index
clip_adjacent_cosine dinov2_adjacent_cosine mean_adjacent_cosine contact_sheet
decision notes clip_adjacent_cosine_context dinov2_adjacent_cosine_context
clip_local_baseline clip_boundary_drop clip_cross_window_centroid_cosine
dinov2_local_baseline dinov2_boundary_drop dinov2_cross_window_centroid_cosine
mean_boundary_drop mean_cross_window_centroid_cosine manual_decision review_notes
review_mode ground_truth automatic_confirmation sequence_boundary_created split_created""".split(),
    },
    "structure": {
        "boundary_zones.csv": "region_id structure_id start_index end_index structure_type".split(),
        "index_structure.csv": "nominal_index structure_id structure_type start_index end_index review_region_id".split(),
        "sequence_core_candidates.csv": "start_index end_index structure_id structure_type".split(),
        "structure_summary.csv": """structure_id structure_type start_index end_index
unique_index_count occurrence_count unique_content_count historical_split_counts
cross_split_exact_content_count""".split(),
    },
    "recurrence": {
        "all_pairs_ranked.csv": PAIR_COLUMNS,
        "manual_review_candidates.csv": [*PAIR_COLUMNS, "review_order"],
    },
}
OCCURRENCE_SCHEMA = dict(
    item.split(":")
    for item in """
frame_id:string content_id:string duplicate_group_id:string source_archive:string
source_member_path:string relative_image_path:string relative_label_path:string
image_filename:string image_basename:string label_filename:string original_split:string
image_sha256:string label_sha256:string label_exists:bool manifest_version:string
image_decode_valid:bool possible_sequence:string possible_frame_index:int64
temporal_inference_confidence:string width:int64 height:int64 channels:int64
image_mode:string image_format:string label_empty:bool label_valid:bool syntax_valid:bool
normalized_values_valid:bool geometry_valid:bool num_objects:int64 classes_present:string
bbox_area_mean:float64 bbox_area_min:float64 bbox_area_max:float64 bbox_width_mean:float64
bbox_height_mean:float64 exact_duplicate:bool cross_split_exact_duplicate:bool
duplicate_occurrence_count:int64 duplicate_splits:string nominal_index:int64
structure_id:string structure_type:string start_index:int64 end_index:int64 review_region_id:string
""".split()
)
NULLABLE = {name for name in OCCURRENCE_SCHEMA if name.startswith("bbox_")} | {
    "review_region_id"
}
