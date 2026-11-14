"""Read-only reconciliation of recovered linkage evidence; never produces a split."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from flir_pipeline.linkage.evidence_assessment import (
    assess_base_links,
    check_sequence_membership,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("evidence", "base", "fixed-publication", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--linkage-id", required=True, help="Stored linkage artifact directory name")
    args = parser.parse_args()
    if Path(args.linkage_id).name != args.linkage_id or args.linkage_id in {".", ".."}:
        raise ValueError("Use a single linkage artifact directory name")
    if args.output.exists():
        raise ValueError("Use a new output directory to preserve previous evidence")
    hashes = {}

    def digest(path):
        value = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes[str(path.resolve())] = value
        return value

    def read(path):
        digest(path)
        if path.suffix == ".json":
            return json.loads(path.read_text(encoding="utf-8"))
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path)

    links = args.evidence / args.linkage_id
    metadata = read(links / "metadata.json")
    for name, expected in metadata["output_checksums"].items():
        if Path(name).name != name or digest(links / name) != expected:
            raise ValueError(f"Linkage checksum mismatch: {name}")
    assignments = read(args.evidence / "occurrence_assignments.parquet")
    expected = metadata["identity"]["sources"]["checksums"]["sequence_set"]
    if (
        digest(args.evidence / "occurrence_assignments.parquet")
        != expected["occurrence_assignments.parquet"]
    ):
        raise ValueError("Sequence assignments not bound to linkage")
    membership = read(args.evidence / "visual_dependency_membership.csv")
    sequence = check_sequence_membership(assignments, membership)
    occurrences = read(links / "candidate_occurrences.parquet")
    source = assignments.set_index("frame_id")
    for column in (
        "sequence_id",
        "video_id",
        "sample_index",
        "timestamp_seconds",
        "source_video_sha256",
    ):
        actual = occurrences.video_frame_id.map(source[column])
        if not actual.eq(occurrences[column]).all():
            raise ValueError(f"Candidate occurrence mismatch: {column}")
    raw = read(links / "content_candidates.parquet")
    mapped = occurrences.merge(
        membership[["sequence_id", "visual_dependency_group_id"]],
        on="sequence_id",
        validate="many_to_one",
    )
    expanded = raw.merge(
        mapped[["video_content_id", "visual_dependency_group_id"]].drop_duplicates(),
        on="video_content_id",
        validate="many_to_many",
    )
    keys = ["labeled_content_id", "visual_dependency_group_id"]
    ranks = (
        expanded.groupby(keys)[["clip_rank", "dinov2_rank"]]
        .min()
        .dropna()
        .rename(
            columns={"clip_rank": "clip_best_rank", "dinov2_rank": "dinov2_best_rank"}
        )
    )
    supplied = read(args.evidence / "shared_group_rank_support.csv")
    pd.testing.assert_frame_equal(
        ranks.sort_index().astype(float),
        supplied.set_index(keys)[list(ranks.columns)].sort_index().astype(float),
        check_index_type=False,
    )
    decisions = pd.concat(
        [read(p) for p in sorted(args.evidence.glob("*_manual_decisions_v1.csv"))],
        ignore_index=True,
    )
    base = read(args.base / "manifest.csv")
    base_receipt = read(args.base / "receipt.json")
    for name in (
        "manifest.csv",
        "experiment.json",
        "data.yaml",
        "environment-protocol.yaml",
    ):
        if digest(args.base / name) != base_receipt["artifacts"][name]:
            raise ValueError(f"BASE checksum mismatch: {name}")
    receipt = read(args.fixed_publication / "receipt.json")
    fixed = {}
    for name in ("protected_records", "test_records"):
        relative = f"sources/proposal/{name}.parquet"
        path = args.fixed_publication / relative
        if digest(path) != receipt["artifacts"][relative]:
            raise ValueError(f"Fixed publication checksum mismatch: {name}")
        fixed[name] = read(path)
    if (
        digest(args.fixed_publication / "sources/proposal/protected_records.parquet")
        != base_receipt["protected_exclusion"]["reference_sha256"]
    ):
        raise ValueError(
            "BASE and current publication use different protected references"
        )
    if set(base.content_id) & set(fixed["protected_records"].content_id):
        raise ValueError("BASE intersects protected content")
    result = assess_base_links(
        base, supplied, decisions, set(fixed["protected_records"].content_id)
    )
    result["historical_candidate_coverage"] = result.content_id.isin(
        raw.labeled_content_id
    )
    result["candidate_status"] = result.historical_candidate_coverage.map(
        {True: "candidate_only", False: "missing_historical_linkage"}
    )
    supported = decisions[decisions.manual_decision == "supported"]
    test_supported = supported[
        supported.labeled_content_id.isin(fixed["test_records"].content_id)
    ]
    summary = {
        "sequence_internal_checks": sequence,
        "linkage_output_checksums_verified": True,
        "assignments_checksum_bound_to_linkage": True,
        "shared_group_support_reproduced": len(ranks),
        "base_images": len(base),
        "manual_status": result.manual_link_status.value_counts().to_dict(),
        "manual_status_by_split": result.groupby(["split", "manual_link_status"])
        .size()
        .to_dict(),
        "historical_candidates_covered": int(
            result.historical_candidate_coverage.sum()
        ),
        "shared_candidates_covered": int((result.shared_candidate_groups > 0).sum()),
        "conditional_protected_dependencies": int(
            result.conditional_protected_dependency.sum()
        ),
        "protected_test_supported_reviews": len(test_supported),
        "sequence_assignments_authorized": 0,
        "training_ready": False,
        "limitations": [
            "VDG producer metadata and source-bound review publications/events absent; CSV decisions retain historical status only.",
            "October decisions require reconciliation; historical support is not current split authority.",
            "Supported BASE and protected test links share groups conditionally; resolve before any split.",
            "Unreviewed and missing linkage cannot be silently assigned singleton independence.",
            "Review video_11min provenance before accepting historical group support.",
            "Changed development population requires a matched control; changed validation metrics are not directly comparable.",
        ],
    }
    summary["manual_status_by_split"] = {
        "/".join(k): int(v) for k, v in summary["manual_status_by_split"].items()
    }
    for path, expected_hash in list(hashes.items()):
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected_hash:
            raise ValueError("Input changed during assessment")
    args.output.mkdir(parents=True)
    result.to_csv(args.output / "base_link_assessment.csv", index=False)
    membership.to_csv(args.output / "historical_group_membership.csv", index=False)
    decisions.to_csv(args.output / "historical_decisions.csv", index=False)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (args.output / "input_checksums.json").write_text(
        json.dumps(hashes, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
