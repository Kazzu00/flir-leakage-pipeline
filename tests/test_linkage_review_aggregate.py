"""Offline aggregate QA: immutable evidence, explicit decisions and no promotion."""

import json
import os
import shutil
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd
import pytest
from test_linkage_review import confirmed_v1_template as confirmed_v1_template
from test_linkage_review import review_template as review_template
from typer.testing import CliRunner

from flir_pipeline.cli import app
from flir_pipeline.linkage.review_aggregate_storage import (
    aggregate_reviews,
    load_source_map,
    verify_aggregate,
)
from flir_pipeline.linkage.review_aggregation import FLAGS
from flir_pipeline.linkage.review_model import KEYS, ReviewConfig, read_csv
from flir_pipeline.linkage.review_storage import (
    init_review,
    inspect_snapshot,
    record_review,
    verify_review,
)
from flir_pipeline.similarity.storage import file_sha256, read_json, write_json


def record(directory, imports, decisions, *, reviewer="reviewer-a", timestamp=None):
    imports.mkdir(parents=True, exist_ok=True)
    path = imports / "decisions.csv"
    decisions.to_csv(path, index=False, lineterminator="\n")
    return record_review(
        directory,
        path,
        reviewer=reviewer,
        source="synthetic manual contact-sheet review",
        timestamp=timestamp or "2026-01-01T00:00:00+00:00",
        output=imports / "revisions",
    )


def initialize(root, paths, image_root, sample, *, context=3):
    root.mkdir(parents=True, exist_ok=True)
    sample_path = root / "sample.csv"
    sample.to_csv(sample_path, index=False, lineterminator="\n")
    paths = replace(paths, calibration_sample=sample_path)
    directory = init_review(
        paths,
        ReviewConfig(context_seconds=context),
        root / "reviews",
        labeled_images_root=image_root / "labeled-images",
        video_images_root=image_root / "video-images",
    )
    return directory, paths


@pytest.fixture(scope="module")
def aggregate_template(confirmed_v1_template, tmp_path_factory):
    image_root, paths, initial = confirmed_v1_template
    root = tmp_path_factory.mktemp("review-aggregate")
    sample = read_csv(paths.calibration_sample)
    directories, bindings = [], {}
    for name, indices, stratum in (
        ("first", [0, 1], "agreement"),
        ("second", [1, 2], "disagreement"),
    ):
        selected = sample.iloc[indices].copy()
        selected["review_stratum"] = stratum
        directory, sources = initialize(root / name, paths, image_root, selected)
        changes = selected[KEYS].copy()
        changes["manual_decision"] = [
            {0: "supported", 1: "ambiguous", 2: ""}[index] for index in indices
        ]
        changes["manual_notes"] = [
            f"Explicit {name} notes for {index}" for index in indices
        ]
        revision = record(directory, root / name / "import", changes, reviewer=name)
        directories.append(revision)
        bindings[read_json(revision / "metadata.json")["calibration_id"]] = sources
    aggregate = aggregate_reviews(directories, bindings, root / "aggregates")
    return root, directories, bindings, aggregate, (image_root, paths, initial)


def test_compatible_revisions_descriptive_counts_lineage_and_flags(aggregate_template):
    _, directories, _, aggregate, _ = aggregate_template
    result = verify_aggregate(aggregate)
    assert result["quality_valid"] and result["source_bound"], result
    assert result["constituent_reviews_reverified"] and result["decisions_replayed"]
    meta = read_json(aggregate / "metadata.json")
    summary = read_json(aggregate / "summary.json")
    assert all(meta[key] is False and result[key] is False for key in FLAGS)
    assert all(summary["semantics"][key] is False for key in FLAGS)
    assert summary["semantics"]["samples_exchangeable"] is False
    assert (
        "not representative accuracy or precision" in summary["semantics"]["statistics"]
    )
    overall = summary["overall_pooled_descriptive"]
    assert overall["query_count"] == 3
    assert overall["decision_counts"] == {
        "supported": 1,
        "ambiguous": 1,
        "unsupported": 0,
        "blank": 1,
    }
    assert overall["decision_rates"]["blank"] == 1 / 3
    assert overall["unresolved_count"] == 2
    assert overall["distinct_group_count"] >= 2
    assert summary["source_revision_count"] == summary["source_calibration_count"] == 2
    assert summary["source_observation_count"] == 4
    assert (
        summary["duplicate_pair_count"]
        == summary["duplicate_observations_not_double_counted"]
        == 1
    )
    assert [row["query_count"] for row in summary["by_stratum"]] == [2, 2]
    assert {row["review_stratum"] for row in summary["by_stratum"]} == {
        "agreement",
        "disagreement",
    }
    assert sum(row["query_count"] for row in summary["by_visual_dependency_group"]) == 3
    assert sum(row["query_count"] for row in summary["by_stratum_and_group"]) == 4
    assert all(
        row["overall"]["query_count"] == 2 for row in summary["by_source_revision"]
    )
    observations = pd.read_parquet(aggregate / "source_observations.parquet")
    pooled = pd.read_parquet(aggregate / "pooled_reviews.parquet")
    duplicates = pd.read_parquet(aggregate / "duplicate_pairs.parquet")
    counts = pd.read_parquet(aggregate / "descriptive_counts.parquet")
    assert len(pooled) == 3 and not pooled.duplicated(KEYS).any()
    assert len(duplicates) == 1 and duplicates.source_observation_count.iloc[0] == 2
    assert set(counts.scope) == {
        "overall_pooled_descriptive",
        "source_revision",
        "source_revision_stratum_and_group",
        "by_stratum",
        "by_visual_dependency_group",
        "by_stratum_and_group",
    }
    source_history = {
        row["source_revision_id"]: row
        for row in read_json(aggregate / "source_revisions.json")
    }
    for directory in directories:
        original, review, history = inspect_snapshot(directory)
        copied = observations.loc[
            observations.source_revision_id.eq(original["artifact_id"]), review.columns
        ]
        pd.testing.assert_frame_equal(
            copied.reset_index(drop=True), review, check_exact=True
        )
        assert source_history[original["artifact_id"]]["metadata"] == original
        assert source_history[original["artifact_id"]]["decision_history"] == history
        assert (
            "visual_dependency_consumer_source_fingerprint"
            in original["identity"]["calibration"]["sources"]
        )


def test_deterministic_identity_order_reuse_and_readonly_sources(
    aggregate_template, tmp_path
):
    _, directories, bindings, original, _ = aggregate_template
    before = {
        str(path): file_sha256(path)
        for directory in directories
        for path in directory.rglob("*")
        if path.is_file()
    }
    rebuilt = aggregate_reviews(list(reversed(directories)), bindings, tmp_path / "out")
    assert rebuilt.name == original.name
    assert aggregate_reviews(directories, bindings, tmp_path / "out") == rebuilt
    moved = tmp_path / "copied-review"
    shutil.copytree(directories[0], moved)
    relocated_aggregate = aggregate_reviews(
        [moved, directories[1]], bindings, tmp_path / "relocated-output"
    )
    assert relocated_aggregate.name == original.name
    for name in (
        "summary.json",
        "pooled_reviews.parquet",
        "source_observations.parquet",
        "duplicate_pairs.parquet",
        "descriptive_counts.parquet",
    ):
        assert (rebuilt / name).read_bytes() == (original / name).read_bytes()
    assert before == {
        str(path): file_sha256(path)
        for directory in directories
        for path in directory.rglob("*")
        if path.is_file()
    }


@pytest.mark.parametrize("decision", ["supported", "unsupported", ""])
def test_conflicting_duplicates_including_blank_fail(
    aggregate_template, tmp_path, decision
):
    _, directories, bindings, _, _ = aggregate_template
    review = read_csv(directories[1] / "review.csv")
    changes = review.loc[review.manual_decision.eq("ambiguous"), KEYS].copy()
    changes["manual_decision"] = decision
    changes["manual_notes"] = "Conflicting explicit manual import"
    conflict = record(
        directories[1],
        tmp_path / "import",
        changes,
        timestamp="2026-01-02T00:00:00+00:00",
    )
    with pytest.raises(
        ValueError, match="Conflicting manual decisions for query/group"
    ):
        aggregate_reviews([directories[0], conflict], bindings, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_identical_pairs_in_same_stratum_not_counted_twice_and_notes_retained(
    aggregate_template, tmp_path
):
    _, directories, bindings, _, _ = aggregate_template
    review = read_csv(directories[0] / "review.csv")
    changes = review[[*KEYS, "manual_decision", "manual_notes"]].copy()
    changes["manual_notes"] = "Independent notes; same recorded decisions"
    second = record(
        directories[0],
        tmp_path / "import",
        changes,
        reviewer="another-reviewer",
        timestamp="2026-01-02T00:00:00+00:00",
    )
    aggregate = aggregate_reviews([directories[0], second], bindings, tmp_path / "out")
    summary = read_json(aggregate / "summary.json")
    assert summary["duplicate_pair_count"] == 2
    assert summary["overall_pooled_descriptive"]["query_count"] == 2
    assert summary["by_stratum"][0]["query_count"] == 2
    assert summary["source_calibration_count"] == 1
    observations = pd.read_parquet(aggregate / "source_observations.parquet")
    assert set(observations.reviewer) == {"first", "another-reviewer"}
    assert observations.manual_notes.nunique() == 3
    assert verify_aggregate(aggregate)["quality_valid"]


def relocated(aggregate_template, tmp_path):
    _, directories, _, aggregate, _ = aggregate_template
    copied = tmp_path / "revision"
    shutil.copytree(directories[0], copied)
    locations = read_json(aggregate / "source_locations.json")
    revision_id = read_json(copied / "metadata.json")["artifact_id"]
    locations["revisions"][revision_id] = str(copied)
    path = tmp_path / "locations.json"
    write_json(path, locations)
    return copied, path


@pytest.mark.parametrize(
    "filename",
    ["metadata.json", "review.csv", "decision_history.json", "contact_sheet"],
)
def test_tampered_constituent_bytes_fail_even_metadata_whitespace(
    aggregate_template, tmp_path, filename
):
    _, _, _, aggregate, _ = aggregate_template
    copied, locations = relocated(aggregate_template, tmp_path)
    assert verify_aggregate(aggregate, locations=locations)["quality_valid"]
    path = (
        next(copied.rglob("*.png"))
        if filename == "contact_sheet"
        else copied / filename
    )
    path.write_bytes(path.read_bytes() + b"\n")
    quality = verify_aggregate(aggregate, locations=locations)
    assert not quality["quality_valid"] and not quality["source_bound"]
    if filename == "metadata.json":
        assert "exact source revision bytes" in quality["error"]


def test_constituent_verification_rebinds_original_sources(
    aggregate_template, tmp_path
):
    _, _, _, aggregate, _ = aggregate_template
    locations = read_json(aggregate / "source_locations.json")
    first = next(iter(locations["calibrations"].values()))
    sample = tmp_path / "changed-sample.csv"
    sample.write_bytes(Path(first["calibration_sample"]).read_bytes() + b"\n")
    first["calibration_sample"] = str(sample)
    location_file = tmp_path / "locations.json"
    write_json(location_file, locations)
    quality = verify_aggregate(aggregate, locations=location_file)
    assert not quality["quality_valid"]
    assert "failed review verification" in quality["error"]


@pytest.mark.parametrize(
    "filename",
    [
        "summary.json",
        "source_revisions.json",
        "source_observations.parquet",
        "pooled_reviews.parquet",
        "duplicate_pairs.parquet",
        "descriptive_counts.parquet",
    ],
)
def test_verify_reconstructs_instead_of_trusting_rehashed_outputs(
    aggregate_template, tmp_path, filename
):
    _, _, _, aggregate, _ = aggregate_template
    copied = tmp_path / "aggregate"
    shutil.copytree(aggregate, copied)
    path = copied / filename
    if filename == "summary.json":
        data = read_json(path)
        data["overall_pooled_descriptive"]["query_count"] += 1
        write_json(path, data)
    elif filename == "source_revisions.json":
        data = read_json(path)
        data[0]["decision_history"][0]["reviewer"] = "fabricated-reviewer"
        write_json(path, data)
    else:
        data = pd.read_parquet(path)
        field = (
            "query_count"
            if filename == "descriptive_counts.parquet"
            else "manual_decision"
        )
        data.loc[0, field] = 999 if field == "query_count" else "unsupported"
        data.to_parquet(path, index=False)
    meta = read_json(copied / "metadata.json")
    meta["output_checksums"][filename] = file_sha256(path)
    write_json(copied / "metadata.json", meta)
    assert not verify_aggregate(copied)["quality_valid"]


@pytest.mark.parametrize("flag", FLAGS)
def test_no_confirmation_ground_truth_or_split_flags(
    aggregate_template, tmp_path, flag
):
    _, _, _, aggregate, _ = aggregate_template
    copied = tmp_path / "aggregate"
    shutil.copytree(aggregate, copied)
    meta = read_json(copied / "metadata.json")
    meta[flag] = True
    write_json(copied / "metadata.json", meta)
    assert not verify_aggregate(copied)["quality_valid"]


def test_duplicate_evidence_must_have_same_protocol(aggregate_template, tmp_path):
    _, _, _, _, (image_root, paths, initial) = aggregate_template
    alternative, sources = initialize(
        tmp_path / "different-context",
        paths,
        image_root,
        read_csv(paths.calibration_sample),
        context=1,
    )
    bindings = {
        read_json(initial / "metadata.json")["calibration_id"]: paths,
        read_json(alternative / "metadata.json")["calibration_id"]: sources,
    }
    with pytest.raises(ValueError, match="Incompatible duplicate evidence"):
        aggregate_reviews([initial, alternative], bindings, tmp_path / "out")


def test_same_query_with_distinct_proposed_groups_is_two_units(
    aggregate_template, tmp_path
):
    _, _, _, _, (image_root, paths, initial) = aggregate_template
    rows = read_csv(initial / "review.csv")
    for row in rows.to_dict("records"):
        groups = {
            occurrence["visual_dependency_group_id"]
            for candidate in json.loads(row["candidate_details_json"])
            for occurrence in candidate["occurrences"]
        } - {row[KEYS[1]]}
        if groups:
            sample = pd.DataFrame(
                [
                    {
                        **{key: row[key] for key in KEYS},
                        KEYS[1]: sorted(groups)[0],
                        "review_stratum": "alternative-group",
                    }
                ]
            )
            break
    else:
        pytest.fail("Synthetic fixture must include occurrence ambiguity across groups")
    alternate, sources = initialize(tmp_path / "alternative", paths, image_root, sample)
    bindings = {
        read_json(initial / "metadata.json")["calibration_id"]: paths,
        read_json(alternate / "metadata.json")["calibration_id"]: sources,
    }
    aggregate = aggregate_reviews([initial, alternate], bindings, tmp_path / "out")
    pooled = pd.read_parquet(aggregate / "pooled_reviews.parquet")
    assert len(pooled) == len(rows) + 1
    assert pooled.labeled_content_id.nunique() == len(rows)
    assert pooled.manual_decision.eq("").all()
    assert verify_aggregate(aggregate)["quality_valid"]


def test_incompatible_group_domains_are_not_pooled(aggregate_template, tmp_path):
    _, _, _, _, (image_root, paths, initial) = aggregate_template
    copied = tmp_path / "dependencies"
    shutil.copytree(paths.visual_dependencies, copied)
    meta = copied / "metadata.json"
    meta.write_bytes(meta.read_bytes() + b"\n")
    alternative, sources = initialize(
        tmp_path / "alternative",
        replace(paths, visual_dependencies=copied),
        image_root,
        read_csv(paths.calibration_sample),
    )
    bindings = {
        read_json(initial / "metadata.json")["calibration_id"]: paths,
        read_json(alternative / "metadata.json")["calibration_id"]: sources,
    }
    with pytest.raises(
        ValueError, match="Incompatible visual dependency group domains"
    ):
        aggregate_reviews([initial, alternative], bindings, tmp_path / "out")


def test_input_contract_and_output_protection(aggregate_template, tmp_path):
    _, directories, bindings, _, _ = aggregate_template
    with pytest.raises(ValueError, match="at least one"):
        aggregate_reviews([], bindings, tmp_path / "empty")
    with pytest.raises(ValueError, match="more than once"):
        aggregate_reviews(
            [directories[0], directories[0]], bindings, tmp_path / "duplicate"
        )
    with pytest.raises(ValueError, match="Missing source paths"):
        aggregate_reviews(directories, {}, tmp_path / "unbound")
    with pytest.raises(ValueError, match="separate from"):
        aggregate_reviews(directories, bindings, directories[0])
    with pytest.raises(OSError):
        aggregate_reviews([tmp_path / "not-a-revision"], bindings, tmp_path / "bad")


def test_cli_source_map_and_aggregate_verify(aggregate_template, tmp_path):
    _, directories, bindings, _, _ = aggregate_template
    source_map = tmp_path / "sources.json"
    payload = {
        "calibrations": {
            key: {
                name: str(value)
                if name == "membership_table"
                else os.path.relpath(value, source_map.parent)
                for name, value in asdict(paths).items()
            }
            for key, paths in bindings.items()
        }
    }
    write_json(source_map, payload)
    assert load_source_map(source_map) == bindings
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "linkage",
            "review",
            "aggregate",
            *map(str, directories),
            "--source-map",
            str(source_map),
            "--output",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 0, result.output
    directory = result.output.strip()
    verified = runner.invoke(app, ["linkage", "review", "aggregate-verify", directory])
    assert verified.exit_code == 0, verified.output
    assert json.loads(verified.output)["quality_valid"]
    # Membership table selection remains explicit even for the legacy producer.
    first = next(iter(payload["calibrations"].values()))
    del first["membership_table"]
    write_json(source_map, payload)
    failed = runner.invoke(
        app,
        [
            "linkage",
            "review",
            "aggregate",
            str(directories[0]),
            "--source-map",
            str(source_map),
            "--output",
            str(tmp_path / "failed"),
        ],
    )
    assert failed.exit_code == 1 and "all ReviewPaths fields" in failed.output


def test_init_revision_is_valid_unresolved_input(aggregate_template, tmp_path):
    _, _, _, _, (_, paths, initial) = aggregate_template
    assert verify_review(initial, paths)["quality_valid"]
    calibration_id = read_json(initial / "metadata.json")["calibration_id"]
    aggregate = aggregate_reviews([initial], {calibration_id: paths}, tmp_path / "out")
    counts = read_json(aggregate / "summary.json")["overall_pooled_descriptive"]
    assert counts["unresolved_count"] == counts["query_count"] == 3


@pytest.mark.parametrize(
    "fault",
    ["missing_revision", "wrong_revision", "invalid_calibrations", "empty_path"],
)
def test_invalid_locations_fail_closed(aggregate_template, tmp_path, fault):
    _, _, _, aggregate, _ = aggregate_template
    located = read_json(aggregate / "source_locations.json")
    key = next(iter(located["revisions"]))
    if fault == "missing_revision":
        del located["revisions"][key]
    elif fault == "wrong_revision":
        located["revisions"][key] = str(aggregate)
    elif fault == "invalid_calibrations":
        located["calibrations"] = []
    else:
        located["revisions"][key] = ""
    path = tmp_path / "locations.json"
    write_json(path, located)
    result = verify_aggregate(aggregate, locations=path)
    assert not result["quality_valid"] and not result["source_bound"]
