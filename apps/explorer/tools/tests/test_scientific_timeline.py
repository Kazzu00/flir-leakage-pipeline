"""Synthetic, offline scientific identity/provenance and presentation invariants."""

import copy
import importlib.util
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

spec = importlib.util.spec_from_file_location(
    "timeline_export", Path(__file__).parents[1] / "export_scientific_timeline.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture
def inputs():
    rows = [
        {
            "frame_id": f"f{i}",
            "content_id": c,
            "basename": f"synthetic_{i}.jpg",
            "possible_sequence": "family-A",
            "possible_frame_index": position,
            "source_archive": "synthetic.zip",
            "temporal_provenance": "filename_heuristic_not_ground_truth",
            "new_split": partition,
        }
        for i, c, position, partition in [
            (0, "c0", 0, "test"),
            (1, "c0", 0, "quarantine"),
            (2, "c1", 5, "train"),
            (3, "c2", 10, "val"),
            (4, "c2", 20, "train"),
        ]
    ]
    records = [
        {
            "record_id": r["frame_id"],
            "content_id": r["content_id"],
            "timeline_id": "axis-A",
            "frame_index": r["possible_frame_index"],
            "filename": r["basename"],
            "cohort": "labeled",
        }
        for r in rows
    ]
    organization = {
        "records": records,
        "contents": [{"content_id": f"c{i}", "preview_key": f"p{i}"} for i in range(3)],
        "media": [],
        "timelines": [
            {
                "timeline_id": "axis-A",
                "source_video_id": None,
                "inferred_family": "family-A",
                "source_archive": "synthetic.zip",
            }
        ],
        "clustering_configurations": [
            {
                "cluster_run_id": "cluster-A",
                "source_kind": "frozen_split_membership",
                "membership_consistency_verified": True,
                "full_clustering_artifact_available": False,
                "source_membership_checksums": {"frozen": "a" * 64},
            }
        ],
        "cluster_memberships": [
            {
                "cluster_run_id": "cluster-A",
                "content_id": f"c{i}",
                "cluster_id": label,
                "is_noise": label == -1,
            }
            for i, label in enumerate([0, -1, 7])
        ],
        "clusters": [{"cluster_run_id": "cluster-A", "cluster_id": label} for label in [0, -1, 7]],
        "split_memberships": [],
        "boundary_zones": [],
    }
    native = {
        "arm": "historical-S1-0",
        "rows": rows,
        "metadata": {
            "candidate_id": "candidate-0",
            "membership_id": "native-membership",
            "identity": {
                "arm": {
                    "arm": "historical-S1-0",
                    "population": "historical",
                    "policy": "S1",
                    "split_seed": 0,
                    "test_id": "test-A",
                }
            },
        },
    }
    return organization, [native]


def build(inputs):
    return module.build_projection(*inputs, expected_historical=5)


def test_multiplicity_clusters_noise_and_zero_index(inputs):
    result = build(inputs)
    assert len(result["records"]) == len(result["runs"][0]["members"]) == 5
    assert [r["position"] for r in result["records"][:2]] == [0, 0]
    assert {r["frame_id"] for r in result["records"] if r["content_id"] == "c0"} == {"f0", "f1"}
    assert {m["cluster_id"] for m in result["cluster_memberships"]} == {0, -1, 7}
    assert result["runs"][0]["split_space_id"] is None
    assert result["runs"][0]["membership_id"] == "native-membership"
    assert module.coverage(result, result["runs"][0])["partitions"]["quarantine"] == 1


def test_no_sequence_boundaries_from_cluster_transitions(inputs):
    result = build(inputs)
    assert len(result["intervals"]) == 1
    assert result["intervals"][0] | {} == {
        "interval_id": "candidate:axis-A",
        "axis_id": "axis-A",
        "sequence_id": None,
        "kind": "candidate_span",
        "review": "not_reviewed",
        "start": 0,
        "end": 20,
        "source_reference": "organization/timelines.json",
        "provenance": "Observed extent of an existing temporal namespace; not a confirmed sequence.",
    }
    assert result["axes"][0]["association"] == "filename_inference"


def test_rejects_preview_of_a_different_content_without_reading_images(inputs):
    inputs[0]["media"] = [
        {
            "preview_key": "p0",
            "content_id": "wrong-content",
            "media_available": True,
            "relative_path": "synthetic.jpg",
        }
    ]
    with pytest.raises(ValueError, match="Preview/content"):
        build(inputs)


@pytest.mark.parametrize(
    "reason", ["unknown_video", "unknown_order", "ambiguous_association", "index_conflict"]
)
def test_exclusion_reasons_account_for_occurrences(inputs, reason):
    org, runs = inputs
    row, record = runs[0]["rows"][-1], org["records"][-1]
    if reason == "unknown_video":
        record["timeline_id"] = None
        row["possible_sequence"] = None
    elif reason == "unknown_order":
        row["possible_frame_index"] = None
        record["frame_index"] = None
    elif reason == "ambiguous_association":
        row["possible_sequence"] = "different-family"
    else:
        row["possible_frame_index"] = 19
    result = build(inputs)
    assert result["records"][-1]["reason"] == reason
    counts = module.coverage(result, result["runs"][0])
    assert counts["total"] == counts["displayed"] + counts["excluded"] == 5
    assert counts["excluded"] == 1


def test_index_conflict_is_not_confused_with_exact_duplicate(inputs):
    org, runs = inputs
    runs[0]["rows"][-1]["possible_frame_index"] = 0
    org["records"][-1]["frame_index"] = 0
    result = build(inputs)
    assert {r["frame_id"] for r in result["records"] if r["reason"] == "index_conflict"} == {
        "f0",
        "f1",
        "f4",
    }


@pytest.mark.parametrize(
    "failure",
    [
        "content",
        "duplicate_frame",
        "duplicate_cluster",
        "unverified_cluster",
        "group_not_cluster",
        "inventory",
        "mixed_run",
        "test_changed",
    ],
)
def test_rejects_integrity_failures(inputs, failure):
    org, runs = inputs
    if failure == "content":
        runs[0]["rows"][0]["content_id"] = "wrong"
    elif failure == "duplicate_frame":
        runs[0]["rows"][-1]["frame_id"] = "f0"
    elif failure == "duplicate_cluster":
        org["cluster_memberships"].append(copy.deepcopy(org["cluster_memberships"][0]))
    elif failure == "unverified_cluster":
        org["clustering_configurations"][0]["membership_consistency_verified"] = False
    elif failure == "group_not_cluster":
        org["cluster_memberships"][0].pop("cluster_id")
        org["cluster_memberships"][0]["group_id"] = "must-not-substitute"
    elif failure == "inventory":
        runs[0]["rows"].pop()
    elif failure == "mixed_run":
        runs[0]["metadata"]["identity"]["arm"]["policy"] = "INVALID"
    else:
        other = copy.deepcopy(runs[0])
        other["arm"] = "historical-S1-1"
        other["metadata"]["identity"]["arm"].update(arm=other["arm"], split_seed=1)
        other["rows"][0]["new_split"] = "train"
        runs.append(other)
    with pytest.raises((ValueError, KeyError)):
        build(inputs)


def write_json(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(module.encoded(data))


@pytest.fixture
def files(inputs, tmp_path):
    org, runs = inputs
    publication = tmp_path / "publication"
    organization = tmp_path / "organization"
    for name, data in org.items():
        write_json(organization / (name + ".json"), data)
    write_json(
        organization / "manifest.json",
        {
            "schema_version": "organization-evidence-v2",
            "record_count": 5,
            "file_sha256": {p.name: module.digest(p.read_bytes()) for p in organization.iterdir()},
        },
    )
    native = runs[0]
    base = publication / "runs" / native["arm"]
    write_json(base / "metadata.json", native["metadata"])
    rows = [
        {**r, "label_ids": "DO_NOT_EXPORT_PROTECTED_ANNOTATIONS", "detector_metric": 999}
        for r in native["rows"]
    ]
    pq.write_table(pa.Table.from_pylist(rows), base / "occurrence_memberships.parquet")
    write_json(
        publication / "summary.json",
        {
            "state": "PROVISIONAL_CANDIDATES",
            "ready_for_training": False,
            "runs": [
                {
                    "arm": native["arm"],
                    "candidate_id": "candidate-0",
                    "membership_id": "native-membership",
                    "population": "historical",
                    "policy": "S1",
                    "seed": 0,
                    "state": "PROVISIONAL_VERIFIED",
                }
            ],
        },
    )
    write_json(
        publication / "provenance.json",
        {"git_commit": "synthetic-commit", "git_worktree_dirty": False},
    )
    write_json(
        publication / "receipt.json",
        {
            "schema_version": "fixed_test_train_val_v1",
            "artifacts": {
                p.relative_to(publication).as_posix(): module.digest(p.read_bytes())
                for p in publication.rglob("*")
                if p.is_file()
            },
        },
    )
    return publication, organization


def test_reproducible_export_and_sources_unchanged_no_protected_fields(files, tmp_path):
    before = {
        str(p): module.digest(p.read_bytes())
        for root in files
        for p in root.rglob("*")
        if p.is_file()
    }
    a, b = tmp_path / "a", tmp_path / "b"
    module.export(*files, a, expected_historical=5)
    module.export(*files, b, expected_historical=5)
    assert {p.name: p.read_bytes() for p in a.iterdir()} == {
        p.name: p.read_bytes() for p in b.iterdir()
    }
    assert {
        str(p): module.digest(p.read_bytes())
        for root in files
        for p in root.rglob("*")
        if p.is_file()
    } == before
    raw = (a / "index.json").read_text()
    assert (
        "DO_NOT_EXPORT" not in raw and '"label_ids"' not in raw and '"detector_metric"' not in raw
    )
    assert len(json.loads((a / "coverage.json").read_text())["runs"]) == 1
    with pytest.raises(ValueError):
        module.export(*files, a, expected_historical=5)
    with pytest.raises(ValueError):
        module.export(*files, files[0] / "inside", expected_historical=5)


@pytest.mark.parametrize("source", ["native", "organization"])
def test_checksum_mismatch_fails_before_output(files, tmp_path, source):
    target = (
        files[0] / "runs/historical-S1-0/occurrence_memberships.parquet"
        if source == "native"
        else files[1] / "records.json"
    )
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(ValueError, match="checksum"):
        module.export(*files, tmp_path / "output", expected_historical=5)
    assert not (tmp_path / "output").exists()
