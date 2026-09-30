"""Synthetic payloads matching observed schemas; no real FLIR identifiers or images."""

import copy
import hashlib
import itertools
import json
import shutil

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from flir_pipeline.cli import app
from flir_pipeline.data.identity import dataset_id_from_manifest
from flir_pipeline.data.variants import make_variant
from flir_pipeline.sequences.experiments.artifacts import inspect, tables
from flir_pipeline.sequences.experiments.native_evidence import read_native
from flir_pipeline.sequences.experiments.native_schema import (
    CSV_COLUMNS,
    NULLABLE,
    OCCURRENCE_SCHEMA,
    PAIR_COLUMNS,
    REQUIRED_FLAGS,
    ROLES,
)
from flir_pipeline.sequences.experiments.real_evidence import (
    discover,
    import_real,
    inspect_real,
)
from flir_pipeline.sequences.experiments.runner import verify
from flir_pipeline.sequences.experiments.sources import ExperimentSources
from flir_pipeline.sequences.experiments.structure import load_structure
from flir_pipeline.similarity.storage import file_sha256, read_json, write_json

FAMILY = "video_11min"
REVIEW_MODE = "assistant_visual_review_with_user_approval"
KINDS = {role: kind for kind, role in ROLES.items()}
# Pin the observed header independently of the importer schema constants.
OBSERVED_CANDIDATE_HEADER = (
    "review_order,core_a,core_b,clip_centroid_cosine,clip_symmetric_ge_090,"
    "clip_symmetric_ge_095,clip_recurrence_score,dinov2_centroid_cosine,"
    "dinov2_symmetric_ge_090,dinov2_symmetric_ge_095,dinov2_recurrence_score,"
    "consensus_score,encoder_agreement_score,clip_rank,dinov2_rank,rank_sum,"
    "mean_reciprocal_rank,joint_symmetric_ge_090,joint_centroid_min"
)


def csv(root, role, name, rows):
    pd.DataFrame(rows, columns=CSV_COLUMNS[role][name]).to_csv(
        root / role / name, index=False
    )


def make_reports(root):
    """Required cardinalities, entirely fabricated identifiers and intervals."""
    root.mkdir()
    summaries = {}
    for kind, role in ROLES.items():
        (root / role).mkdir()
        summaries[role] = {
            "artifact": kind,
            **dict.fromkeys(REQUIRED_FLAGS[role].split(), False),
        }
    summaries["provenance"].update(
        {
            "video_13min": {
                "status": "lineage_supported",
                "candidate_source": "synthetic-source.avi",
                "interpretation": "Source-lineage association, NOT byte identity, NOT ground truth.",
            },
            "video_11min": {
                "status": "unresolved",
                "tested_sources": [
                    "synthetic-a.mp4",
                    "synthetic-b.avi",
                    "synthetic-c.mp4",
                ],
                "internal_timeline_audit": {
                    "unique_nominal_indices": 712,
                    "index_min": 1,
                    "index_max": 712,
                    "ground_truth": False,
                    "sequence_boundaries_created": False,
                    "candidate_transitions_only": True,
                    "both_encoders_bottom_5pct_count": 22,
                },
            },
        }
    )
    summaries["manual"].update(
        query_family=FAMILY,
        review_passes=2,
        transition_count=22,
        region_count=16,
        transition_decision_counts={
            "supported_boundary": 17,
            "unsupported_boundary": 5,
        },
        region_status_counts={
            "supported_transition_region": 12,
            "unsupported_transition_region": 4,
        },
        supported_boundary_zone_count=12,
    )
    summaries["structure"].update(
        query_family=FAMILY,
        nominal_index_count=712,
        occurrence_count=910,
        sequence_core_candidate_count=13,
        boundary_zone_count=12,
        index_coverage_complete=True,
        occurrence_coverage_complete=True,
        cross_structure_exact_content_count=0,
        cross_split_exact_content_count=198,
    )
    summaries["recurrence"].update(
        core_count=13,
        pair_count=78,
        manual_review_candidate_count=11,
        selection_rule={
            "clip_symmetric_ge_090_min": 0.2,
            "dinov2_symmetric_ge_090_min": 0.05,
        },
        ranking={
            "method": "encoder_rank_consensus",
            "primary": "rank_sum",
            "secondary": "joint_symmetric_ge_090",
        },
    )
    for role, summary in summaries.items():
        write_json(root / role / "summary.json", summary)
    zones = [
        dict(
            region_id=f"synthetic-region-{i}",
            start_index=50 + i * 52,
            end_index=50 + i * 52 + (2 if i < 5 else 1),
        )
        for i in range(12)
    ]
    csv(root, "manual", "supported_boundary_zones.csv", zones)
    regions = [{**row, "region_status": "supported_transition_region"} for row in zones]
    regions.extend(
        dict(
            region_id=f"synthetic-unreviewed-{i}",
            start_index=10 + i * 8,
            end_index=11 + i * 8,
            region_status="unsupported_transition_region",
        )
        for i in range(4)
    )
    decisions = []
    for i, region in enumerate(regions):
        ids, labels = [], []
        for j in range(2 if i < 5 or i == 12 else 1):
            order = len(decisions) + 1
            label = "supported_boundary" if i < 12 else "unsupported_boundary"
            row = {c: 0.8 for c in CSV_COLUMNS["manual"]["transition_decisions.csv"]}
            row.update(
                review_order=order,
                transition_id=f"synthetic-transition-{order}",
                left_index=region["start_index"] + j,
                right_index=region["start_index"] + j + 1,
                contact_sheet=f"synthetic/{order}.png",
                decision=label,
                notes="synthetic source note",
                manual_decision=label,
                review_notes="synthetic review",
                review_mode=REVIEW_MODE,
                ground_truth=False,
                automatic_confirmation=False,
                sequence_boundary_created=False,
                split_created=False,
            )
            decisions.append(row)
            ids.append(row["transition_id"])
            labels.append(label)
        region.update(
            transition_ids=json.dumps(ids),
            edge_decisions=json.dumps(labels),
            ground_truth=False,
            sequence_boundary_created=False,
            region_semantics="synthetic uncertain interval",
            exact_boundary_known=False,
        )
    csv(root, "manual", "transition_decisions.csv", decisions)
    csv(root, "manual", "transition_regions.csv", regions)
    structure_zones = [
        {**z, "structure_id": f"synthetic-zone-{i}", "structure_type": "boundary_zone"}
        for i, z in enumerate(zones)
    ]
    cores, start = [], 1
    for i, zone in enumerate([*zones, {"start_index": 713, "end_index": 713}]):
        cores.append(
            dict(
                start_index=start,
                end_index=zone["start_index"] - 1,
                structure_id=f"synthetic-core-{i:02}",
                structure_type="sequence_core_candidate",
            )
        )
        start = zone["end_index"] + 1
    indices = []
    for element in [*cores, *structure_zones]:
        indices.extend(
            dict(
                nominal_index=i,
                structure_id=element["structure_id"],
                structure_type=element["structure_type"],
                start_index=element["start_index"],
                end_index=element["end_index"],
                review_region_id=element.get("region_id", ""),
            )
            for i in range(element["start_index"], element["end_index"] + 1)
        )
    indices.sort(key=lambda row: row["nominal_index"])
    csv(root, "structure", "boundary_zones.csv", structure_zones)
    csv(root, "structure", "sequence_core_candidates.csv", cores)
    csv(root, "structure", "index_structure.csv", indices)
    duplicate_indices = [
        *[r for r in indices if r["structure_type"] == "boundary_zone"][:11],
        *[r for r in indices if r["structure_type"] == "sequence_core_candidate"][:187],
    ]
    duplicates = {r["nominal_index"] for r in duplicate_indices}
    occurrences = []
    for position in indices:
        i = position["nominal_index"]
        for occurrence in range(2 if i in duplicates else 1):
            row = {
                name: (
                    False
                    if dtype == "bool"
                    else 0
                    if dtype == "int64"
                    else 0.25
                    if dtype == "float64"
                    else "synthetic"
                )
                for name, dtype in OCCURRENCE_SCHEMA.items()
            }
            content = hashlib.sha256(f"synthetic-content-{i}".encode()).hexdigest()
            row.update(
                position,
                frame_id=f"synthetic-frame-{i}-{occurrence}",
                content_id=content,
                image_sha256=content,
                label_sha256=hashlib.sha256(
                    f"synthetic-label-{i}".encode()
                ).hexdigest(),
                manifest_version="flir_canonical_v1",
                original_split="train" if occurrence == 0 else "test",
                possible_sequence=FAMILY,
                possible_frame_index=i,
                image_filename=f"synthetic_{i}.jpg",
                source_member_path=f"synthetic/{i}-{occurrence}.jpg",
                duplicate_group_id=content,
                duplicate_occurrence_count=2 if i in duplicates else 1,
                exact_duplicate=i in duplicates,
                cross_split_exact_duplicate=i in duplicates,
                duplicate_splits='["train", "test"]'
                if i in duplicates
                else '["train"]',
                review_region_id=position["review_region_id"] or None,
            )
            occurrences.append(row)
    for row in occurrences[:235]:
        for name in NULLABLE - {"review_region_id"}:
            row[name] = None
    schema = pa.schema(
        [
            pa.field(name, pa.type_for_alias(dtype), nullable=name in NULLABLE)
            for name, dtype in OCCURRENCE_SCHEMA.items()
        ]
    )
    pq.write_table(
        pa.Table.from_pylist(occurrences, schema=schema),
        root / "structure" / "occurrence_structure.parquet",
    )
    frame = pd.read_parquet(root / "structure" / "occurrence_structure.parquet")
    counts = []
    for element in [*cores, *structure_zones]:
        group = frame.loc[frame.structure_id.eq(element["structure_id"])]
        counts.append(
            {
                **element,
                "unique_index_count": group.nominal_index.nunique(),
                "occurrence_count": len(group),
                "unique_content_count": group.content_id.nunique(),
                "historical_split_counts": json.dumps(
                    group.original_split.value_counts().to_dict()
                ),
                "cross_split_exact_content_count": int(
                    group.groupby("content_id").original_split.nunique().gt(1).sum()
                ),
            }
        )
    csv(root, "structure", "structure_summary.csv", counts)
    pairs = []
    for i, (a, b) in enumerate(
        itertools.combinations([c["structure_id"] for c in cores], 2)
    ):
        pairs.append(
            dict(
                zip(
                    PAIR_COLUMNS,
                    [
                        a,
                        b,
                        0.8,
                        0.3 if i < 11 else 0.1,
                        0.05,
                        0.2,
                        0.8,
                        0.1 if i < 11 else 0.01,
                        0.02,
                        0.15,
                        0.175,
                        0.9,
                        i + 1,
                        78 - i,
                        79,
                        0.5 * (1 / (i + 1) + 1 / (78 - i)),
                        0.1,
                        0.8,
                    ],
                    strict=True,
                )
            )
        )
    csv(root, "recurrence", "all_pairs_ranked.csv", pairs)
    pd.DataFrame(
        [{"review_order": i + 1, **r} for i, r in enumerate(pairs[:11])],
        columns=OBSERVED_CANDIDATE_HEADER.split(","),
    ).to_csv(root / "recurrence" / "manual_review_candidates.csv", index=False)


def memory_source(root):
    path = root / "structure" / "occurrence_structure.parquet"
    records = pd.read_parquet(path)
    signature = {
        "dataset_id": dataset_id_from_manifest(records),
        "checksums": {"manifest_sha256": file_sha256(path)},
        "feature_spaces": {"clip": {"synthetic": True}, "dinov2": {"synthetic": True}},
    }
    records["timeline_id"] = FAMILY
    records["family"] = FAMILY
    records["position"] = records.nominal_index
    records["temporal_source"] = "filename_heuristic"
    ids = sorted(records.content_id.unique())
    records["content_row"] = records.content_id.map(
        {value: i for i, value in enumerate(ids)}
    )
    return ExperimentSources(records, ids, {}, signature, FAMILY)


@pytest.fixture(scope="module")
def reports_template(tmp_path_factory):
    root = tmp_path_factory.mktemp("native") / "reports"
    make_reports(root)
    return root


@pytest.fixture
def reports(reports_template, tmp_path):
    root = tmp_path / "r"
    shutil.copytree(reports_template, root)
    return root


def test_native_schema_inspection_is_read_only_and_exact(reports):
    before = {
        p.relative_to(reports): file_sha256(p)
        for p in reports.rglob("*")
        if p.is_file()
    }
    observed = inspect_real(reports, FAMILY)
    assert observed["dry_run"] and not observed["published"]
    assert (
        observed["occurrence_count"] == 910 and observed["nominal_index_count"] == 712
    )
    assert (
        observed["core_candidate_count"] == 13 and observed["boundary_zone_count"] == 12
    )
    assert (
        observed["recurrence_pair_count"] == 78
        and observed["manual_review_candidate_count"] == 11
    )
    assert not observed["canonical_source_verified"] and observed["dataset_id"] is None
    assert observed["dataset_variant"] == "unspecified"
    assert observed["legacy_review_modes"] == [REVIEW_MODE]
    raw = pq.read_table(reports / "structure" / "occurrence_structure.parquet")
    assert raw.num_columns == 46 and raw.num_rows == 910
    assert raw["review_region_id"].null_count == 870
    assert all(raw[name].null_count == 235 for name in NULLABLE - {"review_region_id"})
    assert sum(len(r["source_files"]) for r in observed["producers"].values()) == 14
    assert before == {
        p.relative_to(reports): file_sha256(p)
        for p in reports.rglob("*")
        if p.is_file()
    }


def test_native_import_preserves_legacy_semantics_ids_and_source_snapshots(
    reports, tmp_path
):
    source, output = memory_source(reports), tmp_path / "out"
    dry = import_real(reports, source, output, dry_run=True)
    assert dry["canonical_source_verified"] and not output.exists()
    result = import_real(reports, source, output)
    assert result == import_real(reports, source, output)
    meta, data, summary = (
        inspect(result),
        tables(result),
        read_json(result / "summary.json"),
    )
    assert meta["identity"]["sources"]["input"]["variant_name"] == "unspecified"
    assert "reviewer" not in summary and "reviewed_at" not in summary
    assert not summary["visual_verification_complete"] and not summary["ground_truth"]
    assert (
        not summary["sequence_instances_created"]
        and not summary["visual_dependency_groups_created"]
    )
    assert set(data["intervals"].kind) == {"boundary_zone", "sequence_core_candidate"}
    assert data["cores"].status.eq("candidate").all()
    assert data["cores"].origin.eq("imported_legacy_core_candidate").all()
    assert len(data["cores"]) == 13 and data["observations"].empty
    assert data["native_manual_transition_decisions"].review_mode.eq(REVIEW_MODE).all()
    assert set(data["cores"].element_id) == set(
        data["native_structure_sequence_core_candidates"].structure_id
    )
    assert (
        data["membership"]
        .loc[data["membership"].target.eq("sequence_core"), "evaluation_mask"]
        .sum()
        == 870
    )
    assert (
        not data["membership"]
        .loc[data["membership"].target.eq("sequence_instance"), "evaluation_mask"]
        .any()
    )
    for record in meta["identity"]["sources"]["native_producers"].values():
        for name, digest in record["source_files"].items():
            assert file_sha256(result / record["snapshots"][name]) == digest
    assert verify(result, source)["quality_valid"]
    for record in meta["identity"]["sources"]["native_producers"].values():
        for name in record["source_files"]:
            path = reports / ROLES[record["source_artifact_family"]] / name
            original = path.read_bytes()
            try:
                path.write_bytes(original + b"\n")
                checked = verify(result, source)
                assert not checked["quality_valid"]
                assert "Original consumed source changed" in checked["error"]
            finally:
                path.write_bytes(original)
    path = reports / "provenance" / "summary.json"
    path.write_bytes(path.read_bytes() + b"\n")
    checked = verify(result, source)
    assert (
        not checked["quality_valid"]
        and "Original consumed source changed" in checked["error"]
    )
    with pytest.raises(ValueError, match="Original consumed source changed"):
        load_structure(result, source)


@pytest.mark.parametrize(
    "role,field,value",
    [
        ("manual", "ground_truth", True),
        ("manual", "visual_verification_complete", True),
        ("structure", "sequence_instances_created", True),
        ("recurrence", "candidate_components_are_vdgs", True),
        ("structure", "query_family", "different_family"),
        ("manual", "transition_count", 21),
        ("structure", "occurrence_count", 909),
        ("recurrence", "pair_count", 77),
        ("manual", "schema_version", "unobserved_v9"),
        ("manual", "variant_name", "original_with_hud"),
    ],
)
def test_native_summary_conflicts_fail_closed(reports, role, field, value):
    path = reports / role / "summary.json"
    summary = read_json(path)
    summary[field] = value
    write_json(path, summary)
    with pytest.raises(ValueError):
        inspect_real(reports, FAMILY)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_column",
        "wrong_dtype",
        "null_identity",
        "missing_row",
        "moved_structure",
        "cross_content",
        "bad_region",
        "missing_index",
    ],
)
def test_native_occurrence_structure_conflicts_fail_closed(reports, mutation):
    path = reports / "structure" / "occurrence_structure.parquet"
    data = pd.read_parquet(path)
    if mutation == "missing_column":
        data = data.drop(columns="width")
    elif mutation == "wrong_dtype":
        data["nominal_index"] = data.nominal_index.astype(str)
    elif mutation == "null_identity":
        data.loc[0, "content_id"] = None
    elif mutation == "missing_row":
        data = data.iloc[:-1]
    elif mutation == "moved_structure":
        data.loc[0, "structure_id"] = "foreign-core"
    elif mutation == "cross_content":
        other = data.index[data.structure_id.ne(data.structure_id.iloc[0])][0]
        data.loc[other, "content_id"] = data.content_id.iloc[0]
    elif mutation == "bad_region":
        data.loc[data.structure_type.eq("boundary_zone"), "review_region_id"] = (
            "foreign-region"
        )
    else:
        data.loc[data.nominal_index.eq(712), "nominal_index"] = 711
    data.to_parquet(path, index=False)
    with pytest.raises(ValueError):
        inspect_real(reports, FAMILY)


def test_native_candidates_accept_observed_header_order(reports):
    path = reports / "recurrence" / "manual_review_candidates.csv"
    original = path.read_bytes()
    assert original.decode("utf-8").splitlines()[0] == OBSERVED_CANDIDATE_HEADER
    report = read_native(path.parent / "summary.json")
    candidates = report.tables["manual_review_candidates"]
    assert list(candidates) == OBSERVED_CANDIDATE_HEADER.split(",")
    assert candidates.review_order.tolist() == list(range(1, 12))
    assert path.read_bytes() == original


@pytest.mark.parametrize(
    "mutation",
    [
        "review_order_last",
        "missing_review_order",
        "missing_measurement",
        "extra_column",
        "renamed_column",
        "duplicate_column",
    ],
)
def test_native_candidates_reject_malformed_header(reports, mutation):
    path = reports / "recurrence" / "manual_review_candidates.csv"
    frame = pd.read_csv(path)
    if mutation == "review_order_last":
        frame = frame[[*frame.columns[1:], "review_order"]]
    elif mutation == "missing_review_order":
        frame = frame.drop(columns="review_order")
    elif mutation == "missing_measurement":
        frame = frame.drop(columns="clip_centroid_cosine")
    elif mutation == "extra_column":
        frame["unexpected_column"] = 0
    elif mutation == "renamed_column":
        frame = frame.rename(columns={"review_order": "review_index"})
    else:
        frame = frame.rename(columns={"core_a": "review_order"})
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="manual_review_candidates.csv CSV schema"):
        read_native(path.parent / "summary.json")


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate_reverse",
        "self_pair",
        "missing_pair",
        "missing_candidate",
        "rank_change",
        "foreign_core",
    ],
)
def test_native_recurrence_conflicts_fail_closed(reports, mutation):
    path = (
        reports
        / "recurrence"
        / (
            "manual_review_candidates.csv"
            if mutation in {"missing_candidate", "rank_change"}
            else "all_pairs_ranked.csv"
        )
    )
    data = pd.read_csv(path)
    if mutation == "duplicate_reverse":
        data.loc[1, ["core_a", "core_b"]] = data.loc[0, ["core_b", "core_a"]].to_numpy()
    elif mutation == "self_pair":
        data.loc[0, "core_b"] = data.loc[0, "core_a"]
    elif mutation in {"missing_pair", "missing_candidate"}:
        data = data.iloc[:-1]
    elif mutation == "rank_change":
        data.loc[0, "clip_rank"] = 2
    else:
        for file in ("all_pairs_ranked.csv", "manual_review_candidates.csv"):
            p = reports / "recurrence" / file
            pd.read_csv(p).replace("synthetic-core-00", "foreign-core").to_csv(
                p, index=False
            )
        with pytest.raises(ValueError, match="core IDs disagree"):
            inspect_real(reports, FAMILY)
        return
    data.to_csv(path, index=False)
    with pytest.raises(ValueError):
        inspect_real(reports, FAMILY)


def test_cross_artifact_zone_conflict_and_ambiguity(reports):
    shutil.copytree(reports / "provenance", reports / "duplicate")
    with pytest.raises(ValueError, match="found 2"):
        inspect_real(reports, FAMILY)
    selections = {KINDS["provenance"]: reports / "provenance"}
    assert inspect_real(reports, FAMILY, selections)["occurrence_count"] == 910
    for name in ("supported_boundary_zones.csv", "transition_regions.csv"):
        path = reports / "manual" / name
        data = pd.read_csv(path)
        data.loc[0, "start_index"] += 1
        data.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Cross-artifact boundary zones disagree"):
        inspect_real(reports, FAMILY, selections)


def test_legacy_variant_cannot_be_silently_assigned(reports, tmp_path):
    source = memory_source(reports)
    signature = copy.deepcopy(source.signature)
    signature["dataset_variant"] = make_variant(
        signature["dataset_id"],
        signature["checksums"]["manifest_sha256"],
        "original_with_hud",
    )
    named = ExperimentSources(
        source.records, source.contents, source.embeddings, signature, FAMILY
    )
    with pytest.raises(ValueError, match="unspecified variant"):
        import_real(reports, named, tmp_path / "out")
    source.records.loc[0, "content_id"] = "foreign-content"
    with pytest.raises(ValueError, match="identity differs"):
        import_real(reports, source, tmp_path / "out")


def test_native_cli_inspection_and_import_dry_run(reports, tmp_path, monkeypatch):
    runner = CliRunner()
    response = runner.invoke(
        app,
        [
            "sequences",
            "experiment",
            "inspect-real-evidence",
            "--root",
            str(reports),
            "--family",
            FAMILY,
        ],
    )
    assert response.exit_code == 0, response.output
    assert json.loads(response.output)["published"] is False
    source = memory_source(reports)
    monkeypatch.setattr(
        "flir_pipeline.sequences.experiments.inputs.resolve_inputs",
        lambda *a, **k: ({}, source),
    )
    response = runner.invoke(
        app,
        [
            "sequences",
            "experiment",
            "import-real-evidence",
            "--root",
            str(reports),
            "--family",
            FAMILY,
            "--dry-run",
            "--output",
            str(tmp_path / "out"),
        ],
    )
    assert response.exit_code == 0, response.output
    assert json.loads(response.output)["canonical_source_verified"] is True
    assert not (tmp_path / "out").exists()


def test_normalized_snapshots_are_not_rediscovered_as_producers(reports):
    source = memory_source(reports)
    import_real(reports, source, reports / "normalized")
    assert set(discover(reports, source)) == set(ROLES)


def test_native_dataset_identity_conflicts_even_without_canonical_inputs(reports):
    for role, value in (("manual", "synthetic-one"), ("structure", "synthetic-two")):
        path = reports / role / "summary.json"
        summary = read_json(path)
        summary["dataset_id"] = value
        write_json(path, summary)
    with pytest.raises(ValueError, match="dataset identity conflict"):
        inspect_real(reports, FAMILY)


def test_native_encoder_ranks_preserve_producer_convention(reports, tmp_path):
    for filename in ("all_pairs_ranked.csv", "manual_review_candidates.csv"):
        path = reports / "recurrence" / filename
        frame = pd.read_csv(path)
        frame[["clip_rank", "dinov2_rank"]] -= 1
        frame["rank_sum"] -= 2
        frame.to_csv(path, index=False)
    artifact = import_real(reports, memory_source(reports), tmp_path / "out")
    imported = tables(artifact)["native_recurrence_all_pairs_ranked"]
    expected = pd.read_csv(reports / "recurrence" / "all_pairs_ranked.csv")
    for column in ("clip_rank", "dinov2_rank", "rank_sum"):
        assert imported[column].tolist() == expected[column].tolist()


def test_native_csv_semantics_and_unknown_revision_are_rejected(reports):
    path = reports / "manual" / "transition_regions.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "exact_boundary_known"] = True
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="semantics"):
        inspect_real(reports, FAMILY)
    frame.loc[0, "exact_boundary_known"] = False
    frame.to_csv(path, index=False)
    path = reports / "manual" / "summary.json"
    summary = read_json(path)
    summary["artifact"] = "video11_manual_transition_review_v99"
    write_json(path, summary)
    with pytest.raises(ValueError, match="found 0"):
        inspect_real(reports, FAMILY)
