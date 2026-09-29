"""Read-only adapters for explicitly observed legacy schemas; no evidence promotion."""

import hashlib
import itertools
import json
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from flir_pipeline.data.local_images import declared_file
from flir_pipeline.sequences.experiments.artifacts import inspect, publish
from flir_pipeline.sequences.experiments.native_schema import (
    ADAPTER,
    CSV_COLUMNS,
    FALSE_FLAGS,
    NULLABLE,
    OCCURRENCE_SCHEMA,
    PAIR_COLUMNS,
    REQUIRED_FLAGS,
    ROLES,
)
from flir_pipeline.sequences.experiments.structure import (
    Interval,
    Observation,
    structure_membership,
    validate_intervals,
)
from flir_pipeline.similarity.storage import file_sha256, read_json


@dataclass
class NativeReport:
    path: Path
    kind: str
    summary: dict
    snapshots: dict[str, bytes]
    tables: dict[str, pd.DataFrame]

    @property
    def checksums(self):
        return {
            name: hashlib.sha256(raw).hexdigest()
            for name, raw in self.snapshots.items()
        }


def require(condition, message):
    if not condition:
        raise ValueError(message)


def false_semantics(value, location="summary"):
    """Check analogous nested claims too; absent flags are never invented inputs."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key in FALSE_FLAGS:
                require(
                    item is False, f"Forbidden promoted semantics: {location}.{key}"
                )
            if key in {
                "sequence_instance_count",
                "visual_dependency_group_count",
                "vdg_count",
                "split_count",
            }:
                require(
                    type(item) is int and item == 0,
                    f"Forbidden created groups: {location}.{key}",
                )
            false_semantics(item, f"{location}.{key}")
    elif isinstance(value, list):
        for item in value:
            false_semantics(item, location)


def exact_counts(document, expected, location):
    for key, value in expected.items():
        require(
            type(document.get(key)) is int and document[key] == value,
            f"{location}: revision count mismatch for {key}; expected {value}",
        )


def integer_columns(frame, columns):
    for name in columns:
        values = pd.to_numeric(frame[name], errors="raise")
        require(
            np.isfinite(values).all()
            and (values % 1 == 0).all()
            and values.ge(0).all(),
            f"Invalid nonnegative integer column: {name}",
        )
        frame[name] = values.astype("int64")


def unique_nonempty(frame, column):
    require(
        frame[column].notna().all()
        and frame[column].astype(str).str.strip().ne("").all()
        and frame[column].is_unique,
        f"Missing or duplicate {column}",
    )


def read_native(path, document=None):
    path = Path(path).resolve()
    raw = path.read_bytes()
    summary = json.loads(raw)
    if document is not None:
        require(summary == document, "Native summary changed during discovery")
    kind = summary.get("artifact")
    require(
        kind in ROLES,
        "Uninspected producer schema: expected native artifact declaration",
    )
    require(
        summary.get("schema_version", kind) == kind,
        "Unsupported native producer schema revision",
    )
    role = ROLES[kind]
    for key in REQUIRED_FLAGS[role].split():
        require(summary.get(key) is False, f"{kind} requires false semantics: {key}")
    false_semantics(summary)
    snapshots, tables = {path.name: raw}, {}
    for name, columns in CSV_COLUMNS.get(role, {}).items():
        raw = declared_file(path.parent, name).read_bytes()
        snapshots[name] = raw
        # Keep text (including legacy review_mode and paths) verbatim. Only
        # documented numerical fields are converted for consistency checks.
        table = pd.read_csv(BytesIO(raw), dtype=str, keep_default_na=False)
        require(list(table) == columns, f"Unsupported {kind}/{name} CSV schema")
        for flag in FALSE_FLAGS & set(table):
            require(
                table[flag].str.lower().eq("false").all(),
                f"Forbidden {name}.{flag} semantics",
            )
            table[flag] = False
        tables[Path(name).stem] = table
    if role == "structure":
        name = "occurrence_structure.parquet"
        raw = declared_file(path.parent, name).read_bytes()
        snapshots[name] = raw
        arrow = pq.read_table(BytesIO(raw))
        require(
            arrow.column_names == list(OCCURRENCE_SCHEMA),
            "Unsupported 46-column occurrence schema",
        )
        for name, expected in OCCURRENCE_SCHEMA.items():
            actual = arrow.schema.field(name).type
            valid = (
                (pa.types.is_string(actual) or pa.types.is_large_string(actual))
                if expected == "string"
                else actual == pa.type_for_alias(expected)
            )
            require(valid, f"Unsupported occurrence column type: {name} ({actual})")
            if name not in NULLABLE:
                require(
                    arrow[name].null_count == 0,
                    f"Unexpected null occurrence column: {name}",
                )
        tables["occurrence_structure"] = arrow.to_pandas()
    report = NativeReport(path, kind, summary, snapshots, tables)
    {
        "provenance": validate_provenance,
        "manual": validate_manual,
        "structure": validate_structure,
        "recurrence": validate_recurrence,
    }[role](report)
    unchanged(report)
    return report


def unchanged(report):
    for name, expected in report.checksums.items():
        require(
            file_sha256(declared_file(report.path.parent, name)) == expected,
            f"Original consumed source changed: {report.path.parent / name}",
        )


def named_object(document, name):
    """Locate explicitly named JSON evidence; never derive it from a filename."""
    matches = []

    def visit(value):
        if isinstance(value, dict):
            if name in value:
                matches.append(value[name])
            for nested in value.values():
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)

    visit(document)
    require(
        len(matches) == 1 and isinstance(matches[0], dict),
        f"Missing/ambiguous declared JSON evidence: {name}",
    )
    return matches[0]


def validate_provenance(report):
    first = named_object(report.summary, "video_13min")
    second = named_object(report.summary, "video_11min")
    require(
        first.get("status") == "lineage_supported", "Unsupported source-lineage status"
    )
    require(
        isinstance(first.get("candidate_source"), str)
        and bool(first["candidate_source"]),
        "Missing declared candidate source",
    )
    require(
        second.get("status") == "unresolved",
        "Unresolved family cannot receive provenance",
    )
    require(
        isinstance(second.get("tested_sources"), list)
        and all(isinstance(s, str) for s in second["tested_sources"]),
        "Invalid tested sources",
    )
    audit = named_object(report.summary, "internal_timeline_audit")
    require(
        audit.get("query_family", "video_11min") == "video_11min",
        "Provenance timeline audit family conflict",
    )
    exact_counts(
        audit,
        {
            "unique_nominal_indices": 712,
            "index_min": 1,
            "index_max": 712,
            "both_encoders_bottom_5pct_count": 22,
        },
        "provenance audit",
    )
    require(
        audit.get("ground_truth") is False
        and audit.get("sequence_boundaries_created") is False
        and audit.get("candidate_transitions_only") is True,
        "Invalid candidate-only timeline audit",
    )


def validate_manual(report):
    s, t = report.summary, report.tables
    exact_counts(
        s,
        {
            "review_passes": 2,
            "transition_count": 22,
            "region_count": 16,
            "supported_boundary_zone_count": 12,
        },
        "manual review",
    )
    decisions, regions, zones = (
        t[k]
        for k in (
            "transition_decisions",
            "transition_regions",
            "supported_boundary_zones",
        )
    )
    require(
        (len(decisions), len(regions), len(zones)) == (22, 16, 12),
        "Manual table counts disagree",
    )
    for frame, key in (
        (decisions, "transition_id"),
        (regions, "region_id"),
        (zones, "region_id"),
    ):
        unique_nonempty(frame, key)
    integer_columns(decisions, ("review_order", "left_index", "right_index"))
    require(
        decisions.review_order.is_unique
        and decisions.left_index.lt(decisions.right_index).all(),
        "Invalid transition order or endpoints",
    )
    require(
        decisions.review_mode.str.strip().ne("").all(), "Missing legacy review_mode"
    )
    expected = {"supported_boundary": 17, "unsupported_boundary": 5}
    require(
        decisions.manual_decision.value_counts().to_dict() == expected
        and s.get("transition_decision_counts") == expected,
        "Transition decision counts disagree",
    )
    expected = {"supported_transition_region": 12, "unsupported_transition_region": 4}
    require(
        regions.region_status.value_counts().to_dict() == expected
        and s.get("region_status_counts") == expected,
        "Region status counts disagree",
    )
    for frame in (regions, zones):
        integer_columns(frame, ("start_index", "end_index"))
        require(frame.start_index.le(frame.end_index).all(), "Reversed boundary zone")
    selected = regions.loc[
        regions.region_status.eq("supported_transition_region"), list(zones)
    ]
    equal_sorted(
        zones,
        selected,
        "region_id",
        "Supported zones disagree with legacy reviewed regions",
    )


def equal_sorted(left, right, key, message):
    try:
        pd.testing.assert_frame_equal(
            left.sort_values(key).reset_index(drop=True),
            right.sort_values(key).reset_index(drop=True),
            check_dtype=False,
        )
    except AssertionError as error:
        raise ValueError(message) from error


def validate_structure(report):
    s, t = report.summary, report.tables
    exact_counts(
        s,
        {
            "nominal_index_count": 712,
            "occurrence_count": 910,
            "sequence_core_candidate_count": 13,
            "boundary_zone_count": 12,
            "cross_structure_exact_content_count": 0,
            "cross_split_exact_content_count": 198,
        },
        "structure",
    )
    require(
        s.get("index_coverage_complete") is True
        and s.get("occurrence_coverage_complete") is True,
        "Incomplete declared structure coverage",
    )
    zones, cores, indices, occurrences, counts = (
        t[k]
        for k in (
            "boundary_zones",
            "sequence_core_candidates",
            "index_structure",
            "occurrence_structure",
            "structure_summary",
        )
    )
    require(
        (len(zones), len(cores), len(indices), len(occurrences)) == (12, 13, 712, 910),
        "Structure table counts disagree",
    )
    require(
        zones.structure_type.eq("boundary_zone").all()
        and cores.structure_type.eq("sequence_core_candidate").all(),
        "Unsupported structure type; candidates are not instances",
    )
    for frame in (zones, cores, counts):
        unique_nonempty(frame, "structure_id")
        integer_columns(frame, ("start_index", "end_index"))
    unique_nonempty(zones, "region_id")
    unique_nonempty(occurrences, "frame_id")
    unique_nonempty(indices, "nominal_index")
    integer_columns(indices, ("nominal_index", "start_index", "end_index"))
    require(
        set(indices.nominal_index) == set(range(1, 713)),
        "Nominal index coverage must be exactly 1..712",
    )
    integer_columns(
        counts,
        (
            "unique_index_count",
            "occurrence_count",
            "unique_content_count",
            "cross_split_exact_content_count",
        ),
    )
    columns = ["structure_id", "structure_type", "start_index", "end_index"]
    elements = pd.concat([cores[columns], zones[columns]], ignore_index=True)
    unique_nonempty(elements, "structure_id")
    require(
        elements.start_index.le(elements.end_index).all(), "Reversed structure interval"
    )
    rows = []
    region_ids = zones.set_index("structure_id").region_id.to_dict()
    for item in elements.to_dict("records"):
        rows.extend(
            {
                "nominal_index": i,
                **item,
                "review_region_id": region_ids.get(item["structure_id"], ""),
            }
            for i in range(item["start_index"], item["end_index"] + 1)
        )
    expected = pd.DataFrame(rows)[list(indices)]
    require(
        len(expected) == 712 and expected.nominal_index.is_unique,
        "Structure intervals overlap or leave gaps",
    )
    equal_sorted(
        indices,
        expected,
        "nominal_index",
        "Index assignments disagree with structure intervals/regions",
    )
    require(
        occurrences.structure_type.value_counts().to_dict()
        == {"sequence_core_candidate": 870, "boundary_zone": 40},
        "Expected 870 core and 40 zone occurrences",
    )
    require(
        set(occurrences.nominal_index) == set(indices.nominal_index),
        "Incomplete occurrence index coverage",
    )
    assigned = occurrences[list(indices)].copy()
    assigned["review_region_id"] = assigned.review_region_id.fillna("")
    joined = occurrences[["nominal_index"]].merge(
        indices, on="nominal_index", how="left", validate="many_to_one"
    )
    equal_sorted(
        assigned,
        joined[list(indices)],
        list(indices),
        "Occurrence structure assignments disagree with index_structure",
    )
    require(
        occurrences.groupby("content_id").structure_id.nunique().gt(1).sum()
        == s["cross_structure_exact_content_count"],
        "Content crosses structure elements",
    )
    require(
        int(occurrences.groupby("content_id").original_split.nunique().gt(1).sum())
        == s["cross_split_exact_content_count"],
        "Cross-split exact-content counts disagree",
    )
    equal_sorted(
        counts[columns],
        elements,
        "structure_id",
        "Structure summary element identities disagree",
    )
    for row in counts.itertuples():
        group = occurrences.loc[occurrences.structure_id.eq(row.structure_id)]
        require(
            (
                group.nominal_index.nunique(),
                len(group),
                group.content_id.nunique(),
                int(group.groupby("content_id").original_split.nunique().gt(1).sum()),
            )
            == (
                row.unique_index_count,
                row.occurrence_count,
                row.unique_content_count,
                row.cross_split_exact_content_count,
            ),
            "Structure summary counts disagree with occurrences",
        )
    # Historical split counts remain opaque legacy text: no undocumented parser
    # for a Python-dict/JSON-like cell, and never a fitting label.


def unordered_pairs(frame):
    require(
        frame.core_a.str.strip().ne("").all()
        and frame.core_b.str.strip().ne("").all()
        and frame.core_a.ne(frame.core_b).all(),
        "Empty core ID or recurrence self-pair",
    )
    result = [
        tuple(sorted(pair))
        for pair in frame[["core_a", "core_b"]].itertuples(index=False, name=None)
    ]
    require(len(set(result)) == len(result), "Duplicate/reversed recurrence pair")
    return result


def validate_recurrence(report):
    s, t = report.summary, report.tables
    exact_counts(
        s,
        {"core_count": 13, "pair_count": 78, "manual_review_candidate_count": 11},
        "recurrence",
    )
    require(
        s.get("selection_rule")
        == {"clip_symmetric_ge_090_min": 0.2, "dinov2_symmetric_ge_090_min": 0.05},
        "Unsupported legacy recurrence selection rule",
    )
    require(
        s.get("ranking")
        == {
            "method": "encoder_rank_consensus",
            "primary": "rank_sum",
            "secondary": "joint_symmetric_ge_090",
        },
        "Unsupported legacy recurrence ranking",
    )
    pairs, candidates = t["all_pairs_ranked"], t["manual_review_candidates"]
    require((len(pairs), len(candidates)) == (78, 11), "Recurrence row counts disagree")
    keys, selected = unordered_pairs(pairs), unordered_pairs(candidates)
    cores = set(pairs.core_a) | set(pairs.core_b)
    require(
        len(cores) == 13 and set(keys) == set(itertools.combinations(sorted(cores), 2)),
        "Recurrence must cover all C(13,2) unordered pairs",
    )
    require(set(selected) <= set(keys), "Candidate pair absent from all_pairs_ranked")
    for frame in (pairs, candidates):
        for name in PAIR_COLUMNS[2:]:
            values = pd.to_numeric(frame[name], errors="raise")
            require(
                np.isfinite(values).all(), f"Invalid recurrence measurement: {name}"
            )
            frame[name] = values.astype(float)
        # The observed schema does not specify rank origin or tie handling.
        # Preserve producer ranks rather than imposing a new ranking convention.
    integer_columns(candidates, ("review_order",))
    require(candidates.review_order.is_unique, "Duplicate recurrence review order")
    lookup = dict(zip(keys, range(len(pairs)), strict=True))
    # Match unordered IDs, but retain the producer's original orientation/ranks.
    for index, key in enumerate(selected):
        require(
            np.allclose(
                candidates.iloc[index][PAIR_COLUMNS[2:]].to_numpy(dtype=float),
                pairs.iloc[lookup[key]][PAIR_COLUMNS[2:]].to_numpy(dtype=float),
                rtol=0,
                atol=1e-12,
            ),
            "Candidate measurements/ranks disagree with all_pairs_ranked",
        )


def validate_bundle(found, family, source=None):
    require(
        set(found) == set(ROLES)
        and all(isinstance(v, NativeReport) for v in found.values()),
        "Cannot mix native and normalized producer contracts",
    )
    by_role = {ROLES[k]: v for k, v in found.items()}
    for report in found.values():
        if "dataset_id" in report.summary:
            require(
                isinstance(report.summary["dataset_id"], str)
                and bool(report.summary["dataset_id"].strip()),
                "Invalid declared dataset identity",
            )
    declared_datasets = {
        r.summary["dataset_id"] for r in found.values() if "dataset_id" in r.summary
    }
    require(len(declared_datasets) <= 1, "Cross-artifact dataset identity conflict")
    for report in found.values():
        if "query_family" in report.summary:
            require(
                report.summary["query_family"] == family,
                "Cross-artifact query family conflict",
            )
        for key in ("dataset_variant", "variant_name"):
            if key in report.summary:
                require(
                    report.summary[key] == "unspecified",
                    "Legacy evidence variant must remain unspecified",
                )
        if "dataset_variant_id" in report.summary:
            require(
                source is not None
                and report.summary["dataset_variant_id"]
                == source.signature["dataset_variant_id"],
                "Unestablished legacy dataset variant identity; explicit migration evidence is required",
            )
    require(
        by_role["manual"].summary.get("query_family") == family
        and by_role["structure"].summary.get("query_family") == family,
        "Manual review and structure must declare the query family",
    )
    # This is a declared family key in the provenance JSON, not a filename guess.
    require(
        named_object(by_role["provenance"].summary, family).get("status")
        == "unresolved",
        "Cross-artifact provenance family conflict",
    )
    manual = by_role["manual"].tables["supported_boundary_zones"]
    structure = by_role["structure"].tables
    equal_sorted(
        manual,
        structure["boundary_zones"][list(manual)],
        "region_id",
        "Cross-artifact boundary zones disagree",
    )
    cores = set(structure["sequence_core_candidates"].structure_id)
    pairs = by_role["recurrence"].tables["all_pairs_ranked"]
    require(
        set(pairs.core_a) | set(pairs.core_b) == cores,
        "Recurrence core IDs disagree with sequence_core_candidates",
    )
    require(
        len(pairs) == len(cores) * (len(cores) - 1) // 2,
        "Cross-artifact recurrence pair count conflict",
    )
    occurrences = structure["occurrence_structure"]
    if source is not None:
        require(
            source.family == family
            and source.signature["variant_name"] == "unspecified",
            "Legacy native evidence requires the unspecified variant and matching family",
        )
        records = source.records
        require(
            set(records.frame_id) == set(occurrences.frame_id)
            and records.frame_id.is_unique,
            "Native occurrences do not cover the selected canonical source",
        )
        equal_sorted(
            occurrences[["frame_id", "content_id"]],
            records[["frame_id", "content_id"]],
            "frame_id",
            "Native content/occurrence identity differs from canonical source",
        )
        for name in (
            "image_sha256",
            "label_sha256",
            "manifest_version",
            "original_split",
        ):
            if name in records:
                equal_sorted(
                    occurrences[["frame_id", name]],
                    records[["frame_id", name]],
                    "frame_id",
                    f"Native {name} differs from canonical source metadata",
                )
        equal_sorted(
            occurrences[["frame_id", "nominal_index"]].rename(
                columns={"nominal_index": "position"}
            ),
            records[["frame_id", "position"]],
            "frame_id",
            "Native nominal indices disagree with source temporal metadata",
        )
        require(
            records.timeline_id.eq(family).all(),
            "Native evidence must map to exactly its declared timeline",
        )
        for report in found.values():
            if "dataset_id" in report.summary:
                require(
                    report.summary["dataset_id"] == source.signature["dataset_id"],
                    "Conflicting native dataset identity",
                )
        source.unchanged()
    for report in found.values():
        unchanged(report)
    return {
        "adapter_schema_version": ADAPTER,
        "query_family": family,
        "dataset_id": source.signature["dataset_id"]
        if source
        else next(iter(declared_datasets), None),
        "dataset_variant": "unspecified",
        "dataset_variant_id": source.signature["dataset_variant_id"]
        if source
        else None,
        "canonical_source_verified": source is not None,
        "occurrence_count": len(occurrences),
        "nominal_index_count": occurrences.nominal_index.nunique(),
        "core_candidate_count": len(cores),
        "boundary_zone_count": len(manual),
        "recurrence_pair_count": len(pairs),
        "manual_review_candidate_count": 11,
        "legacy_review_modes": sorted(
            by_role["manual"].tables["transition_decisions"].review_mode.unique()
        ),
        "ground_truth": False,
        "automatic_confirmation": False,
        "split_created": False,
        "exact_boundary_known": False,
        "visual_verification_complete": False,
        "sequence_instances_created": False,
        "visual_dependency_groups_created": False,
    }


def normalized_tables(found, family):
    structure = found["video11_sequence_structure_v1"].tables
    intervals = []
    for name, kind, decision in (
        ("boundary_zones", "boundary_zone", "supported"),
        ("sequence_core_candidates", "sequence_core_candidate", "candidate"),
    ):
        for row in structure[name].itertuples():
            intervals.append(
                dict(
                    element_id=row.structure_id,
                    timeline_id=family,
                    start=int(row.start_index),
                    end=int(row.end_index),
                    kind=kind,
                    decision=decision,
                    notes="Imported legacy evidence; exact boundary unknown; non-ground-truth",
                )
            )
    frame = (
        pd.DataFrame(intervals, columns=list(Interval.model_fields))
        .sort_values(["timeline_id", "kind", "start", "element_id"])
        .reset_index(drop=True)
    )
    result = {
        "intervals": frame,
        "observations": pd.DataFrame(
            columns=[*list(Observation.model_fields)[:-2], "notes", "measurements_json"]
        ),
    }
    # Pair ranks/provenance stay in their original evidence tables/snapshots; no
    # new lineage assignments or confirmed recurrence observations are inferred.
    for kind, report in found.items():
        for name, table in report.tables.items():
            result[f"native_{ROLES[kind]}_{name}"] = table.copy()
    return result


def native_producers(found):
    return {
        kind: {
            "source_artifact_path": str(report.path.parent),
            "source_artifact_family": kind,
            "summary_filename": report.path.name,
            "source_files": report.checksums,
            "adapter_schema_version": ADAPTER,
            "query_family": report.summary.get("query_family"),
            "snapshots": {
                name: f"media/{ROLES[kind]}/{name}" for name in report.snapshots
            },
        }
        for kind, report in found.items()
    }


def import_native(found, source, output, *, dry_run=False):
    validation = validate_bundle(found, source.family, source)
    for report in found.values():
        source.safe_output(output, report.path.parent)
    if dry_run:
        return {
            **validation,
            "dry_run": True,
            "published": False,
            "producers": native_producers(found),
        }
    result = normalized_tables(found, source.family)
    validate_intervals(result["intervals"], source)
    result["membership"], result["cores"] = structure_membership(
        source, result["intervals"]
    )
    result["occurrences"] = source.records.copy()
    for report in found.values():
        unchanged(report)
    source.unchanged()
    return publish(
        Path(output),
        "sequence_structure_review_v1",
        {"adapter": ADAPTER, "exhaustive_boundary_review": False},
        {"input": source.signature, "native_producers": native_producers(found)},
        result,
        {
            **validation,
            "interval_count": len(result["intervals"]),
            "core_count": len(result["cores"]),
            "producer_kind": "native_legacy_bundle",
            "review_semantics": "legacy manual-review evidence, not independently verified visual ground truth",
        },
        extras={
            "native_summaries": {kind: report.summary for kind, report in found.items()}
        },
        media={
            f"media/{ROLES[kind]}/{name}": raw
            for kind, report in found.items()
            for name, raw in report.snapshots.items()
        },
    )


def verify_native(directory, source):
    """Rehash ORIGINAL files, check frozen snapshots, and replay normalization."""
    meta = inspect(directory)
    require(
        meta["identity"]["config"].get("adapter") == ADAPTER,
        "Unsupported native adapter revision",
    )
    producers = meta["identity"]["sources"]["native_producers"]
    require(set(producers) == set(ROLES), "Incomplete native producer inventory")
    found = {}
    for kind, record in producers.items():
        require(
            record["adapter_schema_version"] == ADAPTER
            and record["source_artifact_family"] == kind,
            "Native producer schema identity conflict",
        )
        root = Path(record["source_artifact_path"])
        for name, digest in record["source_files"].items():
            require(
                file_sha256(declared_file(root, name)) == digest,
                f"Original consumed source changed: {root / name}",
            )
            require(
                file_sha256(declared_file(directory, record["snapshots"][name]))
                == digest,
                "Frozen native snapshot changed",
            )
        report = read_native(declared_file(root, record["summary_filename"]))
        require(
            report.kind == kind and report.checksums == record["source_files"],
            "Original native producer identity/inventory changed",
        )
        found[kind] = report
    validation = validate_bundle(found, source.family, source)
    require(native_producers(found) == producers, "Native producer declaration changed")
    require(
        read_json(directory / "native_summaries.json")
        == {k: r.summary for k, r in found.items()},
        "Normalized native summaries differ",
    )
    for name, expected in normalized_tables(found, source.family).items():
        pd.testing.assert_frame_equal(
            expected, pd.read_parquet(directory / f"{name}.parquet")
        )
    summary = read_json(directory / "summary.json")
    require(
        all(summary.get(k) == v for k, v in validation.items()),
        "Normalized native validation summary differs",
    )
