"""Source-bound publication and replay of immutable manual-review aggregates."""

from dataclasses import asdict, fields
from pathlib import Path

import pandas as pd

from flir_pipeline.linkage.review_aggregation import (
    AGGREGATE_SEMANTICS,
    FLAGS,
    KIND,
    aggregate_tables,
)
from flir_pipeline.linkage.review_sources import ReviewPaths, artifact_files
from flir_pipeline.linkage.review_storage import (
    _safe_output,
    inspect_snapshot,
    verify_review,
)
from flir_pipeline.similarity.storage import (
    execution_provenance,
    file_sha256,
    read_json,
    stable_id,
    write_json,
)

TABLES = (
    "source_observations",
    "pooled_reviews",
    "duplicate_pairs",
    "descriptive_counts",
)
OUTPUT_FILES = {
    *(f"{name}.parquet" for name in TABLES),
    "source_revisions.json",
    "source_locations.json",
    "summary.json",
}


def _paths(payload, base):
    """Paths locate sources; only verified bytes determine scientific identity."""
    names = {field.name for field in fields(ReviewPaths)}
    if not isinstance(payload, dict) or set(payload) != names:
        raise ValueError(f"Source map requires all ReviewPaths fields: {sorted(names)}")
    if not all(isinstance(value, str) and value.strip() for value in payload.values()):
        raise ValueError(
            "Source map paths and membership_table must be nonempty strings"
        )
    return ReviewPaths(
        **{
            key: value if key == "membership_table" else (base / value).resolve()
            for key, value in payload.items()
        }
    )


def load_source_map(path):
    data = read_json(path)
    if (
        not isinstance(data, dict)
        or set(data) != {"calibrations"}
        or not isinstance(data["calibrations"], dict)
    ):
        raise ValueError(
            "Source map requires a calibrations object keyed by calibration_id"
        )
    return {
        key: _paths(value, path.parent) for key, value in data["calibrations"].items()
    }


def _collect(directories, bindings):
    if not directories:
        raise ValueError("Provide at least one immutable review revision")
    revisions = []
    seen = set()
    for directory in directories:
        metadata_sha = file_sha256(directory / "metadata.json")
        meta, review, history = inspect_snapshot(directory)
        revision_id, calibration_id = meta["artifact_id"], meta["calibration_id"]
        if revision_id in seen:
            raise ValueError(f"Source revision supplied more than once: {revision_id}")
        seen.add(revision_id)
        if calibration_id not in bindings:
            raise ValueError(
                f"Missing source paths for calibration_id={calibration_id}"
            )
        checks = artifact_files(directory, meta)
        quality = verify_review(directory, bindings[calibration_id])
        if (
            quality.get("quality_valid") is not True
            or quality.get("source_bound") is not True
        ):
            raise ValueError(
                f"Source revision {revision_id} failed review verification: {quality}"
            )
        if (
            artifact_files(directory, meta) != checks
            or checks["metadata.json"] != metadata_sha
        ):
            raise ValueError(
                f"Source revision changed during verification: {revision_id}"
            )
        revisions.append(
            {
                "directory": directory.resolve(),
                "metadata": meta,
                "review": review,
                "history": history,
                "checksums": checks,
            }
        )
    return sorted(revisions, key=lambda item: item["metadata"]["artifact_id"])


def _identity(revisions):
    return {
        "artifact_kind": KIND,
        "artifact_version": 1,
        "semantics": AGGREGATE_SEMANTICS,
        "sources": [
            {
                "source_revision_id": item["metadata"]["artifact_id"],
                "calibration_id": item["metadata"]["calibration_id"],
                "revision_checksums": item["checksums"],
            }
            for item in revisions
        ],
    }


def _source_history(revisions):
    return [
        {
            "source_revision_id": item["metadata"]["artifact_id"],
            "calibration_id": item["metadata"]["calibration_id"],
            "metadata": item["metadata"],
            "decision_history": item["history"],
        }
        for item in revisions
    ]


def _locations(revisions, bindings):
    calibration_ids = sorted({item["metadata"]["calibration_id"] for item in revisions})
    return {
        "revisions": {
            item["metadata"]["artifact_id"]: str(item["directory"])
            for item in revisions
        },
        "calibrations": {
            key: {
                name: str(value if name == "membership_table" else value.resolve())
                for name, value in asdict(bindings[key]).items()
            }
            for key in calibration_ids
        },
    }


def aggregate_reviews(directories, bindings, output):
    """Publish a new aggregate; no constituent file or decision is written."""
    revisions = _collect(directories, bindings)
    tables, summary = aggregate_tables(revisions)
    identity = _identity(revisions)
    directory = output / stable_id(identity)
    protected = [*directories]
    for paths in bindings.values():
        protected.extend(
            value for value in asdict(paths).values() if isinstance(value, Path)
        )
    _safe_output(directory, protected)
    if directory.exists():
        quality = verify_aggregate(directory)
        if not quality["quality_valid"]:
            raise ValueError(
                f"Existing aggregate is incomplete/inconsistent: {quality}"
            )
        return directory
    directory.mkdir(parents=True, exist_ok=False)
    for name, frame in tables.items():
        frame.to_parquet(directory / f"{name}.parquet", index=False)
    write_json(directory / "source_revisions.json", _source_history(revisions))
    write_json(directory / "source_locations.json", _locations(revisions, bindings))
    write_json(directory / "summary.json", summary)
    # Metadata is the completion marker. Do not publish it if a constituent
    # changed while its derived tables were being written.
    for item in revisions:
        if artifact_files(item["directory"], item["metadata"]) != item["checksums"]:
            raise ValueError("Source revision changed during aggregate publication")
    execution = execution_provenance()
    execution["source_sha256"] = {
        path.name: file_sha256(path)
        for path in sorted(Path(__file__).parent.glob("review_*.py"))
    }
    write_json(
        directory / "metadata.json",
        {
            "artifact_kind": KIND,
            "artifact_version": 1,
            "artifact_id": stable_id(identity),
            "identity": identity,
            "semantics": AGGREGATE_SEMANTICS,
            **dict.fromkeys(FLAGS, False),
            "output_checksums": {
                name: file_sha256(directory / name) for name in sorted(OUTPUT_FILES)
            },
            "execution": execution,
        },
    )
    return directory


def verify_aggregate(directory, *, locations=None):
    """Reverify original reviews/sources, replay history and reconstruct all counts."""
    result = {
        "quality_valid": False,
        "source_bound": False,
        **dict.fromkeys(FLAGS, False),
    }
    try:
        meta = read_json(directory / "metadata.json")
        if (
            not isinstance(meta, dict)
            or meta.get("artifact_kind") != KIND
            or type(meta.get("artifact_version")) is not int
            or meta["artifact_version"] != 1
            or meta.get("semantics") != AGGREGATE_SEMANTICS
            or any(meta.get(key) is not False for key in FLAGS)
            or meta.get("artifact_id") != stable_id(meta["identity"])
            or set(meta["output_checksums"]) != OUTPUT_FILES
        ):
            raise ValueError("Invalid aggregate identity/semantics/output contract")
        artifact_files(directory, meta)
        if {
            path.relative_to(directory).as_posix()
            for path in directory.rglob("*")
            if path.is_file()
        } != {*OUTPUT_FILES, "metadata.json"}:
            raise ValueError("Incomplete aggregate publication or undeclared files")
        location_file = locations or directory / "source_locations.json"
        located = read_json(location_file)
        if (
            not isinstance(located, dict)
            or set(located) != {"revisions", "calibrations"}
            or not isinstance(located["revisions"], dict)
            or not isinstance(located["calibrations"], dict)
        ):
            raise ValueError("Locations require revisions and calibrations maps")
        expected = meta["identity"]["sources"]
        if set(located["revisions"]) != {
            item["source_revision_id"] for item in expected
        } or set(located["calibrations"]) != {
            item["calibration_id"] for item in expected
        }:
            raise ValueError(
                "Source locations do not cover exactly the bound revisions/calibrations"
            )
        bindings = {
            key: _paths(value, location_file.parent)
            for key, value in located["calibrations"].items()
        }
        directories = []
        for revision_id, path in sorted(located["revisions"].items()):
            if not isinstance(path, str) or not path.strip():
                raise ValueError("Revision locations must be nonempty path strings")
            source = (location_file.parent / path).resolve()
            if read_json(source / "metadata.json").get("artifact_id") != revision_id:
                raise ValueError(
                    f"Source location does not match revision {revision_id}"
                )
            directories.append(source)
        revisions = _collect(directories, bindings)
        if _identity(revisions) != meta["identity"]:
            raise ValueError("Aggregate differs from its exact source revision bytes")
        tables, summary = aggregate_tables(revisions)
        for name, expected_table in tables.items():
            pd.testing.assert_frame_equal(
                pd.read_parquet(directory / f"{name}.parquet"),
                expected_table,
                check_exact=True,
            )
        if read_json(directory / "summary.json") != summary:
            raise ValueError("Aggregate summary differs from replayed source decisions")
        if read_json(directory / "source_revisions.json") != _source_history(revisions):
            raise ValueError(
                "Aggregate provenance differs from source revision history"
            )
        result.update(
            quality_valid=True,
            source_bound=True,
            constituent_reviews_reverified=True,
            decisions_replayed=True,
            source_revision_count=len(revisions),
            unique_query_group_count=len(tables["pooled_reviews"]),
        )
    except (
        ValueError,
        OSError,
        KeyError,
        TypeError,
        AssertionError,
        IndexError,
    ) as error:
        result["error"] = f"{type(error).__name__}: {error}"
    return result
