"""Immutable manual calibration revisions with replayable decision imports."""

import shutil
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import PIL

from flir_pipeline.data.local_images import declared_file
from flir_pipeline.linkage.review_media import context_plan, render_sheets
from flir_pipeline.linkage.review_model import (
    KIND,
    MANUAL_COLUMNS,
    SEMANTICS,
    ReviewConfig,
    apply_decisions,
    canonical_json,
    decision_summary,
    event_identity,
    read_csv,
    validate_decisions,
)
from flir_pipeline.linkage.review_sources import (
    ReviewPaths,
    artifact_files,
    review_evidence,
)
from flir_pipeline.similarity.storage import (
    execution_provenance,
    file_sha256,
    read_json,
    stable_id,
    write_json,
)

TABLES = (
    "candidates",
    "candidate_occurrences",
    "labeled_occurrences",
    "calibration_sample",
    "visual_dependency_membership",
    "temporal_context",
)
STATIC_FILES = (
    "initial_review.csv",
    "index.html",
    *(f"{name}.parquet" for name in TABLES),
)


def plan_review(paths, config):
    from flir_pipeline.linkage.review_sources import load_review_sources

    sample, labeled, sequences, candidates, occurrences, membership, signature = (
        load_review_sources(paths)
    )
    calibration = {
        "artifact_kind": KIND,
        "artifact_version": 1,
        "configuration": config.model_dump(mode="json"),
        "sources": signature,
        "pillow_version": PIL.__version__,
    }
    calibration_id = stable_id(calibration)
    review, sequences, tables = review_evidence(
        sample, labeled, sequences, candidates, occurrences, membership, calibration_id
    )
    context = context_plan(
        review, tables["candidates"], tables["candidate_occurrences"], sequences, config
    )
    review["contact_sheets_json"] = review.review_query_id.map(
        context.groupby("review_query_id").contact_sheet.agg(
            lambda paths: canonical_json(sorted(set(paths)))
        )
    )
    tables["temporal_context"] = context
    validate_decisions(review)
    return calibration, review, tables


def _identity(calibration, static_checksums, history):
    return {
        "calibration": calibration,
        "static_checksums": static_checksums,
        "decision_event_ids": [event["event_id"] for event in history],
    }


def _safe_output(root, protected):
    root = root.resolve()
    repo = Path(__file__).resolve().parents[3]
    if root.is_relative_to(repo) and not root.is_relative_to(repo / "reports"):
        raise ValueError(
            "Review images and CSVs inside the repository must stay under ignored reports/"
        )
    if any(
        root.is_relative_to(p.resolve()) or p.resolve().is_relative_to(root)
        for p in protected
    ):
        raise ValueError(
            "Review output must be separate from all immutable source artifacts/images"
        )
    if any((parent / "metadata.json").is_file() for parent in root.parents):
        raise ValueError(
            "Review output cannot be nested inside an existing publication"
        )


def _publish_metadata(directory, calibration, static_checksums, history, review):
    if any(
        file_sha256(declared_file(directory, name)) != expected
        for name, expected in static_checksums.items()
    ):
        raise ValueError("Static review evidence changed before publication")
    review.to_csv(directory / "review.csv", index=False, lineterminator="\n")
    write_json(directory / "decision_history.json", history)
    write_json(directory / "summary.json", decision_summary(review))
    identity = _identity(calibration, static_checksums, history)
    outputs = [
        *static_checksums,
        "review.csv",
        "decision_history.json",
        "summary.json",
        *(f"imports/{event['event_id']}.csv" for event in history),
    ]
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
            "calibration_id": stable_id(calibration),
            "identity": identity,
            "semantics": SEMANTICS,
            "ground_truth": False,
            "confirmed_matches_created": False,
            "split_created": False,
            "output_checksums": {
                name: file_sha256(directory / name) for name in outputs
            },
            "execution": execution,
        },
    )


def inspect_snapshot(directory):
    """Validate static evidence and replay explicit decisions; never infer them."""
    meta = read_json(directory / "metadata.json")
    identity = meta["identity"]
    if (
        meta.get("artifact_kind") != KIND
        or meta.get("artifact_version") != 1
        or meta.get("semantics") != SEMANTICS
        or any(
            meta.get(key) is not False
            for key in ("ground_truth", "confirmed_matches_created", "split_created")
        )
        or meta.get("calibration_id") != stable_id(identity["calibration"])
        or meta.get("artifact_id") != stable_id(identity)
    ):
        raise ValueError("Invalid manual calibration identity/semantics")
    ReviewConfig.model_validate(identity["calibration"]["configuration"])
    checksums = artifact_files(directory, meta)
    history = read_json(directory / "decision_history.json")
    static = identity["static_checksums"]
    context = pd.read_parquet(directory / "temporal_context.parquet")
    if set(static) != {*STATIC_FILES, *context.contact_sheet.unique()} or any(
        checksums[name] != digest for name, digest in static.items()
    ):
        raise ValueError("Immutable evidence or contact sheet checksum mismatch")
    expected_files = {
        *static,
        "metadata.json",
        "review.csv",
        "decision_history.json",
        "summary.json",
        *(f"imports/{event['event_id']}.csv" for event in history),
    }
    if (
        set(checksums) != expected_files
        or {
            p.relative_to(directory).as_posix()
            for p in directory.rglob("*")
            if p.is_file()
        }
        != expected_files
    ):
        raise ValueError("Incomplete publication or undeclared review files")
    initial = read_csv(directory / "initial_review.csv")
    validate_decisions(initial)
    if not initial[MANUAL_COLUMNS].eq("").all().all():
        raise ValueError(
            "Initial review must be blank; similarity cannot supply decisions"
        )
    review = initial.copy()
    replayed = []
    for event in history:
        parent_id = stable_id(_identity(identity["calibration"], static, replayed))
        imported = declared_file(directory, f"imports/{event['event_id']}.csv")
        updated, changes = apply_decisions(
            review,
            read_csv(imported),
            reviewer=event["reviewer"],
            source=event["source"],
            timestamp=event["recorded_at_utc"],
        )
        expected = event_identity(
            parent_id,
            file_sha256(imported),
            event["reviewer"],
            event["source"],
            event["recorded_at_utc"],
            changes,
        )
        if not changes or event != expected:
            raise ValueError(
                "Manual decision history cannot be replayed against its import"
            )
        replayed.append(expected)
        review = updated
    if identity != _identity(identity["calibration"], static, replayed):
        raise ValueError("Decision event identity chain mismatch")
    pd.testing.assert_frame_equal(
        read_csv(directory / "review.csv"), review, check_exact=True
    )
    if read_json(directory / "summary.json") != decision_summary(review):
        raise ValueError(
            "Manual calibration summary differs from query-level decisions"
        )
    return meta, review, history


def init_review(
    paths: ReviewPaths,
    config: ReviewConfig,
    output: Path,
    *,
    labeled_images_archive=None,
    labeled_images_root=None,
    video_images_root,
):
    protected = [
        paths.calibration_sample,
        paths.linkage,
        paths.labeled_manifest,
        paths.sequence_set,
        paths.visual_dependencies,
        video_images_root,
        *(p for p in (labeled_images_archive, labeled_images_root) if p is not None),
    ]
    _safe_output(output, protected)
    calibration, review, tables = plan_review(paths, config)
    # Source-based folder name allows deterministic reuse before rendering; the
    # publication identity additionally binds actual rendered bytes and decisions.
    directory = output / stable_id(calibration)
    if directory.exists():
        result = verify_review(directory, paths)
        if not result["quality_valid"]:
            raise ValueError(f"Existing review is incomplete/inconsistent: {result}")
        return directory
    directory.mkdir(parents=True, exist_ok=False)
    for name, frame in tables.items():
        frame.to_parquet(directory / f"{name}.parquet", index=False)
    review.to_csv(directory / "initial_review.csv", index=False, lineterminator="\n")
    render_sheets(
        directory,
        review,
        tables["temporal_context"],
        tables["labeled_occurrences"],
        labeled_images_archive=labeled_images_archive,
        labeled_images_root=labeled_images_root,
        video_images_root=video_images_root,
    )
    static = {
        name: file_sha256(directory / name)
        for name in [
            *STATIC_FILES,
            *sorted(tables["temporal_context"].contact_sheet.unique()),
        ]
    }
    # Rebind the original files after rendering; concurrent source changes cannot
    # acquire a completed metadata marker. Partial output remains preserved.
    if plan_review(paths, config)[0] != calibration:
        raise ValueError("Review sources changed during rendering")
    _publish_metadata(directory, calibration, static, [], review)
    inspect_snapshot(directory)
    return directory


def record_review(
    directory, decisions_path, *, reviewer, source, timestamp=None, output=None
):
    meta, review, history = inspect_snapshot(directory)
    imported_bytes = decisions_path.read_bytes()
    import_sha = file_sha256(decisions_path)
    timestamp = timestamp or datetime.now(UTC).isoformat()
    updated, changes = apply_decisions(
        review,
        read_csv(decisions_path),
        reviewer=reviewer,
        source=source,
        timestamp=timestamp,
    )
    if decisions_path.read_bytes() != imported_bytes:
        raise ValueError("Decision import changed during reading")
    if not changes:
        return directory
    event = event_identity(
        meta["artifact_id"], import_sha, reviewer, source, timestamp, changes
    )
    identity = _identity(
        meta["identity"]["calibration"],
        meta["identity"]["static_checksums"],
        [*history, event],
    )
    output = output or directory.parent
    destination = output / stable_id(identity)
    _safe_output(destination, [directory, decisions_path])
    if destination.exists():
        previous, _, _ = inspect_snapshot(destination)
        if previous["identity"] != identity:
            raise ValueError("Existing review revision identity differs")
        return destination
    destination.mkdir(parents=True, exist_ok=False)
    for name in [
        *meta["identity"]["static_checksums"],
        *(f"imports/{prior['event_id']}.csv" for prior in history),
    ]:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(declared_file(directory, name), target)
    target = destination / "imports" / f"{event['event_id']}.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(imported_bytes)
    if file_sha256(target) != import_sha:
        raise ValueError("Imported decisions no longer match their provenance")
    _publish_metadata(
        destination,
        identity["calibration"],
        identity["static_checksums"],
        [*history, event],
        updated,
    )
    inspect_snapshot(destination)
    return destination


def verify_review(directory, paths: ReviewPaths):
    result = {
        "quality_valid": False,
        "source_bound": False,
        "ground_truth": False,
        "confirmed_matches_created": False,
        "split_created": False,
    }
    try:
        meta, _, _ = inspect_snapshot(directory)
        config = ReviewConfig.model_validate(
            meta["identity"]["calibration"]["configuration"]
        )
        calibration, review, tables = plan_review(paths, config)
        if calibration != meta["identity"]["calibration"]:
            raise ValueError(
                "Manual calibration differs from its bound source artifacts"
            )
        pd.testing.assert_frame_equal(
            read_csv(directory / "initial_review.csv"), review, check_exact=True
        )
        for name, expected in tables.items():
            pd.testing.assert_frame_equal(
                pd.read_parquet(directory / f"{name}.parquet"),
                expected,
                check_exact=True,
            )
        result.update(
            quality_valid=True,
            source_bound=True,
            decisions_replayed=True,
            all_candidate_occurrences_preserved=True,
            one_decision_per_query=True,
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


def summarize_review(directory):
    _, review, _ = inspect_snapshot(directory)
    return decision_summary(review)
