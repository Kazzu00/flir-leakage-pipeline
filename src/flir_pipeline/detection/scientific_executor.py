"""Signed handoff of existing assignments. No split generation or model fitting here."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import yaml

from flir_pipeline.data.yolo_labels import validate_label_bytes
from flir_pipeline.detection.admission import (
    authorized,
    bound_file,
    check_test_isolation,
    training_admission,
)
from flir_pipeline.detection.common_executor import (
    NAMES,
    external_output,
    load_executor_config,
    records_identity,
    verify_executor_plan,
    verify_preparation,
)
from flir_pipeline.detection.materialization import _write_checked
from flir_pipeline.similarity.storage import (
    file_sha256,
    read_json,
    stable_id,
    write_json,
)


def context(document: Path, signature: Path, plan_path: Path, output: Path) -> dict:
    admission = training_admission(document, signature)
    if bound_file(admission["plan"]) != plan_path / "plan.json":
        raise ValueError("Wrong admitted plan locator")
    if Path(admission["output_root"]).resolve() != output.resolve():
        raise ValueError("Output root is frozen")
    plan = verify_executor_plan(plan_path)
    prepared = bound_file(admission["preparation"]).parent
    receipt = verify_preparation(prepared)
    config = load_executor_config(bound_file(admission["config"]))
    if receipt["source_id"] != plan["source_id"]:
        raise ValueError("Plan/preparation publication mismatch")
    is_ab = admission.get("input_format") == "general_ab_assignments_v1"
    if is_ab:
        from flir_pipeline.detection.general_ab import validate_context_contract

        validate_context_contract(admission, plan, receipt)
    elif "input_contract" in plan or "input_contract" in receipt:
        raise ValueError("Explicit A/B signed admission required")
    selected = []
    for run in plan["runs"]:
        if run["identity"]["configuration"] != config:
            raise ValueError("Frozen proposal differs from planned configuration")
        declared_weights = run["identity"]["initialization"]["sha256"]
        if (
            declared_weights is not None
            and declared_weights != admission["weights"]["sha256"]
        ):
            raise ValueError(
                "Approved weights differ from declared plan initialization"
            )
        aliases = [
            a
            for a in run["aliases"]
            if a["population"] == admission["population"]
            and a["policy"] in admission["policies"]
            and a["split_seed"] in admission["split_seeds"]
        ]
        if not aliases:
            continue
        if len(aliases) != 1 and not is_ab:
            raise ValueError("Primary matrix requires distinct strategy identities")
        identity = run["identity"]["records_id"]
        if (
            identity not in receipt["views"]
            or receipt["views"][identity]["test_id"] != admission["test"]["records_id"]
        ):
            raise ValueError("Approved test or prepared membership differs")
        if any(a["arm"] not in receipt["views"][identity]["aliases"] for a in aliases):
            raise ValueError("Strategy alias not bound to preparation")
        selected.append(
            {
                "candidate_run_id": run["run_id"],
                "records_id": identity,
                "detector_seed": run["identity"]["detector_seed"],
                **aliases[0],
                **({"aliases": aliases} if is_ab else {}),
            }
        )
    expected = {
        (p, s, d)
        for p in admission["policies"]
        for s in admission["split_seeds"]
        for d in admission["detector_seeds"]
    }
    if len(selected) != admission["max_runs"] or (
        not is_ab
        and {(r["policy"], r["split_seed"], r["detector_seed"]) for r in selected}
        != expected
    ):
        raise ValueError("Incomplete or duplicated admitted matrix")
    if is_ab:
        from flir_pipeline.detection.general_ab import validate_selected

        validate_selected(selected, expected)
    config["train"]["batch"] = admission["batch"]
    # Identity binds the signed protocol, including all variants and operational freeze.
    for run in selected:
        run["run_id"] = stable_id({"admission": file_sha256(document), "run": run})
    if admission["variant"]["kind"] == "geometry_preserving":
        mapping = read_json(bound_file(admission["variant"]["mapping"]))
        entries = mapping["records"]
        by_frame = {r["frame_id"]: r for r in entries}
        if len(by_frame) != len(entries) or len({r["path"] for r in entries}) != len(
            entries
        ):
            raise ValueError("Variant mapping is not one-to-one")
        for identity in {r["records_id"] for r in selected}:
            rows = pd.read_parquet(
                prepared / "views" / identity / "assignments.parquet"
            )
            active = rows.loc[rows.new_split != "quarantine"].copy()
            for row in active.itertuples():
                if (
                    row.frame_id not in by_frame
                    or by_frame[row.frame_id]["original_sha256"] != row.image_sha256
                ):
                    raise ValueError(
                        "Full paired population requires original hash correspondence"
                    )
            active["transformed_sha256"] = active.frame_id.map(
                lambda f: by_frame[f]["sha256"]
            )
            if active.groupby("transformed_sha256").new_split.nunique().gt(1).any():
                raise ValueError("Transformed exact content crosses active partitions")
    external_output(
        output, plan_path, prepared, document.parent, bound_file(admission["weights"])
    )
    return {
        "admission": admission,
        "admission_sha256": file_sha256(document),
        "prepared": prepared,
        "config": config,
        "runs": selected,
        **({"input_contract": plan["input_contract"]} if is_ab else {}),
    }


def variant_rows(rows: pd.DataFrame, variant: dict) -> pd.DataFrame:
    """Only an attributed, bijective identity-geometry mapping can reuse labels.

    Equal dimensions are checked but do not establish geometry preservation;
    the signed review must attest no crop, warp, resize or correspondence change.
    """
    if variant["kind"] == "original":
        return rows.copy()
    from PIL import Image

    mapping = read_json(bound_file(variant["mapping"]))
    if mapping["variant_id"] != variant["id"] or mapping["geometry"] != "identity_1to1":
        raise ValueError("Variant identity/geometry mismatch")
    entries = mapping["records"]
    by_id = {r["frame_id"]: r for r in entries}
    if len(by_id) != len(entries) or len({r["path"] for r in entries}) != len(entries):
        raise ValueError("Variant mapping is not one-to-one")
    result = rows.copy()
    for idx, row in result.iterrows():
        target = by_id.get(row.frame_id)
        if not target or target["original_sha256"] != row.image_sha256:
            raise ValueError("Missing or mismatched original variant correspondence")
        path = bound_file(target)
        with Image.open(row.image_path) as original, Image.open(path) as transformed:
            if original.size != transformed.size:
                raise ValueError("Geometry changed; original labels cannot be reused")
        result.loc[idx, "original_image_sha256"] = row.image_sha256
        result.loc[idx, "image_sha256"] = target["sha256"]
        result.loc[idx, "image_path"] = str(path)
    if result.groupby("image_sha256").new_split.nunique().gt(1).any():
        raise ValueError("Transformed exact content crosses partitions")
    return result


def write_view(rows: pd.DataFrame, output: Path, strict_geometry: bool = True) -> Path:
    """Isolated writable runtime view; immutable prepare output never receives caches."""
    if output.exists() or not rows.frame_id.is_unique or not rows.image_path.is_unique:
        raise ValueError("Fresh view and unique occurrences required")
    partitions = set(rows.new_split)
    if partitions not in ({"train", "val"}, {"test"}):
        raise ValueError("Development/test views must be separate")
    paths = []
    counts = {}
    for row in rows.itertuples():
        if not re.fullmatch(r"[A-Za-z0-9_-]+", row.frame_id):
            raise ValueError("Unsafe frame identity")
        image = Path(row.image_path)
        label = Path(row.label_path)
        raw = label.read_bytes()
        parsed = validate_label_bytes(raw)
        if (not parsed.syntax_valid or not parsed.normalized_values_valid
                or (strict_geometry and not parsed.geometry_valid)) or any(
            c not in NAMES for c in parsed.classes_present
        ):
            raise ValueError("Unapproved annotation syntax/ontology")
        suffix = image.suffix.lower()
        if suffix not in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"):
            raise ValueError("Unsupported image")
        target = output / "images" / f"{row.frame_id}{suffix}"
        _write_checked(target, image.read_bytes(), row.image_sha256)
        _write_checked(output / "labels" / f"{row.frame_id}.txt", raw, row.label_sha256)
        counts[row.frame_id] = parsed.num_objects
        paths.append(str(target.resolve()))
    rows = rows.copy()
    rows["image_path"] = paths
    rows["num_objects"] = rows.frame_id.map(counts)
    rows.to_parquet(output / "records.parquet", index=False)
    config = {"path": str(output.resolve()), "nc": 5, "names": NAMES}
    for partition in sorted(partitions):
        path = output / f"{partition}.txt"
        path.write_text(
            "\n".join(rows.loc[rows.new_split == partition, "image_path"]) + "\n",
            encoding="utf-8",
        )
        config[partition] = str(path.resolve())
    if partitions == {"test"}:
        # Ultralytics requires train/val keys even for split=test. No development
        # bytes are exposed to this evaluator; the three keys name the sealed test.
        config.update(train=config["test"], val=config["test"])
    path = output / "dataset.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def seal(directory: Path, state: dict, frozen_inputs: dict | None = None) -> dict:
    """Atomic receipt replacement; a crash cannot authorize a half-written checkpoint."""
    files = {
        str(p.relative_to(directory)).replace("\\", "/"): (
            frozen_inputs[p] if frozen_inputs and p in frozen_inputs else file_sha256(p)
        )
        for p in directory.rglob("*")
        if p.is_file() and p.name not in ("execution.json", "execution.partial")
    }
    result = {**state, "files": files}
    temporary = directory / "execution.partial"
    write_json(temporary, result)
    temporary.replace(directory / "execution.json")
    return result


def verify_execution(directory: Path, expected: dict | None = None) -> dict:
    state = read_json(directory / "execution.json")
    if expected and any(state.get(k) != v for k, v in expected.items()):
        raise ValueError("Execution identity/state mismatch")
    if (
        "run" in state
        and state["state"] == "completed"
        and state.get("test_used") is False
    ):
        required = {
            "train/weights/best.pt",
            "train/weights/last.pt",
            "train/args.yaml",
            "train/results.csv",
        }
        if not required <= set(state["files"]):
            raise ValueError("Completed training receipt lacks checkpoint bindings")
    for name, sha in state["files"].items():
        path = directory / name
        if (
            not path.resolve().is_relative_to(directory.resolve())
            or file_sha256(path) != sha
        ):
            raise ValueError("Checkpoint/execution checksum mismatch")
    return state


def verify_test_view(test_root: Path, permission: dict, ctx: dict) -> None:
    """A self-receipted test view is insufficient: rebind it to signed originals."""
    if ctx["admission"].get("input_format") == "general_ab_assignments_v1":
        from flir_pipeline.detection.general_ab import released_test_records

        _, _, expected = released_test_records(permission, ctx)
    else:
        expected = pd.read_parquet(bound_file(permission["test_sources"]["records"]))
    if records_identity(expected) != ctx["admission"]["test"]["records_id"]:
        raise ValueError("Approved test identity differs")
    actual = pd.read_parquet(test_root / "view/records.parquet")
    original = actual.copy()
    if ctx["admission"]["variant"]["kind"] == "geometry_preserving":
        original["image_sha256"] = actual.original_image_sha256
        mapping = read_json(bound_file(ctx["admission"]["variant"]["mapping"]))
        hashes = {r["frame_id"]: r["sha256"] for r in mapping["records"]}
        if any(
            row.image_sha256 != hashes.get(row.frame_id) for row in actual.itertuples()
        ):
            raise ValueError("Test transformed bytes are not approved")
    pd.testing.assert_frame_equal(
        original[expected.columns].sort_values("frame_id").reset_index(drop=True),
        expected.sort_values("frame_id").reset_index(drop=True),
    )
    if set(actual.new_split) != {"test"} or not actual.frame_id.is_unique:
        raise ValueError("Invalid test-only view")
    view = test_root / "view"
    config = yaml.safe_load((view / "dataset.yaml").read_text(encoding="utf-8"))
    expected_config = {
        "path": str(view.resolve()),
        "nc": 5,
        "names": NAMES,
        **{s: str((view / "test.txt").resolve()) for s in ("train", "val", "test")},
    }
    if (
        config != expected_config
        or (view / "test.txt").read_text(encoding="utf-8").splitlines()
        != actual.image_path.tolist()
    ):
        raise ValueError("Test YAML or membership locator changed")
    for row in actual.itertuples():
        image = Path(row.image_path)
        label = view / "labels" / (row.frame_id + ".txt")
        if (
            not image.resolve().is_relative_to(view.resolve())
            or image.stem != row.frame_id
        ):
            raise ValueError("Unsafe test image locator")
        if (
            file_sha256(image) != row.image_sha256
            or file_sha256(label) != row.label_sha256
        ):
            raise ValueError("Test image/label checksum mismatch")
        if validate_label_bytes(label.read_bytes()).num_objects != row.num_objects:
            raise ValueError("Test annotation count changed")


def execute(
    plan: Path,
    output: Path,
    index: int,
    document: Path,
    signature: Path,
    resume_document: Path | None = None,
    resume_signature: Path | None = None,
) -> dict:
    ctx = context(document, signature, plan, output)
    check_test_isolation(ctx["admission"])
    if not 0 <= index < len(ctx["runs"]):
        raise ValueError("Run index outside approved budget")
    run = ctx["runs"][index]
    root = output / run["run_id"]
    # POSIX workers only. A live worker owns this lock; a scheduler restart
    # releases it automatically, without deleting interruption evidence.
    import fcntl

    root.mkdir(parents=True, exist_ok=True)
    with (root / ".worker.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _execute_locked(ctx, run, root, resume_document, resume_signature)


def _execute_locked(
    ctx: dict,
    run: dict,
    root: Path,
    resume_document: Path | None,
    resume_signature: Path | None,
) -> dict:
    previous = None
    if resume_document is not None:
        permission = authorized(resume_document, resume_signature, "resume")
        if (
            permission["admission_sha256"] != ctx["admission_sha256"]
            or permission["run_id"] != run["run_id"]
        ):
            raise ValueError("Resume authorization belongs to another run")
        previous = bound_file(permission["execution"]).parent
        state = verify_execution(
            previous,
            {
                "run": run,
                "admission_sha256": ctx["admission_sha256"],
                "test_used": False,
            },
        )
        required = {
            "train/weights/last.pt",
            "train/weights/best.pt",
            "train/args.yaml",
            "train/results.csv",
        }
        if not required <= set(state["files"]):
            raise ValueError("Resume checkpoint/history must be receipt-bound")
        if state["state"] not in ("interrupted", "failed") or previous.parent != root:
            raise ValueError("Only an interrupted identical run may resume")
    attempts = sorted(root.glob("attempt-*")) if root.exists() else []
    if (previous is None and attempts) or (
        previous is not None and (not attempts or previous != attempts[-1])
    ):
        raise ValueError(
            "Existing attempt preserved; explicit latest-attempt resume required"
        )
    directory = root / f"attempt-{len(attempts):04d}"
    # Prime torch before importing optional runtime, after all admission checks.
    import torch  # noqa: F401
    from ultralytics import settings  # noqa: F401

    from flir_pipeline.detection.scientific_runtime import run_training

    return run_training(ctx, run, directory, previous)


def evaluation_context(
    document: Path, signature: Path, training_document: Path, training_signature: Path
) -> tuple[dict, dict]:
    permission = authorized(document, signature, "evaluate")
    admission = training_admission(training_document, training_signature)
    if permission["admission_sha256"] != file_sha256(training_document):
        raise ValueError("Evaluation release differs from training approval")
    ctx = context(
        training_document,
        training_signature,
        Path(admission["plan"]["path"]).parent,
        Path(admission["output_root"]),
    )
    if set(permission["executions"]) != {r["run_id"] for r in ctx["runs"]}:
        raise ValueError("Entire approved matrix must finish before opening test")
    for run in ctx["runs"]:
        directory = bound_file(permission["executions"][run["run_id"]]).parent
        verify_execution(
            directory,
            {
                "run": run,
                "state": "completed",
                "admission_sha256": ctx["admission_sha256"],
                "test_used": False,
                "checkpoint_selection": "validation_only",
            },
        )
    return permission, ctx


def prepare_test(
    document: Path,
    signature: Path,
    training_document: Path,
    training_signature: Path,
    output: Path,
) -> dict:
    """Custodian-only test materialization after the complete matrix is released."""

    permission, ctx = evaluation_context(
        document, signature, training_document, training_signature
    )
    if ctx["admission"].get("input_format") == "general_ab_assignments_v1":
        from flir_pipeline.detection.general_ab import materialize_released_test

        external_output(output, document.parent, training_document.parent)
        rows = materialize_released_test(permission, ctx, output)
        rows = variant_rows(rows, ctx["admission"]["variant"])
        write_view(rows, output / "view")
        return seal(
            output,
            {
                "state": "sealed_test",
                "admission_sha256": ctx["admission_sha256"],
                "evaluation_sha256": file_sha256(document),
                "test_id": ctx["admission"]["test"]["records_id"],
            },
        )
    return materialize_original_test(permission, ctx, output, document, training_document)


def materialize_original_test(
    permission: dict, ctx: dict, output: Path, *sources: Path
) -> dict:
    """Byte-preserving test view shared by legacy release and automatic matrix gate."""
    from flir_pipeline.detection.annotation_review import source_bytes

    test = permission["test_sources"]
    rows = pd.read_parquet(bound_file(test["records"]))
    if records_identity(rows) != ctx["admission"]["test"]["records_id"]:
        raise ValueError("Test records differ from approved fixed identity")
    labels = {
        r["label_id"]: r for r in read_json(bound_file(test["annotations"]))["records"]
    }
    source_root = Path(test["source_root"])
    external_output(output, source_root, *sources)
    if output.exists():
        raise ValueError("Test destination exists")
    output.mkdir(parents=True)
    images, label_paths = [], []
    for row in rows.to_dict("records"):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", row["frame_id"]):
            raise ValueError("Unsafe test frame")
        image = (
            output
            / "original/images"
            / (row["frame_id"] + Path(row["zip_member_name"]).suffix.lower())
        )
        label = output / "original/labels" / (row["frame_id"] + ".txt")
        _write_checked(
            image,
            source_bytes(source_root, row, row["image_sha256"]),
            row["image_sha256"],
        )
        _write_checked(
            label,
            source_bytes(source_root, labels[row["label_ids"][0]], row["label_sha256"]),
            row["label_sha256"],
        )
        images.append(str(image.resolve()))
        label_paths.append(str(label.resolve()))
    rows["image_path"], rows["label_path"], rows["new_split"] = (
        images,
        label_paths,
        "test",
    )
    rows = variant_rows(rows, ctx["admission"]["variant"])
    write_view(rows, output / "view", strict_geometry=not bool(ctx["admission"].get("policy")))
    return seal(
        output,
        {
            "state": "sealed_test",
            "admission_sha256": ctx["admission_sha256"],
            "evaluation_sha256": permission.get("release_id")
            or file_sha256(sources[0]),
            "test_id": ctx["admission"]["test"]["records_id"],
        },
    )


def evaluate_run(
    document: Path,
    signature: Path,
    training_document: Path,
    training_signature: Path,
    test_root: Path,
    output: Path,
    index: int,
) -> dict:
    permission, ctx = evaluation_context(
        document, signature, training_document, training_signature
    )
    verify_execution(
        test_root,
        {
            "state": "sealed_test",
            "admission_sha256": ctx["admission_sha256"],
            "evaluation_sha256": file_sha256(document),
        },
    )
    verify_test_view(test_root, permission, ctx)
    if not 0 <= index < len(ctx["runs"]):
        raise ValueError("Evaluation index outside approved matrix")
    run = ctx["runs"][index]
    directory = bound_file(permission["executions"][run["run_id"]]).parent
    external_output(output, directory, test_root, document.parent)
    if output.exists():
        raise ValueError("Evaluation output exists")
    import torch  # noqa: F401
    from ultralytics import settings  # noqa: F401

    from flir_pipeline.detection.scientific_runtime import run_evaluation

    return run_evaluation(ctx, run, directory, test_root, output)
