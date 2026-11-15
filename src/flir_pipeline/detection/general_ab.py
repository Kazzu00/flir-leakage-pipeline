"""Explicit occurrence-preserving A/B handoff to the existing YOLO11m worker.

This module neither generates splits nor adjudicates groups. A proposed test may
be inspected and development materialized, but only the existing signed admission
and isolation gates can authorize any model operation.
"""

from __future__ import annotations

import csv
import hashlib
import io
import re
import zipfile
from pathlib import Path, PurePosixPath

import pandas as pd
import yaml
from PIL import Image

from flir_pipeline.data.yolo_labels import validate_label_bytes
from flir_pipeline.detection.common_test_storage import signatures
from flir_pipeline.detection.materialization import _write_checked
from flir_pipeline.similarity.storage import (
    execution_provenance,
    file_sha256,
    read_json,
    write_json,
)

FORMAT = "general_ab_assignments_v1"
POLICIES = ["exact_content_balanced", "dependency_group_balanced"]
REQUIRED = {
    "frame_id",
    "content_id",
    "image_sha256",
    "label_sha256",
    "image_source",
    "image_member",
    "label_source",
    "label_member",
}


def relative_path(root: Path, name: str) -> Path:
    """No absolute, alternate-stream, traversal or symlink escape locators."""
    parts = PurePosixPath(name)
    if (
        not name
        or parts.is_absolute()
        or ".." in parts.parts
        or "\\" in name
        or ":" in name
    ):
        raise ValueError("Unsafe relative input locator")
    path = root / name
    if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink():
        raise ValueError("Input locator escapes its bound root")
    return path


def checked_file(root, binding):
    path = relative_path(root, binding["path"])
    if file_sha256(path) != binding["sha256"]:
        raise ValueError("General A/B input checksum mismatch")
    return path


def read_table(root, binding):
    path = checked_file(root, binding)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream))
    if len(header) != len(set(header)):
        raise ValueError("Duplicate CSV column")
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def validate_membership(rows, test, protected, groups, policy):
    from flir_pipeline.detection.common_executor import active_records

    active_records(rows, test)
    development = rows.loc[rows.new_split.isin(["train", "val"])]
    if set(development.content_id) & set(protected):
        raise ValueError("Protected content in development")
    if set(groups) != set(rows.content_id) or any(not g for g in groups.values()):
        raise ValueError("Accepted groups must cover precisely the occurrence contents")
    protected_groups = {groups[c] for c in protected}
    if any(groups[c] in protected_groups and c not in protected for c in groups):
        raise ValueError("Protected accepted-group closure is incomplete")
    if policy == POLICIES[1]:
        active = rows.loc[rows.new_split != "quarantine"].copy()
        active["accepted_group"] = active.content_id.map(groups)
        if active.groupby("accepted_group").new_split.nunique().gt(1).any():
            raise ValueError("Accepted dependency fragmented in policy B")


def load(publication: Path):
    from flir_pipeline.detection.common_executor import NAMES, records_identity

    binding = file_sha256(publication / "bundle.json")
    descriptor = read_json(publication / "bundle.json")
    if descriptor["schema_version"] != FORMAT:
        raise ValueError("Expected explicit general A/B assignment contract")
    if descriptor["classes"] != {str(k): v for k, v in NAMES.items()}:
        raise ValueError(
            "Executor class catalogue differs; automatic remapping forbidden"
        )
    if (
        not descriptor["population_id"]
        or not re.fullmatch(r"[a-f0-9]{64}", descriptor["dataset_id"])
        or descriptor["test_status"] not in ("PROPOSED", "APPROVED")
    ):
        raise ValueError("Explicit dataset, population and test declaration required")
    records = read_table(publication, descriptor["records"])
    if (
        not REQUIRED <= set(records)
        or not records.frame_id.is_unique
        or not records.frame_id.str.fullmatch(r"[A-Za-z0-9_-]+").all()
        or not records.content_id.eq(records.image_sha256).all()
        or any(
            not records[c].str.fullmatch(r"[a-f0-9]{64}").all()
            for c in ("image_sha256", "label_sha256")
        )
    ):
        raise ValueError("Invalid occurrence/content/annotation identity")
    if "new_split" in records:
        raise ValueError("Assignments must be supplied explicitly per arm")
    for kind in ("image", "label"):
        if not set(records[f"{kind}_source"]) <= set(descriptor["sources"]):
            raise ValueError("Unknown bound source")
        for member in records[f"{kind}_member"]:
            relative_path(publication, member)  # Validate syntax; ZIPs stay read-only.
    test_ids = read_table(publication, descriptor["test_ids"])
    protected = read_table(publication, descriptor["protected_contents"])
    group_table = read_table(publication, descriptor["accepted_groups"])
    if (
        not test_ids.frame_id.is_unique
        or not len(test_ids)
        or not set(test_ids.frame_id) <= set(records.frame_id)
        or not protected.content_id.is_unique
        or not set(protected.content_id) <= set(records.content_id)
        or not group_table.content_id.is_unique
    ):
        raise ValueError("Invalid protected/test/group identity sets")
    test = (
        records.loc[records.frame_id.isin(test_ids.frame_id)]
        .sort_values("frame_id")
        .reset_index(drop=True)
    )
    if not set(test.content_id) <= set(protected.content_id):
        raise ValueError("Test contents must be explicitly protected")
    groups = group_table.set_index("content_id").group_id.to_dict()
    expected = {(p, s) for p in POLICIES for s in range(5)}
    arms = descriptor["arms"]
    if (
        len(arms) != 10
        or {(a["policy"], a["seed"]) for a in arms} != expected
        or len({a["arm_id"] for a in arms}) != 10
    ):
        raise ValueError("Exactly two named policies and seeds 0..4 required")
    assignments, runs, quarantine = {}, [], None
    for arm in arms:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", arm["arm_id"]):
            raise ValueError("Unsafe arm identity")
        membership = read_table(publication, arm["assignments"])
        if (
            set(membership) != {"frame_id", "new_split"}
            or not membership.frame_id.is_unique
            or set(membership.frame_id) != set(records.frame_id)
        ):
            raise ValueError(
                "Each arm must assign all and only the declared occurrences"
            )
        # Preserve IDs and all declared record metadata. Never reinterpret a
        # generic group as a historical S1/S2/S3 policy or rewrite frame IDs.
        rows = (
            records.merge(
                membership[["frame_id", "new_split"]],
                on="frame_id",
                validate="one_to_one",
            )
            .sort_values("frame_id")
            .reset_index(drop=True)
        )
        validate_membership(
            rows, test, set(protected.content_id), groups, arm["policy"]
        )
        excluded = set(rows.loc[rows.new_split == "quarantine", "frame_id"])
        if quarantine is not None and excluded != quarantine:
            raise ValueError("A/B arms must share the same excluded population")
        quarantine = excluded
        assignments[arm["arm_id"]] = rows
        runs.append(
            {
                "arm": arm["arm_id"],
                "population": descriptor["population_id"],
                "policy": arm["policy"],
                "seed": arm["seed"],
                "candidate_id": records_identity(rows),
            }
        )
    if file_sha256(publication / "bundle.json") != binding:
        raise ValueError("General A/B descriptor changed during validation")
    contract = {
        "format": FORMAT,
        "dataset_id": descriptor["dataset_id"],
        "population_id": descriptor["population_id"],
        "test_declared_status": descriptor["test_status"],
        "test_records_id": records_identity(test),
        "source_id": binding,
    }
    return (
        descriptor,
        {"runs": runs, "input_contract": contract},
        assignments,
        test,
        set(protected.content_id),
        groups,
    )


def source_archives(descriptor, source_root):
    return {
        key: checked_file(source_root, value)
        for key, value in descriptor["sources"].items()
    }


def materialize_records(rows, archives, output):
    """Copy precisely the requested bytes; never canonicalize or repair labels."""
    from flir_pipeline.detection.common_executor import NAMES

    images = []
    handles = {k: zipfile.ZipFile(v) for k, v in archives.items()}
    try:
        for r in rows.to_dict("records"):
            values = {}
            for kind in ("image", "label"):
                archive = handles[r[f"{kind}_source"]]
                member = r[f"{kind}_member"]
                if archive.namelist().count(member) != 1:
                    raise ValueError("Missing or ambiguous ZIP member")
                raw = archive.read(member)
                if hashlib.sha256(raw).hexdigest() != r[f"{kind}_sha256"]:
                    raise ValueError(
                        "Image/annotation bytes differ from declared identity"
                    )
                values[kind] = raw
            parsed = validate_label_bytes(values["label"])
            if not parsed.label_valid or any(
                c not in NAMES for c in parsed.classes_present
            ):
                raise ValueError("Invalid annotation or undeclared class")
            suffix = PurePosixPath(r["image_member"]).suffix.lower()
            if suffix not in (
                ".png",
                ".jpg",
                ".jpeg",
                ".bmp",
                ".tif",
                ".tiff",
                ".webp",
            ):
                raise ValueError("Unsupported image format")
            with Image.open(io.BytesIO(values["image"])) as image:
                image.verify()
            target = output / "images" / (r["frame_id"] + suffix)
            _write_checked(target, values["image"], r["image_sha256"])
            _write_checked(
                output / "labels" / (r["frame_id"] + ".txt"),
                values["label"],
                r["label_sha256"],
            )
            images.append(str(target.resolve()))
    finally:
        for archive in handles.values():
            archive.close()
    return images


def prepare(publication, source_root, output):
    from flir_pipeline.detection.common_executor import (
        NAMES,
        VERSION,
        external_output,
        records_identity,
        verify_preparation,
    )

    external_output(output, publication, source_root)
    if output.exists():
        raise ValueError("Preparation destination exists")
    descriptor, summary, assignments, test, protected, groups = load(publication)
    archives = source_archives(descriptor, source_root)
    output.mkdir(parents=True)
    views = {}
    for arm in summary["runs"]:
        rows = assignments[arm["arm"]]
        identity = records_identity(rows)
        if identity in views:
            views[identity]["aliases"].append(arm["arm"])
            views[identity]["policies"].append(arm["policy"])
            continue
        development = rows.loc[rows.new_split.isin(["train", "val"])].copy()
        development["image_path"] = materialize_records(
            development, archives, output / "store"
        )
        view = output / "views" / identity
        view.mkdir(parents=True)
        rows.to_parquet(view / "assignments.parquet", index=False)
        development.to_parquet(view / "records.parquet", index=False)
        for split in ("train", "val"):
            (view / f"{split}.txt").write_text(
                "\n".join(development.loc[development.new_split == split, "image_path"])
                + "\n",
                encoding="utf-8",
            )
        (view / "dataset.yaml").write_text(
            yaml.safe_dump(
                {
                    "path": str(output.resolve()),
                    "nc": 5,
                    "names": NAMES,
                    **{s: str((view / f"{s}.txt").resolve()) for s in ("train", "val")},
                }
            ),
            encoding="utf-8",
        )
        views[identity] = {
            "aliases": [arm["arm"]],
            "policies": [arm["policy"]],
            "test_id": records_identity(test),
        }
    source_archives(
        descriptor, source_root
    )  # Rehash after materialization, no source mutation tolerated.
    receipt = {
        "schema_version": VERSION,
        "source_id": summary["input_contract"]["source_id"],
        "input_contract": summary["input_contract"],
        "protected_contents": sorted(protected),
        "accepted_groups": groups,
        "views": views,
        "ready_for_training": False,
        "test_materialized": False,
        "provenance": execution_provenance(),
        "artifacts": signatures(output),
    }
    write_json(output / "preparation.json", receipt)
    verify_preparation(output)
    return receipt


def verify_prepared_rows(rows, receipt, entry):
    test = rows.loc[rows.new_split == "test"].drop(columns="new_split")
    for policy in entry["policies"]:
        validate_membership(
            rows,
            test,
            set(receipt["protected_contents"]),
            receipt["accepted_groups"],
            policy,
        )


def validate_admission(payload):
    if (
        payload["policies"] != POLICIES
        or payload["split_seeds"] != list(range(5))
        or payload["detector_seeds"] != [42, 43, 44]
        or not payload["population"]
        or type(payload["max_runs"]) is not int
        or not 1 <= payload["max_runs"] <= 30
    ):
        raise ValueError("Invalid explicit A/B matrix or deduplicated budget")


def validate_context_contract(admission, plan, receipt):
    contract = plan.get("input_contract", {})
    if (
        contract.get("format") != FORMAT
        or receipt.get("input_contract") != contract
        or contract["population_id"] != admission["population"]
        or contract["dataset_id"] != admission["dataset_id"]
        or contract["test_records_id"] != admission["test"]["records_id"]
    ):
        raise ValueError("Signed A/B dataset, test or preparation contract mismatch")


def validate_selected(selected, expected):
    observed = [
        (a["policy"], a["split_seed"], run["detector_seed"])
        for run in selected
        for a in run["aliases"]
    ]
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise ValueError("Incomplete or duplicated admitted A/B aliases")
    by_policy = {
        (a["policy"], a["split_seed"], r["detector_seed"]): r["records_id"]
        for r in selected
        for a in r["aliases"]
    }
    if all(
        by_policy[(POLICIES[0], s, d)] == by_policy[(POLICIES[1], s, d)]
        for _, s, d in expected
    ):
        raise ValueError("No distinct A/B contrast; redundant training is not admitted")


def released_test_records(permission, ctx):
    from flir_pipeline.detection.admission import bound_file
    from flir_pipeline.detection.common_executor import records_identity

    test_sources = permission["test_sources"]
    bundle = bound_file(test_sources["general_ab_bundle"])
    descriptor, summary, _, test, _, _ = load(bundle.parent)
    if (
        bundle.name != "bundle.json"
        or summary["input_contract"] != ctx["input_contract"]
        or records_identity(test) != ctx["admission"]["test"]["records_id"]
    ):
        raise ValueError("Released A/B test differs from signed identities")
    return bundle, descriptor, test


def materialize_released_test(permission, ctx, output):
    from flir_pipeline.detection.common_executor import external_output

    bundle, descriptor, test = released_test_records(permission, ctx)
    test_sources = permission["test_sources"]
    source_root = Path(test_sources["source_root"])
    external_output(output, bundle.parent, source_root)
    if output.exists():
        raise ValueError("Test destination exists")
    output.mkdir(parents=True)
    archives = source_archives(descriptor, source_root)
    test["image_path"] = materialize_records(test, archives, output / "original")
    test["label_path"] = test.frame_id.map(
        lambda f: str((output / "original/labels" / f"{f}.txt").resolve())
    )
    test["new_split"] = "test"
    source_archives(descriptor, source_root)
    return test
