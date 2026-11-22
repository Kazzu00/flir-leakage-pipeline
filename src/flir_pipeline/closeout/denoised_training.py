"""Existing candidate restricted by nominal test risk; exploratory training only.

No original-test inference path exists here. Unresolved provenance is not promoted
to eligibility. The strict fully resolved subset is reported separately.
"""

import shutil
from pathlib import Path

import pandas as pd

from flir_pipeline.closeout.io import read_json, seal, transaction, verify, write_json
from flir_pipeline.closeout.oe4 import aug_off, authorize, path, verify_freeze
from flir_pipeline.closeout.oe4_runtime import execute_prepared, gpu_preflight
from flir_pipeline.closeout.training import check_cpu, prepare
from flir_pipeline.similarity.storage import file_sha256

NAMES = {0: "vehicle", 1: "building", 2: "road", 3: "river", 4: "maquinaria"}
VERSION = "denoised_candidate_exploratory_v1"


def materialize(candidate: Path, anchors: Path, publication: Path, output: Path):
    """Do not refit features/groups/splits. Exclude entire groups at nominal risk."""
    verify(candidate)
    verify(anchors)
    audit = read_json(candidate / "audit.json")
    bridge = read_json(anchors / "audit.json")
    if (
        bridge["source_dataset_receipt_sha256"]
        != read_json(candidate / "provenance.json")["source_receipt_sha256"]
    ):
        raise ValueError("Anchor audit belongs to another denoised source")
    if audit["state"] != "CANDIDATE_NOT_FROZEN" or bridge["verified_anchors"] != 0:
        raise ValueError(
            "Expected existing unanchored candidate, not an approved split"
        )
    if bridge["publication_receipt_sha256"] != file_sha256(
        publication / "receipt.json"
    ):
        raise ValueError("Anchor audit publication differs")
    if bridge["protected_records_sha256"] != file_sha256(
        publication / "sources/proposal/test_records.parquet"
    ):
        raise ValueError("Protected occurrence catalogue differs")
    publication_receipt = read_json(publication / "receipt.json")
    bound = publication_receipt.get("artifacts", publication_receipt.get("files", {}))
    for name in (
        "sources/audit/candidate_manifest.parquet",
        "sources/proposal/protection_groups.parquet",
        "sources/proposal/test_records.parquet",
    ):
        if bound.get(name) != file_sha256(publication / name):
            raise ValueError("Used publication snapshot checksum differs")
    rows = pd.read_csv(candidate / "memberships.csv", keep_default_na=False)
    exceptions = pd.read_csv(
        anchors / "correspondence_exceptions.csv", keep_default_na=False
    )
    if not rows.frame_id.is_unique or not exceptions.denoised_frame_id.is_unique:
        raise ValueError("Ambiguous denoised occurrence identity")
    joined = rows.merge(
        exceptions[
            ["denoised_frame_id", "status", "candidate_frame_ids", "candidate_test_ids"]
        ],
        left_on="frame_id",
        right_on="denoised_frame_id",
        validate="one_to_one",
    )
    if len(joined) != len(rows):
        raise ValueError("Incomplete nominal correspondence audit")
    original = pd.read_parquet(publication / "sources/audit/candidate_manifest.parquet")
    protection = pd.read_parquet(
        publication / "sources/proposal/protection_groups.parquet"
    )
    test = pd.read_parquet(publication / "sources/proposal/test_records.parquet")
    protected_groups = set(
        protection.loc[protection.content_id.isin(test.content_id), "group_id"]
    )
    protected_contents = set(
        protection.loc[protection.group_id.isin(protected_groups), "content_id"]
    )
    protected_ids = set(
        original.loc[original.content_id.isin(protected_contents), "frame_id"]
    )
    risky = joined.candidate_frame_ids.map(
        lambda value: bool(set(value.split(";")) & protected_ids)
    )
    risky |= joined.candidate_test_ids.ne("")
    risky |= joined.content_id.isin(protected_contents)
    excluded_groups = set(joined.loc[risky, "group_id"])
    included = joined.loc[~joined.group_id.isin(excluded_groups)].copy()
    strict_groups = excluded_groups | set(
        joined.loc[joined.status.ne("NOMINAL_ONLY"), "group_id"]
    )
    strict = joined.loc[~joined.group_id.isin(strict_groups)]
    # Same original assignment for every retained occurrence; no reallocation.
    if included.groupby("group_id").new_split.nunique().gt(1).any():
        raise ValueError("Existing atomic group crosses candidate splits")
    development = included.loc[included.new_split.isin(["train", "val"])].copy()
    for split in ("train", "val"):
        if (
            development.loc[
                development.new_split.eq(split), [f"class_{i}" for i in range(5)]
            ]
            .sum()
            .le(0)
            .any()
        ):
            raise ValueError("Exploratory development lacks class support")
    report = {
        "state": "EXPLORATORY_ONLY_NOT_PROTECTED_TEST_ELIGIBLE",
        "protocol": VERSION,
        "original_split_reassigned": False,
        "c10_original": False,
        "common_test_evaluation_forbidden": True,
        "confirmatory_denoising_effect": False,
        "independence_proven": False,
        "verified_anchors": 0,
        "excluded_atomic_groups": sorted(excluded_groups),
        "excluded_records": len(joined) - len(included),
        "counts": included.new_split.value_counts().to_dict(),
        "unresolved_retained": int(included.status.ne("NOMINAL_ONLY").sum()),
        "class_counts": included.groupby("new_split")[[f"class_{i}" for i in range(5)]]
        .sum()
        .to_dict("index"),
        "strict_nominally_resolved_counts": strict.new_split.value_counts().to_dict(),
        "strict_val_objects": int(
            strict.loc[strict.new_split.eq("val"), [f"class_{i}" for i in range(5)]]
            .sum()
            .sum()
        ),
        "limits": [
            "Nominal test/group candidates excluded, unresolved provenance remains",
            "No independently verified normal/denoised identity-and-label pairing",
            "Descriptive exploratory training only; never compare to the six protected-test models as denoising effect",
        ],
        "source_candidate_receipt_sha256": file_sha256(candidate / "receipt.json"),
        "source_anchor_receipt_sha256": file_sha256(anchors / "receipt.json"),
        "publication_receipt_sha256": file_sha256(publication / "receipt.json"),
    }
    with transaction(output, candidate, anchors, publication) as stage:
        for row in development.itertuples():
            for kind, column, expected in (
                ("images", "image_path", row.image_sha256),
                ("labels", "label_path", row.label_sha256),
            ):
                source = (
                    candidate / kind / row.new_split / Path(getattr(row, column)).name
                )
                destination = stage / kind / f"{row.frame_id}{source.suffix}"
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
                if file_sha256(destination) != expected:
                    raise ValueError("Exploratory copy checksum differs")
        development["image_path"] = development.frame_id.map(
            lambda value: f"images/{value}.png"
        )
        development["label_path"] = development.frame_id.map(
            lambda value: f"labels/{value}.txt"
        )
        development.to_parquet(stage / "development.parquet", index=False)
        included.to_csv(stage / "candidate_memberships.csv", index=False)
        joined.loc[joined.group_id.isin(excluded_groups)].to_csv(
            stage / "excluded.csv", index=False
        )
        included.loc[included.new_split.eq("test")].to_csv(
            stage / "held_out_internal_test.csv", index=False
        )
        write_json(stage / "audit.json", report)
        seal(stage, {"state": report["state"], "protocol": VERSION})
    return report


def source_rows(dataset: Path):
    receipt = verify(dataset)
    if (
        receipt["protocol"] != VERSION
        or not read_json(dataset / "audit.json")["common_test_evaluation_forbidden"]
    ):
        raise ValueError("Not the isolated exploratory denoised artifact")
    rows = pd.read_parquet(dataset / "development.parquet")
    for column in ("image_path", "label_path"):
        rows[column] = rows[column].map(lambda value: str((dataset / value).resolve()))
        if not all(
            Path(value).is_relative_to(dataset.resolve()) for value in rows[column]
        ):
            raise ValueError("Exploratory data locator escapes source")
    return rows


def prepare_cpu(config: Path, dataset: Path, output: Path):
    cfg, freeze = authorize(config, "train")
    binding = {
        "freeze_sha256": file_sha256(path(cfg["freeze"]) / "receipt.json"),
        "dataset_receipt_sha256": file_sha256(dataset / "receipt.json"),
    }
    if output.exists():
        receipt = verify(output)
        if receipt["state"] != "READY_EXPLORATORY" or any(
            receipt[k] != v for k, v in binding.items()
        ):
            raise ValueError("Exploratory preparation binding changed")
        return receipt
    output.mkdir(parents=True, exist_ok=False)
    try:
        rows = source_rows(dataset)
        baseline = freeze["runs"]["c10"]["baseline_args"]
        checks = {}
        for treatment, controls in (
            ("AUG_ON", baseline),
            ("AUG_OFF", aug_off(baseline)),
        ):
            prepared = prepare(
                rows,
                output / treatment,
                controls,
                path(cfg["initial_weights"]),
                names=NAMES,
            )
            checks[treatment] = check_cpu(prepared, baseline)
        verify_freeze(config)
        seal(
            output,
            {
                "state": "READY_EXPLORATORY",
                "protocol": VERSION,
                **binding,
                "checks": checks,
                "common_test_evaluation_forbidden": True,
            },
        )
    except BaseException as exc:
        seal(output, {"state": "FAILED_CPU_PREPARATION", **binding, "error": str(exc)})
        raise
    return verify(output)


def train(config: Path, dataset: Path, work: Path, index: int):
    if index not in (0, 1):
        raise ValueError("Exploratory array index must be 0/1")
    cfg, freeze = authorize(config, "train")
    preparation = verify(work / "cpu-preparation")
    if (
        preparation["state"] != "READY_EXPLORATORY"
        or preparation["freeze_sha256"]
        != file_sha256(path(cfg["freeze"]) / "receipt.json")
        or preparation["dataset_receipt_sha256"]
        != file_sha256(dataset / "receipt.json")
    ):
        raise ValueError("Exploratory CPU preparation differs")
    env = gpu_preflight()
    treatment = ("AUG_ON", "AUG_OFF")[index]
    baseline = freeze["runs"]["c10"]["baseline_args"]
    prepared = prepare(
        source_rows(dataset),
        work / "runs" / treatment,
        baseline if index == 0 else aug_off(baseline),
        path(cfg["initial_weights"]),
        names=NAMES,
    )
    check_cpu(prepared, baseline)
    state = {
        "state": "RUNNING_EXPLORATORY",
        "protocol": VERSION,
        "policy": f"denoised_restricted_candidate_{treatment}",
        "augmentation": treatment,
        "exploratory": True,
        "test_used": False,
        "common_test_evaluation_forbidden": True,
        "freeze_sha256": preparation["freeze_sha256"],
        "dataset_receipt_sha256": preparation["dataset_receipt_sha256"],
        "authorization": freeze["scientific_authorization"],
        "environment": env,
    }

    def verify_after():
        verify_freeze(config)
        verify(dataset)

    return execute_prepared(prepared, state, verify_after)
