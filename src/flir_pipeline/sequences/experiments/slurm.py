"""SLURM transport and durable receipts; scientific work runs only in workers."""

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from flir_pipeline.sequences.experiments.artifacts import binding, inspect, versions
from flir_pipeline.sequences.experiments.config import SuiteConfig
from flir_pipeline.sequences.experiments.inputs import (
    load_profile,
    local_path,
    resolve_inputs,
)
from flir_pipeline.sequences.experiments.sources import load_sources
from flir_pipeline.sequences.experiments.stages import (
    execute_stage,
    graph,
    review_checkpoint,
)
from flir_pipeline.sequences.experiments.structure import load_structure
from flir_pipeline.similarity.storage import read_json, stable_id

MODULE = "flir_pipeline.sequences.experiments.slurm"
FAILED = {
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
    "PREEMPTED",
    "BOOT_FAIL",
    "DEADLINE",
}
ACTIVE = {
    "PENDING",
    "RUNNING",
    "CONFIGURING",
    "COMPLETING",
    "SUSPENDED",
    "REQUEUED",
    "RESIZING",
}


def now():
    return datetime.now(UTC).isoformat()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def plan_run(profile, family, evidence, dataset_variant=None):
    config, inputs, resources = load_profile(profile)
    paths, source = resolve_inputs(
        inputs, family, require_images=True, dataset_variant=dataset_variant
    )
    evidence = Path(evidence).resolve()
    load_structure(evidence, source)
    identity = dict(
        schema_version="sequence_slurm_run_v1",
        family=family,
        dataset_variant=source.signature["dataset_variant_id"],
        configuration=config.model_dump(mode="json"),
        inputs={k: str(v) for k, v in paths.items()},
        source_signature=source.signature,
        evidence=str(evidence),
        evidence_binding=binding(evidence),
        software=versions(),
        resources=resources.model_dump(mode="json"),
        cwd=str(Path.cwd().resolve()),
    )
    run_id = "seq-" + stable_id(identity)
    root = local_path(resources.run_root) / run_id
    source.safe_output(
        root, evidence, *[v for k, v in paths.items() if k.startswith("images_")]
    )
    stages = graph(config)
    for spec in stages:
        spec["command"] = [
            "env",
            "OMP_NUM_THREADS=1",
            "NUMBA_NUM_THREADS=1",
            "OPENBLAS_NUM_THREADS=1",
            "MKL_NUM_THREADS=1",
            "HF_HUB_OFFLINE=1",
            "TRANSFORMERS_OFFLINE=1",
            sys.executable,
            "-m",
            MODULE,
            "worker",
            str(root / "slurm_run.json"),
            spec["stage"],
        ]
        spec["log"] = str(root / "logs" / f"{spec['stage']}-%j.log")
        # Suite locators must resolve against the common products root.
        output = (
            root / "products"
            if spec["kind"] == "suite"
            else root / "products" / spec["stage"]
        )
        spec["output_root"] = str(output)
        spec["expected_artifact"] = str(output / "<artifact_kind>" / "<content_id>")
        spec["receipt"] = str(root / "receipts" / f"{spec['stage']}.json")
        spec["resources"] = resources.model_dump(exclude={"run_root"}, mode="json")
    plan = dict(
        run_id=run_id,
        identity=identity,
        stages=stages,
        manifest=str(root / "slurm_run.json"),
    )
    return dict(plan=plan, plan_id=stable_id(plan), submissions=[], created_at=now())


def read_run(path):
    manifest = read_json(Path(path))
    if stable_id(manifest["plan"]) != manifest["plan_id"]:
        raise ValueError("Run plan changed; refuse to reuse artifacts or submit")
    if "seq-" + stable_id(manifest["plan"]["identity"]) != manifest["plan"]["run_id"]:
        raise ValueError("Run identity mismatch")
    return manifest


def receipt_path(spec, manifest):
    path = Path(spec["receipt"])
    if not path.exists():
        return None
    receipt = read_json(path)
    artifact = Path(receipt["artifact"])
    if (
        receipt["stage"] != spec["stage"]
        or receipt["plan_id"] != manifest["plan_id"]
        or not artifact.resolve().is_relative_to(Path(spec["output_root"]).resolve())
        or binding(artifact) != receipt["binding"]
    ):
        raise ValueError(f"Invalid immutable stage receipt: {spec['stage']}")
    specs = {s["stage"]: s for s in manifest["plan"]["stages"]}
    if set(receipt["dependency_bindings"]) != set(spec["dependencies"]):
        raise ValueError("Stage receipt has incorrect dependency coverage")
    for dependency, expected in receipt["dependency_bindings"].items():
        previous = read_json(Path(specs[dependency]["receipt"]))
        if binding(Path(previous["artifact"])) != expected:
            raise ValueError("Completed stage dependency changed")
    meta = inspect(artifact)
    if (
        meta["identity"]["sources"].get("input")
        != manifest["plan"]["identity"]["source_signature"]
    ):
        raise ValueError("Completed artifact belongs to different inputs")
    summary = read_json(artifact / "summary.json")
    if summary.get("all_requested_cells_succeeded") is False:
        raise ValueError("Incomplete scientific artifact cannot satisfy a stage")
    return artifact


def submit(manifest, *, dry_run=False):
    if dry_run:
        return manifest
    path = Path(manifest["plan"]["manifest"])
    if path.parent.exists():
        raise ValueError(
            f"Run already exists: {manifest['plan']['run_id']}; inspect status/resubmission-plan"
        )
    path.parent.mkdir(parents=True, exist_ok=False)
    (path.parent / "logs").mkdir()
    (path.parent / "receipts").mkdir()
    atomic_json(path, manifest)
    ids = {}
    for spec in manifest["plan"]["stages"]:
        args = [
            "sbatch",
            "--parsable",
            f"--job-name={manifest['plan']['run_id']}-{spec['stage']}",
            f"--chdir={manifest['plan']['identity']['cwd']}",
            f"--output={spec['log']}",
            f"--cpus-per-task={spec['resources']['cpus_per_task']}",
            f"--mem={spec['resources']['memory']}",
            f"--time={spec['resources']['time']}",
        ]
        for key in ("partition", "account"):
            if spec["resources"].get(key):
                args.append(f"--{key}={spec['resources'][key]}")
        if spec["dependencies"]:
            args.append(
                "--dependency=afterok:" + ":".join(ids[d] for d in spec["dependencies"])
            )
        args.extend(["--wrap", shlex.join(spec["command"])])
        entry = dict(
            stage=spec["stage"],
            job_id=None,
            dependencies=spec["dependencies"],
            dependency_job_ids=[ids[d] for d in spec["dependencies"]],
            command=spec["command"],
            sbatch_command=args,
            output_log=spec["log"],
            state="SUBMITTING",
            submitted_at=now(),
        )
        manifest["submissions"].append(entry)
        atomic_json(path, manifest)
        try:
            result = subprocess.run(args, check=True, capture_output=True, text=True)
            job_id = result.stdout.strip().split(";")[0]
            if re.fullmatch(r"[0-9]+", job_id) is None:
                raise ValueError(f"Unexpected sbatch receipt: {result.stdout!r}")
            ids[spec["stage"]] = job_id
            entry.update(job_id=job_id, state="SUBMITTED")
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            # Submission transport can fail after SLURM accepted a job. Never
            # assume it is absent: operator reconciliation is required.
            entry.update(state="SUBMISSION_UNKNOWN", error=str(error))
            atomic_json(path, manifest)
            raise ValueError(f"Submission stopped; inspect {path}: {error}") from error
        atomic_json(path, manifest)
    return manifest


def scheduler_states(manifest):
    ids = sorted({r["job_id"] for r in manifest["submissions"] if r.get("job_id")})
    if not ids:
        return {}, []
    states, warnings = {}, []
    commands = [
        [
            "sacct",
            "--noheader",
            "--parsable2",
            "--jobs",
            ",".join(ids),
            "--format=JobIDRaw,State,ExitCode",
        ],
        ["squeue", "--noheader", "--jobs", ",".join(ids), "--format=%i|%T|%r"],
    ]
    for command in commands:
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
        except (OSError, subprocess.SubprocessError) as error:
            warnings.append(f"{command[0]} unavailable: {error}")
            continue
        for line in result.stdout.splitlines():
            fields = line.strip().split("|")
            if len(fields) >= 2 and fields[0] in ids:
                state = fields[1].split()[0].rstrip("+")
                if (
                    command[0] == "sacct"
                    and state == "COMPLETED"
                    and fields[2] != "0:0"
                ):
                    state = "FAILED"
                states[fields[0]] = state
    return states, warnings


def status(manifest, *, states=None, review=None):
    if states is None:
        states, warnings = scheduler_states(manifest)
    else:
        warnings = []
    latest = {r["stage"]: r for r in manifest["submissions"]}
    rows, mapped = [], {}
    package = None
    for spec in manifest["plan"]["stages"]:
        entry = latest.get(spec["stage"], {})
        scheduler = states.get(entry.get("job_id"), entry.get("state", "MISSING"))
        detail = None
        try:
            artifact = receipt_path(spec, manifest)
        except (ValueError, KeyError, OSError, AssertionError) as error:
            artifact, state, detail = None, "corrupt", str(error)
        else:
            if artifact is not None:
                state = "completed"
                if spec["stage"] == "review_package":
                    package = artifact
            elif any(
                mapped[d] in {"failed", "blocked_by_dependency", "corrupt", "missing"}
                for d in spec["dependencies"]
            ):
                state = "blocked_by_dependency"
            elif scheduler in FAILED or scheduler == "COMPLETED":
                state = "failed"
                if scheduler == "COMPLETED":
                    detail = "SLURM completed without a verified artifact receipt"
            elif scheduler in {"RUNNING", "COMPLETING"}:
                state = "running"
            elif scheduler in ACTIVE or scheduler == "SUBMITTED":
                state = "pending"
            elif scheduler == "MISSING":
                state = "missing"
            else:
                state = "unknown"
        mapped[spec["stage"]] = state
        rows.append(
            dict(
                stage=spec["stage"],
                state=state,
                slurm_state=scheduler,
                job_id=entry.get("job_id"),
                dependencies=spec["dependencies"],
                artifact=str(artifact) if artifact else None,
                detail=detail,
                log=spec["log"],
            )
        )
    if review is not None and package is None:
        raise ValueError("Cannot attach review before a verified package exists")
    return dict(
        run_id=manifest["plan"]["run_id"],
        counts={
            state: sum(r["state"] == state for r in rows)
            for state in (
                "pending",
                "running",
                "completed",
                "failed",
                "blocked_by_dependency",
                "missing",
                "corrupt",
                "unknown",
            )
        },
        stages=rows,
        scheduler_warnings=warnings,
        review=review_checkpoint(package, review)
        if package
        else {"manual_review_required": None, "review_package": None},
        visual_dependency_groups_created=False,
        split_created=False,
    )


def resubmission_plan(manifest, *, states=None):
    """Deterministic conservative plan, not an implicit cancellation/resubmission.

    Pending jobs with failed afterok parents must be reconciled/cancelled first.
    Lost accounting, ambiguous submissions and orphaned publications block reuse.
    """
    from flir_pipeline.sequences.storage import source_checksums

    identity = manifest["plan"]["identity"]
    paths = identity["inputs"]
    checksums = source_checksums(
        Path(paths["manifest"]),
        Path(paths["clip_features"]),
        Path(paths["dinov2_features"]),
    )
    if (
        checksums != identity["source_signature"]["checksums"]
        or binding(Path(identity["evidence"])) != identity["evidence_binding"]
    ):
        raise ValueError("Original inputs/evidence changed; refuse resubmission")
    if identity["software"] != versions():
        raise ValueError(
            "Code/dependencies changed; refuse resubmission under the old run"
        )
    observed = status(manifest, states=states)
    result = []
    for spec, row in zip(manifest["plan"]["stages"], observed["stages"], strict=True):
        state = row["state"]
        output = Path(spec["output_root"])
        orphan = (
            state != "completed"
            and output.exists()
            and spec["kind"] != "suite"
            and any(output.iterdir())
        )
        if state == "completed":
            action = "reuse_verified"
        elif state == "corrupt" or orphan:
            action = "refuse_artifact_reuse"
        elif Path(spec["receipt"]).with_suffix(".lock").exists():
            action = "reconcile_writer_lock_first"
        elif row["slurm_state"] in ACTIVE or state in {"pending", "running", "unknown"}:
            action = "reconcile_existing_job_first"
        elif state in {"missing", "failed", "blocked_by_dependency"}:
            action = "resubmit_after_dependencies"
        else:
            action = "manual_reconciliation"
        result.append(
            dict(
                stage=spec["stage"],
                action=action,
                dependencies=spec["dependencies"],
                command=spec["command"],
                resources=spec["resources"],
                previous_job_id=row["job_id"],
            )
        )
    return dict(
        run_id=observed["run_id"],
        submitted=False,
        stages=result,
        policy="No automatic resume; generate new afterok job IDs only after reconciliation. Never rerun reuse_verified stages.",
    )


def worker(manifest_path, stage):
    if not os.environ.get("SLURM_JOB_ID"):
        raise ValueError(
            "Compute worker requires SLURM_JOB_ID; do not fit on the login node"
        )
    manifest = read_run(manifest_path)
    identity = manifest["plan"]["identity"]
    if identity["software"] != versions():
        raise ValueError(
            "Code/dependencies changed since submission; preserve the original run"
        )
    specs = {s["stage"]: s for s in manifest["plan"]["stages"]}
    spec = specs[stage]
    paths = {k: Path(v) for k, v in identity["inputs"].items()}
    source = load_sources(
        paths["manifest"],
        paths["clip_features"],
        paths["dinov2_features"],
        identity["family"],
        identity["dataset_variant"],
    )
    if (
        source.signature != identity["source_signature"]
        or binding(Path(identity["evidence"])) != identity["evidence_binding"]
    ):
        raise ValueError("Frozen source/evidence identity changed")
    existing = receipt_path(spec, manifest)
    if existing:
        return existing
    dependencies = {
        name: receipt_path(specs[name], manifest) for name in spec["dependencies"]
    }
    if any(path is None for path in dependencies.values()):
        raise ValueError("Dependency missing verified stage receipt")
    lock = Path(spec["receipt"]).with_suffix(".lock")
    # No stale-lock deletion: an interrupted writer must be reconciled explicitly.
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(os.environ["SLURM_JOB_ID"])
    try:
        output = Path(spec["output_root"])
        if spec["kind"] != "suite" and output.exists() and any(output.iterdir()):
            raise ValueError(
                "Unreceipted artifact exists; inspect before any recomputation"
            )
        artifact = execute_stage(
            spec,
            source,
            SuiteConfig.model_validate(identity["configuration"]),
            Path(identity["evidence"]),
            dependencies,
            output,
            paths,
        )
        source.unchanged()
        if identity["software"] != versions():
            raise ValueError(
                "Code/dependencies changed during computation; stage has no valid receipt"
            )
        atomic_json(
            spec["receipt"],
            dict(
                stage=stage,
                plan_id=manifest["plan_id"],
                artifact=str(artifact.resolve()),
                binding=binding(artifact),
                dependency_bindings={k: binding(v) for k, v in dependencies.items()},
                slurm_job_id=os.environ["SLURM_JOB_ID"],
                completed_at=now(),
            ),
        )
        return artifact
    finally:
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    launch = sub.add_parser("submit")
    launch.add_argument("--family", required=True)
    launch.add_argument("--dataset-variant")
    launch.add_argument(
        "--profile",
        type=Path,
        default=Path("configs/hypatia_sequence_experiments.yaml"),
    )
    launch.add_argument("--evidence", type=Path, required=True)
    launch.add_argument("--dry-run", action="store_true")
    for name in ("status", "resubmission-plan"):
        p = sub.add_parser(name)
        p.add_argument("run_id")
        p.add_argument(
            "--run-root", type=Path, default=Path("reports/sequence_experiments")
        )
        if name == "status":
            p.add_argument("--review", type=Path)
    execute = sub.add_parser("worker")
    execute.add_argument("manifest", type=Path)
    execute.add_argument("stage")
    args = parser.parse_args()
    try:
        if args.operation == "submit":
            result = submit(
                plan_run(
                    args.profile, args.family, args.evidence, args.dataset_variant
                ),
                dry_run=args.dry_run,
            )
            if not args.dry_run:
                result = dict(
                    run_id=result["plan"]["run_id"],
                    jobs={r["stage"]: r["job_id"] for r in result["submissions"]},
                    manifest=result["plan"]["manifest"],
                )
        elif args.operation == "worker":
            import contextlib

            with contextlib.redirect_stdout(sys.stderr):
                result = {"artifact": str(worker(args.manifest, args.stage))}
        else:
            if re.fullmatch(r"seq-[0-9a-f]{16}", args.run_id) is None:
                raise ValueError("Invalid run ID")
            run = read_run(args.run_root / args.run_id / "slurm_run.json")
            result = (
                status(run, review=args.review)
                if args.operation == "status"
                else resubmission_plan(run)
            )
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, OSError, KeyError, AssertionError) as error:
        parser.exit(1, f"Sequence operation failed: {error}\n")


if __name__ == "__main__":
    main()
