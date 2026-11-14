"""Automatic operational freeze and immutable YOLO11m attempts.

The pilot uses development only and cannot count as a scientific matrix cell.
All gates run before optional GPU imports or checkpoint deserialization.
"""

from __future__ import annotations

import copy
import platform
from contextlib import contextmanager
from pathlib import Path

import yaml

from flir_pipeline.detection.admission import bound_file
from flir_pipeline.detection.common_admission import (
    POLICY,
    binding,
    emit,
    receipt,
    verify_admission,
    weights_receipt,
)
from flir_pipeline.detection.common_executor import (
    external_output,
    verify_executor_plan,
)
from flir_pipeline.detection.runtime import (
    environment_signature,
    parse_training_history,
)
from flir_pipeline.detection.scientific_executor import verify_execution
from flir_pipeline.similarity.storage import file_sha256, read_json, stable_id


def source_identity() -> dict:
    """Bind executable code, not a clean-tree claim or only the parent Git commit."""
    root = Path(__file__).resolve().parents[1]
    return {
        p.relative_to(root).as_posix(): file_sha256(p)
        for p in sorted(root.rglob("*.py"))
    }


def compatible_environment(env: dict) -> None:
    if env.get("python", "").split(".")[:2] != ["3", "11"]:
        raise ValueError("Python 3.11 must be recorded in the GPU probe")
    expected = {
        "torch": "2.7.1+cu118",
        "torchvision": "0.22.1",
        "ultralytics": "8.3.203",
    }
    if not env["cuda_available"] or env["cuda"] != "11.8":
        raise ValueError("CUDA 11.8 GPU runtime required; no CPU fallback")
    for package, version in expected.items():
        actual = env["packages"][package]
        if (actual if package == "torch" else actual.split("+")[0]) != version:
            raise ValueError(f"Incompatible {package}")


def probe_evidence(path: Path, config: dict, batch: int) -> dict:
    from flir_pipeline.detection.common_checks import verify_operation

    state = verify_operation(path)
    identity = state["identity"]
    if (
        state["state"] != "completed"
        or identity["device"] != "0"
        or identity["config"] != config
        or identity["batch"] != batch
        or state["actual_epochs"] != 1
        or not state["observed_batches"]
        or set(state["observed_batches"]) != {batch}
    ):
        raise ValueError("Completed matching synthetic GPU probe required")
    env = state["environment"]
    if (
        env["state"] != "completed"
        or env["blockers"]
        or env["config_id"] != stable_id(config)
    ):
        raise ValueError("Probe preflight failed or configuration changed")
    compatible_environment(env["environment"])
    return env["environment"]


def context(gate: Path, weights: Path, batch: int, env: dict, mode: str) -> dict:
    if batch not in (1, 2) or mode not in ("pilot", "matrix"):
        raise ValueError("Explicit batch 1/2 and operation required")
    admission = verify_admission(gate)
    official = weights_receipt(weights)
    plan = verify_executor_plan(bound_file(admission["plan"]).parent)
    for run in plan["runs"]:
        initial = run["identity"]["initialization"]["sha256"]
        if initial is not None and initial != official["weights"]["sha256"]:
            raise ValueError("Plan initial checkpoint differs from official checkpoint")
    compatible_environment(env)
    config = copy.deepcopy(admission["config"])
    config["train"]["batch"] = batch
    if mode == "pilot":
        config["train"].update(epochs=1, close_mosaic=0)
    identity = {
        "policy": POLICY,
        "gate_id": admission["receipt_id"],
        "weights_id": official["receipt_id"],
        "batch": batch,
        "mode": mode,
        "environment": environment_signature(env),
        "source": source_identity(),
        "configuration": config,
    }
    runs = []
    for candidate in plan["runs"]:
        run = {
            "candidate_run_id": candidate["run_id"],
            "records_id": candidate["identity"]["records_id"],
            "detector_seed": candidate["identity"]["detector_seed"],
            "aliases": candidate["aliases"],
            "mode": mode,
        }
        runs.append({**run, "run_id": stable_id({"execution": identity, "run": run})})
    # Internal runtime API retains the name admission; this is explicitly a
    # technical policy, never a synthetic human approval or signed legacy record.
    return {
        "admission": {
            "policy": POLICY,
            "variant": {"id": "original", "kind": "original"},
            "weights": official["weights"],
            "environment": {
                "python": env["python"],
                "signature": environment_signature(env),
            },
            "test": {"records_id": admission["counts"]["test_id"]},
        },
        "admission_sha256": stable_id(identity),
        "identity": identity,
        "prepared": bound_file(admission["preparation"]).parent,
        "config": config,
        "runs": runs,
    }


@contextmanager
def worker_lock(root: Path):
    """OS lock released on process death; evidence directories are never deleted."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".worker.lock").open("a+b") as stream:
        if platform.system() == "Windows":
            import msvcrt

            stream.seek(0)
            if not stream.read(1):
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if platform.system() == "Windows":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def checked_attempt(ctx: dict, run: dict, path: Path, complete: bool = False) -> dict:
    state = verify_execution(
        path,
        {"run": run, "admission_sha256": ctx["admission_sha256"], "test_used": False},
    )
    actual = {
        p.relative_to(path).as_posix()
        for p in path.rglob("*")
        if p.is_file() and p.name not in ("execution.json", "execution.partial")
    }
    if actual != set(state["files"]):
        raise ValueError("Unbound files in immutable training attempt")
    required = {
        "train/weights/last.pt",
        "train/weights/best.pt",
        "train/args.yaml",
        "train/results.csv",
    }
    if not required <= set(state["files"]):
        raise ValueError("No verified checkpoint/history; interrupted work preserved")
    from flir_pipeline.detection.common_checks import check_arguments

    check_arguments(
        yaml.safe_load((path / "train/args.yaml").read_text(encoding="utf-8")),
        ctx["config"],
        run["detector_seed"],
        path / "development/dataset.yaml",
    )
    if state["state"] == "completed":
        parse_training_history(
            path / "train/results.csv", ctx["config"]["train"]["epochs"]
        )
        if state.get("checkpoint_selection") != "validation_only":
            raise ValueError("Checkpoint selection was not validation only")
        if any(
            file_sha256(path / "train/weights" / name) != sha
            for name, sha in state["checkpoint_sha256"].items()
        ):
            raise ValueError("Checkpoint provenance mismatch")
    elif complete or state["state"] not in ("failed", "interrupted"):
        raise ValueError("Training is not complete/recoverable")
    return state


def execute_locked(ctx: dict, run: dict, root: Path, resume: bool) -> dict:
    attempts = sorted(root.glob("attempt-*"))
    previous = attempts[-1] if attempts else None
    if previous:
        state = checked_attempt(ctx, run, previous)
        if state["state"] == "completed":
            return state
        if not resume:
            raise ValueError("Interrupted attempt preserved; use explicit --resume")
    elif resume:
        raise ValueError("No checkpoint belonging to this execution to resume")
    # Preserve the recovered CUDA import order before importing the runtime.
    import torch  # noqa: F401
    from ultralytics import settings  # noqa: F401

    from flir_pipeline.detection.scientific_runtime import run_training

    return run_training(ctx, run, root / f"attempt-{len(attempts):04d}", previous)


def pilot(
    gate: Path,
    weights: Path,
    probe: Path,
    output: Path,
    batch: int = 1,
    index: int = 0,
    resume: bool = False,
) -> dict:
    admission = verify_admission(gate)
    env = probe_evidence(probe, admission["config"], batch)
    ctx = context(gate, weights, batch, env, "pilot")
    if not 0 <= index < len(ctx["runs"]):
        raise ValueError("Pilot index outside matrix")
    external_output(
        output, gate, weights, probe, ctx["prepared"], Path(admission["source_root"])
    )
    root = output / ctx["runs"][index]["run_id"]
    with worker_lock(root):
        return execute_locked(ctx, ctx["runs"][index], root, resume)


def freeze(
    gate: Path,
    weights: Path,
    probe: Path,
    pilot_path: Path,
    output: Path,
    batch: int = 1,
) -> dict:
    admission = verify_admission(gate)
    env = probe_evidence(probe, admission["config"], batch)
    ctx = context(gate, weights, batch, env, "pilot")
    state = read_json(pilot_path / "execution.json")
    if state["run"] not in ctx["runs"]:
        raise ValueError("Pilot belongs to another gate/batch/initialization/source")
    checked_attempt(ctx, state["run"], pilot_path, complete=True)
    if environment_signature(state["environment"]) != environment_signature(env):
        raise ValueError("Pilot and probe environments differ")
    definitive = context(gate, weights, batch, env, "matrix")
    external_output(output, gate, weights, probe, pilot_path, ctx["prepared"])
    return emit(
        output,
        {
            "policy": POLICY,
            "kind": "operational_freeze",
            "gate": binding(gate),
            "weights": binding(weights),
            "probe": binding(probe / "receipt.json"),
            "pilot": binding(pilot_path / "execution.json"),
            "batch": batch,
            "environment": env,
            "identity": definitive["identity"],
            "runs": definitive["runs"],
            "scientific_training_executed": False,
            "pilot_scope": "one development-only epoch; not matrix evidence",
        },
    )


def frozen_context(path: Path) -> dict:
    value = receipt(path, "operational_freeze")
    gate, weights = bound_file(value["gate"]), bound_file(value["weights"])
    admission = verify_admission(gate)
    env = probe_evidence(
        bound_file(value["probe"]).parent, admission["config"], value["batch"]
    )
    pilot_ctx = context(gate, weights, value["batch"], env, "pilot")
    pilot_path = bound_file(value["pilot"]).parent
    run = read_json(pilot_path / "execution.json")["run"]
    if run not in pilot_ctx["runs"]:
        raise ValueError("Unrelated pilot in freeze")
    checked_attempt(pilot_ctx, run, pilot_path, complete=True)
    ctx = context(gate, weights, value["batch"], env, "matrix")
    if ctx["identity"] != value["identity"] or ctx["runs"] != value["runs"]:
        raise ValueError(
            "Operational freeze changed; new software/inputs require a new freeze"
        )
    ctx["freeze"] = value
    return ctx


def train(
    freeze_path: Path, output: Path, index: int = 0, resume: bool = False
) -> dict:
    ctx = frozen_context(freeze_path)
    if not 0 <= index < len(ctx["runs"]):
        raise ValueError("Execution index outside frozen matrix")
    external_output(
        output, freeze_path, ctx["prepared"], bound_file(ctx["admission"]["weights"])
    )
    run = ctx["runs"][index]
    with worker_lock(output / run["run_id"]):
        return execute_locked(ctx, run, output / run["run_id"], resume)


def completed_matrix(ctx: dict, output: Path) -> dict[str, Path]:
    results = {}
    for run in ctx["runs"]:
        attempts = sorted((output / run["run_id"]).glob("attempt-*"))
        if not attempts:
            raise ValueError("Entire matrix must finish before opening test")
        checked_attempt(ctx, run, attempts[-1], complete=True)
        results[run["run_id"]] = attempts[-1]
    return results
