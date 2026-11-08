"""Single-process synthetic CUDA diagnosis; never admits or trains FAC data.

Run each case in a fresh process. This script does not tune allocators, remove
memory limits, download weights, or change the scientific executor/configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import traceback
from pathlib import Path

ENV_KEYS = (
    "CUDA_VISIBLE_DEVICES", "CUDA_MODULE_LOADING", "CUDA_LAUNCH_BLOCKING",
    "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF",
    "PYTORCH_NO_CUDA_MEMORY_CACHING", "CUBLAS_WORKSPACE_CONFIG",
    "LD_LIBRARY_PATH", "LD_PRELOAD", "LOADEDMODULES", "OMP_NUM_THREADS",
    "SLURM_JOB_ID", "SLURM_JOB_GPUS", "SLURM_STEP_GPUS",
    "SLURM_MEM_PER_NODE", "SLURM_MEM_PER_CPU", "SLURM_CPUS_PER_TASK",
)


def attempt(function):
    """Diagnostic failures must not replace the original CUDA exception."""
    try:
        return function()
    except Exception as error:
        return {"unavailable": type(error).__name__, "detail": str(error)}


def read_text(path):
    return attempt(lambda: Path(path).read_text(encoding="utf-8"))


def command(argv):
    def run():
        result = subprocess.run(argv, capture_output=True, text=True, timeout=10, check=False)
        return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    return attempt(run)


def system_snapshot():
    result = {
        "pid": os.getpid(), "hostname": platform.node(), "python": sys.version,
        "executable": sys.executable,
        "environment": {key: os.environ.get(key) for key in ENV_KEYS},
        "limits": read_text("/proc/self/limits"),
        "status": read_text("/proc/self/status"),
        "cgroup": read_text("/proc/self/cgroup"),
        "mountinfo": read_text("/proc/self/mountinfo"),
        "memory_cgroups": {},
    }
    # Resolve the actual v1/v2 memory mount, including cgroup namespaces.
    cgroups, mounts = result["cgroup"], result["mountinfo"]
    if isinstance(cgroups, str) and isinstance(mounts, str):
        for entry in cgroups.splitlines():
            _, controllers, relative = entry.split(":", 2)
            if controllers and "memory" not in controllers.split(","):
                continue
            for mount in mounts.splitlines():
                left, right = mount.split(" - ", 1)
                fields, filesystem = left.split(), right.split()
                if filesystem[0] not in ("cgroup", "cgroup2"):
                    continue
                if filesystem[0] == "cgroup" and "memory" not in filesystem[2].split(","):
                    continue
                root, base = Path(fields[3]), Path(fields[4])
                try:
                    suffix = Path(relative).relative_to(root)
                except ValueError:
                    continue
                leaf = (base / suffix).resolve()
                base = base.resolve()
                if not leaf.is_relative_to(base):
                    continue
                for directory in (leaf, *leaf.parents):
                    if not directory.is_relative_to(base):
                        break
                    for name in ("memory.current", "memory.max", "memory.high", "memory.events",
                                 "memory.limit_in_bytes", "memory.usage_in_bytes", "memory.failcnt"):
                        path = directory / name
                        if path.is_file():
                            result["memory_cgroups"][str(path)] = read_text(path)
    maps = read_text("/proc/self/maps")
    result["cuda_libraries"] = sorted({
        line.split()[-1] for line in maps.splitlines()
        if "/" in line and any(token in line.lower() for token in ("cuda", "cudnn", "cublas"))
    }) if isinstance(maps, str) else maps
    return result


def cuda_snapshot(torch):
    cuda = torch.cuda
    if not cuda.is_initialized():
        return {"initialized": False}
    result = {"initialized": True}
    for name in ("mem_get_info", "memory_stats", "memory_allocated", "memory_reserved",
                 "max_memory_allocated", "max_memory_reserved"):
        result[name] = attempt(getattr(cuda, name))
    for name in ("get_per_process_memory_fraction", "get_allocator_backend"):
        function = getattr(cuda.memory, name, None)
        result[name] = attempt(function) if function else {"unavailable": "API absent"}
    return result


def save_report(path, report):
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "state": "running", "case": args.case, "size": args.size,
        "profile": args.profile, "batch": 1, "nc": 5, "seed": 42,
        "scientific_training_executed": False, "snapshots": [],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_commit": command(["git", "rev-parse", "HEAD"]),
        "source_diff": command(["git", "diff", "--stat"]),
        "initial_system": system_snapshot(), "phase": "imports",
    }
    path = output / "diagnosis.json"
    save_report(path, report)
    torch = None

    def checkpoint(phase):
        report["phase"] = phase
        report["snapshots"].append({"phase": phase, "cuda": cuda_snapshot(torch)})
        save_report(path, report)

    try:
        # Preserve 756cdc0 ordering, before importing any pipeline runtime.
        import torch
        import ultralytics
        from ultralytics import settings  # noqa: F401

        report["versions"] = {"torch": torch.__version__, "ultralytics": ultralytics.__version__,
                              "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version()}
        if torch.__version__.split("+")[0] != "2.7.1" or ultralytics.__version__ != "8.3.203":
            raise ValueError("Expected frozen torch 2.7.1 and ultralytics 8.3.203")
        if not torch.cuda.is_available():
            raise ValueError("CUDA required; no CPU fallback")
        checkpoint("before_configuration")
        if args.profile == "configured":
            from flir_pipeline.detection.runtime import configure_runtime
            configure_runtime()
        from ultralytics.utils.torch_utils import init_seeds
        init_seeds(42, deterministic=True)
        torch.cuda.init()
        checkpoint("cuda_initialized")
        report["flags"] = {
            "grad_enabled": torch.is_grad_enabled(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        }
        report["nvidia_smi"] = command(["nvidia-smi"])
        if os.environ.get("SLURM_JOB_ID"):
            report["slurm"] = command(["scontrol", "show", "job", os.environ["SLURM_JOB_ID"]])
        checkpoint("runtime_ready")
        if args.case == "smoke":
            from flir_pipeline.detection.common_runtime import synthetic_smoke
            synthetic_smoke(args.config, output / "official-smoke", batch=1, cpu=False)
        else:
            from types import SimpleNamespace

            from ultralytics.nn.tasks import DetectionModel
            model = DetectionModel("yolo11m.yaml", ch=3, nc=5).cuda().train()
            model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.937, weight_decay=0.0005)
            batch = {
                "img": torch.rand(1, 3, args.size, args.size, device="cuda"),
                "batch_idx": torch.zeros(5, device="cuda"),
                "cls": torch.arange(5, device="cuda", dtype=torch.float32).reshape(-1, 1),
                "bboxes": torch.tensor([[0.5, 0.5, 0.3, 0.3]] * 5, device="cuda"),
            }
            optimizer.zero_grad(set_to_none=True)
            checkpoint("before_forward")
            if args.case == "forward":
                prediction = model(batch["img"])
                report["prediction_requires_grad"] = [x.requires_grad for x in prediction]
                torch.cuda.synchronize()
                checkpoint("forward_completed")
            else:
                # A hook only marks the completed network forward; it retains no tensors.
                def forward_complete(module, inputs, output):
                    report["phase"] = "loss"
                model.model[-1].register_forward_hook(forward_complete)
                loss, items = model(batch)
                torch.cuda.synchronize()
                total = loss.sum()
                if not torch.isfinite(total).item():
                    raise ValueError("Nonfinite synthetic loss")
                report["loss_items"] = items.detach().cpu().tolist()
                report["loss_total"] = total.detach().item()
                checkpoint("before_backward")
                total.backward()
                torch.cuda.synchronize()
                checkpoint("before_optimizer")
                optimizer.step()
                torch.cuda.synchronize()
                checkpoint("optimizer_completed")
        report["state"] = "completed"
        return 0
    except Exception as error:
        report.update(state="failed", error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc())
        # Capture before cache clearing or exception unwinding destroys evidence.
        if torch is not None:
            report["failure_cuda"] = attempt(lambda: cuda_snapshot(torch))
        return 1
    finally:
        report["final_system"] = attempt(system_snapshot)
        save_report(path, report)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--case", choices=("forward", "loss", "smoke"), default="loss")
    parser.add_argument("--size", type=int, choices=(160, 320, 640), default=640)
    parser.add_argument("--profile", choices=("bare", "configured"), default="configured")
    parser.add_argument("--config", type=Path, default=Path("configs/detection/common_test_yolo11m_v2.yaml"))
    args = parser.parse_args(argv)
    if args.case == "smoke" and (args.size != 640 or args.profile != "configured"):
        parser.error("Official smoke requires configured runtime and 640px")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
