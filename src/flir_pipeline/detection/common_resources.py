"""Operational observations kept separate from scientific/environment identities."""

from __future__ import annotations

import os
import platform


def snapshot() -> dict:
    result = {
        "python": platform.python_version(),
        "pid": os.getpid(),
        "host": platform.node(),
        "scheduler": {
            k: os.environ.get(k)
            for k in (
                "SLURM_JOB_ID",
                "SLURM_ARRAY_JOB_ID",
                "SLURM_ARRAY_TASK_ID",
                "SLURM_MEM_PER_NODE",
                "SLURM_MEM_PER_CPU",
                "SLURM_CPUS_PER_TASK",
                "CUDA_VISIBLE_DEVICES",
                "LOADEDMODULES",
                "PYTORCH_CUDA_ALLOC_CONF",
                "CUBLAS_WORKSPACE_CONFIG",
            )
        },
        "limits": {},
    }
    try:
        import psutil

        memory = psutil.Process().memory_info()
        result.update(rss_bytes=memory.rss, vms_bytes=memory.vms)
    except ImportError:
        # Offline core tests do not install the optional detection dependency.
        result["process_memory_unavailable"] = "psutil not installed"
    if os.name == "posix":
        import resource

        result["limits"] = {
            name: list(resource.getrlimit(getattr(resource, name)))
            for name in ("RLIMIT_AS", "RLIMIT_DATA", "RLIMIT_MEMLOCK", "RLIMIT_STACK")
        }
    try:
        import torch

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            result["cuda"] = {
                "device": torch.cuda.get_device_name(0),
                "free_bytes": free,
                "total_bytes": total,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "cudnn": torch.backends.cudnn.version(),
                "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                "tf32_cudnn": torch.backends.cudnn.allow_tf32,
            }
    except (ImportError, RuntimeError, AttributeError) as exc:
        result["cuda_observation_error"] = str(exc)
    return result
