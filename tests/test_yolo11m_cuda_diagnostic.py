"""Offline tests of evidence preservation, never substitutes for a CUDA test."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/hypatia/yolo11m_cuda_diagnostic.py"
SPEC = importlib.util.spec_from_file_location("cuda_diagnostic", SCRIPT)
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)


def test_smoke_cannot_change_resolution_or_runtime():
    args = diagnostic.parse_args(["output", "--case", "smoke"])
    assert args.size == 640 and args.profile == "configured"
    for override in (["--size", "160"], ["--size", "320"], ["--profile", "bare"]):
        with pytest.raises(SystemExit):
            diagnostic.parse_args(["output", "--case", "smoke", *override])


@pytest.mark.parametrize("argument", ["--dataset", "--weights", "--admission", "--cpu"])
def test_probe_has_no_real_data_or_cpu_fallback_route(argument):
    with pytest.raises(SystemExit):
        diagnostic.parse_args(["output", argument, "anything"])


def test_cuda_snapshot_does_not_change_fraction_or_mask_failed_query():
    def unavailable():
        raise RuntimeError("allocator unavailable after CUDA error")

    def forbidden(*args):
        pytest.fail("Diagnostic changed allocator or cleared evidence")

    cuda = SimpleNamespace(
        is_initialized=lambda: True, mem_get_info=lambda: (20, 24),
        memory_stats=unavailable, memory_allocated=lambda: 1, memory_reserved=lambda: 2,
        max_memory_allocated=lambda: 3, max_memory_reserved=lambda: 4,
        empty_cache=forbidden,
        memory=SimpleNamespace(get_per_process_memory_fraction=lambda: 0.1,
                               set_per_process_memory_fraction=forbidden),
    )
    result = diagnostic.cuda_snapshot(SimpleNamespace(cuda=cuda))
    assert result["get_per_process_memory_fraction"] == 0.1
    assert result["mem_get_info"] == (20, 24)
    assert result["memory_stats"]["unavailable"] == "RuntimeError"
    assert result["get_allocator_backend"]["unavailable"] == "API absent"


def test_missing_torch_is_failed_evidence_not_gpu_success(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setattr(diagnostic, "system_snapshot", lambda: {"fixture": True})
    monkeypatch.setattr(diagnostic, "command", lambda argv: {"fixture": True})
    output = tmp_path / "attempt"
    assert diagnostic.run(diagnostic.parse_args([str(output)])) == 1
    report = json.loads((output / "diagnosis.json").read_text())
    assert report["state"] == "failed"
    assert report["error_type"] == "ModuleNotFoundError"
    assert report["phase"] == "imports"
    assert report["scientific_training_executed"] is False
    assert report["final_system"] == {"fixture": True}
    before = (output / "diagnosis.json").read_bytes()
    with pytest.raises(FileExistsError):
        diagnostic.run(diagnostic.parse_args([str(output)]))
    assert (output / "diagnosis.json").read_bytes() == before


def test_uninitialized_snapshot_does_not_initialize_cuda():
    result = diagnostic.cuda_snapshot(SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: False)))
    assert result == {"initialized": False}


@pytest.mark.parametrize("failure_stage", [None, "forward", "backward", "optimizer"])
def test_loss_vector_and_failure_stage_with_synthetic_backend(tmp_path, monkeypatch, failure_stage):
    events = []

    def operation(stage, event):
        events.append(event)
        if failure_stage == stage:
            raise RuntimeError(f"synthetic CUDA failure during {stage}")

    scalar = MagicMock()
    scalar.detach.return_value.item.return_value = 6.0
    scalar.backward.side_effect = lambda: operation("backward", "backward")
    vector = MagicMock()
    vector.sum.side_effect = lambda: (events.append("sum"), scalar)[1]
    vector.item.side_effect = AssertionError("loss has three components")
    vector.backward.side_effect = AssertionError("scalar reduction required")
    items = MagicMock()
    items.detach.return_value.cpu.return_value.tolist.return_value = [1.0, 2.0, 3.0]
    model = MagicMock()
    model.cuda.return_value.train.return_value = model
    if failure_stage == "forward":
        model.side_effect = RuntimeError("synthetic CUDA failure during forward")
    else:
        model.return_value = (vector, items)
    torch = MagicMock()
    torch.__version__ = "2.7.1+cu118"
    torch.version.cuda = "11.8"
    torch.backends.cudnn.version.return_value = 90100
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.is_grad_enabled.return_value = True
    torch.are_deterministic_algorithms_enabled.return_value = True
    torch.cuda.is_available.return_value = True
    torch.isfinite.return_value.item.return_value = True
    torch.optim.SGD.return_value.step.side_effect = lambda: operation("optimizer", "step")
    for name, module in {
        "torch": torch,
        "ultralytics": SimpleNamespace(__version__="8.3.203", settings={}),
        "ultralytics.utils.torch_utils": SimpleNamespace(init_seeds=lambda *a, **k: None),
        "ultralytics.nn.tasks": SimpleNamespace(DetectionModel=lambda *a, **k: model),
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(diagnostic, "system_snapshot", lambda: {"fixture": True})
    monkeypatch.setattr(diagnostic, "command", lambda argv: {"fixture": True})
    monkeypatch.setattr(diagnostic, "cuda_snapshot", lambda t: {"fixture": True})
    output = tmp_path / "attempt"
    result = diagnostic.run(diagnostic.parse_args([str(output), "--profile", "bare"]))
    report = json.loads((output / "diagnosis.json").read_text())
    if failure_stage:
        assert result == 1 and report["state"] == "failed"
        assert report["phase"] == failure_stage
        assert report["last_checkpoint"] == f"before_{failure_stage}"
        assert report["failure_cuda"] == {"fixture": True}
        assert "synthetic CUDA failure" in report["traceback"]
        assert events == {
            "forward": [], "backward": ["sum", "backward"],
            "optimizer": ["sum", "backward", "step"],
        }[failure_stage]
    else:
        assert result == 0 and report["state"] == "completed"
        assert report["loss_total"] == 6.0
        assert events == ["sum", "backward", "step"]
