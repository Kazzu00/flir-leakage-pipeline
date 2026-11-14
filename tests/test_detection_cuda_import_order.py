"""Regression tests for Hypatia CUDA import initialization order."""

import builtins
import sys
import types

import pytest

from flir_pipeline.detection import cli


@pytest.mark.parametrize(
    "command",
    [
        "common_executor_preflight",
        "common_executor_smoke",
        "common_executor_probe",
    ],
)
def test_gpu_dependencies_load_before_common_runtime(
    monkeypatch, tmp_path, command
):
    events = []

    torch_stub = types.ModuleType("torch")

    ultralytics_stub = types.ModuleType("ultralytics")
    ultralytics_stub.settings = object()

    runtime_stub = types.ModuleType(
        "flir_pipeline.detection.common_runtime"
    )

    def fake_operation(*args, **kwargs):
        return {"state": "completed"}

    for operation in ("preflight", "synthetic_smoke", "probe"):
        setattr(runtime_stub, operation, fake_operation)

    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    monkeypatch.setitem(
        sys.modules, "ultralytics", ultralytics_stub
    )
    monkeypatch.setitem(
        sys.modules,
        "flir_pipeline.detection.common_runtime",
        runtime_stub,
    )

    original_import = builtins.__import__

    def tracked_import(
        name, globals=None, locals=None, fromlist=(), level=0
    ):
        if name == "torch":
            events.append("torch")
        elif name == "ultralytics" and "settings" in (fromlist or ()):
            events.append("ultralytics.settings")
        elif name == "flir_pipeline.detection.common_runtime":
            events.append("common_runtime")

        return original_import(
            name, globals, locals, fromlist, level
        )

    monkeypatch.setattr(
        builtins, "__import__", tracked_import
    )
    monkeypatch.setattr(cli.typer, "echo", lambda result: None)

    getattr(cli, command)(tmp_path / "output.json")

    assert events == [
        "torch",
        "ultralytics.settings",
        "common_runtime",
    ]
