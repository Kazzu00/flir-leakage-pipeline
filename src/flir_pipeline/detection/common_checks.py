"""Pure argument/checkpoint checks usable before optional runtime imports."""

from pathlib import Path

import pandas as pd

from flir_pipeline.detection.common_executor import VERSION
from flir_pipeline.detection.common_test_storage import signatures
from flir_pipeline.similarity.storage import file_sha256, read_json


def check_arguments(args: dict, config: dict, seed: int, dataset: Path) -> None:
    required = {**config["train"], "seed": seed, "split": "val"}
    if any(args.get(key) != value for key, value in required.items()):
        raise ValueError("Effective training configuration changed")
    if Path(args["data"]).resolve() != dataset.resolve():
        raise ValueError("Training dataset locator changed")


def check_resume_state(checkpoint: dict, csv: Path, epochs: int) -> None:
    history = pd.read_csv(csv)
    if (
        not 0 < len(history) < epochs
        or history.epoch.tolist() != list(range(1, len(history) + 1))
        or checkpoint.get("epoch") != len(history) - 1
        or checkpoint.get("optimizer") is None
    ):
        raise ValueError("Checkpoint epoch/optimizer differs from interrupted history")


def verify_operation(output: Path) -> dict:
    """Verify a synthetic receipt without importing the GPU operations module."""
    from flir_pipeline.detection.runtime import parse_training_history

    expected = read_json(output / "receipt.json")["artifacts"]
    actual = signatures(output)
    actual.pop("receipt.json", None)
    if actual != expected:
        raise ValueError("Operation/checkpoint checksum mismatch")
    state = read_json(output / "state.json")
    if state["schema_version"] != VERSION or state["mode"] != "synthetic-smoke":
        raise ValueError("Not a synthetic executor operation")
    if state["state"] == "completed":
        parse_training_history(output / "training/results.csv", 1)
        for name in ("best.pt", "last.pt"):
            if (
                file_sha256(output / "training/weights" / name)
                != state["checkpoint_sha256"][name]
            ):
                raise ValueError("Output checkpoint provenance mismatch")
    return state
