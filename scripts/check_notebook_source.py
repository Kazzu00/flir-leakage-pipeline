"""Check the versioned notebook offline using only the Python standard library."""

from __future__ import annotations

import json
import re
from pathlib import Path


def check_notebook_source(path: Path) -> None:
    """Reject outputs, execution state, attachments, absolute paths and raw hashes.

    Errors identify the rule and cell, never echo potentially sensitive content.
    Code is compiled for syntax without execution or any notebook dependencies.
    """
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("outputs") or cell.get("execution_count") is not None or cell.get("attachments"):
            raise ValueError(f"Cell {index} contains execution state or attachments")
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"notebook cell {index}", "exec")
    def strings(value):
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for child in value.values():
                yield from strings(child)
        elif isinstance(value, list):
            for child in value:
                yield from strings(child)

    serialized = "\n".join(strings(notebook))
    patterns = {
        "absolute path": r"[A-Za-z]:[/\\]|/(?:home|Users|mnt)/",
        "raw hash": r"\b[0-9a-fA-F]{40,64}\b",
        "embedded data": r"base64,|application/vnd\.jupyter\.widget",
    }
    for name, pattern in patterns.items():
        if re.search(pattern, serialized):
            raise ValueError(f"Notebook contains {name}")


if __name__ == "__main__":
    check_notebook_source(Path(__file__).resolve().parents[1] / "notebooks" / "feature_engineering_jp_review.ipynb")
    print("Notebook source is clean and code cells compile.")
