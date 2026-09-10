import ast
import json
import runpy
import subprocess
from pathlib import Path

import pytest


def test_notebook_check_distinguishes_narrative_from_private_state(tmp_path: Path) -> None:
    check = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/check_notebook_source.py"))["check_notebook_source"]
    path = tmp_path / "source.ipynb"
    notebook = {"cells": [{"cell_type": "markdown", "source": ["Context:\n", "Narrative"]}], "metadata": {}}
    path.write_text(json.dumps(notebook))
    check(path)
    notebook["cells"][0]["source"] = ["C:/private/dataset"]
    path.write_text(json.dumps(notebook))
    with pytest.raises(ValueError, match="absolute path"):
        check(path)
    notebook["cells"][0] = {"cell_type": "code", "source": ["print(1)"], "outputs": [], "execution_count": 0}
    path.write_text(json.dumps(notebook))
    with pytest.raises(ValueError, match="execution state"):
        check(path)


def test_review_source_and_builder_paths_are_generic() -> None:
    """Check the path contract without importing optional notebook libraries."""
    root = Path(__file__).resolve().parents[1]
    source = root / "notebooks/feature_engineering_review.ipynb"
    assert sorted((root / "notebooks").glob("feature_engineering*.ipynb")) == [source]
    check = runpy.run_path(str(root / "scripts/check_notebook_source.py"))["check_notebook_source"]
    check(source)
    notebook = json.loads(source.read_text(encoding="utf-8"))
    assert "".join(notebook["cells"][0]["source"]).splitlines()[0] == "# FLIR Feature Engineering — Progress Review"

    tree = ast.parse((root / "scripts/build_feature_engineering_review.py").read_text(encoding="utf-8"))
    assignments = {
        node.targets[0].id: node.value
        for node in tree.body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
    }

    def components(node):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            return components(node.left) + components(node.right)
        if isinstance(node, ast.Name):
            return [node.id]
        return [ast.literal_eval(node)]

    assert components(assignments["REVIEW_DIR"]) == ["REPORT_DIR", "review"]
    assert components(assignments["SOURCE_NOTEBOOK"]) == ["ROOT", "notebooks", source.name]
    assert components(assignments["EXECUTED_NOTEBOOK"]) == ["REVIEW_DIR", "feature_engineering_review.executed.ipynb"]
    assert components(assignments["HTML_REPORT"]) == ["REVIEW_DIR", "feature_engineering_review.html"]


def test_review_outputs_remain_ignored_without_ignoring_source() -> None:
    """Git checks synthetic path names; no dataset or generated output is needed."""
    root = Path(__file__).resolve().parents[1]
    outputs = [
        "reports/feature_engineering/review/feature_engineering_review.executed.ipynb",
        "reports/feature_engineering/review/feature_engineering_review.html",
    ]
    result = subprocess.run(["git", "check-ignore", "--no-index", *outputs], cwd=root, capture_output=True, text=True, check=True)
    assert result.stdout.splitlines() == outputs
    source = subprocess.run(["git", "check-ignore", "--no-index", "notebooks/feature_engineering_review.ipynb"], cwd=root, capture_output=True, text=True)
    assert source.returncode == 1 and not source.stdout
