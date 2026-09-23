import ast
import json
import re
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
    assert "".join(notebook["cells"][0]["source"]).splitlines()[0] == "# FLIR Feature Engineering — Project Report"

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


def test_similarity_review_source_has_complete_narrative_without_execution_state():
    root = Path(__file__).resolve().parents[1]
    source = root/"notebooks/similarity_review.ipynb"
    check = runpy.run_path(str(root/"scripts/check_notebook_source.py"))["check_notebook_source"]
    check(source)
    notebook = json.loads(source.read_text(encoding="utf-8"))
    assert "".join(notebook["cells"][0]["source"]).splitlines()[0] == "# FLIR Similarity and Spatiotemporal Correlation — Project Report"
    sections = ["".join(cell["source"]).splitlines()[0] for cell in notebook["cells"] if cell["cell_type"] == "markdown" and "".join(cell["source"]).startswith("## ")]
    assert len(sections) == 14
    assert all(heading.startswith(f"## {i}. ") for i, heading in enumerate(sections, 1))
    ignored = ["reports/similarity/review/similarity_review.executed.ipynb", "reports/similarity/review/similarity_review.html", "artifacts/similarity/synthetic/cosine_similarity.npy", "reports/similarity/figures/nearest_neighbors_dinov2.png"]
    result = subprocess.run(["git", "check-ignore", "--no-index", *ignored], cwd=root, text=True, capture_output=True, check=True)
    assert result.stdout.splitlines() == ignored


def test_reduction_review_preserves_source_and_fourteen_sections():
    root = Path(__file__).resolve().parents[1]
    source = root/"notebooks/reduction_review.ipynb"
    check = runpy.run_path(str(root/"scripts/check_notebook_source.py"))["check_notebook_source"]
    check(source)
    notebook = json.loads(source.read_text(encoding="utf-8"))
    narrative = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "markdown")
    assert narrative.startswith("# FLIR Dimensionality Reduction — Project Report")
    assert re.findall(r"^## (\d+)\.", narrative, re.MULTILINE) == [str(i) for i in range(1, 15)]
    ignored = ["reports/reduction/review/reduction_review.executed.ipynb", "reports/reduction/review/reduction_review.html",
               "reports/reduction/figures/01_reduction_quality_dinov2.png", "artifacts/reduction/synthetic/coordinates.npy"]
    result = subprocess.run(["git", "check-ignore", "--no-index", *ignored], cwd=root, text=True, capture_output=True, check=True)
    assert result.stdout.splitlines() == ignored


def test_splitting_review_has_eighteen_clean_sections():
    root = Path(__file__).resolve().parents[1]
    source = root / "notebooks/splitting_review.ipynb"
    check = runpy.run_path(str(root / "scripts/check_notebook_source.py"))["check_notebook_source"]
    check(source)
    notebook = json.loads(source.read_text(encoding="utf-8"))
    narrative = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "markdown")
    assert narrative.startswith("# FLIR Cluster-Aware Splitting — Project Report")
    assert re.findall(r"^## (\d+)\.", narrative, re.MULTILINE) == [str(i) for i in range(1, 19)]
    ignored = ["reports/splitting/review/splitting_review.executed.ipynb", "reports/splitting/review/splitting_review.html",
               "reports/splitting/figures/01_split_sizes_comparison.png", "artifacts/splitting/runs/synthetic/split_assignments.parquet"]
    result = subprocess.run(["git", "check-ignore", "--no-index", *ignored], cwd=root, text=True, capture_output=True, check=True)
    assert result.stdout.splitlines() == ignored
