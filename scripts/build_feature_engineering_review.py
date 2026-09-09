"""Build the local feature-engineering progress review."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

from flir_pipeline.features.visualization import generate_feature_engineering_report

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "feature_engineering"
REVIEW_DIR = REPORT_DIR / "jp_review"
SOURCE_NOTEBOOK = ROOT / "notebooks" / "feature_engineering_jp_review.ipynb"
EXECUTED_NOTEBOOK = REVIEW_DIR / "feature_engineering_jp_review.executed.ipynb"
HTML_REPORT = REVIEW_DIR / "feature_engineering_jp_review.html"
REQUIRED_FIGURES = [
    "01_dataset_overview.png",
    "02_original_split_distribution.png",
    "03_class_distribution.png",
    "06_bbox_area_distribution.png",
    "09_pixel_statistics.png",
    "10_entropy_distribution.png",
    "11_laplacian_variance.png",
    "12_unique_vs_records.png",
    "13_cross_split_duplicate_matrix.png",
]


def _find_single(pattern: str) -> Path:
    matches = sorted(ROOT.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"Required local output not found: {pattern}")
    return matches[-1]


def _feature_directories() -> dict[str, Path]:
    directories: dict[str, Path] = {}
    for extractor in ("dinov2", "clip"):
        candidates = sorted(
            (ROOT / "artifacts" / "features" / extractor).glob("*/*")
        )
        valid = [item for item in candidates if (item / "metadata.json").is_file()]
        if valid:
            directories[extractor] = valid[-1]
    return directories


def _assert_source_is_clean() -> None:
    notebook = nbformat.read(SOURCE_NOTEBOOK, as_version=4)
    if any(cell.get("outputs") or cell.get("execution_count") for cell in notebook.cells):
        raise ValueError("The versioned review notebook must not contain outputs.")
    serialized = json.dumps(notebook)
    if str(ROOT) in serialized or "base64," in serialized:
        raise ValueError("The versioned review notebook contains local or embedded data.")


def _build_notebook() -> None:
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    notebook = nbformat.read(SOURCE_NOTEBOOK, as_version=4)
    client = NotebookClient(notebook, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}})
    client.execute()
    nbformat.write(notebook, EXECUTED_NOTEBOOK)
    exporter = HTMLExporter(template_name="lab")
    body, _ = exporter.from_notebook_node(notebook)
    HTML_REPORT.write_text(body, encoding="utf-8")


def main() -> None:
    _assert_source_is_clean()
    manifest = _find_single("data/manifests/*.parquet")
    diagnostics = _find_single("artifacts/features/diagnostics/*/*.parquet")
    generate_feature_engineering_report(
        manifest,
        diagnostics,
        REPORT_DIR,
        feature_dirs=_feature_directories(),
    )
    missing = [name for name in REQUIRED_FIGURES if not (REPORT_DIR / "figures" / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Required review figures were not generated: {missing}")
    _build_notebook()
    print(f"Built {EXECUTED_NOTEBOOK.relative_to(ROOT)}")
    print(f"Built {HTML_REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
