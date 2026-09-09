"""Build the local feature-engineering progress review."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from check_notebook_source import check_notebook_source
from nbclient import NotebookClient
from nbconvert import HTMLExporter

from flir_pipeline.features.visualization import (
    discover_feature_directories,
    generate_feature_engineering_report,
)

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
    if len(matches) != 1:
        raise ValueError(f"Ambiguous local outputs for {pattern}; select the input explicitly")
    return matches[0]


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
    """Build an executed local review from explicit or unambiguous existing outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--diagnostics", type=Path)
    parser.add_argument("--dinov2", type=Path, help="Existing DINOv2 feature directory")
    parser.add_argument("--clip", type=Path, help="Existing CLIP feature directory")
    args = parser.parse_args()
    check_notebook_source(SOURCE_NOTEBOOK)
    manifest = args.manifest or _find_single("data/manifests/*.parquet")
    diagnostics = args.diagnostics or _find_single("artifacts/features/diagnostics/*/*.parquet")
    selected = {name: path for name in ("dinov2", "clip") if (path := getattr(args, name)) is not None}
    if not selected:
        selected = discover_feature_directories(ROOT / "artifacts" / "features")
    generate_feature_engineering_report(
        manifest,
        diagnostics,
        REPORT_DIR,
        feature_dirs=selected,
    )
    missing = [name for name in REQUIRED_FIGURES if not (REPORT_DIR / "figures" / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Required review figures were not generated: {missing}")
    _build_notebook()
    print(f"Built {EXECUTED_NOTEBOOK.relative_to(ROOT)}")
    print(f"Built {HTML_REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
