"""Build the local similarity review from explicitly selected full artifacts."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import nbformat
from check_notebook_source import check_notebook_source
from nbclient import NotebookClient
from nbconvert import HTMLExporter

from flir_pipeline.cli import _default_root
from flir_pipeline.similarity.reporting import FIGURE_NAMES, generate_similarity_report
from flir_pipeline.similarity.storage import file_sha256, read_json, write_json

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "similarity"
REVIEW_DIR = REPORT_DIR / "review"
SOURCE_NOTEBOOK = ROOT / "notebooks" / "similarity_review.ipynb"
EXECUTED_NOTEBOOK = REVIEW_DIR / "similarity_review.executed.ipynb"
HTML_REPORT = REVIEW_DIR / "similarity_review.html"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dinov2", type=Path, required=True, help="Verified full DINOv2 similarity directory")
    parser.add_argument("--clip", type=Path, required=True, help="Verified full CLIP similarity directory")
    parser.add_argument("--comparison", type=Path, required=True, help="Agreement directory: left DINOv2, right CLIP")
    parser.add_argument("--images-archive", type=Path, help="Read-only source; defaults to FLIR_DATA_ROOT/Imagenes.zip")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--examples", type=int, default=3)
    args = parser.parse_args()
    check_notebook_source(SOURCE_NOTEBOOK)
    data_root = _default_root()
    images = args.images_archive or (data_root/"Imagenes.zip" if data_root else None)
    if images is None or not images.is_file():
        raise FileNotFoundError("Provide --images-archive or configure FLIR_DATA_ROOT")
    generate_similarity_report(args.dinov2, args.clip, args.comparison, images, REPORT_DIR, args.seed, args.examples)
    if not all((REPORT_DIR/"figures"/name).is_file() for name in FIGURE_NAMES):
        raise ValueError("Missing required similarity review figure")
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    for name, folder in (("IPYTHONDIR", "ipython"), ("JUPYTER_RUNTIME_DIR", "jupyter-runtime")):
        runtime = ROOT/".cache"/folder
        runtime.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(name, str(runtime))
    notebook = nbformat.read(SOURCE_NOTEBOOK, as_version=4)
    NotebookClient(notebook, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
    nbformat.validate(notebook)
    nbformat.write(notebook, EXECUTED_NOTEBOOK)
    exporter = HTMLExporter(template_name="lab", exclude_input=True, exclude_input_prompt=True, exclude_output_prompt=True)
    body, _ = exporter.from_notebook_node(notebook, resources={"metadata": {"name": "FLIR Similarity and Spatiotemporal Correlation — Progress Review"}})
    HTML_REPORT.write_text(body, encoding="utf-8")
    metadata = read_json(REPORT_DIR/"report_metadata.json")
    metadata["notebook_source_sha256"] = file_sha256(SOURCE_NOTEBOOK)
    for path in (EXECUTED_NOTEBOOK, HTML_REPORT):
        metadata["output_sha256"][path.relative_to(REPORT_DIR).as_posix()] = file_sha256(path)
    write_json(REPORT_DIR/"report_metadata.json", metadata)
    print(f"Built {EXECUTED_NOTEBOOK.relative_to(ROOT)}")
    print(f"Built {HTML_REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
