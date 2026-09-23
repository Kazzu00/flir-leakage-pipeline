"""Build the local dimensionality reduction review from explicit verified sources."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import nbformat
from check_notebook_source import check_notebook_source
from nbclient import NotebookClient
from nbconvert import HTMLExporter

from flir_pipeline.reduction.visualization import generate_reduction_report
from flir_pipeline.similarity.storage import file_sha256, read_json, write_json

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT/"reports"/"reduction"
SOURCE_NOTEBOOK = ROOT/"notebooks"/"reduction_review.ipynb"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dinov2", type=Path, required=True, help="Verified DINOv2 benchmark directory")
    parser.add_argument("--clip", type=Path, required=True, help="Verified CLIP benchmark directory")
    parser.add_argument("--dinov2-similarity", type=Path, required=True, help="Matching posterior provenance")
    parser.add_argument("--clip-similarity", type=Path, required=True, help="Matching posterior provenance")
    args = parser.parse_args()
    check_notebook_source(SOURCE_NOTEBOOK)
    generate_reduction_report({"dinov2": args.dinov2, "clip": args.clip},
                              {"dinov2": args.dinov2_similarity, "clip": args.clip_similarity}, REPORT_DIR)
    review = REPORT_DIR/"review"
    review.mkdir(parents=True, exist_ok=True)
    for name, folder in (("IPYTHONDIR", "ipython"), ("JUPYTER_RUNTIME_DIR", "jupyter-runtime")):
        runtime = ROOT/".cache"/folder
        runtime.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(name, str(runtime))
    notebook = nbformat.read(SOURCE_NOTEBOOK, as_version=4)
    NotebookClient(notebook, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
    nbformat.validate(notebook)
    executed = review/"reduction_review.executed.ipynb"
    nbformat.write(notebook, executed)
    exporter = HTMLExporter(template_name="lab", exclude_input=True, exclude_input_prompt=True, exclude_output_prompt=True)
    body, _ = exporter.from_notebook_node(notebook, resources={"metadata": {"name": "FLIR Dimensionality Reduction — Project Report"}})
    html = review/"reduction_review.html"
    html.write_text(body, encoding="utf-8")
    metadata = read_json(REPORT_DIR/"report_metadata.json")
    metadata["notebook_source_sha256"] = file_sha256(SOURCE_NOTEBOOK)
    for path in (executed, html):
        metadata["output_sha256"][path.relative_to(REPORT_DIR).as_posix()] = file_sha256(path)
    write_json(REPORT_DIR/"report_metadata.json", metadata)
    print(f"Built {executed.relative_to(ROOT)}")
    print(f"Built {html.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
