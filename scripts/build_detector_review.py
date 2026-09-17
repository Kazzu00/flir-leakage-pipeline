"""Render an evidence-gated detector notebook and local HTML with hidden code."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import nbformat
from check_notebook_source import check_notebook_source
from nbclient import NotebookClient
from nbconvert import HTMLExporter

from flir_pipeline.detection.reporting import generate_report
from flir_pipeline.similarity.storage import file_sha256, read_json, write_json

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("artifacts/detection/protocol"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/detection"))
    args = parser.parse_args()
    output = ROOT/"reports/detection"
    source = ROOT/"notebooks/detector_comparison_review.ipynb"
    check_notebook_source(source)
    generate_report(args.plan, args.artifacts, output)
    for key, name in (("IPYTHONDIR", "ipython"), ("JUPYTER_RUNTIME_DIR", "jupyter-runtime")):
        path = ROOT/".cache"/name
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    notebook = nbformat.read(source, as_version=4)
    NotebookClient(notebook, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
    nbformat.validate(notebook)
    review = output/"review"
    review.mkdir(parents=True, exist_ok=True)
    executed = review/"detector_comparison_review.executed.ipynb"
    nbformat.write(notebook, executed)
    exporter = HTMLExporter(template_name="lab", exclude_input=True, exclude_input_prompt=True, exclude_output_prompt=True)
    body, _ = exporter.from_notebook_node(notebook, resources={"metadata": {"name": "FLIR · comparación controlada del detector"}})
    html = review/"detector_comparison_review.html"
    html.write_text(body, encoding="utf-8")
    metadata = read_json(output/"report_metadata.json")
    metadata["notebook_source_sha256"] = file_sha256(source)
    for p in (executed, html):
        metadata["output_sha256"][p.relative_to(output).as_posix()] = file_sha256(p)
    write_json(output/"report_metadata.json", metadata)
    print(f"Built {html.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
