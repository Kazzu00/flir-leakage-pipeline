"""Build the local clustering review and runtime ZIP galleries from verified sources."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import nbformat
from check_notebook_source import check_notebook_source
from nbclient import NotebookClient
from nbconvert import HTMLExporter

from flir_pipeline.cli import _default_root
from flir_pipeline.clustering.experiments import load_families
from flir_pipeline.clustering.visualization import generate_clustering_report
from flir_pipeline.similarity.storage import file_sha256, read_json, write_json

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT/"reports"/"clustering"
SOURCE_NOTEBOOK = ROOT/"notebooks"/"clustering_review.ipynb"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--images-archive", type=Path, help="Read-only ZIP; defaults to FLIR_DATA_ROOT/Imagenes.zip")
    args = parser.parse_args()
    check_notebook_source(SOURCE_NOTEBOOK)
    data_root = _default_root()
    images = args.images_archive or (data_root/"Imagenes.zip" if data_root else None)
    if images is None or not images.is_file():
        raise FileNotFoundError("Provide images-archive or configure FLIR_DATA_ROOT")
    generate_clustering_report(args.comparison, load_families(args.inputs), images, REPORT_DIR)
    review = REPORT_DIR/"review"
    review.mkdir(parents=True, exist_ok=True)
    for name, folder in (("IPYTHONDIR", "ipython"), ("JUPYTER_RUNTIME_DIR", "jupyter-runtime")):
        runtime = ROOT/".cache"/folder
        runtime.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(name, str(runtime))
    notebook = nbformat.read(SOURCE_NOTEBOOK, as_version=4)
    NotebookClient(notebook, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
    nbformat.validate(notebook)
    executed = review/"clustering_review.executed.ipynb"
    nbformat.write(notebook, executed)
    exporter = HTMLExporter(template_name="lab", exclude_input=True, exclude_input_prompt=True, exclude_output_prompt=True)
    body, _ = exporter.from_notebook_node(notebook, resources={"metadata": {"name": "FLIR Density-Based Clustering — Progress Review"}})
    html = review/"clustering_review.html"
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
