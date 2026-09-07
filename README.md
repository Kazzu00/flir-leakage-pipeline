# flir-leakage-pipeline

Research scaffolding for a reproducible thesis pipeline that studies data leakage
in illegal-mining detection from aerial FLIR video frames.

## Research problem

Temporally adjacent video frames can be extremely similar. A random split may
place correlated frames in train, validation, and test, producing an overly
optimistic detector estimate. The methodological hypothesis is that cluster-aware
splits will provide a more realistic estimate of generalization and expose this
leakage.

The planned pipeline is:

`data -> manifest -> embeddings -> similarity -> dimensionality reduction -> clustering -> cluster split -> YOLO detection -> evaluation`

Planned methods include DINOv2, CLIP, cosine similarity, UMAP, PaCMAP, t-SNE,
DBSCAN, HDBSCAN, OPTICS, ARI, AMI, YOLO11, precision, recall, mAP@50, and
mAP@50-95. Pretrained feature extractors remain independent of dataset labels.

## Repository architecture

The `src/flir_pipeline` packages keep ingestion, visual representation, similarity,
reduction, clustering, splitting, detection, and evaluation conceptually separate.
Configuration lives in `configs/`; data manifests and samples live in `data/`;
generated outputs belong in `artifacts/` and `reports/`.

This iteration is scaffolding only. It does not inspect, download, extract, or
process a dataset, and it does not train or run inference for any model.

## Setup

Python 3.11 and [uv](https://docs.astral.sh/uv/) are required:

```powershell
uv sync --extra dev
uv run pytest
uv run ruff check .
uv run ruff format .
uv run flir-pipeline --help
```

Set `FLIR_DATA_ROOT` in a local, ignored `.env` file when the external dataset is
available. Use repository-relative paths in versioned configuration; never commit
absolute machine paths.

## Data and public repository policy

The repository is intended to be public, but real FLIR images, videos, datasets,
archives, model weights, checkpoints, credentials, and `.env` files are external
and must never be committed. Small manifests may be versioned explicitly once their
format is known and they contain no sensitive paths or data.

Development starts on Windows and is intended to continue on an Ubuntu VM. Code
uses `pathlib` and avoids platform-specific path assumptions.

## Next steps

1. Audit the external dataset and define a versioned manifest schema.
2. Add isolated embedding backends and artifact metadata.
3. Implement similarity, reduction, clustering, and leakage-aware split experiments.
4. Add YOLO11 training/evaluation adapters only after the data contract is stable.
