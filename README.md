# FLIR Leakage-Aware Clustering Pipeline

## Overview

A reproducible pipeline for auditing FLIR frames, measuring visual correlation,
grouping related contents and constructing leakage-aware train/validation/test
partitions. Exact copies and nearby scenes can cross a frame-level split;
content identities and indivisible groups make those relationships auditable.

The system compares historical, reproducible random and cluster-aware partitions
using residual visual and inferred temporal relationships. Reduced correlation
has been observed for selected candidates; improved detector generalization
remains untested by a full controlled comparison.

## Pipeline

```text
Data audit & canonicalization
        ↓
Visual representations
        ↓
Similarity analysis
        ↓
Dimensionality reduction
        ↓
Density-based clustering
        ↓
Cluster-aware splitting
        ↓
Interactive inspection
        ↓
Detector evaluation
```

See [inputs, outputs and execution order](docs/pipeline.md).

## Key capabilities

- ZIP-safe audit, annotation QA and canonical occurrence/content identities.
- Independent DINOv2 CLS and CLIP projected-image embeddings, raw/L2 stores,
  pinned model revisions and resumable extraction.
- Content-level cosine similarity, neighborhoods and posterior temporal analysis.
- t-SNE / PaCMAP with preservation metrics and seed stability.
- DBSCAN / OPTICS / HDBSCAN with original-space metrics, ARI/AMI and explicit noise.
- Atomic cluster-aware splitting, seeded baselines and residual cross-split analysis.
- Local Streamlit inspection and VIKUS collection overview.
- Controlled YOLO11n evaluation infrastructure, bootstrap and small CPU pilots.

## Architecture

`src/flir_pipeline/` separates `data`, `features`, `similarity`, `reduction`,
`clustering`, `splitting`, `detection`, `explorer` and `utils`. Metrics live beside
their experiments; optional model/UI dependencies load only where needed.
Source notebooks and report builders consume those components.

See [architecture](docs/architecture.md) and the [data model](docs/data_model.md).

## Quick start

Python 3.11 and `uv` are required. Run commands from the repository root:

```powershell
git clone https://github.com/Kazzu00/flir-leakage-pipeline.git
cd flir-leakage-pipeline
uv sync --locked --extra dev --extra reduction
uv run flir-pipeline --help
```

Create an ignored `.env` using [.env.example](.env.example) as a reference:

```dotenv
FLIR_DATA_ROOT=/path/to/external/flir-data
```

The environment takes precedence over `.env`; data commands also accept explicit
paths. Public clones contain no real data or executed artifacts.
Optional extras are `vision`, `reporting`, `explorer` and `detection`.
`uv sync` installs exactly the selected extras; include all extras you need.
Model weights are separate from the lockfile. See [runbooks](docs/README.md).

## CLI

Use `uv run flir-pipeline <namespace> --help` for options.

| Namespace | Implemented commands |
|---|---|
| `data` | `inventory`, `archive-tree`, `compare-archives`, `build-manifest`, `validate-labels`, `manifest-summary` |
| `features` | `extract`, `diagnostics`, `summary`, `verify`, `visualize-data`, `visualize-embeddings` |
| `similarity` | `compute`, `summary`, `verify`, `compare` |
| `reduction` | `run`, `benchmark`, `verify`, `summary` |
| `clustering` | `run`, `sweep`, `compare`, `verify`, `summary` |
| `splitting` | `build`, `baseline`, `evaluate`, `compare`, `summary`, `verify`, `export-lists` |
| `detection` | `plan`, `materialize`, `environment`, `smoke`, `probe`, `pilot-small`, `freeze`, `run`, `verify` |
| `explorer` | `vikus-build`, `vikus-serve` |

## Interactive exploration

[Streamlit](docs/visualization/streamlit.md) provides detailed cluster/split
inspection, galleries, reconstructed frame playback, timelines, gaps and comparisons:

```powershell
uv run --extra explorer streamlit run apps/cluster_split_explorer.py
```

[VIKUS](docs/visualization/vikus.md) provides a global collection view by cluster,
sequence, split and saved PaCMAP coordinates. For the existing
**C10 — DINOv2 / PaCMAP / DBSCAN** candidate:

```powershell
uv run flir-pipeline explorer vikus-build --candidate C10 --seed 0 --name c10-seed0-local
uv run flir-pipeline explorer vikus-serve --bundle reports/explorer/vikus/c10-seed0-local
```

Both require existing local artifacts and read original ZIPs without mutation.
Playback FPS controls display only; there are no verified timestamps.
The first VIKUS build downloads a checksum-pinned runtime; cached builds support
`--offline`. Image bundles remain private and local.

The separate [Detector & Residual Similarity Explorer](docs/visualization/detector_similarity.md)
shows the 48-cell experiment matrix and frozen residual context now. Run-level
scatters activate only after the complete controlled comparison; pilots are excluded.

```powershell
uv run --extra explorer streamlit run apps/detector_similarity_explorer.py
```

## Data and artifacts

Original ZIPs live outside Git under `FLIR_DATA_ROOT` and remain read-only.
Local manifests, embeddings, assignments, previews, VIKUS bundles, figures,
executed notebooks and HTML are protected by [.gitignore](.gitignore).
Only source code, generic configurations, aggregate documentation and clean
notebook sources are versioned. See [data safety](docs/data_safety.md).

## Reproducibility

Deterministic dataset/space IDs, configurations, seeds, source checksums and
artifact metadata preserve lineage from each `frame_id` to its `content_id`,
embedding row, cluster and partition. Historical membership and labels never
enter visual representation or clustering. Labels support later split balance.
See [reproducibility](docs/reproducibility.md), [configuration status](configs/README.md)
and [pipeline traceability](docs/pipeline_traceability.md).

```powershell
uv run ruff check .
uv run python -m pytest
uv run python scripts/check_notebook_source.py
```

Tests are synthetic and offline, with no FLIR data, model downloads or GPU.

## Current status

| Component | State |
|---|---|
| Audit and canonicalization | Validated: 1657 historical records / 1459 unique contents |
| DINOv2 / CLIP and cosine similarity | Validated on complete content coverage |
| t-SNE / PaCMAP | Validated: 36 executed reductions |
| Density clustering | Validated with limitations: 414 runs; exploratory candidates |
| Cluster-aware splitting | Validated with limitations: 66 runs; residual correlation remains |
| Temporal interpretation | Experimental: filename-derived indices; no verified timing |
| Streamlit / VIKUS | Available for local inspection |
| Detector infrastructure | Available; four small CPU pilots validated |
| Full detector comparison | Pending compute; no final comparative Precision/Recall/mAP results |

Detailed evidence, candidate roles and remaining limits are in [status](docs/status.md).

## Documentation

Start with the [documentation index](docs/README.md): architecture, data model,
pipeline, reproducibility, safety, status and design decisions; functional
analysis, protocols, runbooks and visualization guides.
The [consolidated project report](docs/runbooks/project_report.md) reuses existing
results without rerunning experiments. Technical sources are in
[references](docs/references.md).
