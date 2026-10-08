# FLIR Leakage-Aware Clustering Pipeline

## Overview

`flir-pipeline data video-variant-ingestion build/verify/summary` audits external
image variants directly from ZIP shards with bounded reads and immutable evidence.
Scientific occurrence IDs survive unambiguous repackaging; duplicate indices
remain conflicts. Alignment is candidate-only and the historical labeled universe
stays separate. Available with synthetic offline validation; real ingestion and
downstream consumers remain pending. See the
[variant ingestion runbook](docs/runbooks/video_variant_ingestion.md).

`flir-pipeline explorer export-organization` exports existing frozen M02
record/content, split, clustering, candidate pairs and grouped evidence through
`organization-evidence-v2`. Timelines preserve provenance and optional previews
remain local/ignored. Available with synthetic offline validation; final Hypatia
memberships have not been exported locally. See the
[organization contract and Hypatia runbook](docs/runbooks/organization_evidence_export.md).

`flir-pipeline detection report` implements final evidence-gated detector analysis
and the versioned `detection-export-v1` contract for `flir-pipeline-explorer`.
The owner reports the completed 48-run experiment on Hypatia; those artifacts
are absent from this local clone and have not been reverified here. The command
derives identities from its inputs and refuses incomplete scientific exports.
See the [two-environment reporting runbook](docs/runbooks/detector_final_report.md).

The experimental `flir-pipeline sequences experiment` suite evaluates temporal
boundary zones, visual recurrence and clustering with explicit review masks,
coverage, stability and immutable evidence. It creates no final split or automatic
dependency confirmations. Available with synthetic offline validation; real
Hypatia runs remain pending. See the [Hypatia commands and review workflow](docs/runbooks/sequences.md#suite-experimental-de-temporalidad-y-dependencia-visual)
and [experimental protocol](docs/protocols/sequence_experiments.md). The
[parallel SLURM workflow](docs/runbooks/sequence_operations.md) adds dry-run,
source resolution, durable stage receipts, status and a combined review checkpoint.
Native ingestion now supports the four owner-supplied Hypatia legacy schemas,
with frozen snapshots, cross-report checks and verification of original files.
See the [native import commands](docs/runbooks/native_sequence_evidence.md);
real Hypatia ingestion remains to be executed and verified there.
Generic dataset variants isolate feature stores and experiment identities;
`compare-variants` compares completed suites with optional explicit frame pairing.
See the [future variant example](docs/runbooks/sequence_operations.md#variantes-de-dataset-y-comparación-futura).
Real no-HUD data has not been evaluated; no HUD effect is inferred.

A reproducible pipeline for auditing FLIR frames, measuring visual correlation,
grouping related contents and constructing leakage-aware train/validation/test
partitions. Exact copies and nearby scenes can cross a frame-level split;
content identities and indivisible groups make those relationships auditable.

The system compares historical, reproducible random and cluster-aware partitions
using residual visual and inferred temporal relationships. Reduced correlation
has been observed for selected candidates. The owner reports a complete detector
comparison on Hypatia, pending verification by the final report command there;
detector differences remain descriptive, not causal evidence of leakage inflation.

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
- Reproducible source-video sampling through external FFmpeg/ffprobe, operationally
  validated on Hypatia: a real 5-second smoke and job 737719 (COMPLETED, exit 0:0),
  producing 9648 JPEGs from three source videos at 1 FPS. This does not identify
  sequences. See the [execution evidence](docs/runbooks/data.md#evidencia-real-confirmada-en-hypatia).
- Sampled JPEGs → versioned occurrence/content manifest → shared DINOv2/CLIP
  extraction from a read-only directory, with exact-byte deduplication. This
  bridge has full Hypatia validation reported by the owner, not reverified locally;
  reviewed video sequences are also reported by the owner and remain unverified
  locally. See [evidence boundaries](docs/status.md).
- Independent DINOv2 CLS and CLIP projected-image embeddings, raw/L2 stores,
  pinned model revisions and resumable extraction.
- Content-level cosine similarity, neighborhoods and posterior temporal analysis.
- Occurrence-level sequence candidates from original CLIP/DINOv2, explicit confirmed
  manual boundary consumption, sequence instances and exact-copy dependency components.
  Synthetic validation only; see the [sequence runbook](docs/runbooks/sequences.md).
- Cross-dataset labeled/video candidate linkage: union of independent CLIP/DINOv2
  top-k sets, separate scores and complete occurrence/sequence ambiguity.
  Synthetic validation only; see the [linkage runbook](docs/runbooks/linkage.md).
- Manual group-level linkage calibration with temporal contact sheets, auditable
  decisions, stratified summaries and source-bound aggregation of immutable
  revisions; no confirmed links or split are created. Real Hypatia v1 init/verify
  compatibility is reported by the owner; aggregation is tested synthetically,
  and future calibration strata are not claimed complete.
  See the [manual review contract](docs/runbooks/linkage_review.md).
- Explicit sampled-video similarity v2 with complete occurrence provenance, exact
  summaries and optional streamed full pairs. Numerical reduction/clustering accept
  this contract; local validation is synthetic, real video execution remains pending.
- t-SNE / PaCMAP with preservation metrics and seed stability.
- DBSCAN / OPTICS / HDBSCAN with original-space metrics, ARI/AMI and explicit noise.
  Exact distances are computed on demand with scoped reuse; see the
  [memory audit and limits](docs/analysis/clustering_memory.md).
- Atomic cluster-aware splitting, seeded baselines and residual cross-split analysis.
- Local Streamlit inspection and VIKUS collection overview.
- Controlled YOLO11n evaluation infrastructure, bootstrap and small CPU pilots.

## Architecture

`src/flir_pipeline/` separates `data`, `features`, `similarity`, `reduction`,
`clustering`, `sequences`, `linkage`, `splitting`, `detection`, `explorer` and `utils`. Metrics live beside
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
| `data` | `inventory`, `extract-video-frames`, `build-video-manifest`, `archive-tree`, `compare-archives`, `build-manifest`, `validate-labels`, `manifest-summary` |
| `features` | `extract`, `diagnostics`, `summary`, `verify`, `visualize-data`, `visualize-embeddings` |
| `similarity` | `compute`, `summary`, `verify`, `compare` |
| `reduction` | `run`, `benchmark`, `verify`, `summary` |
| `clustering` | `run`, `sweep`, `compare`, `verify`, `summary` |
| `sequences` | `detect`, `build`, `verify`, `summary` |
| `linkage` | `build`, `verify`, `summary`; `review init/record/summary/verify/aggregate/aggregate-verify` |
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
| Full detector comparison | 48/48 complete on Hypatia reported by owner; final-report verification pending there, no local scientific export |
| Detector report / frontend contract | Implemented; synthetic offline validation, final-only integrity gate and versioned JSON Schema |

Detailed evidence, candidate roles and remaining limits are in [status](docs/status.md).

## Documentation

Start with the [documentation index](docs/README.md): architecture, data model,
pipeline, reproducibility, safety, status and design decisions; functional
analysis, protocols, runbooks and visualization guides.
The [consolidated project report](docs/runbooks/project_report.md) reuses existing
results without rerunning experiments. Technical sources are in
[references](docs/references.md).
