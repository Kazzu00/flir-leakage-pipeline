# FLIR Leakage-Aware Clustering Pipeline

## Overview

Video-derived FLIR frames have strong visual and temporal correlation. A naïve
frame-level split can place exact copies or closely related scenes across
train/val/test, making detector evaluation overly optimistic. This research
repository studies visual grouping and traceable partitioning to investigate
that leakage. Historical exact overlap is already measured; improvements in
detector generalization remain a hypothesis to evaluate.

## Research context

This work contributes to the thesis proposal **“Desarrollo de un pipeline de
agrupación, procesamiento y detección de minería ilegal en videos FLIR de la
Amazonía colombiana.”** It follows the first four CRISP-ML(Q) phases: business
and data understanding, data preparation, modeling, and evaluation. Production
deployment and production monitoring/maintenance are outside the research scope.

## Individual scope

**INDIVIDUAL RESEARCH SCOPE:** dataset characterization, visual representation,
frame similarity, dimensionality reduction, clustering and cluster selection,
cluster-aware partitioning, and evaluation of partition quality. Each scene or
cluster must eventually remain entirely within one partition, with traceability
and adequate class coverage.

## Group integration scope

**GROUP INTEGRATION SCOPE:** noise cleaning, panoptic segmentation and final
group pipeline assembly. These belong to the broader team project and are
integration targets. This repository does not implement those components or
claim the work of other contributors. The proposed handoff is documented in
[architecture](docs/architecture.md).

## Research pipeline

| Stage | Current state |
|---|---|
| Data understanding and exact duplicate audit | DONE |
| Canonical candidate manifest | DONE |
| Feature engineering infrastructure and reporting | IN PROGRESS; real smoke tests validated |
| Full DINOv2 / CLIP extraction | NEXT; neither full run executed |
| Cosine similarity | PLANNED |
| t-SNE / PaCMAP | PLANNED |
| DBSCAN / OPTICS / HDBSCAN | PLANNED |
| Cluster evaluation and selection | PLANNED |
| Cluster-aware splitting and random baseline | PLANNED |
| Detector comparison | PLANNED |

The implementation currently supports **data + features**. A future namespace is
not an implemented experiment. See [current status](docs/current_status.md).

## Current dataset findings

| Aggregate finding | Value |
|---|---|
| Historical records | 1657 |
| Unique exact contents | 1459 |
| Original train / val / test | 1178 / 107 / 372 |
| Exact duplicate groups | 198 |
| Train–val overlap | 57 shared contents |
| Train–test overlap | 141 shared contents |
| Val–test overlap | 0 |
| Validation / test with exact train copies | 53.27% / 37.90% |

All 1657 matched labels are valid, with five classes and 292 empty labels.
The candidate contains **4168 objects**; the full label archive contains **4182**,
including 14 objects in 10 orphan labels. Eight duplicate groups have annotation
conflicts. The [status document](docs/current_status.md) explains the scopes.
Only aggregate results are published here; real identities and hashes remain local.

## Methodology alignment

DINOv2 and CLIP provide independent image representations. Planned similarity
uses cosine on L2-normalized embeddings; **Bhattacharyya requires an explicitly
defined distributional representation** and is not implemented. The proposal's
reduction methods are **t-SNE + PaCMAP**, followed by DBSCAN, OPTICS and HDBSCAN.

Cluster selection will consider stability, visual and temporal coherence, AMI
and ARI. Historical, reproducible random and cluster-based partitions will be
compared using partition similarity and detector Precision, Recall, mAP@50 and
mAP@50–95. No detector comparison has run. The
[traceability table](docs/methodology_traceability.md) and
[design decisions](docs/design_decisions.md) distinguish evidence from plans.

## Architecture

```text
src/flir_pipeline/
  data/          inventory, hashing lineage, canonical manifest, label QA
  features/      DINOv2, CLIP, preprocessing, revisions, storage, diagnostics, plots
  similarity/    planned
  reduction/     planned: t-SNE / PaCMAP
  clustering/    planned: DBSCAN / OPTICS / HDBSCAN
  splitting/     planned
  detection/     future evaluation/integration
  evaluation/    planned
  utils/         streaming SHA256
configs/         active embedding configs; documented future boundaries
docs/            methodology, decisions, architecture, status
notebooks/       narrative source without outputs
scripts/         local report builder and offline notebook check
tests/           synthetic, offline tests
data/manifests/  local real manifests, ignored
artifacts/       local embeddings/diagnostics, ignored
reports/         local figures, tables, executed notebooks and HTML, ignored
```

The [architecture guide](docs/architecture.md) explains
`frame_id → content_id → embedding_row → future cluster_id → future split_id`.

## Reproducibility

Python **3.11** is required. [uv](https://docs.astral.sh/uv/) and the committed
lockfile define dependencies; YAML configs define extractor settings.
`dataset_id` fingerprints the supplied annotated manifest.
`feature_space_id` fingerprints model revision, preprocessing, pooling and
normalization while excluding runtime device and batch size.

Sampling uses an explicit seed (default 0). Metadata records model provenance,
configuration, selected content, seed, Python/library versions and Git commit.
Record a clean commit and hardware details for final experiments: a seed alone
does not guarantee bit-identical results across CPU/GPU or precision modes.

Set `model_revision` to a full Hugging Face commit SHA for final research runs.
New loads capture the resolved SHA when available and use it for the processor
and feature identity. Existing smoke metadata with `unknown` is preserved.
See [revision and cache semantics](docs/design_decisions.md).

## Data safety

Original FLIR datasets are external and treated as read-only. ZIP members are
streamed or decoded in memory; source images and labels are never rewritten.
Real manifests, hash inventories, embeddings, figures, reports, executed
notebooks, model caches/weights, credentials and local environment files are
not versioned. Generated outputs belong under ignored `artifacts/` or `reports/`.
The public notebook contains narrative and code only.

## Installation

```powershell
git clone https://github.com/Kazzu00/flir-leakage-pipeline.git
cd flir-leakage-pipeline
uv sync --locked --extra dev
uv run flir-pipeline --help
```

Core-only use: `uv sync --locked`. Add optional dependencies when needed:

```powershell
uv sync --locked --extra dev --extra vision
uv sync --locked --extra dev --extra vision --extra reporting
```

`uv sync` installs the selected extras exactly; keep the extras you want on each
sync invocation. The vision extra installs libraries, not pretrained model weights.
The first real extraction may download weights; use `--local-files-only` when the
required snapshot is already cached. CI installs only core and dev.

## Environment configuration

Use [.env.example](.env.example) as a reference and create your own ignored
`.env`; keep the example file unchanged and available in Git:

```dotenv
FLIR_DATA_ROOT=/path/to/external/flir-data
```

Replace the illustrative path locally. The CLI reads `FLIR_DATA_ROOT` from the
environment first, then the repository-root `.env`. Data commands accept explicit
paths as an alternative. Artifact roots are command options, not environment
variables. Run the documented commands from the repository root.

## CLI

### Implemented commands

```powershell
uv run flir-pipeline --help
uv run flir-pipeline data --help
uv run flir-pipeline features --help
```

| Group | Commands |
|---|---|
| `data` | `inventory`, `archive-tree`, `compare-archives`, `build-manifest`, `validate-labels`, `manifest-summary` |
| `features` | `extract`, `diagnostics`, `summary`, `verify`, `visualize-data`, `visualize-embeddings` |

With `FLIR_DATA_ROOT` configured and the source ZIPs available:

```powershell
uv run flir-pipeline data inventory --inspect-archives --hash-members
uv run flir-pipeline data compare-archives
uv run flir-pipeline data build-manifest
uv run flir-pipeline data validate-labels
uv run flir-pipeline data manifest-summary data/manifests/flir_canonical_candidate_v1.parquet
```

Example smoke extraction (runs a real model; it is not part of CI):

```powershell
uv run --extra vision flir-pipeline features extract --manifest data/manifests/flir_canonical_candidate_v1.parquet --config configs/embeddings/dinov2_smoke.yaml --limit-content 16 --seed 0 --output-root artifacts/features_smoke
uv run --extra vision flir-pipeline features extract --manifest data/manifests/flir_canonical_candidate_v1.parquet --config configs/embeddings/clip_smoke.yaml --limit-content 16 --seed 0 --output-root artifacts/features_smoke
```

Use the returned local directory with `features summary`, `features verify`, or
`features visualize-embeddings --feature-directory ...`. Verification returns a
nonzero exit code when quality invariants fail. `features diagnostics` needs the
manifest and source ZIP, and `features visualize-data` needs the manifest and
diagnostics Parquet. Use each command's `--help` for options.

### Future planned commands

Similarity, reduction, clustering, splitting, detection and evaluation are
planned stages with no runnable CLI commands. Previous success-returning stubs
were removed; the documented future package boundaries remain.

## Testing

After installing core + dev:

```powershell
uv run ruff check .
uv run pytest
uv run python scripts/check_notebook_source.py
```

Tests use synthetic ZIP images/labels and model stand-ins. They cover manifests,
duplicates, offline adapter compatibility, revision provenance, content mapping,
raw/L2 quality, actual interrupted-batch resume, visualization and report generation.
They do not require FLIR data, a GPU, model downloads or notebook tooling.
CI also checks the source notebook using only the standard library.

## Feature engineering

| Validated real smoke | Contents | Dimension | Representation |
|---|---|---|---|
| `facebook/dinov2-small` | 16 | 384 | CLS token |
| `openai/clip-vit-base-patch32` | 16 | 512 | Projected image embedding |

One embedding is computed per `content_id`; a separate mapping retains every
historical `frame_id`. Raw and L2 arrays are stored separately. Unselected smoke
contents map to row -1, not a fabricated vector. Labels, boxes and
`original_split` do not enter either encoder. The spaces are not concatenated.

Pixel statistics, entropy, Laplacian variance, pHash and dHash remain QA/EDA.
See [configuration status](configs/README.md) for smoke vs research candidates.
Full embeddings for 1459 contents have **not** been executed. For the same model
space, use separate smoke/full output roots because cache signatures reject
different selections.

## Reports

The [feature engineering review notebook](notebooks/feature_engineering_jp_review.ipynb)
is narrative source without outputs. With the required existing local manifest,
diagnostics and smoke artifacts, build the executed notebook and HTML:

```powershell
uv run --extra reporting python scripts/build_feature_engineering_review.py
```

The builder discovers only unambiguous completed artifacts under
`artifacts/features/`; select other locations or multiple runs explicitly with
`--manifest`, `--diagnostics`, `--dinov2` and `--clip`. Supplying either encoder
path uses only explicitly selected encoders. It never loads a model.
Executed outputs go to `reports/feature_engineering/jp_review/`.

Report class charts count presence per historical record; box areas are
per-record means. Embedding tables use actual selected sample sizes and measured
health. N=16 smoke results are distinct from full-dataset characterization.
The local repository review is `reports/code_review/project_review.md`.

## Repository status

**2026-09-09 — Feature engineering.** Data understanding, canonical traceability,
diagnostics and both real smoke tests are complete. The next scientific step is
**full embedding extraction → cosine similarity → t-SNE / PaCMAP →
DBSCAN / OPTICS / HDBSCAN**. See [current status](docs/current_status.md) for
evidence and limitations.

## References

- Acosta-Bernal et al.: cited in the supplied proposal context; exact title/DOI
  awaits confirmation from the proposal bibliography.
- Figueiredo & Mendes (2024), [Analyzing Information Leakage on Video Object
  Detection Datasets by Splitting Images Into Clusters With High Spatiotemporal
  Correlation](https://doi.org/10.1109/ACCESS.2024.3383047).
- Radford et al. (2021), [CLIP](https://arxiv.org/abs/2103.00020).
- Oquab et al. (2023), [DINOv2](https://arxiv.org/abs/2304.07193).
- Wang et al. (2021), [PaCMAP and dimensionality-reduction analysis](https://jmlr.org/papers/v22/20-1061.html).
- McInnes, Healy & Astels (2017), [hdbscan](https://joss.theoj.org/papers/10.21105/joss.00205).
- Studer et al. (2021), [CRISP-ML(Q)](https://www.mdpi.com/2504-4990/3/2/20).
