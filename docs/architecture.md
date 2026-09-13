# Architecture

The package follows data provenance through independent visual representations
to future partition experiments. Existing package boundaries are preserved.

| Layer | Responsibility and boundary | State |
|---|---|---|
| Data | ZIP inventory, streaming hashes, lineage/duplicate analysis, occurrence manifest and non-mutating YOLO QA | ACTIVE |
| Features | DINOv2 CLS / CLIP projected-image adapters, RGB preprocessing, revision tracking, raw/L2 stores, diagnostics, descriptive reports | ACTIVE; both full extractions validated |
| Similarity | Cosine over existing L2 contents, top-k, posterior temporal/split relations, agreement and review | ACTIVE; full cosine validated, inferred temporal analysis partial; Bhattacharyya conditional/planned |
| Reduction | t-SNE and PaCMAP; exact preservation, seed stability, bounded selection and posterior interpretation | ACTIVE; experiment status in current_status.md |
| Clustering | DBSCAN, OPTICS, HDBSCAN and evidence-based cluster selection | FUTURE PLANNED |
| Splitting | Reproducible baselines and indivisible cluster/scene allocation | FUTURE PLANNED |
| Detection / evaluation | Detector comparison, cluster quality, partition similarity and metrics | FUTURE PLANNED; group handoff boundary |

```text
src/flir_pipeline/
  cli.py                   implemented data/features/similarity/reduction commands
  config.py                reserved validated path configuration
  data/
    inventory.py           archive structure, matching and exploratory lineage
    manifest.py            canonical occurrences and label/duplicate reports
    identity.py            shared portable dataset_id
    yolo_labels.py         syntax, normalized coordinates, geometry, canonical boxes
    classes.py             central original/canonical detection names and YAML verification
    annotations.py         occurrence-level instances and per-class normalized geometry
  features/
    base.py                extractor contract and synthetic test adapter
    preprocessing.py       read-only ZIP decoding and in-memory RGB conversion
    dinov2.py / clip.py     independent model adapters
    model_revision.py      requested vs resolved HF provenance
    storage.py             content/record indexes and resumable raw/L2 arrays
    diagnostics.py         independent image QA/EDA
    visualization.py       descriptive local reports
  similarity/
    cosine.py              float32 dot products, deterministic top-k, distributions
    temporal.py            occurrence consensus, multi-split sets, posterior pair tables
    storage.py             config identity, input/output fingerprints, full verification
    comparison.py          content-aligned neighbor Jaccard across independent spaces
    reporting.py           aggregate tables, selected ZIP image grids and figures
  reduction/
    base.py                explicit configs and portable reduction identity
    reducers.py            shared vector-only contract, t-SNE and PaCMAP adapters
    metrics.py             exact ranks, T/C, Jaccard, distance sampling and stability
    storage.py             immutable runs, source binding and executable verification
    benchmark.py           small grids, seed aggregation, predeclared candidates
    visualization.py       posterior temporal/split interpretation and local reports
  clustering/              planned; documentation only
  splitting/               planned; documentation only
  detection/               future evaluation/integration; documentation only
  evaluation/              planned; documentation only
  utils/hashing.py          streaming exact-byte hashes
```

## Identity flow

```text
frame_id (occurrence, all historical records)
   |
   v  many occurrences may share exact bytes
content_id (exact image SHA256)
   |
   v  one row per selected content within one feature_space_id
embedding_row (separate DINOv2 and CLIP stores)
   |
   +--> similarity_space_id: matrix rows, neighbor IDs, posterior content/record lineage
   +--> reduction_space_id: coordinates in the same content_index order
   |
   v  FUTURE assignment, including an explicit noise policy
cluster_id
   |
   v  FUTURE indivisible cluster/scene allocation
split_id
```

`dataset_id` namespaces an annotated manifest version. `feature_space_id`
identifies the representation independently of dataset membership. The combination
of these IDs and the cache signature gives an experiment's lineage. Neither
future cluster_id nor split_id exists yet. `original_split` remains historical
metadata and is not a generated split_id.

Storage writes under `artifacts/features/<extractor>/<dataset_id>/<feature_space_id>`.
Arrays are float32, with raw and L2 versions. Index files map every occurrence
back to its source ZIP member. A `.partial` checkpoint is local execution state;
the `metadata.json` completion marker is written after successful quality checks.
Real manifests, content hashes, arrays and local paths never belong in Git.

Similarity stores use `artifacts/similarity/<extractor>/<dataset_id>/<feature_space_id>/<similarity_space_id>/`.
The numerical layer receives only L2 vectors and indexing. Afterwards,
`data.temporal.audit_temporal_lineage` supplies occurrence provenance; content-level
consensus and split sets annotate pairs. Matrix and pair tables never expand
duplicate occurrences. Standalone verification checks internal integrity; optional
feature/manifest arguments also bind outputs to the original source arrays and
lineage. Comparison stores align queries by content ID, independently of row order.
The local review contains selected image composites; the versioned notebook has
no executed outputs. No future cluster or new split is produced by this layer.

## Reduction boundary

Reduction receives only full original L2 vectors from verified feature spaces.
Original cosine tables form the evaluation reference, independently of fitting.
Output paths extend the feature identity with method/reduction identity; benchmark
directories bind the exact run set, stability summaries and selection rule.
Existing record_index is referenced by fingerprint rather than duplicated.
Metadata retains native preprocessing, library versions, source hashes and the
actual execution commit. Verified completed runs are resumable grid checkpoints;
incomplete publications are preserved. No optimizer checkpoint is fabricated.

Posterior reporting joins by content_id and preserves multiple historical split
memberships. Temporal examples use an explicit nominal-index rule, independently
of coordinates. Neither layer creates clustering assignments. See the [protocol](reduction_protocol.md)
and [runbook](reduction_runbook.md); 2D density is not original-space density.

## Individual research and group handoff

The individual contribution owns characterization, representation, similarity,
reduction, clustering/selection, partitioning and partition-quality evaluation.
Noise cleaning, panoptic segmentation and final assembly belong to the broader
group pipeline. No implementations of these team components are included.

The proposed handoff contract is documentation only: source occurrence/content
identity, relative provenance, representation metadata, and eventually cluster/
split assignments plus evaluation summaries. Any group transform that changes
image bytes must produce new content identity and an explicit parent mapping;
it must preserve historical provenance and record its parameters/version. Group
interfaces must not overwrite original images or claim old embeddings describe
transformed pixels. Concrete schemas await agreement with the other contributors.

## Package and dependency boundaries

CLI imports heavy components inside implemented commands. Importing the package
or requesting help does not download models. Core includes NumPy, pandas, Pillow,
PyArrow, SciPy, matplotlib, tqdm, PyYAML, Pydantic and Typer. SciPy supports the
diagnostic DCT; matplotlib is active in report generation and its synthetic tests.
Pydantic supports the small retained future path contract (`PipelineConfig`),
which is not presented as active environment configuration.

`vision` contains torch, torchvision and Transformers; torchvision is retained
for Transformers image processing even without a direct project import.
`reporting` contains Jupyter/nbformat/nbclient/nbconvert for notebook execution and
HTML. `dev` contains Ruff, pytest/coverage and pre-commit. Unused accelerate,
scikit-learn and future clustering/YOLO/tracking extras were removed from declared
dependencies; they can be introduced with the actual future experiment code.

The data manifest currently uses some private inventory helpers within the data
layer. This coupling is documented technical debt; no broad package refactor was
needed to align the research methodology.
