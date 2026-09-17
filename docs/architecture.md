# Architecture

The package follows data provenance through independent visual representations
to verified partition experiments and controlled detector evaluation infrastructure.
Existing package boundaries are preserved.

| Layer | Responsibility and boundary | State |
|---|---|---|
| Data | ZIP inventory, streaming hashes, lineage/duplicate analysis, occurrence manifest and non-mutating YOLO QA | ACTIVE |
| Features | DINOv2 CLS / CLIP projected-image adapters, RGB preprocessing, revision tracking, raw/L2 stores, diagnostics, descriptive reports | ACTIVE; both full extractions validated |
| Similarity | Cosine over existing L2 contents, top-k, posterior temporal/split relations, agreement and review | ACTIVE; full cosine validated, inferred temporal analysis partial; Bhattacharyya conditional/planned |
| Reduction | t-SNE and PaCMAP; exact preservation, seed stability, bounded selection and posterior interpretation | ACTIVE; experiment status in current_status.md |
| Clustering | DBSCAN, OPTICS, HDBSCAN, original-space metrics, perturbation stability and Pareto candidates | ACTIVE; 414 full runs verified, candidate selection executed |
| Splitting | Reproducible baselines, indivisible groups, class balance and residual partition quality | ACTIVE; 66 verified runs, both encoders, robust Pareto and review |
| Detection | Frozen split/model matrix, immutable dataset views, optional YOLO11 runtime, image statistics/bootstrap and review | INFRASTRUCTURE + SMALL CPU PILOTS VALIDATED; final comparison pending |
| Evaluation | Reserved cross-component boundary; operational metrics live beside their experiments | GROUP INTEGRATION / FUTURE |

```text
src/flir_pipeline/
  cli.py                   implemented data/features/similarity/reduction/clustering/splitting commands
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
  clustering/
    base.py / algorithms.py configs, deterministic identity and vector-only adapters
    metrics.py             original-distance/cosine, medoids, posterior lineage, ARI/AMI
    storage.py             verified source families, immutable run artifacts
    selection.py           bounded Pareto shortlist and explicit noise eligibility
    experiments.py         screening, seed/parameter comparisons and verification
    visualization.py       existing 2D views and runtime ZIP exemplars
  splitting/
    base.py                validated configuration and portable split_space_id
    construction.py        atomic groups, seeded random cuts and profile-count MILP
    metrics.py             complete cross-split NN, quantile/temporal pairs, balance, QA
    selection.py           diverse clustering subset and robust split Pareto
    storage.py             immutable assignments, source binding and recomputation
    experiments.py         baselines, five-seed comparisons, stability and later lists
    visualization.py       nine aggregate figures and local review tables
    cli.py                 lazy construction/evaluation/verification commands
  detection/
    protocol.py            pre-YOLO constraints, immutable plan and matrix identities
    materialization.py     occurrence-preserving byte-checked generated views
    runtime.py             optional YOLO, hardware/batch probe, pilot, gate/resume
    metrics.py             fixed-threshold P/R, pooled AP, image bootstrap, hierarchy
    reporting.py           complete-matrix gate, real context and pending panels
    cli.py                 lazy detector commands, no optional runtime import in CI
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
   v  per-clustering_space_id assignment; noise stays -1
cluster_id
   |
   v  indivisible cluster units + separate singleton noise group_id
new_split (train/val/test), within split_space_id
```

`dataset_id` namespaces an annotated manifest version. `feature_space_id`
identifies the representation independently of dataset membership. The combination
of these IDs and the cache signature gives an experiment's lineage. Candidate
cluster_id assignments and new split_space_id partitions exist. `original_split`
remains historical metadata; historical multi-memberships are never flattened.

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

## Clustering boundary

Clustering families bind the original feature/similarity/manifest signatures and
the selected reduction configuration at seeds 0/1/2. Numerical adapters accept
vectors and content identity only; sequence, class and historical membership
never enter fitting. EvaluationContext joins posterior provenance and retains
the existing original cosine neighbors. Original Euclidean distances define
primary silhouette and medoids independently of the fitted representation.

Run paths are `artifacts/clustering/<encoder>/<dataset_id>/<representation_id>/<algorithm>/<clustering_space_id>/`.
Representation identity is feature_space_id for original controls or
reduction_space_id for reduced inputs. Required files contain one assignment per
content, summaries, exact metrics and completion metadata; noise remains −1.
Original record_index is referenced by fingerprint for all historical occurrences.
This clustering layer creates no split_id; group identifiers are local to one
clustering run and downstream splitting provides its own identity namespace.

Stage A stores the complete screening and bounded shortlist. Stage B references
those runs, computes local parameter agreements, and fits only additional seeds
for reduced shortlist members. Collection metadata binds every run, comparison
pair and selection policy. Source-bound verification recomputes metrics and
medoids; collection verification also reconstructs agreements, their aggregation,
prescribed perturbations, Pareto membership and review references. A changed
reduction family cannot be mixed into a previous screening.

The report reads images from the original ZIP into memory to compose ignored
galleries. It distinguishes fitting space from an existing 2D view, and reports
coverage beside noise-excluding metrics. The [runbook](clustering_runbook.md)
documents commands; [analysis](clustering_analysis.md) records observed results.

## Splitting boundary

Splitting reuses the canonical manifest, audited occurrence annotations and
verified original similarity sources from both encoders. Twelve diverse members
of the existing clustering Pareto form the principal input set. The numerical
assignment layer receives only atomic-unit balance vectors: record/class/empty
counts. Historical membership supplies target ratios and its separate baseline.
Classes never change cluster membership; noise remains label -1 with a separate
singleton group identity. No similarity or temporal objective enters the MILP.

Runs live at `artifacts/splitting/runs/<split_space_id>/`; comparisons live at
`artifacts/splitting/comparisons/<comparison_id>/`. Content and record assignments,
source group snapshots, balance tables, original-space NN/quantile metrics and
temporal summaries are immutable publications with checksums. Standalone QA
reconstructs identity/indivisibility/balance; source-bound QA additionally
recomputes residual metrics and binds labels/cluster sources. Historical content
rows explicitly preserve membership sets, with nullable new_split for overlaps.

Five seeds characterize assignment variability without renaming train/val/test.
Robust constraint/Pareto selection keeps full eligibility diagnostics and at most
three predeclared anchors; seed 0 always represents a selected configuration.
The two observed anchors and their limits are in [analysis](splitting_analysis.md).
The report uses aggregate data only, without fetching images. Later `export-lists`
requires a verified split and already materialized occurrence paths; it never
copies images or silently merges conflicting labels. Real export remains pending.

## Individual research and group handoff

The individual contribution owns characterization, representation, similarity,
reduction, clustering/selection, partitioning and partition-quality evaluation.
Noise cleaning, panoptic segmentation and final assembly belong to the broader
group pipeline. No implementations of these team components are included.

The proposed handoff contract is documentation only: source occurrence/content
identity, relative provenance, representation metadata, and the now-available
cluster/split assignments plus evaluation summaries. Any group transform that changes
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
HTML. `dev` contains Ruff, pytest/coverage and pre-commit. `reduction` contains
scikit-learn, PaCMAP and threadpoolctl; clustering reuses its scikit-learn HDBSCAN
without another library. The optional `detection` extra pins YOLO11-compatible
Ultralytics/PyTorch/torchvision; core and CI do not install it. Unused accelerate
and tracking extras remain absent. Synthetic CI includes the reduction extra.

The data manifest currently uses some private inventory helpers within the data
layer. This coupling is documented technical debt; no broad package refactor was
needed to align the research methodology.
