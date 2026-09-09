# Architecture

The package follows data provenance through independent visual representations
to future partition experiments. Existing package boundaries are preserved.

| Layer | Responsibility and boundary | State |
|---|---|---|
| Data | ZIP inventory, streaming hashes, lineage/duplicate analysis, occurrence manifest and non-mutating YOLO QA | ACTIVE |
| Features | DINOv2 CLS / CLIP projected-image adapters, RGB preprocessing, revision tracking, raw/L2 stores, diagnostics, descriptive reports | ACTIVE; full extraction pending |
| Similarity | Cosine over normalized content embeddings; conditional distribution-based Bhattacharyya | FUTURE PLANNED |
| Reduction | t-SNE and PaCMAP | FUTURE PLANNED |
| Clustering | DBSCAN, OPTICS, HDBSCAN and evidence-based cluster selection | FUTURE PLANNED |
| Splitting | Reproducible baselines and indivisible cluster/scene allocation | FUTURE PLANNED |
| Detection / evaluation | Detector comparison, cluster quality, partition similarity and metrics | FUTURE PLANNED; group handoff boundary |

```text
src/flir_pipeline/
  cli.py                   implemented data/features commands only
  config.py                reserved validated path configuration
  data/
    inventory.py           archive structure, matching and exploratory lineage
    manifest.py            canonical occurrences and label/duplicate reports
    identity.py            shared portable dataset_id
    yolo_labels.py         syntax, normalized coordinates, geometry, canonical boxes
  features/
    base.py                extractor contract and synthetic test adapter
    preprocessing.py       read-only ZIP decoding and in-memory RGB conversion
    dinov2.py / clip.py     independent model adapters
    model_revision.py      requested vs resolved HF provenance
    storage.py             content/record indexes and resumable raw/L2 arrays
    diagnostics.py         independent image QA/EDA
    visualization.py       descriptive local reports
  similarity/              planned; documentation only
  reduction/               planned; documentation only
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
