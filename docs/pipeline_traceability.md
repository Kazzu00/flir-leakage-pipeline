# Pipeline traceability

This map follows inputs through implemented processes to stored artifacts and
downstream consumers. Paths are relative to the repository; real evidence is local
and ignored. Execution state is recorded separately in [status](status.md).

| Input | Process / implementation | Artifact and identity | Downstream consumer |
|---|---|---|---|
| External image/label ZIPs | `data/inventory.py`, `yolo_labels.py`, `annotations.py`, `classes.py` | Inventories, label QA, class/instance tables, source hashes | Canonical manifest and data reports |
| Matched historical occurrences | `data/manifest.py`, `identity.py`, `utils/hashing.py` | Manifest: `dataset_id`, `frame_id`, `content_id`, relative source provenance, `original_split` | Every stage |
| Original names and occurrence lineage | `data/temporal.py` | Inferred sequence/index, confidence, temporal candidates; no verified timestamps | Posterior temporal analysis and display |
| Unique contents and pinned encoder config | `features/dinov2.py`, `clip.py`, `storage.py` | `feature_space_id`; raw/L2 arrays, content/record indices, metadata and quality | Similarity, reduction, original clustering controls |
| Original images | `features/diagnostics.py` | Pixel/entropy/blur/hash diagnostics, separate from embeddings | QA and descriptive reporting |
| Verified original L2 | `similarity/cosine.py`, `storage.py` | `similarity_space_id`; matrix, unordered pairs, directed top-k, checksums | Reduction preservation, clustering/split evaluation |
| Cosine neighborhoods plus posterior manifest metadata | `similarity/temporal.py`, `comparison.py`, `reporting.py` | Temporal/historical relations, cross-encoder Jaccard and review | Descriptive analysis and residual cohorts |
| Independent original L2 and reduction config | `reduction/reducers.py`, `storage.py` | `reduction_space_id`; coordinates, content order and source fingerprints | Clustering and saved visualization layouts |
| Reduction runs and original neighborhoods | `reduction/metrics.py`, `benchmark.py` | Trustworthiness, continuity, Jaccard, sampled Spearman, seed stability and reference selection | Selected exploratory reduction families |
| Original L2 or selected reduction families | `clustering/algorithms.py`, `storage.py` | `clustering_space_id`; assignments, noise -1, medoids, quality and algorithm diagnostics | Stability, selection, splitting, explorers |
| Assignments plus original spaces and posterior provenance | `clustering/metrics.py`, `experiments.py`, `selection.py` | ARI/AMI with two noise policies, coherence, shortlist and Pareto candidates | Diverse split candidate selection |
| Candidate clusters, manifest and occurrence annotations | `splitting/construction.py`, `storage.py` | `split_space_id`; atomic group/content/record assignments, balance and source snapshots | Residual evaluation, explorers, detector views |
| Historical membership / content permutations | `splitting/construction.py` | Historical baseline and reproducible random-content assignments | Common partition comparison |
| Partitions plus both original encoder spaces | `splitting/metrics.py`, `experiments.py`, `selection.py` | Exact overlap, NN/top-k/quantile residuals, temporal fractions, fractures, seed robustness and representatives | Candidate audit before detector training |
| Existing assignments, exact saved coordinates and ZIP images | `explorer/`, `apps/cluster_split_explorer.py` | Streamlit views / ignored VIKUS previews, sprites, occurrence metadata and receipts | Local human inspection |
| Fixed partition selection and detector config | `detection/protocol.py`, `materialization.py` | Plan/matrix identity, occurrence-preserving byte-checked views | Frozen detector runtime |
| Views, fixed weights/config and hardware probe | `detection/runtime.py` | Runtime freeze, pilot/checkpoint metadata; complete runs remain pending | Detector metrics and reporting |
| Detector image statistics | `detection/metrics.py`, `reporting.py` | Fixed-threshold P/R, AP, image bootstrap and gated aggregate report | Controlled comparison; final matrix pending compute |
| Existing tables, figures and validation receipts | `scripts/build_*_review.py`, seven source notebooks | Local HTML/executed notebooks and build receipts | Component reviews and consolidated project report |

## Identity and source binding

`frame_id → content_id → embedding_row → cluster_id → group_id → new_split`
retains all historical occurrences. Space IDs namespace the configuration;
source signatures bind the actual input bytes and selection. Runtime paths,
dates, device and batch do not replace mathematical identity. Reduction and
clustering record relevant implementation versions and source fingerprints.

Completed outputs are verified before reuse. A source-bound check can recompute
metrics; a report availability check reuses prior verification receipts. Neither
should be described as a new experiment. Failed/incomplete artifacts stay
preserved. See [reproducibility](reproducibility.md).

## Interpretation boundaries

- Numerical inputs exclude labels, historical splits and inferred time during
  representation, reduction and clustering. Posterior metrics join that metadata.
- Class counts enter split balance after clustering; cosine and temporal residuals
  are evaluated after allocation. Noise remains -1 with separate singleton groups.
- Reduction preserves content identity, not a guarantee of original density in 2D.
  ARI/AMI compare perturbation assignments, not detection labels as scene truth.
- C10 — DINOv2 / PaCMAP / DBSCAN and C12 — DINOv2 / t-SNE / HDBSCAN retain their
  fixed detector roles. C01 — CLIP / original L2 / OPTICS is a descriptive balance
  reference. The [candidate catalog](analysis/splitting.md#doce-candidatos-evaluados)
  preserves C01–C12 and all original space IDs.
- Bhattacharyya remains conditional on an explicit distributional representation.
  UMAP is outside the implemented reduction protocol.
- Noise cleaning and panoptic segmentation are external integration components;
  neither is implemented here. Production deployment/monitoring are outside this
  pipeline. [Architecture](architecture.md) retains the proposed handoff boundary.
