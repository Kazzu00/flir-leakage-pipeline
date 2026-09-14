# Methodology traceability

Reviewed 2026-09-13 against the documented scope of the thesis proposal: **“Desarrollo de un pipeline de agrupación, procesamiento y detección de
minería ilegal en videos FLIR de la Amazonía colombiana.”** The full proposal
document is not included in this repository; this table traces its documented scope.

DONE means implemented and supported by executed evidence, not an empty package.
Local evidence paths below are intentionally ignored and absent from public clones.
Status is scientific progress, distinct from the A–E maintenance classification in
the local code review. See [current status](current_status.md) for aggregate facts.

| Proposal component | Repository component | Status | Evidence |
|---|---|---|---|
| Dataset characterization (week 6) | `data/inventory.py`, `data/yolo_labels.py`, `data/classes.py`, `data/annotations.py` | DONE | Original YAML/class-order correspondence; 4168 instances, per-class normalized geometry, orphan/conflict QA and full diagnostics; exact bibliography pending |
| Available temporal provenance | `data/temporal.py` | DONE WITH LIMITS | Two inferred sequences, 1657 orderable records by name, no verified timestamps; temporal summary/candidate tables |
| Canonical traceability | `data/manifest.py`, `data/identity.py`, `utils/hashing.py` | DONE | Local candidate v1 manifest: 1657 unique frame_id, 1459 content_id; `tests/test_manifest.py` |
| DINOv2 | `features/dinov2.py`, `features/storage.py` | DONE | Full 1459 × 384 CLS, resolved revision, raw/L2 and all 1657 canonical occurrence mappings verified |
| CLIP | `features/clip.py`, `features/storage.py` | DONE | Full 1459 × 512 projected-image embeddings, resolved revision, raw/L2 and canonical coverage verified |
| Cosine similarity | `similarity/cosine.py`, `storage.py` | DONE | Both full 1459 × 1459 float32 matrices, 1063611 unordered pairs, 29180 directed top-20 edges per encoder; original L2 source verification and executed 14-section HTML |
| Visual/temporal correlation analysis | `similarity/temporal.py`, `reporting.py` | PARTIAL | Full same/different-sequence and frame-delta analysis executed; all 1459 contents have inferred sequence/index, zero verified timestamps; global visual coherence remains unvalidated |
| Bhattacharyya distance | `similarity/`, design decision 8 | PLANNED / CONDITIONAL | Conditional on an explicitly defined distributional representation; none selected |
| t-SNE | `reduction/reducers.py`, `storage.py`, `benchmark.py` | DONE | 18 full runs, perplexity 10/30/50 × seeds 0/1/2 × both encoders; direct L2, 1459 × 2, exact preservation, verified artifacts and report |
| PaCMAP | `reduction/reducers.py`, `storage.py`, `benchmark.py` | DONE | 18 full runs, MN_ratio 0.2/0.5/1.0 × seeds 0/1/2 × both encoders; apply_pca=false, effective pair counts, verified artifacts and report |
| Reduction preservation and seed stability | `reduction/metrics.py`, `benchmark.py`, `visualization.py` | DONE WITH LIMITS | Exact T/C, original/reduced Jaccard and three seed pairs at k=5/10/20, sampled Spearman; four exploratory references under a predeclared rule, not validated clustering inputs |
| DBSCAN | `clustering/algorithms.py` | DONE | 132 full runs; space/seed-specific k-distance epsilon, original controls and four reduced spaces |
| OPTICS | `clustering/algorithms.py` | DONE | 186 full xi runs, max_eps infinite, retained reachability/core/order diagnostics |
| HDBSCAN | `clustering/algorithms.py` | DONE | 96 full scikit-learn EOM runs and membership probabilities; no external library or leaf experiment |
| Stability and visual coherence | `clustering/metrics.py`, `selection.py`, `visualization.py` | DONE WITH LIMITS | Original-space silhouette/cosine/medoids, visual retention, 54 shortlisted configurations, 41 Pareto candidates, six galleries; no exhaustive scene validation |
| Temporal coherence | `clustering/metrics.py` | DONE WITH LIMITS | Posterior sequence fraction/entropy and pair retention at inferred delta 1/5/10, explicit denominators; timestamps still unverified |
| AMI | `clustering/metrics.py`, `experiments.py` | DONE WITH LIMITS | 204 controlled seed/parameter comparisons, all-points/common-clustered, means/minima, N, coverage and triviality |
| ARI | `clustering/metrics.py`, `experiments.py` | DONE WITH LIMITS | Same 204 perturbation comparisons, recomputed from assignments; no object-label ground truth used |
| Cluster-aware splitting | `splitting/` | PLANNED | Every cluster/scene must remain in one partition; preserve content/occurrence mapping and assess class coverage |
| Historical data baseline | `original_split`, report `historical_baseline.csv` | DONE | Original membership preserved and exact overlap audited; detector baseline comparison remains a separate pending stage |
| Random baseline | `splitting/` | PLANNED | Reproducible seeded baseline not generated; must document sampling unit and leakage |
| Cluster-based baseline | `splitting/` | PLANNED | No new split generated |
| Historical inter-partition similarity | `similarity/temporal.py` | DONE WITH LIMITS | Six encoder-specific quantile cohorts, multi-split content sets and non-exact cross-split candidates measured; no automatic leakage claim |
| New-partition correlation evaluation | `splitting/`, `evaluation/` | PLANNED | New splits do not exist; residual correlation and detector comparison remain pending |
| YOLO/detector comparison | `detection/`, `evaluation/` | PLANNED | Compare Precision, Recall, mAP@50 and mAP@50–95 across baselines under a controlled detector protocol |
| Reproducibility | `uv.lock`, configs, identity/storage/revision helpers, CI | IN PROGRESS | Feature, similarity, reduction and clustering stages have full verification, source-only notebooks, metadata, input/output hashes and local receipts; reduction protocol/source snapshot before grid; final experimental reproducibility pending |
| Noise cleaning and panoptic segmentation | Documented group handoff in `architecture.md` | GROUP INTEGRATION | Broader team contribution; neither component is implemented here |
| Final group pipeline assembly | Documented integration boundary | GROUP INTEGRATION | Future handoff contract, not an assembled pipeline |
| Production deployment and monitoring | Outside repository research scope | NOT APPLICABLE | Only the first four CRISP-ML(Q) phases are in scope |

## Methodological boundaries

The committed dimensionality-reduction methods are **t-SNE + PaCMAP**. UMAP is
not part of the current proposal experiment. Its unused dependency was removed;
there was no UMAP implementation or experiment configuration to preserve.

**Bhattacharyya distance requires an explicitly defined distributional
representation.** Arbitrary DINOv2/CLIP vectors are not probability distributions.
Select and justify the distribution, support, normalization and estimation
procedure before considering implementation; do not transform signed embeddings
into arbitrary “probabilities” simply to check a proposal item.

The first four CRISP-ML(Q) phases organize this work: (1) business and data
understanding, (2) data preparation, (3) modeling, (4) evaluation. Quality checks
span phases, but completed data QA does not establish clustering quality or
detector improvement. Production deployment and monitoring are excluded.

The [executed clustering protocol](clustering_protocol.md) defines unique-content
units, explicit noise policies and seed/parameter reference pairs for AMI/ARI.
Detection labels are not scene ground truth. Original L2 controls and selected
2D inputs are compared using original-space metrics; existing PaCMAP is a view
only when illustrating original clustering. Forty-one combinations remain
exploratory candidates, with no final split selected. The bounded week 6 closure uses
DINOv2-small and CLIP ViT-B/32; this does not select a downstream winning encoder.
The [weeks 6–8 matrix](current_status.md) traces the requested closure criteria explicitly;
the full dated proposal is not included in the repository.

Week 9 adds descriptive evidence without selecting an encoder or downstream
configuration. [Executed similarity analysis](similarity_analysis.md) documents
pair units, population statistics, inclusive nested quantiles, filename consensus,
historical cross-split semantics, neighbor agreement and visual inspection limits.

Weeks 9–10 add [36 executed reductions](reduction_analysis.md) under a
[predeclared protocol](reduction_protocol.md). PCA is initialization only; no
preliminary PCA or 3D experiment was run. Candidate references preserve seed 0,
while selection uses all three seeds. The subsequent [clustering analysis](clustering_analysis.md)
evaluates original L2 and those candidate spaces in 414 runs, with explicit
distance scales, noise handling, stability/coherence and coverage. The next
protocol must define indivisible group allocation and residual correlation.
Nonlinear 2D density is not equivalent to original embedding density.
