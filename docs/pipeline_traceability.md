# Pipeline traceability

## Sequence experiments: additional evidence path

The [operational runbook](runbooks/sequence_operations.md) records the parallel
SLURM DAG. `slurm_run.json` maps each logical stage to command, resources, job ID,
dependencies and logs; per-stage receipts bind completed artifacts. This operational
record is separate from scientific feature/fit identities. A combined review
package and `sequence_final_summary_v1` close computation while preserving the
manual gate and false VDG/split flags.

| Input | Implementation | Source-bound output | Interpretation |
|---|---|---|---|
| Canonical/video manifest and complete original encoder stores | `sequences/experiments/sources.py`, `boundary.py` | Current-v1 control, adjacent/multiscale scores, robust local context, ranked candidate zones and occurrences | Original-space temporal evidence; filename inference remains explicit |
| Visual vectors, deterministic grid and seeds | `fitting.py`, existing reduction/clustering adapters | Unique-content assignments, reduction coordinates, effective backend parameters and failed cells | No review labels or historical split enters fitting |
| Normalized external evidence, manifest identity and producer checksums | `structure.py` | Frozen intervals/observations, conservative cores, occurrence membership and masks | Reviewed intervals remain uncertain; external review is not ground truth |
| Distinct cores/instances and original L2 | `recurrence.py` | Directed NN distributions, encoder ranks, exact-copy counts, candidates and diagnostic components | No automatic VDG or sequence merge |
| Cluster assignments and review membership | `evaluation.py`, `transitions.py` | ARI/AMI/V-measure with coverage, purity/entropy, fragmentation/merging, runs/returns and zone hits | Post-hoc comparisons; noise and ambiguous occurrence labels remain explicit |
| Full configured run set | `runner.py` | Comparison, failures, candidate union/intersection and stability with common coverage | Experimental evidence summary without automatic winner or split |
| Candidate evidence and verified image bytes | `review.py`, existing ImageSource | Context sheets, medoids, encoder matches, immutable CSV decisions and replayable history | supported/ambiguous/unsupported remain manual evidence |

All new contracts preserve `ground_truth=false`, `split_created=false` and
`automatic_confirmation=false`. Byte binding and deterministic scientific IDs
are distinct: changing source row order/split metadata invalidates the old source
receipt while leaving the numerical fit invariant. Detailed definitions and
resource/validation boundaries are in the
[protocol](protocols/sequence_experiments.md).

This map follows inputs through implemented processes to stored artifacts and
downstream consumers. Paths are relative to the repository; real evidence is local
and ignored. Execution state is recorded separately in [status](status.md).

| Input | Process / implementation | Artifact and identity | Downstream consumer |
|---|---|---|---|
| External read-only source videos | `data/video_frames.py`, external FFmpeg/ffprobe | Sampled JPEGs, `frames.parquet`, `summary.json`; relative grid, path-based video ID and source checksum | Sampling operationally validated on Hypatia: real 5-second smoke and job 737719 COMPLETED (0:0), 3 source videos / 9648 JPEGs at 1 FPS; next: video manifest bridge |
| Completed sampled JPEGs and checksum-bound receipt | `data/video_manifest.py`, `local_images.py`, `identity.py` | `flir_video_samples_v1`: all occurrences, exact-byte contents, video/grid provenance, decode QA; report binds both inputs and manifest checksum | Existing unique-content extraction via `features/image_source.py`; bridge tested synthetically only, not yet run on Hypatia; sequence identification remains future work |
| External image/label ZIPs | `data/inventory.py`, `yolo_labels.py`, `annotations.py`, `classes.py` | Inventories, label QA, class/instance tables, source hashes | Canonical manifest and data reports |
| Matched historical occurrences | `data/manifest.py`, `identity.py`, `utils/hashing.py` | Manifest: `dataset_id`, `frame_id`, `content_id`, relative source provenance, `original_split` | Every stage |
| Original names and occurrence lineage | `data/temporal.py` | Inferred sequence/index, confidence, temporal candidates; no verified timestamps | Posterior temporal analysis and display |
| Unique contents and pinned encoder config | `features/dinov2.py`, `clip.py`, `storage.py` | `feature_space_id`; raw/L2 arrays, content/record indices, metadata and quality | Similarity, reduction, original clustering controls |
| Contiguous 1 FPS occurrence manifest and both original feature stores | `sequences/detection.py`, `storage.py` | Source/config-bound candidate detection; per-video rounded average-tie ranks, multiscale S and disjoint F3 search intervals | Manual review; never automatic acceptance |
| Recomputed candidates and confirmed manual validation CSV/JSON | `sequences/validation.py`, `construction.py` | Deterministic sequence_set_id/sequence_id; every occurrence assigned once; typed boundary provenance, exact-copy edges/support and connected components | Future split constraints; no split created; real sequences reported by owner, not reverified locally |
| Canonical/video manifests, four paired original L2 stores and reviewed sequence set | `linkage/sources.py`, `candidates.py`, `storage.py` | Source-bound union of encoder top-k candidates, separate cosines/ranks, all candidate video occurrences and full labeled occurrence snapshot | Future visual review; ground_truth=false, no confirmed link, chosen sequence or split |
| Calibration sample, linkage, labeled manifest, sequence occurrences and confirmed visual memberships | `linkage/review_*.py` | All-occurrence temporal sheets, one manual decision per query/proposed group, immutable CSV imports/history and stratified counts/rates | Manual evidence calibration; not ground truth, exact-frame/sequence identification or a leakage-safe split |
| Existing immutable manual review revisions and original source locators | `linkage/review_aggregation.py`, `review_aggregate_storage.py` | Exact revision-file SHA256, revision/calibration IDs, original metadata/history, all observations, compatible unique pairs, duplicate report and descriptive counts | Reverified calibration summary; no automatic decisions, confirmed linkage, sequence assignment or split |
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
| Verified controlled runs and frozen split context | `detection/association.py`, `association_plot.py` | Run × class/overall table, prespecified registry status, 48-cell matrix and gated figure 09 | Separate detector/residual Streamlit app and detector report |
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

For local video samples, `frame_id → content_id → embedding_row` is implemented;
v2 similarity and numerical reduction/clustering can now consume those stores,
with synthetic local validation only. Real downstream execution, sequence discovery
and partition assignment remain pending for that dataset. `record_index` keeps all occurrences,
including smoke exclusions (-1); video provenance is joined from the supplied
manifest. Directory transport changes no mathematical feature ID. Each declared
local occurrence is hash-checked before cache reuse/resume; exact duplicate bytes
must still match, even when that occurrence is not the selected representative.

For video similarity, `record_provenance.parquet` preserves the full occurrence
grid and embedding mapping; content provenance holds membership/occurrence sets.
Temporal minima use all same-source occurrences after cosine/top-k computation.
No sequence or historical split is generated. V2 metadata binds this provenance,
exact quantiles and `pair_storage`; `summary_only_v1` omits the full pair table.
Reduction/clustering use the same matrix/top-k/metadata source contract and never
require that table. Historical v1 IDs and artifacts remain readable without migration.

## Interpretation boundaries

Cross-dataset candidate lineage is normalized: `labeled_occurrences.content_id`
joins `content_candidates.labeled_content_id`; each candidate's `video_content_id`
joins **all** `candidate_occurrences.video_content_id` rows. The latter preserve
video_id/sample_index/sequence_id without selecting a representative occurrence.
Annotation hashes and historical splits survive for every labeled frame_id.
Artifact identity binds both independent dataset IDs, paired mathematical spaces,
input checksums, sequence set and scientific ranking configuration. The bounded
query block size is operational and does not change scores, ranks or identity.

The owner reports successful real Hypatia init/verify for the confirmed visual
dependency v1 adapter: quality_valid/source_bound=true, all candidate occurrences
preserved, one decision per query, and ground_truth/confirmed_matches_created/
split_created=false. This is reported compatibility evidence, not a local real-data
rerun or completion of future calibration strata. Aggregation remains locally
validated with synthetic reviews. It preserves each source sample/history and
deduplicates compatible query/group keys within each reported cell. Different
stratum membership is retained; cell counts can overlap. Global rates describe
the reviewed calibration set and do not estimate representative accuracy/precision.

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

Dataset variants preserve a separate identity chain:
`dataset_id -> dataset_variant_id -> feature_space_id -> experiment artifact`.
The experiment source signature records the variant declaration, parent identity,
manifest/feature checksums and both encoder spaces. Image changes do not imply a
change in temporal identity; explicit correspondence retains both occurrence IDs
and their known indices, method, confidence/evidence and external truth flag.
Cross-variant ARI/AMI excludes non-bijective content relations and reports coverage
without dropping the complete occurrence populations. Metrics do not create
sequences, dependency groups or splits. The real no-HUD comparison remains future
work after base validation; synthetic pixel tests validate infrastructure only.

The native Hypatia adapter binds all 14 consumed source files by SHA256 and raw
snapshot, with source path/family and adapter version. It validates 910 occurrences,
712 nominal indices, 13 core candidates, 12 uncertain zones and 78 recurrence
pairs/11 candidates for the observed revisions. These are acceptance checks of
legacy artifacts, not predictions for future experiments. Full original summaries
and the assistant-review mode survive without asserting truth, exact provenance,
sequence instances, dependency groups or splits. The normalized source retains
the unspecified variant and is verified against the canonical source identities;
any modified original invalidates verification even if its snapshot is intact.
