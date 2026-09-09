# Methodology traceability

Reviewed 2026-09-09 against the thesis proposal requirements supplied for this
review: **“Desarrollo de un pipeline de agrupación, procesamiento y detección de
minería ilegal en videos FLIR de la Amazonía colombiana.”** The full proposal
document was not included; this table traces the supplied methodological scope.

DONE means implemented and supported by executed evidence, not an empty package.
Local evidence paths below are intentionally ignored and absent from public clones.
Status is scientific progress, distinct from the A–E maintenance classification in
the local code review. See [current status](current_status.md) for aggregate facts.

| Proposal component | Repository component | Status | Evidence |
|---|---|---|---|
| Dataset characterization | `data/inventory.py`, `data/yolo_labels.py` | DONE | Synthetic inventory/label tests; local `reports/data_inventory/`, matched-label QA and full diagnostics |
| Canonical traceability | `data/manifest.py`, `data/identity.py`, `utils/hashing.py` | DONE | Local candidate v1 manifest: 1657 unique frame_id, 1459 content_id; `tests/test_manifest.py` |
| DINOv2 | `features/dinov2.py`, `features/storage.py` | IN PROGRESS | Infrastructure and real N=16, 384D CLS smoke validated; full extraction absent |
| CLIP | `features/clip.py`, `features/storage.py` | IN PROGRESS | Infrastructure and real N=16, 512D projected-image smoke validated; full extraction absent |
| Cosine similarity | `similarity/` | PLANNED | Namespace only; intended input is L2-normalized embeddings from each independent encoder |
| Bhattacharyya distance | `similarity/`, design decision 8 | PLANNED | Conditional on an explicitly defined distributional representation; none selected |
| t-SNE | `reduction/` | PLANNED | Namespace only; no reductions executed |
| PaCMAP | `reduction/` | PLANNED | Namespace only; no reductions executed |
| DBSCAN | `clustering/` | PLANNED | No clustering implementation or experiment |
| OPTICS | `clustering/` | PLANNED | No clustering implementation or experiment |
| HDBSCAN | `clustering/` | PLANNED | No clustering implementation or experiment |
| Stability and visual coherence | `clustering/`, `evaluation/` | PLANNED | Cluster selection protocol and visual review pending |
| Temporal coherence | `evaluation/` | PLANNED | Filename-based sequence guesses are exploratory, not verified video timestamps |
| AMI | `evaluation/` | PLANNED | Reference cluster partitions and comparison protocol not yet defined |
| ARI | `evaluation/` | PLANNED | Reference cluster partitions and comparison protocol not yet defined |
| Cluster-aware splitting | `splitting/` | PLANNED | Every cluster/scene must remain in one partition; preserve content/occurrence mapping and assess class coverage |
| Historical baseline | `original_split` in canonical manifest | IN PROGRESS | Historical train/val/test membership and exact overlap audited; detector baseline comparison pending |
| Random baseline | `splitting/` | PLANNED | Reproducible seeded baseline not generated; must document sampling unit and leakage |
| Cluster-based baseline | `splitting/` | PLANNED | No new split generated |
| Inter-partition similarity | `similarity/`, `evaluation/` | PLANNED | Exact-byte historical overlap is available; embedding-based similarity has not been executed |
| YOLO/detector comparison | `detection/`, `evaluation/` | PLANNED | Compare Precision, Recall, mAP@50 and mAP@50–95 across baselines under a controlled detector protocol |
| Reproducibility | `uv.lock`, configs, identity/storage/revision helpers, CI | IN PROGRESS | Offline invariant tests; recorded dataset/feature IDs, seed and Git commit; new resolved-revision support; legacy smoke revisions remain unknown |
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

Future cluster evaluation must define the comparison unit, treatment of noise,
reference partitions for AMI/ARI, and stability protocol. Detection class labels
are not automatically ground-truth scene clusters. Decide whether a reduction is
for visualization or a clustering input and document its geometric implications.
No reduction/clustering combination or final model size is selected in this review.
