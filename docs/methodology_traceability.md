# Methodology traceability

Reviewed 2026-09-09 against the documented scope of the thesis proposal: **“Desarrollo de un pipeline de agrupación, procesamiento y detección de
minería ilegal en videos FLIR de la Amazonía colombiana.”** The full proposal
document is not included in this repository; this table traces its documented scope.

DONE means implemented and supported by executed evidence, not an empty package.
Local evidence paths below are intentionally ignored and absent from public clones.
Status is scientific progress, distinct from the A–E maintenance classification in
the local code review. See [current status](current_status.md) for aggregate facts.

| Proposal component | Repository component | Status | Evidence |
|---|---|---|---|
| Dataset characterization | `data/inventory.py`, `data/yolo_labels.py`, `data/annotations.py` | DONE | Inventory, actual instance vs presence counts, orphan/conflict QA and full diagnostics |
| Available temporal provenance | `data/temporal.py` | DONE WITH LIMITS | Two inferred sequences, 1657 orderable records by name, no verified timestamps; temporal summary/candidate tables |
| Canonical traceability | `data/manifest.py`, `data/identity.py`, `utils/hashing.py` | DONE | Local candidate v1 manifest: 1657 unique frame_id, 1459 content_id; `tests/test_manifest.py` |
| DINOv2 | `features/dinov2.py`, `features/storage.py` | DONE | Full 1459 × 384 CLS, resolved revision, raw/L2 and all 1657 canonical occurrence mappings verified |
| CLIP | `features/clip.py`, `features/storage.py` | DONE | Full 1459 × 512 projected-image embeddings, resolved revision, raw/L2 and canonical coverage verified |
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
| Historical data baseline | `original_split`, report `historical_baseline.csv` | DONE | Original membership preserved and exact overlap audited; detector baseline comparison remains a separate pending stage |
| Random baseline | `splitting/` | PLANNED | Reproducible seeded baseline not generated; must document sampling unit and leakage |
| Cluster-based baseline | `splitting/` | PLANNED | No new split generated |
| Inter-partition similarity | `similarity/`, `evaluation/` | PLANNED | Exact-byte historical overlap is available; embedding-based similarity has not been executed |
| YOLO/detector comparison | `detection/`, `evaluation/` | PLANNED | Compare Precision, Recall, mAP@50 and mAP@50–95 across baselines under a controlled detector protocol |
| Reproducibility | `uv.lock`, configs, identity/storage/revision helpers, CI | IN PROGRESS | Stage-level reproducibility complete: pinned configs, full verification, source-only notebook, local receipts; final experimental reproducibility pending; legacy smoke revisions preserved as unknown |
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
No reduction/clustering combination is selected. The bounded week 6 closure uses
DINOv2-small and CLIP ViT-B/32; this does not select a downstream winning encoder.
The [week 6 matrix](current_status.md) traces the documented commitments explicitly;
the full dated proposal is not included in the repository.
