# Current research status

Reviewed **2026-09-09**. Current phase: feature engineering and preparation for
full embedding extraction. This is an evidence snapshot, not a week-based schedule.
Local artifacts were inspected read-only; real embeddings were not recomputed.

## DONE

- Dataset audit and exact duplicate analysis.
- Canonical candidate v1 manifest with portable occurrence/content identity.
- YOLO label validation, including orphan labels and annotation conflicts.
- Full content-level image diagnostics for 1459 contents.
- DINOv2/CLIP infrastructure, raw/L2 storage, record mapping and checkpointing.
- Real DINOv2 and CLIP smoke validation, N=16 per encoder.
- Feature engineering visualization and narrative notebook/HTML infrastructure.

DONE infrastructure does not mean full representation experiments are complete.
Revision tracking added during review is covered by synthetic adapter tests only.

## CURRENT

Feature engineering: choose and pin the model revision, validate the extraction
configuration on target hardware and prepare full extraction. Research configs
are candidate settings; their larger model sizes and batch size are not finalized.

## NEXT

1. Full DINOv2 extraction over all 1459 unique contents.
2. Full CLIP extraction over all 1459 unique contents, independently.
3. Cosine similarity over normalized embeddings in each space.

## PLANNED

t-SNE, PaCMAP, DBSCAN, OPTICS, HDBSCAN, stability/visual/temporal coherence,
AMI/ARI, cluster selection, cluster-aware splitting, seeded random baseline,
partition-similarity evaluation and controlled detector comparison. No outputs
from these stages exist in the reviewed repository. Bhattacharyya remains
conditional on a justified distributional representation.

## Validated aggregate dataset facts

| Quantity | Count / result | Scope |
|---|---|---|
| Historical records / unique frame_id | 1657 | Canonical candidate |
| Unique content_id | 1459 | Exact original image bytes |
| Original train / val / test | 1178 / 107 / 372 | Historical occurrences |
| Valid matched labels | 1657 | Candidate records |
| Empty matched labels | 292 | Candidate records |
| Classes observed | 5 | Candidate annotations |
| Invalid boxes / geometry | 0 | Validated candidate labels |
| Objects in candidate | 4168 | Sum of `num_objects` across 1657 records |
| Labels in entire label archive | 1667 | Includes 10 orphan labels |
| Objects in orphan labels | 14 | Excluded from candidate |
| Objects in entire label archive | 4182 | 4168 matched + 14 orphan objects |
| Exact duplicate content groups | 198 | 396 involved occurrences |
| Cross-split duplicate occurrences | 396 | Historical partition |
| Train–val / train–test / val–test overlap | 57 / 141 / 0 | Unique shared exact contents |
| Validation with an exact train duplicate | 53.27% | 57 of 107 validation records |
| Test with an exact train duplicate | 37.90% | 141 of 372 test records |
| Duplicate groups with annotation conflicts | 8 | Preserve for annotation policy review |

The earlier reported **4182 objects** describes the entire label archive, not the
candidate's matched annotations. This review reconciled the difference by reading
the 10 orphan labels and confirming their 14 objects. No labels, manifest rows or
original files were changed. Local evidence is recorded in
`reports/code_review/annotation_count_reconciliation.json`.

## Real smoke evidence

| Encoder | Model | Contents | Dimension | Representation | Stored quality_valid |
|---|---|---|---|---|---|
| DINOv2 | `facebook/dinov2-small` | 16 | 384 | CLS token | true |
| CLIP | `openai/clip-vit-base-patch32` | 16 | 512 | Projected image embedding | true |

These are pre-existing real runs. Their stored `model_revision` is `unknown`;
the review preserves that evidence rather than assigning a commit retrospectively.
N=16 establishes shape and normalization behavior, not semantic quality across
1459 contents. A local fake-extractor artifact is software QA, never research data.

## Scientific limitations and next gate

Exact duplicates establish historical cross-split content overlap. They do not
measure the resulting detector performance bias, which requires the future
controlled comparison. Filename-derived sequence/frame fields are exploratory;
validated video provenance is still needed for strong temporal-coherence claims.

Before full extraction, pin the chosen HF model commits, record the environment
and clean Git commit, select separate output roots for smoke/full runs of the same
space, and check target-hardware memory with a small run. Then verify 1459 content
rows, complete mapping of all 1657 frame_id, finite nonzero vectors and raw/L2
consistency. Do not compare encoders by individual feature dimensions.

See [methodology traceability](methodology_traceability.md),
[design decisions](design_decisions.md) and [architecture](architecture.md).
