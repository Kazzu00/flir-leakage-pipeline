# Detector & Residual Similarity Explorer

Local, read-only inspection at **detector-run level**. The existing
[cluster/split explorer](streamlit.md) and [VIKUS](vikus.md) retain image-level
inspection; this application does not add detector metrics to VIKUS.

```powershell
uv run --extra explorer streamlit run apps/detector_similarity_explorer.py
```

Run from the repository root. The default URL is `http://127.0.0.1:8501`.
To run beside the cluster explorer, append `--server.port 8502`.
The sidebar selects local plan/artifact directories; no remote-machine access,
model loading, image decoding, fitting or training is required.

## Evidence and matrix

The default inputs are `artifacts/detection/protocol/` and
`artifacts/detection/`. The frozen plan defines 16 splits × three detector seeds:
**48 cells**, without changing seeds, hyperparameters or candidate assignments.
The application refreshes local evidence every 30 seconds; Refresh evidence
also invalidates its read cache.

| Association state | Available views |
|---|---|
| PENDING | Frozen residual context, registry and matrix; no scientific detector rows |
| PARTIAL | Progress and validation issues, context and matrix; no scientific detector association or strategy summary |
| COMPLETE_CONTROLLED_COMPARISON | Verified run × class/overall table, scatter selectors and descriptive summaries |

Each matrix cell is PENDING, RUNNING, COMPLETE or INVALID. RUNNING is inferred
from an INITIALIZED/TRAINING/TRAINED progress artifact and does **not** establish
that a process is currently alive. An incomplete or corrupt publication remains
unusable. Duplicate runs, wrong splits, different plans/runtimes, changed contexts,
metric definitions, bootstrap settings and failed source-bound checks block the
complete gate. Keep different runtime experiments in separate artifact roots.

Small pilots are excluded even if copied into the runs directory. Their expander
is labelled **NOT SCIENTIFIC COMPARISON** and contains no performance results.
Complete Stage A cells with the exact frozen configuration may count once in
the final matrix, as already specified by the detector protocol; tiny pilots cannot.

## Scatter views and summaries

Selectors cover mAP@50–95, mAP@50, Precision and Recall; Overall and the five
canonical classes; both encoders' NN means and top-0.1% **pair counts**, temporal
Δ≤5 **fraction**, and class deviation in **percentage points**. Strategies use
fixed colors/markers: Historical, Random content-level,
C10 — DINOv2 / PaCMAP / DBSCAN and C12 — DINOv2 / t-SNE / HDBSCAN.

Each point is a real controlled run. Tooltip and table retain strategy, split
seed, detector seed, split_space_id, detector_run_id, X and Y. Missing values stay
undefined and are omitted only from the plot, with an explicit count; the table
retains them. No seed averaging or jitter is applied. Coincident points can overlap.

The seven combinations in [the registry](../../configs/detection/associations.yaml)
show **PRESPECIFIED ASSOCIATION**. Other selector combinations show
**EXPLORATORY VIEW**. Filtering strategies does not turn an exploratory metric/class
combination into a prespecified one. No outcome-dependent associations are added.

Strategy summaries retain fixed display order, run counts, mean/median/sample SD
of mAP@50–95 and mean residuals over unique frozen splits. Detector-seed variation
is computed within each split; split-seed variation describes those means.
Historical has one split, so its between-split SD is undefined, not zero.
There is no winner or global ranking.

## Interpretation and privacy

**Association ≠ causality.** Test examples, composition and difficulty differ
between strategies; residual similarity is not an isolated experimental variable.
Runs sharing a split share its residual context and test examples. These points
are not independent split observations. No p-values, significance stars,
Pearson/Spearman conclusion, regression or causal effect estimate is produced.
Temporal Δ uses inferred filename indices, not verified time.

Optional top-0.1% fractions come only from saved cohort tables whose checksums and
source metadata match the frozen split. Missing sources leave them undefined;
present but inconsistent sources are rejected. Thresholds and similarity matrices
are never recomputed. Originals and scientific artifacts remain read-only.

The app writes no artifacts. The detector report builder separately writes ignored
tables/figures/HTML; see the [runbook](../runbooks/detection.md). Neither local
run metadata nor generated reports should be published with private evidence.
