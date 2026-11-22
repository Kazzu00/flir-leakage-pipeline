# FLIR Leakage-Aware Clustering Pipeline

Cierre de resultados 2026-11-22: [tabla final OE4 y originales verificables](docs/results/oe4_20261122/README.md),
seis configuraciones YOLO11m sobre test común; comparación descriptiva de una
semilla, sin afirmación causal ni de independencia. [Tres datasets de entrega y
sus limitaciones](docs/final_datasets_20261122.md). El cierre publica evidencia
recibida; no repite entrenamientos ni evaluaciones. Los registros siguientes
conservan el estado observado en sus respectivas fechas.

Recuperación OE4 2026-11-22: conteos derivados de la vista histórica, preparación
CPU completa antes de GPU y línea denoised exploratoria aislada.
[Despliegue y comprobación de primera época](docs/runbooks/oe4_recovery_20261122.md).

Hotfix OE4 2026-11-22: autorización comunicada registrada, sin infraestructura
SSH de firmas; protocolo y freeze nuevos, controles de integridad conservados.
[Transferencia y ejecución en Hypatia](docs/runbooks/oe4_hotfix_20261122.md).

## Cierre denoised / OE4 — 2026-11-21

Disponible `flir-pipeline closeout`: importación TAR segura, auditoría de metadata,
adaptador de features por contenido único, grafo/MILP candidato y preparación
de test común/OE4. Se auditó el TAR real de 1.116 pares y se materializaron
372 ocurrencias de test local sin liberar evaluación. DINOv2/CLIP completos y
split denoised candidato 777/172/167 materializados y verificados; sin freeze
científico ni óptimo demostrado. Entrenamientos y evaluación GPU siguen pendientes.
Véase el [runbook y comandos Hypatia](docs/runbooks/oe4_closeout_20261120.md).

## Referencia de integración local — 2026-11-14

Este checkout consolida el backend y el frontend existente en `apps/explorer/`,
conservando su historial. Los datasets, modelos y resultados reales permanecen
externos. Véanse [estado actual](docs/current_status.md),
[integración y reproducción](docs/final_integration.md) y
[cinco objetivos definitivos](docs/methodology_traceability.md).

Los registros fechados siguientes describen sus etapas originales. No se
interpreta B-S2/S3 como clustering CLIP/DINOv2 ni se mezclan las 75 corridas B con
el contrato detector histórico. SIFT/RANSAC se conserva como exploratorio y
queda fuera del camino crítico por decisión de JP. Esta candidata local no
autoriza nuevos entrenamientos, evaluación del test o publicación.

## Consolidated workspace

The persistent integration branch contains the backend at `src/` and the explorer
at `apps/explorer/` (Git subtree with preserved history). Run frontend commands
from that directory: `npm ci`, `npm run build`, `npm test`. Existing backend imports
and CLI names are preserved. Original worktrees and repositories are retained.

The [HUD/no-HUD operational report](docs/runbooks/hud_hypatia_handoff.md) separates
the validated BASE preparation from pending annotation review, missing verified
NO-HUD transfers, and training that has **not** run.

The [preliminary BASE runbook](docs/runbooks/preliminary_base_hypatia.md) now
provides a separate user-accepted positive-only pilot: 883 train / 67 val, original
annotations with visual review pending, 185 empty labels excluded. It is locally
prepared and validated, not trained. NO-HUD has no verified box-transfer population.

The operator subsequently submitted BASE job 791494 (remote completion unverified).
[Expanded NO-HUD recovery](docs/runbooks/no_hud_expansion.md) evaluated all 950
sources and 2,624 candidate pairs; no paired training population is admitted.
YOLO11m integration includes an [automatic technical admission and execution
route](docs/runbooks/automatic_yolo11m.md) for the original common-test protocol.
It preserves occurrence annotations and requires verified probe/pilot evidence
before the definitive matrix. Implemented software and offline tests do not imply
executed FLIR training, validated annotation semantics or verified GPU capacity.
The adjudicated A/B candidate described below is a separate earlier experiment.

The new private adjudicated evaluation candidate applies eight authorized existing-B
annotations and conservative isolation of uncertain dependencies. Automatic checks
and unchanged-MILP replay pass; the test is technically frozen only as an internal
retrospective candidate. A/B memberships are identical, remote preflight is pending,
and no detector training is authorized. See the [scope and limitations](docs/runbooks/adjudicated_internal_evaluation.md).

An explicit general A/B assignment adapter is implemented on a separate integration
branch, with the scientific pipeline frozen at `df3452f`. It preserves occurrence
identities, prepares development only and retains the signed admission gates.
Synthetic integration checks pass; real training, test approval and GPU capacity
remain pending. See the [A/B contract and limitations](docs/runbooks/general_ab_executor.md).

Curated development preparation is implemented as
`detection common-executor prepare-curated-development`. It preserves the supplied
manifest and assignments, verifies exact protected-reference exclusion when metadata
is available, and produces Linux YOLO views explicitly **NOT ADMITTED**. The signed
training boundary is unchanged. See the [audit and contract](docs/runbooks/curated_development.md).

## Overview

YOLO11m common-test execution now has a **signed, default-deny admission route**
for existing prepared splits, checkpoint resume and separate final evaluation.
No real training has been executed; final scientific training is not approved. Test/annotation decisions,
administrator trust and end-to-end signed execution acceptance remain pending.
The operator reported a successful GPU/model preflight on Hypatia; this does not
admit the curated population and was not repeated in this integration. See the
[scientific execution runbook](docs/runbooks/scientific_yolo11m.md).

`flir-pipeline detection ground-truth-quality` audits the original proposed test
and support in existing train/validation arms, producing external static galleries,
descriptive sensitivity tables and pending human forms. `verify-ground-truth-quality`
reproduces the evidence without applying labels or admitting training. See the
[quality and sufficiency workflow](docs/runbooks/ground_truth_quality.md).

`flir-pipeline splitting fixed-test-train-val` builds a six-arm pilot followed by
thirty provisional train/validation configurations with an unchanged proposed test.
`verify-fixed-test-train-val --replay` checks protection, coverage, solver evidence
and actual membership aliases. It never admits training. See the
[fixed-test workflow](docs/runbooks/fixed_test_train_val.md).

`flir-pipeline detection protected-annotation-review` produces external visual
evidence and pending human decision forms for the eight protected annotation
conflicts. `verify-annotation-review` replays original sources; decision validation
never applies labels or approves training. See the [review workflow](docs/runbooks/protected_annotation_review.md).

`flir-pipeline detection audit-readiness`, `plan-common-test` and
`verify-common-test` prepare a separately versioned common-test proposal from
verified audits, splits and review receipts. Internal anchors, protected closure,
quarantine and development pools remain distinct; prospective data is unavailable
until independently audited. Proposals cannot train or freeze the detector.
See the [new draft protocol](docs/protocols/detector_common_test_v1.md) and
[planning commands](docs/runbooks/common_test.md).

`flir-pipeline splitting labeled-candidates` builds S1 random exact-content,
S2 balanced exact-content and S3 visual-candidate sensitivity splits from verified
labeled audits, for historical or expanded populations. It reuses the existing
assignment engine, preserves occurrence annotations and keeps Adendo historical
membership null. `splitting verify-labeled` recomputes metrics and optionally
replays assignments. See the [protocol and CLI](docs/runbooks/labeled_candidates.md)
and [local execution evidence](docs/status.md). These are candidates for review,
not evidence of detector improvement or verified temporal independence.

`flir-pipeline data audit-adendo` consolidates the historical labeled population
and Adendo into a versioned candidate audit, preserving original annotations and
separating exact-byte identity from perceptual/review evidence. Implemented and
executed locally on the three source ZIPs; evidence adjudication remains pending.
It creates no splits or training runs. See the [contract and CLI](docs/runbooks/adendo.md)
and [observed audit results](docs/status.md).
**Congelamiento funcional:** producto sin cambios; comparación de organización
55 passed / 34 failed en ambos brazos, con la misma causa previa. Las 77 pruebas
pertinentes y el benchmark controlado aprobaron; la regresión global sigue sin
estar limpia. El [reporte de cierre científico](SCIENTIFIC_CLOSURE.md) separa
validación técnica, evidencia FAC y condiciones pendientes para evaluar detectores.

## Pipeline general de dependencias — 2026-11-06

`flir-pipeline general` incorpora ingesta de imágenes con labels/vídeos opcionales,
catálogo de N clases, grafo de evidencia auditable y dos políticas comparables:
`exact_content_balanced` y `dependency_group_balanced`. Reutilizan el mismo MILP,
objetivo, ratios y seeds 0–4. FAC se integra mediante un adaptador explícito.
No se infiere temporalidad por nombres ni se aceptan candidatos automáticamente.

**Implemented / validated:** cuatro datasets sintéticos independientes, localización
exacta opcional, métricas, publicaciones inmutables y replay; ejecución provisional
FAC con ambas políticas equivalentes. **Pending scientific evaluation:** aceptación
de dependencias, test común legítimo y comparación detector. No autoriza entrenamiento.
La regresión focalizada pasó 77 pruebas; el estado global se detalla en
[estado vigente](docs/status.md).

```powershell
uv sync --locked --extra dev --extra alignment
uv run python scripts/general_synthetic_demo.py D:/research/general-demo-new
uv run python -c "from flir_pipeline.cli import app; app()" general --help
```

El directorio de salida debe ser nuevo. Entradas: lista YAML de imágenes y etiquetas
YOLO opcionales, o manifiesto canónico de adaptador; configuración genérica en
[`configs/general_pipeline.yaml`](configs/general_pipeline.yaml) y
[`configs/general_images.example.yaml`](configs/general_images.example.yaml).

Documentación de cierre: [contrato](PIPELINE_CONTRACT.md), [método](METHOD.md),
[reproducción desde entorno limpio](REPRODUCE.md), [tests de generalización](GENERALIZATION_TESTS.md),
[caso FAC](FAC_CASE_STUDY.md) y [limitaciones](LIMITATIONS.md).
Los apartados siguientes conservan antecedentes y componentes históricos;
el contrato general es la entrada recomendada para nuevos datasets.

## Overview

**JP methodological correction:** filename indices and supplier timestamps are
auxiliary metadata, never evidence of temporal position. Published nominal axes
are superseded for scientific interpretation by a separate video-grounded review;
no existing split is changed. See the [current review contract](docs/runbooks/jp_video_grounded_review.md).

The JP paired-delivery adapter creates an offline comparative timeline and
**provisional, non-training** reproducible assignments from an existing audited
inventory. It preserves uncertain temporal evidence and protected references;
historical candidate links are never promoted automatically. See
[commands, policy comparison and scientific limits](docs/runbooks/jp_paired_delivery.md).

`flir-pipeline sequences audit` provides **FLIR Audit & Human Review v1.1**:
read-only legacy JSON/CSV/ZIP ingestion, `audit-evidence-v1`, a local SQLite
ledger, versioned human decisions and JSON/CSV exports. The reviewer is integrated
in `flir-pipeline-explorer` → Sequences & linkage → Human audit review.
An audiovisual cut and visual-region dependence are separate judgments;
neither a cut nor a whole video defines an independent split group.
See [installation, synthetic smoke, commands, semantics and limitations](docs/runbooks/audit_human_review.md).
See [v1.1 evidence policy, canonical imports and validation](docs/runbooks/audit_human_review_v11.md).

`flir-pipeline sequences alignment` adds bounded CPU multisplit image-to-video
retrieval with media timestamps, explicit alternatives and local visual evidence.
Automatic output remains candidate/ambiguous/not_found; human verification is a
separate explicit action. See [Phase 2A operation](docs/runbooks/phase2a_multisplit_alignment.md).

`sequences alignment pilot` prepares and serves the 24-case human alignment pilot
on loopback, with separate versioned judgments and JSON/CSV exports. The subsequently
received human export contains 24 accepted alignments and 27 historical decisions.
See the [human review runbook](docs/runbooks/phase2a1_human_alignment_review.md).
No sequences, partitions or detector runs are created by this module.

`sequences alignment dependencies` prepares and serves the Phase 2B pair pilot:
24 directed pairs with separate continuity/spatial judgments, explicit evidence
confirmation, independent reviewer ledgers and append-only adjudication. All pair
decisions remain pending. See the [pair review runbook](docs/runbooks/phase2b_dependency_pilot.md)
and [actual validation and limitations](docs/analysis/phase2b_dependency_validation.md).

`flir-pipeline data video-variant-ingestion build/verify/summary` audits external
image variants directly from ZIP shards with bounded reads and immutable evidence.
Scientific occurrence IDs survive unambiguous repackaging; duplicate indices
remain conflicts. Alignment is candidate-only and the historical labeled universe
stays separate. A real ingestion publication is available locally; bounded reads
were validated in earlier phases. `features extract --video-variant-ingestion`
now uses that ledger through the existing feature engine, with explicit
`--input-root`, source-bound checkpoints and synthetic offline CLI validation.
Real embeddings of this variant and subsequent scientific experiments remain
pending. See the [variant ingestion runbook](docs/runbooks/video_variant_ingestion.md)
and [multishard feature commands](docs/runbooks/features.md#features-desde-publicaciones-multishard).

`flir-pipeline explorer export-organization` exports existing frozen M02
record/content, split, clustering, candidate pairs and grouped evidence through
`organization-evidence-v2`. Timelines preserve provenance and optional previews
remain local/ignored. Available with synthetic offline validation; final Hypatia
memberships have not been exported locally. See the
[organization contract and Hypatia runbook](docs/runbooks/organization_evidence_export.md).

`flir-pipeline detection report` implements final evidence-gated detector analysis
and the versioned `detection-export-v1` contract for `flir-pipeline-explorer`.
The owner reports the completed 48-run experiment on Hypatia; those artifacts
are absent from this local clone and have not been reverified here. The command
derives identities from its inputs and refuses incomplete scientific exports.
See the [two-environment reporting runbook](docs/runbooks/detector_final_report.md).

The experimental `flir-pipeline sequences experiment` suite evaluates temporal
boundary zones, visual recurrence and clustering with explicit review masks,
coverage, stability and immutable evidence. It creates no final split or automatic
dependency confirmations. Available with synthetic offline validation; real
Hypatia runs remain pending. See the [Hypatia commands and review workflow](docs/runbooks/sequences.md#suite-experimental-de-temporalidad-y-dependencia-visual)
and [experimental protocol](docs/protocols/sequence_experiments.md). The
[parallel SLURM workflow](docs/runbooks/sequence_operations.md) adds dry-run,
source resolution, durable stage receipts, status and a combined review checkpoint.
Native ingestion now supports the four owner-supplied Hypatia legacy schemas,
with frozen snapshots, cross-report checks and verification of original files.
See the [native import commands](docs/runbooks/native_sequence_evidence.md);
real Hypatia ingestion remains to be executed and verified there.
Generic dataset variants isolate feature stores and experiment identities;
`compare-variants` compares completed suites with optional explicit frame pairing.
See the [future variant example](docs/runbooks/sequence_operations.md#variantes-de-dataset-y-comparación-futura).
Real no-HUD data has not been evaluated; no HUD effect is inferred.

A reproducible pipeline for auditing FLIR frames, measuring visual correlation,
grouping related contents and constructing leakage-aware train/validation/test
partitions. Exact copies and nearby scenes can cross a frame-level split;
content identities and indivisible groups make those relationships auditable.

The system compares historical, reproducible random and cluster-aware partitions
using residual visual and inferred temporal relationships. Reduced correlation
has been observed for selected candidates. The owner reports a complete detector
comparison on Hypatia, pending verification by the final report command there;
detector differences remain descriptive, not causal evidence of leakage inflation.

## Pipeline

```text
Data audit & canonicalization
        ↓
Visual representations
        ↓
Similarity analysis
        ↓
Dimensionality reduction
        ↓
Density-based clustering
        ↓
Cluster-aware splitting
        ↓
Interactive inspection
        ↓
Detector evaluation
```

See [inputs, outputs and execution order](docs/pipeline.md).

## Key capabilities

- ZIP-safe audit, annotation QA and canonical occurrence/content identities.
- Reproducible source-video sampling through external FFmpeg/ffprobe, operationally
  validated on Hypatia: a real 5-second smoke and job 737719 (COMPLETED, exit 0:0),
  producing 9648 JPEGs from three source videos at 1 FPS. This does not identify
  sequences. See the [execution evidence](docs/runbooks/data.md#evidencia-real-confirmada-en-hypatia).
- Sampled JPEGs → versioned occurrence/content manifest → shared DINOv2/CLIP
  extraction from a read-only directory, with exact-byte deduplication. This
  bridge has full Hypatia validation reported by the owner, not reverified locally;
  reviewed video sequences are also reported by the owner and remain unverified
  locally. See [evidence boundaries](docs/status.md).
- Independent DINOv2 CLS and CLIP projected-image embeddings, raw/L2 stores,
  pinned model revisions and resumable extraction.
- Content-level cosine similarity, neighborhoods and posterior temporal analysis.
- Occurrence-level sequence candidates from original CLIP/DINOv2, explicit confirmed
  manual boundary consumption, sequence instances and exact-copy dependency components.
  Synthetic validation only; see the [sequence runbook](docs/runbooks/sequences.md).
- Cross-dataset labeled/video candidate linkage: union of independent CLIP/DINOv2
  top-k sets, separate scores and complete occurrence/sequence ambiguity.
  Synthetic validation only; see the [linkage runbook](docs/runbooks/linkage.md).
- Manual group-level linkage calibration with temporal contact sheets, auditable
  decisions, stratified summaries and source-bound aggregation of immutable
  revisions; no confirmed links or split are created. Real Hypatia v1 init/verify
  compatibility is reported by the owner; aggregation is tested synthetically,
  and future calibration strata are not claimed complete.
  See the [manual review contract](docs/runbooks/linkage_review.md).
- Explicit sampled-video similarity v2 with complete occurrence provenance, exact
  summaries and optional streamed full pairs. Numerical reduction/clustering accept
  this contract; local validation is synthetic, real video execution remains pending.
- t-SNE / PaCMAP with preservation metrics and seed stability.
- DBSCAN / OPTICS / HDBSCAN with original-space metrics, ARI/AMI and explicit noise.
  Exact distances are computed on demand with scoped reuse; see the
  [memory audit and limits](docs/analysis/clustering_memory.md).
- Atomic cluster-aware splitting, seeded baselines and residual cross-split analysis.
- Local Streamlit inspection and VIKUS collection overview.
- Controlled YOLO11n evaluation infrastructure, bootstrap and small CPU pilots.

## Architecture

`src/flir_pipeline/` separates `data`, `features`, `similarity`, `reduction`,
`clustering`, `sequences`, `linkage`, `splitting`, `detection`, `explorer` and `utils`. Metrics live beside
their experiments; optional model/UI dependencies load only where needed.
Source notebooks and report builders consume those components.

See [architecture](docs/architecture.md) and the [data model](docs/data_model.md).

## Quick start

Python 3.11 and `uv` are required. Run commands from the repository root:

```powershell
git clone https://github.com/Kazzu00/flir-leakage-pipeline.git
cd flir-leakage-pipeline
uv sync --locked --extra dev --extra reduction
uv run flir-pipeline --help
```

Create an ignored `.env` using [.env.example](.env.example) as a reference:

```dotenv
FLIR_DATA_ROOT=/path/to/external/flir-data
```

The environment takes precedence over `.env`; data commands also accept explicit
paths. Public clones contain no real data or executed artifacts.
Optional extras are `vision`, `reporting`, `explorer` and `detection`.
`uv sync` installs exactly the selected extras; include all extras you need.
Model weights are separate from the lockfile. See [runbooks](docs/README.md).

## CLI

Use `uv run flir-pipeline <namespace> --help` for options.

| Namespace | Implemented commands |
|---|---|
| `data` | `inventory`, `extract-video-frames`, `build-video-manifest`, `archive-tree`, `compare-archives`, `build-manifest`, `validate-labels`, `manifest-summary` |
| `features` | `extract`, `diagnostics`, `summary`, `verify`, `visualize-data`, `visualize-embeddings` |
| `similarity` | `compute`, `summary`, `verify`, `compare` |
| `reduction` | `run`, `benchmark`, `verify`, `summary` |
| `clustering` | `run`, `sweep`, `compare`, `verify`, `summary` |
| `sequences` | `detect`, `build`, `verify`, `summary` |
| `linkage` | `build`, `verify`, `summary`; `review init/record/summary/verify/aggregate/aggregate-verify` |
| `splitting` | `build`, `baseline`, `evaluate`, `compare`, `summary`, `verify`, `export-lists` |
| `detection` | `plan`, `materialize`, `environment`, `smoke`, `probe`, `pilot-small`, `freeze`, `run`, `verify` |
| `explorer` | `vikus-build`, `vikus-serve` |

## Interactive exploration

[Streamlit](docs/visualization/streamlit.md) provides detailed cluster/split
inspection, galleries, reconstructed frame playback, timelines, gaps and comparisons:

```powershell
uv run --extra explorer streamlit run apps/cluster_split_explorer.py
```

[VIKUS](docs/visualization/vikus.md) provides a global collection view by cluster,
sequence, split and saved PaCMAP coordinates. For the existing
**C10 — DINOv2 / PaCMAP / DBSCAN** candidate:

```powershell
uv run flir-pipeline explorer vikus-build --candidate C10 --seed 0 --name c10-seed0-local
uv run flir-pipeline explorer vikus-serve --bundle reports/explorer/vikus/c10-seed0-local
```

Both require existing local artifacts and read original ZIPs without mutation.
Playback FPS controls display only; there are no verified timestamps.
The first VIKUS build downloads a checksum-pinned runtime; cached builds support
`--offline`. Image bundles remain private and local.

The separate [Detector & Residual Similarity Explorer](docs/visualization/detector_similarity.md)
shows the 48-cell experiment matrix and frozen residual context now. Run-level
scatters activate only after the complete controlled comparison; pilots are excluded.

```powershell
uv run --extra explorer streamlit run apps/detector_similarity_explorer.py
```

## Data and artifacts

Original ZIPs live outside Git under `FLIR_DATA_ROOT` and remain read-only.
Local manifests, embeddings, assignments, previews, VIKUS bundles, figures,
executed notebooks and HTML are protected by [.gitignore](.gitignore).
Only source code, generic configurations, aggregate documentation and clean
notebook sources are versioned. See [data safety](docs/data_safety.md).

## Reproducibility

Deterministic dataset/space IDs, configurations, seeds, source checksums and
artifact metadata preserve lineage from each `frame_id` to its `content_id`,
embedding row, cluster and partition. Historical membership and labels never
enter visual representation or clustering. Labels support later split balance.
See [reproducibility](docs/reproducibility.md), [configuration status](configs/README.md)
and [pipeline traceability](docs/pipeline_traceability.md).

```powershell
uv run ruff check .
uv run python -m pytest
uv run python scripts/check_notebook_source.py
```

Tests are synthetic and offline, with no FLIR data, model downloads or GPU.

## Current status

| Component | State |
|---|---|
| Audit and canonicalization | Validated: 1657 historical records / 1459 unique contents |
| DINOv2 / CLIP and cosine similarity | Validated on complete content coverage |
| t-SNE / PaCMAP | Validated: 36 executed reductions |
| Density clustering | Validated with limitations: 414 runs; exploratory candidates |
| Cluster-aware splitting | Validated with limitations: 66 runs; residual correlation remains |
| Temporal interpretation | Experimental: filename-derived indices; no verified timing |
| Streamlit / VIKUS | Available for local inspection |
| Detector infrastructure | Available; four small CPU pilots validated |
| Full detector comparison | 48/48 complete on Hypatia reported by owner; final-report verification pending there, no local scientific export |
| Detector report / frontend contract | Implemented; synthetic offline validation, final-only integrity gate and versioned JSON Schema |

Detailed evidence, candidate roles and remaining limits are in [status](docs/status.md).

## Documentation

Start with the [documentation index](docs/README.md): architecture, data model,
pipeline, reproducibility, safety, status and design decisions; functional
analysis, protocols, runbooks and visualization guides.
The [consolidated project report](docs/runbooks/project_report.md) reuses existing
results without rerunning experiments. Technical sources are in
[references](docs/references.md).
