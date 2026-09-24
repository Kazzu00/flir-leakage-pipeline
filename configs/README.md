# Configuration status

`embeddings/*.yaml`, `similarity/*.yaml`, `reduction/*.yaml` and
`clustering/*_research.yaml` and `splits/*.yaml` are executable configurations.
`detection/yolo11n.yaml` configures the implemented detector protocol.
Data preparation uses CLI options; no empty configuration boundary is advertised.

| File | Purpose | Model | Device / batch |
|---|---|---|---|
| `dinov2_smoke.yaml` | Active infrastructure validation | `facebook/dinov2-small` | CPU / 8 starting value |
| `clip_smoke.yaml` | Active infrastructure validation | `openai/clip-vit-base-patch32` | CPU / 8 starting value |
| `dinov2_full.yaml` | Executed full extraction, pinned HF revision | `facebook/dinov2-small` | CPU / 8 validated |
| `clip_full.yaml` | Executed full extraction, pinned HF revision | `openai/clip-vit-base-patch32` | CPU / 8 validated |
| `dinov2_research.yaml` | Planned research candidate | `facebook/dinov2-base` | auto / 8 provisional |
| `clip_research.yaml` | Planned research candidate | `openai/clip-vit-large-patch14` | auto / 8 provisional |

Smoke files configure the encoder, not the sample limit: explicitly pass
`--limit-content 16 --seed 0`. Omitting the limit requests all unique contents.
Full files are the selected small/base model configurations for full extraction:
both completed 1459 unique contents after real N=16 validation using the same pinned
revisions. Research files are larger, unexecuted candidates and do not supersede
that completed choice. Research files are not final validated hardware configurations. Decide the model
size, pin a full HF `model_revision`, and validate a small run on target hardware
before full extraction. `require_resolved_revision` is true for full/research and
false for smoke; see [revision semantics](../docs/design_decisions.md).

Raw and L2 float32 arrays are both required. Alternative storage flags/dtypes and
unsupported pooling are rejected instead of silently ignored. `--batch-size`
and `--device` override YAML runtime settings. The YAML extractor selects the
adapter. Use separate `--output-root` directories for sampled/full runs sharing
the same model and feature space, because the cache refuses different selections.

`similarity/dinov2_research.yaml` and `similarity/clip_research.yaml` configure
the executed cosine analysis: float32, top-20, six quantiles, frame-index
bins and explicit analysis rules. They receive the existing full feature directory
through the CLI; no feature IDs or private paths are hardcoded. They do not select
the larger models from the similarly named `embeddings/*_research.yaml` files.
Numerical tolerance and near-unit diagnostic tolerance are not leakage thresholds.
See [similarity execution and cache semantics](../docs/analysis/similarity.md).

No UMAP experiments are configured. `detection/yolo11n.yaml` predefines a single
controlled detector configuration and target matrix; batch is chosen by a
hardware probe and frozen before real training. CPU Stage B is never automatic.

`detection/associations.yaml` predeclares seven descriptive detector/residual
views before controlled results exist. It does not change `yolo11n.yaml` or the
frozen training plan. Other interactive metric/class combinations are exploratory;
scientific tables and plots require the complete controlled matrix.

`splits/random_baseline.yaml` defines the content-level random baseline with
seeds 0–4. `splits/cluster_aware_research.yaml` defines singleton noise, five
assignment seeds and deterministic SciPy MILP limits. Both derive target record
ratios from the supplied historical manifest when `target_ratios: null`; an
explicit train/val/test ratio tuple is supported. Neither file contains local
paths or clustering IDs. The [splitting protocol](../docs/protocols/splitting.md)
defines diverse candidate selection and posterior evaluation in both encoders;
the [runbook](../docs/runbooks/splitting.md) documents execution and later export.

`clustering/dbscan_research.yaml` defines 18 configurations using min_samples
5/10/20 and k-distance quantiles .80/.85/.90/.95/.97/.99; epsilon is recomputed
in each space/seed. `optics_research.yaml` defines 27 xi configurations and
`hdbscan_research.yaml` defines 12 EOM configurations, generic across encoders
and representations. `clustering/inputs.example.yaml` is a template with
placeholders, to complete locally with existing verified source paths. The
[runbook](../docs/runbooks/clustering.md) explains screening, bounded stability,
verification and reporting.

`reduction/tsne_research.yaml` uses perplexity 10/30/50 and seeds 0/1/2;
`reduction/pacmap_research.yaml` varies MN_ratio 0.2/0.5/1.0 with the same seeds.
Both use 2D and direct full L2 input, with PCA initialization only. Generic configs
receive explicit full feature/similarity directories from the CLI, separately for
each encoder. The parser rejects unsupported methods, hidden pre-PCA and duplicate
or oversized grids. See the [predeclared protocol](../docs/protocols/reduction.md)
and [execution runbook](../docs/runbooks/reduction.md).
