# Configuration status

`embeddings/*.yaml` and `similarity/*.yaml` are executable configuration today. Empty `data/`,
`reduction/`, `clustering/`, `splits/` and `detection/` directories reserve planned
configuration boundaries; they do not imply implemented experiments.

| File | Purpose | Model | Device / batch |
|---|---|---|---|
| `dinov2_smoke.yaml` | Active infrastructure validation | `facebook/dinov2-small` | CPU / 8 starting value |
| `clip_smoke.yaml` | Active infrastructure validation | `openai/clip-vit-base-patch32` | CPU / 8 starting value |
| `dinov2_full.yaml` | Executed full closure, pinned HF revision | `facebook/dinov2-small` | CPU / 8 validated |
| `clip_full.yaml` | Executed full closure, pinned HF revision | `openai/clip-vit-base-patch32` | CPU / 8 validated |
| `dinov2_research.yaml` | Planned research candidate | `facebook/dinov2-base` | auto / 8 provisional |
| `clip_research.yaml` | Planned research candidate | `openai/clip-vit-large-patch14` | auto / 8 provisional |

Smoke files configure the encoder, not the sample limit: explicitly pass
`--limit-content 16 --seed 0`. Omitting the limit requests all unique contents.
Full files are the selected small/base model configurations for the week 6 closure:
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
the executed week 9 cosine analysis: float32, top-20, six quantiles, frame-index
bins and explicit analysis rules. They receive the existing full feature directory
through the CLI; no feature IDs or private paths are hardcoded. They do not select
the larger models from the similarly named `embeddings/*_research.yaml` files.
Numerical tolerance and near-unit diagnostic tolerance are not leakage thresholds.
See [similarity execution and cache semantics](../docs/similarity_analysis.md).

No UMAP, clustering, split or detector experiments are configured.
