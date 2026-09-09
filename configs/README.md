# Configuration status

Only `embeddings/*.yaml` is executable configuration today. Empty `data/`,
`reduction/`, `clustering/`, `splits/` and `detection/` directories reserve planned
configuration boundaries; they do not imply implemented experiments.

| File | Purpose | Model | Device / batch |
|---|---|---|---|
| `dinov2_smoke.yaml` | Active infrastructure validation | `facebook/dinov2-small` | CPU / 8 starting value |
| `clip_smoke.yaml` | Active infrastructure validation | `openai/clip-vit-base-patch32` | CPU / 8 starting value |
| `dinov2_research.yaml` | Planned research candidate | `facebook/dinov2-base` | auto / 8 provisional |
| `clip_research.yaml` | Planned research candidate | `openai/clip-vit-large-patch14` | auto / 8 provisional |

Smoke files configure the encoder, not the sample limit: explicitly pass
`--limit-content 16 --seed 0`. Omitting the limit requests all unique contents.
Research files are not final validated hardware configurations. Decide the model
size, pin a full HF `model_revision`, and validate a small run on target hardware
before full extraction. `require_resolved_revision` is true for research and
false for smoke; see [revision semantics](../docs/design_decisions.md).

Raw and L2 float32 arrays are both required. Alternative storage flags/dtypes and
unsupported pooling are rejected instead of silently ignored. `--batch-size`
and `--device` override YAML runtime settings. The YAML extractor selects the
adapter. Use separate `--output-root` directories for sampled/full runs sharing
the same model and feature space, because the cache refuses different selections.

No UMAP, similarity, clustering, split or detector experiments are configured.
