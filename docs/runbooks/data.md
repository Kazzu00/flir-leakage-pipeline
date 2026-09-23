# Data audit and canonicalization

Run from the repository root with Python 3.11 and `FLIR_DATA_ROOT` pointing to
external read-only originals. Use [.env.example](../../.env.example) as a template
for your ignored `.env`. Public clones do not include the source archives.

```powershell
uv sync --locked --extra dev
uv run flir-pipeline data inventory --inspect-archives --hash-members
uv run flir-pipeline data compare-archives
uv run flir-pipeline data build-manifest
uv run flir-pipeline data validate-labels
uv run flir-pipeline data manifest-summary data/manifests/flir_canonical_candidate_v1.parquet
```

These commands reproduce preparation from originals; they are not required merely
to read existing reports. Use each command's `--help` for explicit input/output
paths. Originals are streamed/read in memory; no source ZIP is rewritten.

The canonical manifest lives under `data/manifests/`, with local inventory,
lineage, label QA and duplicate reports under `reports/`. Follow the directories
returned by the CLI. Preserve prior inventories when archive availability differs.
An absent historical archive is not evidence that canonical coverage is incomplete;
check the actual manifest sources and record the difference.

Validate occurrence/content uniqueness, source hashes, matched versus orphan
labels, exact overlap and conflicting annotations. Keep `original_split` and
all historical occurrences. Never silently repair labels or flatten duplicate
memberships. [Recorded data quality](../analysis/data_quality.md) separates the
canonical annotation universe from the complete labels archive.

Next: [feature extraction and diagnostics](features.md). See [data model](../data_model.md)
and [safety](../data_safety.md) for identity and privacy invariants.
