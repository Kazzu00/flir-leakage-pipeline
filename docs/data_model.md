# Data model

| Identity or field | Meaning | Invariant |
|---|---|---|
| `frame_id` | Historical occurrence: hash of archive basename, normalized member path and image SHA256 | Every occurrence is retained, including exact copies |
| `content_id` | SHA256 of original image bytes | One representation/clustering row per exact content; re-encoding can change identity |
| `dataset_id` | Manifest version and sorted occurrence/image/label identities | Portable across local root paths; annotated dataset identity |
| `embedding_row` | Row within the selected feature store | Explicit mapping; unselected smoke contents use -1 |
| `cluster_id` | Assignment within a `clustering_space_id` | IDs are local to a run; noise remains -1 |
| `group_id` | Atomic split allocation unit | One unit per nonnegative cluster; separate singleton IDs for noise |
| `new_split` | New train/val/test assignment within a `split_space_id` | All occurrences of a content and all members of an atomic group stay together |
| `original_split` | Historical occurrence metadata | Preserved for auditing and baselines; never an encoder/clustering input |

```text
frame_id → content_id → embedding_row → cluster_id → group_id → new_split
             dataset_id / feature_space_id / clustering_space_id / split_space_id
```

The historical baseline can assign a single content to several historical splits.
Its content rows preserve membership sets and nullable `new_split` for overlap;
it is an explicit exception to invariants required of new partitions.

Labels describe occurrence-level boxes. Exact duplicate images can have conflicting
annotations; no representative label is silently substituted. Class presence counts
records, while instance counts include every parsed box. Orphans remain outside
the canonical manifest. See [data quality](analysis/data_quality.md).

Temporal provenance retains archive/member, original name, inferred sequence and
frame index, confidence and derivation rule. `filename_heuristic` is an inference,
not a timestamp. Content-level sequence/index requires occurrence consensus;
missing or conflicting provenance stays explicit. Different sequences have no
comparable frame delta. Display FPS does not establish capture FPS.

Exact-byte duplicates, inferred temporal neighbors and high visual similarity are
different relations. None implies a verified semantic scene boundary. Storage
schemas and source fingerprints are described in [architecture](architecture.md)
and [traceability](pipeline_traceability.md).
