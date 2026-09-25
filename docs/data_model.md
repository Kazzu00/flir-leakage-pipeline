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

## Sampled video occurrences (`flir_video_samples_v1`)

This separate manifest version preserves every row of a completed
`extract-video-frames` output, including exact duplicates and explicitly flagged
decode failures. It does not change the historical manifest or its identity.

`frame_id = SHA256(US.join([manifest_version, video_id, source_video_sha256,
float(sample_fps).hex(), str(sample_index), image_sha256]))`, where US is `\x1f`
and the string is encoded as UTF-8. This binds an occurrence to source-video
bytes, the sampling grid and the exact JPEG. Different temporal samples retain
different frame IDs even when bytes match. Absolute roots and row ordering do
not enter identity. `content_id = image_sha256` still means exact JPEG bytes;
visually identical re-encodings can have different IDs.

The unchanged dataset identity algorithm hashes this manifest version and the
sorted `(frame_id, image_sha256, label_sha256)` tuples. `label_sha256=""` and
`label_exists=false` mean **no label**, not an empty annotation file.
`original_split=""` means **unassigned**, never train/val/test. No label/object
statistics or `possible_sequence` / `possible_frame_index` are fabricated.

All sampling columns are retained, plus `source_video_sha256`, decoded width,
height, channels, mode, format, `image_decode_valid` and `image_error`.
`image_path` is a safe POSIX path relative to `frames-root` / `images-root`.
`relative_image_path` and `source_member_path` are explicit compatibility aliases
of that local path; `source_type="directory"`, `source_archive=""` indicate that
there is no ZIP. The directory feature content index retains these semantics.

`exact_duplicate` is true for all occurrences of a repeated content;
`duplicate_occurrence_count` includes every occurrence, and `duplicate_group_id`
is `duplicate-<content_id>` only for repeated contents. Report `duplicate_records`
counts all members of repeated groups; `redundant_records` counts records beyond
one per content. Deduplication affects embedding rows, never occurrence retention.

`video_id` identifies a source file, not a scene/sequence. `sample_index` indexes
the sampling grid, `timestamp_seconds = sample_index / sample_fps` is relative
grid time, and `source_frame_index_estimate` is a nominal FPS-based estimate,
not an exact decoder index or capture timestamp. Unknown source facts stay null.
Sequence identification and partition assignment for these videos remain future
work, irrespective of historical downstream infrastructure.
