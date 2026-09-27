# Cross-dataset labeled/video candidate linkage

`linkage` generates visual candidates between the canonical labeled manifest and
the independently sampled video manifest. It consumes existing original features
and reviewed video sequences. Local validation uses synthetic inputs only; no
real dataset, image, label file or model is loaded by this implementation task.

## Inputs and scientific policy

Supply seven sources: the canonical labeled manifest, sampled-video manifest,
labeled CLIP/DINOv2 stores, video CLIP/DINOv2 stores, and a completed reviewed
`sequence_instance_set` directory. Each feature store must cover its own manifest
fully with valid float32 raw/L2 arrays, content/record indexes and a resolved model
revision. CLIP pairing must share its mathematical feature space; DINOv2 pairing
must share its own. The labeled and video **dataset IDs may differ**. Each
content_index supplies the row mapping for that particular store.

The kernel compares unique labeled content against unique video content. Existing
L2 vectors remain unchanged; dot products accumulate in float64 with a fixed
kernel independent of block size. Per encoder, order by cosine descending and
then video_content_id ascending for exact ties. Take the union of both top-k
sets. Default k is 10; effective k is min(k, video unique-content count).

Each candidate retains separate `clip_cosine` / `dinov2_cosine`, nullable
`clip_rank` / `dinov2_rank` in the corresponding top-k only, and `clip_topk`,
`dinov2_topk`, `both_topk`. Both cosines remain available even if only one encoder
selected that candidate. `mean_reciprocal_rank` equals
`(1/clip_rank + 1/dinov2_rank)/2`, with missing rank contributing zero. It is not
a probability or acceptance threshold. Raw encoder cosines are never averaged.

Queries are processed in blocks of at most 32, with one encoder score block held
at a time; only its top-k indexes survive before scoring the next encoder. The
small union is rescored with the same kernel. Full cross-dataset matrices are not
published or required in memory. Peak runtime/memory on real inputs remain to be
measured. Slight values outside [-1, 1] from stored float32 normalization are
preserved rather than clipped into artificial ties.

## Commands

From the repository root, using local artifact paths in place of placeholders:

```bash
uv run flir-pipeline linkage build \
  --labeled-manifest artifacts/manifests/LABELED.parquet \
  --video-manifest artifacts/manifests/VIDEO.parquet \
  --labeled-clip artifacts/features/LABELED_CLIP \
  --labeled-dinov2 artifacts/features/LABELED_DINOV2 \
  --video-clip artifacts/features/VIDEO_CLIP \
  --video-dinov2 artifacts/features/VIDEO_DINOV2 \
  --sequence-set artifacts/sequences/sets/SEQUENCE_SET_ID \
  --output artifacts/linkage --top-k 10

uv run flir-pipeline linkage verify artifacts/linkage/LINKAGE_ID \
  --labeled-manifest artifacts/manifests/LABELED.parquet \
  --video-manifest artifacts/manifests/VIDEO.parquet \
  --labeled-clip artifacts/features/LABELED_CLIP \
  --labeled-dinov2 artifacts/features/LABELED_DINOV2 \
  --video-clip artifacts/features/VIDEO_CLIP \
  --video-dinov2 artifacts/features/VIDEO_DINOV2 \
  --sequence-set artifacts/sequences/sets/SEQUENCE_SET_ID

uv run flir-pipeline linkage summary artifacts/linkage/LINKAGE_ID
```

In PowerShell, use a backtick for continuation or put each command on one line.
Build publishes below `--output/LINKAGE_ID` (default root `artifacts/linkage`).
Existing complete publications are verified and reused unchanged. Partial or
inconsistent directories are refused and preserved. Verification requires the
same seven sources and exits nonzero on failure. `--top-k` is positive and belongs
to the build configuration, recovered from metadata during verification.

## Artifact and all-occurrence joins

| File | Contract |
|---|---|
| `metadata.json` | Completion marker; kind `labeled_video_link_candidates`, version 1, ground_truth=false, requested/effective k, dataset/feature-space/sequence-set IDs, configuration, source/output checksums, execution git commit/time and explicit candidate semantics |
| `content_candidates.parquet` | One row per labeled-content/video-content pair; deterministic candidate_id, separate scores/ranks/flags, rank consensus and video occurrence/sequence counts |
| `candidate_occurrences.parquet` | Normalized relation: one row per selected video's source occurrence, with video_content_id, video_frame_id, video_id, sample_index, sequence_id and temporal/feature provenance |
| `labeled_occurrences.parquet` | Full canonical manifest snapshot; every frame_id, content_id, original_split and available annotation/provenance fields |
| `summary.json` | Counts only; populations, candidate pairs, both-top-k pairs, per-query min/max, and pairs with multiple video occurrences/sequence instances |

Expand a candidate to **all** video occurrences by joining on `video_content_id`.
The occurrence table is shared between candidates for the same video content;
it is not a per-candidate representative. Expand labeled provenance by joining
`labeled_content_id` to **all** `labeled_occurrences.content_id` rows. Example:

```python
expanded_video = candidates.merge(occurrences, on="video_content_id", how="left")
expanded_labeled = candidates.merge(
    labeled_occurrences,
    left_on="labeled_content_id",
    right_on="content_id",
    how="left",
)
```

Duplicate labeled occurrences share one query embedding but retain distinct
frame_ids, historical splits and conflicting annotation metadata. Candidate rows
have no assigned sequence. The video occurrence relation may include several
sequence_ids for the same content. An existing sequence is a continuous temporal
segment, not a cluster. No links are auto-confirmed; no visual_dependency_group,
train/val/test assignment or altered splitting semantics is produced.

## Verification and limits

`verify` checks complete feature coverage, original row mappings, numerical health,
mathematical pairing and source checksums. It validates the sequence publication's
checksums/source binding, parses its frozen confirmed review, and reconstructs
its partition using the existing sequence construction and invariant checks.
The authoritative review originated from `boundary_validation.csv`; the accepted
subset never overrides its decisions. Confirmed review remains ground_truth=false;
high_confidence remains annotation only. Linkage never reruns or replaces F3.

This consumer uses the stored review snapshot; the external three-file validation
directory is not an input. Run `sequences verify` with that original directory for
the full independent boundary detector/F3/manual CSV check. No reduction,
clustering or split artifact is needed by linkage.

Candidate verification recomputes the exact union, scores, ranks, flags and all
occurrence joins. It rejects foreign/missing/duplicate content IDs, incorrect
spaces, invalid arrays or indexes, candidate tampering and any collapsed lineage,
even if output checksums were updated. `summary` reads metadata and summary JSON
only; stored counts are not a verification receipt or confirmed-match count.

The reported real counts and zero exact cross-dataset SHA256 overlap are context
provided by the project owner, not assumptions in production logic. Matching
works whether exact overlap is zero or nonzero. Real feature compatibility,
sequence publication compatibility, runtime/memory, candidate counts and visual
relevance still require execution and review on Hypatia. No real artifact is
required in the public checkout or in tests.
