# Clustering memory review against 737e588

This is an operational refactor, not a new scientific protocol. Validation uses
synthetic data only. Full N=8093 clustering and Hypatia RSS measurements remain
pending. Existing reduction artifacts are read only.

## Architecture

`EuclideanDistances` retains vectors without materializing a matrix. Original
evaluation and the original clustering control share one source. Shape/population
checks need no allocation. On request, it executes exactly the former
`squareform(pdist(values.astype(np.float64), metric="euclidean"))` calculation.
No approximation, GPU, alternative distance kernel or disk cache is introduced.

| Consumer | Reuse / release |
|---|---|
| Individual run | Requested space + original evaluation; release on return/error |
| Screening | Reuse across configs of one encoder/representation; release before the next |
| Comparison | Each additional seed run independently; release before the next |
| Source-bound verification | Each run independently; release before the next |
| Direct metrics/exemplars | Only requested sources, released on return/error |
| Selection / signature-only comparison checks | No distances |

Nested retention scopes reuse the same matrix, with cleanup on the outermost
exit, including exceptions. At most two matrices are cached by the sequential
pipeline; an original control needs one. Degenerate silhouettes/empty summaries
need none. A valid cached OPTICS/HDBSCAN run needs none; DBSCAN still derives
epsilon for its ID. Legacy ndarray callers remain supported and own their arrays.
The cache supports one sequential worker and immutable source vectors.

K-distance partitions complete rows in blocks of 256, with unchanged self-index
exclusion and order statistic. In-place partition replaces two NxN copies; the
returned N-vector owns its storage rather than retaining a full partition matrix.
Previous block/cluster temporaries are released before the next is allocated.
Medoid sums/ties, cosine quantiles and silhouette arithmetic remain unchanged.

## Estimated memory at N=8093

| Structure | Decimal GB / binary GiB |
|---|---:|
| One float64 NxN | 0.524 GB / 0.488 GiB |
| Before: 7 spaces × 2 encoders retained | 7.336 GB / 6.832 GiB |
| After loading: retained Euclidean matrices | 0 |
| After: maximum two retained matrices | 1.048 GB / 0.976 GiB |
| Two fully resident float32 cosine maps | 0.524 GB / 0.488 GiB |
| Condensed float64 `pdist` temporary | 0.262 GB |
| New k-distance block | 16.57 MB + boolean checks and N-vector |
| Previous two k-distance copies | 1.048 GB |

For sampled-video numerical clustering, a planning estimate is **3–4 GiB peak
process memory**, not a measured RSS or a hard limit. Identifiable large-array
peaks are around 2.5–3 GB: two distances plus two resident cosine maps (1.572 GB),
two upper-triangle indices for a large cluster (0.524 GB), cosine pair values
(0.262 GB), and quantile/cast workspaces (roughly 0.262 GB). Algorithm fitting and
silhouette occur separately from these pair statistics. One/two dense backend
workspaces put fitting around 2.1–2.6 GB before runtime overhead. Original
384D/512D vectors and their float64 fitting copies add tens of MB.

Before, 7.336 GB of distances plus 0.524 GB of cosine maps and roughly 1–1.6 GB of
temporaries imply **9–9.5 GB of large arrays** before runtime/table overhead. The
guaranteed reduction concerns retained Euclidean matrices: 14 to 2, or 85.7%.
OS residency, native allocators, sklearn version, cluster size and density affect
total RSS. Measure full-process MaxRSS and wall time on Hypatia before selecting
a job reservation; no real-video performance result is claimed.

The same **3–4 GiB planning envelope** applies to sequential sampled-video
screening, comparison and source-bound verification, including their initial
source loading. It is not a guarantee for every backend or allocator:

| Phase | Live large-array estimate / reason |
|---|---|
| Initial source verification | About 2–2.6 GB; sequential reduction ranking dominates, no clustering cache |
| Screening | About 2.5–3 GB; original + active representation persist across configs, plus fitting/evaluation temporaries |
| Comparison | At most the same per-run estimate; seeds/encoders execute sequentially, agreement/selection uses O(N) assignments |
| Source-bound run/collection verification | At most the same evaluation estimate; exact metrics recomputed one run at a time, no refitting |
| Video review after verification | One original matrix for galleries at a time; occurrence tables and figures scale with report size |

Reviewing a comparison verifies its screening and comparison sequentially. This
repeats checks of shared runs but does not overlap their matrices. No source
verification matrices are retained in the returned families.

Remaining O(N²) structures:

- Dense Euclidean matrices and condensed construction temporary; original cosine
  memory maps; noise-excluded silhouette submatrices. The installed sklearn
  silhouette reducer additionally accumulates an `N_clustered × n_clusters`
  float64 array, up to another 0.524 GB. This stage is separate from medoid/pair
  evaluation and fits within the approximate large-array estimate above.
- Per-cluster medoid submatrices, exact intra-cosine pair values/indices and
  quantile workspaces. Quantiles remain exact, with no sampling.
- DBSCAN neighborhoods in dense cases; brute HDBSCAN distance/mutual-reachability
  workspaces. All vector-only CPU adapters, float64 fits, sorting, thread limits
  and sklearn parameters remain unchanged, including OPTICS.
- Historical temporal context still constructs triangular indices and retains
  eligible pairs. If every pair qualified for all three windows, indices alone
  would add about 1.572 GB per encoder. Video skips these pairs; the 3–4 GiB
  estimate does not cover this pathological historical population.
- Source verification is unchanged: v2 similarity compact pairs/statistics, v1
  full pair tables, and sequential reduction distance/rank verification. The
  reduction verifier still computes exact distances for **all required artifacts**
  and has quadratic sort/index temporaries (roughly 2–3 GB of large arrays at
  N=8093, before tables/runtime overhead). It never populates clustering caches.
  Source-validation RSS, especially v1 full tables, is not capped by this cache.
  Reduction ranking retains a float64 score copy and creates full int64 sort
  indices and gathered indices before casting its result to int32. Video v2
  similarity verification retains 21 bytes per unordered pair (about 0.688 GB),
  plus exact float64 quantile workspaces; cosine top-k sorting is already blocked.

Recomputation between comparison seeds and verification runs trades CPU time for
bounded retention. No parallel family execution or process-wide cache is added.

## Compatibility and regression evidence

Before editing code, eager HEAD `737e588f0c7569daafe173e55a29ebba301502fe` executed
a deterministic 90-content synthetic fixture (three blobs plus noise): 18
screening runs and 24 additional seed runs across all three algorithms.
`tests/fixtures/clustering_memory_737e588.json` contains **synthetic-only** digests
of labels, effective parameters, clustering IDs, every metric, medoid summaries,
probabilities, OPTICS diagnostics and eight selection/agreement tables.
Runtime, paths, execution provenance and receipts are excluded.

`tests/clustering_memory_reference.py` defines the configs and projection. The
oracle records NumPy 2.4.6, SciPy 1.17.1 and scikit-learn 1.9.1. Under these
versions tests compare digests exactly. Other versions use a live eager/lazy
comparison, because library versions already enter scientific identity. Tests
never rewrite the oracle; bitwise cross-platform/native-library portability is
not promised by this environment-specific reference.

`tests/test_clustering_memory.py` additionally validates actual collection through
weak references, a maximum of two live matrices across both encoders, requested
spaces only, nested reuse/error cleanup, blocked k-distance equivalence, noise/
singleton/medoid-tie behavior, unchanged exemplars, cache/source corruption
rejection and source-bound screening/comparison verification. It loads real
synthetic t-SNE/PaCMAP artifacts at seeds 0/1/2, proves their bytes unchanged,
rejects corruption and preserves historical `content_cosine_v1`.

Changing `video_id` membership does not create sequence metrics or historical
splits, and selection still omits unavailable criteria. The existing video
integration test now exercises lazy matrices from synthetic v2 artifacts and
rejects invented available temporal metrics. Configs, selection rules, artifact
schemas and scientific IDs are unchanged. New execution hashes/timings differ
naturally; completed caches and receipt-bound comparison IDs remain intact.
