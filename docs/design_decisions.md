# Design decisions

These decisions describe the current implementation and the supplied thesis
methodology. Later sections record the executed similarity/reduction extensions.
Clustering, new splitting and detector training remain unimplemented.

## 1. Separate occurrence from exact content

`frame_id` identifies a portable occurrence: SHA256 of archive basename, normalized
ZIP member path and image SHA256 separated by a unit separator. `content_id` is
the SHA256 of original image bytes. The candidate contains 1657 occurrences,
1459 contents and 198 exact duplicate groups. A re-encoded image can be visually
identical and have a different content_id; exact-byte identity is not semantic
scene identity. `duplicate_group_id` is `duplicate-` plus content_id only for
repeated content. Original bytes and historical occurrences are preserved.

## 2. Compute embeddings once per content_id

Storage decodes one representative occurrence per selected exact content. This
avoids redundant computation and prevents exact copies from inflating local
density in future DBSCAN/HDBSCAN/OPTICS experiments. It does not remove near
duplicates or establish independence between temporally related contents.

## 3. Preserve the mapping to every frame_id

`content_index.parquet` maps each content to its embedding row and representative
source. `record_index.parquet` maps all 1657 historical occurrences to content and
row. Smoke runs use `embedding_row = -1` for contents outside the selected sample;
these rows are not missing or zero embeddings. Full extraction must cover every
content and eliminate these sentinels. Future cluster and split IDs must map back
through this relationship without dropping occurrence history.

## 4. original_split is historical metadata

Normalize `validation -> val` and `training -> train`; use `train`, `val`, `test`
consistently. Historical split membership supports auditing and baseline comparison.
It is not an input to feature extraction, similarity representation, dimensionality
reduction or clustering. Reading it for descriptive plots does not make it a model
input. Future allocation must keep whole clusters/scenes in one partition and
document class coverage and any tradeoffs.

## 5. Labels and boxes are not encoder inputs

DINOv2 receives images and returns CLS tokens. CLIP receives images and returns
projected image embeddings, without text prompts. Neither receives labels, boxes
or classes. Annotation hashes enter dataset provenance so label changes remain
traceable, not the mathematical image representation. Eight duplicate groups
have annotation conflicts; retain that evidence and resolve label policy before
detector evaluation rather than silently choosing a label per content.

## 6. Handcrafted diagnostics remain QA/EDA

Pixel statistics, grayscale entropy, Laplacian variance, pHash and dHash describe
the image data. They are neither automatic exclusion criteria nor automatically
concatenated with learned embeddings. Pixel scaling currently assumes 8-bit
images; other bit depths require a separately justified diagnostic policy. The
custom pHash implementation is a local QA descriptor, not a standardized hash
compatibility claim. It never substitutes for SHA256 exact-content identity.

## 7. Keep DINOv2 and CLIP spaces independent

Each extractor has separate model, preprocessing, pooling and feature-space
metadata. Do not concatenate their embeddings automatically. The selected full
closure models are DINOv2-small (384D) and CLIP ViT-B/32 (512D), validated on all
1459 contents. Existing research configs nominate larger candidates and remain
unexecuted; they do not supersede this bounded model choice.

## 8. Conditional Bhattacharyya and committed reductions

Bhattacharyya distance requires an explicitly defined distributional representation.
No such representation has been selected; implementation is deferred. The proposal
commits to t-SNE and PaCMAP. UMAP is outside that experiment and its unused
dependency has been removed without replacing it with new techniques.

## 9. Model revisions and Hugging Face provenance

`model_revision` in YAML is an optional HF branch, tag or full commit SHA passed
to `from_pretrained(revision=...)`; null uses the HF default. For final experiments
set an explicit full commit SHA. Hugging Face supports revision-specific loading:
[official loading documentation](https://huggingface.co/docs/transformers/models).

After loading the model, the adapter reads Transformers' optional
`model.config._commit_hash`. If it is a full SHA, that is the effective revision
and is also passed to the processor loader. An explicitly requested full SHA is
the fallback if that attribute is absent. A conflicting requested/loaded SHA
raises an error. No separate Hub API request is made; `local_files_only` propagates
to both loaders. An arbitrary local model directory is not automatically a
verifiable HF snapshot: preserve its weight checksums separately.

New metadata fields have distinct meanings:

| Field | Meaning |
|---|---|
| `requested_model_revision` | User-supplied ref or null |
| `resolved_model_revision` | Full immutable commit, or null if unavailable |
| `model_revision` | Resolved commit; otherwise requested ref or `unknown` |
| `model_revision_source` | Config attribute, requested commit, or unresolved |
| `library_versions` | Installed torch and Transformers versions |

`require_resolved_revision: true` in research configs rejects unresolved loads.
Smoke configs allow unresolved metadata with a warning. The optional Transformers
attribute is isolated in a helper and tested with offline stand-ins for both
adapters. The later week 6 closure also validated real N=16 loads and complete extractions
with pinned revisions, resolving both SHAs from the loaded config in offline mode.

Previously validated smoke artifacts record `unknown`. They remain unchanged and
readable by summary/verification/reporting. A new resolved SHA can change
`feature_space_id`; safely identifying weights takes precedence over reusing an
unverifiable cache. We do not relabel old artifacts or claim their revision was
recovered. Device/batch changes still leave mathematical feature identity stable.

## 10. Dataset identity must come from the supplied manifest

The canonical algorithm hashes manifest version and sorted tuples of frame_id,
image SHA256 and label SHA256. A shared helper now computes it directly for data,
features, diagnostics and reports. It preserves the existing canonical v1 identity.
An unrelated `reports/data_manifest/` in the working directory must never supply
another dataset's ID. Unversioned synthetic manifests retain the original feature
fallback algorithm. Optional explicit dataset IDs in the Python storage API are
caller-managed namespaces and must be documented when used.

## 11. Cache integrity and honest reports

A cache signature includes dataset, feature space, selected contents and image
hashes. Completed caches are verified before reuse. Shape, finite values, nonzero
norms, raw/L2 consistency, unique IDs and occurrence mappings determine validity.
Batch checkpoints advance only after arrays are flushed. Metadata is the final
completion marker. One writer is supported per feature directory; concurrent
writers and recovery from a crash during final file promotion remain unsupported.
An incomplete publication is preserved and refused if checkpoint arrays are missing.

Use separate output roots for smoke and full extraction with the same model/space:
selection does not change feature_space_id, and a mismatched selection is refused.
Diagnostics do not provide a resumable multi-run store; separate sampled and full
diagnostic roots to avoid overwriting a previous diagnostic report.

Report inputs are explicit or unambiguous. The builder's default complete mode (`--full`) filters
by canonical dataset and complete coverage, verifies provenance, requires both
encoders, and still rejects multiple eligible runs. Hash ordering never chooses an
experiment. Use `--no-full` and explicit paths to report a sampled artifact.
Class-presence charts count records, not objects; box-area and aspect-ratio
boxplots use individual instances grouped by canonical class name. Sample sizes
and embedding health come from selected artifacts.
Source notebooks have no outputs; executed notebooks and HTML stay local.

## 12. Annotation universes and temporal evidence

Class presence counts each class once per historical occurrence; instances count
every parsed box. The audit reads label ZIP members in memory, checks matched
SHA256/object counts against the manifest, and reports orphans separately.
Duplicate annotation conflicts remain unchanged and visible. Empty/non-empty
are two complementary categories with counts and percentages.

Temporal lineage retains source archive/member, frame/content IDs, original split,
possible sequence/index and confidence. The source is explicitly
`filename_heuristic`; no timestamp is fabricated. Nearby pairs require the same
archive/sequence, different historical splits and index gap <= a stated threshold
(default 1, including equal indices). Exact content equality is a separate flag.
This auditable candidate rule is not an embedding-similarity experiment or proof
of leakage for distinct contents. The retained historical mapping is a data
baseline, not an executed detector baseline.

A full closure additionally compares record/content mappings to the canonical
manifest and requires resolved revision metadata. Equal numerical embeddings for
different contents are permitted. Full numerical validity does not establish
semantic quality, clustering structure or detector improvement.

## 13. Class nomenclature and instance geometry

The original YAML explicitly maps IDs 0–4 to vehicle/building/road/river/SDZI.
Canonical display names follow the publication order confirmed by the project
owner on 2026-09-13; `SDZI` remains the original label for Heavy Machinery (4).
This is order correspondence, not a demonstrated expansion of the source term.
The exact bibliography remains pending; see [evidence and limits](dataset_classes.md).
One immutable catalog supplies report, plot and manifest-summary names.
The generic YOLO parser does not impose the five-class ontology.

Geometry uses the source-precision parsed boxes rather than rounded canonical
tuples intended for conflict comparisons. Historical occurrences each contribute
their own boxes; orphans are excluded and empty labels do not fabricate boxes.
The four metrics are normalized width, height, their product and their ratio.
The ratio of normalized axes differs from pixel aspect ratio on non-square images.
Report sample standard deviation and linear quartiles, retain outliers, and
explicitly count geometry exclusions for invalid labels. No label is repaired.

## 14. Content-level cosine and posterior provenance

Week 9 uses existing float32 L2 vectors without silent normalization, clipping or
diagonal replacement. A full matrix is small enough at the current 1459-content
scale. All-pair statistics use i<j, while top-k tables contain directed edges.
Identical vectors from different contents remain eligible neighbors. Self exclusion
uses content identity; equal scores use ascending content ID as a stable tie break.
Statistics use population standard deviation and linear quantiles. The six upper
cohorts include threshold ties and are nested, so their counts must not be summed.
Encoder-specific quantiles express relative similarity, not a shared leakage cutoff.

Temporal/split metadata is joined only after the dot product and ranking. All
occurrences must agree on known archive/sequence and index before content-level
time is used. Conflicts/missing values stay explicit; different sequences have
null frame_delta. Split membership retains every known train/val/test occurrence.
A distinct-content pair crosses historical splits if some occurrence on each side
has unequal membership; identical multi-split sets can therefore cross. Unknown
provenance is not encoded as a negative relation. Counts describe candidates only.

Similarity identity includes dataset, feature and scientific analysis settings;
runtime paths, timestamps, device and batch are excluded. Because the existing
dataset identity does not include temporal/split fields, a separate posterior
manifest fingerprint prevents reuse after those fields change. Input/output
checksums, executable QA and optional original-source verification protect stored
arrays, row mappings, ranks and annotations. Partial directories are preserved and
refused; a separate output root is required for a distinct source revision under
the same identity. One writer per directory; no distributed writer protocol.

Execution metadata records the actual prior Git commit and dirty-worktree flag
when code has not yet been committed, plus source-file hashes. It is never
retroactively rewritten to claim execution under a later commit. Final verification
receipts can link the committed implementation to the preserved experiment.

## 15. Bounded visual review and encoder agreement

Jaccard aligns query content IDs and compares top-k sets for k=1,5,10,20. k=1 also
measures exact nearest-neighbor agreement. Neither score establishes superiority;
the spaces are never concatenated. Rank correlation was optional and remains
unimplemented to keep this phase focused on interpretable set agreement.

The report verifies the chosen similarity artifacts and the comparison's source
fingerprints, then recomputes Jaccard. Global figures use all applicable pairs or
queries. Galleries use three shared queries selected without replacement from
sorted IDs by NumPy's seeded generator; this is not a stratified coherence audit.
Only selected image bytes are read from the original ZIP and checked before
in-memory decoding. Private IDs/hashes are retained in ignored selection and
inspection tables; displayed tables use aggregate or ordinal identifiers.
Maximum-similarity pairs are inspected even when no pair meets the numerical
near-unit tolerance. RGB pixel equality and exact-byte equality remain separate.
Overlays/menus are visible in source images; their possible influence has not been
isolated experimentally. All image composites, HTML and executed notebooks stay local.

## 16. Direct L2 reduction and source-bound preservation

t-SNE and PaCMAP use independent full L2 spaces, with no input pre-PCA. PCA
initializes low-dimensional positions only. PaCMAP's default preliminary PCA is
disabled explicitly; its remaining scalar rescaling/centering and actual full
input dimension are recorded. The direct grid is feasible at N=1459, avoiding an
additional preprocessing experiment at this stage. 3D is supported by configuration
but only 2D is executed. The numerical adapter accepts no labels, split or time.

Reduction identity includes dataset/feature, method, parameters, seed, dimensions,
preprocessing and implementation versions; operational paths/time/threads are
excluded. Evaluation settings are also included because metrics share the stored
artifact. Input fingerprints protect the original vectors, indices and similarity
reference. Existing record_index is referenced, not duplicated. Completed runs
are immutable checkpoints for a grid; partial outputs are preserved and refused.

Exact full ranks define trustworthiness and continuity, with k=5/10/20 and an
explicit content-ID tie rule. Existing cosine neighbors define set preservation;
Euclidean distances define reduced neighbors. Canonically sampled unique pairs
define complementary Spearman; pair dependencies preclude naive significance
claims. Seed comparisons use neighbor-set Jaccard, not raw coordinate differences.

## 17. Predeclared exploratory references and interpretation

The [protocol](reduction_protocol.md) fixes three configurations per method and
three seeds, then a Pareto/mean-rank rule over T, C, Jaccard, stability and Spearman.
The last criterion has one fifth of the rank weight, with no primacy. Criteria
are correlated; this rule is explicit and exploratory, not a validated objective
for detector performance. All alternatives remain available. Seed 0 always
represents a selected configuration. Missing distance correlation in any seed
precludes candidate eligibility rather than silently averaging fewer seeds.

The report uses a posterior temporal window selected independently of geometry:
first sorted eligible sequence, longest consecutive unambiguous index stretch,
centered window of up to 30 contents. All historical memberships are preserved
as sets. Neither colors nor trajectories influence fitting or selection. Nonlinear
2D density cannot justify density-based clustering without evaluating stability,
coherence, noise and original-space relationships in the next phase.

## 18. Density clustering controls, geometry and identity

Week 10 executes the [density protocol](clustering_protocol.md) on unique
contents: original DINOv2/CLIP L2 controls and the four selected 2D reductions.
Original controls are ablations, preserving t-SNE/PaCMAP in the proposal path.
Euclidean on unit embeddings preserves cosine neighbor order through
d²=2(1−cos), subject to numerical ties. No renormalization or coordinate z-score
is introduced. Nonlinear density remains representation-dependent.

DBSCAN epsilon derives from per-space/per-seed k-distance quantiles, with
min_samples including self. Identical effective parameters are aliases, not
extra experiments. OPTICS uses xi with infinite max_eps; HDBSCAN uses the
installed scikit-learn EOM implementation, avoiding a second library. Its
membership probabilities are stored; absent persistence/outlier metrics are
explicitly unavailable. All adapters use a stable content-ID ordering.

clustering_space_id binds dataset, feature, reduction when applicable,
representation, algorithm, effective/conceptual parameters and implementation
versions, excluding operational paths, dates and device. Input signatures bind
the original and reduced sources. Completed runs are grid checkpoints;
incomplete outputs are preserved. The pre-fit protocol/source snapshot and
actual fit Git/dirty provenance remain intact after verifier improvements.

## 19. Noise, posterior metrics and bounded candidate selection

Noise remains −1 and is never silently converted into singleton clusters.
Primary silhouette excludes noise and uses exact distances in the original
encoder space, even for 2D clustering. Original-distance medoids represent each
cluster. Cosine cohesion weights unique intra-cluster pairs; member-weighted and
median-of-cluster-means summaries are separate diagnostics. Coverage accompanies
every conditional interpretation.

Temporal recall uses all same-sequence pairs within the inferred index window;
noise endpoints fail retention. Visual neighbor coherence excludes noise queries,
keeps noise neighbors as failures and reports query coverage plus an all-query
version. Sequence fraction/entropy and historical membership unions are strictly
post-fit. Labels/classes are not fitting or selection inputs. Temporal coherence
can inform posterior selection, so this is not a geometry-only selection rule.
Sequence purity and cluster count have no automatic better/worse direction.

ARI/AMI are computed on all points and on common clustered points separately,
with N, coverage, triviality and means/minima. Seeds perturb the reduction,
whereas adjacent grid parameters perturb clustering. Original controls have no
fabricated seed experiment. Seed fits apply only to at most three Pareto
shortlist configurations per encoder × representation × algorithm.

Selection uses Pareto criteria without a weighted scalar score. Clearly
degenerate assignments and trivial stability intersections cannot justify a
final candidate; diagnostics such as dominant-cluster or high noise remain
visible rather than imposing an untested aggressive cutoff. The full front
is retained. Figure references maximize original silhouette only to bound
manual review, which selected tiny populations for the original controls.
That preference is not a final split recommendation. The observed broad front
and noise sensitivity require an explicit future partition/noise protocol,
residual-correlation measurements and review of groups of different sizes.
