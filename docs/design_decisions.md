# Design decisions

## 2026-10-08: scientific occurrence identity independent of ZIP packaging

External processed-frame ingestion separates exact image SHA256, named logical
observation, attributable scientific occurrence and physical ZIP-entry identity.
The scientific ID includes collection/series/index and image bytes, never shard
names/checksums/positions. The scientific manifest has no physical locators;
repackaging changes the source-bound artifact but preserves the tested scientific
manifest, dataset and variant identity. No native video-frame identity is inferred.

Repeated indices cannot establish distinct scientific occurrences without an
additional observation discriminator. Every physical entry survives; ambiguous
frame IDs stay null and the existing unique/non-null manifest contract is not
weakened. Such audits withhold the scientific manifest and `dataset_variant_v1`
declaration rather than invent IDs or choose representatives. Corrupt but readable
image bytes retain exact identity with failed decode QA. Candidate timing and
reported manual observations never establish verified alignment or detector
readiness. See the [identity recipes and limits](runbooks/video_variant_ingestion.md).

## 2026-10-05: normalized candidate edges in organization-evidence-v2

A stored labeled/video candidate is an edge between exact content identities,
not a generic group with redundant query/candidate memberships. The v2 producer
copies every authoritative candidate field into `candidate_pairs` once, preserving
IDs, float values, nullable ranks, flags and upstream counts without fitting,
filtering or clipping. Cores/components retain their existing membership semantics.
Consumers join each endpoint through contents to records, timelines and media.
Manifest-level literal semantics prohibit ground truth, automatic confirmation,
sequence identity, confirmed dependency and split constraint interpretations.

Only organization JSON serialization becomes deterministic compact UTF-8; checksum
receipts bind those exact bytes. Detector reports and scientific publications are
unchanged. This is an explicit schema version change, not a v1 semantic mutation;
existing v1 destinations require separate v2 destinations. The owner reports a
completed real v1 export on Hypatia; real v2 execution remains pending. See the
[organization runbook](runbooks/organization_evidence_export.md).

## 2026-10-05: occurrence-scoped compatibility for sequence presentation

Only organization export opts into tolerating manifest serialization drift for
stored structure/recurrence evidence. Dataset identity, immutable artifact
inspection and every existing frame/content/timeline/position check remain
mandatory. The default adapters and scientific sequence source checks remain
byte-strict. Sequence provenance claims exact occurrence binding, never full
manifest table equality; linkage retains its separate stronger table assertion.
The export records both hashes and the verification scope without modifying
scientific sources. See the [organization runbook](runbooks/organization_evidence_export.md).

## 2026-10-05: explicit presentation compatibility for manifest serialization

Organization export may opt into accepting different labeled-manifest Parquet
bytes only when the existing exact full DataFrame comparison (including dtypes)
against linkage's checksum-bound occurrence snapshot and the declared dataset
identity both pass. The shared candidate loader remains byte-strict by default;
manual review signatures and source verification remain unchanged. Organization
source receipts distinguish historical/current hashes and declare exact tabular
verification and reserialization explicitly. No source is rewritten and no
replacement manifest is produced. See the [organization runbook](runbooks/organization_evidence_export.md).

## 2026-10-05: preserved frozen membership is a separate evidence authority

A missing clustering publication may be represented only by the exact stored
membership of every selected split referencing its identity. Equality includes
content, cluster, group and group type after ordering only. Existing split/freeze
checks remain mandatory; a present invalid clustering publication fails closed.
The original identity is retained without inventing a clustering receipt,
configuration or diagnostics. Noise remains unassigned and singleton in splits.
This is presentation provenance, not a new fit or a change to clustering semantics.

## 2026-10-04: occurrence-preserving organization export

The frozen detector plan is the selection authority for final M02 splits and
their source clustering, rather than candidate display aliases or hardcoded
counts. Normalized files preserve record-level historical cross-partition
duplicates and unique-content clustering independently. Counts/ranges summarize
saved memberships; no detector or clustering metric is recalculated.

Existing top-k links retain candidate-pair IDs; existing diagnostic components
join their stored core/content/occurrence evidence. No connected components,
interval memberships or sequence instances are constructed during export.
Filename-derived families are explicitly heuristic timelines with null verified
video IDs. Ambiguous time/annotation consensus remains null. Media is a separate,
optional ignored publication with one preview per content and graceful absence.
Only code, tests, schema and documentation are versioned by this local change.

## 2026-10-03: final-only detector contract and split association unit

The final report extends the existing evidence gate and hierarchical aggregation;
it does not replace the notebook's partial-progress report. Publication requires
every cell in the supplied plan, not a hardcoded experiment identity or count.
Detector seeds are averaged within split before correlations and strategy means.
Single-split SD is undefined, and image-bootstrap intervals are retained per run
rather than averaged. Correlations are descriptive and never leakage effect sizes.

Pydantic models generate a stable JSON Schema and validate the producer bundle;
independent JSON Schema validation is a dev test dependency. Only the lightweight
contract is intended for versioning. Sources and detailed analysis remain ignored.
Two staged destinations are validated before publication, with rollback for normal
I/O failures and an explicit single-writer requirement. This local implementation
does not certify the completed Hypatia experiment reported by the owner.

## 2026-09-28: operational scheduling preserves the experiment

SLURM parallelizes encoders, individual representation configurations/seeds and
algorithm grids. Reduced coordinates are published once and reused without a
second stochastic fit; direct recurrence is reused for cluster support. Scientific
IDs/assignments do not include job IDs, resources or paths. The operational plan
separately binds source checksums, resolved stores, code/dependencies and jobs.
Completed stage receipts require valid immutable publications. Corruption, orphaned
outputs, uncertain submissions and changed sources are never repaired by deleting
or silently recomputing them. Recovery emits a deterministic inspection/resubmission
plan. Manual review remains an explicit guard, with no confirmation-dependent
scientific consumer or split creation in this phase.

## 2026-09-28: separate temporal and visual experimental evidence

The sequence experiment extension keeps source timeline instances, conservative
cores, reviewed uncertainty zones, algorithmic clusters and visual dependency
groups distinct. A new subpackage reuses existing numerical adapters, while the
v1 sequence commitment and density protocol remain unchanged. Agglomerative is
an experimental literature comparison. Fitting uses unique visual contents;
all occurrences and independent encoder mappings survive downstream.

Post-hoc exact-label metrics require unanimous, resolved occurrence membership
and report N/coverage under explicit noise policies. Recurrence combines ranks
or evidence rules, never uncalibrated encoder cosines. Candidate graphs cannot
create VDGs. Incomplete boundary review cannot supply false-positive counts or
candidate precision. Imported schemas are explicit and checksum-bound rather
than inferred from legacy filenames. Manual review imports have immutable,
replayable history; incompatible decisions require external adjudication.

The current detector is retained as an evidence control. Literature-inspired
PaCMAP experiments use the existing 2/3-D protocol, not the 256-D experiment in
Glazner et al.; this is documented as a limitation, not an exact replication.
No new dependencies or final split are introduced. See
[sequence experiment protocol](protocols/sequence_experiments.md).

These decisions describe the current implementation and experimental protocols.
Later sections record the executed similarity/reduction extensions.
Clustering and splitting have executed experiments. Detector infrastructure and
four small CPU pilots are validated; the final controlled detector experiment
remains pending. Earlier dated decisions retain their original scope.

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
extraction models are DINOv2-small (384D) and CLIP ViT-B/32 (512D), validated on all
1459 contents. Existing research configs nominate larger candidates and remain
unexecuted; they do not supersede this bounded model choice.

## 8. Conditional Bhattacharyya and committed reductions

Bhattacharyya distance requires an explicitly defined distributional representation.
No such representation has been selected; implementation is deferred. The primary protocol
uses t-SNE and PaCMAP. UMAP is outside that experiment and its unused
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
adapters. Full extraction validation also covered real N=16 loads and complete extractions
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

Full coverage verification additionally compares record/content mappings to the canonical
manifest and requires resolved revision metadata. Equal numerical embeddings for
different contents are permitted. Full numerical validity does not establish
semantic quality, clustering structure or detector improvement.

## 13. Class nomenclature and instance geometry

The original YAML explicitly maps IDs 0–4 to vehicle/building/road/river/SDZI.
Canonical display names follow the publication order confirmed by the project
owner on 2026-09-13; `SDZI` remains the original label for Heavy Machinery (4).
This is order correspondence, not a demonstrated expansion of the source term.
The exact bibliography remains pending; see [evidence and limits](analysis/dataset_classes.md).
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

Cosine analysis uses existing float32 L2 vectors without silent normalization, clipping or
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

The [protocol](protocols/reduction.md) fixes three configurations per method and
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

Clustering executes the [density protocol](protocols/clustering.md) on unique
contents: original DINOv2/CLIP L2 controls and the four selected 2D reductions.
Original controls are ablations, preserving t-SNE/PaCMAP in the primary reduction path.
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

## 20. Atomic record-weighted partition construction

The subsequent [splitting protocol](protocols/splitting.md) preserves every exact
content and each selected nonnegative cluster as an indivisible unit. Noise keeps
cluster_id=-1 with distinct singleton group IDs. Historical membership supplies
aggregate target ratios only; all new memberships are independently constructed.
Labels enter group balance here, after clustering, without changing memberships.
Occurrence-level counts preserve eight observed annotation-conflict groups.

SciPy MILP is already in the stack. Integer allocation of groups with identical
balance profiles is an exact symmetry reduction, followed by seeded expansion.
Normalized L1 balances record, class and empty-label families. The solver never
receives cosine, sequence or temporal proximity. A deterministic node budget and
recorded gap distinguish optimal results from feasible incumbents; direct primal
checks handle HiGHS stopping codes not recognized by the installed SciPy wrapper.
The five-seed content-permutation baseline balances sizes but not classes, an
explicit confound for attributing downstream differences solely to grouping.

## 21. Residual evaluation and robust candidate selection

Referencias: **C10 — DINOv2 / PaCMAP / DBSCAN**; **C01 — CLIP / original L2 / OPTICS**.

Original matrices and neighbor tables of both encoders evaluate every partition,
irrespective of its clustering encoder. Unique-content NN excludes self, even for
historical contents with multiple memberships. Exact overlap is separate, while
non-exact pairs retain the existing existential cross-membership definition.
Complete matrices avoid truncating cross-split NN searches to stored top-20.
Encoder-specific quantile cohorts preserve inclusive thresholds and ties.

The preliminary 12-candidate rule balances coverage of encoder, representation,
algorithm, noise, cluster counts, ARI/AMI and coherence, without silhouette-only
selection. Final constraints require every seed to preserve identities, cover
all five classes and respect a declared 10% relative record-size tolerance.
Pareto uses worst-seed losses plus variability; representative seed 0 is fixed
in advance. Both full fronts and excluded configurations remain inspectable.

The observed C10 anchor improves visual residuals under both encoders but not
all temporal residuals against historical. C01 is a class-balance anchor with
97.4% noise and no uniform residual advantage. These compromises are reported,
not converted into a universal winner or a claim of eliminated leakage.
Using two encoders is a cross-representation diagnostic, not an independent
held-out dataset. Detector generalization and real temporal validation remain
separate hypotheses and pending experimental stages.

## 31. Detector controls and a measured laptop budget

Referencias: **C10 — DINOv2 / PaCMAP / DBSCAN**; **C12 — DINOv2 / t-SNE / HDBSCAN**; **C01 — CLIP / original L2 / OPTICS**.

The detector candidate review keeps C10 primary and selects C12 through explicit
cross-seed constraints, before YOLO. C01 remains a descriptive balance ablation.
Original occurrence labels, including conflicts, survive materialization. Raw
ZIPs are read-only; generated image lists reuse an ignored occurrence store.

YOLO11n is a single fixed architecture; no per-strategy tuning. Fixed-confidence
P/R is separated from the upstream test-best-F1 display, and image bootstrap
recomputes pooled AP. The primary endpoint is five-class macro mAP50-95. Runtime
weights/configuration are frozen before real-data training, after measuring
device, memory and feasible batch; CPU is not silently treated as CUDA.

Four tiny real CPU pilots validate infrastructure only. Automatic Stage B on CPU
is disabled: extrapolations of roughly 408–593 training hours for 48 runs are
disproportionate on this laptop. No final detector outcome, causal leakage
effect, seed variability or model ranking is inferred from those pilots.
See the detector protocol, runbook and pilot analysis for the executed scope.

## 32. Inspection is a read-only presentation layer

The local Streamlit explorer consumes completed run metadata and immutable
assignments, never fitting or split optimization routines. Joins preserve content
identity and every historical occurrence. A cluster-aware overlay must bind to
its source clustering; noise stays -1 with separate singleton group IDs.
Filename consensus controls playback ordering. Distinct sequences and unknown
provenance cannot be silently concatenated. Gap cuts and bounded GIF pages are
display segments only; requested playback FPS is not a dataset property.

The consolidated project report renders saved median/Q1/Q3 per pre-existing inferred-index
bin. All seven positive-gap bins have support (1438–398800 pairs per bin in the
current artifacts); no bin merging or trend-driven boundaries were needed.
The empty zero-gap bin has count zero and is omitted; future populated zero gaps
remain visible. Y axes are independent, and IQR describes pair dispersion,
not a confidence interval. Original hexbins and scientific artifacts are retained.
This iteration does not change candidates, metrics or experimental completion.

## 33. Source-video sampling precedes sequence identification

`data extract-video-frames` samples the first video stream with external
FFmpeg/ffprobe, rebasing its first PTS to zero before a fixed FPS grid. Nominal
sample time and a rounded FPS-based source-index estimate are explicit derived
coordinates, not capture times or decoder indices. Missing reported properties
stay null. Source-path IDs identify videos; a separate SHA256 binds their bytes.

This preparation stage has synthetic tests and confirmed real operational
validation on Hypatia: a 5-second smoke and completed job 737719, producing
9648 JPEGs from three source videos at 1 FPS. These three sources are not
assumed to be three indivisible sequences. Subsequent
visual/temporal analysis must identify the sequence units that future splits
will preserve. Sampling creates neither those groups nor train/val/test and
does not modify the historical experiment. Integration with content-level
features now uses the separate occurrence/content manifest described below;
repeated samples must not artificially increase density during clustering.

Staging protects prior outputs against decoder failures. A completed summary
and checksum-bound frame table authorize only named generated replacements;
unmanaged files are preserved. A publication marker blocks reuse after an
interruption during the non-atomic final promotion. See the
[data runbook](runbooks/data.md#evidencia-real-confirmada-en-hypatia) for confirmed
execution evidence, and its surrounding sections for the single-writer boundary
and temporal definitions. Operational extraction validation does not validate
scene boundaries, embeddings, clustering, splits, leakage removal or detector
performance for these video samples.

## 34. Video samples reuse the feature pipeline without changing historical identity

`flir_video_samples_v1` namespaces the unlabeled dataset. The portable occurrence
hash includes version, video ID, source-video SHA256, exact float sampling rate,
sample index and JPEG SHA256 (the precise serialization is in [data model](data_model.md)).
Adding rate and source hash prevents an occurrence from silently acquiring different
grid/source provenance when a path-based video ID is reused. Exact JPEG bytes
remain the content unit; the unchanged dataset-ID algorithm accepts the existing
empty-string convention for absent labels. No synthetic label, split or sequence
is assigned. Source-video hashes are inherited from the validated sampling receipt,
not independently recomputed from original videos by this command.

The manifest preserves corrupt image occurrences and reports decode failures;
features refuses them rather than dropping rows or inventing vectors. Missing
images, unsafe paths, duplicate temporal identities, unrecognized receipts and
inconsistent counts/grids are hard errors before publication. Outputs must be
outside the read-only image root. JSON report checksums bind both sampling
metadata inputs and the resulting manifest. Manifest/report promotion is not a
multi-file transaction; validate their checksum correspondence after interruption.

A small image-source adapter replaces the ZIP-specific read inside storage.
Every local declared occurrence is hash-checked before extraction, reuse or resume,
including redundant copies and unselected smoke contents. This costs a full JPEG
read pass but prevents stale content identities from hiding changed duplicate
files. ZIPs keep representative-only access and share the selected-byte checks.
The existing cache-signature fields stay compatible: expected hashes are now
checked against source bytes. Paths and transport do not alter `feature_space_id`.
Old ZIP caches are not migrated; additional source metadata describes new stores.

Use immutable sources and one writer per output. Filesystem containment is checked
at read time, but this is not a lock against concurrent mutation; neither sampling
nor JPEG replacement should run while building/extracting. Existing resume and
metadata-last completion semantics remain. This occurrence/content-to-features
bridge originally had synthetic/offline validation only. The project owner now
reports full Hypatia manifest/features validation; see the separately attributed
evidence in [status](status.md). This change does not revalidate those files. Directory-based
diagnostics/reports and video sequence analysis remain
outside this bridge.

## 35. Explicit video provenance and bounded exact similarity v2

Historical `content_cosine_v1` retains its serialized configuration/ID, artifact
contract, filename consensus, `same_sequence`, `frame_delta` and historical split
semantics. New default fields are excluded from its identity serialization. Its
full pair table is still supported. Deterministic top-k sorting now uses 64-row
blocks with the same score ordering and content-ID tie break.

Video configs explicitly select `content_cosine_v2` / `sampled_video_grid`.
`source video != sequence`; `relative sampling-grid timestamp != capture timestamp`.
The data audit validates declared grid consistency and preserves every occurrence;
receipt/source verification remains the responsibility of build-video-manifest.
There is no inferred sequence ID, label, historical split or representative time.
Content summaries retain video membership and associated occurrence tuples; the
record table retains exact numeric fields and every frame/content/embedding link.

Only the existing L2 content vectors enter cosine/top-k. Posterior video relations
compute independent minima over **all** occurrence pairs sharing a source video.
`same_source_video` means intersecting source sets, including contents present in
several videos. Disjoint sets have null sample/time gaps. Sample units and seconds
have separate bins; FPS is never assumed to be 1. Different source rates can make
the two minima refer to different occurrence pairs. These relations assert neither
scene identity nor confirmed leakage.

V2 uses 21 bytes per unordered pair in compact NumPy arrays plus temporary exact
float64 statistics/quantile workspaces. It never constructs an all-pairs DataFrame
or full triangular row-index arrays. Near-unit pairs and optional full pairs stream
to typed Parquet by matrix row, including the degenerate all-near-unit case.
Default `pair_storage: summary_only_v1` omits only `pair_analysis.parquet`;
`full_streamed_v1` opts in with a different similarity ID. Matrix, complete top-k,
provenance, global and temporal distributions, six exact linear quantiles and all
near-unit pairs remain available. Inclusive quantile thresholds retain ties.

Version, temporal rules, gap bounds/units and pair-storage policy enter similarity
identity. Input fingerprints bind video fields independently of unchanged dataset
and feature identities. Standalone verification reconstructs provenance, neighbors,
compact relations, summaries and persisted pair cohorts; optional sources also bind
original embeddings and manifest. The metadata completion marker, refusal of
incomplete directories and one-writer-per-output rule remain unchanged.

The v2 verifier enforces pair schemas even for empty cohorts. With supplied
features it compares the full mathematical snapshot, exact content index and
occurrence mapping to their sources and validates raw/L2 quality; with a manifest
it also checks complete coverage and resolved revision. Cache reuse from compute
performs that source-bound verification too; matching metadata/checksums alone
cannot accept a cache with a different pooling declaration or content mapping.
Grid auditing rejects
negative relative times and labels/splits/group declarations incompatible with
the unlabeled video-v1 contract. Quantile cohort comparison explicitly uses a
float64 ufunc loop, independent of NumPy scalar-promotion rules.

Reduction still consumes matrix/top-k/metadata, with complete artifact QA. Clustering
can fit/evaluate visually, but sequence recall/coherence and historical split metrics
are explicitly unavailable for video. Its posterior evaluation protocol is v2;
existing Pareto selection records missing temporal criteria as omitted. No source
video is promoted to a cluster. Historical ZIP/sequence HTML builders reject video
early; video summaries and CSVs remain available. Video splitting is explicitly
out of scope until a separate grouping/annotation protocol exists.

Video clustering metadata enumerates unavailable metric groups and explicitly
marks the sequence-temporal denominator unavailable. Verification rejects replacing
these unknown metrics by zero, even if the metric-file checksum is updated.
Historical v1 metadata is not required to carry these video-only declarations.

Local validation uses synthetic inputs only. The target N=8093 workload and its
peak memory/time must be measured later on Hypatia; similarity, scene discovery,
reduction and clustering results on those real contents are not claimed here.

## 36. Scoped exact clustering distances preserve scientific identity

Clustering families retain verified vectors, coordinates and source signatures;
Euclidean matrices materialize only when requested. Scopes reuse at most the
original evaluation matrix and one active representation, releasing both between
screening representations, comparison runs and source-bound verification runs.
The original float64 SciPy kernel is unchanged. Exact row-block k-distance avoids
full partition copies without changing epsilon or tie handling. This is an
operational change, so no protocol/config/identity version is advanced.

A pre-change synthetic oracle from `737e588` binds labels, metrics, medoids, IDs
and selection. Historical cosine v1 and video v2 unavailable-sequence semantics
remain separate; reduction artifacts require no migration. Recomputing after a
scope ends trades runtime for bounded retention. Remaining quadratic algorithm,
metric and source-verification structures, N=8093 estimates and validation limits
are recorded in the [memory audit](analysis/clustering_memory.md).

## 37. Reviewed occurrence sequences are separate from exploratory clusters

`sequences` uses complete contiguous 1 FPS occurrence grids and original CLIP /
DINOv2 L2 stores. Exact copies remain separate timeline occurrences; each encoder
is mapped through its own record_index. Float64 prefix sums, complete windows,
12-decimal rounding before average-tie per-video ranks and multiscale consensus
define candidates. Neighbor midpoint constraints give disjoint F3 search regions.
Neither a high-confidence flag nor a cluster supplies manual acceptance.

The third coarse representative tie-break is P1: the minimum of independently
ranked CLIP/DINOv2 w=1 percentiles over all valid cuts per video, with the same
12-decimal rounding and average ties as the other scales. Raw encoder magnitudes
are not commensurate and their minimum cannot define consensus. P1 only breaks
ties after S and median(P5,P10,P20); final localization remains F3. The explicit
tie-policy literal enters identity, so artifacts using the former raw rule must
be regenerated. Confirmed manual CSV schemas and acceptance rules are unchanged.

Candidate publication and sequence commitment are separate CLI operations. Build
requires the explicit confirmed manual CSV contract, preserves accept/reject and
boundary types, and always records ground_truth=false. The inherited review has
no source-feature binding: compatibility is checked against recomputed candidate
positions, windows and scores, and the new identity binds the three review files
plus the actual manifest and feature checksums. Provisional-source checksums in
that review remain declared provenance, not independently verified evidence.

Every accepted t ends one interval at t-1 and starts the next at t. Sequence IDs
hash the source-bound set and inclusive interval, independently of source video
identity. Exact-copy dependencies use a deterministic star per shared content,
which has the same components as all pairwise edges while remaining linear in
membership. Full supporting occurrences remain available. Components, including
singletons, are future split constraints, not visual groups or actual splits.

Formal verification reconstructs candidates, review consumption, intervals and
dependencies from sources, and independently checks coverage and graph components.
It rejects value tampering even when file checksums are updated. Summary reads
metadata only and explicitly makes no quality claim. Real Hypatia execution and
scientific boundary assessment remain pending; see the [runbook](runbooks/sequences.md).

## 38. Cross-dataset visual linkage preserves candidate and occurrence ambiguity

Different dataset IDs are expected for labeled and sampled-video contents. Pair
each encoder by its mathematical feature configuration, resolved revision,
feature-space ID and dimension; verify complete manifest coverage independently.
Each store's content index defines its own row mapping. Neither historical split
nor annotation metadata enters the ranking kernel.

Rank the original float32 L2 vectors with a fixed float64 dot accumulation and
exact ties ordered by video_content_id. No score rounding, renormalization or
cross-encoder raw-score averaging occurs. For each labeled content, preserve the
union of CLIP and DINOv2 top-k, with separate scores and nullable ranks. Mean
reciprocal rank gives zero to a missing rank and divides by two; it is a scale-free
diagnostic, without an acceptance threshold or match probability. If k exceeds
the video population, retain all contents and record the effective k.

One bounded query block is scored against all video contents, one encoder at a
time. The retained union is rescored for the missing encoder's score. Memory for
cross-scores scales with block size times video population, plus retained top-k
candidates and the input vectors. No approximate index or reduced space is used.

The normalized occurrence relation preserves every source video occurrence and
every reviewed sequence_id. A sequence is an existing continuous segment;
linkage assigns no sequence to labeled content and infers no visual dependency
group. The entire labeled manifest is snapshotted to retain duplicate frame_ids,
original_split and conflicting annotation metadata. Review and future split
constraints remain separate stages.

Immutable publications bind every source file, both datasets, paired spaces,
sequence set and versioned policy. Verification reconstructs candidates, scores,
ranks and normalized lineage, detecting ambiguity loss despite rehashed outputs.
Sequence consumption checks stored review/partition consistency and source
binding; full external manual-review and F3 verification remains `sequences verify`.
`summary` reads JSON counts only and makes no verification claim. Local evidence
is synthetic; the owner reports real consumption compatibility through manual
review init/verify on Hypatia. Global visual assessment remains unestablished.

## 39. Manual calibration records group-level evidence without promoting linkage

The calibration sample proposes one group per labeled query. Every candidate
occurrence and alternative group membership remains in the evidence, with
same-video temporal context that may cross sequence boundaries. A display-only
labeled occurrence never replaces its full annotation/historical lineage.
Visual dependency groups are consumed as existing must-link constraints; the
review does not merge sequences or combine constraints into a new split.

Initialization writes blank decisions. Only explicit imported manual CSV values
can record supported/ambiguous/unsupported or clear a decision to blank. Each
revision preserves its imports, before/after values, reviewer, source and UTC
timestamp. Verification replays those events from an immutable blank baseline.
Source/renderer hashes and deterministic IDs bind evidence and decisions; original
linkage publications remain read-only. Source compatibility uses an explicit
normalized external-membership adapter and an explicit confirmed Hypatia v1 branch.
The latter requires confirmed_manual_visual_dependency_validation/version 1 and
all declared must-link/no-merge/no-split/exact-dependency-preservation semantics.
That producer has no artifact_id or output_checksums: a named consumer source
fingerprint binds SHA256 of metadata, the groups CSV and the explicitly selected
membership table plus producer kind/version and sequence_set_id. No producer
fields are fabricated and the external files remain read-only. Normalized
producers still require their own ID and declared checksums. Verification
rehashes these sources and checks stored PNG evidence; it does not rerender
contact sheets from the original images.

Summary denominators are sampled queries, including blank decisions, within
each stratum/group. Blank and ambiguous cases are unresolved. These are manual
calibration statistics, not representative accuracy. Supported evidence never
becomes an automatic confirmed link or an exact frame/sequence claim. Metadata
always states ground_truth=false, confirmed_matches_created=false and
split_created=false. See the [review runbook](runbooks/linkage_review.md).

The owner subsequently reported successful real Hypatia v1 review init and
quality_valid/source_bound=true from verify, with all candidate occurrences
preserved, one decision per query, and no ground truth, confirmed matches or
split. This is exercised adapter compatibility, not completion of all future
calibration strata or a local revalidation of real artifacts.

## 40. Review aggregation preserves decisions and sample-specific evidence

Aggregates consume only immutable review publications that pass full source-bound
review verification. Their identity binds SHA256 of every constituent file,
including metadata, evidence images and imports. Original revision/calibration
IDs, source fingerprints, notes and event histories remain inspectable; operational
source locators do not define scientific identity. Aggregate verification repeats
constituent verification and reconstructs tables/history/counts, rather than
trusting summaries or checksums of derived values alone.

The unit is a labeled query/proposed group pair. Duplicate decisions must agree
literally, including blank, and share the same source-bound evidence/protocol
apart from sampling CSV/stratum. Conflicts fail without voting or selecting a
newer revision. Compatible duplicates remain in a separate observation/duplicate
report but count once within each cell. All groups share a frozen sequence and
membership domain so identical group strings cannot mask different producers.

Each source revision keeps its own descriptive summary. Global pooled counts
are unique pairs, including unresolved blanks and ambiguous decisions; original
stratum membership is retained, so overlapping cells are not additive. Different
samples are not assumed exchangeable, rates are not representative accuracy or
precision, and neither supported evidence nor encoder ranks create truth. No
sequence, linkage or split assignment is produced. Aggregation is validated with
synthetic inputs; its real execution and future calibration coverage remain open.

## Generic dataset variants and explicit pairing

Variant identity includes dataset identity, name, parent identity and declared
preparation. Source checksums bind immutable snapshots separately. Feature-space
identity describes the mathematical encoder and remains independent of the image
collection; storage and experiment identity include the dataset variant. Existing
undeclared stores are never implicitly relabeled from a CLI selector.

Variant comparison preserves all occurrences. Paired frames express external
correspondence, not byte identity; neither nominal time nor equal content hashes
automatically establishes a pair. ARI/AMI uses unique bijective content relations
under matching fit configurations, with feature-space differences, ambiguous relations and
unpaired coverage reported. Boundary overlap refers to mapped zone membership;
recurrence pairing requires complete mapped core occurrence sets. Descriptive
metrics retain their own masks/denominators and imply no causal effect or winner.
HUD removal is only a future application of this generic infrastructure, with no
real no-HUD experiment or scientific conclusion yet.

## Native legacy evidence without promotion

The owner's observed Hypatia schemas are versioned input contracts. Native
ingestion does not force them into an envelope requiring an invented reviewer,
review timestamp or dataset declaration. Canonical identity is established by
an explicit frame/content/metadata join; legacy variant identity remains
`unspecified`. Revision counts validate these existing artifacts only and never
parameterize future fits or expected results.

All consumed JSON/CSV/Parquet bytes are checksummed and frozen. Verification
requires unchanged originals and replays the normalized tables. Boundary zones
remain uncertain; `sequence_core_candidate` remains a candidate with legacy
origin. Encoder ranks and candidate pairs are preserved, with no confirmed
dependencies or components promoted to VDGs. The legacy assistant review mode
is retained verbatim and is not independent visual ground truth.
