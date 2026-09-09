# Design decisions

These decisions describe the current implementation and the supplied thesis
methodology. No similarity, reduction, clustering, new splitting or training was
implemented during this review.

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

Report inputs are explicit or unambiguous. The builder's `--full` mode filters
by canonical dataset and complete coverage, verifies provenance, requires both
encoders, and still rejects multiple eligible runs. Hash ordering never chooses an
experiment. Class-presence charts count records, not objects; box-area charts use
per-record means. Sample sizes and embedding health come from selected artifacts.
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
