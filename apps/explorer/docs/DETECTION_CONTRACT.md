# Verified detector presentation contract

Source: [flir-leakage-pipeline, publication 0a4ea1f](https://github.com/Kazzu00/flir-leakage-pipeline/tree/0a4ea1f/exports/frontend/detection), branch `feat/hypatia-detector-final-report`. Contract: `detection-export-v1`.

The checked-in manifest reports `COMPLETE_CONTROLLED_COMPARISON`, `scientific_result=true`, `generated_from_verified_artifacts=true`, 48/48 runs, 16 splits, four strategies and three detector seeds. The source reports completion and verification of the Hypatia controlled descriptive comparison. This frontend validates the exported contract; it does not independently repeat training or scientific verification.

Publication commit `0a4ea1f` published the JSON. Generator `source_commit=18d61b35e4c82f41b0c9092e997fb9088f7a10f5` identifies the code that generated it. Plan `a05366b5c2d72834` and model configuration `73416cdcfc4cdff3` are preserved. `snapshot-provenance.json` records publication separately from the unmodified manifest.

## Storage and privacy boundary

The user explicitly authorized versioning this public lightweight contract. Eleven byte-preserved files live in `src/features/detection/snapshot/`: manifest, summary, strategies, splits, runs, classes, support, variance, associations, bootstrap and `schema/detection-export-v1.schema.json`. This is the sole exception to the repository's no-real-artifacts policy. `.gitattributes` disables line-ending conversion for these files; Prettier excludes them so upstream SHA-256 receipts stay valid.

Vite imports this allowlisted directory as asset URLs with `?url&no-inline`. Production gets hash-named JSON assets, independently of any mounted runtime directory. Docker's existing `COPY src` includes this approved contract. No broad `.gitignore` exception, runtime directory copy or Docker mount change is needed. `build.copyPublicDir: false` is retained. General snapshots in `public/runtime/`, `.references`, private data and source artifacts remain excluded. There are no images, weights, archives, credentials or absolute upstream paths in the imported contract.

## Loading and validation

`adapter.ts` reads unknown JSON only after file retrieval, checks each payload/schema byte digest against `manifest.file_sha256`, then validates the complete bundle with strict Zod schemas. `schema.ts` mirrors the exported schema and adds cross-file integrity checks: declared counts, unique run/split/class/association identities, seed membership, support references, protocol IDs, summary consistency, bounds, and null single-split variance. These are integrity checks, not scientific aggregation. The upstream JSON schema is preserved and checksum-checked for traceability; Zod is the executable runtime validator.

`useDetectionSnapshot` owns one deterministic TanStack Query key and a single shared bundle request per session. Files load in parallel; shell and evaluation reuse the same cache. Non-complete scientific state fails closed. Missing files show “Verified detector results are not available.” Invalid contracts show “Detector result contract failed validation.” Technical details are expandable. No demo or stale synthetic detector fallback exists.

Detector evaluation bypasses the unrelated workspace snapshot gate. Invalid/missing workspace artifacts therefore cannot suppress an independently valid detector contract. Other M02 views still enforce their original gate. The detector page hides the unrelated DEMO/MIXED selector and displays REAL / VERIFIED only after successful validation. M01/M03 remain DEMO.

## Presentation and scientific limits

`/organization/evaluation` defaults to the detector comparison. `?view=splits` preserves the separate grouping/split view. All eight legacy M02 redirects remain intact, including `/organization/detector` and `/organization/splits`. No new top-level route was introduced.

- Comparison: split-level means, exported min/max range and strategy means, with exact tables as keyboard alternatives. Symbols and labels complement color. Historical has only one split.
- Variability: exported within-split detector-seed SD, between-split SD and their exported ratio. Single-split variance is undefined, never zero. SD is displayed in percentage points; ratios are not variance ratios or causal decompositions.
- Classes: compact metric matrix and direct link to test composition. No averaging in the UI.
- Test composition: exact exported support per split/class, with no aggregation or normalization.
- Associations: seven pre-specified choices. Each point is an exported split observation. Global Pearson/Spearman and within-strategy centered Pearson are read verbatim; the frontend does not center points, fit a line, calculate correlations, or treat detector seeds as independent splits.
- Traceability: provenance, interpretation/limitations and 48 individual runs are behind accessible drawers. Filters select existing rows. Bootstrap intervals are selected by run and shown with the exported method, confidence level and image-independence caveat.

The main metric ordering describes reported detector performance only, not methodological quality. Different strategies change test composition, class distribution, membership and visual difficulty together. Clusters are not ground-truth sequences, residual dependence can remain, and temporal_at5 is an index-distance proxy (Δ≤5), not seconds. Source interpretation and limitations remain available verbatim in Spanish. The frontend performs only selection, sorting, formatting and chart arrangement.

Fields absent upstream: no per-point centered coordinates, no strategy-level bootstrap intervals, no per-run epoch curve, and no dataset/feature IDs in this detector contract. None are reconstructed. The detector contract stays separate from the sampled-video snapshot; no unvalidated dataset join is attempted. The UI labels support generically as exported class support; the contract does not explicitly encode its unit in a separate metadata field.

The obsolete pending-detector page was removed. The null-metric DEMO protocol entry remains isolated in the demo experiment registry, explicitly described as a placeholder; it does not feed verified evaluation.

## Deterministic update

Regenerate and verify the frontend contract **upstream**, publish that export there, then import it here. The upstream repository is read-only to this command. Node 24 and existing locked npm dependencies are sufficient.

```powershell
npm run sync:detection -- --repo <path-to-flir-leakage-pipeline>
# Or use FLIR_PIPELINE_REPO instead of --repo.
```

The command reads only `exports/frontend/detection/`'s eleven named files, validates the entire bundle before writes, preserves bytes, and writes publication provenance from the upstream branch/HEAD. Do not use an unpublished or dirty export with automatic HEAD provenance. For an already-downloaded published contract:

```powershell
npm run sync:detection -- --export-dir <contract-directory> --branch feat/hypatia-detector-final-report --publication-commit 0a4ea1f
```

Explicit provenance arguments identify the downloaded publication and must be supplied accurately. No timestamps are added by the importer, so repeated imports of the same source/provenance are byte-identical. Unexpected source files are never read or copied. Network access is not required by sync or tests. Checksums bind bundle contents; they are not signatures establishing publisher authenticity. Importing and deploying a new contract requires rebuilding the static app. If a filesystem write is interrupted, rerun sync; runtime validation rejects a partially updated bundle.

After updating: run lint, typecheck, unit tests, build, E2E and format checks; review the diff and provenance. Do not overwrite or recalculate scientific fields locally to satisfy the frontend.
