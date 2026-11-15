# Detector integration validation · 2026-10-04

Branch: `feat/real-detector-results-ui`, created from the clean `feat/ux-v2-sequences-docker` state. No upstream repository was modified, no scientific processing was executed, and no push, merge or deployment was performed.

## Implemented result

Evaluation now defaults to the REAL / VERIFIED detector contract, with explicit missing/invalid/loading states and no demo fallback. Source verification is reported upstream; frontend validation concerns contract integrity. Existing M02 routes and all legacy redirects remain. The grouping/split view is still separate at `?view=splits`.

Source: `flir-leakage-pipeline`, branch `feat/hypatia-detector-final-report`, publication `0a4ea1f`; generator `18d61b35e4c82f41b0c9092e997fb9088f7a10f5`. Eleven source JSON files total 349,672 bytes in `src/features/detection/snapshot/`. All ten manifest-listed SHA-256 digests match. Production JSON assets were compared byte-for-byte with the approved source contract; `dist/runtime/` is absent. Source contract is sufficient for offline tests and frontend execution.

Update: `npm run sync:detection -- --repo <pipeline-repo>` or `FLIR_PIPELINE_REPO`; explicit downloaded-export import also accepts `--export-dir`, `--branch`, `--publication-commit`. [Full procedure and limits](DETECTION_CONTRACT.md).

## File inventory

Created:

- `src/features/detection/schema.ts`, `adapter.ts`, `query.ts`: strict contract, hash/reference integrity, shared TanStack Query.
- `src/features/detection/presentation.ts`, `route.ts`: presentation-safe chart arrangement, labels/formatting and route classification.
- `src/features/detection/DetectionEvaluation.tsx`: state handling and presentation hierarchy.
- `src/features/detection/DetectionSections.tsx`: strategy split/mean plot, variability table, class matrix, support matrix with common visual scale, association scatter and exact tables.
- `src/features/detection/DetectionDetails.tsx`: provenance, source limitations, run filters/table and exported bootstrap intervals.
- `src/features/detection/snapshot/`: eleven approved upstream JSON files; `snapshot-provenance.json` separately records publication metadata.
- `tools/sync-detection.ts`, `.gitattributes`: allowlisted byte-preserving update and checkout behavior.
- `src/test/detection-fixture.ts`, `src/test/detection.test.tsx`, `e2e/detection.spec.ts`.
- `docs/DETECTION_CONTRACT.md`, this validation record.

Modified:

- `src/app/App.tsx`, `src/components/layout/AppShell.tsx`, `src/data/provider.tsx`: independent detector gate/cache and explicit mode; no unrelated workspace requests on the detector route.
- `src/features/organization/EvaluationOverview.tsx`: detector default; compatible split view retained.
- `src/features/overview/Evaluation.tsx`: detector status comes from the validated query.
- `src/data/mock/fixture.ts`: null-metric DEMO protocol remains isolated and explicitly labeled; outdated experiment-pending wording removed.
- `src/styles/global.css`: scoped layout, disclosure, matrices and accessible tab contrast.
- `e2e/real-data.spec.ts`: selects grouping/split explicitly after the intentional default-view change.
- `package.json`, `.prettierignore`, `AGENTS.md`, `README.md`, `docs/ARCHITECTURE.md`, `docs/RESEARCH.md`, `docs/INFORMATION_ARCHITECTURE.md`.

Deleted: obsolete `src/features/organization/detector/Detector.tsx`, replaced by the verified contract presentation. It was classified OBSOLETE; no partial experiment artifact was deleted.

## Executed verification

| Command / check                                                                                                                         | Observed result                                                                        |
| --------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| `npm run sync:detection -- --export-dir .tmp/detection-source --branch feat/hypatia-detector-final-report --publication-commit 0a4ea1f` | Eleven files imported; 48/48 declared runs; hashes and schema valid                    |
| `npm run lint`                                                                                                                          | Pass                                                                                   |
| `npm run typecheck`                                                                                                                     | Pass                                                                                   |
| `npm run test`                                                                                                                          | 58 tests passed in five files; 17 detector tests                                       |
| `npm run build`                                                                                                                         | Pass; approved JSON emitted as hash-named assets                                       |
| `npm run format:check`                                                                                                                  | Pass; upstream byte-preserved JSON excluded intentionally                              |
| `npm run test:e2e`                                                                                                                      | 15 Chromium tests passed, including three detector stories                             |
| Byte integrity and build privacy inspection                                                                                             | Eleven identical production assets; ten matching receipts; no copied runtime directory |
| `git diff --check`                                                                                                                      | Pass                                                                                   |

Detector tests cover required counts, null historical variance, version/state/metric/reference validation, file corruption, pre-specified association definitions, deterministic import and refusal before overwrite, missing/error/loading/ready UI, exact support, filtered runs, provenance and direct use of exported correlations/points. E2E covers 768/1024/1440 px containment, both themes, tabs, keyboard drawer focus, legacy routes and missing workspace independence. Axe checks passed for detector views and the run drawer; this is not accessibility certification. The 9,000-point synthetic test remains infrastructure validation only.

Earlier checks found and resolved inactive-tab contrast in the light theme and the old split-default test expectation. Browser screenshots were visually reviewed; the association legend was explicitly positioned above the plot to separate it from the x-axis label.

Non-failing build messages: Vite reports the intentionally classic `/runtime-config.js` script and the existing chart vendor chunk above 500 kB (about 592 kB minified / 201 kB gzip). These warnings are not suppressed. Test runners report `NO_COLOR`/`FORCE_COLOR` environment overlap. An ECharts disposed-instance warning appeared in an intermediate development run; the final complete suite did not report it and no JavaScript runtime error was observed.

Not executed: Python Ruff/pytest (Python exporter unchanged), scientific training/recomputation, Docker execution, remote CI, deployment, push or merge. Existing Python CI remains unchanged. The TypeScript sync command is covered by the frontend suite.

## UX artifacts and remaining limits

Ignored local screenshots: `test-results/detection-desktop.png`, `test-results/detection-tablet.png`, `test-results/detection-associations.png`. Playwright HTML report is under ignored `playwright-report/`. Tables scroll horizontally where necessary; exact observations and methodology require disclosure. On tablet widths the takeaways stack vertically. The presentation includes English interface labels and original Spanish scientific caveats. A one-minute review with an actual thesis audience has not been conducted.

No scientific values were reconstructed. The export lacks centered point coordinates, strategy-level bootstrap intervals, per-run epoch curves and dataset/feature IDs; these remain unavailable. The historical detector contract is not joined to sampled-video data. Support is labeled as exported class support because no separate unit metadata field is provided.

Recommended next step: rehearse the evaluation view with the thesis team, checking that the split-versus-seed variability and global-versus-within-strategy distinction are understood. Future scientific changes must be regenerated and verified upstream before another deterministic import.
