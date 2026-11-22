# Architecture · UX v2

React components consume `useSnapshot` for workspace data and `useDetectionSnapshot` for the independent verified detector contract; unknown external data is validated by adapters and Zod. Components never parse upstream JSON/CSV/Parquet. The offline Python exporter reads explicitly supplied artifacts, checks consumed bytes and original identities, and publishes an allowlisted snapshot without executing science.

`RealLeakageAdapter` accepts the discriminated union LeakageSnapshotV1/V2. V2 reuses all V1 invariants before validating research extensions. `CompositeDataAdapter` keeps M01/M03 DEMO while clearing the mock M02 collections. Dataset and feature identity continuity remain mandatory; cross-dataset linkage declares a separate labeled dataset with its own occurrences.

## Domain components

- `DatasetOverview`: compact coverage and visual/temporal availability graph.
- `VisualExplorer`: URL view selection and encoder controls for legacy representation views; clustering dispatches directly to `organization-evidence-v2`, without a misleading encoder selector.
- `clustering/Clustering`: configuration and run-local cluster selection through `OrganizationEvidenceProvider` / its Zod adapter. Memoized indices join memberships to unique contents, preview keys and occurrence records. It checks join identities without executing science. Frozen split membership provenance, full-artifact availability and membership verification remain distinct.
- `ClusterVisualGrid` / `ClusterContentDetail`: real local previews, 60-content rendering batches, explicit missing media, selected evidence and bounded occurrence lists. No coordinates are fabricated; `Scatter` and `Gallery` remain available to legacy Embeddings. Run/cluster changes reset content selection and rendering limits.
- `ArtifactStage`: small compatibility/run dispatcher; no dataset or sequence domain logic.
- `evidence-views/ArtifactExplorer`, `SimilarityEvidence`, `SplitEvidence`, `SummaryTables`, `Pager`: independent domain rendering and bounded tables.
- `SequenceExplorer`, `LinkageExplorer`: temporal intervals, evidence provenance, independent encoder scores and reviews.
- `EvaluationOverview`: defaults to the independent detector contract, retaining compatible grouping/split evidence at `?view=splits`.
- `features/detection`: strict Zod contract, byte-hash adapter, shared Query cache, comparison/variability/class/support/association views and traceability drawers. No scientific calculations; see [DETECTION_CONTRACT](DETECTION_CONTRACT.md).
- `TechnicalDetailsDrawer`: Radix modal, focus trap, Escape and trigger-focus restoration; heavy audit content mounts only on request.

Heavy pages use React.lazy/Suspense. Query owns snapshot loading/cache; selections remain local or in URL search parameters. ECharts renders point clouds on Canvas, with no React element per point. Content/source lookups and similarity query indices are memoized; full-content projections remain intact. The sampling strip shows recorded positions, not continuous capture time. Keyboard tables are bounded and disclosed explicitly.

## Static runtime

Vite development retains VITE_* compatibility. A public `window.__FLIR_CONFIG__` object takes precedence and is schema-validated. The container writes it at startup, allowing mode/URL changes without rebuilding. Invalid config fails visibly.

Docker uses locked npm dependencies and digest-pinned Node/nginx base images. Only dist reaches nginx. Runtime snapshots mount read-only; `build.copyPublicDir: false` remains mandatory. No backend, uploads, credentials, scientific execution, WebSockets or cloud deployment exists. See [DOCKER](DOCKER.md).

V1 is still monolithic. Fragment loading is a documented future protocol, not an implemented feature; see [SNAPSHOT_PERFORMANCE](SNAPSHOT_PERFORMANCE.md). Reference clones and all real runtime data remain ignored and read-only.

The explicitly authorized public detector contract is a narrow exception: checked in under `src/features/detection/snapshot/`, emitted as hash-named static assets. It does not unignore or include general runtime snapshots. Its data gate is independent of the workspace snapshot; missing/invalid detector data never falls back to demo.

The organization clustering route also bypasses the legacy snapshot data gate. Its shell identifies exported evidence, and the view reports its own loading, validation, missing-resource and identity-error states. JSON resources under `/runtime/organization-evidence-v2/` and previews resolved by `organizationMediaUrl` remain read-only, local and excluded from production builds. This UI integration does not validate scientific clustering quality or establish temporal identities.
