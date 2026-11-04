# FLIR Pipeline Explorer — permanent agent rules

This repository is the independent frontend `flir-pipeline-explorer`. Scientific work remains in the three upstream repositories. Do not apply the old backend workspace scope to this frontend.

- `.references/` contains ignored, read-only shallow clones. Never modify, commit inside, or publish their contents.
- Inspect `docs/RESEARCH.md` and current code before changing scientific vocabulary. Preserve source disagreements as inconsistent / experimental / unavailable, not as resolved facts.
- Never invent scientific results. Label synthetic data DEMO / MOCK / SAMPLE at the workspace and result level. A lifecycle status is not scientific verification.
- Source video != sequence. Cluster != sequence. Sampling-grid time != verified capture time. Pilot != scientific result.
- Cluster != scene. Candidate boundary != confirmed boundary. Boundary zone != exact cut. Sequence candidate != sequence instance. Review != ground truth. Linkage candidate != confirmed link. Supported group review does not confirm an exact frame/sequence match. Visual recurrence != exact duplication; similarity != temporal continuity. Dataset variant difference != causal effect. Historical dataset != sampled-video dataset.
- M02 has four primary sections: Overview, Visual exploration, Sequences & linkage, Evaluation. Preserve legacy redirects; keep identities and raw reported metadata behind keyboard-accessible disclosure. Availability, execution, integrity and scientific verification are distinct.
- Preserve LeakageSnapshotV1. V2 adds optional, explicitly pending research stages with strict identity validation. Never execute upstream verification that fits or recomputes science from this exporter.
- Docker serves static dist through nginx. Runtime config is public and validated before use; no secrets. Mount snapshots read-only, never COPY them into images. Keep Docker contexts free of .references, personal environment files and real artifacts.
- Real data and runtime snapshots must never be committed. Artifacts are read-only. Only the explicitly invoked offline M02 exporter may read a user-supplied manifest, metadata, indices and arrays for linear integrity checks; never read images, weights, archives, credentials or personal environment files. Never publish source artifacts or copy images from reference clones.
- This frontend visualizes existing data only. No training, denoising, embedding extraction, reduction, clustering, segmentation, or scientific split generation.
- Keep the adapter boundary: components consume `useSnapshot`; adapters validate unknown data with Zod before exposing it. No repository-specific CSV/JSON/Parquet parsing in components.
- Preserve frame/content/embedding/cluster/group/split identities. One clustering row per content; cluster-aware groups are indivisible; noise is explicit and uses singleton allocation groups. Historical overlap is a valid audit state, not silently repaired.
- Unavailable metrics are null, never zero. Retain annotation/evaluation caveats. Do not rank methods by a single score or visual appearance.
- Target WCAG 2.2 AA: semantic HTML, keyboard alternatives to charts, visible focus, explicit labels, contrast and redundant encodings. Automated checks are not certification.
- Use npm and the committed lockfile for the frontend. The isolated Python exporter uses uv and pinned dependencies; it is not a backend or a scientific executor. No scientific commands for frontend changes.
- Dataset and feature identities must match across all imported stages. Validate original identities before anonymizing them; safe aliases are scoped to a snapshot, not join keys across exports.
- Missing M02 scientific stages stay pending/unavailable and must never silently fall back to mock. Keep mock/artifact/reported/unavailable distinguishable at workspace and run level. M01/M03 stay DEMO in MIXED mode.
- Runtime snapshots belong in ignored `public/runtime/` and must not enter the production build. Keep `build.copyPublicDir: false` unless a separately reviewed privacy-preserving asset strategy replaces it.
- Test exporter invariants with synthetic temporary artifacts only; run Ruff and pytest. A 9,000-point UI test verifies infrastructure, never experimental results.
- Run `npm run lint`, `npm run typecheck`, `npm run test`, `npm run build`; run `npm run test:e2e` for UI changes. Fix failures; report anything not executed.
- Before committing or pushing inspect `git status`, `git diff`, `git diff --cached` and tracked files. Commit logical changes; keep generated outputs ignored.
- Do not modify or delete other repositories. No deployment, live API, or experimental result may be described as completed unless executed and verified.

## Explicit detector-contract exception · 2026-10-04

- The user authorized checking in only the public `detection-export-v1` JSON contract under `src/features/detection/snapshot/`. This narrow exception includes the upstream schema and aggregate/run/support/bootstrap outputs; it does not permit images, native artifacts or general runtime snapshots.
- `useDetectionSnapshot` is the canonical detector adapter/query boundary. It is independent of `useSnapshot` and workspace DEMO/MIXED selection; components never parse JSON or recompute science.
- Preserve upstream bytes/checksums and the distinction between publication commit and generator `source_commit`. Update through `npm run sync:detection`; see `docs/DETECTION_CONTRACT.md`.
- The approved detector contract is emitted as hash-named static assets. `build.copyPublicDir: false` and ignored `public/runtime/` remain unchanged. Keep detector failure states explicit with no demo fallback.
