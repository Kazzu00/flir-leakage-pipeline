# Local audit reviewer v1.1

Conclusive reviewed decisions require explicitly consulted, event-bound evidence.
Pending or fully undetermined decisions may remain without it. Existing v1 reviews
without evidence are retained and flagged. Availability filters distinguish
registered images from missing evidence; temporal sorting is grouped by source,
and next-pending navigation follows the filtered order.

An existing PNG/JPEG can be registered through backend `sequences audit attach-evidence`
with event ID, locator, correspondence basis, rationale and reviewer. The backend
freezes bytes and metadata; the UI shows them and requires explicit consultation.
Temporal proximity alone is not identity. There is no bulk extraction. Consult
backend `docs/runbooks/audit_human_review_v11.md` for commands and limits.

CSV/JSON companion histories remain separate and traceable. Historical cross-split
index overlap is displayed as an anomaly, never automatically as semantic leakage.
Sample bounds, nominal times and unknown temporal precision remain distinct.

`/organization/sequences?mode=audit` adds Human audit review to the existing
Sequences & linkage workspace. It consumes the separate `audit-evidence-v1`
local service; organization/detector snapshots remain read-only.

Start the backend from its checkout (Python 3.11, uv):

```powershell
uv sync --extra dev
uv run flir-pipeline sequences audit demo reports/audit-demo
uv run flir-pipeline sequences audit serve --db reports/audit-demo/audit.sqlite3
```

Start this frontend:

```powershell
npm ci
npm run dev -- --port 5173 --strictPort
```

Open `http://127.0.0.1:5173/organization/sequences?mode=audit`. The API is
loopback-only at port 8765 and allows the frontend origin at port 5173.
No remote endpoint, automatic scientific computation or image upload is used.
The backend rejects nonlocal origins; the client rejects nonlocal hostnames.
No runtime evidence is bundled into the production build.

The demo is explicitly SYNTHETIC at workspace and case level. Missing service
or invalid schema produces an error, never a mock fallback. Filters select
source, method, review state and alert kind. Image context is the original
available contact sheet, including before/after panels; open it at full resolution.
Cases without media display an explicit unavailable state. Mark evidence as
consulted yourself; loading a picture is not recorded as human inspection.

Continuity (`continuous`, `shot_change`, `uncertain`) and visual/region dependence
(`same_region`, `different_region`, `uncertain`) are independent. A cut does not
prove independence, and a whole video is not an indivisible sequence. Times are
nominal, not verified alignment. Confidence, reviewer, justification, phenomena
and pending/reviewed state are saved in SQLite by the backend. Corrections append
history; closing/restarting the service or browser does not remove decisions.
Simultaneous stale edits are rejected; refresh before retrying. Decisions are
self-attributed human judgments, not ground truth or a split-generation command.

The Zod adapter `src/data/audit-client.ts` validates API data outside components;
`AuditReview.tsx` presents it using existing React Query. No FAC parsing,
scientific calculations, new UI framework or additional dependencies.

Verification:

Observed v1.1: 94 unit tests and 2 real-service audit E2E tests passed; lint has
0 errors and 3 pre-existing hook warnings; typecheck/build passed. Global E2E:
16 passed, 1 failed (preview wait outside the auditor), 8 skipped. The failed
test passed in isolation, which does not make the global run green. Backend global
pytest was incomplete at its 300-second budget; 49 focused backend tests passed.
Accented Spanish strings are checked in source and browser persistence tests.

```powershell
npm run lint
npm run typecheck
npm run test
npm run build
npm run test:e2e -- e2e/audit-review.spec.ts
```

The E2E test requires the backend in the sibling `flir-leakage-pipeline` directory
or `FLIR_AUDIT_BACKEND` pointing to its local checkout. It uses that checkout's
`.venv`, creates a new synthetic ledger in the OS temporary directory, starts
the real API and restarts it after saving/correcting a decision. It checks image
decoding, exact JSON export persistence, CSV export and rejection of remote Origin.
Keep port 8765 free. No FAC files or model downloads are required. Temporary
synthetic test evidence is retained for inspection; no real data is committed.

Current limitations: no video playback or missing-frame reconstruction, no
automatic matching of windows from separate packages, no inference that close
nominal times refer to the same event, no real human review performed by tests.
The full backend runbook is `docs/runbooks/audit_human_review.md` in its checkout.
