# Runtime snapshot size and future fragments

Read-only structural validation on 2026-10-02 accepted the existing local `LeakageSnapshotV1`: **8,093 contents, 9,648 occurrences, 40 runs, 52,873,506 bytes**. No artifact was rewritten or copied. Parsing through the current Zod schema took approximately 1.5 seconds on this workstation in one run; this is not a general performance benchmark or scientific validation.

Compact-JSON field sizes observed by serialization (formatting overhead excluded):

| Field                  |      Bytes |
| ---------------------- | ---------: |
| Contents / occurrences |  1,821,592 |
| Top-k pairs            | 45,695,437 |
| Reduction coordinates  |  3,126,781 |
| Feature indices        |    823,344 |
| Summaries              |     68,817 |

Pairs dominate the payload. V2 retains V1 arrays without copying them into the research extension; linkage normalizes labeled occurrences and references existing video contents. No vectors, NxN matrices, images, masks or binary content are added. The existing 80 MiB guard remains. Oversized exports fail rather than subsample silently.

Canvas renders persisted points; React renders bounded content rows, six gallery placeholders, 20 sequence intervals or 12 linkage pairs per page. Similarity uses a memoized query index. Technical drawers mount on demand. Heavy domain routes load lazily. ECharts remains a shared large chunk; its build warning is acknowledged, not suppressed.

## Proposed future protocol — not implemented

An immutable `runtime/manifest.json` would declare the schema, snapshot-scoped dataset/feature identities, content index, run metadata, byte limits, SHA256 and root-relative URL for each fragment. Separate `reduction/<run>.json`, `clustering/<run>.json`, `sequences/<id>.json` and `linkage/<id>.json` would load through adapter methods and TanStack Query keys containing snapshot identity + run + checksum.

The manifest must validate before any fetch. Each fragment must validate its original snapshot binding, dataset, feature space, row mapping and declared byte limit before exposure. Loading must be cancellable; failures must remain local/unavailable without mock fallback. Frozen snapshot directories prevent mixing a replaced manifest with stale fragments. Validate the entire dependency chain before enabling AFTER SPLIT. Lazy fragments must not use aliases as join keys across independently exported snapshots.

This migration is deliberately deferred to preserve the existing integration. Do not describe fragment lazy loading as delivered; only page-code lazy loading is active.
