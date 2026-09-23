# Data and artifact safety

Original ZIPs live outside the repository and are accessed through
`FLIR_DATA_ROOT`, configured in the environment or ignored `.env`.
[.env.example](../.env.example) stays versioned as a template; never rename it
away or commit local absolute paths and credentials.

Original archives are read-only: do not rewrite, rename, repackage or extract them
into the source tree. Inventory uses streaming hashes. Feature/report/explorer
code reads selected ZIP members in memory and checks their bytes where required.
Detector materialization creates separate ignored occurrence views; it preserves
source annotations, including conflicts, and never edits the originals.

| Versioned | Local and ignored |
|---|---|
| Source code and synthetic tests | Real images, labels, videos, ZIPs and content hashes |
| Generic configs and lockfile | Weights, caches, embeddings, matrices and assignments |
| Aggregate documentation | Manifests, inventories, detailed diagnostics and receipts |
| Notebook sources without outputs | Executed notebooks, HTML, figures and image previews |
| Explorer implementation | VIKUS runtime bundles, sprites, previews and local metadata |

[.gitignore](../.gitignore) protects `data/`, `artifacts/`, `reports/`, caches,
model formats, media and generated notebooks/HTML. Only `.gitkeep` placeholders
are retained in data/report directories. Check tracked files as well as ignore
rules: `.gitignore` does not untrack an already committed file.

Streamlit and VIKUS are local inspection tools served on loopback. Bundles include
real imagery and must not be published. In-memory GIFs are display caches;
saved previews and sprites belong under ignored report directories.

Preserve incomplete `.partial` state until its dataset, feature identity,
checkpoint metadata and any equivalent final output have been checked. A failed
or resumable artifact is not disposable because it is old. Real report outputs
and experimental configs remain evidence, even after an experiment finishes.
