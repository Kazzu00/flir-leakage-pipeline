# Pipeline

Execute from the repository root using explicit source paths. Existing results
can be inspected without rerunning the pipeline. [Status](status.md) records
what has actually executed; the following is the component dependency order.

| Stage | Input → operation → output | Operational guide |
|---|---|---|
| Audit and canonicalization | External ZIPs → inventory, hashing, label QA → occurrence manifest, diagnostics and aggregate reports | [Data](runbooks/data.md) |
| Representation | Unique content and pinned model configs → independent DINOv2 / CLIP → raw/L2 arrays and occurrence mappings | [Features](runbooks/features.md) |
| Similarity | Verified original L2 → cosine and deterministic neighborhoods → matrices, pairs and posterior provenance | [Similarity](runbooks/similarity.md) |
| Reduction | Original L2 → t-SNE / PaCMAP → coordinates, preservation and seed comparisons | [Reduction](runbooks/reduction.md) |
| Clustering | Original L2 controls or selected reductions → DBSCAN / OPTICS / HDBSCAN → assignments, noise, coherence and ARI/AMI | [Clustering](runbooks/clustering.md) |
| Splitting | Selected clusters, occurrences and annotation counts → atomic allocation / seeded baselines → assignments and residual evaluation in both encoders | [Splitting](runbooks/splitting.md) |
| Inspection | Existing assignments, saved coordinates and selected ZIP images → local exploration → display-only views or ignored bundles | [Explorers](runbooks/explorers.md) |
| Detection | Frozen split plan and byte-checked occurrence views → controlled YOLO11n runtime → checkpoints, metrics, bootstrap and reports | [Detection](runbooks/detection.md) |

Representation, similarity, reduction and clustering use unique `content_id`.
All historical `frame_id` occurrences remain traceable. Labels and historical
splits never enter the encoders or numerical clustering adapters. Split balance
uses labels only after clustering; residual similarity is evaluated afterwards.

Inspector output is not a prerequisite for recomputing numerical metrics, nor
proof of coherent scenes. Full detector comparison remains pending compute.
An N=16 feature smoke or a tiny detector pilot validates infrastructure only.

Use [pipeline traceability](pipeline_traceability.md) to follow stored identities
and downstream consumers, and [design decisions](design_decisions.md) for scientific
boundaries. Source-bound verification may recompute stage metrics; it should be
run deliberately, separately from reading reports or documentation maintenance.
