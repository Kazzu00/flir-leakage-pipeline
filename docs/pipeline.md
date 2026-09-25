# Pipeline

Execute from the repository root using explicit source paths. Existing results
can be inspected without rerunning the pipeline. [Status](status.md) records
what has actually executed; the following is the component dependency order.

| Stage | Input → operation → output | Operational guide |
|---|---|---|
| Audit and canonicalization | External ZIPs → inventory, hashing, label QA → occurrence manifest, diagnostics and aggregate reports | [Data](runbooks/data.md) |
| Video preparation | Source videos → sampled JPEG occurrences → verified `flir_video_samples_v1` manifest and exact-byte content identity | [Data](runbooks/data.md#manifest-de-ocurrencias-muestreadas) |
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

The separate video path is:

```text
source videos
    → sampled frame occurrences (relative sampling grid)
    → occurrence manifest + exact-byte content identity / dedup
    → DINOv2 / CLIP unique-content features (same storage pipeline)
    → future similarity / sequence identification
```

Source-video sampling has confirmed operational validation on Hypatia: a real
5-second smoke and completed job 737719, producing 9648 JPEGs at 1 FPS from three
source videos. See the [execution evidence](runbooks/data.md#evidencia-real-confirmada-en-hypatia).
The occurrence/content-to-features bridge remains tested synthetically only;
`build-video-manifest` has not run on Hypatia. Sampling does not establish
sequences, scene boundaries, embeddings, clustering, splits, leakage removal or
detector performance for these videos. Historical downstream experiments do not
validate the new dataset automatically.
An N=16 feature smoke or a tiny detector pilot validates infrastructure only.

Use [pipeline traceability](pipeline_traceability.md) to follow stored identities
and downstream consumers, and [design decisions](design_decisions.md) for scientific
boundaries. Source-bound verification may recompute stage metrics; it should be
run deliberately, separately from reading reports or documentation maintenance.
