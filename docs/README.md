# Documentation

Start with [pipeline](pipeline.md) for execution order and [status](status.md)
for the boundary between implemented infrastructure, validated experiments and
pending work. Commands assume the repository root; real artifacts remain local.

| System guide | Purpose |
|---|---|
| [Architecture](architecture.md) | Package responsibilities and integration boundaries |
| [Data model](data_model.md) | Occurrences, content, group identity and provenance |
| [Pipeline traceability](pipeline_traceability.md) | Input → process → artifact → consumer |
| [Reproducibility](reproducibility.md) | IDs, configs, versions, checkpoints and verification |
| [Data safety](data_safety.md) | Read-only originals and public-repository boundaries |
| [Design decisions](design_decisions.md) | Representation, metrics, noise and selection rationale |
| [References](references.md) | Existing technical sources and unresolved citations |

| Component | Analysis | Protocol | Runbook |
|---|---|---|---|
| Data | [Quality](analysis/data_quality.md), [class nomenclature](analysis/dataset_classes.md) | [Identity rules](data_model.md) | [Audit and canonicalization](runbooks/data.md) |
| Features | [Extraction evidence](analysis/features.md) | [Representation decisions](design_decisions.md) | [Extract, resume and report](runbooks/features.md) |
| Similarity | [Cosine and temporal relations](analysis/similarity.md) | Rules in the analysis | [Compute and verify](runbooks/similarity.md) |
| Reduction | [t-SNE / PaCMAP](analysis/reduction.md) | [Reduction](protocols/reduction.md) | [Reduction](runbooks/reduction.md) |
| Clustering | [Density and stability](analysis/clustering.md) | [Clustering](protocols/clustering.md) | [Clustering](runbooks/clustering.md) |
| Splitting | [Baselines and residuals](analysis/splitting.md) | [Splitting](protocols/splitting.md) | [Splitting](runbooks/splitting.md) |
| Detection | [Candidate audit and pilots](analysis/detection.md) | [Detection](protocols/detection.md) | [Detection](runbooks/detection.md) |

[Streamlit](visualization/streamlit.md) and [VIKUS](visualization/vikus.md) document
installation, launch, privacy, source joins and display limitations.
The [explorer runbook](runbooks/explorers.md) helps choose an interface;
the [detector/residual explorer](visualization/detector_similarity.md) inspects
controlled runs and the experiment matrix separately from image-level tools;
the [consolidated project report](runbooks/project_report.md) maps its evidence.

The [configuration catalog](../configs/README.md) separates executed settings,
smoke checks and unexecuted larger-model candidates. Source notebook names are
retained for compatibility; their generated reports are not versioned.
