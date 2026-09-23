# Reproducibility

Use Python 3.11 and the committed `uv.lock`. Install only the extras needed for
the operation, retaining all desired extras in each `uv sync` invocation.
The [configuration catalog](../configs/README.md) distinguishes executed full
models, N=16 smoke checks and unexecuted larger-model candidates.

## Identity and provenance

`dataset_id` binds the annotated manifest. `feature_space_id` describes the
mathematical representation: model/revision, preprocessing, pooling, dimension
and defined representation settings. CPU/GPU and batch size are runtime metadata,
not reasons to change that identity. DINOv2 uses CLS; CLIP uses the projected image
embedding. Raw and L2 arrays are separate and never overwrite each other.

`similarity_space_id`, `reduction_space_id`, `clustering_space_id` and
`split_space_id` bind their upstream identities and relevant scientific settings.
See [traceability](pipeline_traceability.md) and [identity decisions](design_decisions.md).
Paths and timestamps do not identify mathematical experiments. Source signatures
and output checksums additionally protect actual bytes, selection and provenance.

Metadata records seeds, versions, effective model revision, preprocessing, pooling,
dimensions, dtype, execution commit and creation time where applicable. Preserve
the actual execution commit/dirty state and pre-fit source/config snapshots;
never retroactively attribute results to a later documentation or verifier commit.

## Resume and verification

Feature checkpoints advance after flushed batches; metadata is the completion
marker. Repeating the same command, manifest, config, seed and output root resumes
or verifies/reuses a complete output. Only one writer per directory is supported.
Recovery during final file promotion is not automatic. Keep smoke and full roots
separate even when they share a `feature_space_id`; selection is cache-bound.

Later stages reuse verified complete runs as grid checkpoints. Incomplete
publications are preserved and refused; use a separate output root where required.
Standalone checks establish internal integrity. Source-bound verification also
binds original inputs and, depending on the stage, recomputes metrics. Numerical
verification does not establish semantic coherence or detector improvement.

Model weights are not in `uv.lock`. Pinned snapshots must exist locally for
`--local-files-only`; an initial authorized extraction can download them explicitly.
Tests and CI use synthetic inputs and stand-ins, never real model downloads.

## Code and report checks

```powershell
uv sync --locked --extra dev --extra reduction
uv run ruff check .
uv run python -m pytest
uv run python scripts/check_notebook_source.py
uv run flir-pipeline --help
uv run python scripts/build_progress_review.py --check
```

The report `--check` reads existing evidence and writes no outputs. Missing local
data in a public clone is reported, not replaced by fabricated results. Building
the report separately requires the `reporting` extra; its seven source notebooks
retain their existing names. Generated HTML/notebooks stay ignored.

If Windows blocks the generated `flir-pipeline` launcher, the same Typer entry
point is accessible with `uv run python -c "from flir_pipeline.cli import app; app()" --help`.
Choose an isolated `UV_PROJECT_ENVIRONMENT` when needed; environment names are
operational details and do not rewrite historical experiment metadata.
