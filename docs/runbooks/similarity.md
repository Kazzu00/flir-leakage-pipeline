# Similarity execution and verification

## Reproducir y verificar

Con Python 3.11, `uv sync --locked --extra dev --extra reporting`, datos originales
externos y ambos directorios completos disponibles. No se necesita `vision` ni
descargar modelos para esta fase. Sustituir los marcadores por directorios locales
explícitos; no seleccionar un smoke por orden alfabético de carpetas.

```powershell
uv run flir-pipeline similarity --help
uv run flir-pipeline similarity compute --feature-directory <dinov2_features> --manifest data/manifests/flir_canonical_candidate_v1.parquet --config configs/similarity/dinov2_research.yaml
uv run flir-pipeline similarity compute --feature-directory <clip_features> --manifest data/manifests/flir_canonical_candidate_v1.parquet --config configs/similarity/clip_research.yaml
uv run flir-pipeline similarity summary <dinov2_similarity>
uv run flir-pipeline similarity summary <clip_similarity>
uv run flir-pipeline similarity verify <dinov2_similarity> --feature-directory <dinov2_features> --manifest data/manifests/flir_canonical_candidate_v1.parquet
uv run flir-pipeline similarity verify <clip_similarity> --feature-directory <clip_features> --manifest data/manifests/flir_canonical_candidate_v1.parquet
uv run flir-pipeline similarity compare --left <dinov2_similarity> --right <clip_similarity>
uv run --extra reporting python scripts/build_similarity_review.py --dinov2 <dinov2_similarity> --clip <clip_similarity> --comparison <comparison_directory> --seed 0 --examples 3
uv run ruff check .
uv run pytest
uv run python scripts/check_notebook_source.py
```

Compute exige verificación de cobertura completa y revisión de modelo resuelta;
reutiliza cachés completas solo si firmas y QA coinciden. Preserva y rechaza
directorios parciales. Verify autónomo recalcula invariantes e integridad; con
los dos argumentos opcionales comprueba además los arrays originales y el manifest.
Su código de salida es distinto de cero ante fallo. El builder comprueba fuentes,
firmas y valores de Jaccard, y lee imágenes desde `FLIR_DATA_ROOT/Imagenes.zip`
o `--images-archive`, en memoria y sin modificar el ZIP.

La ejecución registrada utilizó un entorno uv aislado,
Python 3.11.14 y dependencias del lockfile ya almacenadas, sin reinstalación de
visión. Windows bloquea el launcher generado; el mismo CLI se invocó mediante
`uv run --no-sync python -c "from flir_pipeline.cli import app; app()" similarity ...`.
Los metadatos registran el commit activo anterior al commit de cierre, worktree
dirty y hashes de fuente al calcular. Se preservan; el recibo final enlaza la
verificación con el código comprometido, sin reescribir la procedencia original.


Outputs: matrices, unordered pairs, directed top-k tables and posterior provenance
under `artifacts/similarity/`; local tables/figures/HTML under `reports/similarity/`.
See [observed results and limits](../analysis/similarity.md).
