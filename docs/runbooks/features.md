# Feature extraction and reports

Run from the repository root with Python 3.11, external read-only ZIPs,
`FLIR_DATA_ROOT`, the canonical manifest and pinned configurations.
Prepare data using the [data runbook](data.md).

```powershell
uv sync --locked --extra dev --extra vision --extra reporting
uv run flir-pipeline features diagnostics --manifest data/manifests/flir_canonical_candidate_v1.parquet
```

Diagnostics are QA/EDA, never concatenated with encoder vectors. Use separate roots
for sampled/full diagnostics because their store does not resume multiple runs.

## Extraction and resume

Validar primero las mismas revisiones del completo con salida independiente:

```powershell
uv run --extra vision flir-pipeline features extract --manifest data/manifests/flir_canonical_candidate_v1.parquet --config configs/embeddings/dinov2_full.yaml --limit-content 16 --seed 0 --local-files-only --output-root artifacts/features_revision_smoke
uv run --extra vision flir-pipeline features extract --manifest data/manifests/flir_canonical_candidate_v1.parquet --config configs/embeddings/clip_full.yaml --limit-content 16 --seed 0 --local-files-only --output-root artifacts/features_revision_smoke
```

Comandos de extracción completa registrados:

```powershell
uv run --extra vision flir-pipeline features extract --manifest data/manifests/flir_canonical_candidate_v1.parquet --config configs/embeddings/dinov2_full.yaml --seed 0 --local-files-only --output-root artifacts/features
uv run --extra vision flir-pipeline features extract --manifest data/manifests/flir_canonical_candidate_v1.parquet --config configs/embeddings/clip_full.yaml --seed 0 --local-files-only --output-root artifacts/features
```

Las revisiones están fijadas en los YAML. Se reutilizó la caché; no hubo solicitudes
de pesos durante la ejecución. En una máquina sin esos snapshots, quitar
`--local-files-only` permite descargarlos una primera vez. Las dependencias de
`uv.lock` no incluyen pesos.

Para reanudar, repetir **el mismo comando**, manifest, configuración, semilla y
raíz. El checkpoint avanza después de guardar el lote; una salida completa válida
se verifica y reutiliza. No usar la raíz del smoke para el completo: su selección
es diferente aunque comparta feature_space_id. Un solo escritor por directorio;
la recuperación de un fallo durante la promoción final de archivos no es automática.

## Verificación y reporte

Estos comandos PowerShell seleccionan exclusivamente salidas verificadas completas
del manifest y conservan el rechazo ante más de un candidato completo por encoder:

```powershell
$featurePathsJson = uv run python -c "import json; from pathlib import Path; import pandas as pd; from flir_pipeline.features.visualization import discover_feature_directories; m=pd.read_parquet('data/manifests/flir_canonical_candidate_v1.parquet'); print(json.dumps({k:str(v) for k,v in discover_feature_directories(Path('artifacts/features'),full_manifest=m).items()}))"
if ($LASTEXITCODE -ne 0) { throw "Falló la selección verificada" }
$featurePaths = $featurePathsJson | ConvertFrom-Json
uv run flir-pipeline features verify $featurePaths.dinov2 --manifest data/manifests/flir_canonical_candidate_v1.parquet
uv run flir-pipeline features verify $featurePaths.clip --manifest data/manifests/flir_canonical_candidate_v1.parquet
uv run --extra reporting python scripts/build_feature_engineering_review.py
```

`verify` sin manifest controla la integridad interna y admite muestras con filas
no seleccionadas (-1). Con `--manifest` exige cobertura total, correspondencia
canónica y revisión HF resuelta; devuelve código distinto de cero si falla.

El modo completo predeterminado (`--full`) filtra muestras y datasets distintos, exige ambos encoders completos y
rechaza ambigüedad. Se pueden proporcionar `--dinov2` y `--clip` explícitos;
`--no-full` permite seleccionar muestras sin exigir cobertura completa.
`--labels-archive` permite seleccionar el ZIP; el valor por defecto es
`FLIR_DATA_ROOT/Etiquetas.zip`. `--max-frame-gap 1` documenta el umbral nominal.
El builder también valida el YAML de `FLIR_DATA_ROOT/dataset_split_completo.zip`,
seleccionable mediante `--class-config-archive`, y conserva nombres originales y
canónicos en el catálogo local. Verifica que IDs y nombres no hayan cambiado.
El builder no carga modelos y calcula el estado de cobertura desde las verificaciones.


## Outputs and limits

See [extraction evidence and output inventory](../analysis/features.md),
[reproducibility](../reproducibility.md) and [data safety](../data_safety.md).
An N=16 smoke validates infrastructure; full coverage requires manifest-bound verification.
The report does not load models. Class presence counts records; geometry counts
individual boxes using normalized source coordinates, sample standard deviation
and linear quartiles. Annotation conflicts remain visible.
