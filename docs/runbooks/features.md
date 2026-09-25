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

## Features de frames muestreados locales

Usar primero `data build-video-manifest` y revisar su reporte, especialmente
`decode_failures`. El nuevo flujo conserva el pipeline histórico basado en ZIP:
`--images-archive` y `--images-root` son mutuamente excluyentes. Si no se indica
ninguno, se mantiene el fallback histórico `FLIR_DATA_ROOT/Imagenes.zip`.
Un `--images-root` explícito ignora ese fallback y lee solo las rutas declaradas.

Ejemplos Linux/Hypatia para ejecutar **después** de validar el muestreo y crear
el manifest; no se ejecutaron en esta tarea. Los modelos de estos YAML siguen
siendo DINOv2-small CLS 384D y CLIP ViT-B/32 projected image 512D:

```bash
uv run --no-sync flir-pipeline features extract \
  --manifest data/manifests/flir_video_samples_v1.parquet \
  --images-root /ruta/a/derivados/flir-frames-1fps \
  --config configs/embeddings/dinov2_full.yaml \
  --limit-content 16 --seed 0 --local-files-only \
  --output-root artifacts/video_features_smoke

uv run --no-sync flir-pipeline features extract \
  --manifest data/manifests/flir_video_samples_v1.parquet \
  --images-root /ruta/a/derivados/flir-frames-1fps \
  --config configs/embeddings/dinov2_full.yaml \
  --seed 0 --local-files-only --output-root artifacts/video_features

uv run --no-sync flir-pipeline features extract \
  --manifest data/manifests/flir_video_samples_v1.parquet \
  --images-root /ruta/a/derivados/flir-frames-1fps \
  --config configs/embeddings/clip_full.yaml \
  --seed 0 --local-files-only --output-root artifacts/video_features
```

El entorno debe tener instalado el extra `vision` y los snapshots fijados en caché
para `--local-files-only`; `--no-sync` no instala dependencias ni pesos. Repetir
el smoke con el YAML de CLIP permite revisar ambos encoders antes del completo.
Estos son comandos propuestos, sin confirmar GPUs, modelos disponibles ni estado
del trabajo real. Mantener fuentes inmutables y un solo escritor por store.

El lector local valida contención y SHA256 de **todas** las ocurrencias antes de
crear/reanudar/reutilizar la salida, incluidos duplicados y contenidos fuera del
smoke; después decodifica solo los representantes seleccionados. Esto añade una
lectura completa de JPEGs, evita que una copia modificada quede escondida por la
deduplicación y no aumenta las filas de embeddings. Raw/L2, `record_index` completo,
filas -1 del smoke, checkpoint y metadata como marcador final se conservan.
ZIP/directorio no cambia `feature_space_id`; el dataset tiene identidad separada.
Los outputs deben estar fuera de `images-root`.

Tomar el directorio exacto que imprime cada extracción y verificar:

```bash
uv run --no-sync flir-pipeline features verify /ruta/al/feature-store
uv run --no-sync flir-pipeline features summary /ruta/al/feature-store
uv run --no-sync flir-pipeline features verify /ruta/al/feature-store-completo \
  --manifest data/manifests/flir_video_samples_v1.parquet
```

La última variante requiere cobertura completa y revisión HF resuelta; no usarla
para afirmar completitud de un smoke o del extractor fake. `verify` inspecciona
el store; la comprobación de JPEGs ocurre al extraer/reanudar/reutilizar.
`diagnostics`, notebooks de revisión histórica y exploradores siguen leyendo ZIPs;
no se extienden automáticamente a este dataset sin etiquetas. La procedencia
temporal se recupera uniendo `record_index.frame_id` con el manifest, sin convertir
`video_id` en secuencia ni asumir tiempos de captura.
