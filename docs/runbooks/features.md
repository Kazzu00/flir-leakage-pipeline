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

## Features desde publicaciones multishard

La fase 2B.2 conecta `video_variant_ingestion_v1` al mismo comando `features extract`
y al motor existente de raw/L2, checkpoint y verificación. El flujo está probado
offline con ZIP sintéticos y extractores simulados. **La extracción real de
embeddings de esta variante sigue pendiente**; los ejemplos siguientes son para
ejecución posterior autorizada y no describen resultados de modelos reales.

Indicar la publicación final, su **manifest.parquet original** y la raíz que
contiene los originales declarados en el ledger. Las rutas siguientes son
ejemplos genéricos; el output debe quedar fuera de la publicación y de las fuentes:

```powershell
$variantPublication = "artifacts/video_variants/<artifact_id>"
$variantInputRoot = "D:/datasets/external-variant-inputs"

uv run --offline --no-sync flir-pipeline features extract `
  --manifest "$variantPublication/manifest.parquet" `
  --video-variant-ingestion "$variantPublication" --input-root "$variantInputRoot" `
  --max-open-archives 2 --config configs/embeddings/dinov2_full.yaml `
  --limit-content 16 --seed 0 --local-files-only `
  --output-root artifacts/variant_features_smoke

uv run --offline --no-sync flir-pipeline features extract `
  --manifest "$variantPublication/manifest.parquet" `
  --video-variant-ingestion "$variantPublication" --input-root "$variantInputRoot" `
  --max-open-archives 2 --config configs/embeddings/clip_full.yaml `
  --limit-content 16 --seed 0 --local-files-only `
  --output-root artifacts/variant_features_smoke
```

Estos YAML eligen los encoders existentes y sus revisiones fijadas: DINOv2-small
CLS 384D y CLIP ViT-B/32 projected image 512D. Se necesita el entorno vision y los
snapshots locales; `--offline --no-sync` no instala dependencias y
`--local-files-only` evita descargas de pesos. `--batch-size` y `--device` pueden
sobrescribir sus valores operacionales. `--limit-content` cuenta contenidos únicos;
un smoke no caracteriza globalmente la colección ni constituye extracción completa.
Para una publicación exclusivamente sintética, el mismo comando con
`--extractor fake` y sin `--config` valida infraestructura sin cargar pesos;
usar una raíz de salida independiente y no interpretar sus vectores como DINOv2/CLIP.
Para el completo posterior, omitir ese límite y usar otra raíz, por ejemplo
`artifacts/variant_features_full`.

`--video-variant-ingestion`, `--images-root` y `--images-archive` son mutuamente
excluyentes. El transporte multishard requiere `--input-root` explícito y no
consulta `FLIR_DATA_ROOT` ni el ZIP histórico. `--max-open-archives` es entero
positivo, predeterminado 8 para este transporte; se rechaza en otros transportes,
igual que `--input-root`. Las carpetas/ZIP históricos conservan su fallback.
Si se proporciona `--variant-spec`, debe coincidir con la declaración de ingesta.

Antes de inicializar el encoder, el backend verifica publicación, receipt,
manifest científico, variante, contención de rutas y fuentes declaradas. El lector
verifica SHA256 y directorio central de cada ZIP una vez en la misma sesión que
consume las imágenes. Cada representante se lee por `frame_id`, ordinal y
`ZipInfo`, con SHA256, límites y decode QA; no se extraen ZIP a disco ni se hace
replay de todas las imágenes. Typer rechaza opciones inválidas con código 2; rechazos de
integridad o fallos de ejecución producen código 1 con diagnóstico legible.

Hay una fila de embeddings por `content_id`, ordenada determinísticamente;
`frame_id` desempata el representante sin definir orden temporal. `record_index`
conserva todas las ocurrencias, sus campos científicos y sus localizadores
físicos; las exclusiones del smoke quedan en `embedding_row=-1`.
`source_binding` (`video_variant_feature_source_v1`) vincula dataset, variante,
artifact, receipt, fingerprints de originales y checksums de publicación. Los
índices tienen sus propios checksums. Es parte de la firma de caché/checkpoint,
no del `feature_space_id`; las rutas absolutas de las fuentes tampoco lo son.

Repetir el mismo comando reanuda entre batches o verifica/reutiliza el store final
sin reescribir sus metadatos. Fuentes modificadas, reempaquetadas o una declaración
distinta se rechazan; usar una raíz de salida independiente para otra publicación.
Una raíz física distinta con archivos idénticos y las mismas rutas relativas es
compatible. Stores históricos no se vinculan retrospectivamente. Los errores de
publicación final preservan evidencia y pueden requerir revisión manual.

Verificar el directorio exacto que imprime la extracción:

```powershell
uv run --offline --no-sync flir-pipeline features verify "artifacts/variant_features_smoke/<encoder>/variants/<variant_id>/<feature_space_id>"
uv run --offline --no-sync flir-pipeline features summary "artifacts/variant_features_smoke/<encoder>/variants/<variant_id>/<feature_space_id>"
uv run --offline --no-sync flir-pipeline features verify "artifacts/variant_features_full/<encoder>/variants/<variant_id>/<feature_space_id>" `
  --manifest "$variantPublication/manifest.parquet"
```

`features verify` comprueba arrays, mappings, índices y binding almacenado; **no
abre los originales**. Con manifest exige además cobertura completa y revisión HF
resuelta. La comprobación de originales se realiza al extraer/reanudar/reutilizar.
Una auditoría exhaustiva de todas las imágenes es otra operación, más costosa:
`data video-variant-ingestion verify "$variantPublication" --input-root "$variantInputRoot"`.
Ninguna de ellas demuestra alineación temporal.

Las imágenes sin anotaciones siguen sin anotaciones: `label_exists=false` no es
una anotación negativa verificada. Splits históricos, secuencias y timestamps de
captura no se crean. Índices observados y tiempos candidatos mantienen su estatus
de evidencia candidata. El nombre `hud_reduced` no prueba la eliminación completa
del HUD. Esta población permanece separada del conjunto histórico etiquetado;
el puente no habilita automáticamente similarity, clustering, splitting,
diagnostics, reportes con imágenes ni exports de organización para este contrato.

La validación local 2B.2 añadió 55 casos con `CliRunner`, imágenes pequeñas y
constructores DINOv2/CLIP simulados; los tests bloquean imports de modelos,
conexiones de red y extracción de ZIP a disco. Junto con las regresiones del
motor, lectores, manifest, revisiones y CLI histórica: **253 passed en 247,76 s**.
La regresión del consumidor de configuración YAML en similarity pasó aparte:
**1 passed en 3,61 s**. La suite global, los fallos de organización conocidos y
la ejecución real de modelos quedan fuera de esta validación.
