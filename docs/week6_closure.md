# Cierre controlado hasta semana 6

La matriz de [estado](current_status.md) responde al alcance de la propuesta
transcrito por el usuario. El calendario íntegro no estuvo disponible.
Se cierra la preparación de datos y representaciones; la siguiente fase será
similitud/correlación entre fotogramas.

## Punto de partida y cierre

Ya existían inventario, manifest, validación YOLO, auditoría de duplicados,
diagnósticos completos, adapters, checkpointing y dos smoke reales N=16.
La captura de revisión efectiva ya estaba implementada en el review previo, pero
solo probada con stand-ins; los smoke antiguos seguían con revisión desconocida.

Faltaban conteos separados de imágenes e instancias; una auditoría temporal
tabulada; evidencia real con revisiones resueltas; ambos espacios completos;
verificación contra todos los registros; y una presentación JP en español.
Esta iteración genera esas evidencias y conserva el candidato y artefactos previos.

## Reproducir extracción y reanudación

Desde la raíz del repositorio, con Python 3.11, ZIP externos y
`FLIR_DATA_ROOT` configurado:

```powershell
uv sync --locked --extra dev --extra vision --extra reporting
uv run flir-pipeline data inventory --inspect-archives --hash-members
uv run flir-pipeline data compare-archives
uv run flir-pipeline data build-manifest
uv run flir-pipeline data validate-labels
uv run flir-pipeline data manifest-summary data/manifests/flir_canonical_candidate_v1.parquet
uv run flir-pipeline features diagnostics --manifest data/manifests/flir_canonical_candidate_v1.parquet
```

Estos comandos de preparación son para una reproducción desde los originales.
En este cierre se reutilizaron el manifest y diagnostics completos existentes.
No extraer ni reescribir los ZIP. El CLI admite rutas explícitas; consultar
`--help` para variantes sin variable de entorno.

Validar primero las mismas revisiones del completo con salida independiente:

```powershell
uv run --extra vision flir-pipeline features extract --manifest data/manifests/flir_canonical_candidate_v1.parquet --config configs/embeddings/dinov2_full.yaml --limit-content 16 --seed 0 --local-files-only --output-root artifacts/features_revision_smoke
uv run --extra vision flir-pipeline features extract --manifest data/manifests/flir_canonical_candidate_v1.parquet --config configs/embeddings/clip_full.yaml --limit-content 16 --seed 0 --local-files-only --output-root artifacts/features_revision_smoke
```

Extracciones completas ejecutadas en esta iteración:

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
uv run --extra reporting python scripts/build_feature_engineering_review.py --full
```

`verify` sin manifest controla la integridad interna y admite muestras con filas
no seleccionadas (-1). Con `--manifest` exige cobertura total, correspondencia
canónica y revisión HF resuelta; devuelve código distinto de cero si falla.

`--full` filtra muestras y datasets distintos, exige ambos encoders completos y
rechaza ambigüedad. Se pueden proporcionar `--dinov2` y `--clip` explícitos.
`--labels-archive` permite seleccionar el ZIP; el valor por defecto es
`FLIR_DATA_ROOT/Etiquetas.zip`. `--max-frame-gap 1` documenta el umbral nominal.
El builder no carga modelos y calcula el estado de cierre desde las verificaciones.

## Evidencia y unidades

Cada espacio completo conserva seis archivos: `embeddings_raw.npy`,
`embeddings_l2.npy`, `content_index.parquet`, `record_index.parquet`,
`metadata.json`, `feature_quality.json`. Hay 1459 vectores por encoder,
representados en dos arrays raw/L2, y 1657 mappings históricos.
Raw y L2 no se cuentan como extracciones independientes.

Los nuevos smoke contienen 16 vectores por encoder y permanecen en otra raíz.
No se alteraron los smoke antiguos ni se les asignó una revisión retrospectiva.

Metadata registra modelo, SHA HF efectivo, processor, pooling, dimensiones,
dtypes, semilla, IDs, versiones, commit y fecha. El commit capturado durante la
extracción es el HEAD previo al commit de cierre; los YAML nuevos estaban aún
sin versionar. El recibo local conserva el SHA256 de cada YAML y el comando.
Los encoders y el flujo de extracción no cambiaron respecto al backend ya revisado.
Esto describe el estado real de ejecución, sin atribuirlo a un commit posterior.

Artefactos principales:

- `reports/feature_engineering/jp_review/feature_engineering_jp_review.html`: presentación sin celdas de código.
- `reports/feature_engineering/jp_review/feature_engineering_jp_review.executed.ipynb`: notebook ejecutado con código.
- `reports/feature_engineering/feature_engineering_report.md` y `metadata.json`.
- `tables/annotation_class_distribution.csv`, `annotation_quality.csv`, `empty_annotations.csv`.
- `tables/temporal_summary.csv`, `temporal_coverage.csv`, `temporal_lineage.csv`, `cross_split_temporal_candidates.csv`.
- `tables/historical_baseline.csv`, `historical_split_summary.csv`, `image_geometry_summary.csv`, quality tables.
- `reports/feature_engineering_closure/full_extraction_receipt.json`, logs y auditoría final.

Las rutas de tables anteriores son relativas a `reports/feature_engineering/`.
Todas las salidas reales permanecen locales e ignoradas. El notebook fuente
versionado solo contiene narrativa y código.

## Validación y límites

```powershell
uv run pytest
uv run ruff check .
uv run python scripts/check_notebook_source.py
```

Los tests offline cubren parsing temporal y ausencia de timestamps inventados,
presencia frente a instancias, huérfanas, conflictos, conservación de bytes,
empty/non-empty, revisiones de adapters, normalización, mapping canónico,
selección inequívoca de espacios completos, reanudación interrumpida y reportes.
La ejecución real del builder comprueba además el notebook con las dependencias
de reporting. CI no necesita pesos, GPU ni datos privados.

Las secuencias se derivan de nombres, no de una inspección completa de videos.
No existe timestamp verificado. Los 400 pares próximos de contenido distinto
son candidatos transparentes, no leakage confirmado. Los 8 conflictos siguen
documentados; el estado completo de features no implica anotaciones perfectas.
No se ejecutan similitud, reducción, clustering, nuevos splits ni YOLO.
