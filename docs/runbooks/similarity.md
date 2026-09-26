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

## Video muestreado: contrato v2 y memoria acotada

Usar explícitamente las features completas y el manifest `flir_video_samples_v1`:

```powershell
uv run flir-pipeline similarity compute --feature-directory <dinov2_video_features> --manifest <video_manifest.parquet> --config configs/similarity/dinov2_video_research.yaml
uv run flir-pipeline similarity compute --feature-directory <clip_video_features> --manifest <video_manifest.parquet> --config configs/similarity/clip_video_research.yaml
uv run flir-pipeline similarity verify <video_similarity> --feature-directory <video_features> --manifest <video_manifest.parquet>
uv run flir-pipeline similarity summary <video_similarity>
```

Estos comandos son instrucciones para la futura ejecución en Hypatia, no evidencia
de una ejecución real local. Las configuraciones históricas conservan v1; usarlas
con el manifest de video produce un error explícito. No se descargan modelos ni
se requiere GPU. Si Windows bloquea el launcher, usar el equivalente `uv run python
-c "from flir_pipeline.cli import app; app()" similarity ...` en una sola línea.

V2 conserva `cosine_similarity.npy`, `nearest_neighbors.parquet`,
`content_index.parquet`, `content_provenance.parquet`, `record_provenance.parquet`,
`topk_content_summary.parquet`, `near_unit_pairs.parquet`, snapshots JSON y metadata.
Las tablas son `source_video_similarity.csv`, `sample_index_gap_similarity.csv`,
`timestamp_gap_seconds_similarity.csv`, `quantile_candidates.csv` y
`topk_global_summary.csv`. Los resúmenes incluyen poblaciones/denominadores, media,
std ddof=0 y cuantiles lineales **exactos**, sin muestreo ni aproximación.
Los bins temporales son `(límite anterior, límite actual]`, con primer bin cero;
samples y segundos se declaran por separado. FPS procede de cada video.

`pair_storage: summary_only_v1` es el default explícito de estos YAML: no se escribe
`pair_analysis.parquet`. Para requerirlo, copiar el YAML a una configuración local
y elegir `full_streamed_v1`; cambia el ID del experimento y escribe por bloques.
La verificación entiende ambos contratos. No necesita cargar todos los pares en
pandas, ni siquiera para verificar el archivo opcional o un near-unit masivo.
No borrar parciales para forzar una reejecución; inspeccionarlos y usar otra raíz
cuando no exista una publicación completa válida. Un solo writer por directorio.

Para N=8093, la matriz float32 ocupa aproximadamente 262 MB decimales
y los 32744278 pares usan aproximadamente 688 MB en cuatro arrays compactos.
Los espacios temporales de cálculo de estadísticas se liberan entre estratos;
top-k ordena bloques de 64 filas. Esta es una estimación del diseño, no un MaxRSS
observado: medir memoria/tiempo del compute y de verify en Hypatia. El trabajo
sigue siendo cuadrático; la verificación exacta repite el análisis. Si todos los
pares son near-unit, ese Parquet puede seguir siendo grande, aunque su escritura
y lectura estén acotadas. La cota operativa no cubre todos los algoritmos downstream.

Validación local de escala, **solo sintética**: 8093 vectores normalizados de 8
dimensiones y 9648 occurrences procesaron los 32744278 pares sin full pair Parquet.
La verificación contra embeddings y manifest sintéticos pasó completamente.
Se observó un peak working set de 1971113984 bytes (1,97 GB decimales) en Windows;
compute con verificación interna tardó 119,24 s y la verificación adicional contra
fuentes 88,42 s. La suite de tests estaba ejecutándose concurrentemente: estos
tiempos son diagnósticos locales, no un benchmark de encoders ni una predicción
de Hypatia. El recibo y los inputs sintéticos se conservaron fuera del repositorio.
Los 8093 contenidos FLIR reales no fueron leídos ni validados en esta tarea.

La revisión repitió el ensayo con 8093 vectores **sintéticos** de 512 dimensiones,
IDs de 64 caracteres, 9648 occurrences y FPS de 0,5/2/3 por fuente; BLAS limitado
a un hilo. En la ruta de cómputo revisada se observaron 2056105984 bytes de peak working set
(2,056 GB) y 2734014464 bytes de peak commit (2,734 GB; `peak_pagefile` en Windows).
Compute con verificación interna tardó 116,73 s; la verificación adicional contra
fuentes, 90,10 s. Todas las comprobaciones pasaron, sin full pair Parquet.
La suite de tests también corría concurrentemente. Estos datos corroboran el
orden de magnitud del ensayo anterior; no miden Hypatia ni constituyen resultados
de similitud del dataset real. Los recibos sintéticos permanecen fuera del repo.

Revisión del presupuesto de memoria: los 688 MB son scores float32 (131 MB),
membership bool (33 MB), gaps int64 (262 MB) y gaps float64 (262 MB). Con la matriz
suman aproximadamente 950 MB. Un estrato grande puede añadir una selección de
scores de 131 MB, una conversión float64 de 262 MB y otra copia de trabajo de
cuantiles de 262 MB; las máscaras también ocupan hasta 33 MB cada una.
`matrix_quality` usa operaciones NumPy matriciales temporales; verificar contra
features recalcula una matriz de 262 MB después de liberar los arrays de pares.
Top-k limita la ordenación a 64 × N; Parquet escribe hasta N−1 filas por bloque y
verifica hasta 65536 por batch, conservando metadata de row groups, no todos sus
datos. No hay DataFrame completo de pares, índices triangulares globales, tensor
N×N×D ni join cartesiano de occurrences en v2.

Los embeddings de 384/512 dimensiones requieren aproximadamente 12,4/16,6 MB
por array float32, frente a 0,26 MB en la prueba anterior de 8 dimensiones.
IDs SHA256 largos, Arrow, BLAS, hilos y retención del allocator añaden memoria.
Con el `top_k=20` de estas configuraciones, un presupuesto orientativo de 2–4 GB
por proceso de similitud resulta coherente
con estos buffers; no es una medición de Hypatia ni una cota formal del RSS.
Asignar 8 GB al primer ensayo CPU deja margen para verificarlo allí.

Esta estimación no se extiende a reducción/clustering: el código histórico de
reducción conserva ordenaciones/ranks densos y enumera el triángulo para seleccionar
su muestra; `clustering.load_family` retiene siete matrices float64 de N×N
(original más dos métodos × tres semillas), aproximadamente 3,67 GB por encoder.
Dos familias suman 7,34 GB solamente en distancias, antes de matrices coseno,
algoritmos y temporales. Un cluster que contenga casi toda la población también
activa índices triangulares grandes en su evaluación. Estas limitaciones heredadas
requieren dimensionar cada etapa; la revisión no cambia esos protocolos.

`source video != sequence` y `relative sampling-grid timestamp != capture timestamp`.
La auditoría comprueba consistencia del manifest previamente construido; no vuelve
a verificar el reloj de captura ni a leer los videos. Cada content conserva todas
sus occurrences. Los mínimos de índice y segundos se calculan independientemente
entre occurrences del mismo video; conjuntos fuente disjuntos tienen gaps null.
Proximidad o cosine alto no confirman leakage ni escenas por sí solos.

El cache exige política/versionado, firmas de features/procedencia y verificación
completa vinculada a features y manifest también al reutilizarlo desde compute.
Cambiar tiempo, FPS o source membership invalida su firma incluso si los
vectores son iguales. No cambia los IDs existentes de dataset, features o contenido.
Reducción y clustering no requieren el full pair Parquet. Los builders históricos
de HTML/ZIP/secuencia rechazan v2 expresamente; usar summary/CSV para revisar este
dataset hasta implementar un reporte de video independiente.

La verificación v2 exige el esquema Parquet incluso en cohortes vacías y recalcula
las relaciones/resúmenes. Con fuentes compara además el snapshot matemático,
el content index y el mapping de occurrences con el feature store, comprueba raw/L2
y, con manifest, exige cobertura completa/revisión resuelta. Un flag de metadata
o un checksum actualizado no reemplaza estas comprobaciones. La auditoría rechaza
tiempos relativos negativos y declaraciones de labels/splits/grupos incompatibles
con el manifest de video v1, en vez de descartarlas silenciosamente.
