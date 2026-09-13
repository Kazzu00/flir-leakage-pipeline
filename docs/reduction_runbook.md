# Ejecución reproducible de t-SNE y PaCMAP

El [protocolo](reduction_protocol.md) define las decisiones antes del grid. Los
[resultados agregados](reduction_analysis.md) documentan la ejecución completa.
Los YAML son genéricos: el encoder y sus identidades vienen de los artefactos
explícitamente seleccionados, nunca de una ruta privada escrita en el código.

## Entorno y selección explícita

Desde la raíz del repositorio, con Python 3.11 y uv:

```powershell
uv sync --locked --extra dev --extra reduction --extra reporting
$manifest = 'data/manifests/flir_canonical_candidate_v1.parquet'
$featureDirectory = '<verified full feature directory>'
$similarityDirectory = '<matching verified full similarity directory>'
uv run flir-pipeline reduction --help
```

Sustituir los dos marcadores por las rutas locales del mismo encoder/dataset.
No basta elegir el archivo más reciente: `load_inputs` verifica cobertura completa
contra el manifest, identidades, índices, normas L2, matriz coseno, vecinos y hashes.
No se cargan modelos ni se leen ZIP para ejecutar esta fase. `vision` sigue siendo
un extra separado. El extra `reduction` incluye scikit-learn y PaCMAP fijado en
0.9.1; `uv.lock` fija las dependencias transitivas. No hay descargas durante tests.

En el entorno Windows de esta ejecución el launcher generado fue bloqueado por
el sistema. El equivalente usado, con el mismo Typer app y opciones, fue:
`uv run --no-sync python -c "from flir_pipeline.cli import app; app()" reduction ...`.
El entorno local seleccionado mediante `UV_PROJECT_ENVIRONMENT` y la caché uv
permanecen ignorados. `--no-sync` solo procede después de sincronizar los extras.

## Un run y un benchmark

```powershell
uv run flir-pipeline reduction run --feature-directory $featureDirectory --similarity-directory $similarityDirectory --manifest $manifest --config configs/reduction/tsne_research.yaml --configuration-index 0 --seed 0
uv run flir-pipeline reduction benchmark --feature-directory $featureDirectory --similarity-directory $similarityDirectory --manifest $manifest --config configs/reduction/tsne_research.yaml --config configs/reduction/pacmap_research.yaml
```

Repetir el benchmark con las rutas explícitas del otro encoder. El índice de
configuración individual es 0-based y sigue el producto cartesiano del YAML,
sin contar semillas: t-SNE 0/1/2 → perplexity 10/30/50; PaCMAP → MN 0.2/0.5/1.0.
`run --seed` controla la semilla individual. `benchmark` usa todas las semillas
del YAML y requiere exactamente 0/1/2 para seleccionar candidatos. El protocolo
principal tiene nueve runs por método y encoder, 36 en total. La API permite 3D
mediante `output_dimension: 3`, sin ejecución 3D en esta fase.

`START` y `DONE` muestran progreso por run. Se ejecuta un ajuste cada vez y se
limitan los hilos nativos. Un run terminado y verificado es el checkpoint del
grid; repetir el comando lo reutiliza. No hay checkpoint interno del optimizador.
Las publicaciones incompletas se conservan y se rechazan; no se presentan como
runs terminados. Usar otra raíz explícita para una fuente distinta con la misma
identidad matemática. No se deben escribir concurrentemente las mismas rutas.

## Verificación y artefactos

```powershell
$reductionDirectory = '<returned run or benchmark directory>'
uv run flir-pipeline reduction summary $reductionDirectory
uv run flir-pipeline reduction verify $reductionDirectory --feature-directory $featureDirectory --similarity-directory $similarityDirectory --manifest $manifest
```

`summary` verifica integridad antes de mostrar datos. `verify` sin fuentes revisa
coordenadas, índices, metadata, identidades, hashes y vecinos recalculados. Las tres
opciones de fuente, juntas, añaden cobertura canónica, comparación con entradas y
recálculo de T, C, Jaccard, Spearman y la selección determinista de pares. Un fallo
produce un exit code distinto de cero. El benchmark verifica todos sus runs y
recalcula la selección a partir de sus tablas de métricas/estabilidad.

Cada run se guarda en
`artifacts/reduction/<encoder>/<dataset_id>/<feature_space_id>/<method>/<reduction_space_id>/`.
Contiene coordenadas, content_index, quality y metadata, más vecinos reducidos,
preservación por contenido, métricas y un snapshot de la representación fuente.
El record_index original no se duplica: su huella se conserva en metadata y el
mapping sigue disponible en el directorio de features validado.

El benchmark vive bajo `.../<feature_space_id>/benchmarks/<benchmark_id>/` y guarda
`runs.csv`, estabilidad por contenido/par de semillas, agregados por k/configuración,
alternativas, candidatos, tiempos y metadata. La identidad de reducción incluye
dataset, representación, método, parámetros, semilla, salida, preprocesamiento y
versiones de implementación; excluye ruta, fecha y política de hilos. Los campos
de evaluación también versionan el artefacto para no reutilizar métricas de otro
protocolo. No se llama a esta identidad una promesa de portabilidad bit a bit.

## Reporte local

```powershell
uv run python scripts/check_notebook_source.py
uv run --extra reporting python scripts/build_reduction_review.py --dinov2 <dinov2_benchmark_directory> --clip <clip_benchmark_directory> --dinov2-similarity <dinov2_similarity_directory> --clip-similarity <clip_similarity_directory>
```

El builder comprueba los dos benchmarks y la procedencia posterior correspondiente.
Genera 11 figuras, tablas de todos los runs y un notebook/HTML de 14 secciones en
`reports/reduction/review/`. El HTML oculta código y muestra solo agregados e IDs
de espacios; no muestra content hashes ni rutas privadas. Los índices temporales
son inferidos, no segundos; la selección de ejemplo es independiente de coordenadas.
El notebook público `notebooks/reduction_review.ipynb` carece de outputs.

```powershell
uv run ruff check .
uv run pytest
uv run python scripts/check_notebook_source.py
```

Los tests son sintéticos, offline y CPU. Se contrastan fórmulas con scikit-learn,
se comprueba igualdad de runs pequeños con la misma semilla y se prueban corrupción,
alineación, selección de candidatos y metadata posterior. Datos reales, arrays,
figuras, tablas ejecutadas y HTML se mantienen ignorados por Git.
