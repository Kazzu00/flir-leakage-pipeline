# Reporte final del detector y contrato de frontend

## Estado y dos entornos

Esta etapa implementa software de reporte; no acredita por sí misma la ejecución
del experimento. El responsable reporta **48/48 runs completos en Hypatia**, con
16 splits, cuatro estrategias y tres detector seeds; no queda entrenamiento
pendiente según ese reporte. **No se revalidaron esos resultados en el clon local**.
El portátil conserva un protocolo anterior y pilotos pequeños; no debe
actualizarse ese freeze ni reconstruirse la evidencia para simular el experimento.

El ciclo es: piloto pequeño de infraestructura → Stage A (celdas reutilizables
una vez) → matriz controlada completa → análisis descriptivo verificado → export.
`detection report` solo realiza las dos últimas operaciones. Nunca entrena ni
regenera features, similitud, reducción, clustering, splits o detector runs.

## Ejecución posterior en Hypatia

Transferir primero los commits de código y actualizar el paquete del entorno.
No hace falta instalar torch/Ultralytics para ejecutar este reporte; el
verificador existente contrasta métricas con estadísticas capturadas y checksums.
Se requieren las dependencias core de `pyproject.toml`.

```bash
module purge
module load python/3.11
source ~/.venvs/flir-detection/bin/activate

cd ~/flir-leakage-pipeline
# Si el entorno no tiene instalado el checkout actualizado:
uv pip install --python "$VIRTUAL_ENV/bin/python" -e .

flir-pipeline detection report \
  --plan artifacts/detection/protocol \
  --artifacts artifacts/detection \
  --split-root artifacts/splitting/runs \
  --analysis-output artifacts/detection/final_report \
  --frontend-output exports/frontend/detection
```

El comando usa el clon `~/flir-leakage-pipeline` del usuario de Hypatia, sin
versionar su ruta absoluta privada. Los valores por defecto permiten simplemente
`flir-pipeline detection report`.
`--associations` admite otro archivo de registro validado; el contrato v1 admite
las estrategias historical/random_content/C10/C12 y las cinco clases canónicas.
Identidades, tamaño de matriz, seeds y parámetros se derivan de las fuentes;
no hay IDs, métricas ni un requisito de 48 celdas hardcodeados en el productor.

El consumidor necesita `protocol/`, `runs/` **y `views/`**: `verify_run` utiliza
los recibos de materialización y `records.parquet` para verificar el conjunto de
test. Deben conservarse los archivos de los runs que sus metadata vinculan,
incluidos pesos y estadísticas, aunque nunca se exportan al frontend. La fuente
autoritativa del contexto residual es el plan congelado. Si existe un split bajo
`--split-root`, sus metadata y cohortes opcionales se verifican contra el plan;
su ausencia no inventa fracciones ni invalida el contexto ya congelado.

Un gate incompleto, corrupto, duplicado o incompatible devuelve código 2 y no
publica resultados. Los pilotos se excluyen: si reemplazan una celda esperada,
la matriz permanece incompleta. Un cambio de fuentes durante el reporte impide
la publicación. El comando nunca convierte el freeze local al de Hypatia.

## Salidas y publicación

En `artifacts/detection/final_report/` se generan CSV **y Parquet** para:

- `run_level_metrics`: run × población (overall y cinco clases), con intervalos existentes;
- `split_level_metrics`: una fila overall por split;
- `split_class_metrics`: métricas promediadas por split y clase;
- `strategy_level_metrics` y `class_level_metrics`: medias entre splits;
- `test_support`: instancias por split y clase, sin multiplicarlas por detector seeds;
- `variance_summary` y `prespecified_association_results`;
- `bootstrap_intervals`: intervalos almacenados por run, clase y métrica.

También se generan `REPORT.md`, `receipt.json` y seis figuras PNG/SVG en `figures/`:
`A_overall_performance`, `B_per_class_performance`, `C_variability`,
`D_test_support`, `E_prespecified_associations`, `F_temporal_at5`.
El panel temporal solo existe si el registro contiene una asociación temporal.
Ningún gráfico añade regresiones o asociaciones post-hoc.

El recibo local liga las fuentes consumidas, el código, commit, estado dirty,
versiones, gate y checksums de ambos destinos. El manifiesto público conserva
IDs experimentales y hashes de plan/freeze/registro y archivos exportados; no
publica hashes de imágenes, rutas privadas ni archivos pesados.

Los destinos deben ser disjuntos entre sí y de las fuentes. La generación ocurre
en directorios temporales hermanos y se publica después de validar todo. Ante
fallos ordinarios de escritura se restauran ambos destinos previos. Ejecutar
en directorios dedicados: archivos ajenos al reporte impiden reemplazarlos. Usar
**un solo escritor** por par de destinos. Un corte de energía/terminación abrupta
entre renames no es una transacción entre sistemas de archivos: inspeccionar los
directorios `.final_report-stage-*`/`.detection-stage-*` y sus backups antes de
reintentar, sin borrar evidencias científicas.

## Contrato `detection-export-v1`

Se publica en `exports/frontend/detection/`, fuera de `artifacts/`. No contiene
resultados reales hasta ejecutar el comando con evidencia completa. Este cambio
local versiona solamente el schema, código y documentación, sin summary ficticio.

| Archivo | Unidad y función |
|---|---|
| `manifest.json` | Estado COMPLETE, identidades, conteos, procedencia, limitaciones y hashes de los otros archivos |
| `summary.json` | Entrada principal: protocolo, métricas por estrategia, variabilidad, asociaciones y semántica |
| `runs.json` | Una fila overall por run científico; tiempos/best_epoch/memoria ausentes → null |
| `splits.json` | Una fila por split tras promediar detector seeds; contexto residual congelado |
| `strategies.json` | Resumen con igual peso para cada split; SD entre splits con ddof=1 |
| `classes.json` | Media por estrategia y clase tras promediar primero detector seeds |
| `support.json` | Instancias de test por split y clase, no imágenes independientes |
| `variance.json` | Media de SD entre detector seeds, SD entre splits y su razón |
| `associations.json` | Solo registro YAML; puntos por split, r/ρ globales, r centrado y correlaciones por estrategia |
| `bootstrap.json` | Intervalos **existentes**, por run/clase/métrica; nunca muestras bootstrap |

`bootstrap.json` amplía la estructura propuesta para que el frontend pueda
presentar intervalos sin acceder a fuentes ni recalcularlos. El protocolo de
bootstrap está en `summary.protocol`. No se promedian intervalos por estrategia.

El schema se genera desde los modelos Pydantic del productor y se copia a
`schema/detection-export-v1.schema.json`. Para validar con JSON Schema Draft
2020-12, construir el objeto lógico `{stem_del_archivo: JSON_parseado}` con los
diez archivos de la tabla. El schema exige todas las formas, campos, estrategias,
clases, rangos y null para la SD de una estrategia con un solo split. El productor
añade validaciones de consistencia entre archivos mediante `DetectionExport`.
Tests comparan el schema versionado con el generado y validan el bundle con un
validador JSON Schema independiente (`jsonschema`, extra dev).

## Cálculos y límites científicos

Se reutilizan `load_evidence`, `load_registry`, `prespecified_table` y
`aggregate_runs`. El reporte anterior de progreso (`reporting.py` y figura 09)
permanece compatible; no es el contrato final ni su unidad de asociación.

Para cada asociación se promedian las detector seeds del mismo split y se
preserva un único valor de contexto. Pearson/Spearman globales usan pares
válidos; Spearman usa ranks medios ante empates. El centrado resta las medias
por estrategia de x e y sobre esos mismos pares, seguido de Pearson conjunto.
Histórico aporta un punto global y un residuo centrado cero; no tiene correlación
propia ni variación entre splits. Menos de tres pares o una variable constante
producen null. `n_splits` cuenta todos los puntos; `n_valid_splits` cuenta pares
definidos. Nunca se omiten silenciosamente puntos con datos ausentes.

La interpretación exporta el cambio exacto en |r| y si disminuye al centrar,
sin fijar un umbral post-hoc de significancia. Las razones de variabilidad son
descriptivas, no estimaciones de un modelo de componentes de varianza. Si el
denominador es cero/no definido, la razón es null. Histórico u otra estrategia
con un solo split conserva SD entre splits y razón null, nunca cero.

Todos los outputs conservan que la comparación es descriptiva, no causal;
composición, dificultad y dependencia pueden variar conjuntamente. No se afirma
que C10/C12 eliminen leakage ni que los clústeres sean secuencias ground truth.
`temporal_at5`/Δ≤5 es distancia de índices, no segundos. El bootstrap por imagen
puede subestimar incertidumbre cuando persiste dependencia entre frames.

Las arrays tienen orden estable por estrategia/split/seed/clase o association_id.
No se usa jitter aleatorio; el SVG fija su hash salt y omite fecha. Los timestamps
de manifiesto/recibo cambian; los valores científicos se reproducen con las
mismas fuentes, registro y entorno. No se garantiza igualdad binaria de figuras
entre distintas versiones de matplotlib o fuentes del sistema.

## Integración posterior

1. Desarrollar y validar software con fixtures offline en el clon local.
2. Transferir/actualizar código en Hypatia sin sustituir fuentes congeladas.
3. Ejecutar el comando y revisar gate, recibo, tablas, figuras y limitaciones.
4. Contrastar allí las cifras de referencia comunicadas por el responsable; no son inputs del software.
5. Revisar Git y versionar solo código y export ligero. El commit/push de Hypatia es una operación posterior; este trabajo local no hace push ni merge.
6. Sincronizar el contrato a `flir-pipeline-explorer`, que verifica versión/hashes y solo presenta los valores. No recalcula correlaciones, bootstrap ni agregaciones, ni infiere estado científico desde archivos crudos.

## Validación local de la implementación (2026-10-03)

- `uv run --no-sync python -m pytest -q --basetemp C:/flir-report-tests-20261003`: **804 passed**.
- Bloque detector (tests nuevos, `test_detector_association.py`, `test_detection.py`): **52 passed**; tras los últimos ajustes de figuras/procedencia, los **24 tests nuevos** pasaron nuevamente.
- `uv run --no-sync ruff check .`: aprobado; formato de los cuatro archivos Python nuevos: aprobado.
- `uv run --no-sync python scripts/check_notebook_source.py`: siete notebooks fuente válidos.
- CLI `detection report --help`, rechazo de evidencia incompleta, schema independiente, hashes de fuentes sintéticas y `git diff --check`: aprobados.

Se usó `UV_PROJECT_ENVIRONMENT=.venv-review`. Windows bloqueaba DLLs en el entorno
original; el entorno de revisión utilizó versiones compatibles permitidas por
las restricciones del proyecto: NumPy 2.3.5, pandas 3.0.5, PyArrow 19.0.1,
scikit-learn 1.6.1, PaCMAP 0.9.1, Numba 0.63.1, llvmlite 0.46.0,
Pydantic 2.13.5, matplotlib 3.11.1, pytest 9.1.1 y jsonschema 4.26.0.
No se deshabilitaron políticas del sistema. La primera suite con temporales
dentro de una ruta larga se detuvo tras reproducir un fallo preexistente por
límite de longitud de ruta; la repetición con ruta corta pasó completa.
La ejecución local usa estas versiones, no acredita el entorno exacto de Hypatia
ni sustituye la CI Linux con `uv sync --locked`. Solo `jsonschema` se añadió al
extra dev del proyecto; las elecciones locales de versiones no alteraron el lock.
