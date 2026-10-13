# Splits candidatos etiquetados v1

## Alcance y contrato

`splitting labeled-candidates` consume una publicación completa de `audit-adendo`.
Construye y evalúa particiones candidatas; no adjudica procedencia, extrae features,
entrena modelos ni consume los 48 experimentos históricos del detector.
La publicación de auditoría y los ZIP permanecen intactos; `split_ready=false`
en el recibo original conserva su significado y no se reescribe.

El adaptador verifica los checksums de todos los archivos declarados, recalcula
dataset/audit IDs y compara los manifests JSON/Parquet. Reconstruye las cajas desde
los bytes originales de cada etiqueta, comprueba SHA256, validación, clases,
emparejamiento bidireccional, conteos y cobertura de relaciones exactas.
Las etiquetas inválidas, ambiguas o incompletas bloquean la preparación.
Los huérfanos se validan pero no generan ocurrencias ficticias.
Los conflictos entre anotaciones de contenidos exactos se preservan por ocurrencia.

`--expected-dataset-id` y `--expected-audit-id` fijan una publicación aceptada.
`--source-root` verifica también todos los ZIP originales por streaming SHA256;
si la auditoría tiene evidencia importada, exige `--evidence` para verificarla.
Sin esa opción se valida el snapshot auditado; no se afirma haber releído los ZIP.
Los hashes históricos de la implementación productora identifican esa auditoría;
no se exige que el checkout actual sea idéntico al productor antiguo.

## Políticas y poblaciones

| Política | Unidad y asignación | Interpretación |
|---|---|---|
| S1 | Contenido exacto; permutación con semilla y cortes por ocurrencias | Baseline aleatorio, sin balance por clases |
| S2 | Contenido exacto; MILP existente | Objetivo L1 normalizado de ocurrencias, cinco clases y etiquetas vacías |
| S3 | Componentes conexas de contenido exacto y estados seleccionados; mismo MILP | Sensibilidad a restricciones hipotéticas, no clustering de embeddings ni procedencia confirmada |

`historical` conserva todas las ocurrencias históricas; `expanded` agrega Adendo
con `original_split=null`. Ningún duplicado exacto se elimina. Los ratios explícitos
se toman del YAML; con `target_ratios: null` se calculan **solo sobre el histórico**.
No se fija a cada ocurrencia su split original durante la optimización.

Las relaciones de S3 se seleccionan por `sensitivity_states` y ambos endpoints
deben pertenecer a la población. En particular, el histórico no hereda uniones
a través de Adendo excluido. Se reportan evidencias excluidas, parejas por
ocurrencia y parejas por contenido; múltiples evidencias no cuentan como parejas
independientes. Los labels enteros usados por el verificador existente son
identificadores internos de componentes, no resultados de clustering científico.

Una revisión `rejected` veta toda arista positiva para esa pareja de contenidos.
Si coexiste con `confirmed_provenance`, la política predeterminada `error` detiene
la construcción, también en S1/S2. La alternativa explícita `reject_pair` excluye
esa pareja y registra el conflicto. Un rechazo que contradice identidad exacta
siempre falla. S3 también rechaza una unión transitiva que reincorpora una pareja
rechazada; el rechazo no se traduce en obligación de asignarla a splits distintos.
No se cambian estados ni flags en `relations.json`.

El MILP conserva límites deterministas de nodos y brecha; no usa tiempo como
criterio de parada. Se registran estado, mensaje, factibilidad primal comprobada,
objetivo, cota dual, brecha y límites. Sin incumbent factible no hay fallback.
`optimal_within_tolerance` no demuestra óptimo único. La verificación recalcula
el objetivo desde las membresías. Los objetivos de balance son blandos; no se
garantiza igualdad perfecta ni presencia de todas las clases en cualquier entrada.
`max_relative_record_deviation` es un gate opcional posterior, **no** una restricción
adicional dentro del MILP. Su incumplimiento no prueba inviabilidad matemática.

## Ejecución

Config versionada: [`labeled_candidates_v1.yaml`](../../configs/splits/labeled_candidates_v1.yaml).
Usar directorios nuevos fuera del repositorio y de la auditoría. Ejemplo PowerShell
con variables locales, sin versionar rutas privadas ni IDs de datos reales:

```powershell
$audit = Join-Path $env:FLIR_AUDIT_ROOT 'labeled_consolidation_v1/acceptance-20261012-a'
$destination = Join-Path $env:FLIR_AUDIT_ROOT 'labeled_splits_v1/new-pilot'
uv run flir-pipeline splitting labeled-candidates `
  --audit $audit --config configs/splits/labeled_candidates_v1.yaml `
  --population both --policy all --seeds 0 `
  --expected-dataset-id $acceptedDatasetId --expected-audit-id $acceptedAuditId `
  --source-root $env:FLIR_DATA_ROOT --evidence $evidencePath --output $destination
uv run flir-pipeline splitting verify-labeled $destination --audit $audit --replay
```

`FLIR_AUDIT_ROOT` en este ejemplo es una variable del usuario para el directorio
de auditorías; la CLI no la resuelve automáticamente. Para cinco semillas usar
`--seeds 0,1,2,3,4` y otra salida nueva. `--population` acepta `historical`,
`expanded`, `both`; `--policy` acepta `S1`, `S2`, `S3`, `all`.
Si Windows bloquea el launcher, la misma CLI puede invocarse con
`uv run --no-sync python -c "from flir_pipeline.cli import app; app()" splitting ...`,
sin desactivar controles del sistema.

## Salidas y verificación

Cada ejecución contiene:

- `source_audit/`: snapshot completo ligado por checksums; incluye anotaciones
  privadas originales, por lo que tampoco debe versionarse.
- `configuration.json`, `provenance.json`: configuración, versiones, commit,
  estado dirty, fingerprints del código y alcance de la comprobación de fuentes.
- `historical_reference.json`: descripción del histórico original contaminado,
  conteos de clases y cruces exactos; no es un candidato nuevo.
- `runs/<split_id>/`: membresías por ocurrencia y contenido, grupos atómicos,
  balance de grupos/clases, métricas y metadata del solver.
- `comparison.csv`: tabla por población, política y semilla.
- `summary.json`: recibo final, escrito solo después de verificar los archivos
  persistidos. Sin recibo final, la salida es incompleta y no debe consumirse.

La identidad de cada run incluye dataset, población, auditoría, fuentes, política,
ratios, semilla, configuración e implementación/versiones. La lista de semillas
de la ejecución no altera una identidad individual. `membership_id` representa
la asignación ordenada por ocurrencia; permite comparar membresías entre políticas
sin confundir sus diferentes identidades experimentales. Paths, fechas y duración
del solver son operacionales y no cambian esas identidades.

`verify-labeled` reconstruye grupos desde relaciones, valida conservación de
ocurrencias, propagación, linaje completo y atomicidad; recalcula todas las métricas,
referencia histórica y CSV incluso cuando se han reparado los checksums.
`--audit` liga nuevamente la fuente externa. `--replay` vuelve a ejecutar las
asignaciones y compara membresías, exigiendo el código/versiones registrados.
El replay no se sustituye por igualdad de timestamps o del tiempo del solver.

No se producen métricas coseno nuevas ni métricas temporales: nombres `scene`,
índices de frames y pHash no son verdad temporal. Un cero de cruces candidatos en
S3 verifica la hipótesis impuesta; no demuestra ausencia general de leakage.

## Validación local

Pruebas offline, sintéticas y sin modelos:

```powershell
uv run --no-sync ruff check .
uv run --no-sync pytest tests/test_labeled_splits.py tests/test_splitting.py tests/test_adendo.py tests/test_manifest.py tests/test_inventory.py tests/test_identity.py -q --basetemp $externalFreshTemp
```

Se usa un directorio temporal externo para respetar las barreras de publicación
privada del proyecto. Esta selección cubre el adaptador y regresiones del motor
histórico, identidades, manifest, inventario y auditoría. No equivale a la suite
global de extras experimentales ni revalida los 48 runs del detector.

Resultados de ejecución real y límites: [estado vigente](../status.md).
