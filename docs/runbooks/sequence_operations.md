# Operación de la suite de secuencias en Hypatia

La orquestación reutiliza las funciones de la suite científica. No cambia
algoritmos, semillas, máscaras, umbrales ni la unidad `content_id`. El perfil
predeterminado conserva 154 fits. No se ha ejecutado esta matriz real desde este
checkout; las pruebas son sintéticas, offline y sin modelos.

**Ingesta nativa implementada; ejecución real pendiente.** El adaptador
`hypatia_legacy_evidence_v1` consume los cuatro contratos observados que aportó
el responsable. No exige sidecars, reviewer ni fecha de revisión inexistentes.
La validación local usa fixtures sintéticas con los esquemas/cardinalidades
observados; no constituye una importación real de Hypatia. Véanse las
[validaciones y comandos de ingesta nativa](native_sequence_evidence.md).

## Sincronizar código sin commit ni push

Los cambios locales no llegan a Hypatia mediante `git pull` mientras no estén
publicados. Para transportar este checkout sin datos, empaquetar solamente código,
configuración, tests y documentación. En PowerShell, configurar
`FLIR_HYPATIA_SSH` con el destino SSH explícito autorizado y ejecutar:

```powershell
if (-not $env:FLIR_HYPATIA_SSH) { throw 'Falta FLIR_HYPATIA_SSH' }
$sequenceCodeBundle = Join-Path $env:TEMP 'flir-sequences-code.tar'
tar --exclude=__pycache__ --exclude=*.pyc -cf $sequenceCodeBundle src configs scripts tests docs README.md pyproject.toml uv.lock
if ($LASTEXITCODE -ne 0) { throw 'Falló el empaquetado de código' }
scp $sequenceCodeBundle "${env:FLIR_HYPATIA_SSH}:flir-sequences-code.tar"
```

En Hypatia, desde la raíz del repositorio, inspeccionar el estado y los miembros
del bundle antes de aplicar el código. La copia numerada conserva las versiones
anteriores de los archivos de código; no se extraen `reports/`, `artifacts/`,
datasets, `.env` ni caches. No elimina archivos ni realiza operaciones Git:

```bash
git status --short
tar --list --file "$HOME/flir-sequences-code.tar"
tar --extract --backup=numbered --no-same-owner --file "$HOME/flir-sequences-code.tar"
```

Este transporte no instala dependencias ni valida artefactos. El entorno remoto
debe estar preparado como se describe a continuación. No se ejecutó esta copia
desde el checkout local.

## Preparación e identidad de entradas

Desde la raíz del repositorio, activar el entorno ya preparado con core y el
extra `reduction`. Los scripts utilizan ese `python`; `FLIR_PYTHON` permite
seleccionar explícitamente otro intérprete. No instalan paquetes ni descargan
modelos. No ejecutar el comando serial `suite` en el nodo de login.

El bloque opcional `inputs` del perfil acepta `manifest`, `clip_features`,
`dinov2_features`, `images_root`, `images_archive` y `search_roots`.
Las rutas son relativas al directorio de trabajo del repositorio. Los flags
explícitos prevalecen sobre variables de entorno, y estas sobre el perfil.
Variables: `FLIR_MANIFEST`, `FLIR_CLIP_FEATURES`, `FLIR_DINOV2_FEATURES`,
`FLIR_IMAGES_ROOT` o `FLIR_IMAGES_ARCHIVE` (solo una fuente de imágenes).
No hay IDs de datasets o feature spaces fijados en los scripts.

En ausencia de una ruta de features se buscan metadatos bajo `search_roots`
(por defecto `artifacts`). Se valida la identidad del dataset, cobertura completa,
índices, raw/L2, revision resuelta y representación matemática mediante los
validadores existentes. Un smoke declarado se excluye; un store completo corrupto
se rechaza. Dos stores compatibles requieren selección explícita. La búsqueda
de manifest tampoco elige por fecha o nombre de dataset.

Planificación/importación realizan lectura y validación de entradas; no ejecutan
reducción, clustering, búsqueda de vecinos ni generación de imágenes. La validación
sí puede leer Parquet/features; ejecutarla dentro de una asignación SLURM, como
en los comandos siguientes. Los cálculos experimentales están en workers SLURM.
Las rutas operacionales, checksums y feature
spaces resueltos quedan en `slurm_run.json`; nunca se versiona ese archivo real.

## Flujo y revisión

Con las entradas resueltas una vez mediante entorno o configuración:

```bash
EVIDENCE=$(srun --cpus-per-task=1 --mem=4G --time=00:15:00 \
  flir-pipeline sequences experiment import-real-evidence \
  --family video_11min --root reports --dataset-variant unspecified)

srun --cpus-per-task=1 --mem=4G --time=00:15:00 \
  bash scripts/hypatia/sequence_suite_submit.sh \
  --family video_11min --profile configs/hypatia_sequence_experiments.yaml \
  --dataset-variant unspecified --evidence "$EVIDENCE" --dry-run

srun --cpus-per-task=1 --mem=4G --time=00:15:00 \
  bash scripts/hypatia/sequence_suite_submit.sh \
  --family video_11min --profile configs/hypatia_sequence_experiments.yaml \
  --dataset-variant unspecified --evidence "$EVIDENCE"

srun --cpus-per-task=1 --mem=4G --time=00:15:00 \
  bash scripts/hypatia/sequence_suite_status.sh <run_id>
```

`dry-run` imprime entradas, comandos, recursos, dependencias y ubicaciones
esperadas sin escribir el directorio de ejecución ni llamar a `sbatch`. El `srun`
exterior reserva recursos para validar entradas; no envía el DAG experimental.
Los recursos/partition/account de esa asignación se ajustan según el sitio.
La submission devuelve un solo `run_id` y el mapping etapa→job ID.
Los IDs content-addressed de publicaciones se conocen al terminar cada etapa;
sus rutas exactas quedan en los recibos, sin copiarlos entre comandos.

El DAG ejecuta independientemente los grids temporales, cada representación por
encoder/semilla y la recurrencia directa. Cada representación se calcula una sola
vez y alimenta trabajos separados por algoritmo con su grid de configuraciones.
Agregación, evaluación, transiciones y contraste de recurrencia usan `afterok`.
Una celda científica fallida provoca fallo del worker y bloquea sus dependientes.

El último trabajo genera `sequence_final_summary_v1`: cobertura de clustering,
resultados temporales y de estabilidad, candidaturas recurrentes, concordancia
entre encoders y entre recurrencia directa/clustering, y estado de revisión.
Los detalles acompañan al JSON en Parquet. Mantiene explícitamente
`visual_dependency_groups_created=false`, `split_created=false` y
`leakage_safe_split_exists=false`.

El checkpoint genera un único `sequence_review_package_v1` inmutable: zonas,
pares, matches CLIP/DINOv2, desacuerdos, medoids, contexto temporal y tablas de
soporte por clustering. Abrir la ruta `review_package/media/index.html` mostrada
por status. Copiar `decisions_template.csv` fuera del paquete y completar las
filas efectivamente revisadas. No editar archivos de la publicación.

```bash
REVIEW=$(srun --cpus-per-task=1 --mem=4G --time=00:15:00 \
  flir-pipeline sequences experiment review-import \
  <review_package> <decisiones.csv> --output reports/sequence_review)
srun --cpus-per-task=1 --mem=4G --time=00:15:00 \
  bash scripts/hypatia/sequence_suite_status.sh <run_id> --review "$REVIEW"
```

La revisión requiere pertenencia al paquete y replay válido del historial. Una
revisión parcial mantiene `manual_review_required=true`; todas las consultas
deben tener una decisión válida para liberar el guard de revisión. `ambiguous`
es una decisión de revisión, no una confirmación de dependencia. Esta fase no
implementa consumidores que creen VDGs, fronteras exactas ni splits. El summary
original permanece como snapshot inmutable del checkpoint inicial.

## Estado y recuperación conservadora

Cada ejecución tiene `reports/sequence_experiments/<run_id>/slurm_run.json`,
`receipts/`, `logs/` y `products/`. Se registran comandos exactos, recursos,
dependencias, IDs y la historia de submission. `status` consulta `sacct`/`squeue`
y verifica publicaciones y recibos. Un job COMPLETED sin recibo válido no cuenta
como completado. La falta de accounting se declara; no implica éxito.

```bash
srun --cpus-per-task=1 --mem=4G --time=00:15:00 \
  bash scripts/hypatia/sequence_suite_resubmission_plan.sh <run_id>
```

Este comando produce un plan determinista; no envía ni cancela trabajos.
`reuse_verified` nunca debe recalcularse. `resubmit_after_dependencies` identifica
etapas ausentes/fallidas; sus dependencias necesitan los IDs de los nuevos jobs
en una eventual reanudación. Un job pendiente con una dependencia fallida debe
reconciliarse antes para evitar duplicados. Submissions de aceptación incierta,
locks de writers interrumpidos y publicaciones sin recibo requieren inspección.
Corrupción o cambios de fuentes/código/dependencias impiden reutilizar el run.
No se borran resultados parciales ni publicaciones para forzar una reanudación.

`--run-root` permite consultar ejecuciones cuya raíz operacional fue cambiada
en el perfil. Los recursos por defecto son un presupuesto inicial pendiente de
medición real, no una garantía de consumo o tiempo.

## Variantes de dataset y comparación futura

Infraestructura genérica: una variante identifica una colección y su definición,
no presupone una transformación particular. El dataset real sin HUD sigue
**pendiente de evaluación**, después de validar el pipeline base. El siguiente
ejemplo describe trabajo futuro; no registra una ejecución ni un efecto observado:

```text
original_with_hud ─┐
                  ├─ mismo protocolo experimental ─ compare-variants
no_hud ───────────┘
```

Cada colección requiere su manifest propio con hashes del contenido actual y
ocurrencias preservadas. Registrar la variante no transforma imágenes. `--parent`
referencia el `variant.json` de origen y `--definition` acepta un JSON con la
transformación/preparación declarada. Esos campos afectan `dataset_variant_id`.
El SHA256 del manifest vincula la declaración a bytes concretos.

```bash
BASE=$(flir-pipeline sequences experiment register-variant \
  --manifest "$BASE_MANIFEST" --variant-name original_with_hud)
VARIANT=$(flir-pipeline sequences experiment register-variant \
  --manifest "$VARIANT_MANIFEST" --variant-name no_hud --parent "$BASE")

# Ejecutar para cada colección y cada encoder con sus propias imágenes.
flir-pipeline features extract --manifest "$BASE_MANIFEST" \
  --images-root "$BASE_IMAGES" --extractor clip --variant-spec "$BASE"
flir-pipeline features extract --manifest "$VARIANT_MANIFEST" \
  --images-root "$VARIANT_IMAGES" --extractor clip --variant-spec "$VARIANT"
# Repetir ambas extracciones con --extractor dinov2.

# Mantener el mismo perfil científico; resolver los stores correspondientes.
flir-pipeline sequences experiment suite --profile configs/hypatia_sequence_experiments.yaml \
  --manifest "$BASE_MANIFEST" --dataset-variant original_with_hud \
  --clip-features "$BASE_CLIP" --dinov2-features "$BASE_DINO"
flir-pipeline sequences experiment suite --profile configs/hypatia_sequence_experiments.yaml \
  --manifest "$VARIANT_MANIFEST" --dataset-variant no_hud \
  --clip-features "$VARIANT_CLIP" --dinov2-features "$VARIANT_DINO"

flir-pipeline sequences experiment compare-variants "$SUITE_A" "$SUITE_B"
```

Las rutas impresas por `suite` identifican las publicaciones completas `SUITE_A`
y `SUITE_B`. En SLURM, `sequence_suite_submit.sh --dataset-variant <nombre-o-id>`
acepta el mismo selector; también se admite `inputs.dataset_variant` en el perfil
o `FLIR_DATASET_VARIANT`. Los stores nuevos se publican en
`<output>/<encoder>/variants/<dataset_variant_id>/<feature_space_id>`.
Un cambio de colección conserva el ID matemático del encoder si modelo,
preprocesamiento, pooling y normalización coinciden, pero usa otro store y otro
checkpoint. Un store anterior sin declaración queda `unspecified`: un selector
no lo renombra ni lo convierte automáticamente en `original_with_hud`.

Para evaluar recurrencia, temporal recall y coherencia visual en ambas suites,
proporcionar a cada ejecución su `--structure` revisado y vinculado a sus fuentes.
No trasladar intervalos revisados automáticamente entre variantes. Sin esa
evidencia, la comparación indica los componentes no disponibles.

El emparejamiento es opcional y externo. Su CSV requiere estas columnas exactas:

```csv
left_frame_id,right_frame_id,mapping_method,confidence,evidence,ground_truth
occurrence-a,occurrence-b,documented_transform,1.0,transformation-log-entry,false
```

Los IDs deben existir en las suites y cada ocurrencia aparece como máximo una
vez por lado. `confidence` debe estar en [0,1]; método y evidencia no pueden
estar vacíos; `ground_truth` es una declaración booleana externa que no confirma
automáticamente dependencias. El importador incorpora los content IDs, timeline
e índices conocidos desde las ocurrencias congeladas. No afirma igualdad de bytes.

```bash
PAIRS=$(flir-pipeline sequences experiment import-correspondence \
  "$SUITE_A" "$SUITE_B" "$CORRESPONDENCE_CSV")
flir-pipeline sequences experiment compare-variants "$SUITE_A" "$SUITE_B" \
  --correspondence "$PAIRS"
```

`sequence_variant_comparison_v1` conserva identidades, checksums y espacios de
features de ambos lados. Verifica bytes y recibos de las suites y sus hijos;
no vuelve a leer imágenes originales ni ejecuta fitting. Incluye distribuciones
adyacentes CLIP/DINOv2, zonas, assignments, conteos de clústeres/ruido, estabilidad,
desacuerdos y shortlists recurrentes disponibles, con máscaras y denominadores.
No produce un shortlist nuevo de configuraciones ni selecciona un ganador.

ARI/AMI usa únicamente pares de contenidos únicos con correspondencia biyectiva
entre los pares observados y configuraciones de fitting iguales. Una variante puede cambiar el espacio de
features: esa diferencia se informa por separado, también por run, sin invalidar
la comparación de etiquetas sobre las observaciones emparejadas. Reporta la política de
ruido, cobertura y relaciones excluidas por ambigüedad; sin pairing queda
indefinido. La cobertura parcial nunca elimina las tablas completas originales.
Las zonas se contrastan por pertenencia de ocurrencias emparejadas, no como
cortes exactos. Los pares recurrentes se alinean solo cuando los cores tienen
conjuntos completos de ocurrencias correspondientes. Las diferencias de
protocolo/software/espacio de features y los componentes ausentes quedan
explícitos; no se interpretan como efectos causales de la variante. Comparar
nunca crea secuencias, VDGs ni splits.

En Hypatia, ejecutar comparaciones que carguen Parquet voluminosos dentro de una
asignación SLURM; no hacer esa lectura pesada en el nodo de login. Los comandos
anteriores ilustran el flujo, no una autorización para evaluar ya el dataset
sin HUD.
