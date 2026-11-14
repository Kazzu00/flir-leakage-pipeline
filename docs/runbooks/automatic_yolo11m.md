# Ejecutor YOLO11m: política técnica automática v1

La instrucción del responsable del 2026-11-10 sustituye el requisito de firmas y
revisión humana para este protocolo por comprobaciones técnicas reproducibles.
Esta política **no certifica las anotaciones ni la independencia del test**. No
reescribe estados ni recibos históricos y no utiliza la capa B adjudicada del
paquete A/B. Conserva las etiquetas originales de cada ocurrencia, incluso cuando
dos copias exactas tienen anotaciones distintas. La clase 4 sigue siendo `SDZI`;
la equivalencia con `Heavy Machinery` sigue sin establecerse.

## Alcance y estado

Infraestructura implementada y verificable con fixtures offline. No se ejecutó
entrenamiento FLIR, piloto FLIR ni evaluación del test durante este cambio.
La validación operacional CUDA de esta versión requiere Hypatia. El diagnóstico
792984 sustenta presión de `RLIMIT_AS` como hipótesis principal del OOM, no una
corrección GPU ya demostrada. Se conserva la solicitud Slurm de 32G y 640×640.

La publicación original inspeccionada contiene 372 ocurrencias de test, 142 en la
unión de cuarentena, 30 brazos y 25 membresías distintas. Los brazos históricos
contienen 141 de esas ocurrencias en cuarentena; la restante pertenece a la
población ampliada. El plan deriva 90 identidades y 75 ejecuciones distintas. El
código verifica la matriz y deduplica; **75 no es una constante del ejecutor**.
Las etiquetas originales del test contienen 983 instancias. No se reutiliza el
conteo de 991 de la capa B adjudicada anterior. El soporte por clase distingue
ocurrencias positivas, instancias y contenidos exactos conocidos; ninguno de
estos conteos demuestra por sí solo independencia estadística.

La política es `original_occurrences_technical_v1`. Los adaptadores A/B y contratos
firmados anteriores permanecen como rutas históricas separadas; no son requisitos
de esta ruta. Sus etiquetas, población y recibos no se reinterpretan como la
publicación histórica/ampliada S1/S2/S3.

## Contratos y arquitectura

1. `common_admission.py` verifica publicación, SHA256 de ZIP originales, plan
   reconstruido, preparación y correspondencia de etiquetas por ocurrencia.
   Conserva auditorías y anomalías. Rechaza sintaxis, clases, valores normalizados
   o identidades inválidas; una anomalía de bordes se documenta sin corregir bytes.
2. `verify-weights` compara el checkpoint local con bytes obtenidos mediante HTTPS
   del asset oficial `ultralytics/assets`, release `v8.3.0`, `yolo11m.pt`. Solo esta
   operación explícita requiere red. No deserializa ni escribe una descarga de
   pesos. Declarar una URL o suministrar un hash no basta. Los workers verifican
   el recibo y los bytes locales sin descargar nada.
3. `common_training.py` comprueba probe CUDA completo, ejecuta un piloto de una
   época de desarrollo y crea un freeze con batch común, entorno, código, pesos y
   configuración. El piloto mantiene resolución e inicialización, con presupuesto
   de una época y cierre de mosaic en cero. No selecciona batch por métricas ni
   cuenta como celda definitiva. Batch 1 o 2 debe pasar probe y piloto: ningún
   fallo activa fallback silencioso. Un éxito sintético no demuestra capacidad FLIR.
4. El worker reutiliza `scientific_runtime.py`: SGD, 50 épocas, FP32, AMP/TF32
   desactivados, semillas 42/43/44 y test ausente de los YAML de desarrollo. Cada
   ejecución tiene ID determinista y cada intento un directorio nuevo con lock de
   SO. Repetir una ejecución completa verifica sus artefactos y retorna sin
   entrenar. `--resume` solo admite el último intento de la misma ejecución, con
   `last.pt`, `best.pt`, argumentos e historial vinculados por SHA. Antes de seguir
   comprueba época del checkpoint, optimizador e historial; no extiende el
   presupuesto. Restaurar optimizer/EMA no demuestra continuidad bit a bit del
   RNG. Si SIGKILL deja archivos distintos del recibo atómico, falla conservando
   todo; no rehabilita un checkpoint dudoso.
5. `common_evaluation.py` libera técnicamente el test solo al verificar **todas**
   las ejecuciones definitivas. Prepara una copia común de originales y evalúa
   únicamente `best.pt` seleccionado en validation. Cada evaluación tiene su
   propia copia para cachés. Un piloto o matriz parcial no libera el test.

Los recibos son controles de integridad de un workspace confiable, no firmas de
terceros ni defensa contra alguien que reescriba deliberadamente código y todos
los artefactos. Cambiar código o entradas tras el freeze exige nuevas identidades.
El campo histórico `compute.real_pilot_authorized: false` y los blockers del plan
se preservan como parte del documento original. La nueva política establece
capacidad técnica por su propio recibo, sin cambiar esos valores ni afirmar que
exista una revisión humana. Implementar esta ruta no ejecuta el piloto.

## Métricas y comparación

Se reutilizan el matcher fijado a Ultralytics 8.3.203 y `detection/metrics.py`: P/R
con confidence 0.25 e IoU 0.5; AP con floor 0.001 e IoU 0.50:0.95; NMS IoU 0.7 y
300 detecciones máximas. Se mantiene el macro fijo de cinco clases; una clase sin
soporte no se convierte en una métrica favorable definida.

Exportaciones: estadísticas de matching NPZ, predicciones post-NMS (xyxy en píxeles
de entrada al validador), métricas por imagen/clase, TP/FP/FN, aliases y hashes de
ejecuciones, pesos y test. Las exportaciones se verifican contra las estadísticas
capturadas. Los aliases identifican población, estrategia y semilla de partición;
dos aliases de una misma ejecución no son réplicas independientes.

`compare` separa contrastes entre estrategias dentro de una población y contrastes
de composición entre poblaciones, emparejando semillas. Reporta variabilidad entre
semillas del detector y de partición por separado. Recalcula diferencias pareadas
con los mismos remuestreos por `content_id` exacto conocido. Los intervalos son
condicionales: **no corrigen dependencias visuales/temporales desconocidas ni
establecen independencia estadística**. No son efectos causales del leakage.

La referencia [yolo_eval.py](https://github.com/manugalarza/proyecto-segementacion-panoptica/blob/main/baseline/yolo_eval.py)
fue inspeccionada. Su matching y tratamiento de SDZI no sustituyen nuestro protocolo,
ni se adoptan su tolerancia a labels ausentes o sus pesos FLIR. La captura sigue la
API del [validador 8.3.203](https://github.com/ultralytics/ultralytics/blob/v8.3.203/ultralytics/models/yolo/detect/val.py).

## Comandos Hypatia

Desde el checkout de integración, con rutas de inputs externos suministradas por el
operador. Preparar/transportar los inputs antes de generar recibos en Hypatia: las
rutas absolutas forman parte del contrato. `FLIR_SOURCE_ROOT` contiene ZIP originales;
`FLIR_PUBLICATION` es la publicación de 30 brazos, **no** el paquete adjudicado A/B.
Si la auditoría vincula evidencia adicional, añadir `--evidence "$FLIR_EVIDENCE"`
a `prepare` y `admit`.

El wrapper resuelve su propio checkout y antepone `src` a `PYTHONPATH`: un entorno
compartido con `--no-sync` no debe importar la instalación editable de otra rama.

```bash
export FLIR_REPO="$HOME/flir-yolo11m-auto-executor"
export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/flir-detection"
cd "$FLIR_REPO"
export FLIR_WORK="$HOME/flir-yolo11m-automatic-v1"
mkdir -p "$FLIR_WORK"
# Exportar FLIR_PUBLICATION, FLIR_SOURCE_ROOT y FLIR_INITIAL_WEIGHTS.
run() { bash scripts/hypatia/common_test_yolo11m.sh "$@"; }
run dry-run "$FLIR_PUBLICATION" "$FLIR_WORK/plan"
run prepare "$FLIR_PUBLICATION" "$FLIR_SOURCE_ROOT" "$FLIR_WORK/prepared"
run admit "$FLIR_PUBLICATION" "$FLIR_WORK/plan" "$FLIR_WORK/prepared" \
  "$FLIR_SOURCE_ROOT" "$FLIR_WORK/data-gate.json"
run verify-weights "$FLIR_INITIAL_WEIGHTS" "$FLIR_WORK/official-weights.json"
```

**Una sola verificación sintética inicial:**

```bash
export FLIR_EXECUTOR_MODE=synthetic-smoke
export FLIR_EXECUTOR_OUTPUT="$FLIR_WORK/synthetic"
bash scripts/hypatia/common_test_yolo11m_submit.sh \
  --partition=gpu --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=00:30:00
```

Partición y recursos deben existir en la cuenta Slurm. El script solicita recursos,
sin alterar controles SSH ni límites del proceso. El smoke incluye preflight y una
época sintética a 640, batch 1. No repetir una variante equivalente tras un fallo
sin inspeccionar evidencia y justificar el cambio. Consultar `environment.json`,
`state.json`, `receipt.json` y salida Slurm. Se registran CPU, RAM, GPU, VRAM libre,
total y pico, VMS/RSS, RLIMIT_AS/DATA/MEMLOCK/STACK, asignación y memoria solicitada
a Slurm, versiones, configuración, commit y hashes de fuente.

**Pasos futuros que ejecutan FLIR; no ejecutados durante esta implementación:**

```bash
export FLIR_DATA_GATE="$FLIR_WORK/data-gate.json"
export FLIR_WEIGHTS_RECEIPT="$FLIR_WORK/official-weights.json"
export FLIR_GPU_PROBE="$FLIR_WORK/synthetic/JOBID-0" # operación completa exacta
export FLIR_COMMON_BATCH=1
export FLIR_EXECUTOR_MODE=pilot
export FLIR_EXECUTOR_OUTPUT="$FLIR_WORK/pilots"
bash scripts/hypatia/common_test_yolo11m_submit.sh \
  --partition=gpu --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=01:00:00
run freeze "$FLIR_DATA_GATE" "$FLIR_WEIGHTS_RECEIPT" "$FLIR_GPU_PROBE" \
  "$FLIR_WORK/pilots/PILOT_RUN_ID/attempt-0000" "$FLIR_WORK/freeze.json" --batch 1
export FLIR_TECHNICAL_FREEZE="$FLIR_WORK/freeze.json"
export FLIR_EXECUTOR_PLAN="$FLIR_WORK/plan"
export FLIR_EXECUTOR_MODE=train
export FLIR_EXECUTOR_OUTPUT="$FLIR_WORK/training"
export FLIR_EXECUTOR_CONCURRENCY=1
# Deriva el tamaño de la matriz; no tiene límite constante 75.
bash scripts/hypatia/common_test_yolo11m_submit.sh \
  --partition=gpu --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=12:00:00
# Reanudar una única ejecución interrumpida verificable:
export FLIR_EXECUTOR_INDICES=0
export FLIR_EXECUTOR_RESUME=1
bash scripts/hypatia/common_test_yolo11m_submit.sh \
  --partition=gpu --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=12:00:00
unset FLIR_EXECUTOR_INDICES FLIR_EXECUTOR_RESUME
```

Evaluación futura después de la matriz completa:

```bash
run release-test "$FLIR_WORK/freeze.json" "$FLIR_WORK/training" "$FLIR_WORK/test-release.json"
run prepare-test "$FLIR_WORK/test-release.json" "$FLIR_WORK/common-test"
# Dentro de asignación GPU, repetir por cada índice derivado del freeze.
run evaluate "$FLIR_WORK/test-release.json" "$FLIR_WORK/common-test" "$FLIR_WORK/evaluations" --index 0
# Solo con todas las evaluaciones completas; consume predicciones ya guardadas.
run compare "$FLIR_WORK/test-release.json" "$FLIR_WORK/common-test" \
  "$FLIR_WORK/evaluations" "$FLIR_WORK/comparison"
```

Las pruebas offline verifican contratos y dobles de runtime, no GPU ni resultados
científicos. Queda verificar en Hypatia: corrección efectiva del OOM bajo RLIMIT_AS,
versiones, probe, piloto FLIR, batch común, interrupción y resume de Ultralytics
real. Solo después corresponde ejecutar la matriz y evaluar el test. El protocolo
YOLO11n histórico de 48 ejecuciones permanece separado y sin modificaciones.
