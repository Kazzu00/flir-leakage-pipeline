# YOLO11m v2 con test común provisional

Infraestructura técnica, sin aprobación científica. `controlled_detector_v1` y
sus 48 ejecuciones no se modifican ni se reutilizan como inicialización.
El estado vigente del repositorio está en `docs/status.md` (no existe
`docs/current_status.md`). No se ejecutaron entrenamientos FLIR ni probes GPU
durante esta implementación.

## Contrato y bloqueos

`common_test_yolo11m_v2.yaml` conserva 50 épocas propuestas, resolución 640,
SGD, FP32, semillas 42/43/44 y las definiciones históricas de métricas. La
inicialización científica prevista es `yolo11m.pt` oficial. No admite pesos
FLIR de JP. Batch sigue pendiente: no se presume que 8 quepa en GPU.

El plan reconstruye y verifica la publicación completa antes de producir 30
brazos / 90 identidades experimentales. Solo agrupa registros completos iguales,
incluyendo anotaciones, procedencia y cuarentena, con configuración e
inicialización iguales y la misma semilla del detector. Si se mantienen las 25
membresías observadas, se esperan 75 ejecuciones candidatas; el comando calcula
el número, no lo impone. Los alias conservan brazo, población, política, split
seed y candidate ID. Un hash declarado de pesos sigue siendo una dependencia
no verificada. Los IDs son **provisionales**, no IDs de experimentos congelados.

La preparación verifica ZIP originales en lectura y escribe solamente fuera
del repositorio y de las fuentes. Conserva todas las asignaciones y procedencia
en Parquet, pero materializa únicamente train/val. Test (372 ocurrencias) y
quarantine nunca aparecen en los YAML operacionales. Cada ocurrencia conserva
sus bytes y etiqueta; las vistas comparten el mismo almacén físico. El nombre
original del ID 4 es `SDZI`; `Heavy Machinery` en el catálogo interno no acredita
equivalencia. La discrepancia queda en los recibos; no se corrigen anotaciones.

**Train y resume están bloqueados incondicionalmente**, incluso con JSON que
declare `approved=true`. No existe `--force`. No se implementa una vía de
entrenamiento FLIR hasta disponer de un contrato de admisión verificable:

- revisión humana atribuida y verificable por una raíz de confianza externa;
- decisiones de anotación, suficiencia y clases ligadas a hashes de entradas;
- aprobación del test y freeze de membresías, métricas, presupuesto y protocolo;
- pesos oficiales con origen contrastado y SHA256, nunca solo un nombre;
- entorno, commit, dependencias y batch común congelados tras piloto autorizado
  sin test; procedencia e identidad de `last.pt` para cualquier reanudación;
- resultados completos inmutables, historial de interrupciones, recibos de
  checkpoints y evaluación de test solo bajo autorización verificable.

Ese contrato y la ejecución/evaluación científica posterior permanecen pendientes.
Los smoke interrumpidos se preservan; se reintentan en directorio nuevo, sin
deserializar `last.pt`. `--resume` real falla antes de cargar pesos o tocar GPU.
No se afirma que entrenamiento científico o recuperación estén validados.

## Windows: planificación sin GPU

Desde el repositorio, con entorno core/dev preparado:

```powershell
$publication = Join-Path $env:FLIR_DATA_ROOT 'audits/fixed_test_train_val_v1/five-seeds-20261013-a'
$work = Join-Path $env:FLIR_DATA_ROOT 'executor-yolo11m-v2'
uv run flir-pipeline detection common-executor dry-run $publication "$work/plan"
uv run flir-pipeline detection common-executor verify "$work/plan"
uv run flir-pipeline detection common-executor status "$work/plan"
# sourceRoot debe contener los ZIP originales con sus nombres registrados.
$sourceRoot = $env:FLIR_DATA_ROOT
uv run flir-pipeline detection common-executor prepare $publication $sourceRoot "$work/prepared"
uv run flir-pipeline detection common-executor verify "$work/prepared" --kind preparation
```

Si el audit registra evidencia adicional, pasar `--evidence RUTA` a `prepare`.
No se asume una ubicación de datos privada. `FLIR_DATA_ROOT` debe estar exportada
en el shell; estos comandos no cargan `.env` automáticamente. Los outputs son
nuevos e inmutables; no se reparan ni sobrescriben silenciosamente.

## Hypatia: entorno y pruebas operacionales

Preparar previamente Python 3.11 y el entorno bloqueado con los extras
`detection,dev`, respetando torch 2.7.1/cu118, torchvision 0.22.1 y Ultralytics
8.3.203. No instalar ni descargar dentro del job. Adaptar módulos y recursos
a lo disponible en Hypatia; no hay supuestos de VRAM, tiempo o partición.

```bash
export FLIR_REPO="$PWD"
export FLIR_EXECUTOR_OUTPUT=/ruta/externa/executor-yolo11m-v2/operations
bash scripts/hypatia/common_test_yolo11m.sh preflight "$FLIR_EXECUTOR_OUTPUT/preflight.json"
# CPU solo mediante petición explícita; no representa capacidad GPU:
bash scripts/hypatia/common_test_yolo11m.sh synthetic-smoke "$FLIR_EXECUTOR_OUTPUT/cpu-smoke" --cpu
export FLIR_EXECUTOR_MODE=probe
export FLIR_EXECUTOR_CONCURRENCY=1
# Definir estos valores conforme al clúster; no son estimaciones del proyecto.
bash scripts/hypatia/common_test_yolo11m_submit.sh \
  --partition="$GPU_PARTITION" --gres=gpu:1 --cpus-per-task="$CPUS" \
  --mem="$RAM" --time="$WALLTIME"
bash scripts/hypatia/common_test_yolo11m_status.sh "$JOB_ID"
bash scripts/hypatia/common_test_yolo11m.sh verify /ruta/al/probe/batch-1 --kind operation
```

Smoke usa arquitectura YOLO11m con inicialización aleatoria y diez imágenes
sintéticas por partición, a 640, una época, sin pesos descargados ni test. Probe
mide memoria libre y máxima en GPU real y ensaya batch 1 y 2; un OOM queda
registrado y cualquier otro fallo aborta. Su candidato de batch **no es un
freeze ni garantía de capacidad sobre FLIR**. No hay fallback automático a CPU.
La verificación comprueba recibos, historial validation y checkpoints sin cargar
pickle. Los resultados reportan estado completed/failed/interrupted; el plan
científico sigue pending con bloqueos y no declara resultados reutilizables.

La plantilla SLURM soporta índices de array y concurrencia; el submit científico
falla antes de `sbatch`. No enviar manualmente jobs reales para intentar eludir
el bloqueo: el worker también lo aplica.

## Verificación offline focalizada

```bash
uv run --offline pytest tests/test_common_executor.py tests/test_detection.py tests/test_common_test.py tests/test_fixed_test.py
uv run --offline ruff check src/flir_pipeline/detection tests/test_common_executor.py
```

Los tests usan datos sintéticos y dobles del runtime, sin modelos, Internet o
GPU. La suite completa corresponde a CI. No confundir esos tests con un smoke
Ultralytics ejecutado en Hypatia.
