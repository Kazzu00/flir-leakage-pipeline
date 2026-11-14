# Primer entrenamiento YOLO11m autorizado

**Estado: BLOQUEADO para datos reales.** Código y pruebas sintéticas disponibles;
no se ejecutó entrenamiento, materialización real ni un job Hypatia. Base `756cdc0`.
El contrato, esquema de firmas, disponibilidad inspeccionada de A/B/C y límites
están en [scientific_admission_v1.md](../protocols/scientific_admission_v1.md).

## Antes de enviar un job

1. **Decisión científica:** adjudicar anotaciones/ontología, suficiencia por clase,
   política de dependencias y test fijo. Aprobar A ampliada, S1/S2/S3, split seed 0,
   detector seeds 42/43/44: nueve entrenamientos, configuración y batch comunes.
   Las 30 particiones actuales siguen PROVISIONAL; no cambiar sus flags ni reconstruirlas.
2. **Insumos:** reutilizar el plan existente y `prepare`/`verify --kind preparation`.
   Preparar solo si falta la vista train/val; no repetir splits/auditorías. Peso
   oficial local con origen revisado y SHA256, entorno bloqueado, commit limpio y
   batch respaldado por prueba de capacidad sin test. Nunca descargar pesos en jobs.
3. **Autoridad:** el administrador configura `/etc/flir-pipeline/allowed_signers`
   root-owned y una cuenta POSIX no root. El custodio impide a esa cuenta leer
   ZIP originales, test, etiquetas protegidas y galerías. La aplicación comprueba
   los roots declarados pero no descubre copias ocultas. No hay autoridad instalada
   ni aprobación creada por esta implementación.
4. **Firma:** completar fuera del repo la [plantilla bloqueada](../../configs/detection/scientific_admission.example.json)
   con hashes y decisiones reales. El revisor firma el documento con OpenSSH,
   namespace `flir-scientific-admission-v1`. Un JSON `approved=true` no sirve.

## Lanzamiento del primer run

```bash
export FLIR_REPO=/checkout/limpio
export FLIR_EXECUTOR_PLAN=/external/plan
export FLIR_EXECUTOR_OUTPUT=/external/scientific-runs
export FLIR_ADMISSION=/external/approvals/admission.json
export FLIR_ADMISSION_SIGNATURE=/external/approvals/admission.json.sig
export FLIR_EXECUTOR_MODE=train
export FLIR_EXECUTOR_CONCURRENCY=1
export FLIR_EXECUTOR_INDICES=0
cd "$FLIR_REPO"
bash scripts/hypatia/common_test_yolo11m.sh admission-check \
  "$FLIR_EXECUTOR_PLAN" "$FLIR_EXECUTOR_OUTPUT" "$FLIR_ADMISSION" "$FLIR_ADMISSION_SIGNATURE"
# Debe devolver 9; Hypatia determina recursos, no este ejemplo.
bash scripts/hypatia/common_test_yolo11m_submit.sh \
  --partition="$GPU_PARTITION" --gres=gpu:1 --cpus-per-task="$CPUS" \
  --mem="$RAM" --time="$WALLTIME"
```

Submit y worker comprueban admisión. Cada run deja identidad, argumentos efectivos,
ambiente, historia y hashes en `RUN/attempt-NNNN/execution.json`. El trainer solo
recibe desarrollo. Tras revisar el primer run sin cambiar condiciones, enviar
índices `1-8`; no repetir 0. Cinco semillas de split requieren aprobación nueva
de presupuesto y una decisión explícita sobre reutilizar los nueve resultados.

## Reanudación y evaluación

- **Resume:** confirmar fin del job y verificar recibo/checkpoints del último intento.
  Obtener firma de propósito `resume`, ligada al documento train, run_id y SHA256
  de execution.json. Exportar `FLIR_EXECUTOR_RESUME=1`, `FLIR_RESUME_ADMISSION`,
  `FLIR_RESUME_SIGNATURE` y el índice único; reutilizar submit. Se crea otro intento,
  no se sobrescribe el previo ni se añaden épocas. Bitwise RNG resume no garantizado.
  Una escritura truncada no se repara automáticamente.
- **Test:** después de completar la matriz, firma separada `evaluate` ligada a los
  nueve recibos y a las fuentes del test. El custodio ejecuta
  `common-executor prepare-test EVAL_JSON EVAL_SIG TRAIN_JSON TRAIN_SIG TEST_OUTPUT`.
  En el entorno de evaluación, ejecutar
  `common-executor evaluate EVAL_JSON EVAL_SIG TRAIN_JSON TRAIN_SIG TEST_OUTPUT EVAL_OUTPUT --index 0`
  por índice, con salida nueva. Best se seleccionó por validation; no hay tuning de
  test. Se registran P/R, mAP50/mAP50–95 y métricas por clase. Sin bootstrap en esta ruta mínima.

**B sin HUD:** archivos presentes, pero publicación inspeccionada con cero mappings
aprobados; no reutilizar etiquetas hasta admitir correspondencia 1:1 y geometría.
**C videos:** tres fuentes presentes, sin contrato supervisado admitido; inferencia
sin ground truth no produce mAP. No se generaron frames nuevos.

## Validación y límites

```bash
uv run --offline --no-sync python -m pytest tests/test_scientific_executor.py tests/test_common_executor.py tests/test_detection_cuda_import_order.py tests/test_detection.py -q
uv run --offline --no-sync ruff check src/flir_pipeline/detection tests/test_scientific_executor.py tests/test_common_executor.py
```

Pruebas offline con datos y dobles sintéticos; test de firma usa OpenSSH local.
Resultado de la intervención: 68 casos focalizados distintos aprobados (27 nuevos),
Ruff global, ayuda CLI, rechazo por defecto y sintaxis de scripts comprobados.
Smoke operacional de runtime, capacidad GPU y señales SLURM en Hypatia pendiente.
No se ejecutó la suite global, ni se alteraron resultados/exports/checkpoints YOLO11n.
