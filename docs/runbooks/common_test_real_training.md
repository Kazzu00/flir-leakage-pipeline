# Preparación de entrenamiento real YOLO11m con test común

Estado de esta rama: implementación inicial del **entrenamiento de desarrollo**,
sin evaluación del test ni aprobación científica concedida. No reemplaza
`common_test_yolo11m_v2`, las 48 corridas antiguas ni ningún recibo previo.
La admisión debe ser atribuida a un revisor independiente. No modificar
`require_admission` ni escribir un `approved=true` para eludir el gate.

## Qué está implementado

- `common-executor train-approved`: nueva ruta de entrenamiento YOLO11m solo
  con `train` y `val`, usando la matriz deduplicada del plan. Cada run arranca
  desde **el mismo** checkpoint original de Ultralytics firmado mediante SHA256.
- Comprobación de firmas OpenSSH SSHSIG con allowed signers externo y SHA256 del
  anclaje de confianza obtenido de una fuente distinta del checkout.
- Revalidación integral de plan/preparación, igualdad de las identidades del test,
  exclusión del test en los YAML, controles y clases.
- Rechazo de cualquier ejecución sin aprobación, con batch o versiones divergentes,
  y de carpetas parciales. Los resultados completos registran seed, tiempos,
  argumentos efectivos, history, checksum de `best.pt`/`last.pt` y recibo.
- `common-executor verify-approved-run`: contrasta firma, estado, epochs y
  hashes **sin deserializar pesos**.
- En `FLIR_EXECUTOR_MODE=train-approved`, el submit comprueba la admisión
  **antes** de pedir recursos Slurm y el worker vuelve a verificarla.
- La ruta `train` original continúa bloqueada deliberadamente.

## Bloqueos que no desaparecen por escribir código

La revisión real de 8 conflictos, los formularios de las 372 imágenes,
suficiencia por clase, semántica de clase 4, decisión sobre test retrospectivo,
dependencias y freeze con JP **no están acreditados aquí**.
El firmante debe verificar las fuentes de manera independiente. La bandera
`validated_non_test_pilot` en la firma no sustituye la realización y verificación
operacional del piloto; ese flujo y la reanudación verificada siguen pendientes.
El comando `--resume` rechaza todas las solicitudes hasta implementar
procedencia de checkpoints parciales y pruebas de reanudación. **No borrar
checkpoints interrumpidos.** La evaluación final sobre el test común, el bootstrap
pareado y la comparación completa son implementaciones separadas pendientes.

## Contrato de admisión firmado (fuera de Git)

El archivo JSON debe incluir los campos que valida
`src/flir_pipeline/detection/common_admission.py`:
`schema_version=flir_common_test_training_admission_v1`,
`decision=approved_for_real_training`, `reviewer_identity`, `review_date`,
`rationale`, las nueve `decisions` verdaderas y específicamente adjudicadas,
`plan_id`, SHA256 de `plan/receipt.json` y `prepared/preparation.json`,
`frozen_test_id`, SHA256 del checkpoint oficial, origen aprobado,
mapa original de cinco clases y `freeze`: GPU, batch 1/2, 50 épocas, 640,
SGD, versiones fijadas, FP32, TF32=false, checksum defaults, checksum piloto
y constancia atribuida del piloto de desarrollo.

El revisor autorizado firma **exactamente los bytes** del JSON:

```bash
ssh-keygen -Y sign -f /ruta/privada/llave_firmante \
  -n flir-yolo11m-common-test /ruta/externa/authorization.json
```

El archivo `allowed_signers` lo prepara un responsable externo al proceso
que edita el repo. Ejemplo de sintaxis: `revisor@institucion.edu ssh-ed25519 AAAA...`.
No versionar llaves privadas, firmas, datos o la aprobación privada. El
administrador de confianza publica a la operación el SHA256 real del
`allowed_signers` por un canal independiente.

## Ejecución (solo DESPUÉS de los gates humanos y del piloto)

Estas son variables de ejemplo, no nombres reales de archivos disponibles:

```bash
cd ~/flir-leakage-pipeline
git switch feat/common-test-real-training-gate
source ~/.venvs/flir-detection/bin/activate
export UV_PROJECT_ENVIRONMENT="$VIRTUAL_ENV"
export FLIR_REPO="$PWD"
export FLIR_EXECUTOR_MODE=train-approved
export FLIR_EXECUTOR_CONCURRENCY=1
export FLIR_EXECUTOR_PLAN=/ruta/externa/plan
export FLIR_EXECUTOR_PREPARATION=/ruta/externa/prepared
export FLIR_EXECUTOR_OUTPUT=/ruta/externa/real-runs
export FLIR_EXECUTOR_CHECKPOINT=/ruta/externa/yolo11m.pt
export FLIR_EXECUTOR_AUTHORIZATION=/ruta/externa/authorization.json
export FLIR_EXECUTOR_SIGNATURE=/ruta/externa/authorization.json.sig
export FLIR_EXECUTOR_SIGNERS=/ruta/externa/allowed_signers
export FLIR_EXECUTOR_REVIEWER=revisor@institucion.edu
export FLIR_TRUSTED_SIGNERS_SHA256=sha256_64_hex_de_canal_independiente

# NO enviar todavía sin aprobación válida, piloto y configuración operacional:
bash scripts/hypatia/common_test_yolo11m_submit.sh \
  --partition="$GPU_PARTITION" --gres=gpu:1 \
  --cpus-per-task=4 --mem=32G --time=12:00:00
```

La llamada `sbatch` solo tiene sentido tras verificar `RLIMIT_AS`, los
módulos reales del clúster y el tiempo necesario en una reserva autorizada.
Este ejemplo NO afirma que 32 GiB o 12 horas sean suficientes. No mezclar
resultados de distintas reservas/runtime. Validar CI y el receipt de cada
run; no computar los aliases repetidos como réplicas independientes.

Para evaluar finalmente el test común se requiere una fase posterior que
materialice el **mismo** test congelado una sola vez, conserve predicciones
pareadas por frame_id y compute P/R 0.25 IoU 0.5, AP mAP@50 y mAP@50–95,
intervalos compatibles con la dependencia y contraste del diseño completo.
Nunca seleccionar checkpoint, política o umbral usando test.
