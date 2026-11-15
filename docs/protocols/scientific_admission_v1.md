# Contrato de admisión científica firmada para YOLO11m

Estado: **infraestructura implementada, entrenamiento real BLOQUEADO**. Base
`fix/hypatia-cuda-import-order`, `756cdc0`. No se ejecutó Hypatia, entrenamiento,
preparación real, nuevo split, auditoría ni extracción de features en esta intervención.
Los resultados históricos YOLO11n y su executor no cambian.

## Disponibilidad efectivamente inspeccionada

Rutas relativas a la raíz externa de datos; no se publican archivos ni hashes privados.
Estas son observaciones de archivos/metadatos existentes, no una repetición de auditoría.

| Variante | Evidencia local inspeccionada | Admisión actual |
|---|---|---|
| A: original ampliado | `fac/{Imagenes.zip,Adendo.zip,Etiquetas.zip}` existentes; `audits/fixed_test_train_val_v1/five-seeds-20261013-a/summary.json`: 30 candidatos, 25 membresías, ready=false. `executor-yolo11m-v2/plan/{plan,receipt,provenance}.json` existe; no preparación encontrada en ese directorio | Bloqueada: test propuesto y decisiones humanas pendientes. Selección primaria: S1/S2/S3 ampliados, split seed 0, detector 42/43/44 = 9 runs |
| B: sin HUD | Seis ZIP `fac/frames_sin_hud-…`; `audits/no_hud_label_transfer_v1/review-20261015-a/summary.json`: 19.800 imágenes, cero verificadas y cero etiquetas candidatas, ready=false. `audits/no_hud_correspondence_v1/pilot-20261015-a/results/summary.json`: 20 pares, cero identidades verificadas, cero validaciones humanas; geometría de nueve pares pasa screening, no identidad | **No admite transferencia de etiquetas**. Los candidatos y dimensiones coincidentes no acreditan correspondencia 1:1 ni geometría idéntica |
| C: videos fuente | `fac/Videos/{video30min-11to22.mp4,VideoCodefest_008-46min.avi,video_2hour.mp4}` existentes. El contrato local ya inspeccionado de representación conserva muestreo: 9.648 ocurrencias / 8.093 contenidos; no se decodificaron videos ni regeneraron manifests. `experimentos/flir-correspondencia-v2-20261020/manifest.json` declara evidencia exploratoria sin ground truth temporal | Inferencia exploratoria no equivale a evaluación supervisada. Sin un manifest de anotaciones admitido no se admite entrenamiento ni mAP. No se implementa otro sampler ni se inventan etiquetas |

La revisión `audits/detector_evaluation_protocol_v1/acceptance-review-20261013-a/acceptance_report.md`
aprueba infraestructura, **no entrenamiento ni test definitivo**. El informe registra
ocho conflictos protegidos y 141 contenidos del test expuestos a entrenamiento histórico.
Es evidencia documental leída, no revalidación de esas cifras. Los experimentos de
correspondencia posteriores también declaran explícitamente límites exploratorios.

## Autoridad y freeze

`approved=true` no se consulta. El worker verifica una firma OpenSSH sobre los
bytes exactos del documento, namespace `flir-scientific-admission-v1`, principal
`flir-scientific-approver`. La raíz fija `/etc/flir-pipeline/allowed_signers`, todos
sus padres y el propio archivo deben pertenecer a root, sin escritura de grupo/otros
ni symlinks. El worker debe ser POSIX **no root**. No existe argumento para sustituir
la raíz de confianza ni un comando del proyecto para autofirmar aprobaciones.

Un administrador y el responsable científico deben acordar quién puede firmar y
provisionar su **clave pública**. La clave privada nunca va al checkout, al job o al
paquete de experimentos. La firma atribuye una decisión humana; no demuestra que
esa decisión sea científicamente correcta. Revocación: retirar la clave del allowlist;
el worker vuelve a comprobar firma y vigencia en cada operación, no cancela jobs ya iniciados.

La plantilla [scientific_admission.example.json](../../configs/detection/scientific_admission.example.json)
es deliberadamente inválida: test PROPOSED, batch null y hashes/fechas pendientes.
Se completa **fuera del repositorio**, tras resolver las decisiones. Liga:

- Revisión atribuida y hashes de decisiones sobre anotaciones, ontología, suficiencia,
  dependencias, test, presupuesto, origen de pesos y capacidad/batch sin test.
- Plan y preparación existentes, configuración propuesta, commit limpio y `uv.lock`.
  El test usa `records_id` de `preparation.json`, no se confunde con otros `test_id`.
- Peso local `.pt` con SHA256 y URL oficial fija; revisión de origen documentada.
  No se descarga nada en SLURM; no basta renombrar un peso FLIR a yolo11m.pt.
- Batch entero, política/semillas/población, variante y salida fija. Configuración
  compartida de 50 épocas/640/SGD/FP32; no autotuning ni selección por test.
- `environment.python` exacto y `environment.signature` igual al resultado de
  `runtime.environment_signature(preflight_report['environment'])`: paquetes, CUDA,
  GPU, defaults Ultralytics y hardware estable. No se congelan memoria libre ni RAM disponible.

El custodio debe aislar por cuenta/mounts/ACL todos los ZIP originales, galerías,
anotaciones protegidas y cualquier copia del test. Sus rutas se declaran en
`test.protected_roots`; el worker rechaza acceso de lectura/traversal. Solo monta
preparación de desarrollo y decisiones redactadas sin contenido del test.
La aplicación no puede descubrir copias no declaradas ni reemplaza el aislamiento
del sistema operativo. No pasar al trainer un directorio que contenga test aunque
el YAML no lo mencione.

## Primer entrenamiento autorizado

1. Resolver decisiones enumeradas abajo. Reutilizar plan y preparación válidos.
   Si falta la preparación, ejecutar una sola vez el `common-executor prepare`
   existente con el custodio: verifica las asignaciones y escribe solo train/val;
   no reconstruye las 30 particiones. No ejecutar `dry-run` ni auditorías por rutina.
2. Preparar fuera del job el entorno bloqueado (`detection`, Python 3.11), los pesos
   oficiales y su revisión de origen. Reutilizar preflight/probe existentes si
   corresponden al entorno; no asumir que batch 2 cabe sobre FLIR. La decisión de
   capacidad debe indicar qué prueba sin test la respalda; cualquier piloto FLIR
   adicional necesita autorización específica, no lo ejecuta este runbook automáticamente.
3. Fijar el commit final limpio, hashes y ambiente; obtener la firma de la autoridad.
   `ssh-keygen -Y sign -f CLAVE_DEL_REVISOR -n flir-scientific-admission-v1 admission.json`
   es una operación del revisor, **no del worker**. Revisar el documento antes de firmar.
4. Verificar y enviar solo el índice 0 de la matriz admitida:

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
# Debe devolver 9. Recursos proporcionados por Hypatia, no estimados aquí.
bash scripts/hypatia/common_test_yolo11m_submit.sh \
  --partition="$GPU_PARTITION" --gres=gpu:1 --cpus-per-task="$CPUS" \
  --mem="$RAM" --time="$WALLTIME"
```

El submit comprueba admisión antes de `sbatch`; el worker la repite, verifica
aislamiento y entorno y comprueba arquitectura medium/detect al cargar el peso.
La lista ordenada es `context(...)["runs"]`; cada recibo conserva índice implícito,
run_id, candidate_id, brazo, política, split seed, detector seed y variante.
El índice 0 no implica que una estrategia sea preferida. Tras revisar el primer
run sin cambiar el protocolo, enviar índices `1-8`; no reenviar 0. No hay push ni submit automático.

## Interrupción y reanudación

Cada run tiene `attempt-0000`, `attempt-0001`, etc. Se copian únicamente los insumos
de desarrollo a un área runtime escribible para aislar cachés; los originales y
la preparación no cambian. `execution.json` registra hashes, historia y checkpoint.
En cada checkpoint completo se actualiza atómicamente un recibo recuperable sin
rehashear todas las imágenes. SIGTERM solicita cierre; una escritura interrumpida
o SIGKILL entre checkpoint y recibo puede dejar evidencia **no reanudable**, que
se preserva y bloquea. No se reparan hashes automáticamente.

Tras confirmar que el job terminó, el revisor firma un documento de propósito
`resume` con las mismas claves de identidad/vigencia del encabezado, más:

```json
{
  "admission_sha256": "SHA256 del documento train",
  "run_id": "run aprobado",
  "execution": {"path": "/external/scientific-runs/RUN/attempt-0000/execution.json", "sha256": "SHA256"}
}
```

Es un fragmento, requiere `schema_version`, `purpose`, `reviewer`, `not_before`,
`expires_at`. Exportar `FLIR_EXECUTOR_RESUME=1`, `FLIR_RESUME_ADMISSION`,
`FLIR_RESUME_SIGNATURE` y el índice único apropiado; usar el mismo submit.
Se restaura optimizer/EMA de last.pt en **un intento nuevo**, conservando CSV/best
previos. Un lock POSIX impide dos workers para el mismo run. No se reanuda un run
completo ni se amplían épocas. No se garantiza continuidad bit a bit del RNG;
las interrupciones forman parte del registro experimental.

## Evaluación protegida

No ocurre automáticamente al terminar train. La autoridad firma un segundo
documento, propósito `evaluate`, ligado al SHA256 del documento train y al mapa
`executions: {run_id: {path, sha256}}` de **todos los nueve recibos completados**.
Así se fija best.pt elegido por validation antes de abrir el test. Añade:

```json
{
  "test_sources": {
    "records": {"path": "/external/publication/sources/proposal/test_records.parquet", "sha256": "SHA256"},
    "annotations": {"path": "/external/publication/sources/audit/annotation_audit.json", "sha256": "SHA256"},
    "source_root": "/custodian/original-archives"
  }
}
```

El custodio ejecuta `common-executor prepare-test EVAL_JSON EVAL_SIG TRAIN_JSON TRAIN_SIG TEST_OUTPUT`.
Se reutilizan localizadores ZIP, hashes, parser YOLO y escritura comprobada del
pipeline. El evaluator ejecuta `common-executor evaluate EVAL_JSON EVAL_SIG TRAIN_JSON TRAIN_SIG TEST_OUTPUT EVAL_OUTPUT --index 0`
en proceso GPU separado, con una salida nueva por índice. También exige la firma
vigente y el entorno congelado; los workers de entrenamiento no reciben TEST_OUTPUT.
El test materializado se contrasta de nuevo con las filas y bytes aprobados: un
recibo local reescrito no autoriza otra población ni otras etiquetas.

Se escriben Precision/Recall a umbral fijo, mAP@50, mAP@50–95, estadísticas por
imagen y tabla por clase mediante los evaluadores existentes. Clase ausente sigue
siendo indefinida, no cero ni macro adaptativa. No se ejecuta bootstrap en esta
ruta mínima (`bootstrap_executed=false`). ID 4 conserva `SDZI`; no se resuelve por
código su discrepancia con Heavy Machinery. La autoridad debe aceptar explícitamente
la ontología original o tramitar una versión nueva, no alterar las etiquetas aquí.

## Variantes y comparación pareada

La misma ruta admite `variant.kind=geometry_preserving` con `id`, `version`,
`geometry=identity_1to1`, `mapping:{path,sha256}` y `review:{path,sha256}` firmados.
Mapping JSON: `variant_id`, `geometry`, `records` con `frame_id`, `original_sha256`,
`path` absoluta y `sha256` transformado. Debe cubrir toda la población activa
incluido test, con ocurrencias y destinos únicos. Se comprueban dimensiones,
hashes y ausencia de contenido transformado exacto entre particiones; **la revisión
atribuida debe acreditar geometría**, no basta el tamaño ni la similitud visual.
No se admite filtrado silencioso de imágenes sin mapping.

A/B comparten frame_id, etiquetas, membresías, semillas, inicialización, presupuesto
y evaluación; cambian únicamente variante/bytes de imagen admitidos. Ambas
admisiones deben fijarse antes de observar sus resultados; la decisión de apertura
del test debe esperar ambas matrices si se estudia el efecto pareado. El gate
automatizado exige completar la matriz del documento actual, no coordina documentos
independientes: el revisor debe enlazar las dos liberaciones. Esta comparación
permanece bloqueada por el mapping real de B. C no puede usar esta interfaz como
atajo para inventar supervisión; requeriría un contrato etiquetado admitido nuevo.

## Decisiones todavía pendientes y límites operacionales

1. Adjudicar ocho conflictos protegidos y aceptar/versionar la ontología; si cambian
   etiquetas, los hashes viejos dejan de ser admisibles. No firmar archivos pendientes
   como sustituto de la revisión real.
2. Decidir suficiencia por clase, límites retrospectivos del test y política de
   dependencias candidatas/confirmadas. Firmar test y asignaciones existentes solo
   si esa decisión es metodológicamente defendible.
3. Aprobar nueve runs, métricas/budget/batch y su evidencia de capacidad sin test.
   La extensión a cinco semillas exige nueva aprobación explícita de 45 identidades;
   no se lanza automáticamente ni debe repetir resultados ya existentes sin una
   decisión de reutilización de evidencia. No se implementan aliases entre admisiones.
4. Aprobar origen/SHA256 de pesos oficiales y freeze de entorno/commit. No se encontró
   preparación ni pesos en el directorio executor inspeccionado; no se presume
   disponibilidad en Hypatia ni se afirma ausencia en todo el sistema.
5. Administrador: provisionar raíz de confianza, cuenta no root, aislamiento de test,
   mounts read-only y recursos SLURM. Esto es una propuesta de operación pendiente
   de adopción, no una autoridad creada por el agente.
6. Smoke operacional autorizado en Hypatia pendiente. Pruebas offline con dobles no
   acreditan CUDA, memoria, señales del scheduler ni continuidad numérica del runtime.

La implementación toma el comportamiento de resume/save_dir del
[trainer oficial Ultralytics 8.3.203](https://github.com/ultralytics/ultralytics/blob/v8.3.203/ultralytics/engine/trainer.py).
La subclase solo verifica argumentos y redirige el intento; no altera optimizer,
función de pérdida, asignaciones o métricas. No se cambió el executor YOLO11n.

## Comprobaciones focalizadas

```bash
uv run --offline --no-sync python -m pytest tests/test_scientific_executor.py tests/test_common_executor.py tests/test_detection_cuda_import_order.py tests/test_detection.py -q
uv run --offline --no-sync ruff check src/flir_pipeline/detection tests/test_scientific_executor.py tests/test_common_executor.py
bash -n scripts/hypatia/common_test_yolo11m.sh
bash -n scripts/hypatia/common_test_yolo11m_submit.sh
bash -n scripts/hypatia/common_test_yolo11m.sbatch
```

Los tests nuevos usan archivos/firmas temporales sintéticos, sin red, pesos reales
ni GPU. OpenSSH debe estar instalado para el test criptográfico. En Windows se
usó `python -m pytest` porque Control de aplicaciones bloqueó el launcher pytest;
no se cambió esa política del sistema.
