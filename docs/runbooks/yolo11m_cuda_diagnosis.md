# Diagnóstico CUDA de YOLO11m — 2026-11-07

Estado: corrección del probe y diagnóstico implementados; causa raíz del OOM y
smoke oficial GPU a 640 **pendientes de comprobación remota**. No cambia el
protocolo científico, la admisión firmada ni el ejecutor experimental.

## Comparación que ya se puede sostener

| Evidencia | Alcance | Resultado / límite |
|---|---|---|
| Corrección `756cdc0` | Importar torch y settings de Ultralytics antes de common_runtime | Sigue presente en la CLI actual; pruebas offline del orden pasan |
| Git: `756cdc0`, `a6dab04`, `1480852`, `b5c8858` | `runtime.py`, `common_runtime.py`, YAML común YOLO11m | Contenido idéntico en los cuatro commits |
| Éxito de octubre, comunicado en el documento de contexto | YOLO11m, tensor 2×3×160×160, forward/backward | No se recibió su log original ni fórmula de pérdida; no valida entrenamiento a 640 |
| Logs originales de seis jobs GPU suministrados por el responsable | `DetectionModel` directo, 1×3×640×640, nc=5 | Error dentro del forward; todavía no se ejecutó el cálculo numérico de la pérdida |
| Control CPU suministrado | Pérdida vectorial, reducción, backward, SGD | Completado según log y contabilidad Slurm; no acredita CUDA |
| Smoke del common-executor | Una epoch, imágenes sintéticas, imgsz=640, batch=1, AMP desactivado | No ejecutado en esta intervención hasta comprobar el diagnóstico |

Los scripts GPU fallidos no importan `configure_runtime` ni el adaptador A/B.
Por tanto, esos componentes no son necesarios para reproducir el fallo observado.
Esto no identifica todavía la causa del allocator/driver ni demuestra que el
smoke completo funcione. Tampoco debe confundirse `cudnn.deterministic=False`
en un probe con cambiar el protocolo científico.

## Error demostrado y corrección mínima

El probe recibido hacía `loss.item()` y `loss.backward()` inmediatamente después
de `loss, loss_items = model(batch)`. La pérdida de detección contiene tres
componentes. La corrección operacional consiste en añadir **`loss = loss.sum()`**
antes de esas llamadas. Ya estaba aplicada en el control CPU y el probe posterior
de cuDNN. El [trainer oficial 8.3.203](https://github.com/ultralytics/ultralytics/blob/v8.3.203/ultralytics/engine/trainer.py)
realiza la misma reducción. El ejecutor del proyecto delega en ese trainer;
no necesita que se cambie su algoritmo.

Esta corrección elimina un error posterior del probe; **no corrige por sí sola
el OOM dentro del forward**. El nuevo diagnóstico prueba esa reducción y conserva
el error original con su etapa y memoria. Las pruebas con dobles sintéticos no
se presentan como ejecución PyTorch/GPU.

## Hipótesis delimitadas

- Límite de memoria por proceso: requiere leer la fracción efectiva en el
  proceso fallido. [PyTorch 2.7.1](https://github.com/pytorch/pytorch/blob/v2.7.1/torch/cuda/memory.py)
  permite consultarla. No se establece una fracción nueva ni se altera un límite.
- Entorno de módulos/bibliotecas: los scripts recibidos no cargan explícitamente
  `python/3.11 cuda/11.8`, a diferencia de la ruta histórica comunicada. No prueba
  que esos módulos estuvieran ausentes del entorno heredado. Registrar los
  módulos y bibliotecas realmente cargados antes de atribuir causalidad.
- Límites del job/proceso: el ReqMem observado es RAM del job, no una medida de
  VRAM. La contabilidad MaxRSS de varios fallos GPU es insuficiente para describir
  el proceso Python; recoger `/proc/self/limits`, status y cgroups en ese proceso.
- Fragmentación o errores CUDA asíncronos: los fallos comunicados con caché
  deshabilitada y sincronización no bastan para asignar una causa. No aplicar
  recomendaciones genéricas del texto de la excepción como si fueran diagnóstico.

`model(tensor)` también construye autograd cuando está habilitado y sus parámetros
lo requieren. `model(batch)` inicializa el criterio y añade la pérdida; el mero
cambio de tipo de entrada no prueba que antes se ejecutara sin gradientes.

## Ejecución acotada

`scripts/hypatia/yolo11m_cuda_diagnostic.sbatch` solicita **un job** con los recursos
del antecedente recibido: partición gpu, una GPU, dos CPU, 12 GiB y 20 minutos.
No se envía automáticamente ni se reintenta. Requiere rutas explícitas:

```bash
export FLIR_REPO=/path/to/existing/executor
export FLIR_PYTHON=/path/to/existing/venv/bin/python
export FLIR_DIAGNOSTIC_SCRIPT=/path/to/yolo11m_cuda_diagnostic.py
export FLIR_CUDA_OUTPUT=/path/to/new/external/attempt
sbatch /path/to/yolo11m_cuda_diagnostic.sbatch
```

La matriz usa seis intérpretes nuevos, uno por caso: forward 640 sin
configure_runtime; pérdida 640 sin configure_runtime; pérdida 640/160/320 con
configure_runtime; y pérdida 640 con los módulos históricos explícitos. Todos
usan seed 42, cinco clases, batch 1 y autograd. Los cinco objetos sintéticos del
probe solo ejercitan el contrato de pérdida, no reproducen una escena FAC.
Los tamaños 160/320 son controles diagnósticos, no resoluciones científicas
alternativas. Una comparación con el script original debe considerar que allí
se usaba un único objeto sintético y no se fijaba seed.

Cada `diagnosis.json` conserva etapas, memoria, fracción del allocator, variables
permitidas, versiones, bibliotecas CUDA cargadas, cgroups, límites y traceback.
No captura credenciales ni el entorno completo. Los resultados son privados:
contienen rutas operacionales y deben quedar fuera del repositorio público.
Un proceso nuevo evita heredar un contexto CUDA fallido. No se limpia la caché
antes de capturar el fallo y no se anula ninguna restricción administrativa.

Si el diagnóstico permite justificar una corrección, comprobarla con la misma
entrada y entorno antes de ejecutar **una** epoch del smoke oficial:

```bash
PYTHONPATH="$FLIR_REPO/src" "$FLIR_PYTHON" "$FLIR_DIAGNOSTIC_SCRIPT" \
  /path/to/new/official-smoke-attempt --case smoke --size 640 --profile configured
```

Este caso llama al `synthetic_smoke` existente. Nunca se reduce su resolución,
se activa AMP, se reemplaza YOLO11m, se usa CPU como fallback o se consume test
protegido. El job diagnóstico no lo ejecuta automáticamente. Un microstep exitoso
no equivale a completar el smoke ni autoriza entrenamiento científico.

## Validación local y límites

Pruebas offline cubren pérdida vectorial, fallo durante forward, preservación de
evidencia, lectura sin modificación de límites, ausencia de fallback CPU y
rechazo de cambiar la resolución del smoke. Se ejecuta además la regresión de
admisión, common-executor, A/B y orden de importación. Los resultados y hashes
de los logs recibidos se registran en un paquete operacional privado separado.
No se afirma CI global verde ni se modifican resultados históricos.
