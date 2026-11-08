# Diagnóstico CUDA de YOLO11m — 2026-11-07

Estado tras job 792984: **agotamiento del espacio de direcciones virtuales del
proceso es la explicación mejor sustentada**. Corrección operacional preparada:
solicitar 32G de RAM al scheduler. La intervención causal y el smoke oficial GPU
a 640 **todavía no se han ejecutado**. No cambia el protocolo, la admisión ni el
código experimental. No se declara recuperada la GPU antes de esa comprobación.

## Auditoría íntegra del arnés — 2026-11-08

Se revisaron los seis JSON completos (647 valores hoja cada uno), sus trazas,
stdout, stderr y códigos de salida. Los 23 archivos recibidos conservan sus
hashes. No hay errores de captura `unavailable`; nvidia-smi y scontrol retornaron
0. Los stderr vacíos son consecuencia del except que escribe diagnosis.json y
retorna 1, no una omisión de errores ni prueba de éxito.

El defecto encontrado es **de etiquetado de etapa**: phase=before_forward era
el último checkpoint, no la ubicación del fallo. Se separan ahora last_checkpoint
y phase activa (forward/backward/optimizer, además de setup/smoke). El código
computacional es idéntico por comparación AST excluyendo únicamente esas
asignaciones de etiquetas. No se cambian tensores, sincronizaciones, imports,
semillas, pérdida, optimizador ni llamadas CUDA.

| Variante | Punto donde se notificó el error | Memoria CUDA asignada antes → al fallar |
|---|---|---:|
| inherited-forward-640 | SiLU, forward sin criterio de pérdida | 81.71 → 525.30 MiB |
| inherited-loss-640 | SiLU, forward invocado desde loss | 81.71 → 525.30 MiB |
| configured-loss-640 | BatchNorm, forward de C2f/Conv | 81.71 → 209.71 MiB |
| configured-loss-160 | Atención C2PSA: producto matricial, cublasCreate | 77.31 → 109.46 MiB |
| configured-loss-320 | SiLU, forward de C2f/Conv | 78.19 → 152.10 MiB |
| october-modules-loss-640 | BatchNorm, forward de C2f/Conv | 81.71 → 209.71 MiB |

La inicialización CUDA, transferencia del modelo y creación de tensores sí se
ejecutaron. Las trazas recorren capas del modelo y la memoria asignada aumenta
durante forward. Ninguno alcanzó la reducción de pérdida, backward o SGD.step.
No hay evidencia de un fallo compartido de parser, imports, versiones o tipos
del arnés. Como CUDA es asíncrono, la traza identifica dónde se notificó el
error, no demuestra por sí sola qué kernel anterior lo originó.

### Comparación honesta del control a 160 con octubre

| Aspecto | Octubre, según evidencia comunicada | Job 792984 |
|---|---|---|
| Construcción | YOLO("yolo11m.yaml") | DetectionModel("yolo11m.yaml", ch=3, nc=5) |
| Tensor | randn(2,3,160,160), distribución normal | rand(1,3,160,160), uniforme |
| Objetivo | Forward/backward reportados; fórmula de pérdida no disponible | Pérdida nativa de detección, cinco cajas, reducción sum |
| Optimización | No consta código ni llamada SGD verificable | SGD se construye antes del forward; step no se alcanza |
| Torch/Ultralytics | 2.7.1+cu118 / 8.3.203 reportados | Mismas versiones observadas |
| cuDNN, TF32, determinismo | No constan flags efectivos históricos | cuDNN 90100, TF32 desactivado en configurado, determinismo activo |
| Módulos | python/3.11 y cuda/11.8 reportados | Ninguno heredado; control adicional carga ambos y falla |
| Límites del proceso | No disponibles en el antecedente | RLIMIT_AS soft=hard=13.2 GiB |

No se encontró el script/log original de esa ejecución de octubre en los
artefactos locales consultados. El control actual a 160 es una microprueba de
pérdida, **no una reproducción exacta** de aquel tensor-smoke. La diferencia de
memoria efectiva frente a octubre aún no puede cuantificarse. Por ello no se
atribuye una regresión a PyTorch, CUDA o al cluster. El import-order de 756cdc0
se conserva y precede a common_runtime; su corrección no elimina el límite AS.

Se mantiene la única intervención ya preparada: reserva de 32G por Slurm y
microstep 640, seguida como máximo por un smoke oficial. No se repiten las seis
variantes ni se añade una reconstrucción ficticia del ensayo de octubre. El
script auditado tendrá SHA distinto por las etiquetas, diferencia documentada
que no cambia la computación del control.

## Resultado de los seis diagnósticos

Todos devuelven exit code 1 y estado failed. Los stderr están vacíos porque el
script conserva la excepción en diagnosis.json; no indican éxito. Los stdout
son idénticos y contienen la descripción del modelo, sin resultados de pérdida.
Los seis recibos identifican el mismo script y executor 1480852, sin diff de
fuente reportado. Ninguno completó forward, backward o SGD.

| Caso | Error | VRAM libre GiB | VmSize GiB | Margen hasta RLIMIT_AS MiB |
|---|---|---:|---:|---:|
| Forward 640, sin configure_runtime | CUDA OOM | 20.551 | 13.147816 | 53.437 |
| Loss 640, sin configure_runtime | CUDA OOM | 20.551 | 13.145420 | 55.890 |
| Loss 640, configurado | CUDA OOM | 20.891 | 13.149662 | 51.546 |
| Loss 160, configurado | CUBLAS_STATUS_NOT_INITIALIZED | 20.953 | 13.156631 | 44.409 |
| Loss 320, configurado | CUDA OOM | 20.928 | 13.163509 | 37.366 |
| Loss 640, módulos octubre | CUDA OOM | 20.891 | 13.177235 | 23.312 |

VmSize se captura tras el error, no durante la llamada fallida; no permite
identificar el mmap concreto rechazado. El límite soft y hard de RLIMIT_AS es
14,173,392,076 bytes (13.2 GiB) en todos los casos. RLIMIT_RSS y la reserva del
job son 12 GiB; RSS observado es 2.93–3.08 GiB. Los límites de datos y memoria
bloqueada son ilimitados. No confundir espacio virtual, RAM residente y VRAM.

El allocator es native, fracción efectiva 1.0, sin configuración personalizada.
Los cinco OOM tienen num_ooms=1; el fallo cuBLAS tiene num_ooms=0. La memoria GPU
asignada por PyTorch es 109–525 MiB. Los cgroups de memoria accesibles pertenecen
a slurmd/system.slice, no a un límite de memoria por job; sus failcnt permanecen
en cero. Esto no prueba ausencia de toda restricción del sistema.

Versiones iguales: torch 2.7.1+cu118, CUDA 11.8, cuDNN 90100, Ultralytics 8.3.203.
En todos: autograd y algoritmos deterministas activos, cudnn.benchmark=False,
matmul TF32=False. El perfil sin configure_runtime conserva cudnn TF32=True;
el configurado usa False. Ambos fallan. CUDA_MODULE_LOADING termina en LAZY y
CUBLAS_WORKSPACE_CONFIG en :4096:8; LD_PRELOAD está ausente.

El entorno heredado no tenía módulos. La última prueba cargó anaconda,
python/3.11 y cuda/11.8 y amplió LD_LIBRARY_PATH, pero cuBLAS, cudart y cuDNN
continuaron cargándose desde los paquetes nvidia del mismo venv y libcuda desde
el driver 520.61.05. El resultado no mejoró. No hay evidencia para cambiar
torch/cuDNN, reinstalar CUDA, alterar determinismo o reordenar los imports.

## Interpretación causal y corrección operacional

[RLIMIT_AS](https://man7.org/linux/man-pages/man2/getrlimit.2.html) limita las
reservas de espacio virtual y puede causar ENOMEM aun con memoria física libre.
La concentración de seis fallos cerca del mismo techo, incluso a 160px, favorece
esta explicación frente a falta de VRAM o al cálculo de la pérdida. El techo es
exactamente 110% de la RAM solicitada: compatible con una política como
[VSizeFactor de Slurm](https://slurm.schedmd.com/slurm.conf.html#OPT_VSizeFactor),
pero aún falta consultar la configuración efectiva del scheduler. No se afirma
que ese parámetro esté confirmado ni que las seis observaciones prueben por sí
solas la causalidad.

La corrección del worker es una directiva `#SBATCH --mem=32G`, provisional y
sobrescribible al enviar el job. Solicita recursos legítimamente a Slurm; no
ejecuta setrlimit, ulimit de escritura ni cambia la fracción CUDA. No equivale
a pedir 32G de VRAM ni a demostrar que esa RAM sea el mínimo necesario.

El éxito comunicado de octubre a 160px no incluye límites efectivos ni log
original. No se puede afirmar que tuviera el mismo RLIMIT_AS. La corrección de
importación 756cdc0 ya estaba preservada en los seis procesos fallidos. Se
conserva sin repetirla ni atribuirle una regresión que Git no muestra.

## Único job de verificación preparado, no enviado

Usar `scripts/hypatia/yolo11m_cuda_verify.sbatch`, con las mismas variables de
ruta del diagnóstico. Solicita una GPU en el mismo nodo, dos CPU, 32G de RAM y
20 minutos. No añade módulos ni modifica límites. Registra la política Slurm
accesible y ejecuta el mismo microstep configured-loss-640 en un proceso nuevo.
Solo si pasa inicia **una** epoch del smoke oficial a 640, batch 1, AMP=False,
inicialización aleatoria, sin FAC, sin test y sin reintentos.

```bash
sbatch /path/to/yolo11m_cuda_verify.sbatch
```

Condiciones para evaluar la intervención:

1. Verificar reserva efectiva de 32G y nuevo RLIMIT_AS en ambos diagnosis.json.
   Si sigue en 13.2 GiB, la intervención no quedó aplicada; no anular el límite.
2. Comparar versiones, script, entorno, bibliotecas, flags y seed con el caso
   configurado de 792984. Cualquier otra diferencia limita la inferencia causal.
3. Exigir microstep completo (loss/backward/SGD), luego smoke completed con una
   epoch, batch observado 1 y recibo del ejecutor válido. Stderr vacío no basta.
4. Si alguno falla, preservar evidencia y detenerse; no repetir jobs ni cambiar
   automáticamente resolución, modelo, AMP, CPU, límites o protocolo.
5. Un éxito sintético no admite entrenamiento FAC ni congela el batch científico.

Las pruebas locales verifican el flujo con sustitutos offline: una sola llamada
al smoke tras un microstep exitoso, propagación del fallo y rechazo de salida
preexistente. La capacidad CUDA y la causalidad requieren esta ejecución Hypatia.

## Antecedentes conservados del diagnóstico inicial

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
