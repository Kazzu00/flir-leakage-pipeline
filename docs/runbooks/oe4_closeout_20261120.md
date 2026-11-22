# Cierre local denoised y OE4

Este adaptador prepara datos y operaciones reproducibles. No acredita una ejecución
en Hypatia ni otorga aprobación científica. `CODEX_TASK.md` no estaba en el checkout;
se utilizó el brief adjunto `CODEX_BRIEF_FLIR_OE4_DENOISED_20261120.md`, contrastado
con código y artefactos. Base: integración `3ea7590`, descendiente de `main` y del
checkout inicial. Trabajo en `feat/oe4-closeout-denoised-20261120`; `main` intacto.

## Verificado localmente

El TAR real se leyó sin modificarlo. La salida externa contiene 1.116 imágenes,
1.116 labels y 1.116 filas únicas OK; se decodificaron todas las imágenes. Hay
1.116 contenidos únicos. Soporte de objetos por ID 0..4: 529, 3697, 1431, 190,
241 (6.088 total). No se observaron cajas fuera de rango ni overflow de bordes.
`split_original`: 803/151/162, únicamente metadata histórica.

Las dos versiones de clases difieren solo en el nombre de clase 4 y las dos del
CSV solo en su cabecera. Se preservan sus bytes, orden TAR y checksums en
`metadata_original/` y `members_inventory.csv`. `resumen.json` concuerda con
los conteos, familias y procedencia de ganadores; `faltantes.csv` contiene
cabecera y cero filas. La equivalencia semántica SDZI/maquinaria se atribuye a
Laura según el brief; no se adjudica ni se recodifican etiquetas.

Doce nombres del video2 difieren del CSV en 1 ms. Se conservan ambos valores y
se reportan en `audit.json`, con tolerancia explícita de una unidad de precisión
de milisegundo. No se afirma que sean timestamps nativos. El primer intento
fallido quedó conservado y separado; nunca se publicó como dataset íntegro.

El error inicial de los dos metadatos adicionales se reprodujo ejecutando el
importador preliminar externo sobre un TAR sintético antes de verificar la
corrección. Los tests permanentes incluyen metadatos duplicados, corrupción,
rutas peligrosas, enlaces y correspondencias inválidas; no consumen datos reales.

Se verificó la publicación local `five-seeds-20261013-a` con el contrato existente
y los ZIP consumidos. Se materializaron 372 ocurrencias únicas y 983 objetos:
12/589/270/93/19 por ID. Es una copia **MATERIALIZED_NOT_RELEASED**, sin evaluación.
No se afirma paridad con la copia remota hasta su preflight ni disyunción con los
tres entrenamientos remotos. Los archivos originales no se modificaron.

## Splits denoised

`closeout/splits.py` usa el MILP existente, conserva grupos indivisibles y verifica
restricciones después de asignar. No copia splits históricos ni crea anclas por
el nombre «test». Combina igualdad de contenido, identidad de frame reportada,
escenas nominales por video y dataset de origen, ventana temporal configurable y
relaciones de coseno de ambos encoders. Conflictos de anotación por contenido o
frame de origen ponen el componente entero en cuarentena. El video completo no
es una unidad de entrada; el cierre transitivo puede formar componentes grandes.

Configuración propuesta `denoised_dependency_candidate_v1`: 2 s, targets
70/15/15, seed 42 y sensibilidades de coseno 0.983503/0.988371 heredadas del kit.
Son umbrales propuestos, no calibrados para denoised. La extracción completa se
ejecutó localmente en CPU, offline, en un entorno externo instalado desde
`uv.lock`: DINOv2 1.116 × 384 y CLIP 1.116 × 512. Ambos stores tienen cobertura
completa, raw/L2, índices verificados y revisiones resueltas; no presentan
NaN, Inf ni normas cero. Se reutilizaron los wrappers y storage originales,
sin reutilizar embeddings históricos ni alterar los pesos cacheados.

El grafo completo forma 93 componentes y 1.170 relaciones: 1.002 temporales,
156 nominales y 12 visuales; el componente mayor contiene 229 registros.
No se detectaron grupos para cuarentena. Son relaciones conservadoras
reportadas o heurísticas; no ground truth de escenas.

**Split real materializado como CANDIDATE_NOT_FROZEN**: train 777, val 172,
test 167. Las cinco clases están presentes en las tres particiones. Hay cero
cruces de contenido exacto, frame de origen, grupos o relaciones registradas.
El MILP verificó factibilidad primal y alcanzó el límite de 128 nodos:
gap 0.2661, sin prueba de optimalidad ni de óptimo único. No se sustituyó por
una heurística. Una segunda ejecución reprodujo exactamente las asignaciones.
`Dataset Con Splits/` contiene imágenes, labels, YAML, membresías, grafo,
metadata, procedencia, checksums y evidencia del solver. La salida inicial
`split-eligibility/` sin features se conserva como intento histórico sustituido.

El replay independiente volvió a decodificar las 1.116 imágenes del TAR,
comparó los bytes de cada copia materializada y reconstruyó las 1.170 relaciones
y los 93 componentes sin importar el constructor de producción. Verificó
raw/L2 y hashes de ambos stores. Todas las comprobaciones pasaron. La auditoría
independiente del test común también confirmó las 372 ocurrencias y sus bytes.

C10 permanece como referencia: DINOv2-small CLS 384d/L2, PaCMAP 2D seed 0,
DBSCAN min_samples 10, eps 0.598 en su espacio/protocolo original. El q85 del kit
no es C10; no se ejecuta ni se transfiere eps a una nueva escala. Este adaptador
no reemplaza la infraestructura de reducción/clustering del proyecto. Anclar
denoised al test anterior sigue bloqueado hasta disponer de correspondencias
explícitas de origen y resolver ambigüedades; no existe freeze aprobado nuevo.

La auditoría local de correspondencias contra el catálogo original produjo
774 coincidencias nominales y 342 sin correspondencia nominal; 230 de las
coincidencias apuntan a candidatos del test. Ninguna se promovió a ancla verificada.
Se entregan las 1.116 excepciones con sus identidades candidatas en
`protected-anchor-audit-v2/`. La revisión inicial que no contemplaba que el
catálogo guarda basenames sin extensión se conserva como intento sustituido;
la versión v2 corrige únicamente esa comparación nominal y tiene test de regresión.

## Integración del kit

| Prototipo | Integración y corrección |
|---|---|
| preparar_denoised_flir | Importación transaccional estricta, cuatro metadatos reconocidos, evidencia de fallo, hashes y QA completo |
| make_embeddings | Adaptador al storage y encoders existentes; unique content, raw/L2, revisión resuelta y resume |
| split_denoised | Grafo explícito y MILP existente; sin normalización nominal de sufijos que invente identidad ni anclas ignoradas |
| materialize_common_test | Verificador de publicación, locadores ZIP originales, test portable, conteos y recibo |
| materialize_class_subset | Filtrado conjunto de GT/predicciones en estadísticas capturadas, sin recodificación |
| audit_runs | Preflight de recibos, argumentos completos, desarrollo, pesos, overlap y versiones |
| yolo_closeout | Workers GPU por Slurm, matrices congeladas, AUG_OFF y last epoch 50, vistas privadas para cachés |
| aggregate_test | Seis evaluaciones completas, mismo test/configuración, replay de métricas capturadas |

Se preservan APIs históricas. Los módulos nuevos reutilizan `verify_fixed_test`,
`source_bytes`, `verify_execution`, `write_view`, `capture_validator`, métricas
del detector y `milp_assignment`. Los tests del kit se sustituyen por pruebas
de invariantes con fixtures sintéticos; no se copia código exploratorio sin revisión.

## Listo para Hypatia: contrato y configuración

Todas las rutas están en `configs/closeout/oe4_20261115.yaml`, con `$HOME` y rutas
comunicadas en el brief. `FLIR_OE4_CONFIG` permite seleccionar otro archivo sin
editar scripts; `FLIR_OE4_PYTHON` selecciona un entorno detector existente. Por
defecto se usa `$HOME/.venvs/flir-detection/bin/python`. Se requieren Python 3.11,
Ultralytics 8.3.203, Torch 2.7.1 y torchvision 0.22.1. No se modifica ese entorno.

Preflight en login solo importa core, lee metadata y calcula hashes. Registra
READY o BLOCKED; los fallos se conservan en carpetas `freeze.blocked-*` y permiten
reintentar sin sobrescribir. Verifica publicación, ZIPs, test de 372, recibos de
las tres corridas, 50 épocas, argumentos completos, AUG_ON, pesos iniciales
oficiales y train/val efectivos. Una identidad no mapeable falla; no se adivina.
La auditoría distingue frame_id, contenido, grupos protegidos, relaciones
registradas y candidatos temporales. Ausencia de cruces conocidos no demuestra
independencia. Contaminación confirmada bloquea y queda reportada, sin alterar test.

Se reutiliza el recibo `official-weights.json` del protocolo existente; su ausencia
o incompatibilidad es un bloqueo de procedencia. No se descarga ni sustituye un
checkpoint automáticamente. La precondición remota no pudo verificarse localmente.
El preflight GPU comprueba versiones, CUDA y una operación dentro de Slurm.

Los tres baselines son **AUG_ON / validation epoch 50**, según el brief; sus
métricas remotas no se recertificaron localmente. Los nuevos entrenamientos
AUG_OFF parten del mismo `yolo11m.pt`, nunca de best/last. Todos los argumentos
no operacionales se conservan y deben coincidir entre brazos; los valores de
aumentación, incluidos bgr, cutmix, erasing y close_mosaic, quedan en cero;
auto_augment=None, multi_scale=False. `copy_paste_mode` es inerte con probabilidad
cero y se conserva. La configuración 8.3.203 se contrastó con su
[fuente oficial](https://github.com/ultralytics/ultralytics/blob/v8.3.203/ultralytics/cfg/default.yaml).

No se evalúa test hasta que terminen los tres AUG_OFF y exista una liberación
válida. La evaluación ejecuta los tres last originales y los tres last AUG_OFF
con idénticos parámetros: split=test, sin TTA, 640, batch 1, floor=.001,
NMS IoU=.7, class-aware, max_det=300, FP32; P/R fija a confidence .25.
El macro compartido 0/1 filtra tanto GT como predicciones después del NMS común;
mantiene el presupuesto de 300 detecciones de cinco clases. No es un modelo
reentrenado a dos clases. Las clases sin soporte permanecen indefinidas.

Dentro del job se inspecciona el checkpoint seleccionado por path/hash y se
contrasta su historia embebida con las 50 filas del CSV. Se admite `epoch=-1`
solo con esa historia completa, porque el
[stripping final de Ultralytics](https://github.com/ultralytics/ultralytics/blob/v8.3.203/ultralytics/utils/torch_utils.py)
reemplaza el marcador de época. El adaptador solo lee checkpoints; nunca aplica
stripping ni modifica originales. La tabla conserva el nombre original SDZI para
ID 4, sin adoptar la ontología denoised.

## Bloqueos científicos

La nueva ablación y su liberación no reciben automáticamente la autorización del
protocolo técnico histórico de 75 corridas. Se conserva el verificador de firmas
existente en `detection/admission.py`. Los documentos externos train/evaluate
deben incluir `protocol=oe4_closeout_v1`, `freeze_sha256` del recibo nuevo y los
campos de identidad/revisor/vigencia de `flir-scientific-admission-v1`. Evaluate
debe ligar además `aug_off_receipts`, mapping de las tres políticas a sus SHA256.
Los paths están preconfigurados bajo `approvals/`. No se crearon firmas, revisores,
ni `approved=true`. Si faltan autoridad o documentos válidos, submit falla antes
de solicitar GPU. Resolver una decisión científica no equivale a parchear scripts.

## Comandos definitivos

Transferir el bundle y su `.sha256` al HOME remoto. El PowerShell opcional
`deploy_to_hypatia.ps1 -Bundle ...` usa solo un alias Hypatia ya existente en SSH;
si es ambiguo requiere `-SshAlias` de esa configuración. Solo transfiere código.
No se ejecutó SSH, transferencia, SharePoint ni sbatch durante el trabajo local.

```bash
set -euo pipefail
cd "$HOME"
sha256sum -c flir-oe4-closeout-20261121-code.tar.gz.sha256
test ! -e flir-oe4-closeout-20261121-code
tar --keep-old-files -xzf flir-oe4-closeout-20261121-code.tar.gz
cd "$HOME/flir-oe4-closeout-20261121-code"
bash scripts/hypatia/preflight_oe4.sh
```

Solo con READY y admisión científica válida para la fase:

```bash
bash scripts/hypatia/submit_oe4.sh train
```

Tras completar las tres corridas y su liberación de evaluación:

```bash
bash scripts/hypatia/submit_oe4.sh evaluate
```

Tras completar el array de evaluación:

```bash
bash scripts/hypatia/aggregate_oe4.sh
squeue -u "$USER"
sacct --format=JobID,State,ExitCode,Elapsed
```

No usar `sacct --format=StdOut`. Los logs se crean antes del submit. No se borra,
reutiliza ni reanuda automáticamente una operación fallida; su evidencia se conserva.

La fase denoised es independiente de OE4 y ya tiene features y split candidato
materializados localmente. No requiere repetir extracción para usar esa copia.
El siguiente comando es solo para una reproducción nueva en GPU, en una raíz
de trabajo sin salidas finales previas. Después de transferir la carpeta
auditada `Dataset Sin Split` a `$HOME/flir-denoised-20261121/`, y con modelos
oficiales cacheados y dependencias vision en `FLIR_DENOISED_PYTHON`:

```bash
mkdir -p "$HOME/flir-denoised-20261121/logs"
sbatch --partition=gpu --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=04:00:00 --output="$HOME/flir-denoised-20261121/logs/%j.out" --error="$HOME/flir-denoised-20261121/logs/%j.err" scripts/hypatia/denoised_closeout.sbatch
```

Sin cachés/dependencias no se afirma que esa fase esté operativamente lista; el
job deja un reporte BLOCKED y preserva los checkpoints de extracción. Ninguno de
los comandos GPU se ejecutó localmente. El bundle no contiene datos ni modelos.

## Validación y paquete

```powershell
uv run --no-sync python -m pytest tests/closeout -q
uv run --no-sync python -m ruff check .
git diff --check
uv run --no-sync python scripts/closeout/build_bundle.py <salida-externa.tar.gz>
```

El paquete lleva código, configuraciones, tests, documentación, lockfile y un
manifiesto SHA256 portable. Tar/gzip usan orden/mtime/modos fijos. La auditoría
independiente `scripts/closeout/audit_outputs.py` vuelve al TAR y las copias sin
importar el importador de producción; verifica bytes, labels, soporte y constraints.
Las salidas de tests, auditoría, historial de errores y hashes reales se conservan
fuera de Git. El informe final distingue resultados de suite focalizada/global y
fallos observados; no se atribuyen automáticamente al código previo.

La suite global ejecutada produjo 1.629 aprobadas y tres fallos: dos por rutas
temporales largas en Windows y uno por identidad de source/gate durante la
ejecución. Los tres pasaron juntos al repetir con checkout estable y ruta corta
(46,69 s); no se presenta la primera ejecución como verde. La suite final del
adaptador tiene 49 pruebas aprobadas, incluye replay independiente que detecta
relaciones omitidas y conserva evidencia de un solver inviable. Ruff global
aprobado. Los scripts Bash se verifican también con ejecución simulada desde el
spool de Slurm y rutas con espacios; no equivale a una ejecución en Hypatia.

La auditoría real de preflight local verificó publicación/test y reportó BLOCKED
por ausencia local de los tres recibos remotos, del recibo de pesos oficiales y
del runtime detector. Las comprobaciones independientes no se cancelan por esos
faltantes. El bundle incluye la suite `tests/closeout`; la suite global del
repositorio se ejecuta y reporta por separado. Un paquete solo se genera desde
un checkout limpio con commit, para que el manifiesto no atribuya cambios sin
commit a una revisión anterior.
