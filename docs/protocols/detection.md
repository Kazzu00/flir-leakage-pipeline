# Protocolo preespecificado de comparación del detector

Fecha de definición: 2026-09-16, antes de cualquier entrenamiento del detector
en esta fase. Al definir el protocolo no existía una arquitectura previamente
configurada. Se adopta **YOLO11n**, una sola
arquitectura, por la preferencia explícita del proyecto y el presupuesto del
portátil. No se comparan arquitecturas ni se modifica el preprocessing original.

## Pregunta y límites de interpretación

¿Cómo cambian las métricas del detector al mantener íntegros los grupos
visualmente correlacionados? Historical, random y cluster-aware tienen ejemplos
distintos en test. Las diferencias mezclan composición/dificultad, entrenamiento
y correlación residual; **no identifican un efecto causal del leakage**.
No se presupone una dirección de cambio ni se selecciona una métrica posterior.

## Selección previa al detector

Referencias: **C10 — DINOv2 / PaCMAP / DBSCAN**; **C12 — DINOv2 / t-SNE / HDBSCAN**; **C01 — CLIP / original L2 / OPTICS**.

Se revisan C01/C05/C07/C09/C10/C12 del frente de splitting existente, sin
reoptimizar asignaciones. C10 se conserva como candidato visual principal.
Para un segundo candidato, los cinco split seeds deben mantener: cinco clases
por partición, cero duplicados exactos cross-split, cero fracturas, desviación
relativa de tamaños ≤2%, desviación media de clases ≤2 pp, noise <80%, retención
mínima entre pares de semillas ≥60%, peor correlación temporal Δ≤5 no superior
al peor C10, y NN medio y pares top-0.1% de ambos encoders inferiores al histórico.
Son restricciones descriptivas explícitas elegidas al revisar splits, antes de
observar YOLO; no umbrales inferenciales ni una suma ponderada de métricas.

**C12** es el único segundo candidato que satisface estas restricciones. Su
ventaja temporal y de similitud media compensa un peor balance; no domina C10
en todos los ejes. C01 (97.4% noise) queda como referencia descriptiva de balance.
El archivo local `candidate_audit.csv` conserva las 30 filas, ARI/AMI, noise,
HM, tamaños, clases, NN, pares, temporalidad Δ≤1/5/10 y estabilidad de asignación.
Los controles originales no tienen una semilla de reducción ficticia.

Representantes split seed 0:

| Estrategia | split_space_id | Representación / agrupación |
|---|---|---|
| historical | 9cbd4bf45de58e21 | partición original, excepción histórica de duplicados |
| random_content | c2e75560b458bddd | asignación reproducible de content_id íntegro |
| C10 | 88ccf4e12335a83f | DINOv2 → PaCMAP → DBSCAN |
| C12 | 79a8ca09669699fc | DINOv2 → t-SNE → HDBSCAN |

El `plan.json` local congela todos los IDs, dataset, clustering, hashes de
asignaciones, noise singleton, ratios, objetos por clase y registros por split.
Se verifica identidad/checksums/invariantes de los 66 splits de origen. No
cambia ninguna asignación después de observar métricas del detector.

## Matriz y métricas

Referencias: **C10 — DINOv2 / PaCMAP / DBSCAN**; **C12 — DINOv2 / t-SNE / HDBSCAN**.

Diseño objetivo: historical × detector seeds 42/43/44; random_content, C10 y C12
× split seeds 0/1/2/3/4 × detector seeds 42/43/44: **48 celdas independientes**.
La misma lista de semillas aplica a todas las estrategias. Una eventual
reducción de presupuesto requiere una revisión explícita del protocolo antes
del experimento final, basada en coste, nunca en resultados favorables.

Primaria: macro mAP@50–95. Secundarias: macro mAP@50, Precision y Recall.
Dominio: Heavy Machinery Recall y mAP@50; se presentan además sus otras dos
métricas y las cuatro métricas de cada clase. Macro significa las cinco clases
fijas. Si falta soporte, se informa indefinido; no se reemplaza por cero ni se
reduce silenciosamente el conjunto de clases. Para clases presentes sin
predicciones, P/R/AP se definen como cero.

P/R se evalúan a confidence **0.25**, IoU **0.5**, fijados ahora. AP integra
predicciones con confidence ≥0.001, IoU de matching 0.50:0.05:0.95, NMS IoU 0.7
y max_det 300. P/R tienen un matching separado tras el filtro 0.25. El P/R
«best F1» que imprime Ultralytics es diagnóstico de la biblioteca, no la métrica
principal del reporte; elegir ese umbral sobre test violaría este protocolo.
Se comprueba paridad de AP con la biblioteca fijada, tolerancia 0.001 por empates.

Bootstrap por imagen: 1000 remuestreos con reemplazo, seed 20260916, intervalo
percentil 95%. Se recalcula AP sobre detecciones agrupadas de cada remuestreo,
no se promedian AP por imagen. Se cuentan remuestreos sin soporte de alguna
clase; sus métricas indefinidas no se convierten en cero. Esta aproximación no
corrige la dependencia residual entre fotogramas y puede subestimar incertidumbre.

Se conservan resultados por run, estadísticas count/mean/std muestral/median/
min/max por detector seed dentro de split, y variabilidad entre medias de
split seeds por separado. Se reportan diferencias descriptivas frente a
historical, CIs por run y dispersión, sin pruebas indiscriminadas de p-values.

## Configuración y hardware

Fuente versionada: `configs/detection/yolo11n.yaml`. Ultralytics 8.3.203,
PyTorch 2.8.0 y torchvision 0.23.0, en extra opcional `detection`. Pretrained
YOLO11n del asset oficial, hash SHA256 registrado localmente. Cada run nuevo
inicia desde esos mismos pesos, nunca desde otro split.

Configuración candidata: 50 epochs completas, imgsz 640, SGD, lr0 0.01,
lrf 0.01, schedule lineal, momentum 0.937, weight_decay 0.0005, warmup 3,
patience 0, FP32, AMP desactivado, deterministic=True, TF32 desactivado,
workers 0, resolución fija y ninguna transformación adicional de los originales.
Todas las augmentations y pérdidas están explicitadas en YAML; los restantes
defaults quedan vinculados al hash del archivo de la biblioteca y a su versión.
El preprocessing interno del detector y las augmentations son iguales para todos.

**No se asume GPU, CUDA ni VRAM.** Antes del primer entrenamiento real se
registran CPU, núcleos, RAM total/disponible, torch.cuda.is_available(), build
CUDA de PyTorch, GPU y VRAM total/libre cuando existan. Un probe sintético de
forward/backward/optimizer a imgsz 640 ensaya batches CPU [1,2] o CUDA [1,2,4,8].
Se selecciona el mayor probado con margen de memoria: CUDA <70% de VRAM libre
inicial; CPU >2 GiB RAM disponible. No se usa una cifra de VRAM inventada.
El probe no representa aún escenas reales densas ni el coste del dataloader.

`runtime_freeze.json`, creado antes de datos reales, fija device, batch elegido,
imgsz, pesos, versiones, defaults y configuración. Device y batch forman parte
del espacio experimental. Las rutas, fechas y nombre de máquina no entran al
detector_run_id. CUDA/CPU jamás se mezclan en una misma comparación final.

## Protocolo escalonado y presupuesto del portátil

1. Smoke opcional sintético en CPU: comprueba API, parser, cinco clases y
   bootstrap; no representa el dataset FLIR ni selecciona hiperparámetros.
2. Probe de hardware/batch, seguido del freeze operacional.
3. **Piloto pequeño real de infraestructura**, seed 42 y split seed 0 por
   estrategia: 24 imágenes train, 12 val y 20 test, seleccionadas reproduciblemente
   dentro de su partición original para cubrir clases presentes; 2 epochs,
   imgsz y batch congelados. Mosaic no se cierra en este piloto corto. Su muestreo
   deliberado de cobertura y tamaño impide interpretarlo científicamente.
   Se conserva mapping a las asignaciones originales, métricas, hardware,
   tiempos, checkpoints aislados y recibos. Nunca se incorpora a Stage A/B.
4. Extrapolación explícita de tiempo por imagen/epoch y matriz 48 runs. Incluye
   overhead del piloto y excluye evaluación final; es estimación, no tiempo medido
   del experimento completo. En CPU **Stage B automático está deshabilitado**.
   Se reporta coste y se detiene antes de decenas de entrenamientos.
5. Con capacidad suficiente: Stage A completa a 50 epochs, split seed 0 y
   detector seed 42 en las cuatro estrategias; gate de outputs, cobertura,
   metadatos, inicialización, clases y epochs. No se interpreta aislado como
   comparación final. Si falla, se corrige infraestructura, no hiperparámetros
   para mejorar métricas. La automatización GPU ejecuta Stage B solo tras el gate.
6. Stage B recorre las 48 celdas. Las cuatro celdas idénticas del piloto completo
   pueden reutilizarse una vez tras su validación; el piloto pequeño nunca se
   reutiliza. Resume solo desde last.pt del mismo run con identidad verificada.

Una GPU en otro equipo no se presupone accesible. Un futuro protocolo final CPU
reducido debe fijar su matriz antes de resultados finales y conservar controles
idénticos por estrategia; esta revisión no autoriza cambiar condiciones por split.

## Datos, selección y trazabilidad

Las vistas generadas reutilizan un almacén de imágenes y labels originales por
frame_id, fuera de Git. Se leen ZIPs en modo read-only, verificando hashes del
manifest. No se colapsan annotations conflictivas por content_id. El YAML usa
0 Vehicles, 1 Buildings, 2 Roads, 3 Rivers, 4 Heavy Machinery; sin Background.
Los vacíos permanecen vacíos. Cada vista se valida contra el split congelado.

Validation elige best.pt por mAP@50–95; test solo se evalúa después del training.
La validación histórica carece de Vehicles y Heavy Machinery: se conserva esa
limitación, no se corrige ni se atribuye soporte inexistente. Las comparaciones
de selección de checkpoint están también condicionadas por esa composición.
La selección exacta de best_epoch se registra durante training; el CSV redondeado
se conserva como comprobación auxiliar, no como fuente exacta de empates.

Por run: identidad, dataset/split/config/seed, versiones, CPU/GPU/CUDA/RAM,
epochs/best_epoch, batch, resolución, optimizer, tiempo, commit y hashes del
código, pesos, métricas, cinco clases, bootstrap y estadísticas por imagen.
Todos estos artifacts son locales e ignorados. El chequeo de comparación exige
la matriz completa y los mismos controles; outputs parciales nunca certifican
una comparación controlada completa.

## Fuentes de la integración

- [YOLO11, documentación oficial](https://docs.ultralytics.com/models/yolo11/).
- [Configuración de entrenamiento](https://docs.ultralytics.com/modes/train/).
- [Validador fijado a 8.3.203](https://github.com/ultralytics/ultralytics/blob/v8.3.203/ultralytics/models/yolo/detect/val.py).
- [Definiciones de métricas fijadas a 8.3.203](https://github.com/ultralytics/ultralytics/blob/v8.3.203/ultralytics/utils/metrics.py).

Este documento preespecifica el experimento; por sí solo no acredita ejecución
del piloto, Stage A, Stage B ni una conclusión sobre generalización.
