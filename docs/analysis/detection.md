# Detector: auditoría de candidatos y piloto observado

## Detector y contexto residual: infraestructura disponible

La capa `detection/association.py` y el
[explorador local](../visualization/detector_similarity.md) unen runs controlados
con el contexto residual congelado. El estado observado al implementar esta capa
es **PENDING: 0/48 controlled runs**, con 16 contextos y cuatro small pilots
excluidos. Esto acredita infraestructura; no añade resultados científicos.

La figura `09_metrics_vs_residual_similarity` conserva su nombre y muestra
PENDING COMPUTE junto al registro de siete asociaciones. Solo una matriz completa
y controlada permite publicar puntos o tablas de asociación. Las combinaciones
preespecificadas están en el [protocolo](../protocols/detection.md#asociaciones-preespecificadas-con-contexto-residual).

Asociación ≠ causalidad: los ejemplos de test, composición y dificultad cambian;
la similitud residual no se manipula de forma aislada. No se añade regresión,
significancia estadística, ranking de estrategias ni interpretación de los
pilotos como performance científica. Las tablas del análisis siguiente conservan
sus resultados anteriores.

## Alcance de los pilotos registrados

Esta fase acredita infraestructura y **cuatro pilotos pequeños en CPU**.
No acredita Stage A completo, Stage B, un detector final ni diferencias de
generalización entre estrategias. La decisión de detener el protocolo largo
sigue la instrucción de presupuesto del portátil, no resultados de mAP.

## Auditoría antes de YOLO

La colección de 66 splits pasó checksums, identidades, balance e invariantes.
Se inspeccionaron los seis candidatos Pareto y sus cinco semillas. Las tablas
completas de 30 filas están en `artifacts/detection/protocol/candidate_audit.csv`.
Los valores siguientes corresponden al representante seed 0, salvo estabilidad.

| Candidato | Encoder / representación / algoritmo | Clusters | Noise % | ARI / AMI parámetros | ARI / AMI semillas de reducción | Retención media de split entre seeds |
|---|---|---:|---:|---:|---:|---:|
| C01 | CLIP / original L2 / OPTICS | 2 | 97.40 | .533 / .544 | no aplica | .604 |
| C05 | CLIP / t-SNE / OPTICS | 21 | 46.26 | .679 / .818 | .714 / .853 | .781 |
| C07 | DINOv2 / original L2 / OPTICS | 24 | 67.58 | .719 / .772 | no aplica | .624 |
| C09 | DINOv2 / PaCMAP / HDBSCAN | 44 | 19.26 | .372 / .747 | .489 / .788 | .615 |
| C10 | DINOv2 / PaCMAP / DBSCAN | 34 | 3.22 | .856 / .925 | .755 / .877 | .893 |
| C12 | DINOv2 / t-SNE / HDBSCAN | 22 | 12.54 | .657 / .856 | .919 / .961 | .946 |

ARI/AMI mostrados son la versión all-points de las perturbaciones ya ejecutadas;
el tratamiento de noise, cobertura y versión common-clustered se conserva en el
CSV. No se equipara estabilidad de clustering con retención del nombre de split.

| Candidato | Registros train/val/test | Desviación clases pp | HM train/val/test | NN DINOv2 / CLIP | Pares top-0.1% DINOv2 / CLIP |
|---|---|---:|---|---:|---:|
| C01 | 1178 / 107 / 372 | .0398 | 92 / 8 / 29 | .8743 / .9566 | 229 / 311 |
| C05 | 1178 / 107 / 372 | .0398 | 92 / 8 / 29 | .8434 / .9444 | 75 / 57 |
| C07 | 1178 / 107 / 372 | .0398 | 92 / 8 / 29 | .8434 / .9517 | 65 / 66 |
| C09 | 1178 / 107 / 372 | .0463 | 92 / 8 / 29 | .8014 / .9442 | 33 / 56 |
| C10 | 1178 / 107 / 372 | .4694 | 95 / 7 / 27 | .8004 / .9431 | 6 / 7 |
| C12 | 1174 / 107 / 376 | 1.3756 | 87 / 8 / 34 | .7888 / .9377 | 7 / 14 |

Los seis mantienen cero exact overlap y cero fracturas. La desviación relativa
máxima de tamaño en seed 0 es cero salvo C12: 1.0753%. No se reemplaza ese tamaño
por un objetivo nominal durante materialización.

| Candidato | Temporal Δ≤1 % | Δ≤5 % | Δ≤10 % |
|---|---:|---:|---:|
| C01 | 27.33 | 33.02 | 35.81 |
| C05 | 20.31 | 26.97 | 30.27 |
| C07 | 21.42 | 25.83 | 29.67 |
| C09 | 16.83 | 25.38 | 29.35 |
| C10 | 15.16 | 23.93 | 29.31 |
| C12 | 13.77 | 21.36 | 25.57 |

**Decisión:** C10 principal, C12 segundo. C12 satisface las restricciones
documentadas para todas sus semillas: temporal Δ≤5 entre 20.85–21.61%, frente
a 22.13–24.46% de C10; desviación de clases entre 1.3034–1.3756 pp. Su retención
mínima de split es .8992. Sus NN medios son inferiores al histórico en ambos
encoders. A cambio, tiene peor balance y más pares CLIP top-0.1% que C10.

C05/C07 no mejoran conjuntamente NN medio respecto al histórico y su temporalidad
es superior al peor C10; C09 tampoco satisface la restricción temporal y tiene
retención mínima de split .5387. C01 conserva 97.4% noise y no mejora ambos NN;
queda como ablation descriptiva, sin entrenamiento. No se construyó un score.

El histórico tiene 198 contenidos exactos cross-split, 328/277 pares extremos,
NN medios .8410/.9486 y temporal Δ≤5 14.76%. C10/C12 reducen la dependencia visual
medida, pero no superan la temporalidad histórica en Δ≤5. Esa distinción se
conserva: no se afirma eliminación de todo leakage.

## Plan, vistas y hardware observados

Referencias: **C10 — DINOv2 / PaCMAP / DBSCAN**; **C12 — DINOv2 / t-SNE / HDBSCAN**.

Plan `ace1ffd6bb4775d0`; freeze del runtime CPU `9897062efdb32450`.
Dieciséis vistas corresponden a historical y cinco semillas de random_content,
C10 y C12. La matriz objetivo tiene 48 celdas con detector seeds 42/43/44.
El plan local contiene todos los IDs, hashes, ratios y clases, sin publicar
identidades privadas en Git. Los IDs seed 0 están en el protocolo.

Hardware detectado: AMD Ryzen 7 7730U, 8 cores/16 hilos, RAM utilizable
23,451,787,264 bytes (~21.84 GiB), unos 6.2 GiB disponibles al probe. Windows
expone AMD Radeon integrada; `torch.cuda.is_available()` es False y
`torch.version.cuda` es None. No se inventa VRAM CUDA. PyTorch 2.8.0+cpu,
torchvision 0.23.0+cpu, Ultralytics 8.3.203, Python 3.11.14.

Probe sintético a 640: batches 1 y 2 pasaron forward/backward/optimizer; el proceso
observado utilizó aproximadamente 497 y 637 MiB respectivamente. Son RSS
observados, no un pico muestreado de RAM. Se fijó **CPU, batch 2, imgsz 640**
antes del primer entrenamiento real; mismo YOLO11n pretrained en las cuatro
estrategias. Los ZIP originales permanecieron read-only.

## Pilotos pequeños ejecutados y coste

Referencias: **C10 — DINOv2 / PaCMAP / DBSCAN**; **C12 — DINOv2 / t-SNE / HDBSCAN**.

Cada estrategia usó 24/12/20 imágenes, dos epochs, detector seed 42 y split seed 0.
Se seleccionaron subconjuntos deterministas con cobertura de clases presentes
dentro de cada partición. Esa selección deliberada invalida interpretar sus
métricas como estimación de generalización del conjunto completo.

| Estrategia | Training observado s | Evaluación + bootstrap s | Estimación full 50 epochs h, por wall time del piloto |
|---|---:|---:|---:|
| historical | 30.49 | 14.42 | 10.39 |
| random_content | 30.20 | 14.34 | 10.29 |
| C10 | 51.06 | 14.65 | 17.40 |
| C12 | 28.64 | 14.05 | 9.73 |

Los cuatro entrenamientos suman **140.39 s**; evaluación/bootstrap suma **57.46 s**.
El primer piloto incluye mayor overhead. No se atribuye esa diferencia a la
estrategia de partición. Las duraciones de callbacks de epoch incluyen un
callback adicional de validación final; el cálculo alternativo usa solo el
índice de la última epoch de entrenamiento, sin contarlo dos veces.

Extrapolación por wall time total del piloto: **592.58 h** para 48 runs.
Alternativa con última epoch observada, reduciendo startup: **408.33 h**.
Equivale aproximadamente a **17–25 días continuos de cómputo**. No es un intervalo
de confianza ni una medición de entrenamientos completos; el escalamiento
lineal simplifica warmup, dataloader, carga de CPU y costes distintos de validation.
No incluye la evaluación/bootstrap final. Ambas estimaciones justifican detener
el lanzamiento automático de Stage B en este portátil.

## Validación y límites

Validado en los cuatro pilotos: checkpoints aislados; inicialización idéntica;
configuración efectiva; cinco clases test; todos los IDs test; conteos de objetos
del loader en train/val/test; métricas reconstruidas desde estadísticas por imagen;
1000 remuestreos bootstrap reproducidos exactamente; checksums de outputs.
Los labels vacíos y conflictos por ocurrencia permanecen intactos. La validación
histórica sigue sin Vehicles ni Heavy Machinery.

Verificación de software: Ruff sin errores; suite completa de 137 pruebas
aprobada; después se validaron las 14 pruebas específicas de detection,
incluido el guard CPU añadido al cierre (138 pruebas distintas en el árbol
final). Seis notebooks fuente limpios y compilables. HTML ejecutado sin errores,
18 secciones, nueve imágenes embebidas, código oculto y checksums comprobados.
Los cuatro pilotos tienen IDs distintos y `metrics.json`, métricas por clase,
`training_summary.json` y `metadata.json`, derivados sin reescribir el recibo
original de ejecución. No se entrenó dentro de pytest ni se usó Internet en tests.

No se publican métricas pequeñas como resultado final, ni sus CIs como evidencia
sobre las estrategias. No hay variabilidad de training seeds o split seeds
estimada por estos pilotos, que usan solo 42/0. No hay efecto del detector
cuantificado de forma controlada ni conclusión de mejora/disminución de mAP.

**No ejecutado:** Stage A completo a 50 epochs, Stage B, entrenamiento en GPU,
resume interrumpido en GPU y comparación científica multi-seed. El resume
implementado usa el mecanismo de Ultralytics y registra reanudaciones; no se
certifica equivalencia bit a bit con una ejecución ininterrumpida.

El notebook/HTML tiene 18 secciones y nueve archivos de figuras: una muestra
correlación residual observada; ocho paneles de métricas están explícitamente
pendientes. Siguiente paso: fijar un presupuesto final viable y una matriz común
antes de los resultados finales, o ejecutar el protocolo existente en un runtime
con capacidad confirmada. No cambiar condiciones individualmente por estrategia.
