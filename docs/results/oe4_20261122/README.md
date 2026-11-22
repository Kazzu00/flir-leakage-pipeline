# Resultados finales OE4 — 22 de noviembre de 2026

Protocolo `oe4_closeout_v2`: seis configuraciones YOLO11m, 50 épocas, seed 42,
políticas histórica, aleatoria y C10, cada una con AUG_ON y AUG_OFF. Evaluación
reportada por el responsable en Hypatia: array Slurm `809088`; agregación `809094`.
Este cierre publica los bytes recibidos; no repite entrenamiento ni evaluación.

El test común contiene **372 imágenes/ocurrencias y 983 objetos anotados**,
comprobados en el `records.csv` local cuyo SHA coincide con el recibo oficial.
Su identidad es el par de hashes del campo `test`
del [recibo original](receipt.json): primero `test_receipt_sha256`, después
`test_records_sha256`. La coincidencia de los seis tests se exige en
[`aggregate()`](../../../src/flir_pipeline/closeout/oe4_runtime.py).
El recibo agregado local no contiene los conteos ni los seis recibos nativos.
El recibo local anterior del test tiene bytes distintos al remoto; se acredita
la igualdad del `records.csv`, no la del recibo de liberación. No se volvió a
auditar el payload de imágenes/GT ni los checkpoints.

> Descriptive one-seed internal comparison; no causal or independence claim

Es una comparación interna descriptiva de una semilla. No acredita causalidad,
independencia estadística, ausencia total de leakage ni superioridad general de
una política. Los tests de las variantes Sin HUD/Sin Ruido son diferentes y no
se mezclan con este test. [Entrega de datasets](../../final_datasets_20261122.md).

## Archivos y transformación exacta

- [comparison.csv](comparison.csv): original intacto, doce registros; única fuente
  numérica. SHA-256 `5e690f88ed9a9d9c4f820fd65e660394c99c8ae51109156d2d3646ee15da4bb4`.
- [receipt.json](receipt.json): original intacto, estado `AGGREGATED`, protocolo,
  identidad del test, checksum e interpretación.
- [oe4_final_metrics_20261122.csv](oe4_final_metrics_20261122.csv): seis registros,
  orden histórica ON/OFF, aleatoria ON/OFF, C10 ON/OFF; diez columnas, ocho métricas.

Las métricas permanecen en escala 0–1. El exportador conserva **cada token decimal
como texto**, sin convertirlo a float ni redondearlo, y comprueba la proyección
inversa de las seis filas a las doce celdas originales. No calcula nuevas métricas.
Git conserva los bytes de CSV/recibo mediante `.gitattributes`.

Desde la raíz, para verificar el archivo publicado:

```powershell
uv run --no-sync python -m flir_pipeline.closeout.final_metrics docs/results/oe4_20261122/comparison.csv docs/results/oe4_20261122/receipt.json docs/results/oe4_20261122/oe4_final_metrics_20261122.csv --check
```

Para reconstruirlo, sustituir la ruta de salida por un archivo nuevo y omitir
`--check`. El comando rechaza un CSV distinto del SHA oficial, cohortes/celdas
duplicadas o incompletas y valores no finitos/fuera de rango. Pruebas sintéticas:
`uv run python -m pytest tests/closeout/test_final_metrics.py -q`.
Para instalar el entorno, véase [REPRODUCE](../../../REPRODUCE.md).

Procedencia remota informada: `~/flir-oe4-recovery-20261122/aggregate/comparison.csv`.
Copia operativa local: directorio externo `00_ENTREGABLES/Resultados_OE4`, junto
con los originales y la tabla final. No se publican rutas personales absolutas.

## Aumentación y políticas

La política histórica conserva su pertenencia original y contaminación conocida;
aleatoria y C10 corresponden a las vistas congeladas de desarrollo registradas
por el preflight, manteniendo el test común. C10 es la candidata de agrupamiento
documentada; no equivale a las políticas S2/S3 ni al split denoised.

AUG_ON reutiliza los argumentos efectivos de cada entrenamiento baseline.
AUG_OFF conserva los controles metodológicos (incluidos modelo, épocas, semilla,
optimizador y datos) y pone a cero las claves `AUG_ZERO` de
[`oe4.py`](../../../src/flir_pipeline/closeout/oe4.py), incluidos HSV, transformaciones
geométricas, flips, mosaic, mixup, cutmix y copy_paste; `auto_augment=None` y
`multi_scale=False`. Los valores ON exactos se recuperan del freeze/`args.yaml`,
no se infieren de defaults. Ambos tratamientos se evalúan con `augment=False`.

## Definición de las métricas

[`capture_validator(0.25)`](../../../src/flir_pipeline/detection/runtime.py) filtra
predicciones a confianza ≥0,25 y vuelve a emparejarlas con GT a IoU 0,50. Se suman
TP y predicciones por clase sobre todas las imágenes. En
[`evaluate()`](../../../src/flir_pipeline/detection/metrics.py), Precision es
TP/predicciones (cero si no hay predicciones) y Recall es TP/objetos GT. Después
se toma la media aritmética por clase, sin ponderar por soporte. Una clase sin
soporte deja la macro correspondiente indefinida; nunca se sustituye por cero.

mAP@50 integra la curva precision–recall por clase a IoU 0,50 con interpolación
de 101 puntos. mAP@50–95 promedia AP sobre diez IoU, de 0,50 a 0,95. Se conservan
las detecciones a confianza mínima 0,001 del validador para AP; el filtro 0,25
corresponde únicamente a Precision/Recall. NMS: IoU 0,7, por clase, `max_det=300`.

- `five_class_macro`: media fija de las cinco clases.
- `shared_01_macro`: media de IDs 0 y 1, vehículo y edificio. `filter_shared()`
  restringe GT y predicciones después del NMS común de cinco clases; conserva
  el presupuesto `max_det=300`. No es otro modelo ni otra inferencia de dos clases.

`metric_exports()` genera ambas cohortes; `aggregate()` verifica los recibos y
reproduce las métricas desde `image_stats.npz` antes de escribir `comparison.csv`.
Ese es el procedimiento del productor, no una operación repetida en este cierre.

Los valores P/R de la consola Ultralytics **tienen otra definición**: en la
versión fijada 8.3.203, `ultralytics.utils.metrics.ap_per_class` selecciona
`smooth(f1_curve.mean(0), 0.1).argmax()` y extrae allí `p_curve`/`r_curve`.
Son valores al punto de máximo F1 medio suavizado, no P/R a confianza fija 0,25.
No se copian ni se mezclan con las macros del CSV oficial.

## Checkpoints y alcance verificable

La selección es `train/weights/last.pt` de la época 50; `best.pt` se conserva
pero no se utiliza para esta comparación. Las raíces de AUG_ON por política están
en [`configs/closeout/oe4_20261115.yaml`](../../../configs/closeout/oe4_20261115.yaml),
campo `runs`, bajo `~/flir-oe4-minimal-20261115/scientific-runs/`. AUG_OFF procede
de `~/flir-oe4-recovery-20261122/aug-off/{policy}/train/weights/last.pt`.

Cada recibo nativo de evaluación, en
`~/flir-oe4-recovery-20261122/evaluations/{AUG_ON|AUG_OFF}/{policy}/receipt.json`,
registra `weights_sha256`, `training_receipt_sha256`, `freeze_sha256` y la identidad
del test. El ejecutor valida metadatos/history de 50 épocas, argumentos y hashes.
**Estos seis recibos y hashes de pesos no están en los originales locales
entregados para este cierre**: no se inventan ni se afirma su verificación local.
El SHA verificable publicado aquí acredita el CSV, no por sí solo los checkpoints.
Recuperar sus recibos publicables sigue pendiente si se necesita esa auditoría
independiente; no requiere volver a evaluar. Pesos e imágenes permanecen privados.
