# Fase 2A: alineación visual multisplit

Este módulo propone correspondencias de procedencia entre imágenes originales y
cuadros de un vídeo. No estima ground truth, no declara secuencias independientes,
no utiliza etiquetas YOLO y no modifica el ledger Human Audit v1.1.
Los resultados observados están en el [informe de validación de Fase 2A](../analysis/phase2a_multisplit_validation.md).
La referencia metodológica es el [protocolo piloto v0.1](human_review_pilot_protocol_v01.md),
leído como instrucción de trabajo. El piloto de 24 juicios humanos y la medición
de acuerdo entre revisores son posteriores; no se simulan con resultados automáticos.

## Ejecución

```powershell
uv sync --extra dev --extra alignment
$facRoot = Read-Host 'Directorio FAC original (solo lectura)'
uv run flir-pipeline sequences alignment run --root $facRoot --video Videos/VideoCodefest_008-46min.avi --output reports/phase2a-plan --sample-only
uv run flir-pipeline sequences alignment run --root $facRoot --video Videos/VideoCodefest_008-46min.avi --output reports/phase2a-run --audit-db reports/audit-review-real/audit.sqlite3
uv run flir-pipeline sequences alignment verify reports/phase2a-run
```

Abrir `reports/phase2a-run/index.html` localmente. No subir imágenes ni informes
reales al repositorio o a servicios externos. No es necesario modificar el frontend:
el informe enlaza láminas, alternativas, controles y procedencia. Es una superficie
local de inspección; las acciones React de validación quedan fuera de esta entrega.

Para comparar las cuatro fronteras en láminas de cuatro paneles, en otro directorio:

```powershell
uv run python scripts/report_phase2a_diagnostics.py --root $facRoot --evidence reports/phase2a-run --output reports/phase2a-diagnostics
```

Este informe vuelve a decodificar los PTS seleccionados y compara sus hashes RGB
con los candidatos guardados. No juzga continuidad ni dependencia de región.

`--dry-run` equivale a `--sample-only`: inventaría y selecciona sin decodificar vídeo.
`--config archivo.json` permite modificar presupuestos y parámetros explícitos del
modelo `alignment.models.Config`. Defaults: semilla 20261103, 12 por split,
máximo 60 imágenes, 90.000 cuadros decodificados y 1.800 segundos de búsqueda.
El inventario y hashing previos no se incluyen en el presupuesto de decodificación;
sus costos deben distinguirse del costo de matching.

## Inventario y muestra

Se recorren directorios y tablas de miembros ZIP; solo se leen imágenes cuyos
nombres completos cumplen `video_13min_N.(jpg|jpeg|png)`. Los nombres aumentados
quedan contados como exclusiones. Las etiquetas no se abren. Cada ocurrencia conserva
archivo, miembro y ordinal físico, split, índice, dimensiones, SHA-256 y anomalías.
Archivos corruptos y splits desconocidos quedan explícitos. Los nombres repetidos
no se fusionan por índice ni por igualdad de bytes.

La selección inicial usa exclusivamente `Imagenes.zip`, después de comprobar que
es la procedencia con la partición histórica esperada. Los otros ZIP permanecen
en el manifiesto, sin sustituir silenciosamente esa fuente. Se ordena por índice
dentro de cada split, se divide en 12 estratos de igual cantidad y se elige una
ocurrencia aleatoria por estrato. Se añaden 199, 200, 349, 350, 399, 400, 499, 500
y 501 si existen. Faltantes y déficits se exportan. Esta muestra dirigida no estima
desempeño poblacional; el split no interviene en ninguna medida de similitud.

## Búsqueda y soporte

- `1978 + N` es únicamente el centro de búsqueda nominal. Ventana inicial ±8 s,
  intervalo solicitado 0,5 s. La tolerancia es una decisión operacional conservadora,
  no una distribución de error temporal estimada.
- Controles en tiempo cero y a ±120 s del centro; si no hay soporte geométrico,
  búsqueda ampliada ±60 s con intervalo solicitado de 3 s. Los controles con soporte
  también pueden generar alternativas fuera de ventana. No hay búsqueda global
  exhaustiva: `not_found` solo describe el dominio y presupuesto inspeccionados.
- Refinamiento de dos modos separados temporalmente a tasa nativa en ±0,3 s.
  Los vecinos y los cinco primeros candidatos se conservan en láminas.
- Resize 384×216, región interior fija (x=38:346, y=22:194) para reducir HUD,
  CLAHE, correlación normalizada, distancia dHash y SIFT (hasta 500 puntos).
  Matching L2 con razón 0,75 y homografía RANSAC de 3 píxeles.
- Se publican medidas independientes: correlación, huella, correspondencias,
  inliers, proporción, cobertura espacial y residual. Ranking lexicográfico,
  no combinación ponderada convertida en probabilidad. Umbrales exploratorios:
  12 inliers, proporción ≥0,35, cobertura ≥0,03; no calibrados con datos FAC.
  dHash se conserva para evaluar filtrado, pero no descarta automáticamente cuadros:
  no se calibró una poda segura frente a cambios térmicos o de tono. Esta elección
  conservadora aumenta el costo de SIFT y evita afirmar validado un filtro perceptual.
- Modos separados por más de 2 s con al menos 80% del soporte del mejor generan
  `ambiguous`. Esto no prueba unicidad cuando no se detectan alternativas. Un zoom,
  tono o HUD compartido puede dificultar o sesgar la búsqueda: revisión requerida.

Los controles temporales lejanos no son negativos verdaderos de identidad de región.
Los negativos sintéticos tienen construcción conocida. Los controles reales se
documentan con PTS, soporte y láminas para inspección, sin inventar etiquetas.

## Tiempo y reproducibilidad

PyAV hace seek hacia atrás a un keyframe y decodifica secuencialmente. Cada cuadro
retiene `pts`, `time_base`, su producto en segundos, duración reportada, tiempo
solicitado y primer tiempo realmente decodificado. `decode-log.json` registra
PTS devueltos y pasos efectivos, que pueden diferir de la frecuencia solicitada.
Los timestamps localizan cuadros del medio, no prueban el instante exacto de captura
del JPEG. El intervalo candidato abarca soporte observado; no es un intervalo de
confianza estadística. No se transforma el muestreo 5 FPS previo en tiempo exacto.

En el AVI local, `ffprobe` expone `best_effort_timestamp` mientras omite `pts` en
la salida de cuadros inspeccionada. El valor coincide con `frame.pts` de PyAV:
es tiempo del medio reconstruido por el decodificador, no un reloj original de
captura. La comparación independiente y su tolerancia se conservan en el reporte.

El run ID vincula inventario, selección, vídeo, configuración, hashes del código,
versiones de librerías y ledger de referencia. Repetir el mismo run verifica y devuelve
los archivos existentes. Cambiar parámetros o entradas exige otro directorio; no se
sobrescriben ejecuciones. Las láminas incluyen originales, candidatos y vecinos;
los controles tienen lámina aparte. `checksums.json` cubre los archivos publicados.
Se comprueban stats de originales y hash del ledger antes/después.

## Contratos y verificación humana

`alignment-evidence-v1` tiene JSON Schema raíz y CSV con `case_json` íntegro para
roundtrip. `manifest.json`, `selection.json` y los hashes enlazan cada caso a su
imagen. Los enlaces a IDs de Human Audit son **sugerencias por índice**, con base
explícita `index_reference_candidate_only_not_visual_identity`. No se adjunta una
lámina a un evento automáticamente ni se modifican sus TemporalObservation.

El pipeline solo emite `candidate`, `ambiguous` o `not_found`. `human_verified`
pertenece al contrato separado `alignment-human-review-v1` y exige una acción
humana explícita: revisor, justificación, localizador, lámina consultada y PTS de un
candidato documentado. Se registra como archivo independiente y vinculado por hash:

```powershell
uv run flir-pipeline sequences alignment review reports/phase2a-run decision.json reports/phase2a-reviews/revision-001.json
```

La CLI valida pertenencia e integridad; no puede certificar que una persona miró
la imagen ni sustituir su juicio. Correcciones usan otro archivo, conservando la
revisión anterior. No ejecutar este comando con anotaciones inventadas por el agente.
Para continuidad y región, aplicar los dos ejes del protocolo sin inferir uno del otro.

## Pruebas

```powershell
uv run ruff check .
uv run python -m pytest tests/test_alignment.py tests/test_audit_review.py tests/test_audit_v11.py tests/test_smoke.py -q
```

Las pruebas de alineación necesitan el extra CPU `alignment`; sin él se omiten
explícitamente, no se cuentan como aprobadas. CI instala ese extra. Fixtures
sintéticos con FFV1 prueban PTS, transformaciones, negativos, ambigüedad, inventario,
integración con SQLite, exportaciones, límites, corrupción y revisión humana.
