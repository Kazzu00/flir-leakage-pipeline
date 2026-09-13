# Estado del proyecto

Revisado **2026-09-13**. **Semanas 6, 7 y 8 COMPLETED** y **semana 9: coseno DONE,
análisis temporal PARTIAL**, según la evidencia descrita a continuación y los criterios
explícitos de preparación y representaciones de la solicitud de esta revisión.
El cierre anterior agrupaba ambas extracciones bajo «hasta semana 6»; aquí se
distinguen preparación (6), DINOv2 (7) y CLIP/comparación descriptiva (8).
El documento íntegro y el calendario fechado de la propuesta no forman parte del
repositorio; no se certifican compromisos adicionales ni su aprobación formal.

| Semana | Estado | Evidencia ejecutada / revalidada |
|---|---|---|
| 6 — preparación y caracterización | COMPLETED | Inventario actualizado; manifest/identidades; labels; duplicados; nombres de clases; 4168 instancias; cuatro métricas de bbox por clase; empty annotations; 1459 diagnostics; procedencia temporal disponible |
| 7 — DINOv2 | COMPLETED | implemented + smoke validated históricamente + full extraction completed; 1459 × 384, 1657 mappings y revisión efectiva revalidados |
| 8 — CLIP y comparación descriptiva | COMPLETED | implemented + smoke validated históricamente + full extraction completed; 1459 × 512 y 1657 mappings revalidados; comparación descriptiva sin ranking |
| 9 — similitud y correlación descriptiva | COSINE DONE / TEMPORAL PARTIAL | Dos matrices completas verificadas; pares, top-20, análisis temporal/histórico, Jaccard y HTML ejecutado; temporalidad aún inferida |

Las extracciones completas ya existían desde 2026-09-09. Esta revisión verifica
arrays, índices, cobertura y metadata; **no vuelve a extraer embeddings**.
La referencia bibliográfica exacta de la nomenclatura sigue pendiente. El orden
de clases fue confirmado explícitamente por el responsable del proyecto;
la [evidencia y su límite](dataset_classes.md) se conservan sin inventar una cita.

## Semana 9: evidencia completa de similitud

Se reutilizaron los embeddings completos, después de verificarlos nuevamente
contra el manifest. **No se ejecutó extracción adicional** en esta fase.
Para cada encoder se guardaron y verificaron una matriz float32 **1459 × 1459**,
**1063611 pares únicos i<j** y **29180 vecinos dirigidos top-20** (rangos 1–20,
sin self-neighbors). Se preserva el mapping a las 1657 ocurrencias.

| Resultado | DINOv2 | CLIP |
|---|---:|---:|
| similarity_space_id | fb754789ca5c0115 | 05d98f0f34211249 |
| Media / mediana global | 0.460609 / 0.472356 | 0.823653 / 0.830576 |
| Desviación estándar poblacional | 0.195429 | 0.070360 |
| Mediana rank 1 | 0.933983 | 0.971477 |
| Rank 1 en misma secuencia inferida | 1453/1459 = 99.5888% | 1448/1459 = 99.2461% |
| Top-20 en misma secuencia inferida | 28657/29180 = 98.2077% | 28512/29180 = 97.7108% |
| Candidatos cross-split en top 0.1% | 328 de 1064 | 277 de 1064 |

Jaccard medio entre encoders: **0.281700 / 0.287730 / 0.289354 / 0.294129**
para k=1/5/10/20. Rank 1 coincide exactamente en **28.169979%** de consultas.
No se infiere superioridad de un encoder.

Los 1459 contenidos tienen consenso de secuencia/índice inferidos en sus
ocurrencias: 531747 pares de misma secuencia con Δ conocido y 531864 de distinta
secuencia con Δ nulo. Hay cero conflictos/desconocidos en esta ejecución, pero
**cero timestamps verificados**. Las medianas por los rangos de Δ poblados
disminuyen de 0.890798 (Δ=1) a 0.522038 (Δ>100) en DINOv2 y de 0.954006 a
0.849469 en CLIP. Es una asociación descriptiva agregada, sin causalidad ni FPS.

Se revisaron los pares de máximo coseno (0.995615 / 0.996466): IDs, hashes de
bytes y píxeles RGB distintos; misma secuencia inferida y Δ=1 en ambos casos.
Ningún par está dentro de la tolerancia numérica |cos−1| ≤ 1e−6. Este diagnóstico
no define un umbral semántico de near-duplicate o leakage.

El [registro de semana 9](similarity_analysis.md) contiene percentiles, rangos de
Δ, los seis cuantiles de candidatos, Jaccard, reglas y comandos reproducibles.
Evidencia local: `artifacts/similarity/` y `reports/similarity/`.
El HTML `reports/similarity/review/similarity_review.html` tiene 14 secciones y
11 figuras, incluidas consultas compartidas con semilla 0 y revisión de máximos.
Se comprobaron ejecución sin errores, código oculto, ausencia de rutas privadas
y hashes de contenido en texto, y figuras legibles. Artefactos e imágenes están
ignorados por Git; el notebook versionado carece de outputs.

La fase descriptiva solicitada está cerrada. La validación espaciotemporal
permanece **PARTIAL**, porque la cobertura nominal no valida tiempos reales ni
coherencia visual global. No se implementaron ni ejecutaron reducción,
clustering, nuevas particiones o YOLO; Bhattacharyya sigue condicionado.

## COMPLETADO en este alcance

- Diseño inicial y arquitectura del pipeline.
- Inventario y auditoría de ZIP, imágenes y etiquetas, conservando los originales.
- Manifest canónico candidato v1, identificadores y mapping ocurrencia/contenido.
- Caracterización separada de imágenes e instancias por clase con nombres canónicos.
- Estadísticas de ancho, alto, área y ratio normalizados por instancia/clase;
  boxplots de área y ratio por clase, conservando extremos.
- Auditoría de duplicados exactos y conflictos de anotación.
- Caracterización temporal disponible, con heurística y confianza explícitas.
- Línea base reproducible de datos: pertenencia histórica preservada, sin nuevos splits.
- Diagnósticos sobre los 1459 contenidos únicos.
- Pipeline DINOv2/CLIP con checkpoints, metadata, raw/L2 y validaciones.
- **DINOv2-small completo: 1459 × 384**, con mapping de 1657 registros y revisión resuelta.
- **CLIP ViT-B/32 completo: 1459 × 512**, con mapping de 1657 registros y revisión resuelta.
- Reporte técnico en español: 16 secciones, notebook ejecutado local y HTML sin código visible.
- Similitud coseno completa por encoder y revisión independiente de 14 secciones;
  análisis posterior de secuencia, Δ y pertenencias históricas, y comparación Jaccard.

El estado completo de features exige que ambos espacios pasen
`features verify --manifest ...`. Los smoke N=16 no satisfacen esa condición.

## Evidencia agregada verificada

| Cantidad | Resultado | Universo |
|---|---|---|
| Registros / frame_id únicos | 1657 | Candidato canónico |
| content_id únicos | 1459 | Bytes originales exactos |
| Train / val / test histórico | 1178 / 107 / 372 | Ocurrencias |
| Etiquetas asociadas válidas / faltantes | 1657 / 0 | Candidato |
| Anotaciones vacías / no vacías | 292 / 1365 | 17.62% / 82.38% |
| Clases | Vehicles (0), Buildings (1), Roads (2), Rivers (3), Heavy Machinery (4) | Anotaciones canónicas; nombre original de 4: SDZI |
| Etiquetas con geometría inválida | 0 | Candidato |
| Objetos del candidato | 4168 | 1657 etiquetas asociadas |
| Etiquetas huérfanas / objetos excluidos | 10 / 14 | Fuera del manifest |
| Etiquetas / objetos del archivo completo | 1667 / 4182 | Incluye huérfanas |
| Grupos duplicados exactos / ocurrencias involucradas | 198 / 396 | Candidato |
| Exceso de registros sobre contenidos únicos | 198 | 1657 − 1459 |
| Solapamiento exacto train–val / train–test / val–test | 57 / 141 / 0 | Contenidos compartidos |
| Validación / test con copia exacta en train | 53.27% / 37.90% | 57/107 y 141/372 |
| Grupos con anotaciones consistentes / conflictivas | 190 / 8 | Duplicados preservados |

**4168 + 14 = 4182**: los dos totales de objetos corresponden a universos distintos.
El builder relee las etiquetas en memoria, verifica sus SHA256 y recuenta cada caja.
No usa estas cifras documentales como fuente de verdad ni corrige anotaciones.

| Clase | Imágenes que contienen la clase | Instancias / cajas | % de instancias |
|---|---:|---:|---:|
| Vehicles (0) | 55 | 92 | 2.2073 |
| Buildings (1) | 685 | 2295 | 55.0624 |
| Roads (2) | 619 | 1010 | 24.2322 |
| Rivers (3) | 401 | 642 | 15.4031 |
| Heavy Machinery (4), original SDZI | 121 | 129 | 3.0950 |

Ambas columnas cuentan anotaciones de ocurrencias históricas. Una imagen con
varias cajas de una clase aporta una presencia y varias instancias.

| Clase | Área normalizada: media / mediana | Ratio normalizado: media / mediana |
|---|---:|---:|
| Vehicles (0) | 0.004799 / 0.002711 | 0.671457 / 0.550129 |
| Buildings (1) | 0.051098 / 0.036322 | 0.861200 / 0.806626 |
| Roads (2) | 0.109588 / 0.062671 | 1.283726 / 0.673839 |
| Rivers (3) | 0.121339 / 0.095755 | 1.504121 / 1.083354 |
| Heavy Machinery (4) | 0.145936 / 0.129796 | 1.062787 / 1.007319 |

`bbox_geometry_by_class.csv` contiene también ancho/alto, count, std muestral,
Q1, Q3, min y max para las cuatro métricas: 20 filas, 4168 instancias válidas,
cero exclusiones geométricas. El ratio usa ancho/alto YOLO normalizados, no
dimensiones en píxeles. Las diferencias observadas no establecen causalidad.
Background describe los 292 labels vacíos, sin agregar clase ni cajas.

## Evidencia de preparación y disponibilidad actual

| Criterio de semana 6 | Evidencia |
|---|---|
| Inventario / originales | Inventario nuevo en `reports/academic_revision/inventory/`, con lectura y hashing; originales preservados |
| Validación / manifest | 1657 labels asociados, válidos, hashes/clases/conteos contrastados; 10 huérfanos separados |
| frame_id / content_id / dataset_id | 1657 ocurrencias únicas, 1459 contenidos; identidad del manifest consistente con ambos espacios y diagnostics |
| Mapping / split histórico | Índices completos contrastados con manifest; `historical_baseline.csv` preserva los 1657 registros |
| Duplicados | 198 grupos, 8 conflictos visibles; solapamiento exacto histórico recontado |
| Clases / instancias / geometría | YAML original validado, catálogo central con nombres originales y canónicos; tabla de instancias y 20 estadísticas por clase/métrica |
| Vacías / diagnostics | 292 vacías y 1365 no vacías; cobertura y finitud de diagnostics sobre los 1459 contenidos |
| Temporalidad | Secuencia/índice inferido y ausencia de timestamps verificados explícitos |
| Trazabilidad | Configuración de métricas, hashes locales, commit, fuentes y reporte ejecutado; notebook público sin outputs |

El inventario histórico enumeraba seis ZIP. Actualmente hay cinco: falta el
archivo adicional `video_13min_778.zip`; no se ha eliminado durante esta tarea.
Los cinco ZIP presentes coinciden con los hashes del inventario histórico.
Los originales canónicos `Imagenes.zip` y `Etiquetas.zip` están disponibles;
el candidato y los espacios completos conservan su cobertura. Se preserva el
inventario anterior y se documenta la diferencia de disponibilidad.

## Procedencia temporal disponible

| Secuencia inferida | Registros | Contenidos | Rango de índices | Confianza |
|---|---:|---:|---|---|
| video_11min | 910 | 712 | 1–712 | medium |
| video_13min | 747 | 747 | 1–781 | medium |

Los 1657 registros tienen secuencia e índice inferibles por nombre; 0 carecen de
ese orden inferido y **0 tienen timestamps verificados**. No se certifican dos
videos fuente ni se infieren segundos o FPS. El rango no implica muestreo uniforme.

Regla: mismo archivo/secuencia, splits distintos, diferencia de índice ≤ 1,
incluidos índices iguales. Produce **598 pares candidatos**: 198 con bytes
idénticos y 400 de contenido distinto. Los últimos son proximidad nominal;
no constituyen leakage confirmado. Este análisis de preparación es distinto de
los pares de contenidos por coseno añadidos en semana 9. La validación
espaciotemporal completa sigue necesitando mejor procedencia temporal.

## Espacios completos y smoke conservados

| Extractor | Modelo | N | Dimensión | Pooling | feature_space_id |
|---|---|---:|---:|---|---|
| DINOv2 | facebook/dinov2-small | 1459 | 384 | cls_token | c6df9d274f46cca7 |
| CLIP | openai/clip-vit-base-patch32 | 1459 | 512 | projected_pooler_output | 585246e6ed6c4cf8 |

Ambos pasaron verificación de arrays, finitud, normas no nulas, consistencia
raw/L2, índices únicos y completos, mapping de los 1657 registros, identidad del
dataset y revisión efectiva HF. Los vectores no tienen que ser todos diferentes.

Revalidación 2026-09-13: raw/normalized sin NaN ni Inf; zero_norm_count=0;
L2, shapes, content_index, record_index, unicidad de IDs, metadata y cobertura
completa válidos. Metadata conserva processor config, pooling, float32, semilla,
revisión resuelta, Python/torch/Transformers, commit y created_at originales.

Se ejecutaron primero **16 contenidos adicionales de validación por encoder**
con las revisiones fijadas y salida separada en `artifacts/features_revision_smoke/`.
Los smoke anteriores N=16, con revisión `unknown`, permanecen sin cambios.
Un mismo feature_space_id puede identificar la muestra y el completo; la selección
se controla con la firma de caché y las raíces separadas.

Se reutilizaron snapshots locales, en CPU, batch 8, sin precisión mixta.
Python 3.11.9, torch 2.14.0, Transformers 5.16.1. Tiempos observados de los comandos
completos: DINOv2 169.84 s; CLIP 128.27 s. No son un benchmark.

Los modelos y el backend de extracción procedían del commit existente al ejecutar;
los YAML fijados estaban aún sin commit. Se conservan ese commit en metadata y
los hashes de configuración en el recibo local, sin reescribir la procedencia.
El commit de cierre versiona las configuraciones, auditorías y reporte.
Ver [ejecución reproducible](week6_closure.md).

## PENDIENTE / SIGUIENTE

Validación adicional de procedencia temporal/coherencia visual; Bhattacharyya
cuando exista una representación distribucional justificada; t-SNE; PaCMAP; DBSCAN; OPTICS;
HDBSCAN; evaluación de estabilidad/coherencia, AMI/ARI y selección de agrupamiento;
partición por clústeres y baseline aleatorio; entrenamiento comparativo;
evaluación y análisis/reproducibilidad final. No se ejecutaron estas etapas.

Los controles numéricos no prueban calidad semántica ni mejora de detección.
La línea base de datos no implica que un detector baseline esté entrenado.

Evidencia local de semanas 6–8: `reports/feature_engineering/` y
`reports/feature_engineering_closure/`, más `reports/academic_revision/`.
La revisión actual usa un entorno uv aislado, Python 3.11.14, core/dev/reporting
del lockfile; no modifica los metadatos históricos de extracción en Python 3.11.9.
Datos, arrays, hashes de contenido,
notebook ejecutado y HTML permanecen fuera de Git.

## Validación previa: revisión de semanas 6–8

- `uv run --no-sync pytest`: **47 passed**, offline y sintéticos.
- `uv run --no-sync ruff check .`: **All checks passed**.
- `uv run --no-sync python scripts/check_notebook_source.py`: fuente limpia y código compilable.
- `uv run --no-sync python scripts/build_feature_engineering_review.py`: notebook
  ejecutado y HTML generados; se comprobó ausencia de errores y código oculto,
  más inspección visual de las cuatro figuras de clases/instancias/geometría.
- CLI de inventario actualizado y verificación de ambos encoders contra manifest:
  salida correcta. En esta máquina el launcher `flir-pipeline.exe` fue bloqueado
  por Windows; se invocó el mismo Typer app mediante
  `uv run --no-sync python -c "from flir_pipeline.cli import app; app()" ...`.

Los comandos anteriores usaron `UV_PROJECT_ENVIRONMENT=.venv-academic` tras
`uv sync --locked --offline --extra dev --extra reporting` con Python 3.11.14.
`uv` se instaló localmente en `.cache/uv-tool`; los entornos anteriores apuntaban
a un Python ya no disponible y se conservaron. La instalación no cambió `uv.lock`.
Recibo local: `reports/academic_revision/verification_receipt.json`, junto con
logs de CLI y el inventario actualizado. En ese cierre la siguiente etapa era
similitud/correlación; la evidencia de semana 9 se añadió después, sin reducción,
clustering, splits nuevos ni YOLO.

## Validación de semana 9

- `uv run --no-sync pytest`: **64 passed**, offline, sintéticos, sin modelos/GPU.
- `uv run --no-sync ruff check .`: comprobación de código y notebooks fuente.
- Checker de ambos notebooks: sin outputs privados ni errores de compilación.
- CLI `similarity compute`, `summary`, `verify`, `compare` y `--help`; los verify
  completos contrastan matrices con arrays L2 originales y procedencia canónica.
- Reporte construido con core/dev/reporting, sin cargar modelos; revisión visual
  de las 11 figuras y comprobaciones programáticas del HTML ejecutado.
- Mismo entorno uv aislado, Python 3.11.14. El launcher Windows se sustituye por
  `uv run --no-sync python -c "from flir_pipeline.cli import app; app()" ...`.
- Metadata conserva el commit realmente activo, worktree dirty y hashes de fuente
  al computar. El recibo local de validación registra el código final sin atribuir
  retrospectivamente las extracciones o el cálculo inicial a otro commit.
