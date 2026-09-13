# Estado del proyecto

Revisado **2026-09-13**. **Semanas 6, 7 y 8 COMPLETED**, según los criterios
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

Las extracciones completas ya existían desde 2026-09-09. Esta revisión verifica
arrays, índices, cobertura y metadata; **no vuelve a extraer embeddings**.
La referencia bibliográfica exacta de la nomenclatura sigue pendiente. El orden
de clases fue confirmado explícitamente por el responsable del proyecto;
la [evidencia y su límite](dataset_classes.md) se conservan sin inventar una cita.

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
no constituyen leakage confirmado. La correlación espaciotemporal completa
requiere la siguiente fase y mejor validación de procedencia cuando sea posible.

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

Similitud entre fotogramas (coseno por espacio); Bhattacharyya cuando exista una
representación distribucional justificada; t-SNE; PaCMAP; DBSCAN; OPTICS;
HDBSCAN; evaluación de estabilidad/coherencia, AMI/ARI y selección de agrupamiento;
partición por clústeres y baseline aleatorio; entrenamiento comparativo;
evaluación y análisis/reproducibilidad final. No se ejecutaron estas etapas.

Los controles numéricos no prueban calidad semántica ni mejora de detección.
La línea base de datos no implica que un detector baseline esté entrenado.

Evidencia local: `reports/feature_engineering/` y
`reports/feature_engineering_closure/`, más `reports/academic_revision/`.
La revisión actual usa un entorno uv aislado, Python 3.11.14, core/dev/reporting
del lockfile; no modifica los metadatos históricos de extracción en Python 3.11.9.
Datos, arrays, hashes de contenido,
notebook ejecutado y HTML permanecen fuera de Git.

## Validación de esta revisión

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
logs de CLI y el inventario actualizado. La siguiente etapa sigue siendo
similitud/correlación; no se ejecutaron reducción, clustering, splits nuevos ni YOLO.
