# Estado del proyecto

Revisado **2026-09-09**. **Cierre de datos, trazabilidad, caracterización e ingeniería
de características**, contrastado con los compromisos hasta semana 6 suministrados
en la solicitud. No se recibió el documento íntegro ni el calendario fechado de
la propuesta; no se certifican compromisos adicionales ni su aprobación formal.

## COMPLETADO en este alcance

- Diseño inicial y arquitectura del pipeline.
- Inventario y auditoría de ZIP, imágenes y etiquetas, conservando los originales.
- Manifest canónico candidato v1, identificadores y mapping ocurrencia/contenido.
- Caracterización de registros, instancias por clase y partición histórica.
- Auditoría de duplicados exactos y conflictos de anotación.
- Caracterización temporal disponible, con heurística y confianza explícitas.
- Línea base reproducible de datos: pertenencia histórica preservada, sin nuevos splits.
- Diagnósticos sobre los 1459 contenidos únicos.
- Pipeline DINOv2/CLIP con checkpoints, metadata, raw/L2 y validaciones.
- **DINOv2-small completo: 1459 × 384**, con mapping de 1657 registros y revisión resuelta.
- **CLIP ViT-B/32 completo: 1459 × 512**, con mapping de 1657 registros y revisión resuelta.
- Reporte para JP en español: 16 secciones, notebook ejecutado local y HTML sin código visible.

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
| Clases | 0, 1, 2, 3, 4 | Anotaciones canónicas |
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

| Clase | Imágenes que contienen la clase | Instancias / cajas |
|---|---:|---:|
| 0 | 55 | 92 |
| 1 | 685 | 2295 |
| 2 | 619 | 1010 |
| 3 | 401 | 642 |
| 4 | 121 | 129 |

Ambas columnas cuentan anotaciones de ocurrencias históricas. Una imagen con
varias cajas de una clase aporta una presencia y varias instancias.

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
`reports/feature_engineering_closure/`. Datos, arrays, hashes de contenido,
notebook ejecutado y HTML permanecen fuera de Git.
