# Data quality and canonical audit

Observed evidence from the existing canonical candidate and local audit receipts.
No source annotations are repaired. See [class nomenclature](dataset_classes.md).

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

| Control de preparación | Evidencia |
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
archivo adicional `video_13min_778.zip`; su ausencia ya consta en el inventario registrado.
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
los pares de contenidos por coseno del análisis de similitud. La validación
espaciotemporal completa sigue necesitando mejor procedencia temporal.
