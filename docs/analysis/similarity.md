# Similitud visual y relaciones temporales

Ejecutado y revisado 2026-09-13. **Cosine similarity DONE**: ambos espacios
completos, artefactos verificados, análisis posterior, comparación y HTML.
**Temporal correlation PARTIAL**: análisis descriptivo completo sobre procedencia
inferida; no hay timestamps verificados ni auditoría global de coherencia visual.
No se ejecutaron reducción, clustering, nuevos splits ni detector.

## Universo, identidad y reglas

Se reutilizaron los embeddings completos existentes, comprobados contra el
manifest canónico: 1459 contenidos distintos por bytes y 1657 ocurrencias.

| Encoder | Modelo / representación | feature_space_id | similarity_space_id |
|---|---|---|---|
| DINOv2 | facebook/dinov2-small; CLS, 384D | c6df9d274f46cca7 | fb754789ca5c0115 |
| CLIP | openai/clip-vit-base-patch32; imagen proyectada, 512D | 585246e6ed6c4cf8 | 05d98f0f34211249 |

Cada matriz es **1459 × 1459 float32**; la fórmula N(N−1)/2 da **1063611
pares únicos i<j**. Cada tabla top-20 contiene **29180 relaciones dirigidas**,
1459 consultas, rangos 1–20 y cero self-neighbors. Labels, cajas, secuencia y
split histórico no intervienen en el producto ni en el ranking. La metadata se
incorpora después. Los vectores raw/L2 originales no se modifican.

La identidad del análisis incluye dataset, feature space, métrica, k, dtype,
cuantiles, rangos de Δ, tolerancias y versiones de las reglas. Excluye fecha,
rutas y parámetros de dispositivo/batch. Los archivos de entrada y la procedencia
posterior llevan firmas adicionales: cambiar un split o secuencia no cambia el
producto, pero impide reutilizar anotaciones históricas de una caché anterior.

La matriz se calcula como X·Xᵀ sobre L2 validado, sin renormalización ni clipping.
La tolerancia numérica es 1e−5. Empates exactos de vecinos se resuelven por ID de
contenido ascendente. Todos los resúmenes usan desviación estándar poblacional
(`ddof=0`) y cuantiles lineales. No se asume independencia entre pares.

## Distribución global y vecindarios

| Estadística global | DINOv2 | CLIP |
|---|---:|---:|
| Count | 1063611 | 1063611 |
| Media | 0.460609 | 0.823653 |
| Std poblacional | 0.195429 | 0.070360 |
| Mínimo | -0.110048 | 0.459454 |
| Q1 | 0.320457 | 0.779949 |
| Mediana | 0.472356 | 0.830576 |
| Q3 | 0.604691 | 0.874900 |
| Máximo | 0.995615 | 0.996466 |
| p90 | 0.709861 | 0.909305 |
| p95 | 0.765594 | 0.927237 |
| p97.5 | 0.811430 | 0.939901 |
| p99 | 0.865117 | 0.951965 |
| p99.5 | 0.901112 | 0.959410 |
| p99.9 | 0.954069 | 0.975326 |

Las escalas son diferentes; un coseno mayor no demuestra superioridad.

| Métrica por contenido | DINOv2 media / mediana | CLIP media / mediana |
|---|---:|---:|
| Rank 1 | 0.917865 / 0.933983 | 0.970054 / 0.971477 |
| Media top-5 | 0.889516 / 0.903604 | 0.961138 / 0.963023 |
| Media top-10 | 0.869556 / 0.882238 | 0.954960 / 0.957319 |
| Media top-20 | 0.844330 / 0.856698 | 0.947123 / 0.950471 |

Rank 1: std **0.060614 / 0.014817**; Q1 **0.891644 / 0.962277**;
Q3 **0.961483 / 0.979993**; mínimo **0.527662 / 0.895283**;
máximo **0.995615 / 0.996466** (DINOv2 / CLIP). Las tablas locales contienen
count, todas las medidas anteriores y percentiles para cada resumen top-k.

## Secuencia y distancia de índices

Todas las ocurrencias de cada contenido concuerdan en archivo, secuencia e
índice inferido. **1459/1459** contenidos utilizables, cero ambiguos/desconocidos
y **cero timestamps verificados**. La confianza nominal es medium.
`video_11min`: 712 contenidos, índices 1–712. `video_13min`: 747 contenidos,
índices 1–781. Los nombres no certifican dos videos ni un FPS.

| Relación | Pares | DINOv2 media / mediana / Q1 / Q3 | CLIP media / mediana / Q1 / Q3 |
|---|---:|---|---|
| Misma secuencia | 531747 | 0.517329 / 0.539150 / 0.378593 / 0.675182 | 0.846198 / 0.855149 / 0.806220 / 0.897457 |
| Distinta secuencia | 531864 | 0.403902 / 0.418064 / 0.279904 / 0.535281 | 0.801113 / 0.807915 / 0.761431 / 0.849627 |

Rank 1 en misma secuencia: **1453/1459 (99.588759%)** DINOv2 y
**1448/1459 (99.246059%)** CLIP. Top-20: **28657/29180 (98.207676%)** y
**28512/29180 (97.710761%)**. El denominador conocido coincide con el total;
la API conserva ambos y reporta desconocidos para otras ejecuciones.

Δ es la diferencia absoluta de índices inferidos para una misma secuencia
con índice válido. Para secuencias distintas/desconocidas queda nulo. Los rangos
se eligieron tras inspeccionar diferencias observadas entre 1 y 780. El bin 0
permanece como diagnóstico vacío, no se inventan pares para llenarlo.

| Δ | Pares | DINOv2 media / mediana | CLIP media / mediana |
|---|---:|---:|---:|
| 0 | 0 | — | — |
| 1 | 1438 | 0.795438 / 0.890798 | 0.929447 / 0.954006 |
| 2–5 | 5721 | 0.696524 / 0.749633 | 0.901481 / 0.920373 |
| 6–10 | 7068 | 0.635047 / 0.670933 | 0.883329 / 0.895770 |
| 11–25 | 20847 | 0.586489 / 0.618540 | 0.869167 / 0.882188 |
| 26–50 | 33772 | 0.553433 / 0.588011 | 0.860492 / 0.873421 |
| 51–100 | 64101 | 0.535804 / 0.563895 | 0.854759 / 0.865808 |
| >100 | 398800 | 0.502027 / 0.522038 | 0.840660 / 0.849469 |

Las medianas disminuyen entre estos rangos poblados en ambos espacios. La amplia
dispersión y similitudes altas a Δ grandes permanecen visibles en hexbins y
cuartiles locales. Esto no demuestra una ley temporal, causalidad ni leakage.

## Pertenencias históricas y cuantiles

Conjuntos por contenido: `{train}` 980; `{test}` 231; `{val}` 50;
`{train,test}` 141; `{train,val}` 57. No se asigna un único split a los últimos.
Cross-split significa que existe una ocurrencia de cada contenido en particiones
distintas; también puede ocurrir entre conjuntos multi-split iguales.

Todos los pares siguientes tienen content_id distintos. Las cohortes incluyen
coseno ≥ umbral propio del encoder, conservan empates y son **anidadas**.
No deben sumarse. Cada nivel tiene cero casos de secuencia desconocida,
relación cross-split desconocida o procedencia insuficiente en esta ejecución.

| Encoder | Top % | Umbral | Pares | Misma secuencia | Distinta secuencia | Cross-split candidatos |
|---|---:|---:|---:|---:|---:|---:|
| DINOv2 | 10 | 0.709861 | 106362 | 98567 | 7795 | 51851 |
| DINOv2 | 5 | 0.765594 | 53181 | 52197 | 984 | 25231 |
| DINOv2 | 2.5 | 0.811430 | 26591 | 26507 | 84 | 11717 |
| DINOv2 | 1 | 0.865117 | 10637 | 10637 | 0 | 3751 |
| DINOv2 | 0.5 | 0.901112 | 5319 | 5319 | 0 | 1487 |
| DINOv2 | 0.1 | 0.954069 | 1064 | 1064 | 0 | 328 |
| CLIP | 10 | 0.909305 | 106362 | 99159 | 7203 | 51190 |
| CLIP | 5 | 0.927237 | 53181 | 52399 | 782 | 24710 |
| CLIP | 2.5 | 0.939901 | 26591 | 26484 | 107 | 11509 |
| CLIP | 1 | 0.951965 | 10637 | 10621 | 16 | 3852 |
| CLIP | 0.5 | 0.959410 | 5319 | 5315 | 4 | 1594 |
| CLIP | 0.1 | 0.975326 | 1064 | 1064 | 0 | 277 |

Son **high-similarity cross-split candidates**, sin confirmación automática de
leakage. Esta caracterización no cuantifica su efecto en un detector.

## Acuerdo de vecindarios

Comparación `4cc4e5505aaa556f`, misma población, consultas alineadas por ID.

| k | Jaccard medio | Mediana | Q1 | Q3 |
|---|---:|---:|---:|---:|
| 1 | 0.281700 | 0 | 0 | 1 |
| 5 | 0.287730 | 0.25 | 0.111111 | 0.428571 |
| 10 | 0.289354 | 0.25 | 0.111111 | 0.428571 |
| 20 | 0.294129 | 0.25 | 0.142857 | 0.428571 |

Rank 1 coincide exactamente en **28.169979%** de las consultas. La discrepancia
describe vecindarios distintos; no indica cuál es mejor. No se combinaron
embeddings ni se añadió una métrica opcional de acuerdo de rankings.

## Revisión visual y límites

Tres consultas compartidas, seleccionadas sin reemplazo sobre IDs ordenados,
`numpy.random.default_rng(0)`, cada una con top-5. El muestreo no es estratificado;
las tres consultas seleccionadas pertenecen a `video_11min`. No representa una
validación humana completa ni cobertura visual de ambas secuencias. Las imágenes
conservan overlays y, en algunos casos, menús; no se aisló su influencia.

Se inspeccionaron también los máximos. DINOv2: **0.995615**, índices nominales
1/2; CLIP: **0.996466**, índices 689/690; misma secuencia inferida y Δ=1.
En ambos pares son distintos los IDs, hashes de bytes y píxeles RGB decodificados.
El máximo de CLIP tiene pertenencias `{train,val}` en ambos contenidos y relación
cross-split; el de DINOv2 solo `{train}`. Los IDs/hashes concretos están en la
tabla local de inspección, excluidos del texto del reporte y del repositorio.
Hay **cero pares** con |cos−1|≤1e−6. Esa tolerancia numérica no es un cutoff
semántico de near-duplicate; los máximos se revisan aunque no la alcancen.

## Reproduction

Commands, inputs, cache behavior and verification are in the
[similarity runbook](../runbooks/similarity.md).

## Artefactos y validación

Local por encoder:
`artifacts/similarity/<encoder>/<dataset_id>/<feature_space_id>/<similarity_space_id>/`.
Incluye matriz, nearest_neighbors, índices, provenance por contenido/ocurrencia,
pair_analysis, resúmenes top-k por contenido/globales, near_unit_pairs, tablas de
secuencia/Δ/split/cuantiles, snapshot de features, metadata, checksums y quality.
La comparación contiene Jaccard por consulta/k y resumen global. Ninguno se versiona.

Fuente: `notebooks/similarity_review.ipynb`. Outputs locales:
`reports/similarity/review/similarity_review.executed.ipynb` y
`reports/similarity/review/similarity_review.html`. El HTML ejecutado presenta
14 secciones, 11 figuras y código oculto, sin rutas privadas ni hashes de contenido
en el texto visible. Las imágenes originales no se copian individualmente;
las composiciones generadas también quedan ignoradas.

Figuras en `reports/similarity/figures/`:

1. `01_global_similarity_distribution_dinov2.png`
2. `02_global_similarity_distribution_clip.png`
3. `03_rank1_similarity_comparison.png`
4. `04_same_vs_different_sequence_similarity.png`
5. `05_similarity_vs_frame_delta_dinov2.png`
6. `06_similarity_vs_frame_delta_clip.png`
7. `07_cross_split_high_similarity_quantiles.png`
8. `08_neighbor_agreement_dinov2_clip.png`
9. `nearest_neighbors_dinov2.png`
10. `nearest_neighbors_clip.png`
11. `maximum_similarity_pairs.png`

Validación: **64 tests passed**, offline/sintéticos; Ruff y checker de dos
notebooks fuente; CLI y verificación completa de ambos espacios; revisión visual
de figuras y comprobación programática del HTML. Los tests cubren incluso que
cambiar metadata posterior no modifica similitudes/ranking, aunque invalide
la caché, y que una imagen alterada se rechaza al generar la galería.

El alcance de este análisis termina en similitud. Los resultados posteriores
de [reducción](reduction.md), [clustering](clustering.md) y [particiones](splitting.md)
se documentan por separado. La comparación completa del detector sigue pendiente
según [status](../status.md). **Bhattacharyya PLANNED / CONDITIONAL** exige una representación
distribucional explícitamente justificada.
