# Análisis de particiones — semanas 10–11

Ejecución completa del protocolo principal: **66 runs verificados**, sobre
**1459 contenidos y 1657 registros** por run. Una baseline histórica, cinco
random content-level y 12 candidatos × cinco semillas. No se recalcularon
embeddings ni se entrenó un detector. Los originales permanecieron read-only.

**Conclusión observada:** C10 reduce correlación visual residual bajo ambos
encoders frente al histórico y a random, con tamaños y cobertura válidos. No
mejora todas las relaciones temporales frente al histórico. C01 es el ancla de
balance de clases, pero no demuestra una mejora visual conjunta frente al
histórico. Ninguno es un ganador universal ni una validación del detector.

## Construcción y procedencia

La comparación local es `a19ba503f085f569`. Se reutilizaron las matrices
DINOv2 `fb754789ca5c0115` y CLIP `05d98f0f34211249`, y la colección Pareto de
clustering `3609392b792b668b`. Los targets se derivaron del manifest:
train **0.7109233554616777**, val **0.06457453228726615**, test
**0.22450211225105612**: 1178/107/372 registros.

Noise policy **singleton**, semillas **0–4** en random y cada candidato. MILP
SciPy 1.17.1/HiGHS con perfiles intercambiables, balance L1 normalizado de
registros, instancias y vacíos, 32 nodos y gap solicitado 0.001. De 60 solves,
35 terminaron óptimos dentro de tolerancia y 25 como incumbentes factibles;
gap máximo observado **0.5854542093**. No se atribuye optimalidad a estos últimos.
El código HiGHS 16 fue expuesto como status 4 por SciPy; se comprobaron todas
las restricciones primales antes de aceptar el incumbente. Se preservó el
diagnóstico original. La reproducción posterior reconstruyó exactamente las
**65 asignaciones nuevas**, con el código final, sin reutilizar el solver cache.

La formulación, reglas de diversidad y selección están en el
[protocolo](splitting_protocol.md); comandos en el [runbook](splitting_runbook.md).
Las 4168 instancias, 292 labels vacíos y ocho grupos con conflictos de anotación
se revalidaron. No se corrigieron labels ni se eligieron etiquetas representativas.

## Doce candidatos evaluados

Dos por encoder × representación desde los 41 Pareto previos; algoritmo faltante
y maximin de rangos complementan el ancla de menor ruido. No se seleccionó por
silhouette. Los controles originales se mantienen como ablation de representación.


| Candidato | Encoder | Representación | Algoritmo | clustering_space_id | Noise | Clusters |
|---|---|---|---|---|---|---|
| C01 | clip | original_l2 | optics | 0208da483a57db96 | 0.973955 | 2 |
| C02 | clip | original_l2 | dbscan | bf8bb9fceae6f259 | 0.071967 | 8 |
| C03 | clip | pacmap | hdbscan | 06938d9d76d03bde | 0.256340 | 9 |
| C04 | clip | pacmap | dbscan | 895e5dccdfb6c938 | 0.000000 | 4 |
| C05 | clip | tsne | optics | 1c94a28a079022d2 | 0.462646 | 21 |
| C06 | clip | tsne | dbscan | 707b3a23c8f203b9 | 0.038382 | 2 |
| C07 | dinov2 | original_l2 | optics | e3ad8a743e6b55e2 | 0.675805 | 24 |
| C08 | dinov2 | original_l2 | dbscan | ed4aab613e943b33 | 0.000685 | 2 |
| C09 | dinov2 | pacmap | hdbscan | 9db01bfbc49efaaf | 0.192598 | 44 |
| C10 | dinov2 | pacmap | dbscan | f11bb99234c9de13 | 0.032214 | 34 |
| C11 | dinov2 | tsne | dbscan | 2097fab0f4a8f61c | 0.014393 | 2 |
| C12 | dinov2 | tsne | hdbscan | 854148e2dbb27bfd | 0.125428 | 22 |

## Validez, Pareto y candidatos finales

Los 65 nuevos runs tienen cero contenidos exactos cross-split. Los 60
cluster-aware tienen cero fracturas de los clusters utilizados. El histórico
conserva **198 contenidos exactos cross-split**, 57 train–val y 141 train–test;
396 ocurrencias implicadas. Un histórico auditado no satisface los invariantes
de una partición nueva y se maneja como excepción explícita.

Seis candidatos satisfacen cobertura de las cinco clases en todos los splits y
tolerancia relativa de registros ≤10% en todas las semillas. Los seis permanecen
en el Pareto de ocho criterios; los otros seis se conservan en resultados con
su exclusión. C03 conserva tamaños, pero pierde cobertura; otros grupos dominantes
producen particiones muy desbalanceadas. Sus ceros de correlación no justifican
seleccionarlos. Los valores siguientes son los **peores entre cinco semillas**.


| Candidato | Elegible | Pareto | Desv. tamaño relativa | Desv. clases pp | Top0.1% DINO fracción | Top0.1% CLIP fracción |
|---|---|---|---|---|---|---|
| C01 | True | True | 0.000000 | 0.039754 | 0.252820 | 0.298872 |
| C02 | False | False | 0.774194 | 12.015711 | 0.005639 | 0.003759 |
| C03 | False | False | 0.000000 | 1.932062 | 0.164474 | 0.149436 |
| C04 | False | False | 0.774194 | 14.685152 | 0.000000 | 0.000000 |
| C05 | True | True | 0.000000 | 0.044942 | 0.070489 | 0.064850 |
| C06 | False | False | 0.709677 | 19.805727 | 0.000940 | 0.007519 |
| C07 | True | True | 0.000000 | 0.039754 | 0.076128 | 0.090226 |
| C08 | False | False | 0.997312 | 13.333333 | 0.000000 | 0.000000 |
| C09 | True | True | 0.000000 | 0.309925 | 0.033835 | 0.059211 |
| C10 | True | True | 0.009346 | 1.137127 | 0.005639 | 0.012218 |
| C11 | False | False | 0.785047 | 13.488099 | 0.000000 | 0.000000 |
| C12 | True | True | 0.010753 | 1.375578 | 0.006579 | 0.015977 |

Las anclas finales fijadas por la regla son:

- **C10**, DINOv2 → PaCMAP → DBSCAN: `f11bb99234c9de13`, 34 clusters,
  noise 3.2214%; mínimos del peor top 0.1% en ambos encoders entre elegibles.
  Representante seed 0: **`88ccf4e12335a83f`**.
- **C01**, CLIP original → OPTICS: `0208da483a57db96`, dos clusters,
  noise 97.3955%; ancla de menor desviación de clases. Representante seed 0:
  **`643cd594f431cfc8`**. Agrupa solo 38 contenidos; su utilidad como control de
  balance no demuestra que singleton noise resuelva correlación residual.

La semilla representante fue prefijada como 0, no elegida por sus métricas.
El frente completo también conserva C05, C07, C09 y C12. C12 tiene menor fracción
temporal que C10, pero no desplaza automáticamente las anclas fijadas por el protocolo.

## Tamaños, cobertura y anotaciones vacías

Orden de las ternas: **train / val / test**. El conteo de contenidos históricos
por split suma 1657 porque 198 contenidos pertenecen a más de un split; el universo
global histórico sigue siendo 1459. Las particiones nuevas sí suman 1459.


| Partición | Registros | Contenidos | Heavy Machinery | Vacíos | Desv. media clases pp |
|---|---|---|---|---|---|
| historical | 1178 / 107 / 372 | 1178 / 107 / 372 | 110 / 0 / 19 | 229 / 16 / 47 | 3.384756 |
| C10 | 1178 / 107 / 372 | 1029 / 95 / 335 | 95 / 7 / 27 | 207 / 19 / 66 | 0.469353 |
| C01 | 1178 / 107 / 372 | 1047 / 98 / 314 | 92 / 8 / 29 | 207 / 19 / 66 | 0.039754 |

Random tiene 1178/107/372 registros en cuatro semillas; seed 2 produce
1177/107/373. Heavy Machinery en random: train 87–98, val 3–12, test 25–33.
Vacíos: train 199–210, val 20–22, test 61–71. El histórico no tiene Vehicles
ni Heavy Machinery en validation. Ambas representantes finales tienen las cinco
clases en las tres particiones, aunque la representación C10 conserva solo dos
instancias Vehicles en val: cobertura no equivale a suficiencia estadística.

Instancias reales por clase (no imágenes ni contenidos) en representantes finales:


**C10**

| class_name | train | val | test |
|---|---|---|---|
| Buildings | 1621 | 153 | 521 |
| Heavy Machinery | 95 | 7 | 27 |
| Rivers | 457 | 41 | 144 |
| Roads | 719 | 64 | 227 |
| Vehicles | 60 | 2 | 30 |

**C01**

| class_name | train | val | test |
|---|---|---|---|
| Buildings | 1632 | 148 | 515 |
| Heavy Machinery | 92 | 8 | 29 |
| Rivers | 457 | 41 | 144 |
| Roads | 718 | 65 | 227 |
| Vehicles | 65 | 6 | 21 |

Las tablas locales incluyen además imágenes con presencia, contenidos por clase,
porcentajes, targets y desviación absoluta en puntos porcentuales, conservando
las cinco clases. El background se reporta aparte.

## NN cross-split y retención de vecinos

NN sobre la matriz completa, un valor por contenido y sin self-content. En el
histórico se usa existencia de memberships distintos entre dos contenidos, sin
aplanar conjuntos ni mezclar duplicados exactos. Coseno entre encoders no comparte
escala. Cada fila corresponde a la representante seed 0.


| Partición | Encoder | mean | median | Q1 | Q3 | p90 | p95 | p99 | max |
|---|---|---|---|---|---|---|---|---|---|
| historical | dinov2 | 0.841023 | 0.848119 | 0.803105 | 0.898118 | 0.936903 | 0.967750 | 0.991784 | 0.994964 |
| historical | clip | 0.948599 | 0.952314 | 0.937414 | 0.964149 | 0.973025 | 0.982706 | 0.993436 | 0.996466 |
| C10 | dinov2 | 0.800351 | 0.818989 | 0.759431 | 0.859820 | 0.893428 | 0.913319 | 0.946285 | 0.963081 |
| C10 | clip | 0.943083 | 0.948832 | 0.933269 | 0.959735 | 0.966952 | 0.970508 | 0.974949 | 0.982206 |
| C01 | dinov2 | 0.874337 | 0.890699 | 0.831517 | 0.932423 | 0.959551 | 0.970890 | 0.987415 | 0.995615 |
| C01 | clip | 0.956567 | 0.958604 | 0.946425 | 0.970231 | 0.980412 | 0.986811 | 0.991518 | 0.993905 |

| Partición | Encoder | Top1 fracción | Top5 fracción | Top10 fracción | Top20 fracción |
|---|---|---|---|---|---|
| historical | dinov2 | 0.192598 | 0.243729 | 0.283071 | 0.334647 |
| historical | clip | 0.228239 | 0.269637 | 0.308156 | 0.351062 |
| C10 | dinov2 | 0.047293 | 0.062097 | 0.087320 | 0.131871 |
| C10 | clip | 0.094585 | 0.131734 | 0.166415 | 0.215559 |
| C01 | dinov2 | 0.260452 | 0.307334 | 0.323029 | 0.345819 |
| C01 | clip | 0.288554 | 0.321316 | 0.341193 | 0.360212 |

## Pares de alta similitud por cuantil

Cantidades de pares únicos i<j cross-split. Cohortes inclusivos con empates y
thresholds originales propios del encoder. Los cohortes son anidados, no sumables.
Random es la **media de cinco seeds**, por eso algunas cantidades son fraccionarias.


**DINOV2**

| Top % | Threshold | Pares en cohorte | Historical | Random media | C10 | C01 |
|---|---|---|---|---|---|---|
| 10.000000 | 0.709861 | 106362 | 51851 | 46721.800000 | 38839 | 45500 |
| 5.000000 | 0.765594 | 53181 | 25231 | 23366.600000 | 14687 | 21533 |
| 2.500000 | 0.811430 | 26591 | 11717 | 11747.800000 | 4917 | 9747 |
| 1.000000 | 0.865117 | 10637 | 3751 | 4727.600000 | 700 | 3233 |
| 0.500000 | 0.901112 | 5319 | 1487 | 2346.600000 | 205 | 1479 |
| 0.100000 | 0.954069 | 1064 | 328 | 478.000000 | 6 | 229 |

**CLIP**

| Top % | Threshold | Pares en cohorte | Historical | Random media | C10 | C01 |
|---|---|---|---|---|---|---|
| 10.000000 | 0.909305 | 106362 | 51190 | 46221.800000 | 43845 | 46855 |
| 5.000000 | 0.927237 | 53181 | 24710 | 23016.600000 | 19234 | 22942 |
| 2.500000 | 0.939901 | 26591 | 11509 | 11492.000000 | 8020 | 10960 |
| 1.000000 | 0.951965 | 10637 | 3852 | 4609.600000 | 2405 | 3967 |
| 0.500000 | 0.959410 | 5319 | 1594 | 2313.600000 | 851 | 1814 |
| 0.100000 | 0.975326 | 1064 | 277 | 466.800000 | 7 | 311 |

## Temporalidad residual y fragmentación de secuencias

Pares de contenidos distintos, misma secuencia inferida y Δ de índices, no segundos.
Los denominadores son 1438 / 7159 / 14227 / 35074 para Δ≤1/5/10/25. No hay
unknowns en esta ejecución; esa cobertura no verifica el origen temporal real.


| Δ máximo | Pares | Historical count | Historical fracción | Random fracción media | C10 count | C10 fracción | C01 count | C01 fracción |
|---|---|---|---|---|---|---|---|---|
| 1 | 1438 | 203 | 0.141168 | 0.441446 | 218 | 0.151599 | 393 | 0.273296 |
| 5 | 7159 | 1057 | 0.147646 | 0.439167 | 1713 | 0.239279 | 2364 | 0.330214 |
| 10 | 14227 | 2232 | 0.156885 | 0.439109 | 4170 | 0.293105 | 5095 | 0.358122 |
| 25 | 35074 | 6434 | 0.183441 | 0.438724 | 12536 | 0.357416 | 13807 | 0.393653 |

En historical, random y ambas representantes finales, las dos secuencias
identificadas atraviesan los tres splits: fracciones confinadas a 1/2/3 splits
= **0 / 0 / 1**. No se exigió que un video completo fuera indivisible.

C10 seed 0 tiene Δ≤5 cross-split **23.93%**, frente a **43.92%** random medio,
pero **14.76%** histórico. Incluso Δ≤1 queda ligeramente peor: 15.16% frente a
14.12%. No se afirma mejora temporal uniforme. Esta diferencia respecto a conteos
previos de auditoría temporal deriva del universo: aquí son pares de contenidos
distintos, no todos los pares de ocurrencias ni duplicados exactos.

## Fracturas y estabilidad entre semillas

Cada porcentaje de fractura evalúa los clusters del mismo candidato; noise queda
excluido. Historical usa sus conjuntos completos, random promedia cinco semillas.


| candidate_label | cluster_aware | historical | random_content |
|---|---|---|---|
| C01 | 0.000000 | 0.500000 | 1.000000 |
| C02 | 0.000000 | 0.375000 | 0.925000 |
| C03 | 0.000000 | 0.888889 | 1.000000 |
| C04 | 0.000000 | 1.000000 | 1.000000 |
| C05 | 0.000000 | 0.809524 | 1.000000 |
| C06 | 0.000000 | 1.000000 | 1.000000 |
| C07 | 0.000000 | 0.416667 | 0.983333 |
| C08 | 0.000000 | 0.500000 | 1.000000 |
| C09 | 0.000000 | 0.590909 | 1.000000 |
| C10 | 0.000000 | 0.470588 | 1.000000 |
| C11 | 0.000000 | 1.000000 | 1.000000 |
| C12 | 0.000000 | 0.818182 | 1.000000 |

Fracción de contenidos que conserva el mismo nombre de split, diez pares de semillas:

| candidate_label | mean | min | max |
|---|---|---|---|
| C01 | 0.603564 | 0.524332 | 0.682659 |
| C02 | 0.986155 | 0.979438 | 0.993831 |
| C03 | 0.893900 | 0.843043 | 0.947224 |
| C04 | 1.000000 | 1.000000 | 1.000000 |
| C05 | 0.781151 | 0.695682 | 0.860864 |
| C06 | 0.997121 | 0.995888 | 0.998629 |
| C07 | 0.624058 | 0.562714 | 0.739548 |
| C08 | 1.000000 | 1.000000 | 1.000000 |
| C09 | 0.615147 | 0.538725 | 0.719671 |
| C10 | 0.892666 | 0.815627 | 0.969842 |
| C11 | 1.000000 | 1.000000 | 1.000000 |
| C12 | 0.946059 | 0.899246 | 0.982180 |
| random_content | 0.557711 | 0.546950 | 0.571624 |

Una asignación muy estable puede ser inadecuada: C04/C08/C11 repiten soluciones
desbalanceadas. Esta estabilidad no sustituye cobertura ni correlación residual.

Variabilidad de random; std poblacional sobre cinco seeds, no intervalos de confianza:


| metric | mean | std_population | min | max |
|---|---|---|---|---|
| max_relative_record_deviation | 0.000538 | 0.001075 | 0.000000 | 0.002688 |
| class_deviation_pp | 1.660290 | 0.220230 | 1.274283 | 1.878440 |
| dinov2_nn_mean | 0.896996 | 0.000877 | 0.895923 | 0.898563 |
| clip_nn_mean | 0.963612 | 0.000532 | 0.962808 | 0.964224 |
| dinov2_top001_pairs | 478.000000 | 31.419739 | 423.000000 | 521.000000 |
| clip_top001_pairs | 466.800000 | 44.664975 | 406.000000 | 530.000000 |
| temporal_at5 | 0.439167 | 0.005329 | 0.431904 | 0.445314 |

## Artefactos, figuras y validación

- Run directories: `artifacts/splitting/runs/<split_space_id>/`, assignments por
  contenido y registro, balance, metadata, quality, métricas y trazabilidad local.
- Comparación: `artifacts/splitting/comparisons/a19ba503f085f569/` con 66 filas,
  fracturas, retención de memberships, variabilidad, Pareto y dos representantes.
- Reporte: `reports/splitting/review/splitting_review.html` y
  `splitting_review.executed.ipynb`, 18 secciones, código oculto.
- Figuras: `01_split_sizes_comparison.png`, `02_class_balance_comparison.png`,
  `03_exact_duplicate_overlap.png`, `04_cross_split_nn_similarity_dinov2.png`,
  `05_cross_split_nn_similarity_clip.png`, `06_high_similarity_cross_split_pairs.png`,
  `07_temporal_cross_split_rates.png`, `08_cluster_fracture_comparison.png`,
  `09_split_candidate_pareto.png`, bajo `reports/splitting/figures/`.
- **124 tests offline sintéticos aprobados**, Ruff aprobado; CLI de comparación
  ejecutada, superficie de los siete comandos probada; notebook fuente limpio.
- Las 66 publicaciones y métricas se verificaron contra fuentes; la posterior
  reconstrucción de las 65 asignaciones nuevas coincidió exactamente. Los logs y
  recibos de verificación quedan en `reports/splitting/execution/`.
- No se versionan assignments reales, content IDs, embeddings, matrices, imágenes,
  labels ni HTML/notebooks ejecutados. Solo código, configs, narrativa agregada y
  notebook fuente. El commit/push se identifica en el cierre de ejecución.

## Límites y siguiente paso

La selección sigue siendo exploratoria sobre los mismos datos; evaluar ambos
encoders no equivale a disponer de un test externo. El solver balancea clases
y random no, por lo que una futura ablación singleton balanceado permitiría
aislar mejor el efecto exclusivo del agrupamiento. Las diferencias entre semillas
y la brecha de MILP limitan la exhaustividad de la comparación. La selección de
12 candidatos no caracteriza todos los 41 del frente previo.

C01 mantiene mucho ruido y puede ser peor que el histórico en CLIP; no debe
presentarse como solución a leakage. C10 es un candidato visualmente favorable
con compromisos temporales y pocas instancias de algunas clases en val. Persisten
ocho conflictos de anotación y multiplicidades históricas. No se ha demostrado
causalidad ni mejora en Precision, Recall o mAP.

**Cierre: protocolo principal ejecutado y validado, con límites**, no detector
validado. Siguiente: revisar casos y anotaciones de los candidatos, fijar un
protocolo común de comparación del detector y solo después materializar imágenes.
El exportador está implementado y probado sintéticamente, pero no se ejecutó
exportación real, copia/movimiento de imágenes, similarity-components, naive
record-random ni entrenamiento YOLO.
