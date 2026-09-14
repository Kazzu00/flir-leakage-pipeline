# Semana 10: clustering ejecutado y candidatos exploratorios

Ejecución local del **2026-09-13** (America/Bogota; metadata en UTC). Se evaluaron
**1459 contenidos únicos por run**, conservando el mapping a 1657 ocurrencias.
Se verificaron previamente los dos espacios completos, las matrices/vecinos de
similitud y las 36 reducciones existentes. No se extrajeron embeddings ni se
reajustaron reducciones. La validación temporal sigue **PARTIAL**: secuencia e
índice proceden de nombres, con cero timestamps verificados.

## Protocolo y ejecución

El [protocolo preregistrado](clustering_protocol.md) y sus fuentes/configs se
copiaron localmente antes del primer ajuste. Se ejecutaron 342 runs de Fase A:
18 DBSCAN + 27 OPTICS + 12 HDBSCAN por cada uno de los seis espacios. Los
cuantiles k-distance no produjeron parámetros efectivos idénticos ni epsilon
no positivo en este grid; no se omitieron configuraciones. La shortlist tiene
54 configuraciones, tres por encoder × representación × algoritmo, frente a
269 miembros del Pareto de A. Las 36 configuraciones reducidas de la shortlist
añadieron 72 ajustes (semillas 1/2); las 18 originales solo evalúan parámetros.

**Total: 414 runs completos y 204 comparaciones de assignments**: 108 pares de
semillas y 96 pares locales de parámetros, contados por candidato de referencia
(un par simétrico puede documentarse para dos referencias). No se ajustaron
semillas para todo el grid. Todos los inputs de fit son vectores e identidad de
fila, sin labels, split histórico ni procedencia temporal.

| Algoritmo | Fase A | Adicionales B | Total | Ajuste acumulado (s) |
|---|---:|---:|---:|---:|
| DBSCAN | 108 | 24 | 132 | 13.220 |
| OPTICS | 162 | 24 | 186 | 617.158 |
| HDBSCAN | 72 | 24 | 96 | 15.532 |

Tiempos de fit sumados **645.910 s**, sin evaluación, reporte ni verificaciones;
no constituyen un benchmark de hardware. Se utilizó CPU, un hilo nativo,
Python 3.11.14, NumPy 2.4.6, pandas 3.0.5 y scikit-learn 1.9.1 del lockfile.
HDBSCAN EOM proporciona probabilidades de pertenencia, guardadas en sus 96 runs;
esta implementación no expone outlier scores ni cluster persistence. No se
instaló la librería externa ni se ejecutó la variante leaf: EOM ya produjo
alternativas no degeneradas en los seis espacios.

El fit usa Euclidean sobre L2 original o coordenadas 2D existentes, sin
renormalización ni z-score. Para vectores unitarios, d²=2(1−cos), con el mismo
orden de vecinos salvo redondeo/empates. Las escalas 2D son propias de cada
reducción y semilla; DBSCAN recalcula epsilon por la misma regla de cuantil.
Los adapters ordenan canónicamente por content_id para resolver sensibilidad al
orden sin ponderar copias históricas.

## Resultados completos y degeneración

Hay **39 runs con un solo cluster**, cero all-noise, 18 nearly-all-noise
(fracción ≥0.95) y 54 dominant-cluster (grupo mayor ≥0.90 del total).
Estos flags pueden solaparse y no se suman. Fase A contiene 38 single-cluster;
una variante de semilla añade el caso restante. El rango observado es **1–65
clusters** y **noise 0–0.984921**. Los ceros/unos de grupos quedan fuera de la
shortlist; los flags restantes son diagnósticos y se conservan con su cobertura.
Los 414 resultados, incluyendo desfavorables, permanecen en `all_runs.csv` y HTML.

La selección final deja **41 candidatos Pareto**, de 54 evaluados: 13 presentan
comparaciones locales common-clustered triviales y/o una variante de semilla
degenerada. No se impone estabilidad perfecta ni se relaja el protocolo. Los
41 elegibles permanecen no dominados: el frente amplio muestra que los criterios
no identifican un ganador único. Número de grupos y pureza de secuencia se
describen, sin asignarles una dirección artificial de optimización. Split y
clases no se usan como criterio. No se ejecutó análisis opcional de clases.

## Original vs t-SNE vs PaCMAP: referencias de revisión

Estas **seis referencias** maximizan silhouette original dentro del frente final
de cada encoder/representación, únicamente para limitar las figuras. Todas
resultaron OPTICS xi=0.10, max_eps infinito. **No son seis ganadores** ni una
selección de algoritmo: las configuraciones alternativas DBSCAN/HDBSCAN siguen
en el frente completo al final de este documento.

| Ref | encoder | representation | algorithm | clustering_space_id | Grupos | Noise | Silhouette original | Coseno intra | Secuencia dominante |
|---|---|---|---|---|---|---|---|---|---|
| R1 | clip | original_l2 | optics | 0208da483a57db96 | 2 | 0.973955 | 0.560925 | 0.970055 | 1.000000 |
| R2 | clip | pacmap | optics | 8bbf87a3d1ea83c8 | 16 | 0.656614 | 0.215438 | 0.927741 | 0.972056 |
| R3 | clip | tsne | optics | 0f31ddafa5b7a5b4 | 4 | 0.876628 | 0.350595 | 0.918829 | 1.000000 |
| R4 | dinov2 | original_l2 | optics | 45edde2676999382 | 2 | 0.958191 | 0.686930 | 0.918593 | 1.000000 |
| R5 | dinov2 | pacmap | optics | ed6f6719599cc7e1 | 15 | 0.609321 | 0.226463 | 0.798940 | 0.998246 |
| R6 | dinov2 | tsne | optics | 9a8afc2627221c30 | 37 | 0.523646 | 0.275229 | 0.865037 | 0.998561 |

R1/R4 originales solo agrupan 38/61 contenidos: cobertura 2.6045%/4.1809%.
Sus silhouettes altas no justifican particionar toda la población. El máximo
de silhouette tiene un sesgo de revisión hacia subpoblaciones compactas; por
ello se publica el frente completo y no se adopta esta preferencia para el split.
Las referencias reducidas cubren más contenidos con menor silhouette original.
Eso no prueba una mejora global: cambian cobertura, grupos y escala geométrica.

En estas referencias, DINOv2 t-SNE retiene más pares temporales que DINOv2
PaCMAP, mientras CLIP PaCMAP retiene más que CLIP t-SNE. Son configuraciones
distintas y seleccionadas para inspección; no permiten ordenar encoders ni
métodos en general. Los cosenos absolutos de CLIP y DINOv2 tampoco comparten
calibración semántica.

## Retención temporal y visual

Temporal recall usa pares únicos de la misma secuencia inferida con Δ≤k,
incluyendo en el denominador los que tocan noise. Un par con dos labels −1
**no** se considera retenido. La métrica visual usa top-k originales dirigidos,
excluye queries noise y cuenta vecinos noise como no retenidos. Su cobertura es
1−noise_fraction; la versión con todas las queries también se conserva localmente.
La fracción dominante pondera miembros con secuencia conocida; aquí la cobertura
de procedencia entre agrupados es 1, lo que no valida tiempos reales.

| Ref | temporal_recall@1 | temporal_recall@5 | temporal_recall@10 | visual_neighbor_coherence@5 | visual_neighbor_coherence@10 | visual_neighbor_coherence@20 |
|---|---|---|---|---|---|---|
| R1 | 0.024339 | 0.021651 | 0.018627 | 0.978947 | 0.973684 | 0.878947 |
| R2 | 0.203755 | 0.146808 | 0.118718 | 0.836727 | 0.782834 | 0.683533 |
| R3 | 0.093185 | 0.074591 | 0.066282 | 0.948889 | 0.918333 | 0.872222 |
| R4 | 0.040334 | 0.037854 | 0.034863 | 0.996721 | 0.993443 | 0.981967 |
| R5 | 0.285118 | 0.213019 | 0.170240 | 0.893333 | 0.846316 | 0.766316 |
| R6 | 0.365090 | 0.286912 | 0.230056 | 0.932950 | 0.883597 | 0.721799 |

## Estabilidad entre semillas de reducción

Media y mínimo sobre seed0–1, seed0–2 y seed1–2. All-points trata −1 como una
categoría; common-clustered usa solo la intersección no-noise. N y cobertura se
guardan para cada par. N<2 es indefinido; una partición de un grupo es trivial.
Los originales no tienen semilla de reducción: **N/A**, no estabilidad perfecta.

| Ref | ARI all: media (mín) | AMI all: media (mín) | ARI common: media (mín) | AMI common: media (mín) | Cobertura common mínima |
|---|---|---|---|---|---|
| R1 | N/A | N/A | N/A | N/A | N/A |
| R2 | 0.373314 (0.356577) | 0.539506 (0.517170) | 0.994008 (0.988018) | 0.995463 (0.991775) | 0.203564 |
| R3 | 0.578671 (0.368007) | 0.767768 (0.651652) | 1.000000 (1.000000) | 1.000000 (1.000000) | 0.122687 |
| R4 | N/A | N/A | N/A | N/A | N/A |
| R5 | 0.267237 (0.253319) | 0.600677 (0.577281) | 0.963981 (0.898008) | 0.987667 (0.969490) | 0.291295 |
| R6 | 0.797227 (0.763827) | 0.885164 (0.871125) | 1.000000 (1.000000) | 1.000000 (1.000000) | 0.439342 |

El acuerdo common-clustered cercano a 1 no implica asignaciones estables para
los 1459 contenidos. Por ejemplo R5 tiene ARI medio all-points 0.267237 frente a
0.963981 common-clustered, con cobertura mínima 0.291295. También debe revisarse
la variación de quién entra/sale de noise. Los dos resultados no son intercambiables.

## Robustez local de parámetros

Cuantil epsilon adyacente para DBSCAN, xi adyacente para OPTICS, min_samples o
min_cluster_size adyacentes para HDBSCAN, cambiando una coordenada a la vez.
Se reutilizan runs de Fase A, incluso si el vecino es degenerado. En las seis
referencias OPTICS xi=0.10, el único vecino es xi=0.05, por eso media=mínimo.

| Ref | ARI all: media (mín) | AMI all: media (mín) | ARI common: media (mín) | AMI common: media (mín) | Cobertura common mínima |
|---|---|---|---|---|---|
| R1 | 0.532803 (0.532803) | 0.543893 (0.543893) | 1.000000 (1.000000) | 1.000000 (1.000000) | 0.026045 |
| R2 | 0.787852 (0.787852) | 0.848147 (0.848147) | 1.000000 (1.000000) | 1.000000 (1.000000) | 0.299520 |
| R3 | 0.471103 (0.471103) | 0.606743 (0.606743) | 1.000000 (1.000000) | 1.000000 (1.000000) | 0.123372 |
| R4 | 0.355362 (0.355362) | 0.424217 (0.424217) | 1.000000 (1.000000) | 1.000000 (1.000000) | 0.041809 |
| R5 | 0.740445 (0.740445) | 0.863113 (0.863113) | 1.000000 (1.000000) | 1.000000 (1.000000) | 0.377656 |
| R6 | 0.681537 (0.681537) | 0.830481 (0.830481) | 1.000000 (1.000000) | 1.000000 (1.000000) | 0.454421 |

## Candidatos Pareto completos

Todos corresponden al original o semilla 0 de las reducciones seleccionadas.
`ms`=min_samples, `mcs`=min_cluster_size. DBSCAN q indica la regla conceptual y
epsilon el valor efectivo para este espacio; las otras semillas usan su propio
epsilon. HDBSCAN usa EOM, epsilon=0 y allow_single_cluster=false. Las tablas
locales incluyen todas las métricas, medias/mínimos ARI/AMI y cada comparación.

| encoder | representation | algorithm | clustering_space_id | Parámetros | Grupos | Noise | Silhouette original | Coseno intra | temporal_recall@5 | visual_neighbor_coherence@10 |
|---|---|---|---|---|---|---|---|---|---|---|
| clip | original_l2 | dbscan | bf8bb9fceae6f259 | ms=5; q=0.85; eps=0.342002 | 8 | 0.071967 | 0.043803 | 0.836276 | 0.843135 | 0.980724 |
| clip | original_l2 | optics | 000389f634346938 | ms=5; mcs=10; xi=0.03 | 21 | 0.788211 | 0.346066 | 0.964014 | 0.137030 | 0.864401 |
| clip | original_l2 | optics | 0208da483a57db96 | ms=10; mcs=10; xi=0.1 | 2 | 0.973955 | 0.560925 | 0.970055 | 0.021651 | 0.973684 |
| clip | pacmap | dbscan | 293f660b7413de2f | ms=20; q=0.85; eps=1.128590 | 8 | 0.029472 | 0.120855 | 0.880552 | 0.730549 | 0.961017 |
| clip | pacmap | dbscan | 895e5dccdfb6c938 | ms=20; q=0.99; eps=2.064257 | 4 | 0.000000 | 0.047825 | 0.838753 | 0.909485 | 0.984236 |
| clip | pacmap | dbscan | 8b78f58b254aa25e | ms=10; q=0.85; eps=0.574760 | 28 | 0.033585 | 0.045405 | 0.908189 | 0.582344 | 0.876241 |
| clip | pacmap | hdbscan | 05fa234a0f64ae34 | ms=20; mcs=50; EOM | 3 | 0.039068 | 0.088984 | 0.837528 | 0.854030 | 0.984522 |
| clip | pacmap | hdbscan | 06938d9d76d03bde | ms=5; mcs=50; EOM | 9 | 0.256340 | 0.131400 | 0.905468 | 0.484286 | 0.891797 |
| clip | pacmap | hdbscan | 1c676172a4ed0b08 | ms=20; mcs=20; EOM | 5 | 0.006169 | 0.060393 | 0.837620 | 0.874563 | 0.981310 |
| clip | pacmap | optics | 6dbc765bf568e44f | ms=10; mcs=10; xi=0.03 | 57 | 0.333790 | 0.129744 | 0.939425 | 0.277413 | 0.639095 |
| clip | pacmap | optics | 8bbf87a3d1ea83c8 | ms=10; mcs=20; xi=0.1 | 16 | 0.656614 | 0.215438 | 0.927741 | 0.146808 | 0.782834 |
| clip | pacmap | optics | d886c648457019f8 | ms=20; mcs=30; xi=0.1 | 8 | 0.705963 | 0.180656 | 0.909563 | 0.147926 | 0.870396 |
| clip | tsne | dbscan | 1b7cf5a28f66f18a | ms=20; q=0.85; eps=5.598408 | 5 | 0.046607 | 0.119722 | 0.858928 | 0.825395 | 0.976779 |
| clip | tsne | dbscan | 707b3a23c8f203b9 | ms=20; q=0.9; eps=6.311909 | 2 | 0.038382 | 0.077754 | 0.828200 | 0.907110 | 0.994654 |
| clip | tsne | hdbscan | 550362337e02f013 | ms=20; mcs=30; EOM | 9 | 0.235093 | 0.179420 | 0.910207 | 0.528146 | 0.933871 |
| clip | tsne | hdbscan | 5bb5c274dc217f3c | ms=5; mcs=50; EOM | 4 | 0.079507 | 0.122576 | 0.859753 | 0.791172 | 0.981087 |
| clip | tsne | hdbscan | 61e8ba3cf0696d4a | ms=5; mcs=10; EOM | 44 | 0.224126 | 0.123874 | 0.919961 | 0.397542 | 0.807951 |
| clip | tsne | optics | 0f31ddafa5b7a5b4 | ms=20; mcs=30; xi=0.1 | 4 | 0.876628 | 0.350595 | 0.918829 | 0.074591 | 0.918333 |
| clip | tsne | optics | 1c94a28a079022d2 | ms=10; mcs=20; xi=0.05 | 21 | 0.462646 | 0.160160 | 0.920553 | 0.301439 | 0.866454 |
| clip | tsne | optics | 359b80164379a5a1 | ms=5; mcs=30; xi=0.1 | 8 | 0.725154 | 0.230037 | 0.897747 | 0.151837 | 0.919950 |
| dinov2 | original_l2 | dbscan | 2c135d2c1a3f1754 | ms=10; q=0.97; eps=0.838867 | 2 | 0.006854 | 0.265312 | 0.471028 | 0.985752 | 0.998827 |
| dinov2 | original_l2 | dbscan | ed4aab613e943b33 | ms=10; q=0.99; eps=0.928652 | 2 | 0.000685 | 0.263846 | 0.467571 | 0.996508 | 0.998628 |
| dinov2 | original_l2 | optics | 36f4e25772097bd1 | ms=10; mcs=20; xi=0.1 | 5 | 0.887594 | 0.484225 | 0.901556 | 0.087442 | 0.995732 |
| dinov2 | original_l2 | optics | 45edde2676999382 | ms=20; mcs=20; xi=0.1 | 2 | 0.958191 | 0.686930 | 0.918593 | 0.037854 | 0.993443 |
| dinov2 | original_l2 | optics | e3ad8a743e6b55e2 | ms=10; mcs=10; xi=0.03 | 24 | 0.675805 | 0.362030 | 0.894855 | 0.214974 | 0.932347 |
| dinov2 | pacmap | dbscan | 25dc30caae026d9c | ms=5; q=0.8; eps=0.323575 | 65 | 0.122687 | 0.148036 | 0.809400 | 0.423383 | 0.770547 |
| dinov2 | pacmap | dbscan | f11bb99234c9de13 | ms=10; q=0.85; eps=0.598159 | 34 | 0.032214 | 0.060389 | 0.672261 | 0.592122 | 0.871884 |
| dinov2 | pacmap | hdbscan | 06cb8e8c0a0bb896 | ms=5; mcs=10; EOM | 54 | 0.166552 | 0.134443 | 0.784481 | 0.390837 | 0.754605 |
| dinov2 | pacmap | hdbscan | 967d6b1161af99dd | ms=5; mcs=50; EOM | 7 | 0.037697 | 0.049947 | 0.540661 | 0.718955 | 0.947721 |
| dinov2 | pacmap | hdbscan | 9db01bfbc49efaaf | ms=10; mcs=10; EOM | 44 | 0.192598 | 0.156146 | 0.769072 | 0.400894 | 0.802122 |
| dinov2 | pacmap | optics | 255cf54a2598f056 | ms=20; mcs=30; xi=0.1 | 11 | 0.647019 | 0.223255 | 0.772618 | 0.208130 | 0.905437 |
| dinov2 | pacmap | optics | c07705fe79d2f1a8 | ms=20; mcs=10; xi=0.03 | 30 | 0.282385 | 0.147952 | 0.775791 | 0.349770 | 0.791309 |
| dinov2 | pacmap | optics | ed6f6719599cc7e1 | ms=10; mcs=30; xi=0.1 | 15 | 0.609321 | 0.226463 | 0.798940 | 0.213019 | 0.846316 |
| dinov2 | tsne | dbscan | 2097fab0f4a8f61c | ms=20; q=0.95; eps=7.291948 | 2 | 0.014393 | 0.073241 | 0.485042 | 0.899567 | 0.991516 |
| dinov2 | tsne | dbscan | a790a460d245cdb5 | ms=5; q=0.8; eps=1.878334 | 53 | 0.122687 | 0.151893 | 0.773872 | 0.475625 | 0.860703 |
| dinov2 | tsne | hdbscan | 7a46f5678288a69a | ms=20; mcs=20; EOM | 13 | 0.210418 | 0.159039 | 0.711024 | 0.484146 | 0.942535 |
| dinov2 | tsne | hdbscan | 854148e2dbb27bfd | ms=5; mcs=20; EOM | 22 | 0.125428 | 0.144445 | 0.743983 | 0.505797 | 0.909953 |
| dinov2 | tsne | hdbscan | fba6246cf104ca03 | ms=20; mcs=50; EOM | 4 | 0.098698 | 0.014914 | 0.483620 | 0.713228 | 0.984639 |
| dinov2 | tsne | optics | 22d306e4c9a60222 | ms=20; mcs=30; xi=0.1 | 12 | 0.352981 | 0.170783 | 0.712148 | 0.421986 | 0.967903 |
| dinov2 | tsne | optics | 8c9cc9991a41b173 | ms=20; mcs=20; xi=0.1 | 12 | 0.398218 | 0.188195 | 0.720722 | 0.408437 | 0.968337 |
| dinov2 | tsne | optics | 9a8afc2627221c30 | ms=10; mcs=10; xi=0.1 | 37 | 0.523646 | 0.275229 | 0.865037 | 0.286912 | 0.883597 |

## Inspección de exemplars e historial

El HTML contiene **17 figuras**: cinco diagnósticos globales (k-distance,
estructura/noise, cohesión original, estabilidad e historial), seis scatters
y seis galerías R1–R6. Las galerías tienen **144 posiciones** con imágenes leídas
y verificadas desde ZIP en runtime; los roles pueden repetir contenidos.
Se revisaron visualmente las 17 figuras. El notebook fuente tiene 18 secciones;
el ejecutado/HTML muestran las tablas completas y ocultan código.

Observación de esta revisión, sin anotación exhaustiva de escenas: R1/R4 muestran
subgrupos de vistas muy semejantes; los miembros lejanos de varios grupos
reducidos cambian encuadre o escala. R2 incluye un grupo de mayor entropía con
menús superpuestos sobre vistas de distintas secuencias, y un grupo mediano con
contrastes extremos. R3 contiene un grupo de imágenes visualmente ruidosas;
R6 incluye un grupo mediano con encuadres diferentes de una zona excavada.
Esto plantea una **hipótesis** de influencia de menús/contraste/ruido y semejanza
de escena en la representación; la galería no demuestra sus causas ni valida
todos los miembros. Mayor entropía puede ser cero si todos los grupos tienen
una sola secuencia: no se fabrica diversidad para llenar un panel.

| Ref | historical_train_only_clusters | historical_val_only_clusters | historical_test_only_clusters | historical_multisplit_clusters | historical_unknown_clusters |
|---|---|---|---|---|---|
| R1 | 0 | 0 | 1 | 1 | 0 |
| R2 | 3 | 0 | 1 | 12 | 0 |
| R3 | 0 | 0 | 0 | 4 | 0 |
| R4 | 0 | 0 | 1 | 1 | 0 |
| R5 | 2 | 0 | 1 | 12 | 0 |
| R6 | 16 | 1 | 2 | 18 | 0 |

La unión de memberships conserva todas las ocurrencias originales. Un grupo
históricamente multi-split no es penalizado ni equivale por sí solo a leakage
confirmado. Los 198 duplicados exactos y ocho conflictos de anotación conocidos
se preservan sin corrección silenciosa.

## Verificación, trazabilidad y límites

Cada run conserva labels int32, índice, resumen con medoides en L2 original,
metrics, quality y metadata; HDBSCAN agrega probabilidades y OPTICS diagnósticos
de reachability/core/order. La identidad de 16 caracteres depende de dataset,
feature, reducción, representación, algoritmo, parámetros y versiones; no de
rutas, dispositivo ni fecha. Los snapshots conservan el commit realmente activo
y dirty state del ajuste, sin atribuirlo retrospectivamente al commit de cierre.

Se verificaron los 414 runs contra features, similitud, manifest y reducciones,
recalculando métricas y medoides; 204 pares ARI/AMI y la selección se reconstruyen.
Tras los ajustes se reforzó el verificador para reconstruir también agregados,
topología de perturbaciones y referencias, y rechazar cambios de familia reducida
antes de nuevas semillas. No cambió el fit, las métricas ni la regla Pareto.
Se preservan protocolo/configs y snapshot previo; el recibo final distingue las
fuentes originales de las comprobaciones posteriores.

Esta es una ejecución completa y verificada del protocolo exploratorio, no una
validación de escenas, split sin leakage ni mejora del detector. El frente se
evalúa sobre esta misma población, con criterios correlacionados y sin holdout
de selección. Fase B evalúa solo la shortlist; parámetros vecinos y tres seeds
no cubren toda la incertidumbre. Las galerías son muestras dirigidas y los datos
temporales siguen siendo inferidos. No se hicieron bootstrap, ablation de overlays,
análisis de clases, variantes leaf, nuevos splits, balanceo ni YOLO.

## Siguiente paso y reproducción

Validación de cierre: **100 tests passed**, Ruff **All checks passed**, cuatro
notebooks fuente limpios; cinco comandos CLI y sus ayudas validados. HTML
ejecutado sin errores, código oculto, UTF-8 correcto, 18 secciones y 17 figuras,
sin hashes protegidos, rutas privadas ni NaN en texto visible. Se comprobaron
`.gitignore`, archivos públicos, `git status`, `git diff` y staging antes del
commit; los recibos y logs ejecutados permanecen locales.

Definir el protocolo de cluster-aware splitting con una política explícita para
noise y cobertura. Revisar candidatos de distinta granularidad del frente; las
referencias de silhouette con cobertura mínima no son una elección final útil
por defecto. No convertir noise automáticamente en singletons independientes.
Mantener indivisibles cada grupo y todas las ocurrencias de cada content_id;
medir overlap exacto y relaciones visuales/temporales residuales entre splits.
Comparar después con histórico y aleatorio reproducible, y evaluar cobertura de
clases/conflictos como restricciones de partición posteriores. No se genera un
split en esta fase.

Comandos y reglas de visualización: [runbook](clustering_runbook.md).
Artefactos locales ignorados: `artifacts/clustering/`, `reports/clustering/`;
HTML: `reports/clustering/review/clustering_review.html`.
Código principal: `src/flir_pipeline/clustering/`; configs genéricos:
`configs/clustering/`; notebook fuente: `notebooks/clustering_review.ipynb`.
Los resultados reales y content IDs/medoides quedan fuera de Git.

Fuentes técnicas: [DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html),
[OPTICS](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.OPTICS.html),
[HDBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html)
y [silhouette](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html).
Las convenciones de min_samples de scikit-learn incluyen self; no copiar sin
conversión parámetros de la librería externa hdbscan.
