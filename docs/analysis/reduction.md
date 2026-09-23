# Reducción dimensional — registro de ejecución completa

**2026-09-13: 36 runs completos**, sobre 1459 content_id únicos por encoder:
18 t-SNE y 18 PaCMAP. DINOv2 conserva 384 dimensiones de entrada y CLIP 512;
todas las salidas son float32 1459 × 2. Se revalidaron fuentes, coordenadas,
alineación, finitud y métricas contra los embeddings originales. No se generaron
etiquetas de clustering, nuevas particiones ni resultados de detector.

El [protocolo predefinido](../protocols/reduction.md) y los YAML se conservaron con
huellas locales antes del primer ajuste. Los [comandos](../runbooks/reduction.md)
permiten reproducir una ejecución individual, los grids y el reporte. No se
reextrajeron features, se renormalizaron archivos L2 ni se leyeron/modificaron ZIP.

## Configuraciones ejecutadas

t-SNE: perplexity 10/30/50, semillas 0/1/2, max_iter=1000, learning_rate=auto
(efectivo 50), early_exaggeration=12, init=pca, metric=euclidean,
method=barnes_hut, angle=0.5, n_iter_without_progress=300, min_grad_norm=1e-7.
PaCMAP: n_neighbors=10, MN_ratio=0.2/0.5/1.0, FP_ratio=2, lr=1,
num_iters=[100,100,250], distance=euclidean, apply_pca=false, init=pca,
knn_backend=faiss, semillas 0/1/2. Base controlada MN_ratio=0.5.

Ambos operan directamente sobre L2 completo; PCA solo inicializa coordenadas.
PaCMAP registra su transformación afín nativa y conteos/muestreo de pares.
La evaluación usa k=5/10/20 y 100000 pares únicos, semilla 0. No se ejecutó 3D.
Versiones del ajuste: Python 3.11.14, NumPy 2.4.6, SciPy 1.17.1,
scikit-learn 1.9.1, PaCMAP 0.9.1, FAISS CPU 1.15.0, Numba 0.67.0.
Las versiones transitivas y parámetros efectivos completos están en metadata.

## Referencias seleccionadas

La regla Pareto + media de rangos usa las tres semillas y cinco criterios, con
igual peso; la semilla representativa siempre es 0. Esta tabla muestra T/C/J y
Spearman del run de referencia; estabilidad es la media de los tres pares de
semillas a k=20. Son candidatos exploratorios para la siguiente evaluación.


| encoder | method | label | seed | reduction_space_id | trustworthiness@20 | continuity@20 | jaccard@20 | stability@20 | spearman | fit_seconds | mean_criterion_rank |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dinov2 | pacmap | MN_ratio=1 | 0 | 111e0d9dd4a45458 | 0.966748 | 0.959097 | 0.420561 | 0.649487 | 0.389608 | 1.033619 | 1.800000 |
| dinov2 | tsne | perplexity=30 | 0 | 476786e28eb2bcb8 | 0.976688 | 0.976196 | 0.483902 | 0.974223 | 0.430447 | 12.997342 | 1.600000 |
| clip | pacmap | MN_ratio=1 | 0 | 05838df32347c8d8 | 0.959741 | 0.952572 | 0.355739 | 0.631375 | 0.435227 | 1.061841 | 1.400000 |
| clip | tsne | perplexity=30 | 0 | e273672b3e6b812e | 0.968002 | 0.967658 | 0.407854 | 0.968087 | 0.496726 | 14.862725 | 1.800000 |


## Tiempos observados

Ajustes seriales en CPU local, con un hilo nativo/Numba/FAISS. La primera carga
de biblioteca puede aumentar backend_total; la verificación y agregación añaden
tiempo. Estas cifras no constituyen un benchmark de hardware.

El resumen registrado conserva **269.038 s** de ajuste: t-SNE **251.284 s** y
PaCMAP **17.753 s**, sin incluir toda la verificación, agregación y reporte.


| encoder | method | runs | fit_total_seconds | fit_min_seconds | fit_median_seconds | fit_max_seconds | backend_total_seconds | evaluation_total_seconds |
|---|---|---|---|---|---|---|---|---|
| clip | pacmap | 9 | 8.980045 | 0.849770 | 0.958262 | 1.181506 | 11.156860 | 54.838801 |
| clip | tsne | 9 | 130.116567 | 11.112164 | 14.645185 | 17.716474 | 131.131885 | 54.809539 |
| dinov2 | pacmap | 9 | 8.773318 | 0.782713 | 0.913543 | 1.149976 | 11.033442 | 55.894830 |
| dinov2 | tsne | 9 | 121.167874 | 11.318200 | 12.997342 | 15.764729 | 122.235566 | 54.486564 |


## Métricas por run: DINOV2

T=trustworthiness; C=continuity exacta; J=Jaccard original–reducido.


| método | configuración | seed | reduction_space_id | fit s | T@5/10/20 | C@5/10/20 | J medio@5/10/20 | Spearman |
|---|---|---|---|---|---|---|---|---|
| tsne | perplexity=10 | 0 | 26b2513fa20793dc | 12.170074 | 0.990378 / 0.984663 / 0.974753 | 0.988046 / 0.982811 / 0.975157 | 0.552112 / 0.499453 / 0.466000 | 0.429434 |
| tsne | perplexity=10 | 1 | de56f2138f86b08b | 11.318200 | 0.989770 / 0.984118 / 0.974055 | 0.987982 / 0.982724 / 0.975047 | 0.552512 / 0.498852 / 0.465478 | 0.426675 |
| tsne | perplexity=10 | 2 | 05198d1422c250e7 | 11.556262 | 0.990415 / 0.984575 / 0.974474 | 0.988024 / 0.982815 / 0.975123 | 0.551519 / 0.498810 / 0.465943 | 0.427475 |
| tsne | perplexity=30 | 0 | 476786e28eb2bcb8 | 12.997342 | 0.990840 / 0.985209 / 0.976688 | 0.987897 / 0.982973 / 0.976196 | 0.543553 / 0.510376 / 0.483902 | 0.430447 |
| tsne | perplexity=30 | 1 | c75668a88ebaa901 | 13.115029 | 0.990911 / 0.985199 / 0.976589 | 0.987927 / 0.983000 / 0.976216 | 0.546879 / 0.509763 / 0.483507 | 0.433394 |
| tsne | perplexity=30 | 2 | 1c542ead12e25d1e | 12.948160 | 0.990664 / 0.985138 / 0.976660 | 0.987941 / 0.983033 / 0.976249 | 0.544448 / 0.509709 / 0.483683 | 0.432773 |
| tsne | perplexity=50 | 0 | e64a5da733bb71cd | 15.764729 | 0.991734 / 0.986356 / 0.979171 | 0.987708 / 0.983475 / 0.977462 | 0.526051 / 0.504652 / 0.484675 | 0.457209 |
| tsne | perplexity=50 | 1 | 84adcbe99fb9aa89 | 15.588095 | 0.991492 / 0.986690 / 0.979361 | 0.987705 / 0.983458 / 0.977358 | 0.525474 / 0.504919 / 0.486273 | 0.455687 |
| tsne | perplexity=50 | 2 | 09401a35abbcbaee | 15.709983 | 0.991783 / 0.986617 / 0.979082 | 0.987646 / 0.983431 / 0.977410 | 0.526929 / 0.504515 / 0.484528 | 0.455882 |
| pacmap | MN_ratio=0.2 | 0 | a88ab74ab18b4692 | 0.782713 | 0.977699 / 0.975625 / 0.964921 | 0.974169 / 0.963124 / 0.946169 | 0.290781 / 0.391655 / 0.419701 | 0.365236 |
| pacmap | MN_ratio=0.2 | 1 | 11675da17db72474 | 0.879868 | 0.974912 / 0.972192 / 0.963350 | 0.977552 / 0.969594 / 0.957393 | 0.291116 / 0.394201 / 0.420807 | 0.317786 |
| pacmap | MN_ratio=0.2 | 2 | 4d2be45c989b0f75 | 0.870322 | 0.976479 / 0.975470 / 0.966747 | 0.977611 / 0.968684 / 0.955113 | 0.300358 / 0.401231 / 0.423721 | 0.405420 |
| pacmap | MN_ratio=0.5 | 0 | 486548b127956739 | 1.133078 | 0.976980 / 0.974869 / 0.966176 | 0.979408 / 0.971315 / 0.958606 | 0.289544 / 0.391734 / 0.421125 | 0.398721 |
| pacmap | MN_ratio=0.5 | 1 | e6909869fa72de51 | 0.913543 | 0.975610 / 0.972094 / 0.962306 | 0.974849 / 0.963891 / 0.947215 | 0.291899 / 0.397176 / 0.416513 | 0.342231 |
| pacmap | MN_ratio=0.5 | 2 | 893f7c339ae40ae2 | 0.895216 | 0.979087 / 0.977184 / 0.968955 | 0.976681 / 0.964546 / 0.945782 | 0.298041 / 0.397233 / 0.429760 | 0.365624 |
| pacmap | MN_ratio=1 | 0 | 111e0d9dd4a45458 | 1.033619 | 0.976292 / 0.974506 / 0.966748 | 0.979054 / 0.971279 / 0.959097 | 0.288627 / 0.393004 / 0.420561 | 0.389608 |
| pacmap | MN_ratio=1 | 1 | b7da69f6287bb39b | 1.149976 | 0.976617 / 0.974258 / 0.963646 | 0.974255 / 0.960449 / 0.939602 | 0.290071 / 0.398495 / 0.417976 | 0.341052 |
| pacmap | MN_ratio=1 | 2 | a5a4bd71408c14ad | 1.114984 | 0.978966 / 0.976283 / 0.967135 | 0.976362 / 0.964133 / 0.942287 | 0.302800 / 0.397888 / 0.427866 | 0.350599 |


## Métricas por run: CLIP

T=trustworthiness; C=continuity exacta; J=Jaccard original–reducido.


| método | configuración | seed | reduction_space_id | fit s | T@5/10/20 | C@5/10/20 | J medio@5/10/20 | Spearman |
|---|---|---|---|---|---|---|---|---|
| tsne | perplexity=10 | 0 | 56ea1353276cf437 | 11.535945 | 0.989717 / 0.979399 / 0.964475 | 0.984000 / 0.974607 / 0.962857 | 0.486360 / 0.432953 / 0.386410 | 0.522786 |
| tsne | perplexity=10 | 1 | effcc85521f0a2e9 | 11.112164 | 0.989445 / 0.979484 / 0.964558 | 0.983960 / 0.974526 / 0.962720 | 0.486681 / 0.432884 / 0.387340 | 0.522535 |
| tsne | perplexity=10 | 2 | 6306ceaa113c770f | 11.663332 | 0.989476 / 0.979634 / 0.964635 | 0.984026 / 0.974617 / 0.962849 | 0.485533 / 0.433032 / 0.387643 | 0.521015 |
| tsne | perplexity=30 | 0 | e273672b3e6b812e | 14.862725 | 0.989444 / 0.981167 / 0.968002 | 0.984305 / 0.976281 / 0.967658 | 0.490119 / 0.445452 / 0.407854 | 0.496726 |
| tsne | perplexity=30 | 1 | 8b4ddacfcef094df | 14.645185 | 0.989522 / 0.981266 / 0.968100 | 0.984343 / 0.976300 / 0.967629 | 0.489735 / 0.445658 / 0.407787 | 0.495368 |
| tsne | perplexity=30 | 2 | 07421d331a9d01d5 | 14.039015 | 0.989380 / 0.981328 / 0.967899 | 0.984329 / 0.976259 / 0.967569 | 0.489205 / 0.445761 / 0.407654 | 0.495710 |
| tsne | perplexity=50 | 0 | e90ffe5dbd2cca6d | 17.716474 | 0.989266 / 0.982556 / 0.971131 | 0.984421 / 0.976894 / 0.968393 | 0.470326 / 0.439814 / 0.408927 | 0.482239 |
| tsne | perplexity=50 | 1 | 03c68613aa6eabf0 | 17.255883 | 0.988746 / 0.982345 / 0.970987 | 0.984355 / 0.976922 / 0.968438 | 0.471069 / 0.439711 / 0.409241 | 0.480525 |
| tsne | perplexity=50 | 2 | 177b92c1287a25d5 | 17.285843 | 0.988830 / 0.982392 / 0.971110 | 0.984369 / 0.976928 / 0.968462 | 0.469834 / 0.438725 / 0.408700 | 0.482390 |
| pacmap | MN_ratio=0.2 | 0 | 4d51dd419d4b1bce | 0.849770 | 0.971020 / 0.968728 / 0.959752 | 0.979656 / 0.969512 / 0.954307 | 0.247914 / 0.332499 / 0.354839 | 0.423236 |
| pacmap | MN_ratio=0.2 | 1 | 59c8242f0ca80912 | 0.866751 | 0.972703 / 0.970366 / 0.964475 | 0.980149 / 0.971013 / 0.955729 | 0.249268 / 0.336005 / 0.364283 | 0.428039 |
| pacmap | MN_ratio=0.2 | 2 | 19ada5a37ebab01c | 0.925436 | 0.971745 / 0.968376 / 0.957400 | 0.979486 / 0.970266 / 0.954440 | 0.260091 / 0.340447 / 0.356766 | 0.401401 |
| pacmap | MN_ratio=0.5 | 0 | ae80f8c9018441c5 | 0.954631 | 0.968288 / 0.965418 / 0.956117 | 0.979619 / 0.970426 / 0.954719 | 0.247359 / 0.332093 / 0.356112 | 0.415853 |
| pacmap | MN_ratio=0.5 | 1 | 0a50e7b7cc7e93e7 | 1.014236 | 0.968288 / 0.966249 / 0.960453 | 0.979668 / 0.970353 / 0.954932 | 0.240293 / 0.324120 / 0.361921 | 0.430869 |
| pacmap | MN_ratio=0.5 | 2 | fbc602a54d77333c | 0.958262 | 0.974684 / 0.971417 / 0.963120 | 0.978368 / 0.968407 / 0.951175 | 0.262884 / 0.343296 / 0.364009 | 0.479292 |
| pacmap | MN_ratio=1 | 0 | 05838df32347c8d8 | 1.061841 | 0.971411 / 0.968820 / 0.959741 | 0.979608 / 0.969853 / 0.952572 | 0.250375 / 0.337266 / 0.355739 | 0.435227 |
| pacmap | MN_ratio=1 | 1 | a7ee859627213a93 | 1.181506 | 0.972250 / 0.970431 / 0.964434 | 0.979636 / 0.969886 / 0.953052 | 0.247620 / 0.333917 / 0.364090 | 0.458715 |
| pacmap | MN_ratio=1 | 2 | 6f15847d0a8fb39e | 1.167613 | 0.973728 / 0.970704 / 0.961034 | 0.979741 / 0.970379 / 0.953966 | 0.264701 / 0.344653 / 0.361311 | 0.414011 |


Cada `metrics.json` y `reports/reduction/tables/runs.csv` también conservan
mediana/Q1/Q3 de Jaccard para cada k y run; el HTML muestra las 108 filas de
run × k con esos cuatro agregados. Los valores por content_id están en Parquet
local. Spearman compara 1−coseno original con distancia euclidiana reducida;
no se infiere independencia estadística entre pares ni preservación perfecta.

## Estabilidad y alternativas

Jaccard entre semillas 0/1, 0/2 y 1/2, agregado por contenido. Los cuantiles
descriptivos completos están en `configuration_stability.csv` y el HTML.


| encoder | method | label | k | mean | median | Q1 | Q3 |
|---|---|---|---|---|---|---|---|
| dinov2 | pacmap | MN_ratio=0.5 | 5 | 0.388339 | 0.428571 | 0.250000 | 0.666667 |
| dinov2 | pacmap | MN_ratio=0.5 | 10 | 0.550992 | 0.538462 | 0.333333 | 0.818182 |
| dinov2 | pacmap | MN_ratio=0.5 | 20 | 0.644387 | 0.666667 | 0.481481 | 0.818182 |
| dinov2 | pacmap | MN_ratio=0.2 | 5 | 0.384290 | 0.428571 | 0.250000 | 0.428571 |
| dinov2 | pacmap | MN_ratio=0.2 | 10 | 0.543294 | 0.538462 | 0.333333 | 0.666667 |
| dinov2 | pacmap | MN_ratio=0.2 | 20 | 0.633923 | 0.666667 | 0.481481 | 0.818182 |
| dinov2 | pacmap | MN_ratio=1 | 5 | 0.390947 | 0.428571 | 0.250000 | 0.666667 |
| dinov2 | pacmap | MN_ratio=1 | 10 | 0.557722 | 0.538462 | 0.333333 | 0.818182 |
| dinov2 | pacmap | MN_ratio=1 | 20 | 0.649487 | 0.666667 | 0.481481 | 0.818182 |
| dinov2 | tsne | perplexity=10 | 5 | 0.965113 | 1.000000 | 1.000000 | 1.000000 |
| dinov2 | tsne | perplexity=10 | 10 | 0.957418 | 1.000000 | 1.000000 | 1.000000 |
| dinov2 | tsne | perplexity=10 | 20 | 0.948152 | 1.000000 | 0.904762 | 1.000000 |
| dinov2 | tsne | perplexity=30 | 5 | 0.975141 | 1.000000 | 1.000000 | 1.000000 |
| dinov2 | tsne | perplexity=30 | 10 | 0.970726 | 1.000000 | 1.000000 | 1.000000 |
| dinov2 | tsne | perplexity=30 | 20 | 0.974223 | 1.000000 | 1.000000 | 1.000000 |
| dinov2 | tsne | perplexity=50 | 5 | 0.954696 | 1.000000 | 1.000000 | 1.000000 |
| dinov2 | tsne | perplexity=50 | 10 | 0.951542 | 1.000000 | 1.000000 | 1.000000 |
| dinov2 | tsne | perplexity=50 | 20 | 0.950788 | 1.000000 | 0.904762 | 1.000000 |
| clip | pacmap | MN_ratio=0.5 | 5 | 0.378045 | 0.428571 | 0.250000 | 0.666667 |
| clip | pacmap | MN_ratio=0.5 | 10 | 0.530228 | 0.538462 | 0.333333 | 0.666667 |
| clip | pacmap | MN_ratio=0.5 | 20 | 0.625087 | 0.666667 | 0.481481 | 0.818182 |
| clip | pacmap | MN_ratio=0.2 | 5 | 0.378196 | 0.428571 | 0.250000 | 0.666667 |
| clip | pacmap | MN_ratio=0.2 | 10 | 0.530773 | 0.538462 | 0.333333 | 0.666667 |
| clip | pacmap | MN_ratio=0.2 | 20 | 0.612099 | 0.600000 | 0.428571 | 0.818182 |
| clip | pacmap | MN_ratio=1 | 5 | 0.387945 | 0.428571 | 0.250000 | 0.666667 |
| clip | pacmap | MN_ratio=1 | 10 | 0.541639 | 0.538462 | 0.333333 | 0.666667 |
| clip | pacmap | MN_ratio=1 | 20 | 0.631375 | 0.666667 | 0.481481 | 0.818182 |
| clip | tsne | perplexity=10 | 5 | 0.979003 | 1.000000 | 1.000000 | 1.000000 |
| clip | tsne | perplexity=10 | 10 | 0.971384 | 1.000000 | 1.000000 | 1.000000 |
| clip | tsne | perplexity=10 | 20 | 0.960011 | 1.000000 | 0.904762 | 1.000000 |
| clip | tsne | perplexity=30 | 5 | 0.970737 | 1.000000 | 1.000000 | 1.000000 |
| clip | tsne | perplexity=30 | 10 | 0.966986 | 1.000000 | 1.000000 | 1.000000 |
| clip | tsne | perplexity=30 | 20 | 0.968087 | 1.000000 | 1.000000 | 1.000000 |
| clip | tsne | perplexity=50 | 5 | 0.944786 | 1.000000 | 1.000000 | 1.000000 |
| clip | tsne | perplexity=50 | 10 | 0.934570 | 1.000000 | 0.818182 | 1.000000 |
| clip | tsne | perplexity=50 | 20 | 0.931451 | 1.000000 | 0.904762 | 1.000000 |


Agregados usados para seleccionar: cada fila reúne las tres semillas; T/C/J
y estabilidad promedian k=5/10/20. Los rangos se comparan por encoder/método.


| encoder | method | label | trustworthiness_mean | continuity_mean | preservation_mean | stability_mean | spearman_mean | pareto_nondominated | mean_criterion_rank |
|---|---|---|---|---|---|---|---|---|---|
| dinov2 | pacmap | MN_ratio=0.5 | 0.972585 | 0.964699 | 0.370336 | 0.527906 | 0.368859 | True | 2.000000 |
| dinov2 | pacmap | MN_ratio=0.2 | 0.971933 | 0.965490 | 0.370397 | 0.520502 | 0.362814 | True | 2.200000 |
| dinov2 | pacmap | MN_ratio=1 | 0.972717 | 0.962946 | 0.370810 | 0.532719 | 0.360420 | True | 1.800000 |
| dinov2 | tsne | perplexity=10 | 0.983022 | 0.981970 | 0.505631 | 0.956895 | 0.427861 | False | 2.600000 |
| dinov2 | tsne | perplexity=30 | 0.984211 | 0.982381 | 0.512869 | 0.973363 | 0.432205 | True | 1.600000 |
| dinov2 | tsne | perplexity=50 | 0.985810 | 0.982850 | 0.505335 | 0.952342 | 0.456259 | True | 1.800000 |
| clip | pacmap | MN_ratio=0.5 | 0.966004 | 0.967519 | 0.314676 | 0.511120 | 0.442005 | True | 2.400000 |
| clip | pacmap | MN_ratio=0.2 | 0.967174 | 0.968284 | 0.315790 | 0.507023 | 0.417559 | True | 2.200000 |
| clip | pacmap | MN_ratio=1 | 0.968061 | 0.967633 | 0.317741 | 0.520320 | 0.435984 | True | 1.400000 |
| clip | tsne | perplexity=10 | 0.977869 | 0.973796 | 0.435426 | 0.970133 | 0.522112 | True | 2.200000 |
| clip | tsne | perplexity=30 | 0.979568 | 0.976075 | 0.447692 | 0.968604 | 0.495934 | True | 1.800000 |
| clip | tsne | perplexity=50 | 0.980818 | 0.976576 | 0.439594 | 0.936936 | 0.481718 | True | 2.000000 |


## Interpretación y límites observados

En las cuatro referencias, T@20 se sitúa entre 0.9597 y 0.9767, mientras Jaccard@20
medio va de 0.3557 a 0.4839. Esto muestra que penalizar poco los rangos de vecinos
intrusos no equivale a conservar sus conjuntos exactos. En este protocolo, las
referencias t-SNE tienen estabilidad@20 de 0.9742/0.9681 y las PaCMAP de
0.6495/0.6314 (DINOv2/CLIP). Es una observación del grid y no una conclusión
universal sobre los métodos ni su calidad semántica.

Los valores por run permiten comparar preservación local, conjuntos exactos y
orden de distancias; una T cercana a 1 no significa que todos los vecinos sean
los mismos. La estabilidad de la proyección tampoco valida automáticamente su
estructura semántica. No se establece superioridad de DINOv2 sobre CLIP ni de
un método por su apariencia. La regla es exploratoria y los criterios están
correlacionados; se conservan todas las alternativas para análisis posterior.

Las 11 figuras responden a preservación (2), geometría de referencias (4),
estabilidad (2), interpretación temporal (2) y pertenencias históricas (1).
Los ejemplos temporales usan la misma ventana central del primer tramo elegible
en ambos encoders, sin consultar coordenadas. Los índices son inferidos y no
hay timestamps verificados. El overlay histórico conserva todas las pertenencias
de cada contenido; no asigna arbitrariamente un split a copias exactas.
La ventana observada corresponde a índices 342–371 de video_11min. La inspección
de ambas figuras muestra tramos locales y saltos entre regiones de la proyección;
no se atribuyen automáticamente a cambios reales de escena ni a leakage.

Notebook fuente: `notebooks/reduction_review.ipynb`, sin outputs. Evidencia local:
`reports/reduction/review/reduction_review.executed.ipynb` y
`reports/reduction/review/reduction_review.html`, 14 secciones, código oculto.
Tablas, figuras, coordenadas, hashes e índices reales permanecen ignorados por Git.
Los logs/recibos están en `reports/reduction/execution/`.

Las ejecuciones conservan el commit realmente activo y worktree dirty, junto con
huellas de fuente; no se reescribe metadata para atribuir los ajustes al commit de
cierre. Se conserva una copia local del código numérico/configuraciones/protocolo
anterior al primer fit y un recibo de verificación del código final.

## Consumo por clustering

El [protocolo de clustering](../protocols/clustering.md) desarrolla la evaluación
de DBSCAN/OPTICS/HDBSCAN en los embeddings L2 originales y en las referencias
candidatas, con grillas acotadas y distancias/escalas propias de cada espacio.
La densidad 2D no equivale a densidad original. Contrastar estabilidad entre
semillas/perturbaciones, coherencia visual y temporal, ruido y cobertura. AMI/ARI
requieren asignaciones reales y una política explícita para ruido y comparación.
Conservar la unidad content_id y las 1657 ocurrencias. La partición debe
mantener grupos íntegros y medir correlación residual antes de comparar detectores.
Los resultados de esa fase están en [clustering](clustering.md); no forman parte
del experimento de reducción descrito aquí. Bhattacharyya sigue condicionado.
