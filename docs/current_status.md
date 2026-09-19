# Estado del proyecto

Revisado **2026-09-17** (consolidación documental; sin nuevos experimentos). **Semanas 6, 7 y 8 COMPLETED**, **semana 9: coseno DONE,
análisis temporal PARTIAL** y **semanas 9–10: reducción DONE**, **semana 10: clustering y candidatos DONE WITH LIMITS**, **semanas 10–11: splitting DONE WITH LIMITS**, según la evidencia descrita a continuación y los criterios
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
| 9–10 — reducción dimensional | DONE | 18 t-SNE + 18 PaCMAP completos, ambos encoders; T/C/Jaccard/Spearman, estabilidad de tres semillas, cuatro referencias, verificación con fuentes y HTML de 14 secciones/11 figuras |
| 10 — clustering y selección exploratoria | DONE WITH LIMITS | 414 runs completos, tres algoritmos/seis espacios; métricas originales, 204 ARI/AMI de perturbaciones, 54 shortlist, 41 Pareto, verificación con fuentes y HTML de 18 secciones/17 figuras; sin split nuevo |
| 10–11 — cluster-aware splitting y baselines | DONE WITH LIMITS | 66 runs verificados; historical + random 0–4 + 12 candidatos × cinco seeds; ambos encoders y temporal; cero exact overlap en 65 nuevos runs, cero fracturas en 60 cluster-aware; seis Pareto, dos representantes; 18 secciones/nueve figuras; 65 asignaciones reconstruidas exactamente |
| Detector — protocolo e infraestructura | IMPLEMENTED + SMALL CPU PILOTS VALIDATED; FINAL PENDING | Historical/random_content/C10/C12 congelados; 16 vistas; cuatro pilotos 24/12/20 imágenes × dos epochs, batch 2/640 CPU, cinco clases test y 1000 bootstrap revalidados; Stage A/B completos no ejecutados por coste |

Las extracciones completas ya existían desde 2026-09-09. Esta revisión verifica
arrays, índices, cobertura y metadata; **no vuelve a extraer embeddings**.
La referencia bibliográfica exacta de la nomenclatura sigue pendiente. El orden
de clases fue confirmado explícitamente por el responsable del proyecto;
la [evidencia y su límite](dataset_classes.md) se conservan sin inventar una cita.

## Reporte acumulativo de avance

`notebooks/progress_review.ipynb` consolida las etapas existentes en 22 secciones
en español, con pregunta, fuentes y hallazgo por sección. Reutiliza 12 figuras,
presenta dos resúmenes temporales nuevos desde cuartiles registrados y añade un
diagrama del pipeline. El HTML local, con código oculto, se construye
con `uv run --extra reporting python scripts/build_progress_review.py` en
`reports/progress/review/`; no reemplaza los reportes específicos.

El builder verifica disponibilidad, checksums registrados, consistencia de
resúmenes, identidad/conteos del manifest y privacidad del HTML. Reutiliza los
recibos de validación numérica anteriores; no vuelve a ejecutar experimentos ni
modifica resultados. Los faltantes se muestran como `missing / invalid`.
El estado científico no cambia. Véase [guía y mapa de fuentes](progress_review.md).

La iteración de inspección del **2026-09-19** añade un [explorador local](cluster_split_explorer.md)
Streamlit opcional para clusters, splits, comparación y noise singleton. Lee
asignaciones existentes y ZIPs en memoria; el playback separa secuencias y gaps
inferidos, con FPS exclusivamente de visualización. No cambia C10/C12, métricas
ni experimentos. Los hexbins permanecen en similarity_review; el acumulativo usa
mediana/Q1–Q3 y conteos en los siete bins ya poblados.

## Detector: infraestructura y piloto del portátil

Se seleccionaron **historical, random_content, C10 y C12** antes de YOLO.
C10 conserva su papel principal; C12 reduce similitud media y temporalidad
residual a cambio de peor balance. C01 queda como referencia descriptiva.
Plan local `ace1ffd6bb4775d0`: 16 splits, 48 celdas objetivo, detector seeds
42/43/44 y split seeds 0–4. No se reoptimizó ninguna asignación.

Implementados protocolo y runbook, vistas por ocurrencia con hashes de bytes,
extra opcional Ultralytics, prueba de hardware/batch, freeze, piloto, gate
Stage A/B, resume aislado, P/R a umbral fijo, AP, bootstrap y agregación jerárquica.
El freeze CPU es `9897062efdb32450`. El portátil detectado tiene Ryzen 7 7730U,
8 núcleos/16 hilos, ~21.84 GiB de RAM utilizable, sin CUDA en PyTorch.

**Ejecutado y revalidado:** 16 vistas del dataset; probe de batches 1/2 a 640;
cuatro pilotos reales pequeños de YOLO11n con CPU/batch 2/imgsz 640, dos epochs,
24/12/20 imágenes y seeds 0/42. Training total 140.39 s; evaluación/bootstrap
57.46 s. Se reconstruyeron métricas y 1000 bootstrap de cada piloto; hashes,
soporte de annotations del loader, cinco clases test y configuración efectiva
coinciden. Los originales no se alteraron.

**Pendiente:** Stage A completo a 50 epochs, Stage B, métricas científicas
multi-seed, variabilidad e interpretación del detector. El piloto extrapola
~408–593 horas para los 48 runs (dos métodos, no intervalo de confianza).
Se detuvo antes del protocolo costoso CPU por instrucción explícita del usuario.
No hay ejecución GPU ni prueba de resume interrumpido en GPU.

Notebook/HTML de 18 secciones; una figura de contexto real y ocho paneles
explícitamente pendientes. Evidencia local en `artifacts/detection/` y
`reports/detection/review/`. Detalle en [análisis del piloto](detector_comparison_analysis.md),
[protocolo](detector_experiment_protocol.md) y [runbook](detector_runbook.md).

## Semana 9: evidencia completa de similitud

El apartado siguiente conserva el cierre de semana 9; la evidencia posterior de
reducción se registra después, sin reescribir los experimentos anteriores.

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
coherencia visual global. Durante esa fase no se implementaron ni ejecutaron
reducción, clustering, nuevas particiones o YOLO; Bhattacharyya sigue condicionado.

## Semanas 9–10: reducción completa y referencias exploratorias

Se revalidaron ambos espacios completos y las matrices/vecinos de semana 9.
Se ejecutaron **36 runs: 18 t-SNE y 18 PaCMAP**, sobre **1459 contenidos únicos**
por run, con salida float32 **1459 × 2**, finitud, índices alineados y rango 2.
El ajuste usa únicamente L2 original, 384D/512D por separado. No hubo pre-PCA,
concatenación ni renormalización de archivos; PCA se usa solo para inicializar.
PaCMAP registra además su transformación afín interna con `apply_pca=false`.

Grid: t-SNE perplexity 10/30/50, 1000 iteraciones máximas, learning rate auto
(efectivo 50); PaCMAP n_neighbors 10, FP_ratio 2, MN_ratio 0.2/0.5/1.0,
fases 100/100/250. Cada configuración usa semillas 0/1/2. El protocolo y código
numérico se conservaron antes del primer fit; sus huellas no cambiaron durante
el grid. El tiempo de ajuste sumado fue **269.038 s**: t-SNE 251.284 s y PaCMAP
17.753 s, sin incluir toda la verificación, agregación y reporte.

| Encoder | Método / parámetro seleccionado | reduction_space_id, semilla 0 | T@20 | C@20 | J@20 | Estabilidad@20 | Spearman |
|---|---|---|---:|---:|---:|---:|---:|
| DINOv2 | t-SNE, perplexity 30 | 476786e28eb2bcb8 | 0.976688 | 0.976196 | 0.483902 | 0.974223 | 0.430447 |
| DINOv2 | PaCMAP, MN_ratio 1.0 | 111e0d9dd4a45458 | 0.966748 | 0.959097 | 0.420561 | 0.649487 | 0.389608 |
| CLIP | t-SNE, perplexity 30 | e273672b3e6b812e | 0.968002 | 0.967658 | 0.407854 | 0.968087 | 0.496726 |
| CLIP | PaCMAP, MN_ratio 1.0 | 05838df32347c8d8 | 0.959741 | 0.952572 | 0.355739 | 0.631375 | 0.435227 |

T/C/J/Spearman corresponden al run semilla 0; estabilidad es Jaccard de vecinos
entre los tres pares de semillas. Se preservan métricas de todas las ejecuciones,
k=5/10/20 y media/mediana/Q1/Q3 de Jaccard. Spearman usa 100000 pares canónicos
compartidos, semilla 0. La regla de selección se fijó antes: frente no dominado y
media de rangos de cinco criterios. Las cuatro referencias son **candidatas**,
sin afirmar calidad de clustering ni superioridad entre encoders.

Los [resultados por run](reduction_analysis.md), [protocolo](reduction_protocol.md)
y [runbook](reduction_runbook.md) explican parámetros, unidades y trazabilidad.
Se generó y verificó el HTML local `reports/reduction/review/reduction_review.html`:
14 secciones, 11 figuras, sin errores, código oculto, sin rutas privadas ni hashes
de contenido visibles. Se inspeccionaron las 11 figuras. La ventana temporal usa
los mismos 30 contenidos, índices inferidos 342–371 de video_11min, en ambos
encoders. Los colores históricos conservan todas las pertenencias por contenido.
Datos y outputs reales permanecen ignorados por Git.

Al cerrar reducción, el siguiente paso era contrastar DBSCAN/OPTICS/HDBSCAN.
Durante esa fase no se ejecutó 3D, clustering, AMI/ARI, nuevos splits ni YOLO.
La evidencia posterior de clustering se registra a continuación; la validación
temporal sigue PARTIAL y densidad 2D no equivale a densidad original.

## Semana 10: clustering completo y selección exploratoria

**414 runs sobre 1459 contenidos únicos**: 132 DBSCAN, 186 OPTICS y 96 HDBSCAN.
El screening ejecutó 342; la shortlist de 54 configuraciones añadió 72 ajustes
de semillas reducidas 1/2. Los controles L2 originales se conservaron junto a
t-SNE y PaCMAP. No se recalcularon features, matrices ni reducciones.

Se observaron 39 single-cluster, cero all-noise, 18 nearly-all-noise y 54
dominant-cluster (flags solapables); rango 1–65 grupos y noise 0–0.984921.
Se evaluaron silhouette en el encoder original excluyendo noise, cosenos
intracluster exactos, medoides originales, secuencia dominante/entropía,
retención temporal@1/5/10 y visual@5/10/20, con cobertura explícita.

**204 comparaciones ARI/AMI**: 108 entre semillas y 96 de parámetros vecinos.
Se conservan all-points y common-clustered con medias, mínimos, N y cobertura;
los controles originales no tienen semilla ficticia. El Pareto de B deja **41
candidatos**, sin ganador único ni suma arbitraria de pesos. Los seis ejemplos
de figuras maximizan silhouette dentro del frente; R1/R4 originales agrupan solo
38/61 contenidos (2.60%/4.18%), por lo que no son recomendaciones de split.

Se verificaron los 414 artefactos contra las fuentes y se recalcularon métricas,
medoides, acuerdos y selección. HTML/notebook ejecutados: 18 secciones, 17 figuras,
seis galerías y 144 posiciones de imágenes verificadas desde ZIP en memoria.
Se revisaron las 17 figuras; las galerías muestran tanto vistas semejantes como
cambios de encuadre, contraste y overlays. Esa revisión no valida cada escena.

[Resultados completos y 41 IDs candidatos](clustering_analysis.md),
[protocolo previo al grid](clustering_protocol.md) y [runbook](clustering_runbook.md).
Artefactos y outputs reales permanecen en `artifacts/clustering/` y
`reports/clustering/`, ignorados por Git. No se introdujeron clases/split/tiempo
en fit ni se corrigieron conflictos de anotación.

Al cerrar semana 10, el siguiente paso era el protocolo de cluster-aware
splitting. En esa fase no se generaron splits ni balanceo. La ejecución posterior
se registra a continuación; durante esa fase no se ejecutó YOLO. El piloto
pequeño posterior del detector se describe al inicio de este documento.

## Semanas 10–11: particiones ejecutadas y evaluadas

**66 runs**, todos con 1459 contenidos y 1657 registros verificados. El histórico
preserva sus memberships y 198 contenidos exactos cross-split. Las cinco
particiones random content-level y 60 cluster-aware tienen **cero duplicados
exactos cross-split**; los 60 cluster-aware tienen **cero fracturas**. Se conservaron
las 4168 instancias, 292 labels vacíos y ocho conflictos de anotación sin corregir.

La selección preparatoria tomó 12 de los 41 Pareto, con diversidad de encoders,
original/t-SNE/PaCMAP y DBSCAN/OPTICS/HDBSCAN, sin silhouette como criterio.
Noise singleton y semillas 0–4. SciPy MILP balanceó registros, clases y vacíos
sin optimizar similitud; 35 solves óptimos dentro de tolerancia y 25 incumbentes
factibles con gap explícito. Reproducción posterior: **65/65 nuevas asignaciones
idénticas**. Targets derivados del manifest: 1178/107/372 registros.

Se evaluaron NN cross-split completos, rank-1/top-5/10/20, seis cohortes por
cuantil de ambos encoders, ventanas temporales Δ≤1/5/10/25, fragmentación de
secuencia, fracturas y robustez de nombres de split. Seis configuraciones
satisfacen las restricciones y quedan Pareto; dos anclas representantes seed 0:

| Candidato | split_space_id | Registros train/val/test | HM train/val/test | Vacíos train/val/test | Pares top0.1% DINO / CLIP |
|---|---|---|---|---|---|
| C10 — DINOv2/PaCMAP/DBSCAN, ancla visual | 88ccf4e12335a83f | 1178/107/372 | 95/7/27 | 207/19/66 | 6 / 7 |
| C01 — CLIP/original/OPTICS, ancla de balance | 643cd594f431cfc8 | 1178/107/372 | 92/8/29 | 207/19/66 | 229 / 311 |

El histórico tiene 328/277 pares extremos; random, medias 478/466.8. C10 reduce
la media NN en ambos encoders, pero su fracción temporal Δ≤5 es **23.93%**, frente
a random **43.92%** e histórico **14.76%**. No hay mejora uniforme: C01 mantiene
97.4% de ruido y su media NN supera al histórico en ambos encoders. El histórico
carece de Vehicles y Heavy Machinery en val; ambas representantes cubren cinco
clases, con apenas dos instancias Vehicles en val de C10.

[Resultados agregados](splitting_analysis.md), [protocolo](splitting_protocol.md)
y [runbook](splitting_runbook.md). HTML local de 18 secciones/nueve figuras y
notebook fuente limpio; **124 tests sintéticos y Ruff aprobados**. Artifacts,
assignments y reportes ejecutados permanecen ignorados. El protocolo principal
está ejecutado y validado; la generalización del detector y temporalidad real
siguen pendientes. No hubo exportación real, materialización ni YOLO.

## COMPLETADO en este alcance

- Diseño inicial y arquitectura del pipeline.
- Inventario y auditoría de ZIP, imágenes y etiquetas, conservando los originales.
- Manifest canónico candidato v1, identificadores y mapping ocurrencia/contenido.
- Caracterización separada de imágenes e instancias por clase con nombres canónicos.
- Estadísticas de ancho, alto, área y ratio normalizados por instancia/clase;
  boxplots de área y ratio por clase, conservando extremos.
- Auditoría de duplicados exactos y conflictos de anotación.
- Caracterización temporal disponible, con heurística y confianza explícitas.
- Línea base histórica preservada; nuevos splits derivados separados y auditados.
- Diagnósticos sobre los 1459 contenidos únicos.
- Pipeline DINOv2/CLIP con checkpoints, metadata, raw/L2 y validaciones.
- **DINOv2-small completo: 1459 × 384**, con mapping de 1657 registros y revisión resuelta.
- **CLIP ViT-B/32 completo: 1459 × 512**, con mapping de 1657 registros y revisión resuelta.
- Reporte técnico en español: 16 secciones, notebook ejecutado local y HTML sin código visible.
- Similitud coseno completa por encoder y revisión independiente de 14 secciones;
  análisis posterior de secuencia, Δ y pertenencias históricas, y comparación Jaccard.
- Reducción t-SNE/PaCMAP completa: 36 runs, preservación/estabilidad, cuatro referencias
  exploratorias y reporte independiente de 14 secciones, con artefactos verificados.

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

Validación adicional de procedencia temporal y escenas; Bhattacharyya cuando
exista una representación distribucional justificada; revisión de grupos y
conflictos de los splits candidatos y ejecución final controlada del detector
con presupuesto viable. El protocolo, 16 vistas y cuatro pilotos pequeños CPU
ya existen; los 48 entrenamientos finales y sus conclusiones no se ejecutaron.
Particiones, política singleton, random y comparación
residual ya se ejecutaron en semanas 10–11 con los límites registrados arriba.
DBSCAN/OPTICS/HDBSCAN, métricas, AMI/ARI y selección exploratoria sí se ejecutaron
en semana 10, con las limitaciones y candidatos registrados arriba.

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

## Validación de semanas 9–10

- `uv run --no-sync pytest`: **81 passed**; `uv run --no-sync ruff check .`:
  **All checks passed**. Tests sintéticos offline, incluyendo métricas exactas, invariancia geométrica,
  repetición por semilla, índices, corrupción, selección y procedencia posterior.
- CLI `reduction run/benchmark/summary/verify` y ayudas validadas; los dos benchmarks
  completos pasan verificación con features, similitud y manifest, recalculando métricas.
- Los tres notebooks fuente pasan el checker; el nuevo notebook ejecutado/HTML
  pasó comprobaciones de 14 secciones, 11 figuras, código oculto y privacidad.
- Python 3.11.14 con core/dev/reduction/reporting del lockfile; sin GPU ni modelos.
  Ajustes con scikit-learn 1.9.1, PaCMAP 0.9.1 y una política de hilos explícita.
- Recibo local `reports/reduction/execution/verification_receipt.json`, logs,
  snapshot del protocolo/fuentes antes del grid y metadata original conservados.

## Validación de semana 10

- `uv run --no-sync pytest`: **100 passed**, 99.19 s en esta ejecución; pruebas
  sintéticas offline, sin datos FLIR, modelos, Internet ni GPU.
- `uv run --no-sync ruff check .`: **All checks passed**; cuatro notebooks
  fuente sin outputs y con todas sus celdas de código compilables.
- CLI `clustering run/sweep/compare/verify/summary` y las cinco ayudas validadas.
  El run individual reutilizó un artefacto completo; los 414 runs no incluyen
  ese acceso a caché como otro experimento.
- Verificador final con fuentes: métricas/medoides recalculados para los 414
  runs, 204 ARI/AMI y agregados/topología de perturbaciones/selección verificados.
- HTML ejecutado sin errores, 18 secciones, 17 figuras, código oculto, sin
  rutas privadas, hashes completos ni NaN visibles; revisión visual de las
  17 figuras. Se guardaron probabilidades en 96 HDBSCAN y diagnósticos en 186 OPTICS.
- Protocolo, configuraciones y fuentes del ajuste preservados en snapshot previo.
  Después se reforzaron guards/verificación en `experiments.py`; no cambiaron
  ajuste, métricas ni selección. Metadata mantiene el commit activo y estado
  dirty reales del experimento, sin reatribuirlos al commit posterior de cierre.
- Recibo local `reports/clustering/execution/verification_receipt.json` y logs
  `final_commands.json`; todos los artefactos/reportes reales están ignorados
  por Git. Se comprobó ausencia de content IDs reales y rutas privadas en los
  archivos públicos antes de publicar.
