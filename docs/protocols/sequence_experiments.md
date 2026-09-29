# Experimentos de estructura temporal y dependencia visual

Infraestructura experimental, separada de `sequences detect/build` v1 y del
protocolo histórico `content_density_v1`. No crea secuencias, VDGs ni splits.
Todas las publicaciones conservan `ground_truth=false`, `split_created=false`
y `automatic_confirmation=false`. La revisión manual tampoco constituye ground
truth. Una ejecución correcta no valida semánticamente los grupos.

## Conceptos e hipótesis evaluables

| Concepto | Interpretación permitida |
|---|---|
| sequence_instance | Segmento continuo de una línea de origen, suministrado mediante revisión explícita |
| sequence_core | Complemento conservador de zonas revisadas, o intervalo explícito; candidato, no instancia confirmada |
| boundary_zone | Intervalo inclusivo de incertidumbre; su pico diagnóstico no es un corte exacto |
| cluster | Agrupamiento algorítmico de contenido visual único; evidencia complementaria |
| visual_dependency_group | Restricción futura que mantiene juntas varias instancias; no fusiona sus identidades temporales |
| recurrence candidate | Evidencia para revisión; un componente conectado candidato no es un VDG |

Preguntas: ¿los cambios L2 coinciden con zonas revisadas?, ¿cuánto fragmenta o
mezcla cada agrupamiento los cores?, ¿coincide la recurrencia por vecinos con
la co-pertenencia a clústeres?, ¿qué evidencia permanece al cambiar semillas
o parámetros? No se declara un ganador ni ausencia de leakage automáticamente.

## Correspondencia bibliográfica y diferencias

[Figueiredo y Mendes (2024)](https://doi.org/10.1109/ACCESS.2024.3383047)
estudia CLIP → t-SNE → DBSCAN/OPTICS/Agglomerative y partición por agrupamientos,
con comparaciones ARI/AMI. Aquí esa ruta aporta evidencia complementaria; no se
ejecuta su paso de partición. Agglomerative usa un adaptador experimental
explícito, sin ampliar silenciosamente el protocolo de densidad comprometido.

[Glazner et al. (2025), v1](https://arxiv.org/html/2511.13944v1), propone features
→ PaCMAP → HDBSCAN y evalúa AMI/V-measure frente a identidades de video conocidas.
Su experimento usa **PaCMAP de 256 dimensiones** y, entre otros, **DINOv3**.
Esta suite usa los adaptadores existentes de **2/3 dimensiones y DINOv2**:
es una comparación inspirada, no una reproducción exacta del artículo.
Las referencias no demuestran superioridad universal ni validez de un split FLIR.

## Fuentes, identidades y cobertura

Se verifica el manifest completo y ambos stores raw/L2 con el QA existente,
incluyendo modelo, revisión, pooling, dimensiones, normalización e índices.
Cada encoder mantiene su propio `embedding_row`; `content_row` alinea el
subconjunto canónico ordenado por `content_id`. Fitting recibe exclusivamente
vectores y IDs únicos: ni split histórico, bounding boxes, zonas ni labels manuales.
Cada artefacto pertinente conserva una tabla completa de ocurrencias.

Para `flir_video_samples_v1`, la línea es `video_id/sample_index`. Para el
manifest histórico es `possible_sequence/possible_frame_index`, **inferencia por
nombre**, sin timestamps ni procedencia física confirmada. Las copias históricas
del mismo contenido en la misma posición cuentan una sola vez en la señal
temporal. El mismo contenido en otra posición conserva esa ocurrencia. Contenidos
distintos en una misma posición nominal se rechazan por ambigüedad. Los gaps
interrumpen ventanas; no se interpolan imágenes.

El ID científico de entrada ignora orden de filas y split histórico. Cada
publicación también liga los bytes completos de sus fuentes: reordenar el
manifest cambia el binding/publicación, aunque no cambie el fitting ni los
candidatos canónicos. Esa diferencia es intencional.

## Señales temporales

Se reutiliza `centroid_changes` sobre originales L2. En cada corte y ventana
completa w: `1 - cos(sum(x[t-w:t]), sum(x[t:t+w]))`, normalizando ambos
centroides. Norma cero deja el resultado indefinido. w=1 conserva además coseno
adyacente. Cada encoder/segmento/ventana obtiene percentiles con redondeo a 12
decimales y empates promedio, siguiendo el helper existente.

También se ejecuta `detect_tables` v1 como control `current_v1:consensus`, con
su configuración completa persistida en `boundary.current_v1`. Sus scores/eventos
se guardan aparte; sus intervalos de búsqueda se presentan como zonas candidatas.
La posición localizada permanece diagnóstico, nunca una frontera comprometida.
En familias sin timestamps se guarda NaN; no se fabrica una escala de segundos.

El contexto local excluye el corte consultado: mediana, MAD, número de valores,
exceso respecto de mediana y z robusto. MAD cero deja z indefinido; nunca se
sustituye por infinito ni cero. La candidatura requiere cambio absoluto mínimo,
exceso local y percentil configurados. El multiescala exige todas las ventanas;
el consenso exige ambos encoders. CLIP, DINOv2, consenso y adjacent-only quedan
separados como ablaciones. Las ventanas se miden en muestras disponibles; el
gap máximo admisible se persiste. Los intervalos agrupan candidatos cercanos y
conservan el contexto anterior. No se localiza un corte exacto ni se acepta por
similitud baja solamente.

## Evaluación posterior y ruido

Targets: `sequence_core`, `sequence_instance`, `known_source_video`. El último
requiere referencias explícitas: jamás se usa la familia inferida como verdad
de origen. Sin revisión, la suite aún ejecuta señales/clustering/transiciones,
pero deja explícitamente pendiente la evaluación de pertenencia y recurrencia.

Una etiqueta de contenido es evaluable solo si **todas sus ocurrencias** tienen
etiqueta conocida, fuera de zonas, y coinciden. Copias que aparecen en dos cores
quedan fuera de métricas de etiqueta única y permanecen en recurrencia/ocurrencias.
`evaluation_mask` explica las exclusiones. ARI, AMI, homogeneity, completeness y
V-measure se acompañan siempre de N, cobertura de contenidos y ocurrencias.
Se reportan por separado incluyendo ruido como una etiqueta y excluyéndolo;
N<2 produce null y particiones triviales se marcan.

Purity temporal pondera la fracción dominante por miembros evaluados no-noise.
Entropía es Shannon en bits. Fragmentación cuenta clústeres no-noise por target
y registra ruido aparte; merging cuenta targets presentes por clúster.
Temporal recall@1/5/10 usa pares de contenidos distintos dentro de un mismo
target/timeline, deduplicados sobre todas las ocurrencias, con delta nominal
positivo ≤ umbral. No equivale a leakage confirmado. Coherencia visual@5/10/20
usa vecinos originales L2 y queries no-noise, con denominador explícito;
si N no soporta k, queda indefinida, no se cambia k silenciosamente.

## Recurrencia y consenso

Cada par de cores disjuntos conserva, por encoder: coseno de centroides,
NN A→B/B→A, mediana/p95, fracciones sobre cada umbral, soporte simétrico mínimo
y NN mutuos. Los matches individuales permiten revisión. Los grupos usan
contenidos únicos; sus ocurrencias no se descartan. Los contenidos exactos
compartidos son evidencia de copia, separada de similitud y de confirmación visual.

Selección amplia: soporte mínimo en cualquiera de los encoders. Refinada:
soporte mínimo en ambos y peor rango promedio de sus medianas dentro de la
fracción configurada. Los rangos se calculan independientemente por encoder.
Se guardan desacuerdos, tasas candidatas y advertencias por tasa > umbral.
Los defaults son **parámetros exploratorios**, no calibración validada sobre FLIR.
La calibración exige revisión externa y nuevas configuraciones inmutables.

Las aristas `exact_copy_edges` conservan el content_id compartido y sus dos
elementos temporales. Son evidencia exacta por identidad de contenido del
manifest; no crean un VDG ni fusionan instancias. Los cores generados por
complemento llevan `status=candidate`, `origin` explícito y decisión manual
vacía. Una zona `ambiguous` también excluye sus frames de evaluación exacta,
sin convertirse por ello en un corte soportado.

La recurrencia por clústeres excluye noise, cuenta clústeres compartidos y el
soporte por miembros en cada dirección. Soporte distribuido exige cantidad y
fracción mínimas en ambos cores; un miembro incidental queda diferenciado.
Hay tablas de acuerdo/desacuerdo con recurrencia directa y Spearman entre
soporte y rango invertido, solo si ambos varían y hay al menos tres pares.

## Transiciones, estabilidad y ablaciones

RLE por línea/grilla; los gaps y noise interrumpen transiciones. Se registran
longitud posterior, persistencia tras N muestras, cambios aislados, retornos
A→B→A y gaps de recurrencia. El retorno de un cluster no fusiona instancias.
Se evalúa coincidencia con intervalos revisados inclusivos ± tolerancia, nunca
con un punto de referencia inventado. Recall cuenta cada zona una vez.
Con revisión parcial, transiciones externas no son falsos positivos demostrados:
`candidate_precision` y `false_transitions` son null. Solo una declaración
explícita de revisión exhaustiva habilita esas métricas frente a evidencia manual.

Se reutiliza `assignment_agreement`: ARI/AMI all-points y common-clustered con
cobertura, trivialidad y estabilidad de noise. Semillas se comparan con el resto
de parámetros fijo; perturbaciones locales difieren en un solo parámetro del
mismo espacio; otras representaciones quedan identificadas aparte. ARI≈1 con
cobertura baja no demuestra estabilidad global. Se añade Jaccard de candidatos
de transición, pares recurrentes y posiciones cubiertas por intervalos temporales;
dos sets vacíos producen null, no estabilidad perfecta.

La tabla comparativa cruza encoder, representación, algoritmo, semilla,
parámetros y métricas con cobertura. Ablaciones de frontera: adjacent-only,
multiescala, CLIP, DINOv2, consenso, transiciones, unión/intersección diagnóstica.
Las uniones/intersecciones producen zonas candidatas para revisión. La tabla
retiene objetivos y cobertura sin un ranking universal ni winner automático.

## Publicación y límites operacionales

Artefactos inmutables bajo `<root>/<artifact_kind>/<artifact_id>/`, con config,
fuentes, versiones, checksums de todos los archivos, ID del contenido lógico,
summary, tablas Parquet y recibo de metadata. No se sobrescribe un directorio
incompleto. Verify comprueba inventario exacto, bytes, identidad lógica, flags,
fuentes suministradas y dependencias/children; no vuelve a ajustar un reductor
estocástico ni certifica la verdad científica de una revisión.

Clustering reutiliza sus cotas/caches existentes y aún puede requerir O(N²)
memoria. Recurrencia y vecinos usan bloques B×N. No hay medición de RSS/tiempo
completa de esta suite en Hypatia. El perfil de 154 fits es exploratorio y puede
exceder los recursos del launcher de ejemplo. Las celdas inválidas quedan en
`failures.parquet`; `all_requested_cells_succeeded=false` impide presentarlas
como ejecutadas con éxito. La CLI publica el resultado parcial y devuelve código
1 cuando hay celdas fallidas, para que SLURM no lo trate como éxito completo.
Sin celdas exitosas no se publica una suite evaluada.

La ingesta nativa de evidencia legacy sigue el
[contrato observado y runbook](../runbooks/native_sequence_evidence.md).
`sequence_core_candidate` se mantiene como evidencia candidata fuera de zonas,
sin promoverla a instancia ni a etiqueta de fitting. Ranks/shortlists heredados
se preservan en tablas `native_*`; no reemplazan resultados nuevos calculados con
features. Los JSONs originales conservan el alcance de las asociaciones de
lineage. Los archivos reales siguen pendientes de una importación verificada en
Hypatia, aun cuando el adaptador y las pruebas sintéticas estén disponibles.
