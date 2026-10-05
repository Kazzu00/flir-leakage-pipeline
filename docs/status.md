# Project status

## Evidencia organizativa M02 para frontend (2026-10-04)

**Infraestructura implementada; export real pendiente en Hypatia.**
`explorer export-organization` consume el plan/freeze existente, resuelve sus
splits y clustering seleccionado, y exporta ocurrencias, contenidos únicos,
membresías, timelines y evidencia candidata/revisión existente. No regenera
clustering/splits/linkage ni modifica el contrato del detector.

El schema `organization-evidence-v1` valida relaciones y semántica conservadora;
los previews son opcionales e ignorados por Git. Hay fixtures sintéticas para
duplicados, particiones históricas, seeds, ruido, candidatos, zonas inclusivas,
media y publicación con rollback. No se ejecutó un export científico real local.
Véase el [runbook y alcance de validación](runbooks/organization_evidence_export.md).

Regresión general sintética/offline: **838 passed**, con 40 FutureWarning de
pandas en código preexistente de clustering/association. Ruff global, formato,
ayuda CLI, siete notebooks fuente y checks de diff aprobados. Esta evidencia
valida software, no las membresías reales del experimento final.
Revisión final tras precisar diferencias de bytes de labels, sin afirmar
conflictos semánticos: **34 passed** en el bloque de organización.

## Reporte final del detector (2026-10-03)

**Infraestructura de reporte implementada; ejecución científica en Hypatia
pendiente de revalidación.** El responsable reporta 48/48 runs científicos
completos en Hypatia y ningún entrenamiento pendiente. El clon local conserva
un plan anterior y pilotos pequeños, sin esa matriz final; no se sustituye su
freeze ni se publican resultados científicos ficticios.

`detection report` verifica una matriz completa, agrega detector seeds antes de
splits/asociaciones y produce reporte, tablas, seis figuras y contrato
`detection-export-v1` para `flir-pipeline-explorer`. El estado COMPLETE solo se
publica al pasar el gate de fuentes del entorno de ejecución. Los estados
históricos del detector que aparecen debajo describen la evidencia local previa.
Véase el [runbook](runbooks/detector_final_report.md).

Validación local sintética/offline: **804 passed** en la suite completa;
**52 passed** en el bloque detector y **24 passed** en la revisión final de los
tests nuevos. Ruff global, formato de los cuatro archivos nuevos Python, siete
notebooks fuente, ayuda CLI y chequeos de diff pasaron. El schema se contrastó
con JSON Schema Draft 2020-12. Se usó un entorno de revisión compatible con las
restricciones DLL de Windows y una ruta temporal corta para la suite; versiones
y comandos están en el runbook. No hay type checker adicional configurado.
Estas pruebas no verifican los 48 runs reales de Hypatia ni publican su export.

## Ingesta nativa de evidencia legacy (2026-09-29)

**Adaptador implementado; ejecución real pendiente.** Los esquemas aportados por
el responsable ya tienen importación nativa mediante `hypatia_legacy_evidence_v1`.
Se comprueban las 46 columnas/tipos y nulos admitidos, las 910 ocurrencias/712
índices, la distribución 870/40, 13 cores candidatos/12 zonas, 78 pares/11
candidatos y la consistencia entre reports. Los 14 archivos consumidos quedan
ligados por SHA256 con snapshots sin cambios; verificar exige originales intactos
y replay de la normalización. No se inventan reviewer, fecha ni ground truth.

La revisión legacy conserva su `review_mode`; los cores siguen candidatos, las
zonas no son cortes exactos y el linaje no se convierte en byte identity. La
variante queda `unspecified`. No hay instancias, VDGs, splits ni asignaciones de
procedencia nuevas. El [runbook nativo](runbooks/native_sequence_evidence.md)
incluye inspección sin escritura, dry-run con fuentes, importación y verify en
SLURM. La evidencia local procede de fixtures sintéticas, no de una importación
ejecutada sobre los archivos reales de Hypatia.

Validación de cierre: `uv run --no-sync python -m pytest -q`, con Hugging Face
y Transformers en modo offline y un hilo para OMP/Numba: **773 passed**
(17 min 00 s), incluidas **33 pruebas de ingesta nativa**. Ruff global, formato
de los 26 archivos nuevos/de experimentos, siete notebooks fuente, sintaxis de
cuatro scripts Bash, ayudas CLI y `git diff --check` pasaron. La regresión cubre
también la infraestructura de variantes y la operación por etapas. No se
sincronizó código remoto, no se enviaron jobs reales y no hubo commit ni push.

## Variantes de dataset (2026-09-29)

**Infraestructura implementada.** Registro genérico con dataset/variante/origen,
checksums y definición; stores y checkpoints separados por variante; selección
explícita en CLI, resolución de features y SLURM. `compare-variants` consume
suites completas e inmutables, conserva ambas poblaciones y contrasta métricas
disponibles, zonas, clustering, recurrencia, estabilidad y shortlists. El pairing
opcional valida ocurrencias, método, confianza/evidencia y flag externo de verdad,
con cobertura y exclusiones explícitas; no afirma identidad de bytes.

El [ejemplo futuro](runbooks/sequence_operations.md#variantes-de-dataset-y-comparación-futura)
documenta `original_with_hud` y `no_hud` bajo el mismo protocolo. **No se evaluó
el dataset real sin HUD**; no hay conclusiones sobre su efecto ni un ganador
automático. La comparación no crea secuencias, VDGs ni splits. Los stores antiguos
sin declaración permanecen `unspecified`, sin modificación ni relabeling.

Validación focalizada: `uv run --no-sync python -m pytest
tests/test_dataset_variants.py tests/test_sequence_operations.py
tests/test_features.py -q`: **39 passed** (3 min 19 s), incluyendo las 18 pruebas
de variantes. Ruff global, formato de los 23 archivos nuevos/de experimentos y
help de registro, pairing, comparación, suite y extracción pasaron.
Regresión completa posterior: `uv run --no-sync python -m pytest -q`:
**740 passed** (15 min 13 s). `git diff --check` pasó. No hubo extracción de
modelos reales, experimentos con datos sin HUD, jobs remotos, commit ni push.

## Actualización: experimentos de secuencias (2026-09-28)

**Available; validación local sintética/offline.** `sequences experiment` integra
el detector actual como control, ventanas multiescala configurables, baselines
locales robustos, grids CLIP/DINOv2 × original L2/PaCMAP/t-SNE ×
DBSCAN/OPTICS/HDBSCAN/Agglomerative experimental, evaluación posterior con máscaras
y cobertura, recurrencia directa y por clústeres, transiciones diagnósticas,
estabilidad y ablaciones. Incluye publicaciones inmutables, importación explícita
de evidencia, paquetes visuales e historial manual. Véase el
[runbook](runbooks/sequences.md#suite-experimental-de-temporalidad-y-dependencia-visual).

La validación sintética incluye ambos reductores reales y los cuatro algoritmos.
Recibo local de implementación: `uv run --no-sync python -m pytest -q` completó
**722 passed** (13 min 27 s). Las **14 pruebas operacionales** también pasaron
en ejecución separada sobre los últimos controles. Ruff global, formato de los
archivos afectados, siete notebooks fuente y sintaxis de los cuatro scripts Bash
pasaron. No hay
un type checker configurado adicional. El launcher `.exe` de Windows quedó
bloqueado por Control de aplicaciones (4551); help y operaciones de la CLI se
validaron mediante su entrada Python y CliRunner, como permite el runbook.
No es un experimento FLIR completo ni valida dependencias visuales reales.
El perfil de Hypatia, su presupuesto y la calibración de umbrales están
**Pending compute / review**. Los resultados comunicados sobre video_11min y
video_13min no se codifican como constantes ni se presentan como revalidados.
**Operacionalización In progress.** Están implementados resolución de stores,
dry-run, DAG SLURM (82 jobs / 154 fits del perfil), recibos, status, plan
conservador de recuperación, paquete visual combinado y summary final. Las
pruebas comparan resultados seriales y por etapas, incluyendo reutilización de
coordenadas y recurrencia. No se ha enviado ningún job real desde este checkout.

**Bloqueo de esquema resuelto:** tras el rechazo de autenticación SSH, el
responsable entregó los contratos observados de las cuatro familias. El adaptador
nativo descrito arriba reemplaza la limitación previa a envelopes explícitos.
La ejecución y verificación con los archivos reales de Hypatia aún están
pendientes; no se presentan las fixtures como resultados reales. No se creó
ningún split ni se confirmaron automáticamente VDGs, fronteras o secuencias. Los
resultados históricos documentados debajo conservan su alcance anterior.

Documentación reorganizada el **2026-09-23** desde el código y la evidencia local
existente, sin ejecutar experimentos ni cambiar resultados. **Validated** indica
comprobación dentro del protocolo, no validación semántica exhaustiva ni mejora
del detector. **Available** indica infraestructura utilizable; **Experimental**,
incertidumbre interpretativa; **Pending compute**, una ejecución completa pendiente;
**Planned**, trabajo adicional condicionado.

| Componente | Estado | Evidencia y límites |
|---|---|---|
| Auditoría y manifest | Validated | 1657 frame_id, 1459 content_id, 198 grupos duplicados; ocho conflictos de anotación preservados |
| Muestreo de videos fuente | Operationally validated on real data | Smoke real de 5 s y job Hypatia 737719 COMPLETED, ExitCode 0:0; 3 videos fuente → 9648 JPEGs a 1 FPS, Parquet/grilla validados y fuentes read-only; no identifica secuencias ni define splits |
| Puente de frames muestreados a features | Full validation reported by owner on Hypatia; not reverified locally | 9648 ocurrencias / 8093 contenidos; DINOv2 8093 × 384 y CLIP 8093 × 512; esta etapa no genera secuencias ni splits |
| Similitud de video v2 y contrato downstream | Available; synthetic local validation | Procedencia por grilla, mínimos sobre todas las occurrences, summaries/cuántiles exactos, pares opcionales por bloques; reducción/clustering numéricos compatibles, ejecución real pendiente |
| Secuencias de video y dependencias exactas | Available; synthetic local validation; real sequences reported by owner | `sequences detect/build/verify/summary`; cortes multiescala sobre originales L2, revisión confirmada obligatoria, cobertura de occurrences y componentes exactos; publicación real no revalidada localmente, sin split ni ground truth |
| Linkage etiquetado → video | Available; synthetic local validation | `linkage build/verify/summary`; unión top-k CLIP/DINOv2 entre datasets, scores separados, todas las ocurrencias/secuencias y anotaciones preservadas; compatibilidad de consumo real reportada por la revisión manual, sin afirmar relevancia visual global |
| Calibración manual etiquetado → grupo visual | Available; real init/verify compatibility reported on Hypatia | Contrato confirmado v1 ejercitado: init exitoso y verify quality_valid/source_bound=true, todas las ocurrencias preservadas y una decisión por consulta. Sin ground truth, enlaces confirmados ni split; no implica completar todos los estratos |
| Agregación de revisiones manuales | Available; synthetic local validation | `linkage review aggregate/aggregate-verify`; revisiones inmutables revalidadas, duplicados compatibles contados una vez y conflictos rechazados, conteos descriptivos por revisión/estrato/grupo; ejecución real de agregación pendiente |
| Caracterización y diagnostics | Validated | 4168 instancias canónicas, 292 labels vacíos, geometría por clase y 1459 diagnostics; huérfanos separados |
| DINOv2 | Validated | Full 1459 × 384 CLS; raw/L2, revisión resuelta y 1657 mappings |
| CLIP | Validated | Full 1459 × 512 projected-image; raw/L2, revisión resuelta y 1657 mappings |
| Coseno | Validated | Por encoder: 1459 × 1459, 1063611 pares únicos y 29180 vecinos dirigidos top-20 |
| Interpretación temporal | Experimental | Secuencia/índice inferidos por nombre; cero timestamps verificados |
| t-SNE / PaCMAP | Validated with limitations | 18 + 18 runs, preservación y tres semillas; cuatro referencias exploratorias |
| DBSCAN / OPTICS / HDBSCAN | Validated with limitations | 132 / 186 / 96 runs; 204 ARI/AMI, shortlist de 54, 41 Pareto; escenas sin validación exhaustiva |
| Splitting y baselines | Validated with limitations | 66 runs; 65 nuevos sin overlap exacto; 60 cluster-aware sin fracturas; seis Pareto |
| Streamlit / VIKUS | Available | Inspección local; no valida escenas automáticamente |
| Detector YOLO11n | Available; small pilots validated | 16 vistas y cuatro pilotos CPU; no comparación científica final |
| Detector ↔ residual similarity | Available; PENDING | Registro de siete asociaciones, figura 09 y explorador local; 0/48 runs controlados, pilotos excluidos |
| Comparación completa del detector | Pending compute | Stage A/B y 48 runs objetivo aún no ejecutados |
| Bhattacharyya | Planned / conditional | Requiere una representación distribucional explícita |

## Data and representations

El histórico conserva 1178/107/372 registros train/val/test y 57/141/0 contenidos
compartidos train–val/train–test/val–test. El archivo de etiquetas contiene 4182
objetos: 4168 del candidato y 14 de diez etiquetas huérfanas. La referencia
bibliográfica exacta de la nomenclatura sigue pendiente; el orden fue confirmado
por el responsable del proyecto y contrastado con el YAML original.

Las extracciones completas ya existían desde 2026-09-09. Los smoke N=16 se
conservan separados y no caracterizan globalmente el espacio de embeddings.
El inventario registrado conserva cinco ZIP disponibles de seis históricos;
falta `video_13min_778.zip`, sin pérdida de cobertura del candidato canónico.
Véanse [auditoría](analysis/data_quality.md), [clases](analysis/dataset_classes.md)
y [representaciones](analysis/features.md).

## Source-video sampling: real execution confirmed

La evidencia confirmada por el responsable del proyecto registra el job SLURM
**737719** en **Hypatia, Universidad de los Andes**: `COMPLETED`, `ExitCode=0:0`,
`Elapsed=00:15:28`, `MaxRSS` del batch `175016K` y stderr vacío. Se usaron Python
3.11.10, FFmpeg/ffprobe 9.0.2, `sample_fps=1.0` y `jpeg_quality=2`.

Los tres videos produjeron **2770 + 660 + 6218 = 9648 JPEGs**. Se validaron 9648
filas y 9648 `image_path` únicos en `frames.parquet`, índices contiguos desde 0
por video, tiempos coherentes con `sample_index / 1 FPS` y fuentes read-only.
También se validó previamente un smoke real: 5 segundos a 30 FPS, 150 frames
fuente → 5 JPEGs a 1 FPS. El [runbook](runbooks/data.md#evidencia-real-confirmada-en-hypatia)
conserva configuración y resultados agregados; los recibos exactos permanecen locales.

Esta evidencia valida operacionalmente la extracción reproducible de frames.
Los tres videos son **fuentes, no tres secuencias**. No valida límites de escenas,
embeddings de estas muestras, clustering, train/val/test, eliminación de leakage
ni rendimiento del detector. Posteriormente, el responsable reportó validación
del manifest y features completos en Hypatia: 9648 occurrences, 8093 content_id,
1555 contenidos duplicados, 3110 registros participantes, 1555 redundantes y cero
fallos de decode. Por video, los conteos reportados son 660/2770/6218. DINOv2
8093 × 384 y CLIP 8093 × 512 conservan los 9648 mappings; el responsable reporta
cobertura completa, metadata coincidente y revisiones resueltas válidas en ambos.
Esta es evidencia comunicada por el responsable, no una revalidación local de
esos artefactos. No se versionan sus manifests, hashes ni outputs privados.

La adaptación de similitud v2 se valida localmente con datos sintéticos: ninguna
ejecución de similitud, reducción o clustering sobre los 8093 contenidos reales
se realizó aquí. Esta ruta no infiere secuencias, no genera labels ni
splits, y memoria/tiempo completos deben medirse posteriormente en Hypatia. Los
resultados históricos de las demás etapas no cambian.

## Cross-dataset linkage: implementation and evidence boundary

La etapa `linkage` consume los dos manifests, cuatro stores originales de features
y un set de secuencias revisadas. El responsable reporta que las ocurrencias de
video ya cuentan con secuencias en Hypatia; esta actualización no inspecciona ni
revalida ese artefacto real. La evidencia local de la etapa nueva consiste en
fixtures sintéticos y verificación de sus invariantes, incluyendo contenido
repetido en varias secuencias y conflictos de anotación históricos.

Se conservan separados los cosenos de CLIP y DINOv2; solo los ranks alimentan el
consenso opcional. `ground_truth=false`, sin confirmación automática, asignación
de secuencia al contenido etiquetado, grupos visuales ni nuevos splits. Los
recursos completos y la relevancia visual global no se establecen mediante el
éxito de consumo del artefacto. Véase el [runbook](runbooks/linkage.md).

La calibración manual consume grupos visuales confirmados reportados por el
responsable; no reconstruye ni confirma esos grupos. El esquema confirmado v1
inspeccionado en Hypatia está soportado explícitamente, sin exigirle ID ni
checksums de productor. La calibración liga sus tres archivos mediante SHA256
e identidad del consumidor. Se conservó también el adaptador normalizado y se
probaron ambos con datos/imágenes sintéticos. El responsable reportó ejecución
real exitosa del adaptador v1 en Hypatia: `review init` completó y `review verify`
devolvió `quality_valid=true`, `source_bound=true`,
`all_candidate_occurrences_preserved=true` y `one_decision_per_query=true`,
con `ground_truth=false`, `confirmed_matches_created=false` y `split_created=false`.
Este resultado se atribuye al responsable y no se revalidó localmente. No implica
que todos los estratos futuros estén revisados ni una evaluación representativa.
`supported/ambiguous/unsupported` evalúan evidencia
de enlace al grupo, sin identificar un frame o una secuencia exacta. Las tasas
por estrato/grupo no son accuracy representativa. El nuevo flujo no crea ningún
split leakage-safe. Véase el [contrato y flujo manual](runbooks/linkage_review.md).

## Clustering memory and sampled-video review infrastructure

La ruta independiente `clustering video-review` está disponible para runs y
comparisons de `flir_video_samples_v1`, con validación sintética local. Conserva
todas las ocurrencias, verifica JPEGs locales y produce galerías, pertenencias y
líneas de tiempo por video fuente con secuencias desconocidas. No modifica el
reporte histórico ni crea escenas/splits. La revisión de los videos reales sigue
pendiente. Véase el [runbook](runbooks/clustering.md#revisión-independiente-de-videos-muestreados).

La refactorización conserva los resultados sintéticos de referencia de `737e588`:
distancias exactas bajo demanda, hasta dos matrices retenidas y k-distance por
bloques. Es validación de infraestructura; el pico RSS y el clustering completo
de N=8093 continúan pendientes. Véase la [auditoría de memoria](analysis/clustering_memory.md).

## Candidate partitions

**C10 — DINOv2 / PaCMAP / DBSCAN** es la referencia visual principal:
seed 0, split `88ccf4e12335a83f`, 1178/107/372 registros, cinco clases por split,
cero duplicados exactos cross-split y 6/7 pares top-0.1% en DINOv2/CLIP.
El histórico tiene 328/277 y las medias random 478/466.8. Su fracción temporal
inferida Δ≤5 es 23.93%, frente a 14.76% histórica y 43.92% random.

**C12 — DINOv2 / t-SNE / HDBSCAN** es el candidato secundario del protocolo del
detector, con menor similitud NN media y temporalidad residual, a costa del
balance. **C01 — CLIP / original L2 / OPTICS** permanece como referencia
descriptiva de balance, con 97.4% noise. Las anclas de splitting (C10/C01) y la
selección previa al detector (C10/C12) corresponden a decisiones distintas;
ninguna constituye un ganador universal.

Las 65 asignaciones nuevas fueron reconstruidas exactamente en la validación
registrada. El MILP balancea clases y random no: ese contraste limita atribuir
un cambio posterior exclusivamente al agrupamiento. Resultados completos en
[clustering](analysis/clustering.md), [splitting](analysis/splitting.md) y
[selección del detector](analysis/detection.md).

## Detector execution boundary

Historical, random_content, **C10 — DINOv2 / PaCMAP / DBSCAN** y
**C12 — DINOv2 / t-SNE / HDBSCAN** se seleccionaron antes de YOLO.
Plan `ace1ffd6bb4775d0`: 16 splits, 48 celdas objetivo, detector seeds 42/43/44
y split seeds 0–4. Runtime CPU congelado: `9897062efdb32450`.

Los cuatro pilotos usan 24/12/20 imágenes, dos epochs, batch 2, imgsz 640 y CPU.
Se revalidaron métricas y 1000 bootstrap por piloto. Training total 140.39 s;
evaluación/bootstrap 57.46 s. La extrapolación registrada de 408–593 horas para
48 runs usa dos métodos, no un intervalo de confianza. El hardware registrado
no expone CUDA; el protocolo largo se detuvo por presupuesto.

Quedan pendientes Stage A completo a 50 epochs, Stage B, comparación multi-seed,
variabilidad e interpretación científica. No hay ejecución GPU ni prueba real de
resume interrumpido en GPU. El HTML conserva una figura de contexto y ocho paneles
pendientes. Véanse [análisis](analysis/detection.md), [protocolo](protocols/detection.md)
y [runbook](runbooks/detection.md).

## Reports and next validation

Los siete notebooks fuente siguen activos, sin outputs versionados. El
[reporte consolidado](runbooks/project_report.md) tiene 22 secciones, 14 figuras
(12 reutilizadas y dos resúmenes temporales) y un diagrama. Los recibos previos
respaldan las validaciones numéricas; una interfaz disponible no cambia el estado
experimental.

El siguiente trabajo metodológico es revisar escenas y conflictos de candidatos,
mejorar la procedencia temporal y ejecutar el protocolo controlado del detector
con presupuesto viable. [Streamlit](visualization/streamlit.md) y
[VIKUS](visualization/vikus.md) apoyan esa revisión. La coherencia visual global,
la eliminación de toda dependencia y la mejora en Precision/Recall/mAP siguen
sin demostrarse.

## Secuencias revisadas: infraestructura disponible, ejecución real pendiente

La etapa independiente `sequences` implementa detección/localización multiescala,
consumo explícito de revisión manual confirmada, sequence instances y dependencias
por contenido exacto, conservando todas las ocurrencias. La revisión real existe
solo en Hypatia según el responsable; no se copió ni se ejecutó aquí. Sus 13
aceptaciones sobre tres fuentes implican una expectativa de 16 instancias que
debe comprobarse allí junto con todos los invariantes, no un resultado observado
localmente. La evidencia local es sintética/offline.

Clustering sigue siendo exploratorio y no suministra fronteras. La revisión
permanece ground_truth=false. Esta etapa no crea splits ni modifica las métricas
históricas o los componentes grupales. Próximo paso: ejecutar detect/build/verify
con las fuentes y revisión congeladas, medir recursos y conservar el recibo local.
El [runbook de secuencias](runbooks/sequences.md) especifica contrato, comandos,
identidades y límites de la procedencia heredada.
