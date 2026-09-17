# Reporte acumulativo de avance

El [notebook fuente](../notebooks/progress_review.ipynb) reúne el avance académico
en 22 secciones. La narrativa está en español; cada sección comienza con una
pregunta y termina con un hallazgo. Las etiquetas OBSERVADO, VALIDADO, LIMITACIÓN
y PENDIENTE separan resultados, validaciones existentes y trabajo futuro.

## Construcción y alcance

Desde la raíz del repositorio, con los artefactos locales de las etapas ya
disponibles:

```powershell
uv run python scripts/build_progress_review.py --check
uv run --extra reporting python scripts/build_progress_review.py
```

`--check` comprueba fuentes y selecciones sin escribir outputs. La construcción
ejecuta únicamente las celdas de lectura/presentación del notebook. No carga
modelos, embeddings, matrices de similitud, coordenadas ni ZIPs originales.
No calcula métricas experimentales, genera particiones ni ejecuta YOLO.
Las tablas se seleccionan y formatean desde resultados registrados; los conteos
del manifest se contrastan con sus resúmenes. Las conversiones de proporciones
a porcentaje y de horas a días son exclusivamente de presentación.

Outputs locales, ignorados por Git:

- `reports/progress/review/progress_review.html`: informe con código oculto y
  figuras embebidas, sin depender de los otros HTML para leerlo.
- `reports/progress/review/progress_review.executed.ipynb`: ejecución local.
- `reports/progress/review/build_receipt.json`: inventario de fuentes con hashes,
  estado por sección, comprobaciones y checksums de salida; no se publica.

Una fuente ausente se marca `missing`; si cambia frente al checksum del recibo
anterior o sus resúmenes no concuerdan, se marca `invalid`. Las secciones afectadas
no publican tablas ni hallazgos numéricos y el builder termina con código 1.
Nunca completa huecos con cifras recordadas ni recalcula la etapa ausente.
Un cambio del estado final del detector requiere revisar la narrativa.

## Mapa de evidencia por sección

Las rutas siguientes son relativas al repositorio. Cada sección del HTML incluye
su lista de fuentes desplegable; el recibo local vincula el archivo exacto leído.

| Secciones | Fuentes existentes principales | Qué respaldan |
|---|---|---|
| 1–2 · Problema y pipeline | Contexto metodológico; `reports/detection/report_metadata.json` | Alcance individual y comparación final pendiente |
| 3 · Dataset | `data/manifests/flir_canonical_candidate_v1.parquet`; `reports/feature_engineering/tables/{dataset_summary,historical_split_summary,duplicate_summary}.csv` | Registros, contenidos, splits históricos y overlaps exactos |
| 4 · Clases | `reports/feature_engineering/tables/{class_catalog,object_instances_by_class,annotation_quality}.csv` | Nombres, instancias, vacíos y huérfanos excluidos |
| 5 · Bounding boxes | `reports/feature_engineering/tables/bbox_geometry_by_class.csv`; `reports/feature_engineering/metadata.json` | Medianas existentes y definición de geometría YOLO normalizada |
| 6 · Features | `reports/feature_engineering/tables/embedding_health_{dinov2,clip}.csv`; `reports/feature_engineering_closure/final_audit.json`; manifest | Cobertura completa, dimensiones, pooling y validación previa |
| 7 · Coseno | `reports/similarity/tables/{feature_spaces,global_similarity,topk_global_summary}.csv` | Pares únicos, media/mediana global y rank-1 |
| 8 · Visual/temporal | `reports/similarity/tables/{temporal_neighbors,sequence_similarity,frame_delta_similarity,temporal_coverage}.csv` | Misma secuencia, medianas por relación y Δ; timestamps no verificados |
| 9 · Acuerdo | `reports/similarity/tables/neighbor_agreement_summary.csv` | Acuerdo rank-1 y Jaccard@1/5/10/20 |
| 10 · Reducción | `reports/reduction/tables/{runs,candidate_reference_metrics,input_spaces}.csv` | Grid, parámetros, seeds y métricas de las cuatro referencias |
| 11–12 · Clustering | `reports/clustering/summary.json`; `reports/clustering/tables/{all_runs,candidates,references}.csv` | Grid, Pareto, ARI/AMI registrados y ejemplo R6 |
| 13–14 · Splitting y C10 | `reports/splitting/summary.json`; `reports/splitting/tables/{runs,clustering_candidates}.csv`; `docs/splitting_protocol.md` | Invariantes, noise singleton, MILP, tamaños y balance de C10 seed 0 |
| 15 · Histórico/random/C10 | `reports/splitting/tables/{runs,metric_variation}.csv` | Histórico y C10 seed 0 frente a medias random seeds 0–4 ya calculadas |
| 16 · C12 | Las dos tablas anteriores; `artifacts/detection/protocol/plan.json`; `reports/detection/tables/split_context.csv` | Compromiso entre NN medio, temporalidad, balance y pares extremos; selección secundaria |
| 17 · Detector | `reports/detection/{report_metadata,verification_receipt}.json`; `reports/detection/tables/pilot_validation.csv`; `artifacts/detection/{environment.json,protocol/plan.json}`; `docs/detector_comparison_analysis.md` | Hardware registrado, configuración de pilotos y ausencia de resultados finales |
| 18 · Cómputo | `artifacts/detection/compute_budget.json`; plan y metadata del detector | Dos extrapolaciones existentes para la matriz final; no intervalo de confianza |
| 19–21 · Estado, resultados, límites | Fuentes anteriores; `docs/current_status.md` | Estado derivado de evidencia, conflictos de anotación e incumbentes MILP |
| 22 · Próximos pasos | `artifacts/detection/protocol/plan.json` | Plan pendiente, sin ejecutarlo |

Los `report_metadata.json` de similitud, reducción, clustering, splitting y
detector comprueban los hashes de los archivos reutilizados cuando hay un
checksum registrado. La caracterización inicial no tiene ese mismo inventario
de checksums: se registra su snapshot actual y se comprueba consistencia con
manifest y recibos existentes. Esto no es una nueva verificación numérica de
embeddings, reducción, clustering o asignaciones.

## Figuras seleccionadas

Catorce PNG se reutilizan byte a byte; se incorpora además un diagrama HTML
sencillo del pipeline (15 figuras en total). No se regeneran gráficos de etapas.

| Sección | Carpeta bajo `reports/` | Figura existente |
|---|---|---|
| 3 | `feature_engineering/figures/` | `01_dataset_overview.png`; `13_cross_split_duplicate_matrix.png` |
| 4 | `feature_engineering/figures/` | `15_class_instances.png` |
| 5 | `feature_engineering/figures/` | `06_bbox_normalized_area_by_class.png`; `17_bbox_aspect_ratio_by_class.png` |
| 7 | `similarity/figures/` | `03_rank1_similarity_comparison.png` |
| 8 | `similarity/figures/` | `05_similarity_vs_frame_delta_dinov2.png`; `06_similarity_vs_frame_delta_clip.png` |
| 10 | `reduction/figures/` | `05_tsne_dinov2_reference.png`; `03_pacmap_dinov2_reference.png` |
| 12 | `clustering/figures/` | `R6_dinov2_tsne_clusters.png` |
| 15 | `splitting/figures/` | `06_high_similarity_cross_split_pairs.png`; `04_cross_split_nn_similarity_dinov2.png`; `07_temporal_cross_split_rates.png` |

Las figuras de splitting conservan sus leyendas originales, incluido C01 como
referencia de balance y variación entre semillas. La tabla principal distingue
histórico, media random de cinco seeds y C10 seed 0. R6 es un ejemplo de clustering,
no el clúster fuente de C10. No se muestran paneles de métricas de pilotos YOLO.

## Validación y límites de la revisión

```powershell
uv run ruff check .
uv run pytest
uv run python scripts/check_notebook_source.py
```

Los tests nuevos son sintéticos y offline: faltantes, corrupción de fuentes,
smoke frente a cobertura completa, promedios frente a representantes, cambios
del estado del detector, privacidad y notebook limpio. El builder comprueba
22 secciones, 15 figuras en una construcción completa, ausencia de código visible,
rutas privadas e identificadores reales del manifest en el texto/atributos
visibles. Comprueba que las fuentes leídas no cambien durante la construcción.
Los hashes de imágenes no se publican en el HTML ni en el notebook fuente.

La inspección visual de los PNG complementa el control textual, porque los textos
dentro de imágenes raster no se auditan con el parser HTML. La inspección del
layout interactivo depende de disponer de un navegador; no se confunde una
auditoría estructural con una revisión visual completa del HTML.

Esta consolidación no cambia el estado científico: validación temporal parcial,
clustering y splitting con límites, comparación controlada del detector pendiente.
