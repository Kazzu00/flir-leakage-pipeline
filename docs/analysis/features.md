# Visual representations: recorded extraction evidence

Full extraction and smoke evidence remain distinct. These are preserved results,
not experiments executed during documentation maintenance.

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
El commit posterior versiona las configuraciones, auditorías y reporte.
Ver [ejecución reproducible](../runbooks/features.md).


## Evidencia y unidades

Cada espacio completo conserva seis archivos: `embeddings_raw.npy`,
`embeddings_l2.npy`, `content_index.parquet`, `record_index.parquet`,
`metadata.json`, `feature_quality.json`. Hay 1459 vectores por encoder,
representados en dos arrays raw/L2, y 1657 mappings históricos.
Raw y L2 no se cuentan como extracciones independientes.

Los nuevos smoke contienen 16 vectores por encoder y permanecen en otra raíz.
No se alteraron los smoke antiguos ni se les asignó una revisión retrospectiva.

Metadata registra modelo, SHA HF efectivo, processor, pooling, dimensiones,
dtypes, semilla, IDs, versiones, commit y fecha. El commit capturado durante la
extracción es el HEAD previo al commit de publicación; los YAML nuevos estaban aún
sin versionar. El recibo local conserva el SHA256 de cada YAML y el comando.
Los encoders y el flujo de extracción no cambiaron respecto al backend ya revisado.
Esto describe el estado real de ejecución, sin atribuirlo a un commit posterior.

Artefactos principales:

- `reports/feature_engineering/review/feature_engineering_review.html`: reporte sin celdas de código.
- `reports/feature_engineering/review/feature_engineering_review.executed.ipynb`: notebook ejecutado con código.
- `reports/feature_engineering/feature_engineering_report.md` y `metadata.json`.
- `tables/annotation_class_distribution.csv`, `annotation_quality.csv`, `empty_annotations.csv`.
- `tables/class_catalog.csv`, `object_instances_by_class.csv`, `bbox_instances.csv`, `bbox_geometry_by_class.csv`.
- `tables/representation_comparison.csv`: modelo, N, dimensión, pooling y L2, sin ranking de encoders.
- `figures/06_bbox_normalized_area_by_class.png`, `figures/17_bbox_aspect_ratio_by_class.png`.
- `tables/temporal_summary.csv`, `temporal_coverage.csv`, `temporal_lineage.csv`, `cross_split_temporal_candidates.csv`.
- `tables/historical_baseline.csv`, `historical_split_summary.csv`, `image_geometry_summary.csv`, quality tables.
- `reports/feature_engineering_closure/full_extraction_receipt.json`, logs y auditoría final.

Las rutas de tables anteriores son relativas a `reports/feature_engineering/`.
Todas las salidas reales permanecen locales e ignoradas. El notebook fuente
versionado solo contiene narrativa y código.
