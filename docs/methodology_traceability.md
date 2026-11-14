# Trazabilidad con los cinco objetivos definitivos

Referencia: `ppg_j.solorzanod_202620.pdf`, objetivos en su página impresa 2.
El expediente privado conserva archivo, SHA y objetivos textuales. La tabla
separa infraestructura y evidencia; no sustituye la revisión académica de JP.

| Objetivo | Módulos/configuración | Artefacto → experimento → resultado → informe | Estado y brecha |
|---|---|---|---|
| OE1. Caracterizar correlación espaciotemporal | `features/`, `similarity/`; `configs/embeddings/`, `configs/similarity/` | Stores raw/L2 e índices → extracción completa y coseno → QA y caracterización → análisis features/similarity | Visual ejecutado; temporalidad canónica limitada. Bhattacharyya no ejecutada: metodología la condiciona a distribuciones definidas; justificación y límite deben quedar explícitos. |
| OE2. Comparar reducción y clustering de densidad | `reduction/`, `clustering/`; configs t-SNE/PaCMAP/DBSCAN/OPTICS/HDBSCAN | 36 reducciones y 414 runs → shortlist/estabilidad → ARI/AMI, ruido y coherencia con cobertura → análisis clustering | Comparación técnica acreditada; no hay ground truth exhaustivo de escenas ni continuidad temporal. No requiere SIFT/RANSAC. |
| OE3. Particionar por clústeres indivisibles y cobertura | `splitting/`; `cluster_aware_research.yaml` | source_groups y asignaciones → 60 splits cluster-aware → atomicidad/cobertura/solver → análisis splitting | Implementación y salidas verificadas. C10 documentado; nueva admisión bajo test fijo actual pendiente. No confundir B-S2/S3 con C10. |
| OE4. Comparar histórica, aleatoria y clústeres con coseno y detector | `splitting/metrics.py`, `detection/`; protocolos detector separados | Comparación residual retrospectiva; contrato histórico de 48 corridas; B de 75 YOLO11m por separado → reportes/exportes → métricas con identidad y limitaciones | El contrato histórico declara ejecución completa; recuperación de recibos nativos se distingue de validar sus bytes exportados. B no mide directamente el efecto de C10. El contraste actual YOLO11m de clústeres con test común no se da por completado. |
| OE5. Ensamblar limpieza, partición y panóptica | Ingesta de variantes y fuentes de imagen; contratos externos; `apps/explorer` como consumidor | Fuente externa versionada → identidad padre/hijo, geometría, catálogo y split → consumidor de Manuela → recibo integrado | Interfaces parciales; falta acreditar recorrido conjunto sobre una población admitida. Reutilizar Laura/Manuela sin atribuir sus algoritmos al desarrollo individual. |

Los módulos y comandos de reproducción existentes están en
[pipeline_traceability](pipeline_traceability.md), [reproducibilidad](reproducibility.md)
y [runbooks](runbooks/). Las rutas reales, hashes de datos, 75 identidades,
checkpoints y recibos se conservan en el índice privado externo, no en Git.

Para cualquier resultado, el índice externo relaciona objetivo → módulo/commit
→ configuración/checksum → artefacto/identidad → experimento/semilla/checkpoint
→ métrica/alcance → informe. Ausencias permanecen pendientes, no cero. Los
análisis de variantes, los splits transductivos antiguos y las evaluaciones
comunes actuales mantienen dominios diferentes.
