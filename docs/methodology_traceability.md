# Trazabilidad con los cinco objetivos definitivos

Cierre OE4 2026-11-22: recibidos CSV/recibo agregados del test común; SHA validado,
tabla final reproducida sin alterar valores. Se acredita la publicación de esos
resultados recibidos para seis configuraciones (una semilla), no una nueva
auditoría local de los seis checkpoints. Recuperados los seis recibos de evaluación
y comprobadas sus identidades/metadatos; pesos, recibos de entrenamiento/freeze
y payloads estadísticos no se rehasharon. [Definiciones, procedencia y límites](results/oe4_20261122/README.md).

Recuperación 2026-11-22: conteos de etiquetas de la vista nativa, sin exigirlos
al manifest fuente. Denoised avanza solo como exploración separada con grupos de
riesgo nominal excluidos y test común prohibido; no se acredita efecto causal
del denoising ni independencia de sus correspondencias no resueltas.
[Evidencia y ejecución](runbooks/oe4_recovery_20261122.md).

Hotfix OE4 2026-11-22: se registra la autorización de JP comunicada por el
responsable para experimentos/evaluación; no se afirma verificación documental
ni criptográfica. El cambio administrativo no modifica la comparación AUG_ON /
AUG_OFF, los datos, la semilla, las 50 épocas ni la liberación de test después
de completar los tres entrenamientos. [Contrato v2](runbooks/oe4_hotfix_20261122.md).

Actualización local 2026-11-21: el [cierre denoised/OE4](runbooks/oe4_closeout_20261120.md)
añade evidencia de ingesta de la variante externa (OE5), grafo/MILP candidato (OE3)
y preparación verificable de test común y ablación AUG_OFF (OE4). Se ejecutaron
ambos encoders completos y el MILP denoised: candidato 777/172/167 con 93 grupos
indivisibles y cero cruces registrados. No acredita freeze científico,
independencia temporal absoluta, óptimo del solver ni ejecución detector nueva.
El algoritmo C10 original permanece separado de las sensibilidades de la variante.

Referencia: `ppg_j.solorzanod_202620.pdf`, objetivos en su página impresa 2.
El expediente privado conserva archivo, SHA y objetivos textuales. La tabla
separa infraestructura y evidencia; no sustituye la revisión académica de JP.

| Objetivo | Módulos/configuración | Artefacto → experimento → resultado → informe | Estado y brecha |
|---|---|---|---|
| OE1. Caracterizar correlación espaciotemporal | `features/`, `similarity/`; `configs/embeddings/`, `configs/similarity/` | Stores raw/L2 e índices → extracción completa y coseno → QA y caracterización → análisis features/similarity | Visual ejecutado; temporalidad canónica limitada. Bhattacharyya no ejecutada: metodología la condiciona a distribuciones definidas; justificación y límite deben quedar explícitos. |
| OE2. Comparar reducción y clustering de densidad | `reduction/`, `clustering/`; configs t-SNE/PaCMAP/DBSCAN/OPTICS/HDBSCAN | 36 reducciones y 414 runs → shortlist/estabilidad → ARI/AMI, ruido y coherencia con cobertura → análisis clustering | Comparación técnica acreditada; no hay ground truth exhaustivo de escenas ni continuidad temporal. No requiere SIFT/RANSAC. |
| OE3. Particionar por clústeres indivisibles y cobertura | `splitting/`; `cluster_aware_research.yaml` | source_groups y asignaciones → 60 splits cluster-aware → atomicidad/cobertura/solver → análisis splitting | Implementación y salidas verificadas. C10 documentado; nueva admisión bajo test fijo actual pendiente. No confundir B-S2/S3 con C10. |
| OE4. Comparar histórica, aleatoria y clústeres con coseno y detector | `splitting/metrics.py`, `detection/`, `closeout/`; protocolos separados | Comparación residual; contrato histórico de 48 corridas; B de 75 YOLO11m separado; OE4 v2: CSV/recibo y seis recibos nativos → tabla final y procedencia publicable | SHA del CSV y metadatos/identidades de seis recibos de evaluación comprobados. Comparación descriptiva seed 42, sin causalidad/independencia; bytes de checkpoints/payloads y recibos de entrenamiento/freeze no verificados localmente. B y contrato histórico permanecen separados. |
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
