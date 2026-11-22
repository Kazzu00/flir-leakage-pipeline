# Contratos y evidencia

Fuentes ejecutables: `src/contracts/index.ts` para el sobre de UI y `src/contracts/leakage.ts` para artifacts M02. Los tipos se derivan de Zod. Cada adapter valida antes de exponer datos a React.

## Snapshot de UI

`SnapshotSchema` conserva `schemaVersion: 1`, módulos, dataset, registro de runs y colecciones demo. `origin` en dataset/run/métrica admite `mock`, `artifact`, `reported`, `unavailable`. Predicciones de M03 siguen siendo mock. En modo mixto `leakage` contiene el dominio artifact; las antiguas colecciones demo M02 quedan vacías y `similarity` es `null`. No se generan coordenadas o asignaciones fake para satisfacer el sobre. Una semilla desconocida es `null`, no cero.

La fixture DEMO conserva sus invariantes de unicidad, tamaños/medoids, ruido singleton, referencias y asignaciones por ocurrencia. Historical admite overlap explícito; Random/content conserva cada contenido; Cluster-aware mantiene íntegros los grupos.

## LeakageSnapshotV1

JSON con `schemaVersion: "LeakageSnapshotV1"`, `dataset`, `contents`, `runs`, `exportPolicy`. El exportador construye exclusivamente campos conocidos; Zod rechaza campos desconocidos del snapshot, runs y registros de procedencia.

| Campo                    | Semántica                                                                                                                          |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------- |
| dataset                  | Alias, versión `flir_video_samples_v1`, origen artifact, cobertura de ocurrencias/contenidos/duplicados                            |
| contents                 | Una entrada por contenido y todas sus ocurrencias, incluidas membresías multivideo                                                 |
| occurrence               | Alias frame/video, índice y segundos de grilla; secuencia y captura siempre `null`                                                 |
| run                      | Etapa/método tipados, encoder, semilla nullable, aliases dataset/feature/similarity/reduction/clustering/split y configuration     |
| contentIndex             | Orden completo `contentId → embeddingRow` de cada feature space; nunca vectores                                                    |
| coordinates              | `contentId, embeddingRow, x, y` de una reducción persistida; vacío si no se exportó ese run para visualización                     |
| pairs                    | Aristas top-k dirigidas con rank, coseno, relación de fuente y gaps mínimos reportados                                             |
| summaries                | Tablas reportadas por fuente, sample gap, grid gap, top-k, configuración/estabilidad; etiquetas fijas o aliases, nunca texto libre |
| labels                   | Un cluster por contenido, ruido -1 y medoid reportado                                                                              |
| assignments              | Cada frame y su contenido con train/validation/test; ausente si no existe split                                                    |
| metrics / parameters     | Valores numéricos finitos o null mediante lista permitida; sin paths ni metadata arbitraria                                        |
| integrity / metricOrigin | `export-validated` para controles de transporte; métricas normalmente `reported`                                                   |

### Invariantes

- Conteos y unicidad de contenidos/ocurrencias; todas las ocurrencias conservan el mapping.
- Mismo dataset en todas las etapas; referencias existentes, etapa y encoder compatibles; continuidad del feature space.
- DINOv2: 384 dimensiones, CLS; CLIP: 512, projected pooler output. Cobertura completa, no smoke tests mezclados con el dataset.
- Índice único y secuencial de features. Coordenadas finitas con cobertura completa y el mismo mapping de filas.
- Un método admitido por etapa; las coordenadas solo pertenecen a reducción, pares a similarity, etiquetas a clustering y asignaciones a splits.
- Clusters únicos por contenido; ruido explícito. Split con cobertura exacta de ocurrencias. Cluster-aware exige clustering compatible y no divide clústeres; ruido se trata como grupo singleton.
- Detector exige referencia de split. El exportador rechaza pilotos como resultados finales y exige controles declarados de evaluación.
- Source video nunca se convierte en sequenceId; timestamps de captura no se deducen de la grilla.
- Los errores del validador se convierten en códigos seguros, no se imprimen paths ni datos rechazados en el DOM.

Los aliases son locales al snapshot, no claves científicas globales. La continuidad original se valida en el exporter antes de sustituir IDs. Estos controles no recomputan métricas ni demuestran validez experimental. La evidencia científica y la configuración completa de reproducción permanecen en los artifacts originales.

La fixture `src/test/fixtures/leakage-synthetic.json` es completamente sintética y compartida con los tests Python. No es un snapshot de resultados reales. Los límites de tamaño, inputs nativos admitidos y campos no exportados se documentan en [REAL_DATA_INTEGRATION](REAL_DATA_INTEGRATION.md).

## LeakageSnapshotV2

`src/contracts/leakage-v2.ts` mantiene los campos V1 y todas sus invariantes, cambia únicamente el discriminador y añade `research` validado por schemas composables en `research.ts`. V1 continúa estricto: no acepta campos V2 desconocidos. La ausencia de una sección V2 se normaliza a `{ state: 'pending', artifact: null }`; un objeto presente inválido se rechaza.

Estados explícitos: implemented, artifact_available, executed, verified, experimental, review_required, pending, unavailable, demo. Son declaraciones de evidencia separadas de integridad de exportación. El exporter actual no promueve publicaciones a `verified` y mantiene métricas reportadas. Un artefacto pendiente no puede contener resultados.

- `sequences`: zonas/candidatos/instancias diferenciados, fuente y extremos de grilla, decisiones y referencias de revisión, recurrencia. Las instancias requieren declaración vinculada de revisión del sequence set; esto no inventa una decisión manual por intervalo. Las zonas aceptadas/rechazadas requieren revisión correspondiente.
- `experiments`: encoder, representación, método (incluido Agglomerative experimental), cobertura, agreement/stability nullable, máscara, ablación, inmutabilidad y métricas reportadas admitidas. No hay ranking automático.
- `evidence`: fuente native/legacy/external/manual, imported/verified/pending y canonical binding nullable. No se inventan reviewer ni fecha.
- `linkage`: dataset etiquetado independiente, contents/occurrences etiquetados normalizados, referencias al contenido de video y a todas sus secuencias, cosenos independientes. Candidato no equivale a confirmación; el productor actual no autoriza `confirmed`.
- `linkageReview`: evidencia por candidato cuando corresponda. `groupReview` mantiene separada la calibración real por labeled content / proposed visual dependency group con supported/unsupported/ambiguous/pending. No modifica estados de candidatos exactos.
- `reviewAggregation`: conteos descriptivos con confirmed null, enlace al mismo linkage, sin convertir supported en precisión ni confirmación.
- `variants`: declaraciones con identidad del mismo dataset; diferencias no establecen causalidad. El selector avanzado solo aparece cuando hay más de una.

Los IDs son aliases locales al snapshot. Ningún alias crea identidad científica entre dos exportaciones. Todos los registros extra tienen vocabulario controlado y límites. Null significa unavailable, nunca cero implícito. `exportPolicy` conserva su valor V1 para el bloque común; las extensiones identifican su integridad por `consumed-files-validated`.

Las fixtures `leakage-v2-synthetic.json` (producida por el exporter Python) y `research-fixture.ts` (estados de interacción) son exclusivamente sintéticas. No son evidencia de un experimento FLIR ejecutado.
