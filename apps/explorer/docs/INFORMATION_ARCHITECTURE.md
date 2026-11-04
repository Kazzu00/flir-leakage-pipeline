# Information architecture · UX v2

M02 responde cuatro preguntas con cuatro destinos principales:

| Sección             | Ruta                       | Pregunta                                         |
| ------------------- | -------------------------- | ------------------------------------------------ |
| Overview            | `/organization`            | ¿Qué datos y análisis están disponibles?         |
| Visual exploration  | `/organization/explore`    | ¿Qué contenidos se relacionan visualmente?       |
| Sequences & linkage | `/organization/sequences`  | ¿Qué evidencia temporal y entre datasets existe? |
| Evaluation          | `/organization/evaluation` | ¿Qué consecuencias downstream pueden evaluarse?  |

Overview muestra conteos del snapshot, representaciones y dos ramas: estructura visual (similitud/reducción/clustering) y temporal (secuencias/experimentos/revisión). Ambas contextualizan linkage y la futura organización/evaluación; las flechas no implican que todas las etapas se hayan ejecutado ni que linkage sea un requisito de la partición histórica.

Visual exploration usa `?view=embeddings|similarity|reduction|clustering`. Los controles muestran encoder, método/representación y selección de run solo cuando hay alternativas. Las coordenadas guardadas ocupan la vista principal; el detalle de contenido y la grilla conservan selección. Los parámetros, semillas, IDs de espacios y métricas completas se abren en Technical details. El modo DEMO mantiene fixtures explícitas, separadas de artifacts.

Sequences & linkage usa `?mode=sequences|linkage`, sin crear páginas para evidencia legacy ni agregación. Intervalos candidatos/revisados conservan vocabulario, patrón y texto. Las revisiones de grupos se muestran separadas de candidatos de enlace. Las variantes solo aparecen en detalles cuando hay más de una declaración; este selector inspecciona metadata y no cambia silenciosamente la población.

Evaluation usa `?view=splits|detector` y abre el detector por defecto. El contrato público verificado `detection-export-v1` presenta comparación, variabilidad, clases, composición de test y asociaciones exportadas, con procedencia/limitaciones/runs en drawers. Es independiente del snapshot sampled-video y de su selector DEMO/MIXED. La vista `splits` conserva su estado pendiente cuando no existe evidencia compatible; no se deduce dependencia residual desde una métrica de clustering ni desde la ausencia de duplicados. Ver [DETECTION_CONTRACT](DETECTION_CONTRACT.md).

## Compatibilidad

| Ruta antigua (prefijo `/organization`) | Destino                                  |
| -------------------------------------- | ---------------------------------------- |
| `/dataset`                             | `/organization`                          |
| `/embeddings`                          | `/organization/explore?view=embeddings`  |
| `/similarity`                          | `/organization/explore?view=similarity`  |
| `/reduction`                           | `/organization/explore?view=reduction`   |
| `/clustering`                          | `/organization/explore?view=clustering`  |
| `/groups`                              | `/organization/sequences`                |
| `/splits`                              | `/organization/evaluation?view=splits`   |
| `/detector`                            | `/organization/evaluation?view=detector` |

Home mantiene source video → M01 → M02 → M03 → evaluación, con enlaces nativos y una bifurcación ligera dentro de M02. No enumera todas las subetapas. M01/M03 y las rutas globales conservan su alcance previo.
