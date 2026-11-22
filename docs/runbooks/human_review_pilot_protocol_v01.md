# FLIR — Protocolo de revisión humana y piloto v0.1

**Proyecto:** Auditoría de procedencia, continuidad temporal y dependencia visual de imágenes FLIR.
**Fecha:** 2026-11-03.
**Estado:** propuesta operativa para piloto; **no** resultados validados ni definición definitiva de secuencias.
**Herramienta:** `FLIR Human Audit v1.1`, interfaz local `/organization/sequences?mode=audit` y contrato `audit-evidence-v1` (revisión 1.1).
**Ámbito inicial:** familia `video_13min` de `VideoCodefest_008-46min.avi`; ampliación posterior a las demás fuentes FAC.

## 1. Objetivo y alcance

Documentar decisiones humanas reproducibles sobre pares/ventanas de observaciones, distinguiendo: (a) continuidad **audiovisual** y (b) dependencia **visual o geográfica**. El objetivo es obtener evidencia para evaluar posteriormente agrupaciones de imágenes en unidades de dependencia para particionamiento. Ninguna anotación por sí misma constituye un grupo final, una demostración de independencia ni una medición de leakage en el rendimiento del detector.

**Unidad de anotación:** un evento candidato con una ventana temporal identificada, sus evidencias antes/después y, cuando sea necesario, contexto ampliado. **Unidad de particionamiento futura:** secuencias/grupos de observaciones relacionadas —por definir y validar—; ni fotograma individual ni vídeo fuente completo por defecto.

## 2. Precondiciones para abrir un caso

1. El evento tiene `event_id` estable, recurso origen identificado, método/versiones y tiempo o índice etiquetado como **nominal** si no se ha verificado la alineación.
2. Las evidencias (fotogramas o mosaicos) están **vinculadas al evento por referencia explícita o correspondencia revisada**, no solo por cercanía numérica. Para medios externos usar el procedimiento `attach-evidence` de v1.1; conservar hash, localizador, justificación y responsable.
3. Registrar qué se ha consultado; no marcar `reviewed` con conclusión y confianza si no hay evidencia trazable consultada. Si falta material, dejar `pending` o una decisión totalmente `uncertain/undetermined`, según admita el formulario.
4. No consultar etiquetas YOLO ni métricas de rendimiento del detector para decidir continuidad o identidad de región. Si es viable, ocultar el nombre del split durante la primera valoración visual; consultar ese dato después para auditoría de cross-split.
5. No editar datos FAC originales, importaciones históricas ni resultados previos. Exportar y respaldar el ledger SQLite y revisiones antes y después del piloto.

## 3. Dos ejes independientes de clasificación

### 3.1 Continuidad audiovisual (`continuity`)

- **`continuous`**: se observa continuidad de la toma a través de la transición, con desplazamiento, zoom o cambio de tonalidad plausiblemente progresivos. Pueden existir fallos breves de señal o ganancia; en esos casos justificar la persistencia de referencias antes/después, no deducir automáticamente continuidad del fallo.
- **`shot_change`**: evidencia positiva de discontinuidad de la toma o cambio abrupto de encuadre que excede el movimiento normal observado localmente. Un fundido, flash, congelación o pantalla negra **por sí solos no bastan** para inferir un corte.
- **`uncertain`**: faltan imágenes, hay oclusión, señal perdida, zoom extremo, salto temporal o referencias insuficientes. Preferirlo a un juicio binario sin respaldo.

### 3.2 Dependencia visual o de región (`dependence`)

- **`same_region`**: coincidencias identificables y espacialmente coherentes de referencias persistentes del terreno (por ejemplo, un cruce vial + la disposición relativa de edificios, o la forma de un meandro + el camino asociado). Un objeto aislado genérico no basta. Incluir las referencias concretas en `justification`.
- **`different_region`**: evidencia positiva de localizaciones distintas; por ejemplo, hitos claramente incompatibles y contexto suficiente o metadatos geográficos fiables que permitan excluir la identidad. **Una correlación baja, un cambio de toma o la falta de coincidencias no demuestran por sí mismos `different_region`.**
- **`uncertain`**: no se puede establecer ni descartar identidad de región con la evidencia disponible. Este es el estado por defecto ante ambigüedad.

**Combinaciones válidas:** `shot_change` + `same_region`; `continuous` + `same_region`; `shot_change` + `uncertain`; `uncertain` + `same_region`. Evitar inferir independencia por un cambio audiovisual. Una continuidad de cámara no equivale por sí sola a una etiqueta de identidad geográfica si el campo de visión se ha desplazado demasiado.

## 4. Fenómenos observables (`phenomena`)

Marcar únicamente lo efectivamente observado: `zoom`, `camera_motion`, `gain_change`, `polarity_change`, `signal_loss`, `unknown`. Se pueden marcar varios. No marcar `polarity_change` salvo que existan indicios de inversión térmica; no todo cambio de contraste lo es.

## 5. Confianza y justificación

- **`high`**: referencias persistentes inequívocas o discontinuidad clara en varias imágenes; evidencia suficiente, localizador exacto y una alternativa plausible descartada.
- **`medium`**: patrón compatible y varias pistas, pero alguna ambigüedad (movimiento, baja resolución, solapamiento parcial).
- **`low`**: indicios débiles; preferir `uncertain` para la dimensión que no puede sustentarse.
- **`undetermined`**: falta evidencia o no corresponde atribuir confianza a una conclusión.

La interfaz v1.1 usa **una confianza global**; cuando ambas dimensiones tienen distinta solidez, reportar la menor de las afirmaciones conclusivas y explicar la diferencia por escrito. No atribuir confianza estadísticamente calibrada a estas etiquetas humanas.

Formato recomendado para `justification`:

> Observación antes: …; observación después: …; referencias persistentes o incompatibles: …; explicación alternativa (zoom, ganancia, señal): …; límite de la evidencia: …; decisión de continuidad: …; decisión de región: …

No usar como justificación «índices consecutivos», «r bajo», «lo clasificó Codex» o «están en splits distintos» sin evidencia visual.

## 6. Orden de revisión por caso

1. Verificar `event_id`, archivo fuente y ubicación del evento; registrar cualquier ausencia de evidencia.
2. Revisar la secuencia **anterior**: identificar al menos dos referencias del terreno, si existen.
3. Revisar la secuencia **posterior** y buscar persistencia, correspondencia geométrica o incompatibilidad explícita.
4. Examinar el instante de transición; separar cambio audiovisual de cambio térmico, pérdida de señal, paneo y zoom.
5. Si hay dudas, consultar un intervalo más amplio y/o los fotogramas del dataset alineados con la fuente; anotar la evidencia adicional exacta.
6. Elegir **cada dimensión por separado**, confianza y fenómenos; registrar justificación con limitaciones.
7. Guardar en el ledger; exportar copia de decisiones al terminar el bloque. No reconciliar duplicados con decisiones existentes sin respaldos y recibo.

## 7. Diseño del piloto: 24 casos

La muestra será **dirigida para comprobar utilidad y consistencia del protocolo**, no una muestra aleatoria para estimar desempeño global del detector. El plan es:

| Estrato | n | Propósito |
|---|---:|---|
| Alertas fuertes con aparente discontinuidad | 6 | Probar si los cambios se sostienen al ampliar contexto |
| Alertas fuertes con probable continuidad/cambio de ganancia | 6 | Capturar falsos cortes por cámara/sensor |
| Eventos `unresolved_low_similarity` | 6 | Probar manejo de incertidumbre |
| Transiciones sin alerta 5 FPS, seleccionadas con semilla fija | 6 | Controles negativos del detector de cambios (sin prejuzgar continuidad) |
| **Total** | **24** | |

**Inclusiones diagnósticas obligatorias:** disponer de evidencia para las fronteras históricas 199–200, 349–350, 399–400 y 499–501 cuando existan imágenes y correspondencia verificable. Incluirlas dentro de los estratos anteriores si califican, o documentarlas como una **submuestra adicional no representativa**. No rotularlas como verdad terreno. Evitar que todas las ventanas provengan de un único tipo de transición. Registrar `selection_reason`, `sampling_seed`, estrato, presencia de evidencia y split *después* de la clasificación visual.

Si un estrato carece de seis casos con evidencia verificable, **no reemplazarlo silenciosamente**: reportar el faltante, registrar exclusiones y completar la muestra tras la fase de alineación. No generar falsos «controles negativos» solo porque no haya alerta; son controles del sistema de detección, no negativos de continuidad.

## 8. Consistencia entre revisores

**Ideal:** dos revisores anotan independientemente los 24 casos, sin ver la decisión del otro, y se comparan por eje. La v1.1 puede registrar revisiones sucesivas de una decisión, pero **esto no equivale a dos anotaciones independientes**: usar ledgers/copias independientes o ampliar el diseño antes de medir acuerdo interevaluador.

**Si hay un solo revisor:** repetir una submuestra ciega (por ejemplo, ocho casos, tras varios días) y reportar acuerdo intraevaluador, no interevaluador. Comparar matrices por eje y discusiones de desacuerdo; el estadístico kappa solo si es apropiado y siempre junto con conteos y tamaño de muestra. No afirmar confiabilidad general a partir de 24 casos.

Los desacuerdos se resuelven en una fase de adjudicación documentada, conservando decisiones originales, revisión y responsable. No reemplazar opiniones previas sin historial.

## 9. Criterios de aceptación del piloto

- [ ] 24 casos seleccionados con estrato, razón y trazabilidad; exclusiones documentadas.
- [ ] Cada caso `reviewed` conclusivo tiene evidencia consultada asociada y localizador reproducible.
- [ ] Las dos dimensiones fueron valoradas independientemente; `uncertain` permanece disponible.
- [ ] Se registraron fenómenos visibles y justificación específica.
- [ ] La interfaz permite corregir y reabrir decisiones sin pérdida de historial.
- [ ] Los resultados se exportaron en JSON/CSV y el SQLite se respaldó.
- [ ] Se reportó la cantidad de casos sin evidencia y por qué.
- [ ] Se realizó una revisión de consistencia (doble evaluación independiente o repetición intraevaluador explícitamente identificada).
- [ ] Se elaboró una matriz de desacuerdos y criterios que requieren ajuste.
- [ ] No se declararon secuencias ni leakage definitivo por la sola anotación.

## 10. Decisión de cierre

Entregar un informe piloto con: tabla de los 24 casos, referencias de evidencias, decisiones por eje, confianza, notas, discrepancias y recomendaciones para v0.2 del protocolo. **No crear el grafo ni los splits nuevos hasta revisar la calidad de alineación y dependencia.**

### Estado de hallazgos iniciales (hipótesis; no ground truth)

- 199–200: continuidad visual probable.
- 349–350: cambio de toma probable.
- 399–400: continuidad visual probable.
- 499–501: cambio de toma probable, con índice 500 ausente del conjunto histórico.

Estas hipótesis ayudan a muestrear casos diagnósticos, pero **no deben mostrarse al evaluador antes de registrar su primera decisión**.
