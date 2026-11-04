# Fase 2A.1 — piloto preparado, revisión humana pendiente

Base: `feat/flir-phase2a-multisplit-alignment`, commits `a95940a` y `8b671fb`.
Trabajo aislado en `feat/flir-phase2a1-human-alignment-review`. Se preservaron los
cambios locales ajenos de `organization_contract.py` y `repository_audit.md`.
Frontend sin modificaciones; servicio local de revisión integrado en la CLI backend.

## Selección real reproducible

Fuente: paquete completo de Fase 2A, verificado antes de preparar. Semilla 20261103.
24 imágenes distintas: **9 train, 6 val y 9 test**. Composición:

| Estrato | Índices |
| --- | --- |
| Ocho diagnósticos | 199, 200, 349, 350, 399, 400, 499, 501 |
| Cuatro adicionales train | 539, 736, 6, 168 |
| Cuatro adicionales val | 357, 372, 364, 351 |
| Cuatro adicionales test | 410, 421, 490, 436 |
| Cuatro difíciles | 630 (train, abstención), 263 (test), 88 (train), 48 (train) |

Los adicionales se eligen por split entre los casos de Fase 2A no usados por las
fronteras ni por el 630. Los otros tres difíciles tienen los menores márgenes de
inliers entre top-1 y las alternativas dentro del remanente. No se seleccionaron
por etiquetas YOLO ni rendimiento del detector.

El 630 ocupa una plaza especial: **23 propuestas automáticas encontradas y una
abstención**, no 24 matches confirmables. El índice 500 sigue ausente, sin sustitución.
Se conservan los IDs excluidos y la razón de selección de cada caso en `pilot.json`.

## Evidencia disponible

Se reextrajeron únicamente PTS existentes, con comparación de sus hashes RGB:
120 alternativas (cinco por caso) y 24 controles existentes. No hubo nueva búsqueda.
Se generaron **380 imágenes**: 24 originales visuales, 120 cuadros candidatos,
120 visualizaciones de correspondencias, 92 superposiciones válidas y 24 controles.
Las 28 homografías restantes no generan superposición.

Las 120 reconstrucciones concuerdan en cantidad de matches e inliers con las
métricas originales. Se conservan los valores originales de correlación, dHash,
cobertura, proporción de inliers, error geométrico y PTS, junto con la matriz y
puntos reconstruidos. Una homografía plausible sigue sin verificar identidad.
Las alternativas cercanas no se ocultan. El original conserva su procedencia y
hash original aunque la copia visual servida sea PNG.

## Estado humano y análisis

**24 pendientes; cero anotaciones humanas FAC; cero `human_verified_top1`.**
Los pendientes iniciales son estados implícitos, no decisiones fabricadas.
El análisis inicial tiene proporciones `null`, no una exactitud de cero.
La confianza automática figura como no disponible/no calibrada; las bandas
descriptivas son 22 casos con ≥40 inliers, uno con <40 y una abstención.

Los juicios `correct_match`, `incorrect_match`, `uncertain` pertenecen a alineamiento.
No se emiten etiquetas de continuidad o dependencia geográfica. Se registran
confianza humana, evidencia consultada, revisor, justificación, control y posible
alternativa mejor; las correcciones son versiones adicionales. El reporte usa la
última versión y mantiene denominadores explícitos y un mínimo operacional de cinco.

## Validación técnica observada

El chequeo de Chromium cargó **380/380 imágenes FAC** sin errores de runtime,
recorriendo los 24 casos. Reinició el servicio y comprobó historial real idéntico,
con cero anotaciones. No se hizo POST de decisiones sobre FAC.

En otro piloto exclusivamente sintético se cargaron 168 imágenes. Se rechazó una
conclusión sin evidencia; después se guardó una revisión y una corrección a
`incorrect_match` con alternativa mejor. Las dos versiones y el export JSON exacto
persistieron tras terminar y volver a iniciar el proceso. Se verificó CSV, historial,
selección de alternativa y persistencia de español con tildes. Las anotaciones
sintéticas no se trasladaron al piloto real.

La regresión focalizada terminó con **76 pruebas aprobadas en 29,50 s** y Ruff
global aprobado. Repetir la preparación verificó 382 archivos sin cambiar sus bytes;
la preparación inicial había decodificado 2.195 cuadros, sin búsqueda de nuevos PTS.
El segundo chequeo Chromium repitió satisfactoriamente carga y persistencia.
El wheel incluye explícitamente `pilot_ui.html`; se
comprobó igualdad de bytes con el archivo fuente. Las suites globales completas y
las suites frontend no se repitieron; no se declaran aprobadas por esta tarea.

Una primera prueba HTTP de pytest fue bloqueada por la prohibición global de red.
Se adaptó la prueba de handlers a transporte en memoria sin eliminar esa prohibición.
Las conexiones loopback reales se ejercitan por separado en el chequeo Chromium.
Un primer wheel omitía el HTML por la exclusión general de reportes; se añadió una
inclusión explícita y la segunda construcción se verificó correctamente.

## Artefactos locales

- `reports/phase2a1-pilot`: paquete inmutable, selección, procedencia y medios.
- `reports/phase2a1-reviews.sqlite3`: ledger separado, sin anotaciones reales.
- `reports/phase2a1-validation/selection.md`: tabla de selección con IDs.
- `reports/phase2a1-validation/initial-analysis.json`: análisis inicial pendiente.
- `reports/phase2a1-validation/initial-reviews.json` y `.csv`: 24 pendientes
  identificados explícitamente, sin atribuir decisiones humanas.
- `reports/phase2a1-validation/provenance.json`: verificación de evidencia fuente,
  hashes y versiones locales; comprobación registrada después de la preparación.
- `reports/phase2a1-browser-*`: resultados y capturas de chequeos de navegador.

El paquete Fase 2A verificó íntegro después de la preparación; el hash del ledger
histórico coincide con el registrado en Fase 2A. No se modificaron FAC, etiquetas,
splits ni registros anteriores. No se entrenó, agrupó, construyó grafo, recalibró,
subió datos ni realizó push, merge o despliegue.

## Límites

La muestra es dirigida y pequeña. No permite afirmar desempeño poblacional ni
calibración; solo hay un caso de abstención y las bandas de soporte están desbalanceadas.
Los controles son negativos propuestos, sujetos a juicio humano. Las visualizaciones
geométricas trabajan sobre la región interior reducida de Fase 2A, mientras el original
se ofrece a resolución completa. No sustituyen inspección de detalle cuando sea necesaria.
El servicio es local y no autentica identidades de revisores; las decisiones son
autoatribuidas. No se implementó adjudicación ni acuerdo entre evaluadores.

Siguiente paso: una persona aplica el [runbook](../runbooks/phase2a1_human_alignment_review.md)
y registra sus juicios. Hasta entonces, los 24 casos permanecen pendientes.
