# Project status

## Propuesta de evaluación del detector con test común (2026-10-13)

**Infraestructura de planificación/verificación implementada y ejecutada sobre
las publicaciones reales. LISTO PARA REVISIÓN DEL PROTOCOLO; no listo para entrenar.**
Se verificaron recibos, snapshots, identidades y checksums de la auditoría
etiquetada, matriz de splits y revisión de aceptación. Se conservaron las fuentes
y, para la propuesta interna, se revalidaron los ZIP originales y la evidencia
importada. No se generaron nuevas asignaciones train/val ni métricas del detector.

La matriz previa contiene 30 configuraciones, 25 membresías y **25 tests distintos**,
con intersección común vacía; cinco aliases S2/S3 históricos no constituyen réplicas
independientes. Hay 24 membresías con cobertura completa de las cinco clases.

La propuesta interna conserva como anclas las 372 ocurrencias/contenidos del test
histórico. El cierre conservador común a todos los brazos protege 373 contenidos
y 514 ocurrencias; 142 ocurrencias quedan en cuarentena fuera de train/val y del
test evaluado (141 copias históricas y una ocurrencia Adendo candidata). Se
confirmaron 141 contenidos del antiguo test presentes en entrenamiento histórico
y ocho grupos exactos protegidos con conflictos de anotación. Las relaciones
perceptuales permanecen hipótesis, no procedencia confirmada.

| Clase YOLO | Instancias en test | Imágenes / contenidos en test |
|---|---:|---:|
| 0 | 12 | 12 |
| 1 | 589 | 193 |
| 2 | 270 | 175 |
| 3 | 93 | 76 |
| 4 | 19 | 18 |

Quedan 1.144 ocurrencias / 1.087 contenidos de desarrollo históricos y 1.192 /
1.135 ampliados. Los 30 brazos diseñados comparten exactamente test y protección.
Cumplen las condiciones necesarias de soporte por grupos, pero su factibilidad
de asignación train/val es **NOT_SOLVED**. El protocolo describe la extensión
bidireccional del motor existente; no reutiliza las 30 asignaciones anteriores.

Publicaciones nuevas fuera del repositorio, bajo `detector_evaluation_protocol_v1`:
`internal-proposal-20261013-b` (ID `a31a24468674f045`, **PROPOSED**, verificación CLI
independiente aprobada) y `prospective-readiness-20261013-a` (ID
`9e83d40ba65459fd`, **UNAVAILABLE**, sin población prospectiva admitida).
Ambas conservan revisión pendiente, ausencia de aprobación/freeze y
`ready_for_training=false`. La salida interna inicial `-a` se conserva incompleta,
sin recibo final: una comparación de índices pandas tras serialización Parquet
falló y se corrigió antes de publicar en otro directorio.
La revisión de aceptación posterior reprodujo y corrigió un defecto del gate de
fuentes: una revisión previa con dictamen rechazado y checksums recalculados
pasaba la comprobación de conteos. Ahora se exige `verdict=APROBADO`; este
dictamen previo no equivale a aprobar el nuevo protocolo del detector. La
regresión focalizada posterior a la corrección pasó **23/23** pruebas.

Los exports agregados verifican 48 runs / 16 splits históricos. Los bytes del
checkout tenían diferencias CRLF/LF frente a sus checksums publicados: se creó
una copia externa de los blobs exactos de Git, verificados contra el manifest,
con recibo de transporte. No se modificaron los exports ni se flexibilizó su
verificador. Esto no valida pesos, predicciones ni errores por imagen; esos
artefactos no están disponibles en la raíz local inspeccionada y siguen pendientes.

Faltan revisión de calidad de conflictos, decisión sobre suficiencia del soporte,
aprobación explícita y freeze, reconstrucción/verificación de train/val, adaptador
de entrenamiento validado, checksum del checkpoint inicial y batch/entorno
definitivos. El test interno no demuestra independencia retrospectiva. El contrato
prospectivo está diseñado; su ingesta/admisión aún no está implementada.
Validación: **126 passed** en pruebas sintéticas/offline de test común, detector,
splitting, candidatos etiquetados, reporte final y asociación; Ruff global y
`git diff --check` aprobados. No se ejecutó la suite global en esta intervención.
No se entrenaron modelos, eligieron ganadores, modificaron frontend, splitter
histórico, configuración/freeze originales ni resultados previos. No hubo
commit, push ni merge. Véanse el [protocolo](protocols/detector_common_test_v1.md)
y la [CLI y artefactos](runbooks/common_test.md).

## Splits candidatos etiquetados v1 (2026-10-13)

**Infraestructura implementada, piloto real de seis candidatos y matriz real de
30 configuraciones ejecutados y verificados.** Son particiones candidatas para
revisión; no se entrenó YOLO11n ni se ejecutaron DINOv2/CLIP sobre Adendo.
La publicación aceptada pasó verificación de dataset/audit IDs, todos sus
artefactos, los tres ZIP originales y la evidencia importada. El adaptador
reconstruyó anotaciones por ocurrencia: 1.657 históricas + 49 Adendo, 1.508 contenidos
consolidados; los ocho conflictos exactos y los diez huérfanos permanecen documentados.
Adendo conserva `original_split=null`; ratios derivados solo de 1.657 históricos.

Resultados **observados**, rangos entre semillas 0–4. Desviación de clases:
promedio absoluto, en puntos porcentuales, de proporciones de instancias por clase
frente a la población respectiva. Desviación de tamaño: máximo relativo por split.

| Población | Política | Runs | Grupos | Desv. tamaño (%) | Desv. clases (pp) | Cruces candidatos |
|---|---|---:|---:|---:|---:|---:|
| Histórica | S1 | 5 | 1459 | 0–0,269 | 1,274–1,878 | No disponibles |
| Histórica | S2 | 5 | 1459 | 0 | 0,040–0,058 | No disponibles |
| Histórica | S3 | 5 | 1459 | 0 | 0,040–0,058 | No disponibles |
| Ampliada | S1 | 5 | 1508 | 0,149–0,759 | 0,757–1,621 | 8–27 / 35 |
| Ampliada | S2 | 5 | 1508 | 0,149 | 0,049–0,071 | 11–24 / 35 |
| Ampliada | S3 | 5 | 1473 | 0,149 | 0,048–0,064 | 0 / 35 |

En todos los candidatos: cero contenidos exactos entre splits, cero fracturas
obligatorias y cobertura completa de ocurrencias. La referencia histórica
original retiene sus 198 contenidos cruzados; no fue modificada.
Las 41 evidencias candidatas describen 35 parejas por ocurrencia. No existen
parejas visuales candidatas internas al histórico en esta publicación:
S3 histórico equivale a S2 en membresía para las cinco semillas; no aporta
evidencia independiente. Hay 25 membresías distintas entre las 30 configuraciones.
Los seis runs piloto se reproducen dentro de la matriz por identidad y membresía
(no son seis configuraciones adicionales). El replay de los **30/30** candidatos
reprodujo las asignaciones con el código/versiones registrados; se verificó además
la igualdad con la auditoría externa y se recalcularon todas las métricas.

S1 histórico semilla 2 tiene cero instancias de Vehicles en validation. Los otros
29 candidatos cubren las cinco clases en las tres particiones. Se conserva esta
observación, sin cambiar la semilla ni reparar el baseline silenciosamente.
Los 20 runs MILP tienen factibilidad primal comprobada: 12 reportan óptimo dentro
de tolerancia y ocho estado 4 con solución factible; brecha máxima observada
0,03305 (3,305 %). No se demuestra óptimo único. Los S2/S3 históricos tienen
tamaños 1178/107/372; los ampliados 1213/110/383. Se conserva por run distribución
de clases, vacíos, cohortes y tamaños de grupos. El balance global no garantiza
balance por cohorte: S3 ampliado asigna entre 2 y 19 ocurrencias Adendo a test,
según semilla; no se incorporó un objetivo adicional de balance por cohorte.

S3 ampliado impone las relaciones seleccionadas como restricciones hipotéticas:
el cero residual sobre esas aristas **no valida** procedencia ni independencia.
No se calcularon nuevas similitudes coseno ni métricas temporales. Menor
desviación descriptiva no prueba superioridad científica o mejora del detector.
S2 histórico permite revisar una comparación en la misma población; S2/S3
ampliados permiten revisar la sensibilidad a las relaciones candidatas. Antes
de un entrenamiento posterior deben fijarse criterios de cobertura, tratamiento
de conflictos y comparación entre poblaciones; no se selecciona un ganador aquí.

Validación: **82 passed** en seis archivos de pruebas offline/sintéticas
(29 pruebas nuevas y 53 regresiones), Ruff global aprobado y `git diff --check`
aprobado. La suite global no se ejecutó en esta intervención. Los 48 runs
históricos del detector no se consumieron, modificaron ni revalidaron.
Un primer piloto incompleto se preservó sin recibo final: falló la comparación
de nulos del CSV (None/NaN), luego corregida y cubierta por pruebas. La publicación
válida usó otra salida. Los artefactos reales permanecen fuera de Git, bajo
`labeled_splits_v1`, en directorios nuevos. No hubo commit, push, merge ni rebase.
Véase el [contrato, CLI y verificación](runbooks/labeled_candidates.md).

## Consolidación etiquetada candidata (2026-10-12)

**Auditoría implementada y ejecutada sobre los tres ZIP reales; consolidación
candidata, sin nuevos splits ni entrenamientos.** `data audit-adendo` conserva
ocurrencias, contenido exacto, etiquetas originales y evidencia revisable.
La ejecución local observó 1.706 imágenes / 1.508 contenidos únicos: 1.657 imágenes
históricas y 49 de Adendo. Confirmó 10 etiquetas huérfanas históricas, 292 labels
vacíos válidos, 4.168 objetos históricos emparejados y 215 de Adendo (clases
0/1/4: 84/115/16). No hubo errores sintácticos/geometría ni identidades exactas
compartidas entre las dos cohortes. Se conservaron 198 grupos duplicados exactos
históricos (396 ocurrencias, 198 adicionales respecto de contenidos únicos),
ocho con conflictos de anotación; los cruces son 57 train–val y
141 train–test.

Con pHash Hamming ≤8 se generaron 35 parejas candidatas que involucran seis imágenes
Adendo. Las seis relaciones aportadas por el responsable se importaron además
como evidencia candidata, sin revisión visual nueva: 41 registros de evidencia
candidata sobre 35 parejas, no 41 correspondencias independientes. Las seis
presentan anotaciones diferentes. No encontrar candidatos para las otras 43
imágenes no demuestra independencia. `split_ready` sigue falso.

Validación de aceptación: **53 passed** en pruebas sintéticas de auditoría, manifest, inventario,
identidad y splitting; Ruff global aprobado. Verificados checksums de los outputs,
identidades de huérfanos y conflictos de los seis vínculos importados. Una ejecución
global no fue válida en el entorno core/dev: 550 passed, 137 skipped, 288 failed y
219 errors, con fallos observados por la barrera `safe_output` al usar un basetemp
dentro del repositorio y por extras experimentales ausentes como scikit-learn.
No se revalidaron los 48 experimentos de Hypatia. Los artefactos y
la evidencia importada permanecen fuera del repositorio. Véase el
[contrato y ejecución](runbooks/adendo.md).

## Ingesta de variantes visuales externas (2026-10-08)

**Ingesta real disponible localmente; puente de features implementado y validado
con datos sintéticos; embeddings reales de la variante y alineación pendientes.**
`data video-variant-ingestion build/verify/summary` publica
`video_variant_ingestion_v1` mediante staging, checksum receipts y un escritor
exclusivo. La lectura multishard acotada conserva cada entrada física, mientras
el manifest científico y las identidades existentes son independientes del
empaquetado para observaciones atribuibles. Índices repetidos permanecen ambiguos
y bloquean ese manifest, sin perder entradas ni inventar IDs.

Una publicación real local contiene 19.800 PNG repartidos entre seis ZIP.
Las fases previas validaron una lectura puntual de tres ocurrencias en tres
shards, sin verificar exhaustivamente todas las imágenes. El nombre externo
`hud_reduced` no demuestra eliminación completa del HUD. No hay anotaciones YOLO
ni correspondencia temporal exacta verificada; esta población sigue separada
del universo histórico etiquetado.

Las fases 1/2A/2B.1 añadieron lector multishard, factoría de fuentes y binding
`video_variant_feature_source_v1` al motor existente. La fase 2B.2 añade opciones
al mismo `features extract`, exige publicación/manifest original/input-root y
difiere la construcción del modelo hasta verificar fuentes. Selección por contenido,
mapping completo y procedencia física conservan las identidades científicas.
La validación es offline/sintética con constructores simulados; no se ejecutaron
modelos reales ni nuevos embeddings de esta variante. La fase 2B.1 pasó 255
pruebas focalizadas. En 2B.2, **253 passed** (55 casos nuevos y 198 regresiones)
en 247,76 s, más **1 passed** de regresión del consumidor CLI de configuración
de similarity en 3,61 s. Ruff sobre los archivos Python modificados/nuevos,
ayuda CLI y diff checks aprobaron. La suite global no se revalidó: los 34 fallos locales de
organización y el incidente ambiental de DLL previos no se corrigieron aquí.
Véanse [uso y límites de features](runbooks/features.md#features-desde-publicaciones-multishard).

La evidencia temporal es exclusivamente candidata/reportada. No se crean labels,
alineación verificada, secuencias, splits, entrenamiento ni exports nuevos, y los
extractores matemáticos permanecen intactos.
Validación inicial de ingesta: **60 passed** en pruebas nuevas y **60 passed** en regresión de datos
relacionada (identidad, inventario, manifest histórico y muestreo mock), sin
modelos ni datos reales. En esa fase aprobaron Ruff global, formato de archivos
nuevos, ayuda CLI y `git diff --check`; no se ejecutó la suite completa. Véanse
[contrato, límites y comandos](runbooks/video_variant_ingestion.md).

## Organización v2: pares normalizados (2026-10-05)

El responsable comunica un export v1 real completado en Hypatia; no se revalidó
localmente. El productor v2 separa todos los pares candidatos en `candidate_pairs`
y conserva cores/componentes y sus membresías. Usa JSON compacto con checksums
sobre bytes exactos. **Export real v2 pendiente**; la validación local es sintética.
Los apartados previos conservados debajo documentan el estado histórico v1.
Véase el [contrato vigente y migración](runbooks/organization_evidence_export.md).

## Membresía congelada para organización (2026-10-05)

Implementado fallback de presentación para publicaciones de clustering ausentes:
consume exclusivamente membresías verificadas de todos los splits seleccionados,
exige igualdad exacta entre seeds y conserva su procedencia por SHA256. Una
publicación presente pero inválida no activa el fallback. Configuración y
diagnósticos upstream no disponibles permanecen null; no se ejecuta clustering.
La evidencia operacional comunicada por el responsable en Hypatia no fue
revalidada localmente ni prueba que el export real haya terminado. Véase el
[runbook](runbooks/organization_evidence_export.md) para validación sintética y
el paso pendiente de export en Hypatia.

Validación offline/sintética: **70 passed** en organización/linkage y **874 passed**
en la suite completa, con 40 warnings preexistentes. Ruff global, ayuda CLI,
schema independiente y checks de diff aprobados. No se publicaron datos reales.

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
