# Secuencias revisadas sobre la grilla de video

Infraestructura `sequences` para `flir_video_samples_v1`, validada con fixtures
sintéticos. El manifest, los modelos ya extraídos y la revisión real permanecen
fuera de Git. No se ha ejecutado ni validado aquí la construcción real en Hypatia.
No se entrena ningún modelo, no se leen JPEGs/videos ni se crea train/val/test.

## Entradas y comandos

Usar Python 3.11 y el entorno core/dev del repositorio. No se necesitan los extras
vision/reduction, GPU, acceso a Internet ni matrices de similitud. Se requieren:

- Manifest completo a 1 FPS, `sample_index` contiguo desde cero por video.
- Directorios completos CLIP y DINOv2: raw/L2 float32, content/record index y
  metadata con revisión resuelta. Cada encoder conserva su propio embedding_row.
- Para build/verify del conjunto: directorio de revisión manual confirmada.

```powershell
uv run flir-pipeline sequences --help
uv run flir-pipeline sequences detect --manifest <manifest.parquet> --clip-features <clip_features> --dinov2-features <dinov2_features> --config configs/sequences/research.yaml --output artifacts/sequences
uv run flir-pipeline sequences summary <candidate_directory>
uv run flir-pipeline sequences verify <candidate_directory> --manifest <manifest.parquet> --clip-features <clip_features> --dinov2-features <dinov2_features>
uv run flir-pipeline sequences build --detection <candidate_directory> --manifest <manifest.parquet> --clip-features <clip_features> --dinov2-features <dinov2_features> --validation <confirmed_review_directory> --output artifacts/sequences
uv run flir-pipeline sequences verify <sequence_set_directory> --manifest <manifest.parquet> --clip-features <clip_features> --dinov2-features <dinov2_features> --validation <confirmed_review_directory>
uv run flir-pipeline sequences summary <sequence_set_directory>
```

Si Windows bloquea los launchers, usar `uv run --no-sync python -c
"from flir_pipeline.cli import app; app()" sequences ...` en una sola línea.
El flag `--no-sync` presupone un entorno ya preparado. Análogamente, pytest se
ejecuta con `uv run --no-sync python -m pytest`.

`detect` publica únicamente candidatos. `build` verifica la detección contra las
fuentes y consume exclusivamente `decision=accept`. Una anotación
`high_confidence=false` no excluye un evento aceptado. `summary` lee dos JSON,
declara `verification_performed=false` y `quality_valid=null`; nunca sustituye
`verify`. Este último exige fuentes, reconstruye los resultados y sale con código
distinto de cero ante fallo. La calidad estructural no certifica verdad semántica.

## Regla numérica versionada

Para cada video y encoder se reconstruyen todas las ocurrencias desde record_index.
En el corte t y ventana w se comparan los centroides de `[t-w,t)` y `[t,t+w)`,
normalizados por separado: cambio = `1 - cos(centroide_antes, centroide_después)`.
Se usan los embeddings L2 originales, sumas prefijas float64 y ventanas completas.
Un centroide de norma cero deja el corte indefinido (NaN); no se inventa un score.
No se concatenan encoders ni se deduplican las grillas temporales.

Los cambios se redondean a **12 decimales antes de ordenar**. Cada percentil es
`average_rank / valid_count`, independientemente por encoder, video y ventana,
sobre todos los cortes válidos para esa ventana. No es `(rank-1)/(N-1)` ni un
percentil calculado solo dentro de candidatos o ventanas de búsqueda.

`Pw=min(percentil_CLIP_w,percentil_DINOv2_w)` para w=1,3,5,10,20.
`S=min(P5,P10,P20)`. Se retienen cortes con S≥0.95 y se fusionan candidatos
consecutivos del mismo video con distancia ≤3 muestras. El representante maximiza,
en orden, S, mediana(P5,P10,P20), P1 y después minimiza el índice.
P1 es el mínimo de `clip_percentile_w1` y `dinov2_percentile_w1`: los cambios
adyacentes de cada encoder se redondean a 12 decimales y se convierten a percentiles
independientemente por video, sobre todos los cortes válidos, con average ties /
valid_count. Las magnitudes raw de CLIP y DINOv2 tienen escalas distintas y no se
combinan directamente. `P1` en temporal_scores y `coarse_P1` en candidate_events
identifican inequívocamente este consenso de percentiles. P1 (también denominado
F1 como diagnóstico) solo sirve como tercer desempate grueso; nunca es un umbral
de aceptación ni sustituye F3 en la localización final.
`high_confidence` indica que algún candidato del evento tiene S≥0.975.

La localización fina maximiza F3=P3 en ±20 muestras del representante grueso.
Entre centros a<b, el corte `floor((a+b)/2)` pertenece al evento izquierdo y el
siguiente índice inicia el intervalo derecho. Estos intervalos son disjuntos;
cada evento conserva su propia región aunque un vecino tenga un pico mayor.
Los empates F3 prefieren cercanía al representante y después el índice menor.
Se registran los intervalos incluso si sus extremos no tienen una ventana F3
completa; la selección utiliza únicamente posiciones F3 válidas.

Los defaults implementan el protocolo solicitado. Umbrales, merge_gap y radio
pueden cambiarse explícitamente en YAML y producen otro ID. Ventanas, política de
percentiles, redondeo, representación y regla de aceptación se validan mediante
Pydantic; claves desconocidas se rechazan. La revisión anterior debe coincidir
con los nuevos candidatos antes de poder reutilizarse.

La política `score_median_P1_stable_percentile_consensus_lower_index` entra en la
identidad del artefacto; la anterior política de mínimo raw se rechaza y sus
candidatos deben regenerarse. El contrato CSV confirmado no incorpora columnas
nuevas obligatorias. Según el responsable, la revisión real ya usaba percentiles
instantáneos y sus eventos no se decidieron por ese tercer desempate: se esperan
las mismas posiciones, pendientes de comprobar en Hypatia sin hardcodearlas.

## Contrato de revisión confirmada

El directorio contiene exactamente las tres entradas relevantes siguientes; no se
modifican ni se generan automáticamente confirmaciones:

1. `boundary_validation.csv`: tabla autoritativa, incluidos los rechazos.
2. `accepted_boundary_candidates.csv`: mismo esquema; se verifica que sea el
   subconjunto exacto `accept` de la primera tabla.
3. `metadata.json`: contrato confirmado y procedencia de la revisión.

Columnas CSV:

```text
event_id,video_id,coarse_sample_index,search_start,search_end,
localized_sample_index,localization_shift,f3_score,persistent_stable_min,
persistent_stable_median,high_confidence,repeat_partner_sample_index,
decision,boundary_type,notes,review_status,sequence_boundary_committed,
manual_confirmation
```

`event_id` es un entero del reporte manual, no un ID de secuencia. Los eventos se
vinculan por `(video_id,coarse_sample_index)`; la numeración manual se conserva.
Se exige cobertura de todos los eventos recomputados, unicidad, ventanas y corte
localizado idénticos y scores coincidentes con tolerancia absoluta 1e-12.
Si la regla diagnóstica previa difiere, el build falla: investigar la diferencia
antes de revisar/reconfirmar un nuevo artefacto; nunca trasladar decisiones
silenciosamente. `repeat_partner_sample_index` es diagnóstico opcional y no
define límites ni aristas. Las dependencias se reconstruyen desde content_id.

Metadata obligatoria:

| Campo | Condición |
|---|---|
| artifact_kind | confirmed_manual_boundary_validation |
| ground_truth | false booleano |
| diagnostic_source | cadena no vacía, conservada como procedencia |
| review_status | confirmed_manual_review |
| manual_confirmation_complete | true booleano |
| manual_confirmation_required_before_sequence_commit | false booleano |
| sequence_boundaries_committed | booleano; false es esperado y válido |
| confirmation_timestamp_utc | fecha/hora ISO con zona UTC |
| event_count, accepted_count, rejected_count, boundary_type_counts | deben coincidir con CSV |
| source_checksums | hashes SHA256 de las fuentes provisionales declaradas |

Cada fila debe tener `review_status=confirmed_manual_review` y
`manual_confirmation=true`. Las únicas decisiones son accept/reject. Accept
requiere scene_change, degradation_transition o transition_interval; reject
requiere tipo reject. `sequence_boundary_committed=false` en la entrada describe
una construcción pendiente y no anula un accept. Se conservan notas, tipo y fecha.
Un transition_interval se compromete como el corte revisado t, conservando su
tipo; no se infiere un intervalo de incertidumbre no suministrado.

La revisión heredada **no contiene binding del manifest ni de features**. Sus
checksums provisionales se conservan como declaraciones sin afirmar que aquí se
releyeron esas fuentes. El nuevo conjunto vincula los bytes de los tres archivos
confirmados y verifica geometría/scores contra las fuentes suministradas. Esto
detecta revisiones incompatibles, pero no autentica a la persona revisora ni
convierte su decisión en ground truth.

## Publicación, identidades y dependencias exactas

Directorios inmutables:

```text
artifacts/sequences/candidates/<detection_id>/
  temporal_scores.parquet
  candidate_events.parquet
  summary.json
  metadata.json
artifacts/sequences/sets/<sequence_set_id>/
  temporal_scores.parquet
  candidate_events.parquet
  manual_review.parquet
  boundaries.parquet
  sequence_instances.parquet
  occurrence_assignments.parquet
  dependency_edges.parquet
  dependency_support.parquet
  summary.json
  metadata.json
```

Metadata se publica al final, tras comprobar escritura y cobertura. Directorios
incompletos se preservan y rechazan; no hay borrado automático ni overwrite.
Un solo writer por publicación e inputs inmutables durante compute/verify.
Los resultados completos se reutilizan únicamente tras reconstrucción y QA.

Los IDs usan el helper existente: primeros 16 hex de SHA256 de JSON canónico,
claves ordenadas y sin NaN. detection_id vincula dataset_id, checksum del manifest,
checksums de metadata/raw/L2/índices de ambos features, configuración matemática
de ambos encoders y configuración del algoritmo. sequence_set_id añade el ID de
detección y la firma de los tres archivos confirmados. Esa firma es SHA256
completo del JSON canónico del mapa nombre→checksum. Paths y tiempo de ejecución
no entran al ID; cambiar los bytes de un input sí cambia la publicación aunque
sus campos operacionales no cambien feature_space_id.

Cada sequence_id deriva de sequence_set_id, video y extremos inclusivos. Video
no es sequence_id. El corte t deja la secuencia previa en t−1 y abre la siguiente
en t; sin cortes aceptados queda una instancia para ese video. Esto no afirma que
el video contenga una única escena real. Se registran extremos, timestamps de
grilla, conteos, fuente y referencias a ambas fronteras. Todos los frame_id,
content_id y los dos embedding_row permanecen en occurrence_assignments.

Por cada content_id presente en varias secuencias se ordenan los sequence_id y se
conecta el primero con cada uno de los demás: `exact_shared_content_star_v1`.
La estrella produce los mismos componentes que la clique completa, con O(k)
aristas por contenido compartido. dependency_support conserva todas las
ocurrencias justificativas, y las aristas incluyen content_id y conteos a ambos
lados. No es una enumeración exhaustiva de parejas de ocurrencias/secuencias.
Cada componente, incluidos singletons, recibe un ID determinista basado en el
conjunto ordenado de sus sequence_id y sequence_set_id. Un duplicado aislado puede
unir transitivamente componentes grandes: es una dependencia exacta observada,
no una afirmación de similitud visual global ni una implementación visual_group.

## Verificación y límites científicos

Verify comprueba fuentes/checksums, feature identity/pooling/revisión/cobertura,
configuración y políticas, IDs, todos los archivos esperados y sus valores
recomputados, revisión confirmada, exclusión de rechazos, cobertura total única,
intervalos sin gaps/overlaps, identidad fuente, aristas exactas y componentes/IDs.
También audita directamente la cobertura y el grafo, además de reconstruirlos.
Modificar una tabla y actualizar su checksum no basta para aprobar QA.
Ni clustering, reducciones, labels ni split histórico son entradas del algoritmo.

La memoria de detección es O(N_video × D) más tablas O(N_occurrences), con un
encoder/video activo; se leen los arrays de features mediante mmap. El QA
preexistente de features usa temporales O(N_content × D). La construcción y las
dependencias tienen memoria lineal; no hay matrices NxN ni joins cartesianos.
Los archivos Parquet se cargan como tablas lineales. Memoria/tiempo en Hypatia
siguen pendientes de medición; no se afirma una cota de RSS observada.

La revisión confirmada real comunicada por el responsable contiene 14 eventos,
13 aceptados y uno rechazado sobre tres fuentes. Al ejecutarla posteriormente se
esperan 16 instancias, consecuencia de fuentes + cortes aceptados, **no una
constante del código ni evidencia de una ejecución realizada aquí**. Comprobar
también los bloques/pares exactos repetidos registrados en el reporte local, sin
versionar IDs ni posiciones privadas. No basta comparar el conteo final.

Pruebas sintéticas: redondeo/empates, ventanas y bordes, consenso multiescala,
merging, representantes, localización por puntos medios, mapeos independientes,
rechazos/provisionales, cobertura, copias exactas/componentes transitivos,
determinismo, corrupción con checksum reescrito y summary liviano.

```powershell
uv run --no-sync python -m pytest
uv run --no-sync ruff check .
uv run --no-sync ruff format --check src/flir_pipeline/sequences tests/test_sequences.py
uv run --no-sync python scripts/check_notebook_source.py
git diff --check
```

Siguiente paso: ejecutar detect/build/verify con las fuentes completas y revisión
congelada en Hypatia, investigar cualquier diferencia de localización y registrar
el recibo. Después podrá diseñarse un protocolo de split que mantenga indivisibles
los componentes exactos y mida correlación residual. Esta implementación no
demuestra ausencia de toda dependencia ni mejora del detector.
