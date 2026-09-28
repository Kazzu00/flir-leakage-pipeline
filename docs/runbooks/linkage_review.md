# Calibración manual de evidencia etiquetado → grupo visual

`linkage review init/record/summary/verify` prepara evidencia y registra una
valoración humana por consulta etiquetada y grupo visual propuesto. La revisión
es `ground_truth=false`; nunca confirma enlaces automáticamente ni crea un split.
Un grupo visual es una restricción must-link para un futuro particionamiento.
Conserva las secuencias como segmentos temporalmente continuos e independientes.

La pregunta es «¿la evidencia apoya el enlace con este grupo?». No se evalúa el
frame exacto de origen ni se identifica una única sequence_instance. Los cosenos
CLIP y DINOv2 permanecen separados. El acuerdo de ranks es evidencia para el
revisor, sin umbral de aceptación ni promoción automática a verdad.

## Contratos de entrada

La muestra CSV requiere `labeled_content_id`,
`proposed_visual_dependency_group_id` y `review_stratum`, con **una consulta
etiquetada única por fila**. Los campos adicionales se conservan en
`sample_metadata_json`. `init` deja todas las decisiones en blanco, incluso si
la muestra contiene columnas de decisiones anteriores. Para evaluar otro grupo
para la misma consulta, preparar otra calibración explícita; no multiplicar
decisiones según la cantidad de ocurrencias.

Se consumen también la publicación `linkage`, su manifest etiquetado exacto y el
directorio del set de secuencias que produjo sus occurrence assignments. El
adaptador comprueba sus identidades, checksums y joins; no recalcula embeddings
ni cambia la revisión de límites. El QA numérico completo sigue correspondiendo
a `linkage verify`, y el QA independiente del detector F3/revisión de límites a
`sequences verify` con sus fuentes originales.

El lector distingue dos contratos explícitos. Ambos consumen un directorio
`--visual-dependencies` con `metadata.json` y una tabla seleccionada mediante
`--membership-table`. La membresía debe cubrir cada secuencia del set exactamente
una vez, incluyendo singletons. No se completan filas ni se generan grupos.

**Productor confirmado de Hypatia, v1.** Se reconoce exclusivamente mediante
`artifact_kind="confirmed_manual_visual_dependency_validation"` y
`artifact_version=1` (entero). Requiere todas estas declaraciones:

- `ground_truth=false`, `review_status="confirmed_manual_review"` y
  `manual_confirmation_complete=true`.
- `sequence_set_id` coincidente con el set de secuencias consumido.
- `semantic_role="must_link_constraint_for_leakage_safe_split"`.
- `sequence_instances_merged=false`, `split_created=false` y
  `exact_duplicate_dependencies_preserved=true`.

Este productor **no tiene `artifact_id` ni `output_checksums`**. No se exigen,
inyectan ni escriben esos campos en el artefacto confirmado. El consumidor calcula
SHA256 de los bytes exactos de `metadata.json`, `visual_dependency_groups.csv` y
la tabla de membresía seleccionada. Registra en su propia metadata:

- `visual_dependency_files`: los tres SHA256 calculados por el consumidor.
- `visual_dependency_consumer_source_identity`: kind/version del productor,
  sequence_set_id y checksums, bajo el contrato
  `consumer_bound_visual_dependency_source_v1`.
- `visual_dependency_consumer_source_fingerprint`: SHA256 del JSON canónico de
  esa identidad (claves ordenadas, UTF-8, sin espacios separadores).

Esta huella es una identidad **del consumidor**, no un ID declarado por el
productor. `verify` vuelve a calcular los tres hashes y rechaza cualquier cambio
respecto de la calibración, aunque sea solo whitespace. El CSV de grupos se liga
por sus bytes; el lector no reconstruye ni reinterpreta su definición de grupos.
Las columnas adicionales de membresía, como procedencia, intervalos,
exact_duplicate_dependency_group_id, visual_dependency_status, decision_basis y
ground_truth, se conservan en la tabla auxiliar. Para el artefacto real se indica
explícitamente `--membership-table visual_dependency_membership.csv`.

**Productores normalizados.** Se conserva el contrato anterior, más estricto en
cuanto a identidad/checksums declarados:

- Una tabla CSV o Parquet seleccionada por `--membership-table` (por defecto
  `membership.csv`), con `sequence_id` y `visual_dependency_group_id`.
- Metadata con `artifact_kind` y `artifact_id` declarados por el productor;
  `sequence_set_id` coincidente; `ground_truth=false`;
  `review_status="confirmed_manual_review"`;
  `manual_confirmation_complete=true`; y `output_checksums` con SHA256 de la
  tabla y de todos sus outputs declarados. Si existen `split_created` o
  `sequence_instances_merged`, deben ser false.

En esta rama la identidad externa se conserva sin inventar su algoritmo y se
verifican todos los checksums declarados. El kind confirmado de Hypatia con una
versión desconocida falla; nunca se redirige a esta rama como fallback.
Los tests reproducen el esquema informado desde Hypatia con datos sintéticos;
no equivalen a ejecutar la calibración sobre sus archivos reales.
El responsable reportó posteriormente una ejecución real exitosa en Hypatia:
`review init` completó y `review verify` devolvió `quality_valid=true`,
`source_bound=true`, `all_candidate_occurrences_preserved=true` y
`one_decision_per_query=true`, con `ground_truth=false`,
`confirmed_matches_created=false` y `split_created=false`. Esto valida la
compatibilidad ejercitada del adaptador v1; no significa que todos los estratos
futuros hayan sido revisados ni constituye una evaluación representativa.
Los artefactos reales no se revalidaron en este checkout.
La confirmación de **membresías de grupos** no confirma enlaces etiquetado→grupo.
La combinación futura de restricciones visuales y exactas no se ejecuta aquí.

Las imágenes son fuentes read-only: un ZIP etiquetado mediante
`--labeled-images-archive` o un directorio mediante `--labeled-images-root`
(exactamente uno), y un directorio de muestras mediante `--video-images-root`.
Se reutiliza `ImageSource`: rutas declaradas y comprobación SHA256 al leer cada
imagen mostrada. No se extraen ZIPs ni se cambian imágenes, labels o manifests.

## Flujo CLI

Ejemplo de Bash con placeholders; en PowerShell usar una línea o backticks:

```bash
uv run flir-pipeline linkage review init \
  --calibration-sample reports/linkage/calibration_sample.csv \
  --linkage artifacts/linkage/LINKAGE_ID \
  --labeled-manifest artifacts/manifests/LABELED.parquet \
  --sequence-set artifacts/sequences/sets/SEQUENCE_SET_ID \
  --visual-dependencies reports/sequence_diagnostics/visual_dependency_validation_confirmed_v1 \
  --membership-table visual_dependency_membership.csv \
  --labeled-images-archive /external/labeled-images.zip \
  --video-images-root /external/video-samples \
  --context-seconds 3 --output reports/linkage/manual_calibration

uv run flir-pipeline linkage review record INIT_DIRECTORY \
  --decisions reports/linkage/manual_decisions.csv \
  --reviewer REVIEWER_ID --source "Revision humana de contact sheets"

uv run flir-pipeline linkage review summary REVISION_DIRECTORY

uv run flir-pipeline linkage review verify REVISION_DIRECTORY \
  --calibration-sample reports/linkage/calibration_sample.csv \
  --linkage artifacts/linkage/LINKAGE_ID \
  --labeled-manifest artifacts/manifests/LABELED.parquet \
  --sequence-set artifacts/sequences/sets/SEQUENCE_SET_ID \
  --visual-dependencies reports/sequence_diagnostics/visual_dependency_validation_confirmed_v1 \
  --membership-table visual_dependency_membership.csv
```

`init` imprime un directorio cuyo nombre es `calibration_id`. Copiar `review.csv`
fuera de ese directorio y editar únicamente `manual_decision` / `manual_notes`.
También se puede importar un CSV parcial con las dos claves de consulta/grupo
y esas dos columnas manuales. Valores permitidos: `supported`, `ambiguous`,
`unsupported` o blanco. No se aceptan sinónimos ni conversiones de booleanos.
Una fila ausente deja intacta su decisión; una decisión en blanco presente puede
borrar una decisión previa y queda auditada como tal.

`record` crea otra publicación, nombrada por su ID de revisión, en el mismo
padre o bajo `--output`. Conserva las fuentes/evidencia, importa el CSV original
y registra reviewer, source, fecha UTC y valores anteriores/nuevos. Las
publicaciones previas permanecen intactas. `--timestamp-utc` permite aportar una
fecha UTC explícita; por defecto se registra la hora de importación. Fechas
anteriores a la decisión que se cambia se rechazan. Sin cambios, se reutiliza la
revisión previa. Una importación repetida con igual contenido y fecha es idéntica.
No editar archivos dentro de ninguna publicación completada.

## Evidencia y reproducibilidad

Se publica `review.csv` con una fila por consulta/grupo, `initial_review.csv`
siempre en blanco, `decision_history.json`, los CSVs importados, `summary.json`
y `metadata.json` como marcador final. Las tablas auxiliares conservan la muestra,
candidatos, todas sus ocurrencias, todas las ocurrencias etiquetadas con sus
splits históricos/conflictos de anotación, membresías y contexto temporal.
`candidate_details_json` y `contact_sheets_json` mantienen los joins desde cada
fila de revisión. No se selecciona una ocurrencia como identidad definitiva.

Las imágenes se organizan por estrato/grupo propuesto/consulta. Hay una contact
sheet por **cada ocurrencia candidata**, con contexto ±3 segundos muestreados
por defecto (configurable 0–30). La ventana se limita al mismo video fuente,
puede cruzar límites de secuencia/grupo y muestra ambas identidades. Las
ocurrencias de grupos alternativos también permanecen visibles. La imagen
etiquetada seleccionada por frame_id es solo una representación visual; todas
sus ocurrencias y metadatos sobreviven en la tabla auxiliar.

`index.html` permite inspeccionar las sheets. La identidad de calibración liga
fuentes, protocolo y versión de Pillow; la identidad de publicación además liga
los bytes de evidencia renderizada y eventos manuales. Fechas de ejecución no
cambian la calibración; las fechas de decisiones sí forman parte de su historia.
Los tests sintéticos comprueban que dos ejecuciones de `init` generan PNGs
idénticos en el mismo entorno de Pillow. Esto es una prueba de generación;
no describe lo que hace `verify`.

`verify` reconstruye los joins/contextos desde las fuentes, valida checksums,
identidades, una fila por consulta/par, vocabulario y denominadores, y reproduce
toda la historia desde los CSVs importados. Comprueba que la revisión inicial
esté en blanco y que no haya confirmación ni split oculto. Las imágenes generadas
se verifican por su checksum ligado a la identidad. `verify` comprueba evidencia
almacenada y joins ligados a las fuentes; **no vuelve a renderizar las contact
sheets desde las imágenes originales**, no abre esas imágenes ni hace una nueva
evaluación visual.

## Interpretación de los resúmenes

`summary` valida la publicación y presenta conteos/tasas globales, por estrato,
por grupo visual y por la intersección estrato×grupo. La unidad es la consulta,
no el número de candidatos u ocurrencias. El denominador de todas las tasas es
el total de consultas de la celda, **incluidos los blancos**. Se informa además
el número de grupos distintos y pendientes: blancos + ambiguos.

Son estadísticas descriptivas de una muestra estratificada y revisada
manualmente. No son accuracy, estimaciones representativas de rendimiento ni
ground truth. `supported` no afirma coincidencia exacta de frame, identificación
exacta de secuencia, ausencia de leakage o existencia de un split leakage-safe.
La ejecución y revisión reales en Hypatia permanecen separadas de los tests
offline sintéticos de infraestructura.

## Agregar revisiones inmutables

`aggregate` consume exclusivamente directorios ya publicados por `review init`
o `review record`. Cada entrada debe pasar el contrato completo de `review verify`,
incluyendo sus fuentes originales y el replay de importaciones manuales. No basta
un recibo anterior ni la suma de summaries. Una revisión inicial en blanco es una
entrada válida: sus casos siguen sin resolver. Repetir el mismo ID de revisión
en los argumentos es un error explícito.

Las revisiones no contienen rutas absolutas de sus fuentes. Por ello se requiere
`--source-map`, un JSON con una entrada por `calibration_id` (obtenido de su
`metadata.json`). Varias revisiones de la misma calibración comparten esa entrada.
Ejemplo con placeholders; las rutas relativas se resuelven desde el JSON:

```json
{
  "calibrations": {
    "CALIBRATION_ID_A": {
      "calibration_sample": "sample_a.csv",
      "linkage": "../../artifacts/linkage/LINKAGE_ID",
      "labeled_manifest": "../../artifacts/manifests/LABELED.parquet",
      "sequence_set": "../../artifacts/sequences/sets/SEQUENCE_SET_ID",
      "visual_dependencies": "../sequence_diagnostics/visual_dependency_validation_confirmed_v1",
      "membership_table": "visual_dependency_membership.csv"
    }
  }
}
```

Añadir la entrada de `CALIBRATION_ID_B` con su muestra y fuentes cuando corresponda.
Los seis campos son obligatorios, incluyendo la selección explícita de membresía.
No se necesitan las imágenes originales para verificar/agregar: se comprueban los
bytes de las contact sheets almacenadas, sin volver a renderizarlas.

```bash
uv run flir-pipeline linkage review aggregate REVISION_A REVISION_B \
  --source-map reports/linkage/review_sources.json \
  --output reports/linkage/manual_calibration_aggregates

uv run flir-pipeline linkage review aggregate-verify \
  reports/linkage/manual_calibration_aggregates/AGGREGATE_ID
```

Se publica bajo `--output/AGGREGATE_ID`. Su identidad determinista liga los IDs de
revisión/calibración y SHA256 de **todos** los archivos de cada revisión, incluido
`metadata.json`, PNGs e importaciones. El orden de argumentos y las rutas de
transporte no cambian la identidad; cambiar bytes de una revisión sí la cambia.
Una publicación existente solo se reutiliza después de verificarla.

El dominio de grupos debe ser idéntico: mismo set de secuencias y mismos bytes de
la publicación de membresías visuales, incluida la tabla seleccionada. No se
comparan IDs de grupo de productores diferentes como si fueran equivalentes.
Las muestras pueden diferir; esto no supone intercambiabilidad estadística.

La unidad agregada es **labeled_content_id + proposed_visual_dependency_group_id**.
Una consulta con dos grupos propuestos en calibraciones diferentes son dos pares;
el `query_count` agregado cuenta pares, nunca ocurrencias candidatas. Para un par
repetido se requiere la misma decisión literal y evidencia compatible: todas las
fuentes originales excepto el CSV de selección de la muestra, configuración de
revisión/renderizado y detalles de candidatos/ocurrencias idénticos. Estrato,
metadata de muestreo, notas y autoría pueden diferir y se conservan por separado.
No se aplica tolerancia de cosenos, votación, selección de la revisión más reciente
ni decisión automática. Cualquier discrepancia de decisión, **incluido blanco
frente a una decisión**, detiene la publicación con el par y revisiones implicados.
Una corrección requiere una revisión manual explícita y escoger las revisiones
apropiadas en la siguiente agregación; el agregado nunca modifica las entradas.

Outputs:

| Archivo | Unidad y procedencia |
|---|---|
| `source_observations.parquet` | Todas las filas originales, con notas, reviewer/source/timestamp, evidencia y IDs de revisión/calibración |
| `pooled_reviews.parquet` | Una decisión original sin modificar por par compatible; listas de revisiones, calibraciones y estratos |
| `duplicate_pairs.parquet` | Pares repetidos explícitos, cantidad de observaciones y todas sus revisiones |
| `descriptive_counts.parquet` | Conteos/tasas por revisión, estrato, grupo, estrato×grupo, revisión×estrato×grupo y total descriptivo |
| `source_revisions.json` | Metadata completa de cada fuente, fingerprints originales e historia manual íntegra |
| `source_locations.json` | Rutas operacionales para volver a localizar cada revisión y sus fuentes; no son la identidad científica |
| `summary.json` | Resúmenes descriptivos, duplicados, casos sin resolver y denominadores explícitos |
| `metadata.json` | Marcador final, identidad, checksums, procedencia de ejecución y semántica |

Todos los flags `ground_truth`, `confirmed_matches_created`, `split_created` y
`automatic_confirmation` permanecen false. El agregado no asigna secuencias, no
crea enlaces confirmados ni combina cosenos CLIP/DINOv2.

Los duplicados compatibles cuentan una sola vez por celda. Si un par pertenece a
dos estratos originales, aparece en ambos, pero solo una vez en el total. Por ello
**los conteos por estrato no son necesariamente aditivos**. Se conservan además
los summaries de cada revisión con sus denominadores originales. Blancos y
ambiguos permanecen sin resolver. El campo `overall_pooled_descriptive` presenta
conteos/tasas descriptivos del conjunto de calibración revisado, no estimaciones
representativas de accuracy o precision; no pondera muestras como intercambiables.

`aggregate-verify` vuelve a verificar cada revisión y sus fuentes, reproduce sus
eventos y reconstruye todas las tablas/resúmenes/procedencias. Rechaza alteraciones
aunque se hayan actualizado los checksums de las tablas derivadas. No sustituye
una nueva revisión visual. Las revisiones y sus fuentes deben seguir disponibles.
Si se trasladan sin cambiar bytes, preparar una copia externa de
`source_locations.json` con las rutas nuevas y pasar `--locations COPIA.json`;
no editar la publicación congelada. Las rutas de estos JSONs son operacionales
y privadas: mantenerlos fuera de Git, igual que todos los outputs de revisión.

La agregación tiene validación local sintética. El éxito real de `init/verify`
reportado en Hypatia no implica que se haya ejecutado allí la agregación ni que
la cobertura de calibración esté completa.
