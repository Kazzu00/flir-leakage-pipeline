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
