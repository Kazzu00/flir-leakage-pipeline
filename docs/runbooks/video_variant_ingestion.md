# Ingesta de variantes visuales externas

Esta etapa de comprensión/preparación audita imágenes externas sin ejecutar la
transformación que las produjo. El contrato es `video_variant_ingestion_v1`;
el manifest científico utiliza `flir_video_variant_occurrences_v1`.
No consume anotaciones históricas, crea labels, identifica secuencias, ajusta
clustering, genera splits ni entrena detectores. Los ZIP son shards físicos,
nunca videos o secuencias. La variante `hud_reduced` es un nombre configurable:
no significa eliminación completa del HUD.

## Estado y autorización de ejecución

Infraestructura implementada y validada con fixtures pequeñas, sintéticas y
offline. La ejecución sobre los PNG/video originales y la alineación exhaustiva
están pendientes. Los tres contrastes visuales reportados por el responsable
pueden documentarse como observaciones externas; no verifican todos los frames.
No se incorporaron datos, hashes reales ni exports científicos a esta etapa.

## Cuatro identidades independientes

| Identidad | Receta y significado |
|---|---|
| `content_id` / `image_sha256` | SHA-256 de los bytes descomprimidos de la imagen. No es hash de píxeles ni identidad semántica. |
| `observation_id` | SHA-256 del JSON canónico `[manifest_version, collection_id, series_id, observed_index]`. Clave de observación nombrada; en un conflicto no garantiza una ocurrencia única. |
| `frame_id` | SHA-256 del JSON canónico `[manifest_version, collection_id, series_id, observed_index, image_sha256]`, únicamente si el índice identifica una entrada dentro de la serie y sus bytes son legibles. |
| `entry_id` | SHA-256 del JSON canónico `["zip-entry-v1", archive_key, archive_sha256, member_ordinal, original_member_name]`. Identidad física, independiente de la identidad científica. |

El JSON canónico usa claves ordenadas, separadores `,`/`:`, UTF-8, escape ASCII
y rechaza NaN/Inf. Las listas anteriores fijan el orden de los campos.
Los índices se leen con el grupo nombrado `index` de una regex de configuración
aplicada mediante `fullmatch` al basename original. Son enteros decimales ASCII
entre cero y el máximo int64. Los ceros iniciales no distinguen observaciones.

Las pruebas verifican que cambiar el root, nombres y compresión de ZIP, orden
de entradas, cantidad de shards y carpetas internas **conserva** `frame_id`,
el manifest, `dataset_id` y `dataset_variant_id`, siempre que se preserven
colección, serie, índices, bytes y definición de variante y que los selectores
sigan atribuyendo cada imagen a la misma serie. Renombrar índices/series,
cambiar bytes o cambiar la declaración de variante tiene otra semántica.

La identidad del dataset reutiliza `dataset_id_from_manifest`, sin modificar
el algoritmo histórico: versión más tuplas ordenadas de `frame_id`,
`image_sha256`, `label_sha256`. Se reutiliza `dataset_variant_v1` con el checksum
exacto del manifest y la transformación declarada. No se inventa un dataset padre
a partir de la presencia de un MP4; el parent permanece null en esta fase.

### Ocurrencias ambiguas

Se conserva **una fila por entrada física de imagen**, incluso si dos entradas
comparten nombre, índice o bytes. La lectura usa `ZipInfo`, no búsqueda por nombre.
`orig_filename` preserva el spelling del ZIP antes de las normalizaciones de
`ZipInfo.filename` en Windows. Nunca se normaliza traversal para hacerlo aceptable.

Índices repetidos dejan `identity_status=ambiguous_index` y `frame_id=null` en
todas las entradas implicadas, aunque sus bytes difieran. Selectores solapados,
imágenes sin serie, índices no interpretables y bytes no legibles también quedan
sin identidad científica resuelta. No se agrega un sufijo arbitrario ni se escoge
una copia como representante científico.

El contrato existente exige `frame_id` únicos y no nulos. Si alguna ocurrencia
es ambigua/no identificable, o algún ZIP no pudo inventariarse íntegramente,
la publicación conserva su auditoría pero **no incluye `manifest.parquet` ni
`variant.json`**, y `dataset_id`/`dataset_variant_id` son null. Esta es una
limitación explícita de v1; resolver multiplicidad lógica exige evidencia de
identificación adicional en un protocolo posterior. No se pierde ninguna
ocurrencia inventariada al aplicar esta restricción.

## Configuración

Véase el [ejemplo genérico](../../configs/data/video_variant_ingestion.example.yaml).
Las rutas son POSIX relativas a `input-root`; el root proviene de CLI o
`FLIR_DATA_ROOT`/`.env`. La configuración enumera archivos: no hay descubrimiento
recursivo. Puede declarar cualquier cantidad de shards, videos y series.
Un ZIP puede contener varias series, mediante `member_pattern` disjuntos;
cada imagen debe coincidir con una sola serie.

El rango esperado requiere ambos extremos y un paso positivo. Se detectan
faltantes/duplicados globalmente por serie, independientemente del shard.
Los faltantes se almacenan como intervalos compactos sin construir listas
proporcionales al rango esperado. Sin extremos declarados, la auditoría es
`observed_interior_only`: no prueba ausencia de pérdidas antes/después de los
extremos observados. El conteo de contenidos se calcula sobre bytes disponibles.
`duplicate_records` incluye todos los miembros de contenidos repetidos;
`redundant_records` cuenta las copias adicionales.

El CSV auxiliar queda ligado por checksum, opcionalmente comparado con un hash
esperado. En v1 **no se interpretan sus columnas ni se usan sus conteos**:
los ZIP son la autoridad. No se infiere su esquema ni se inventan reconciliaciones.
Los errores de lectura y decode permanecen explícitos; una imagen corrupta con
bytes legibles aún posee identidad exacta, pero falla integridad de decode.

## Correspondencia temporal y evidencia

`video_key` identifica una referencia declarada; `source_video_id` liga los bytes
del archivo, sin identificar escenas. El MP4 tiene SHA-256 y metadata declarada
separada de metadata observada con ffprobe. Ninguno prueba que un PNG proceda
de un frame nativo concreto.

La tabla `alignment_candidates` distingue índice observado, video candidato,
tiempo candidato y regla. `candidate_time`, si se declara, calcula:

```text
candidate_timestamp_seconds = (observed_index - index_origin) / fps + offset_seconds
```

FPS/origen/offset son parámetros explícitos, nunca defaults deducidos de un
nombre de video o de la cantidad de imágenes. Tiempos negativos/no finitos
quedan null y generan una incidencia. `alignment_status` es `unverified`,
`candidate` o `ambiguous`; `verification_scope` permanece `none`.
`verified_native_frame_index` y `verified_timestamp_seconds` permanecen null.

Las observaciones son `externally_reported`, conservan descripción, índice/tiempo
reportados y referencias opcionales a evidencia ligada por checksum. Reviewer,
fecha e índice pueden ser null; no se inventan. No se extrapola una observación
al resto de la serie ni se convierte una confianza externa en probabilidad.
Esta fase no tiene importador de decisiones de alineación verificada; añadirlo
requerirá un contrato separado con alcance y procedencia de decoder/PTS explícitos.

## Publicación

| Archivo | Unidad / uso |
|---|---|
| `config.json`, `sources.json` | Configuración normalizada y fingerprints de los insumos declarados. |
| `archives.parquet` | Un shard, checksum, tamaño, conteos y error de inventario. |
| `entries.parquet` | Cada entrada, incluso directorios y archivos auxiliares; decode de imágenes seleccionadas. |
| `video_sources.parquet` | Cada referencia de video y procedencia de su metadata. |
| `occurrences.parquet` | Todas las entradas físicas de imagen y su identidad lógica resuelta o ambigua. |
| `index_issues.parquet` | Conflictos, faltantes compactos y errores de reglas candidatas. |
| `alignment_candidates.parquet` | Una fila por ocurrencia física; sin alineación verificada. |
| `observations.parquet` | Evidencia externa explícitamente suministrada; tabla vacía tipada si no se aporta. |
| `manifest.parquet`, `variant.json` | Solo cuando la identidad científica de todas las imágenes inventariadas es atribuible. |
| `integrity.json` | Integridad, cobertura/alcance, límites de uso y conteos; sin métricas de leakage. |
| `metadata.json`, `receipt.json` | Contrato, identidad, checksums, implementación y ejecución. |

El manifest científico excluye referencias ZIP: se recuperan mediante
`frame_id → occurrences → entry_id`. Esta separación mantiene también su
checksum estable en el reempaquetado probado, no solo el dataset ID.
Ausencia de anotaciones: `label_exists=false`, `label_sha256=""`,
`original_split=""`; no hay `label_empty` ni conteos de objetos fabricados.

El artifact ID liga configuración, snapshots de fuentes, checksums exactos de
salida y hashes de implementación. Puede cambiar con reempaquetado aunque las
identidades científicas permanezcan iguales. La ejecución registra fecha UTC,
Python/dependencias, commit, worktree dirty, checksum del YAML original y versión
de ffprobe cuando la inspección se solicita y el binario informa su versión.

Solo se publica en un root disjunto del input y, dentro del checkout, bajo
`artifacts/` o `reports/` ignorados. Se limita tamaño descomprimido por entrada,
ratio de compresión, total declarado por archivo, número de entradas y
dimensiones/píxeles antes de decode. Se mantiene una imagen acotada en memoria,
no toda la colección, y tablas proporcionales al inventario.

Se comprueban checksums de fuentes antes y después de leerlas. Un lock exclusivo
impone un escritor por output-root y no se rompe automáticamente. Staging y
destino final están en el mismo filesystem. Se escriben metadata/receipt al
final, se verifica staging y se promueve el directorio mediante rename.
Una excepción conserva el staging `.partial`; nunca se consume como publicación
final ni se borra automáticamente. Solo se elimina el staging redundante de la
invocación actual si existe un final equivalente e íntegro. Un lock de un proceso
terminado exige inspección manual. No hay resume de ingesta en v1.

## CLI

Ejemplos para ejecución posterior autorizada; las variables son localizadores.

```powershell
uv run --no-sync flir-pipeline data video-variant-ingestion build `
  --config "$INGESTION_CONFIG" --input-root "$INPUT_ROOT" `
  --output-root artifacts/video_variants

uv run --no-sync flir-pipeline data video-variant-ingestion summary "$ARTIFACT"
uv run --no-sync flir-pipeline data video-variant-ingestion verify "$ARTIFACT"
uv run --no-sync flir-pipeline data video-variant-ingestion verify "$ARTIFACT" `
  --input-root "$INPUT_ROOT"
```

`build` publica una auditoría y puede finalizar correctamente con
`integrity_valid=false`; debe revisarse el JSON de salida. `ingestion_completed`
significa que terminó esa auditoría, no que desaparecieron sus conflictos.
`summary` verifica tablas almacenadas, sin leer ZIP originales. `verify` sin
root comprueba checksums, tipos, IDs y replay de evidencia derivada; declara
`source_bound=false`. Con root verifica todos los insumos y vuelve a leer/decode
las imágenes: cuesta una ingesta completa y declara `source_bound=true`.
No repite ffprobe ni verifica científicamente su metadata o la correspondencia.

`--ffprobe-bin ffprobe` en build opta por inspección de metadata del primer stream
con el helper existente; no extrae ni alinea frames. Un error queda registrado.
Los tests de CLI usan `CliRunner`. Si Windows bloquea el console script, el
entrypoint Python existente sigue disponible:

```powershell
uv run --no-sync python -c "from flir_pipeline.cli import app; app()" `
  data video-variant-ingestion --help
```

## Puente de features y consumidores pendientes

`integrity_valid=true` no significa alineación verificada ni dataset listo para
detector: `alignment_verified=false`, `labels_available=false`,
`detector_ready=false`, `consumer_adapter_available=false` son explícitos.

El lector multishard y el puente al motor existente de features están implementados
y probados sintéticamente. `features extract --video-variant-ingestion` requiere
el manifest original y un `--input-root` explícito. Verifica las fuentes antes
de cargar el modelo, resuelve por ledger/ordinal, deduplica solo por contenido y
conserva cada ocurrencia. Véanse los [comandos de smoke, resume y verificación](features.md#features-desde-publicaciones-multishard).
La disponibilidad de este consumidor es estado del código; los flags, checksums
y receipts de publicaciones de fases anteriores no se reescriben.

Los consumidores temporales siguen pendientes. **No pasar este manifest a los
comandos históricos de similarity, secuencias, clustering, splitting o detection**:
no es `flir_video_samples_v1` ni grilla de muestreo verificada. La extracción real
de embeddings de esta variante permanece pendiente; no se infiere eliminación
completa del HUD a partir del nombre externo.

Organización v2 tampoco incorpora automáticamente este kind mediante
`--evidence-root`. Su extensión/versionado y el soporte multishard de previews
son posteriores. El contrato pendiente de clustering, el frontend y los exports
existentes permanecen ajenos a esta implementación.

## Validación local de esta primera fase

```powershell
uv run --no-sync ruff check .
uv run --no-sync python -m pytest tests/test_video_variant_ingestion.py tests/test_zip_image_collection.py -q
uv run --no-sync python -m pytest tests/test_identity.py tests/test_inventory.py tests/test_manifest.py tests/test_video_frames.py -q
git diff --check
```

Resultado: **60 passed** en pruebas nuevas y **60 passed** en regresión relacionada.
Ruff global, formato de los siete archivos Python nuevos y ayuda CLI aprobaron.
Los tests incluyen reempaquetado con checksum de manifest conservado, duplicados
ambiguos, series solapadas, límites/ZIP64, decode, cambios durante staging,
publicación interrumpida y replay de fuentes. Solo produjeron fixtures/artifacts
sintéticos bajo directorios temporales de pytest. No se ejecutó la suite completa,
ingesta real, extracción de features, GPU, experimento posterior ni export.
