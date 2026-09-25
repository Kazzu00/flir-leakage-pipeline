# Data audit and canonicalization

Run from the repository root with Python 3.11 and `FLIR_DATA_ROOT` pointing to
external read-only originals. Use [.env.example](../../.env.example) as a template
for your ignored `.env`. Public clones do not include the source archives.

```powershell
uv sync --locked --extra dev
uv run flir-pipeline data inventory --inspect-archives --hash-members
uv run flir-pipeline data compare-archives
uv run flir-pipeline data build-manifest
uv run flir-pipeline data validate-labels
uv run flir-pipeline data manifest-summary data/manifests/flir_canonical_candidate_v1.parquet
```

These commands reproduce preparation from originals; they are not required merely
to read existing reports. Use each command's `--help` for explicit input/output
paths. Originals are streamed/read in memory; no source ZIP is rewritten.

The canonical manifest lives under `data/manifests/`, with local inventory,
lineage, label QA and duplicate reports under `reports/`. Follow the directories
returned by the CLI. Preserve prior inventories when archive availability differs.
An absent historical archive is not evidence that canonical coverage is incomplete;
check the actual manifest sources and record the difference.

Validate occurrence/content uniqueness, source hashes, matched versus orphan
labels, exact overlap and conflicting annotations. Keep `original_split` and
all historical occurrences. Never silently repair labels or flatten duplicate
memberships. [Recorded data quality](../analysis/data_quality.md) separates the
canonical annotation universe from the complete labels archive.

Next: [feature extraction and diagnostics](features.md). See [data model](../data_model.md)
and [safety](../data_safety.md) for identity and privacy invariants.

## Muestreo reproducible de videos fuente

`data extract-video-frames` implementa una etapa de **preparación** para los
videos fuente. Además de las pruebas sintéticas, se confirmaron en **Hypatia**
un smoke real con FFmpeg y la extracción completa del job **737719**:
**3 videos → 9648 JPEGs a 1 FPS**, con validación operacional de sus metadatos.
La [evidencia real](#evidencia-real-confirmada-en-hypatia) se registra abajo.
No cambia `data inventory` ni requiere una nueva
dependencia Python para video. FFmpeg y ffprobe son ejecutables externos; se
requiere una versión de FFmpeg que soporte `-fps_mode` (5.1 o posterior).

Ejemplo genérico para Linux/Hypatia, desde la raíz del repositorio:

```bash
uv run flir-pipeline data extract-video-frames \
  --videos-root /ruta/a/videos-fuente \
  --output-root /ruta/a/derivados/flir-frames-1fps \
  --sample-fps 1.0 \
  --ffmpeg-bin "$HOME/.conda/envs/flir-video/bin/ffmpeg" \
  --ffprobe-bin "$HOME/.conda/envs/flir-video/bin/ffprobe" \
  --jpeg-quality 2 \
  --no-overwrite
```

Las rutas son ejemplos que deben sustituirse. Los dos directorios deben ser
disjuntos: ninguno puede contener al otro. Los videos fuente permanecen
**read-only**; el comando no escribe dentro de su directorio, no toca ZIPs,
labels ni particiones históricas. No usa `FLIR_DATA_ROOT` implícitamente.
Guardar los derivados fuera del repositorio o bajo un directorio ignorado como
`artifacts/`; JPEGs, Parquet, JSON y hashes reales no se versionan.

| Parámetro | Significado / default |
|---|---|
| `--videos-root` | Obligatorio; búsqueda recursiva de `.mp4`, `.mov`, `.avi`, `.mkv`, sin distinguir mayúsculas en la extensión |
| `--output-root` | Obligatorio; destino de JPEGs y metadatos, separado de las fuentes |
| `--sample-fps` | Frecuencia positiva y finita; default `1.0` |
| `--ffmpeg-bin`, `--ffprobe-bin` | Ruta al ejecutable o nombre en `PATH`; defaults `ffmpeg`, `ffprobe`; se expande `~` |
| `--jpeg-quality` | Cuantizador JPEG de 1 a 31; default `2`, alta calidad; menor valor implica mayor calidad, no un porcentaje |
| `--overwrite / --no-overwrite` | Default `--no-overwrite`; reemplazo restringido a artefactos administrados y reconocidos |

### Muestreo y trazabilidad

Los videos se ordenan por su ruta relativa POSIX, con comparación sensible a
mayúsculas. Cada `video_id` es `video_` más los primeros 16 caracteres del SHA256
de esa ruta UTF-8, incluida su extensión. Así se distinguen nombres iguales en
directorios distintos y se conserva el ID al trasladar el directorio raíz.
Renombrar un video cambia su ID; reemplazar sus bytes conserva ese ID de ruta
pero cambia el `source_sha256` registrado. **No es un `content_id`, un `frame_id`
histórico ni un identificador de secuencia.** Una colisión de IDs se rechaza.

ffprobe inspecciona `v:0` mediante JSON; se extrae ese mismo stream con un único
proceso FFmpeg por video. No se carga el video completo en memoria. La cadena es:

```text
setpts=PTS-STARTPTS,fps=fps=<sample_fps>:start_time=0:round=near:eof_action=pass
```

Se resta el primer PTS del video y se genera una grilla desde cero. `fps` puede
descartar o repetir frames; `-fps_mode passthrough` evita un segundo remuestreo.
`eof_action=pass` explicita el tratamiento del último intervalo. Los nombres
empiezan en `frame_000000.jpg`; las filas se construyen a partir de los archivos
realmente producidos, no de una cuenta inferida desde la duración. Una salida
vacía o con numeración discontinua se rechaza. Se usa el encoder MJPEG con un
hilo, se desactiva la autorrotación para conservar las dimensiones codificadas
y se omite la copia de metadatos del contenedor a los JPEGs. Véanse los filtros
oficiales [fps](https://ffmpeg.org/ffmpeg-filters.html#fps) y
[setpts](https://ffmpeg.org/ffmpeg-filters.html#setpts_002c-asetpts), y la
[documentación de ffprobe](https://ffmpeg.org/ffprobe.html).

```text
<output-root>/
  video_<hash-de-ruta>/
    frame_000000.jpg
    frame_000001.jpg
    ...
  frames.parquet
  summary.json
```

`frames.parquet` contiene una fila por JPEG, ordenada por video y sample_index:

| Campo | Interpretación |
|---|---|
| `video_id` | ID estable de la ruta relativa del video |
| `source_video` | Ruta relativa a `videos-root`, con `/` |
| `sample_index` | Posición de la muestra, entera y desde cero |
| `timestamp_seconds` | `sample_index / sample_fps`; tiempo nominal relativo de la grilla |
| `source_fps` | `avg_frame_rate` válido, o `r_frame_rate` como fallback; admite fracciones como `30000/1001`; null si ambos son desconocidos |
| `source_frame_index_estimate` | `floor(timestamp_seconds * source_fps + 0.5)`; estimación del índice nominal desde cero, redondeada al entero más cercano con empates hacia arriba; null sin FPS válido |
| `source_width`, `source_height` | Dimensiones codificadas reportadas para el stream de video |
| `source_duration_seconds` | Duración del stream, con fallback a la del contenedor; null si ninguna está disponible |
| `source_nb_frames` | Número de frames reportado por ffprobe; null si falta o es `N/A`, sin estimarlo desde la duración |
| `codec_name` | Códec fuente reportado por ffprobe |
| `sample_fps` | Frecuencia de muestreo solicitada |
| `image_path` | Ruta relativa a `output-root`, con `/` |

**`source_frame_index_estimate` NO es un índice exacto obtenido del decoder.**
Con frame rate variable, discontinuidades o redondeo, no identifica necesariamente
el frame que FFmpeg eligió. No se acota artificialmente usando `nb_frames`.
`timestamp_seconds` tampoco es el PTS original del frame seleccionado ni un
timestamp real de captura; no permite sincronizar videos ni inferir fechas.
Los valores desconocidos quedan nulos. La duración del contenedor puede incluir
otros streams; `duration_field` registra cuándo se utilizó ese fallback.

`summary.json` registra `processed_videos`, `total_frames`, `sample_fps`,
`jpeg_quality`, `read_only_source: true`, versiones FFmpeg/ffprobe y Python,
SHA256 del módulo implementador, filtro temporal y políticas de extracción.
Por video conserva la ruta/ID, SHA256 de sus bytes, dimensiones, códec,
duración, FPS, fracciones originales, campo elegido para FPS/duración,
`source_nb_frames` y `extracted_frames`. No guarda rutas absolutas ni la línea
de comandos local. También identifica productor/esquema y el checksum del
Parquet para reconocer una salida previa antes de reemplazarla.

Los IDs, orden y grilla son deterministas. Para reproducir los JPEGs deben
preservarse los mismos bytes fuente, parámetros, implementación y build de
FFmpeg; no se promete igualdad byte a byte entre builds o plataformas diferentes.
Los hashes y las versiones son evidencia de procedencia, no validación visual.

### Sobrescritura y fallos

Sin `--overwrite`, la existencia de metadatos previos bloquea la ejecución antes
de escribir resultados. Los conflictos en las carpetas de todos los videos se
revisan antes de extraer. Con `--overwrite`, se comprueban productor, esquema,
checksum del Parquet y correspondencia de rutas/conteos con `summary.json`.
Un archivo arbitrario llamado `summary.json` o `frames.parquet` no se reemplaza.
Los enlaces que redirigen destinos se rechazan.

La ejecución prepara todos los JPEGs y metadatos en un directorio temporal
dentro de `output-root`. Si ffprobe o FFmpeg falla, se informa el video y el
stderr relevante, se limpia ese temporal y se conservan los resultados previos.
La publicación reemplaza solamente los archivos administrados; retira sus
frames obsoletos, incluidos los de videos que ya no están en la entrada, y
preserva archivos ajenos y directorios. Un `frame_*.jpg` que colisione con el
espacio de nombres administrado sin estar registrado se rechaza, no se borra.

Esto requiere espacio para la salida anterior y el nuevo muestreo completo.
Se admite **un único escritor por output-root**. La promoción final no es una
transacción atómica de múltiples archivos: una interrupción en ese punto deja
`.video-frames-publishing`, bloquea la reutilización y exige conservar la salida
para inspección y ejecutar en otra raíz. No se implementa checkpoint/resume;
los temporales de un proceso terminado abruptamente no se reutilizan ni se
borran en ejecuciones posteriores automáticamente.

### Evidencia real confirmada en Hypatia

Evidencia confirmada por el responsable del proyecto para la ejecución real de
`data extract-video-frames` en **Hypatia, Universidad de los Andes**. Esta
actualización registra el resultado recibido; no reejecuta la extracción.

| Entorno / ejecución | Valor confirmado |
|---|---|
| Python | 3.11.10 |
| FFmpeg / ffprobe | 9.0.2 / 9.0.2 |
| SLURM batch job | 737719 |
| State / ExitCode | COMPLETED / 0:0 |
| Elapsed | 00:15:28 |
| MaxRSS del batch | 175016K |
| stderr | Vacío |

Configuración: `sample_fps=1.0`, `jpeg_quality=2`, `autorotate=false`, `threads=1`.
Filtro de muestreo registrado:

```text
setpts=PTS-STARTPTS,fps=fps=1.0:start_time=0:round=near:eof_action=pass
```

Los tres streams fuente reportaron códec **h264**:

| Video fuente | Resolución | source_fps | source_duration_seconds | source_nb_frames | extracted_frames |
|---|---|---|---|---|---|
| `VideoCodefest_008-46min.avi` | 1280×720 | 30000/1001 ≈ 29.97003 | 2769.800367 | 83011 | 2770 |
| `video30min-11to22.mp4` | 1920×1080 | 30 | 660.0 | 19800 | 660 |
| `video_2hour.mp4` | 1280×720 | 30 | 6217.333333 | 186520 | 6218 |

Resultado y comprobaciones confirmadas:

- `processed_videos=3`, `total_frames=9648`.
- `frames.parquet`: 9648 filas y 9648 `image_path` únicos.
- Índices de muestra contiguos desde 0 en cada uno de los tres videos.
- `timestamp_seconds` coherente con `sample_index / 1 FPS`.
- Fuentes preservadas read-only.

Previamente se ejecutó un **smoke real de 5 segundos** sobre una fuente de
30 FPS y 150 frames fuente. La extracción a 1 FPS produjo 5 JPEGs, tiempos
`0,1,2,3,4` y `source_frame_index_estimate=0,30,60,90,120`; se validaron
`summary.json` y `frames.parquet`. Estos índices siguen siendo estimaciones
nominales, no índices exactos del decoder ni tiempos reales de captura.

La documentación versionada conserva esta evidencia agregada y portable.
`summary.json` mantiene la procedencia exacta local, incluidos los SHA256;
no se incorporan aquí hashes concretos ni rutas absolutas de Hypatia.
Las 9648 rutas únicas **no demuestran 9648 contenidos visuales únicos**:
el manifest occurrence/content y su deduplicación todavía no se han ejecutado
en Hypatia. El puente `build-video-manifest` conserva validación sintética.

### Límite metodológico

Muestrear frames **no equivale a identificar secuencias** y esta etapa **no crea
train/val/test**. Los tres videos procesados son fuentes, no tres
secuencias ya validadas. La unidad que eventualmente deberá mantenerse íntegra
entre particiones será la **secuencia visualmente relacionada**, que debe
identificarse y evaluarse posteriormente; no se impone que sea el video completo
ni el frame individual.

La salida prepara una representación temporal muestreada. El puente siguiente
define sus ocurrencias y contenidos exactos para reutilizar la extracción
CLIP/DINOv2. La alta correlación o repetición de frames muestreados no demuestra
leakage sin evaluar particiones.

La ejecución real valida operacionalmente la extracción reproducible. No valida
todavía identificación de secuencias, límites de escenas, embeddings de estas
9648 muestras, clustering de estos videos, particiones, eliminación de leakage
ni rendimiento del detector.

## Manifest de ocurrencias muestreadas

`data build-video-manifest` consume exactamente el `output-root` del muestreo.
No requiere FFmpeg, videos originales ni modelos. Ejemplo posterior para Hypatia,
**documentado, no ejecutado en esta tarea**:

```bash
uv run --no-sync flir-pipeline data build-video-manifest \
  --frames-root /ruta/a/derivados/flir-frames-1fps \
  --output data/manifests/flir_video_samples_v1.parquet \
  --report-output reports/video_manifest_1fps
```

Los tres parámetros son obligatorios. Manifest y reportes deben quedar **fuera**
de `frames-root`; los JPEGs y los metadatos de muestreo permanecen read-only.
Antes de usar este comando, confirmar que el escritor del muestreo ha terminado;
el job de muestreo 737719 ya está confirmado como COMPLETED. Esa confirmación no
constituye una ejecución de `build-video-manifest`: este comando todavía solo
tiene pruebas sintéticas y no se ha ejecutado en Hypatia.

El comando rechaza productor/esquema desconocido, checksum incorrecto del Parquet,
`.video-frames-publishing`, rutas no portables/traversal, escapes por symlink,
archivos ausentes, muestras temporales repetidas y discrepancias de videos,
conteos, grilla o propiedades respecto al summary. Admite N videos y conserva
propiedades opcionales desconocidas como nulas; no busca imágenes adicionales.

Lee cada JPEG, calcula SHA256 y valida decodificación con PIL. Cada muestra
conserva su `frame_id`; bytes iguales comparten `content_id`. Imágenes corruptas
permanecen en el manifest con `image_decode_valid=false` y `image_error`; **no se
excluyen silenciosamente**. `features extract` rechaza esos manifests para que
se inspeccione la causa antes de producir embeddings.

`reports/video_manifest_1fps/summary.json` registra dataset/version, conteos de
ocurrencias y contenidos, grupos duplicados, registros por video, fallos de
decodificación y sus rutas relativas, `read_only_source=true`, Git/implementación
y SHA256 de `frames.parquet`, summary de muestreo y manifest final. Antes de
consumir una pareja manifest/reporte tras una interrupción, comprobar
`manifest_parquet_sha256`: la publicación de ambos archivos no es atómica.

La identidad y los aliases de compatibilidad se especifican en
[data model](../data_model.md#sampled-video-occurrences-flir_video_samples_v1).
No se fabrican etiquetas, escenas ni train/val/test. El siguiente paso operativo
es [extraer features locales](features.md#features-de-frames-muestreados-locales);
la identificación de secuencias visuales sigue pendiente para estos videos.
