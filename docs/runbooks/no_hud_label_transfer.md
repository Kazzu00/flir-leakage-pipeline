# Transferencia candidata HUD/no-HUD v1

Etapa de preparación e integridad de anotaciones. No es un experimento de detector,
ni una aprobación de ground truth. Los 47 labels vacíos del test propuesto no son
la población sin HUD. Las fuentes se seleccionan explícitamente por ZIP.

## CLI

```powershell
uv run flir-pipeline data no-hud-label-transfer prepare `
  --source-root $env:FLIR_DATA_ROOT `
  --audit <auditoria-vigente> --proposal <propuesta-verificada> `
  --audit-evidence <evidencia-original-de-la-auditoria.json> `
  --archive <shard-1.zip> --archive <shard-2.zip> `
  --output <publicacion-externa-nueva> --gallery-limit 60

uv run flir-pipeline data no-hud-label-transfer verify `
  --publication <publicacion-externa> --replay
```

La auditoría debe ser exactamente la vinculada a la propuesta protegida.
Se reutilizan los verificadores de ambas publicaciones y los hashes de los ZIP
originales. Una auditoría antigua incompatible no se corrige silenciosamente.
Las salidas deben estar fuera del repositorio y separadas de las fuentes.
El recibo se escribe al final; una carpeta sin recibo es incompleta.

## Búsqueda y estados

Se decodifican todas las imágenes seleccionadas mediante lecturas ZIP, sin
extracción. Se registran SHA-256, formato, dimensiones, orientación EXIF,
localizador físico, índice nominal, errores y posibles etiquetas contiguas.
Un índice `frame_N` es una heurística, no una correspondencia con `video_N`.

La recuperación compara thumbnails centrales de gris, 32×24, centrados y
normalizados, mediante correlación. Devuelve tres contenidos fuente únicos por
destino y conserva todas sus ocurrencias/anotaciones. No utiliza split ni labels
para puntuar. Siempre devuelve vecinos cuando hay fuentes: **no existe un umbral
de identidad ni una declaración de ausencia basada en este ranking**. El margen
entre vecinos tampoco es probabilidad. La galería muestra los candidatos de mayor
score y todas las parejas con evidencia explícita; no es una muestra representativa.

Estados: `pending_review`, `verified`, `rejected`. Las imágenes ilegibles o con
localizador ambiguo se rechazan para esta preparación. Sin evidencia por pareja,
incluso la igualdad visual permanece pendiente. Una geometría no soportada queda
pendiente, pues no demuestra por sí sola que sean fotogramas distintos.

## Evidencia y límites de transferencia

La opción `--evidence` admite JSON con una lista `pairs`. Cada entrada requiere:

- `target_id` del inventario y `source_frame_id` de la auditoría.
- `reviewer` y `evidence_reference`: responsable y procedencia de la atribución.
- `source` y `target`, cada uno con `image_sha256`, `video_sha256` y
  `native_frame_index`. El video y el índice nativo deben coincidir.
- `modified_regions_complete: true` y `modified_regions`: rectángulos normalizados
  `[xmin, ymin, xmax, ymax]` de todas las zonas modificadas, documentados externamente.

La evidencia queda ligada por SHA-256 y conservada por pareja. **Es procedencia
externa atribuida; la herramienta no decodifica el video para demostrarla de forma
independiente.** No se debe fabricar este archivo a partir de nombres o scores.
La confirmación general de que existen fotogramas etiquetados no aporta ese mapping.

El raster fuente debe coincidir exactamente con el destino fuera de las regiones
declaradas, después de identidad o escalado uniforme completo con bilinear de PIL.
La orientación debe ser 1, el aspect ratio idéntico y al menos 75% del destino
debe quedar sin máscara. Estos son límites conservadores de implementación, no
umbrales calibrados sobre el dataset. Las regiones modificadas no se estiman para
hacer pasar una pareja. JPEG distinto, inpainting fuera de máscara, cambios de
color, recorte, desplazamiento, giro y otras interpolaciones quedan pendientes.

En esas dos transformaciones soportadas las coordenadas YOLO normalizadas no
cambian. Se recuperan y verifican los bytes originales, se validan clases y cajas
y se copian sin redondeo ni clipping. Cajas que intersectan regiones modificadas
se señalan para revisión. Conflictos entre anotaciones del mismo contenido o
posibles labels preexistentes en destino bloquean la copia automática.
No se crean objetos ni se corrigen etiquetas. Las etiquetas vacías conservan su
significado original, sin convertir imágenes sin anotación en negativos.

## Protección y publicaciones

Cada pareja verificada crea una relación `same_native_frame` con su ocurrencia
fuente. Otras ocurrencias de bytes idénticos crean `exact_content_dependency`:
igualdad de contenido no afirma identidad temporal adicional.
Todas son restricciones indivisibles para consumidores futuros. La función
`enforce_identity_splits` exige ambos extremos, igual partición y ausencia de
contenido protegido en desarrollo. **Los splitters existentes no consumen todavía
esta nueva publicación**; su incorporación queda bloqueada hasta integrar y
validar ese contrato. No se alteran los 30 brazos ni el test propuesto.

`inventory.json`, `correspondences.json`, `frame_identity_relations.json`,
`class_statistics.json`, `candidate_labels/`, `gallery.html`, `gallery/`,
`inputs.json`, `source_bindings.json`, `summary.json` y `receipt.json` conservan la
evidencia fuera de Git. Un directorio de labels vacío significa cero transferencias,
no etiquetas vacías generadas. El verificador comprueba cobertura del recibo,
fuentes originales y protección; `--replay` reproduce las salidas científicas y
visuales. El recibo operacional contiene fecha, versiones, commit y hashes.

Todas las anotaciones permanecen candidatas. `ready_for_training=false` siempre.

## Verificación temporal pendiente cuando falta el registro de extracción

No basta con verificar PNG → video: también hay que verificar imagen etiquetada
→ el mismo fotograma nativo. El procedimiento propuesto, aún no ejecutado, es:

1. Recuperar el script/comando y versión del procesamiento sin HUD, sus parámetros,
   máscara, resize/crop/orientación, y el registro de extracción de las imágenes
   etiquetadas. Ligar cada fuente y registro mediante SHA-256. Si no existen,
   mantener la limitación explícita en vez de reconstruirlos como hechos.
2. Inspeccionar el stream original y registrar frame nativo, PTS/time base, tasa
   nominal y variable, decoder y versión. La fórmula `(index-1)/30` solo define
   centros de búsqueda; no sustituye los PTS ni prueba el índice inicial.
3. Seleccionar anclas distribuidas en inicio, final, transiciones y tramos de
   movimiento, buscar ventanas de fotogramas vecinos y documentar alternativas.
   Recuperación visual puede reducir la búsqueda; escenas estáticas o repetidas
   pueden impedir distinguir fotogramas consecutivos.
4. Verificar simultáneamente la evolución temporal de vecinos y puntos/regiones
   espaciales fuera del HUD. Registrar correspondencias, residuos y regiones
   visibles en común. Estimar offset, stride o segmentos solo como hipótesis y
   contrastarlos en anclas independientes; no extrapolar una ancla a 19.800 PNG.
5. Verificar cada pareja propuesta y la transformación contra el raster completo,
   incluyendo bordes para detectar recortes. Si el procesamiento es con pérdida,
   definir y validar un protocolo adicional de tolerancias y controles negativos
   antes de ampliar v1; no ajustar tolerancias para aceptar los candidatos actuales.
6. Vincular la imagen etiquetada y el PNG al mismo video SHA-256/índice nativo.
   Una pareja temporalmente indistinguible permanece pendiente. Documentar al
   responsable, evidencia y alternativas descartadas; revisar las cajas sobre
   ambas imágenes sin inventar objetos antes ocultos por el HUD.

Solo después de ese trabajo corresponde suministrar evidencia por pareja,
repreparar una publicación nueva y revisar las etiquetas candidatas. El uso de
datos protegidos en esta auditoría no permite incorporarlos al desarrollo ni
optimizar configuraciones del detector con ellos.
