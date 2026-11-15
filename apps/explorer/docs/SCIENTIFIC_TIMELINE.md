# Línea temporal científica M02

> Fuente intermedia preservada. La vista activa aplica ahora admisión estricta
> de tres vídeos; ver [contrato vigente](VERIFIED_VIDEO_TIMELINE.md). Las familias
> inferidas y los clusters de referencia descritos abajo no entran en ese mount.

Implementada en `/organization/sequences` (vista inicial, también `?mode=timeline`).
La evidencia temporal anterior permanece en `?mode=sequences`; linkage y revisión
humana conservan sus rutas. Es una vista de publicaciones existentes, no un
ejecutor científico. No cambia clustering, secuencias, particiones, test o modelos.

## Qué representa

Tres pistas comparten coordenadas y desplazamiento: marcas de ocurrencias
coloreadas por clustering, intervalos temporales exportados y marcas de split
por ocurrencia. La coordenada es el índice de aparición, nunca el orden de
clustering. Los duplicados conservan distintos `frame_id`; si comparten índice,
se apilan sin inventar un orden entre ellos. Los gaps del índice no son segundos.

El color se calcula por `(cluster_run_id, cluster_id)` y no depende del split.
`-1` es ruido, `0` es un clúster válido y una membresía ausente es «Sin asignación».
La tabla accesible y Anterior/Siguiente permiten seleccionar los mismos frames
con teclado. Zoom, desplazamiento y selección de índices son comunes a las tres
pistas. Una selección exploratoria no crea una secuencia.

Población, política y semilla seleccionan **un único run** disponible. La ejecución
de clustering se elige por separado como referencia de contenido; no se afirma
que haya generado S1/S2/S3. La distribución de splits del intervalo cuenta cada
ocurrencia y sus clústeres, sin reemplazar la pista por un split dominante.

Los intervalos candidatos son únicamente el alcance observado de namespaces
temporales ya existentes. No se calculan cortes a partir de cambios de color ni
se convierten zonas de frontera en secuencias. El contrato distingue candidatos,
secuencias revisadas y secuencias verificadas; la conexión actual solo dispone de
tramos candidatos. Para mostrar secuencias revisadas se requiere una fuente
explícita de sus identidades, intervalos y decisiones; el frontend no las adjudica.

## Fuentes y unión reproducible

`tools/export_scientific_timeline.py` es un adaptador offline independiente del
backend. Lee únicamente JSON/Parquet explícitos de:

- Una publicación `fixed_test_train_val_v1`: receipt, summary, provenance y
  `runs/*/{metadata.json,occurrence_memberships.parquet}`.
- `organization-evidence-v2`: manifest, records, contents, timelines,
  cluster_memberships, split_memberships, clusters, boundary_zones,
  clustering_configurations y media (solo metadata, no archivos de imagen).

Cada archivo consumido debe estar ligado por checksum a su receipt o manifest.
Se comprueban las identidades de run/población/política/semilla, unicidad de
ocurrencias, inventario histórico y test común. El inventario histórico debe
coincidir en `frame_id` con el cohort etiquetado de organización, además de su
conteo esperado. No se unen las poblaciones completas por asumir dataset_id
equivalentes: las ocurrencias compartidas se vinculan exactamente por identidad
y contenido. Las fuentes se vuelven a comprobar al finalizar.

La membresía de clustering se obtiene exclusivamente de
`cluster_memberships.json`, por `(cluster_run_id, content_id)`. Se comprueban
configuración, consistencia, noise y referencias. `group_id` no es un clúster y
no se consulta como sustituto. La fuente actual conserva membresías en artifacts
congelados y declara indisponible el artifact de clustering original: no se
reconstruyen parámetros, diagnósticos, embeddings o métodos ausentes.

Se preservan `candidate_id`, `membership_id` y `test_id`. Si la publicación no
exporta `split_space_id`, permanece null y se muestra como no exportado;
`membership_id` no se renombra para completar ese campo.

El Parquet se proyecta a columnas de identidad, nombre, procedencia, índice y
split. No se exportan anotaciones, clases, labels, métricas de selección,
predicciones, pesos ni imágenes. Las miniaturas disponibles se referencian desde
el mount local existente, sin copiarlas o generarlas.

## Exportar y conectar

Configurar variables con las rutas locales de las dos fuentes y un **destino
nuevo**, fuera de ellas. Con uv y las dependencias fijadas en el script:

```powershell
uv run tools/export_scientific_timeline.py "$env:FLIR_CANDIDATE_PUBLICATION" "$env:FLIR_ORGANIZATION_EXPORT" public/runtime/scientific-timeline-v1
```

El destino no se sobrescribe. Para repetir, elegir otro directorio y contrastar
sus hashes antes de actualizar el mount por una operación explícita. Los tests
usan exclusivamente fuentes sintéticas temporales y no requieren Hypatia o GPU.

Se producen `index.json`, `coverage.json`, `timeline_assigned.csv`,
`timeline_unassigned.csv` y `receipt.json`. Los CSV del export offline incluyen
`run_id`, población, política y semilla: la misma ocurrencia puede aparecer en
varios runs, sin que eso aumente el número de ocurrencias científicas únicas.
El navegador descarga CSV **solo del run seleccionado**, con el clustering de
referencia seleccionado. Los filtros de pantalla no alteran esos totales.

El adaptador TypeScript valida el contrato estricto con Zod, comprueba el SHA-256
de `index.json` contra su receipt y contrasta la procedencia. No hay fallback a
DEMO ante un error o ausencia. El export local vive en `public/runtime/`, ignorado
por Git y excluido del build mediante `build.copyPublicDir: false`.

## Cobertura observada localmente · 2026-11-13

La copia local existente contiene 30 configuraciones, Historical/Expanded,
S1/S2/S3 y semillas 0–4. No equivale a 30 entrenamientos. Su estado publicado es
provisional y su test figura como PROPOSED; la vista no cambia ni determina el
estado científico de publicaciones posteriores.

En `historical-S1-0` se conservaron y contabilizaron **1.657 ocurrencias**. Son
representables en **dos familias documentadamente inferidas de nombres**, no en
dos vídeos originales verificados. No se verificaron posiciones contra vídeos
originales en este trabajo y no se presentan timestamps reales.

En Expanded se conservan 1.706 ocurrencias; 49 quedan fuera de los ejes temporales
por vídeo desconocido y permanecen en el reporte y dataset. Los cuatro motivos
admitidos son `unknown_video`, `ambiguous_association`, `unknown_order` e
`index_conflict`. Un índice repetido por el mismo contenido conserva todas sus
ocurrencias; contenidos distintos en un mismo índice se señalan como conflicto.

La repetición del export produjo cinco archivos byte a byte idénticos. Esto
valida la proyección de presentación, no el descubrimiento temporal, la revisión
de anotaciones ni los resultados del detector. Hypatia rechazó SSH no interactivo:
se verificó la copia local contra sus receipts, sin revalidación remota.

## Validación y límites

Las pruebas cubren integridad de fuentes, identidad, multiplicidad, índice cero,
orden, recurrencias, ruido, ausencia de clúster, aislamiento de semillas/test,
CSV, exclusiones, reproducibilidad y estados no disponibles. Chromium comprueba
las tres pistas, teclado, zoom/scroll, selección, descargas y accesibilidad
automática en temas claro y oscuro. Los fixtures se identifican como sintéticos;
las comprobaciones reales son opt-in:

```powershell
$env:FLIR_SCIENTIFIC_TIMELINE_REAL = '1'
npx playwright test e2e/scientific-timeline.spec.ts --workers=1
```

La accesibilidad automática no es una certificación WCAG. La implementación no
prueba que una familia nominal corresponda a un vídeo original, que un tramo sea
una secuencia confirmada o que una recurrencia visual implique continuidad o
leakage. Siguen pendientes las publicaciones de posiciones/relaciones revisadas
y los gates científicos de test y experimento. No se ejecutó entrenamiento,
SLURM, extracción, reducción, clustering ni particionamiento nuevo.
