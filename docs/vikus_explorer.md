# VIKUS: inspección local de Scene / Cluster

VIKUS complementa la inspección del proyecto con un canvas WebGL de imágenes,
zoom y filtros por dimensiones. Permite comparar la apariencia de los grupos
y recorrer una proyección existente como colección visual. Cada elemento es
un **content_id**; un clúster es una **candidate scene**, pendiente de validación
humana. La implementación y su smoke no constituyen un nuevo experimento.

| Herramienta | Responsabilidad |
|---|---|
| VIKUS | Overview de colección, clústeres, secuencias, splits y layouts guardados |
| [Streamlit Explorer](cluster_split_explorer.md) | Inspección detallada, playback, timelines, gaps y comparación de particiones |

Ambas muestran los mismos IDs de clustering, split y cluster. Streamlit conserva
sus vistas y controles; no se incorporan capturas reales al reporte versionado.

## Instalación, build y serve

Desde la raíz del repositorio, Python 3.11 y `FLIR_DATA_ROOT` en el entorno o
`.env` local. Pillow pertenece al core; VIKUS no necesita Streamlit, Node, GPU,
FFmpeg, modelos ni los extras de reducción/detección para construir el bundle.

```powershell
uv sync --locked
uv run flir-pipeline explorer vikus-build --candidate C10 --seed 0 --name c10-seed0-local
uv run flir-pipeline explorer vikus-serve --bundle reports/explorer/vikus/c10-seed0-local
```

Abre **http://127.0.0.1:8765/**. Se requiere un navegador de escritorio con WebGL;
se recomienda una ventana de al menos 1100 px para los controles. El servidor
sólo escucha en loopback. Ctrl+C lo detiene. `--port` cambia el puerto local.

El primer build descarga exclusivamente el código público fijado de VIKUS a
`.cache/upstream/`. No transmite el manifest ni imágenes. Los builds posteriores
pueden usar `--offline`; `--runtime-archive <zip>` admite el mismo ZIP fijado
previamente descargado. Se verifica su SHA256. Una cache ausente o distinta
produce error, sin sustituir la versión. Servir el bundle no requiere Internet.

`--candidate` es un alias leído de la tabla existente de candidatos. No hay
selección ni lógica especial para C10. También se puede utilizar:

```powershell
uv run flir-pipeline explorer vikus-build --split-space-id <existing-id>
uv run flir-pipeline explorer vikus-build --clustering-space-id <existing-id>
uv run flir-pipeline explorer vikus-build --split-space-id <baseline-id> --clustering-space-id <existing-id>
```

`--manifest`, `--candidates`, `--clustering-root`, `--split-root`,
`--reduction-root` y `--data-root` ajustan las fuentes locales. Discovery excluye
incompletos, `.partial`, quality inválida e identidades ambiguas. `--seed` elige
un split que ya existe. No ejecuta el optimizador. Un baseline requiere indicar
explícitamente el clustering sobre el cual se superpone.

Un clustering reducido exige encontrar **su reducción fuente exacta**. Se
verifican IDs, feature space, encoder, dimensión, cobertura por contenido y
checksums, incluidos los fingerprints de entrada del clustering. Una reducción
t-SNE adicional del mismo dataset/feature space puede indicarse mediante
`--extra-reduction <directory>`; se admite un layout por método. No se elige
arbitrariamente otro run cuando la fuente falta. No se importan ni ejecutan
reducers, modelos, clustering, generación de splits o detector.

## Navegación y límites

- **Clusters:** group layout por cluster_id, orden numérico y **Noise (-1)**
  separado al final. El ruido conserva -1; sus grupos singleton provienen del
  split existente y se muestran en metadata.
- **Sequences:** group layout por archivo fuente/secuencia inferida; dentro de
  cada grupo, índice inferido y desempate estable por content_id. Los índices
  ambiguos/desconocidos quedan al final y vacíos en metadata, sin elegir una
  ocurrencia arbitrariamente. No es una reproducción temporal.
- **Cluster-aware split:** grupos train / validation / test. Se pueden combinar
  con el filtro de cluster. Los baselines tienen su propio título; si el mismo
  contenido histórico tiene varias pertenencias, aparecen todas, por ejemplo
  `train + test`, en un solo elemento visual, sin inventar una asignación única.
- **PaCMAP visual similarity:** coordenadas existentes, unidas por content_id.
  El CSV conserva x/y originales con precisión suficiente para round-trip.
  El canvas aplica traslación y escala uniforme, con inversión de Y para la
  pantalla. No ajusta distancias, colisiones, densidades ni distribuye imágenes
  para mejorar la apariencia. El solapamiento es posible. Los puntos filtrados
  conservan sus posiciones y se atenúan. No es ground truth geometry.
- **t-SNE visual similarity:** opcional, sólo desde un run alineado explícito.

Rueda para zoom, arrastre para desplazar, clic en una imagen para detalle.
**Enfocar selección** amplía los contenidos activos sin modificar el layout.
**Restablecer filtros** recupera la colección. La vista global de muchos grupos
tiene miniaturas pequeñas: filtra y enfoca para inspeccionar sus miembros.
Los grupos nativos de VIKUS forman columnas desde abajo; el orden inferido
avanza de izquierda a derecha y de abajo hacia arriba.

Filtros separados: **Cluster, Sequence, Split, Noise, Class presence**. Las
dimensiones se combinan con AND. Dentro de Cluster/Sequence/Noise, varios
valores usan OR. Las dimensiones de arrays (clases y memberships de split)
requieren todos los valores seleccionados (AND), conforme al crossfilter de
esta versión. Clic en el título de una dimensión la restablece. La lista es
desplazable; pasar el cursor resalta una selección provisional.

La secuencia e índice proceden de la heurística de nombres y del consenso de
todas las ocurrencias. No se conocen timestamps/FPS verificados. Proximidad
temporal, duplicado exacto y similitud visual son conceptos distintos. Ver un
grupo coherente no valida científicamente una escena ni demuestra leakage.

## CSV y assets

```text
reports/explorer/vikus/<bundle>/
  index.html, LICENSE.md, js/, css/, font/, img/, flir.js, flir.css
  bundle_receipt.json
  data/
    config.json, data.csv, timeline.csv, info.md
    layouts/pacmap.csv          # y tsne.csv sólo si se solicita
    images/<content_id>.jpg
    sprites/manifest.json
    sprites/sheet-000.jpg       # atlas adicional cada 256 contenidos
```

`data.csv` tiene una fila por contenido y campos `id`, `keywords`,
`_cluster_id`, `_sequence_id`, `_frame_index`, `_new_split`,
`_original_split_membership`, `_noise`, `_encoder`, `_representation`,
`_algorithm`, IDs de experimentos, grupo, número de ocurrencias e imagen
representativa. `_frame_occurrences` conserva **todas** las ocurrencias con
frame_id, splits histórico/nuevo, archivo, secuencia e índice. El panel de
detalle expone este mapping. No se usan las etiquetas para alterar los grupos.

`_contains_vehicles`, `_contains_buildings`, `_contains_roads`,
`_contains_rivers`, `_contains_heavy_machinery` expresan presencia post-hoc en
alguna ocurrencia. Valores true/false/unknown distinguen ausencia conocida de
anotación desconocida. `_empty_annotation` admite true/false/mixed/unknown;
mixed conserva casos con anotaciones vacías y no vacías del mismo contenido.
`_annotation_conflict` señala hashes de labels distintos. No los corrige.

**year=0** es un placeholder técnico constante y reproducible. No representa
año ni índice. Se excluye del detalle y de los layouts; `timeline.csv` tiene
sólo encabezados. Las etiquetas de los grupos son clusters/secuencias/splits.

`FrameReader` abre ZIPs en modo lectura, comprueba los bytes contra el manifest
y decodifica en memoria. Las previews son RGB/JPEG quality=92, subsampling=0,
lado máximo de detalle 1024 px. Miniaturas de hasta 128 px, en celdas de 128 px
de atlas 2048×2048, formato **pixi-packer v1**. Sólo resize LANCZOS y encoding,
con relación de aspecto conservada; sin crop, CLAHE, filtros ni enhancement.
No se requiere el script externo de preprocessing de VIKUS ni su infraestructura
de embeddings. Los sprites se identifican por el mismo id del CSV.

## Versión, licencia y adaptaciones

Upstream: [cpietsch/vikus-viewer](https://github.com/cpietsch/vikus-viewer),
commit [ccb4e7e3d921521b01b2760b51bf4d1b090b844b](https://github.com/cpietsch/vikus-viewer/tree/ccb4e7e3d921521b01b2760b51bf4d1b090b844b).
[Licencia MIT](https://github.com/cpietsch/vikus-viewer/blob/ccb4e7e3d921521b01b2760b51bf4d1b090b844b/LICENSE.md),
Christopher Pietsch y colaboradores. La licencia completa acompaña cada bundle.
SHA256 del ZIP fijado:
`1295c3cb04e03317337ee29205438d7552c601d0834ac3b0640223c2a062e3f0`.

No se versiona una copia del runtime externo. `vikus_upstream.py` selecciona
assets estáticos y aplica parches comprobados contra ese commit a la copia
local generada. Una discrepancia de contexto falla explícitamente. Adaptaciones:

1. `index.html`: cabecera, controles y assets FLIR; se retira el fallback remoto
   obsoleto para Internet Explorer.
2. `canvas.js`: inicio en Clusters; orden por índice de presentación y orden de
   grupos explícito; margen para controles y encuadre de selección; escala XY
   uniforme; eliminación del anillo aleatorio de puntos filtrados.
3. `flir.js` / `flir.css`: parseo de arrays JSON para crossfilter, conteo activo,
   reset/enfoque y distribución de controles. No cambian memberships ni x/y.

El runtime fijado utiliza D3 v3, Vue 2 y PIXI 5.2.2, incluidos localmente. Config:
`loader.items/timeline/info/layouts/textures`, `filter.type=crossfilter`,
`filter.dimensions`, `sortArrays` y `detail.structure`. Las rutas parten de la
raíz del bundle. El servidor permite scripts locales y `unsafe-eval` requerido
por Vue; bloquea conexiones/recursos externos mediante CSP. Los enlaces de
attribution son navegación manual, no cargas de datos del viewer.

## Reproducibilidad y privacidad

Todo el bundle, recibos, capturas y sprites queda bajo `reports/**`, ya ignorado;
la descarga pública queda bajo `.cache/`, también ignorado. **No publicar ni
subir el bundle**, ni servirlo en una interfaz pública. El servidor comprueba
integridad y limita las rutas a los archivos del recibo, sin directory listing
ni endpoints de escritura. No registra URLs con content IDs en access logs.

`bundle_receipt.json` registra versión, hashes del exporter/adaptador y runtime,
Pillow, IDs, coordenadas fuente, conteos y hashes de cada archivo generado.
También registra commit/estado dirty y SHA256 de ZIPs fuente. Se verifica que
manifest, artifacts y ZIPs conservan sus hashes al terminar. El fingerprint de
exportación cambia con los inputs o la implementación. Un bundle idéntico
íntegro se reutiliza; un destino distinto/incompleto no se sobrescribe. Usa un
nombre nuevo o deja que se genere automáticamente. Estos recibos son evidencia
de exportación e integridad, nunca métricas científicas nuevas.

## Validación

`tests/test_vikus.py` usa imágenes y artifacts sintéticos, offline: metadata,
ocurrencias, schema/config, grupos y ruido, anotaciones conflictivas, CSV de
coordenadas con índice permutado y round-trip exacto, incompatibilidades,
mapping de sprites, paths seguros, invariantes read-only y dispatch HTTP en
memoria. No descarga upstream/modelos ni conecta a servidores durante los tests.

```powershell
uv run ruff check .
uv run python -m pytest
uv run python scripts/check_notebook_source.py
```

El smoke real y la inspección del navegador se ejecutan fuera de CI; sus assets
y evidencias permanecen locales. El estado de los experimentos y la validación
humana de candidate scenes se mantienen separados de la infraestructura.

Smoke local del 2026-09-19: bundle C10/seed 0 con **1459 imágenes únicas**, 1657
ocurrencias, seis atlas, 34 clústeres y 47 contenidos de ruido. Las secuencias
inferidas contienen 712/747 contenidos; el split existente asigna 1029/95/335
contenidos a train/validation/test. No confundir estos conteos por contenido
con las ocurrencias históricas. Se comprobó igualdad exacta de x/y mediante
round-trip del CSV frente al array PaCMAP original, cobertura de imágenes y
sprites y hashes de ZIPs sin cambios.

El navegador local cargó las cuatro vistas y los filtros de cluster, secuencia,
split, ruido y clases; los conteos visibles coincidieron con el CSV, sin errores
JavaScript observados. La suite completa pasó 185 tests (22 del exporter), Ruff
y siete notebooks fuente limpios. Capturas y recibo del smoke quedan sólo en
`reports/explorer/`. No se añadió t-SNE al bundle C10 ni se ejecutaron nuevos
experimentos; la validación humana de escenas continúa pendiente.
