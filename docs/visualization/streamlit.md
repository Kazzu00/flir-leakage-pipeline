# FLIR Cluster & Split Explorer

Herramienta local de **inspección humana en modo lectura**. Abre clustering runs,
asignaciones y resúmenes existentes. No ejecuta modelos, similitud, reducción,
clustering, optimización de splits ni detección. La inspección no acredita una
nueva validación científica de escenas ni una selección distinta de candidatos.

[VIKUS Viewer](vikus.md) complementa esta herramienta con el overview
de imágenes por cluster/candidate scene, secuencia, split y proyecciones guardadas.
Streamlit conserva detalle, playback, timelines, gaps y comparación. El overview
muestra `clustering_space_id`; junto con `cluster_id` y el ID de split permite
localizar el mismo grupo en ambas interfaces.

## Instalar y lanzar

Desde la raíz del repositorio, con Python 3.11 y `FLIR_DATA_ROOT` configurado en
el entorno o en el `.env` local:

```powershell
uv sync --locked --extra explorer
uv run --extra explorer streamlit run apps/cluster_split_explorer.py
```

Abre `http://127.0.0.1:8501`. `.streamlit/config.toml` limita el servidor a loopback,
desactiva telemetría y evita servir archivos estáticos. No desplegar públicamente.
Para conservar las herramientas de desarrollo/reporting en el entorno:

```powershell
uv sync --locked --extra dev --extra reduction --extra reporting --extra explorer
```

Streamlit es un extra opcional; Pillow ya pertenece al core. No se requieren
FFmpeg, codecs de video, pesos ni GPU. La forma de lanzamiento sigue la
[documentación de Streamlit](https://docs.streamlit.io/develop/concepts/architecture/run-your-app).

## Fuentes y discovery

Referencias: **C10 — DINOv2 / PaCMAP / DBSCAN**; **C12 — DINOv2 / t-SNE / HDBSCAN**.

- Manifest canónico local: por defecto
  `data/manifests/flir_canonical_candidate_v1.parquet`.
- `artifacts/clustering/**/metadata.json`: `artifact_kind=clustering_run`, índice
  de contenido, labels, resumen de clústeres, métricas y quality publicados.
- `artifacts/splitting/**/metadata.json`: `artifact_kind=split_run`, asignaciones
  por ocurrencia/contenido, grupos fuente y quality publicados.
- `reports/splitting/tables/clustering_candidates.csv`, opcional, aporta alias
  descriptivos como C10/C12 por `clustering_space_id`; no crea ni elige runs.
- ZIPs originales identificados por el manifest, relativos a `FLIR_DATA_ROOT`.

Las rutas del manifest y las raíces de artifacts se ajustan en **Fuentes locales**.
Discovery excluye metadata incompleta, quality inválida, identidades repetidas y
directorios `.partial`; muestra avisos y conserva los archivos. Sólo los runs del
dataset del manifest llegan a los selectores. Los archivos leídos al abrir un run
se contrastan con sus checksums registrados, y se comprueban cobertura y mapping
por IDs. Esto no recalcula métricas científicas ni sustituye sus recibos previos.

Seleccionar un split cluster-aware restringe el selector al clustering fuente
exacto. Historical y random pueden superponerse sobre los clusterings del mismo
dataset. Los controles encoder/representación/algoritmo muestran combinaciones
realmente disponibles. **Actualizar discovery** refresca el catálogo en memoria.

## Navegación

- **Cluster browser:** elige `cluster_id`; muestra contenidos, todas sus
  ocurrencias históricas, secuencias, rangos de índices, memberships históricos y
  asignaciones del split seleccionado. Coseno intraclúster, secuencia dominante
  y medoid proceden del resumen guardado, cuando están disponibles.
- **Browse by split:** TRAIN / VALIDATION / TEST filtra contenidos con al menos
  una ocurrencia en esa partición. Puedes abrir sus clústeres o noise y recorrer
  timelines, galerías y previews. En historical/random un clúster puede estar en
  varias particiones; el filtro no redefine ese clúster. Las métricas guardadas
  siguen describiendo el clúster completo.
- **Compare partitions:** selecciona dos runs y una secuencia. Las dos bandas
  coloreadas preservan cada ocurrencia, incluso duplicados exactos del histórico;
  la tabla muestra asignación previa/nueva y cluster fuente de cada lado. Los
  grupos e IDs completos están en **Show technical metadata**.
- **Noise / singleton browser:** navega cada contenido con `cluster_id=-1`.
  En los splits cluster-aware, muestra el grupo singleton existente y su split.
  No transforma el ruido en clústeres inventados.

Las galerías muestran 12 contenidos por página, ordenados por secuencia e índice
inferido, con desempate estable por contenido. Cada miniatura usa una ocurrencia
representativa para leer los bytes; el panel técnico conserva **todas** las
ocurrencias. Las pertenencias históricas múltiples no se reducen a un split único.

## Secuencia visual reconstruida

**Reconstructed frame sequence** reproduce un contenido exacto por posición en
orden inferido. No es una reconstrucción del video original: faltan timestamps y
FPS verificados, pueden faltar frames y un clúster puede ser discontinuo.

Cada secuencia, identificada también por su archivo fuente, tiene su propio
timeline/preview. No se concatenan secuencias distintas. La procedencia temporal
reutiliza la regla de consenso existente: todas las ocurrencias deben concordar
en secuencia e índice para ordenar un contenido. Conflictos/desconocidos quedan
en la galería y se excluyen del playback; no se inventa posición temporal.

El umbral de gap configurable (25 por defecto; 0 desactiva la segmentación)
separa únicamente **segmentos de playback**. Los gaps grandes se listan y el GIF
rotula los saltos entre índices. Otro control limita cada preview a 60/120/240
frames para acotar memoria, sin muestrear ni descartar frames: se recorren los
segmentos/páginas disponibles. La segmentación no cambia memberships ni métricas.

FPS de visualización: 2/4/8/12. Siempre se muestra:

> Playback FPS is for visualization only; source timing is unknown.

GIF cuantiza duración a 10 ms; 8/12 FPS son aproximados. El navegador también
puede introducir demoras. El FPS nunca se registra como atributo del dataset.

## Lectura, cache y privacidad

Pillow decodifica miembros ZIP en memoria. Cada imagen leída debe coincidir con
el SHA256 del manifest. No se extraen archivos ni se escriben fuentes/artifacts.
La cache Streamlit de previews es sólo en memoria, limitada a ocho entradas y
ligada a clustering, cluster, secuencia, IDs ordenados, contenidos, FPS, ancho,
segmentación, versión de renderer y estado de los ZIPs. No es evidencia científica.

`reports/**`, GIFs, imágenes, assignments y notebooks ejecutados están ignorados
por Git. No publicar capturas ni previews reales. La UI puede mostrar IDs y rutas
locales bajo paneles técnicos: está destinada exclusivamente al equipo local.

## Validación

Referencias: **C10 — DINOv2 / PaCMAP / DBSCAN**; **C12 — DINOv2 / t-SNE / HDBSCAN**.

Los tests de `tests/test_explorer.py` son offline/sintéticos y cubren discovery,
identidad, mapping contenido/ocurrencias, memberships, secuencias, orden, gaps,
ruido, filtros, comparación, planes, cache y lectura/GIF sin cambios del ZIP.
No levantan Streamlit en CI. El smoke con datos locales se mantiene separado de
las pruebas unitarias y no convierte la inspección visual en experimento.

Smoke local del 2026-09-19: 414 clustering runs y 66 particiones descubiertos;
C10/C12 seed 0 abren con 1459 contenidos y 1657 ocurrencias. Se comprobaron las
cuatro vistas, las tres particiones, separación de secuencias y generación GIF
mediante Streamlit AppTest; el servidor HTTP respondió correctamente. Hashes de
ZIPs fuente y metadata científica antes/después coinciden. El recibo y un GIF
de seis frames permanecen locales en `reports/explorer/`. No hubo navegador
conectado para inspección visual de la UI; esa revisión humana queda pendiente.
