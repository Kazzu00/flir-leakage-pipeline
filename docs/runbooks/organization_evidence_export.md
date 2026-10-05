# Export de evidencia de organización M02

Infraestructura de presentación de evidencia existente para
`flir-pipeline-explorer`: `/organization/explore`, `/organization/sequences` y
`/organization/evaluation`. La implementación local usa fixtures sintéticas.
**No se ha exportado ni revalidado aquí la membresía real de Hypatia.**

## Flujo y fuentes

Implementación y pruebas locales → export real en Hypatia → sincronización
local del frontend. No hay nuevos fits, splits, componentes, fronteras ni
confirmaciones automáticas. El contrato del detector permanece intacto.

`explorer export-organization` requiere:

- Manifest canónico y `protocol/plan.json` + `runtime_freeze.json` del experimento.
  `verify_plan` valida la identidad; se comprueba también la vinculación y
  configuración del freeze. La selección es `plan.identity.splits`: **no** se
  buscan aliases C10/C12 en nombres de carpetas ni se codifican IDs/seeds/conteos.
- Runs de splitting resolubles por identidad, con metadata exactamente ligada
  por `split_metadata_sha256` al plan. Se reutilizan `discover_runs`,
  `load_split`, `checked_file` y `with_split`, comprobando también el índice de
  asignación por contenido y los conteos congelados.
- Los runs de clustering referenciados por esos splits. `inspect_clustering`
  verifica configuración, identidad y checksums; `load_cluster` verifica índice,
  representantes, labels y summaries. Solo se exportan las configuraciones
  seleccionadas; no todos los experimentos exploratorios. Se conservan las
  probabilidades de membresía si existe su array autoritativo; no se inventan
  probabilidades para otros algoritmos. Las distancias/orden de OPTICS se copian
  de `algorithm_diagnostics.npz`: infinito se representa con valor null y flag
  `*_infinite=true`, separado de un diagnóstico ausente (flag null).
- Linkage opcional bajo `--linkage-root` y directorios adicionales repetibles
  `--evidence-root`. Las fuentes no presentes se registran como limitación;
  una publicación seleccionada corrupta o de tipo desconocido falla.

El universo comunicado para comprobar **después en Hypatia** es 16 splits:
historical × 1, random_content × 5, C10 × 5, C12 × 5; aproximadamente 1657
registros etiquetados y 1459 contenidos. No son constantes ni resultados de
las pruebas. Un plan anterior válido exportará su propia selección: el operador
debe suministrar el freeze final, no sustituirlo. Este comando comprueba evidencia
organizativa; no verifica la finalización de entrenamientos del detector.

## Contratos de linkage soportados

| Fuente existente | Adaptador y salida |
|---|---|
| `labeled_video_link_candidates` | `load_candidate_context`, compartido con revisión manual. Requiere `--sequence-root` para resolver el `sequence_set_id` ligado. Cada `candidate_id` se conserva como `kind=candidate_pair`; no se construyen componentes nuevos. Query, candidato, scores separados y todas sus ocurrencias permanecen visibles. |
| `sequence_structure_review_v1` | `experiments.artifacts.inspect/tables`; cores desde `cores` y membresías desde `membership`. Las zonas se copian desde `intervals`. El adaptador nativo revalida además originales mediante `verify_native`. |
| `sequence_recurrence_v1` | Join de `diagnostic_components` → `core_content_membership` → membresía de ocurrencias de la estructura ligada. Requiere esa estructura exacta dentro de las raíces suministradas. No se ejecuta `diagnostic_components` ni se asigna el interior de intervalos. |
| `labeled_visual_dependency_manual_calibration` | `inspect_snapshot` y `verify_review`; requiere `--review-source-map` con el formato existente de `linkage review aggregate`. Decisiones externas en `reviews.json`, sin convertir grupos propuestos en membresías confirmadas. |

Los experimentos de estructura/recurrencia de esta versión deben estar ligados
al mismo manifest etiquetado (pueden ser un subconjunto por familia). Una
estructura de otro dataset se rechaza; no se busca correspondencia por filename.
Los videos fuente se incorporan mediante el contrato etiquetado→video que sí
declara ese vínculo. Los reports legacy crudos deben pasar por su importador
existente antes de suministrarse; el exportador no interpreta formatos ad hoc.
No se consumen agregados de revisión como si fueran revisiones individuales:
suministrar los directorios de las revisiones originales y su mapa de fuentes.

Las raíces pueden ser el propio directorio de una publicación o una colección.
Se ignoran directorios `.partial`, y los artefactos experimentales de fitting,
métricas/suites no se convierten en grupos. Una recurrencia sin su estructura,
una identidad duplicada, un tipo no soportado o una referencia desconocida
produce un error explícito. Preferir raíces específicas para la evidencia M02.

## Comando para Hypatia

Después de transferir estos commits y actualizar el paquete en el entorno
existente, ejecutar desde el checkout. No modificar ni recrear el freeze:

```bash
cd ~/flir-leakage-pipeline
uv run --no-sync flir-pipeline explorer export-organization \
  --manifest data/manifests/flir_canonical_candidate_v1.parquet \
  --plan artifacts/detection/protocol \
  --split-root artifacts/splitting/runs \
  --clustering-root artifacts/clustering \
  --linkage-root artifacts/linkage \
  --sequence-root artifacts/sequences \
  --output exports/frontend/organization \
  --media-output artifacts/frontend/organization-media \
  --include-previews
```

`FLIR_DATA_ROOT`/`.env` resuelve los ZIPs read-only; también se admite
`--data-root`. Para miniaturas de frames muestreados, añadir
`--video-images-root "$VIDEO_IMAGES_ROOT"`, apuntando a la raíz que contiene los
`image_path` declarados. Su ausencia solo afecta las miniaturas de esos videos.

Las ubicaciones reales de los experimentos de secuencias/revisiones de Hypatia
no están comprobadas en este clon. Añadir, usando **sus rutas existentes**:

```bash
# Opciones adicionales del mismo comando; las variables son localizadores.
--evidence-root "$M02_STRUCTURE_PUBLICATION" \
--evidence-root "$M02_RECURRENCE_PUBLICATION" \
--evidence-root "$M02_MANUAL_REVIEW_REVISION" \
--review-source-map "$M02_REVIEW_SOURCE_MAP"
```

No inventar esos paths ni copiar artefactos antiguos para simular los finales.
No se necesita GPU, modelos, entrenamiento ni acceso a Internet. El comando
utiliza las dependencias core. La CLI devuelve código 2 ante fuentes inválidas.

## Contrato normalizado

Schema version: `organization-evidence-v1`. Todos los JSON están en
`exports/frontend/organization/`:

| Archivo | Unidad y claves |
|---|---|
| `manifest.json` | Plan/freeze/manifest, commit y estado dirty, recibos de fuentes, alcance de verificación, conteos, limitaciones, definiciones y SHA256 de los demás archivos. Timestamp aislado aquí. |
| `records.json` | Una ocurrencia; `record_id` conserva exactamente `frame_id`, con `content_id`, cohort, filename, clases y procedencia. |
| `contents.json` | Un contenido exacto; todos sus `record_ids`, representante para preview y consenso temporal/de anotación. |
| `splits.json` | Una identidad de split; estrategia, seed, fuente, configuración de clustering y conteos train/val/test. Soporte de clases copiado del plan. |
| `split_memberships.json` | Una fila `(split_space_id, record_id)`, incluyendo estrategia, seed, partición y contenido. |
| `clustering_configurations.json` | Configuración seleccionada, representación, algoritmo, parámetros, reducción e identidades. Conteos/ruido contrastados con labels almacenados. |
| `clusters.json` | Clave compuesta `(cluster_run_id, cluster_id)`; tamaños, videos explícitos y rangos por timeline. `-1` es ruido. |
| `cluster_memberships.json` | Una fila `(cluster_run_id, content_id)` con label, ruido y probabilidad autoritativa o null. |
| `linkage_groups.json` | Clave `(evidence_artifact_id, linkage_group_id)`; tipo, fuentes, metadata original, conteos y rangos. |
| `linkage_memberships.json` | Una fila `(evidence_artifact_id, linkage_group_id, content_id)`; registros vinculados, índices observados, rol y referencias a cores. |
| `boundary_zones.json` | Intervalos originales inclusivos, decisión y timeline; nunca un corte exacto. |
| `reviews.json` | Revisión original × query; decisión externa, contenido y referencia al grupo propuesto upstream. No crea una membresía. |
| `timelines.json` | Video explícito o familia inferida dentro de un archive; puntos por contenido y posición, preservando todos los registros. |
| `media.json` | Una entrada por contenido, preview key, ruta relativa, dimensiones, SHA256 y disponibilidad. |
| `schema/organization-evidence-v1.schema.json` | JSON Schema Draft 2020-12 generado desde los modelos Pydantic. |

Para validar con JSON Schema, construir el objeto lógico
`{stem_del_archivo: JSON_parseado}` de los 14 JSON raíz. El productor valida ese
objeto con `OrganizationExport` antes de publicarlo y después de serializarlo.
El modelo añade validaciones de IDs, coverage, conteos, ruido, grupos indivisibles
y consistencia entre tablas que JSON Schema por sí solo no expresa. Los tests
ejercitan también un validador independiente (`jsonschema`, extra dev).

`record_count`/`unique_content_count` cuentan todo el bundle. Los campos
`labeled_record_count`/`labeled_unique_content_count` delimitan el cohort del
detector: los registros adicionales de video nunca reciben splits inventados.
Las anotaciones se mantienen por registro; `annotation_consensus` distingue
`identical_label_bytes`, `different_label_bytes` y `unavailable`. Diferentes
hashes no prueban un conflicto semántico: podrían deberse a formato u orden.
Ante bytes distintos, las clases del contenido quedan null, conservando las
de cada registro. No se recalcula ni sustituye la auditoría de anotaciones.
Un representante
sirve para mostrar una imagen, no para resolver ambigüedades temporales.

## Semántica y joins del frontend

- Particiones: `train`, `val`, `test`. El histórico admite que varios registros
  de un contenido estén en particiones distintas. Los demás splits preservan
  contenidos y clústeres no ruido indivisibles.
- Un clúster es evidencia visual algorítmica. Un componente/candidate core es
  diagnóstico. Ninguno es una secuencia confirmada ni ground truth. Las flags
  de verdad, confirmación automática y creación de secuencias permanecen false.
- `source_video_id` solo contiene IDs explícitos de la grilla de video.
  Las familias `possible_sequence` ya existentes producen timelines con
  `temporal_source=filename_heuristic` y `source_video_id=null`. No se crean
  nuevas inferencias. El `timeline_id` compuesto es una clave de presentación,
  no un identificador de secuencia científica.
- Un contenido que aparece en varios tiempos/videos conserva todas las
  ocurrencias y posiciones. El consenso ambiguo queda null; no se selecciona un
  timestamp arbitrario. `timestamp_seconds` de video es tiempo relativo a la
  grilla, no tiempo de captura. `source_frame_index_estimate` sigue siendo una
  estimación separada de `source_frame_index` exacto, que queda null.
- `boundary_zone` conserva ambos extremos y `inclusive=true`, `exact_cut=false`.
  Los huecos entre miembros no se rellenan. Un rango min/max no asegura cobertura
  continua. Los span/index counts son resúmenes de membresías, no métricas nuevas.
- Ruido `cluster_id=-1` se muestra como no asignado. No es una secuencia ni un
  grupo indivisible global. Las probabilidades almacenadas no son ground truth.

El frontend puede unir record→content→preview, content→cluster/group y
record→split por las claves anteriores. Las membresías de grupo/cluster se
superponen a timelines sin duplicar las asignaciones de todos los splits en
cada punto. Un inspector usa `(strategy, split_seed, partition)` para filtrar
registros y obtiene la galería mediante `content_id`; conserva ambos conteos.
No debe inferir nuevos grupos, scores, timestamps ni membresías por proximidad.

## Previews, sincronización y publicación

`--include-previews` genera un JPEG RGB de máximo 256×256 por contenido único,
reutilizando `FrameReader` para ZIPs y `ImageSource` para frames de video. Ambos
verifican SHA256 contra la fuente; no se extraen ZIPs completos. Un preview
ausente/inválido produce `media_available=false`, campos de archivo null y
motivo explícito; la membresía permanece. Errores de escritura del destino sí
cancelan la publicación. Sin la opción, todas las entradas dicen `not_requested`.

Las imágenes permanecen en `artifacts/frontend/organization-media/`, ignorado
por Git, con `media-receipt.json`. No hay cambio de `.gitignore` ni autorización
implícita de publicación de imágenes. Sincronizar, después del export real:

1. Los JSON/schema hacia el directorio de datos local que configure el frontend.
2. El directorio de media hacia su raíz local de previews; resolver
   `media.relative_path` respecto a esa raíz independiente.
3. Verificar SHA256 del manifest y de las imágenes disponibles. Si falta media,
   usar placeholder y mantener visibles las relaciones y metadata.

La sincronización no está automatizada ni se modifica `flir-pipeline-explorer`.
No se generan Parquet, embeddings, pesos, imágenes originales ni reportes HTML
en el contrato. Este cambio solo versiona código, tests, schema y documentación.

Se escriben directorios temporales hermanos y se validan todas las relaciones
y fuentes antes del reemplazo. Se reutiliza el mecanismo de publicación del
reporte detector con rollback ante errores ordinarios de I/O. Los destinos no
pueden superponerse a fuentes, entre sí ni contener archivos ajenos. Las fuentes
se hashean antes de consumirlas y justo antes de publicar. Los originales nunca
se modifican. Un solo escritor por destino; ante un corte abrupto entre renames,
inspeccionar los `.organization-stage-*`/backups antes de reintentar.

Orden estable por IDs, estrategias, seeds, particiones e índices. Con los mismos
bytes upstream y entorno, los JSON científicos son byte-estables salvo
`manifest.json` (fecha/procedencia). Las miniaturas son deterministas en el mismo
entorno Pillow. No se garantiza igualdad binaria entre versiones de codecs.

## Validación local

Los tests nuevos cubren ocurrencias duplicadas, leakage histórico exacto,
seeds múltiples, grupos C10/C12 sintéticos, ruido/probabilidades, componentes,
índices no contiguos, zonas inclusivas, videos múltiples, media ausente/generada,
referencias inválidas, IDs duplicados, corrupción, determinismo y rollback.
Se comprueba que el exportador no llama fitting, generadores de splits,
constructores de componentes ni evaluadores numéricos de clustering/splitting.

```powershell
$env:UV_PROJECT_ENVIRONMENT = '.venv-review'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:OMP_NUM_THREADS = '1'
$env:NUMBA_NUM_THREADS = '1'
uv run --no-sync python -m pytest tests/test_organization_export.py tests/test_organization_linkage.py -q
uv run --no-sync ruff check .
uv run --no-sync python -m pytest -q
uv run --no-sync python -c "from flir_pipeline.cli import app; app()" explorer export-organization --help
```

No hay type checker adicional configurado. La validación sintética no demuestra
que el freeze local sea el final ni que existan las membresías finales en este
clon. No se ejecutó un experimento FLIR ni se publicaron datos reales.

Validación focalizada de esta implementación (2026-10-04): **34 passed** en
87.72 s, con schema independiente, fuentes intactas, previews y rollback.
Ruff global y formato de los cinco archivos Python nuevos aprobados; ayuda CLI,
siete notebooks fuente y `git diff --check` aprobados. Windows bloqueaba una DLL
de pandas en los entornos preexistentes: se usó `.venv-review` con pandas 2.3.3,
NumPy 2.3.5, PyArrow 19.0.1 y Pydantic 2.13.5. No se cambiaron dependencias del
proyecto ni el lockfile; estas versiones corresponden al entorno local de prueba.

Regresión completa con las variables offline anteriores:
`uv run --no-sync python -m pytest -q --basetemp C:/flir-org-full-final`:
**838 passed**, 40 FutureWarning de pandas en `clustering/selection.py` y
`detection/association.py`, ambos preexistentes y sin cambios en esta tarea.
Tras precisar el vocabulario de `annotation_consensus` sin equiparar diferencias
de bytes a conflictos semánticos, se repitió el bloque afectado: **34 passed**
en 85.13 s; Ruff global y formato aprobados nuevamente. La suite general no se
repitió para ese ajuste limitado de terminología/schema.
