# Integración local de artifacts de M02

Infraestructura implementada para lectura de artifacts existentes de `flir-leakage-pipeline`. DEMO sigue siendo el modo predeterminado. REAL muestra M02 desde artifacts y mantiene M01/M03 en DEMO: el workspace se identifica como **MIXED**. No hay API, entrenamiento, ejecución de jobs ni integración real de los otros repositorios.

```text
Manifest + artifacts científicos existentes (read-only)
  → tools/export_leakage_snapshot.py
  → LeakageSnapshotV1 / LeakageSnapshotV2 (JSON saneado, local e ignorado)
  → Zod: src/contracts/leakage.ts + leakage-v2.ts + research.ts
  → RealLeakageAdapter → CompositeDataAdapter
  → TanStack Query → rutas existentes de organización
```

## Generar el snapshot

Requisitos: Python 3.11+ y `uv`. El script declara versiones de NumPy, pandas y PyArrow mediante PEP 723; no importa el pipeline científico, torch, Transformers ni modelos. La instalación inicial requiere acceso a dependencias públicas. Los tests y la exportación no descargan datos ni modelos.

Ejemplo PowerShell con **placeholders**, desde este frontend:

```powershell
uv run tools/export_leakage_snapshot.py `
  --manifest "<ARTIFACT_ROOT>/manifest/flir_video_samples_v1.parquet" `
  --dinov2-features "<DINO_FEATURE_RUN_DIR>" `
  --dinov2-similarity "<DINO_SIMILARITY_V2_RUN_DIR>" `
  --dinov2-reduction "<DINO_REDUCTION_BENCHMARK_DIR>" `
  --clip-features "<CLIP_FEATURE_RUN_DIR>" `
  --clip-similarity "<CLIP_SIMILARITY_V2_RUN_DIR>" `
  --clip-reduction "<CLIP_REDUCTION_BENCHMARK_DIR>" `
  --output "public/runtime/leakage-snapshot.json"
```

Solo `--manifest` y `--output` son obligatorios. Preferir omitir etapas ausentes. Una etapa omitida queda pendiente. Cualquier ruta proporcionada que no exista o no valide **hace fallar toda la exportación**; no se degrada silenciosamente a pending. Similarity requiere features del mismo encoder; reduction requiere ambos. Los inputs declarados de manifest/features/similarity/reduction deben existir y validar. Nunca se busca el run más reciente.

`--*-reduction` acepta un directorio de run o de benchmark. Para un run individual se exportan todas sus coordenadas sin declararlo candidato. Para un benchmark se leen únicamente los runs enumerados en su metadata y las selecciones persistidas en `candidates.csv`; se exportan métricas de todos ellos y todas las coordenadas de los candidatos declarados. No se recalcula selección, estabilidad, métricas ni reducción. Se comprueba la referencia y el checksum de metadata de cada run. Un run sin coordenadas exportadas mantiene su ficha de métricas y un estado explícito de visualización no disponible.

Si existen etapas posteriores **compatibles con el mismo dataset**, añadir sus carpetas de run:

```powershell
# Opciones adicionales del mismo comando de exportación:
# --clustering "<CLUSTER_RUN_DIR>"
# --splitting "<SPLIT_RUN_DIR>"
# --detection "<COMPLETE_DETECTOR_RUN_DIR>"
```

Estas tres opciones son repetibles. No aceptar un split histórico de otro dataset. Clustering sobre una reducción requiere que esa reducción esté en la exportación; para original L2, la galería, las etiquetas y la trazabilidad pueden mostrarse sin scatter. Un detector requiere un split exportado y compatible. Los pilotos o formatos no soportados se rechazan: no se reinterpretan como métricas finales.

El archivo de salida debe ser nuevo y estar fuera de las carpetas de entrada. La publicación usa un archivo temporal y enlace exclusivo, sin sobrescribir snapshots anteriores. Para actualizar, escoger otro nombre y configurar la URL correspondiente. No se modifica ni borra ningún artifact fuente, `.partial`, ZIP, imagen o peso. La salida es la única escritura del exportador, aparte de su temporal de publicación.

En Hypatia se puede ejecutar el mismo script con rutas explícitas del servidor y un `--output` fuera del árbol de artifacts. Transferir **solo ese JSON saneado** a `public/runtime/` local por el mecanismo autorizado del proyecto. El frontend no se conecta a Hypatia ni gestiona credenciales o transferencias.

## Arrancar en REAL/MIXED

```powershell
$env:VITE_DATA_MODE = "real"
$env:VITE_LEAKAGE_SNAPSHOT_URL = "/runtime/leakage-snapshot.json"
npm run dev
```

Abrir `http://127.0.0.1:5173/organization/reduction`. El selector **Data mode** permite volver a DEMO y probar REAL sin cambiar archivos. El modo no se persiste en localStorage. Una recarga vuelve al valor de entorno. `.env.example` documenta los valores; no se crea ni versiona un archivo personal de entorno.

`public/runtime/*` está ignorado, salvo `.gitkeep`. Vite tiene `build.copyPublicDir: false`: no copia snapshots locales a `dist`. La integración real se prueba con el servidor local, no mediante una publicación pública del snapshot. Los artifacts saneados siguen siendo información de investigación; anonimizar nombres no autoriza publicarlos.

## Alcance y evidencia

| Etapa          | Datos consumidos y visualización                                                                                                                  |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| Manifest       | Solo `flir_video_samples_v1`; ocurrencias, contenidos únicos, duplicados y todas las membresías por video                                         |
| DINOv2 / CLIP  | Metadata, content/record index, calidad; raw y L2 se inspeccionan en bloques, sin enviarlos al navegador; dimensiones 384/512 y pooling explícito |
| Similarity v2  | Resumen global, cuantiles, top-k por contenido, resúmenes por relación de fuente y gaps; no matriz N×N ni histogramas inventados                  |
| t-SNE / PaCMAP | Coordenadas persistidas, métricas, configuración/semilla, selección de candidatos y tablas de estabilidad del benchmark                           |
| Clustering     | Etiquetas, medoids reportados, métricas numéricas admitidas, ruido -1, scatter si hay coordenadas compatibles, galería sin imágenes y procedencia |
| Splits         | Membresías por ocurrencia; cobertura y atomicidad comprobadas; BEFORE/AFTER solo con un split cluster-aware compatible                            |
| Detector       | Métricas globales admitidas de un run COMPLETE ligado al split; sin pesos ni predicciones privadas; pilotos rechazados                            |

Los campos numéricos se exportan mediante lista permitida. No todas las métricas anidadas upstream están normalizadas: una métrica no soportada no se inventa ni se convierte en cero. Ausencia de etapa → **pending/unavailable**, sin fallback a mock dentro de M02. M01/M03 conservan sus fixtures independientes, no reciben el dataset real.

**`export-validated` no equivale a verificación científica.** Se valida identidad del manifest, coincidencia dataset/feature/similarity/reduction, alineación de índices, cobertura completa, invariantes raw/L2, checksum de archivos consumidos y referencias entre etapas. Se verifica la firma de procedencia de similarity, porque `dataset_id` por sí solo no codifica cambios temporales. Los receipts de features y de metadata/vecinos de similarity se comparan con los de reduction. La matriz coseno no se lee: su hash declarado se enlaza entre receipts, sin validar sus bytes. No se recalculan distancias, métricas, clustering, splits ni evaluación del detector. Las métricas son **reported**; el resultado de exportación no reemplaza los verificadores ni los metadatos originales del pipeline.

La temporalidad conserva `sampleIndex` y `gridSeconds` persistidos. `sequenceId` y `captureTimestamp` permanecen `null`. Las múltiples ocurrencias de un contenido pueden pertenecer a distintos videos; no se selecciona una fuente representativa como si fuera la única. Source video ≠ sequence; cluster ≠ sequence; proximidad temporal no confirma leakage.

## Privacidad y límites

No se exportan rutas, usuarios, nombres de archivo, texto libre de metadata, hashes originales de contenido, imágenes, videos, labels, arrays de embeddings, pesos, tokens ni trazas de excepciones. Los IDs son aliases ordinales deterministas para la misma selección de entrada (`content-000001`, `frame-000001`, etc.). Se validan los IDs originales **antes** de sustituirlos. Los aliases están acotados a un snapshot: no sirven para unir exportaciones de datasets diferentes ni sustituyen los IDs científicos en el registro experimental.

Contrato estricto y errores públicos seguros: `snapshot-missing`, `snapshot-malformed`, `schema-mismatch`, `identity-mismatch`, `snapshot-unavailable`. Solo se admiten URLs del mismo origen. Un error ofrece reintento y retorno explícito a DEMO; nunca intercambia resultados silenciosamente.

Límites explícitos: 20.000 contenidos, 100.000 ocurrencias, 256 runs, 1.000.000 aristas top-k por run y 80 MiB por snapshot. Si se exceden, se rechaza la exportación; no hay submuestreo silencioso. Canvas renderiza las proyecciones; las tablas paginan 20–25 filas y los selectores de vecinos muestran 100 coincidencias buscables. Se prueba con 9.000 contenidos sintéticos. Esto verifica infraestructura y responsividad, no caracteriza un espacio de embeddings real.

## Validación y estado

Tests sintéticos compartidos entre exporter y Zod en `src/test/fixtures/leakage-synthetic.json`. Su generador `tools/tests/make_fixture.py` crea datos ficticios en temporales; nunca lee artifacts privados. El test Python compara su salida con la fixture, y TypeScript valida y representa exactamente ese contrato.

```powershell
uvx ruff==0.16.9 check tools
uvx ruff==0.16.9 format --check tools
uv run --with numpy==2.2.6 --with pandas==2.2.3 --with pyarrow==20.0.0 --with pytest==8.4.2 python -m pytest tools/tests -q
npm run lint
npm run typecheck
npm run test
npm run build
npm run format:check
npm run test:e2e
```

Inspección de contratos upstream: commit `e59400b4d797cff80f0eaede8997e37a2db6a4bb`, clon temporal ignorado y read-only. No se modificó `.references`. La implementación admite sus formatos, pero **no se ejecutó una exportación con artifacts privados** ni se certificó el estado de los experimentos en Hypatia. Siguiente comprobación operativa: ejecutar el exportador con las rutas explícitas del usuario y verificar el snapshot local. No se debe marcar clustering/splitting/evaluación científica como completos por la existencia de esta interfaz.

## Evolución V2 y artifacts temporales

Sin nuevas opciones se conserva la salida V1. `--schema-version v2` añade `research` con etapas pendientes. Proporcionar cualquier argumento nuevo selecciona V2 automáticamente. El snapshot conserva íntegramente `contents`, ocurrencias e índices V1.

| Argumento                           | Input explícito admitido                                                                                                              |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `--sequences`                       | Publicación `sequence_boundary_candidates` o `sequence_instance_set`                                                                  |
| `--sequence-experiment` (repetible) | `sequence_experiment_suite_v1` / `sequence_clustering_experiment_v1`                                                                  |
| `--sequence-evidence` (repetible)   | Evidencia normalizada `sequence_structure_review_v1` / `sequence_external_evidence_v1`, incluida ingesta `hypatia_legacy_evidence_v1` |
| `--linkage`                         | `labeled_video_link_candidates`, con sequence set exportado compatible                                                                |
| `--linkage-review`                  | `labeled_visual_dependency_manual_calibration`, unido al linkage y sequence set suministrados                                         |
| `--linkage-aggregate`               | Agregación de revisión con historial enlazado al linkage y revisión explícitos                                                        |
| `--dataset-variant` (repetible)     | Declaración JSON de variante del mismo dataset                                                                                        |

Ejemplo adicional al comando con manifest y ambos feature stores:

```powershell
# --schema-version v2
# --sequences "<EXISTING_SEQUENCE_PUBLICATION>"
# --sequence-experiment "<EXISTING_SUITE_PUBLICATION>"
# --sequence-evidence "<EXISTING_NORMALIZED_EVIDENCE>"
# --linkage "<EXISTING_LINKAGE_PUBLICATION>"
# --linkage-review "<EXISTING_MANUAL_CALIBRATION>"
# --linkage-aggregate "<EXISTING_REVIEW_AGGREGATE>"
# --dataset-variant "<EXISTING_VARIANT_JSON>"
```

`tools/export_research.py` consume nombres de archivos fijos, verifica checksums de los archivos consumidos y comprueba al finalizar que no cambiaron. Verifica IDs originales de dataset/features antes de crear aliases. Sequence sets conservan cobertura completa de occurrences e intervalos; linkage mantiene todas las membresías del contenido de video y todas las occurrences etiquetadas. La identidad del dataset etiquetado se reconstruye desde las columnas de identidad de su manifest persistido, sin abrir imágenes ni labels.

No se invoca `sequences verify`, `linkage verify`, fitting, imports científicos ni replay computacional. `consumed-files-validated` no equivale a verificar todos los archivos de la publicación ni sus resultados. No se abren media, contact sheets, fuentes originales de evidencia, credenciales, pesos o ZIP. El frontend tampoco interpreta archivos Parquet directamente.

Los runs experimentales exportan vocabulario inspeccionado y métricas numéricas admitidas de las tablas persistidas. Campos sin una medida compatible permanecen null; no se promedian masks ni comparaciones entre runs para inventar agreement/stability global. Los límites o esquemas no soportados fallan explícitamente. La evidencia legacy de un dataset histórico no se puede unir a un manifest sampled-video distinto.

Limitaciones: no imágenes; no confirmación de enlaces exactos en el productor actual; recurrencia sin score compatible conserva marcador con score null; variante como inspección de declaración, no cambio automático de población ni conclusión causal. No se ejecutó una nueva exportación con fuentes privadas. Sí se validó estructuralmente el snapshot local V1 existente; tamaño y preservación en [SNAPSHOT_PERFORMANCE](SNAPSHOT_PERFORMANCE.md).

Para servir un snapshot montado en contenedor y cambiar su URL sin rebuild, ver [DOCKER](DOCKER.md).
