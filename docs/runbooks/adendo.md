# Consolidación candidata de datos etiquetados

`flir-pipeline data audit-adendo` implementa preparación y auditoría, sin construir
splits, grupos de dependencia, embeddings ni entrenamientos. Lee los ZIP mediante
`ZipInfo` sin extracción. Los resultados son evidencia candidata, no un dataset
aprobado para entrenar. No modifica el manifiesto histórico ni sus consumidores.

## Ejecución

En PowerShell, establecer `FLIR_DATA_ROOT` y `FLIR_AUDIT_ROOT` fuera del repositorio:

```powershell
uv run flir-pipeline data audit-adendo `
  --images-zip "$env:FLIR_DATA_ROOT/Imagenes.zip" `
  --labels-zip "$env:FLIR_DATA_ROOT/Etiquetas.zip" `
  --adendo-zip "$env:FLIR_DATA_ROOT/Adendo.zip" `
  --output "$env:FLIR_AUDIT_ROOT/run-01" `
  --evidence "$env:FLIR_AUDIT_ROOT/prior_evidence.json" `
  --phash-threshold 8
```

`--evidence` es opcional; los tres ZIP y `--output` son obligatorios. El directorio
de salida debe ser nuevo. No hay sobrescritura ni reanudación automática. Ante un
fallo de escritura, una carpeta sin `summary.json` no constituye una publicación
completa; inspeccionarla y usar otra salida. Los originales se hashean antes y
después de la lectura. Nunca se escriben ni se extraen imágenes o ZIP.

## Contratos e identidades

- `candidate_manifest.parquet` y `candidate_manifest.json`: una fila por imagen,
  incluida cualquier ocurrencia duplicada o con QA fallido. `frame_id` usa la
  receta histórica (nombre del ZIP, miembro normalizado, SHA256 de imagen).
  `content_id = image_sha256` identifica contenido exacto. Ambos locators,
  normalizado y literal (`zip_member_name`), quedan registrados con offset ZIP.
- `annotation_audit.json`: todas las etiquetas, incluidos huérfanos, validación
  YOLO, cajas a precisión original, conteos y bytes originales codificados en
  base64. `label_id` identifica cada ocurrencia de etiqueta; nunca se fusionan.
- `relations.json`: aristas entre `frame_id` con `content_id` en ambos extremos,
  evidencia, puntuación, distancia pHash observada, diferencias de anotación y
  estado de revisión. Una pareja puede tener varias evidencias atribuidas.
- `summary.json`: recibo final con conteos, advertencias, configuración, identidad
  de dataset/auditoría, versiones, commit base, hashes del código utilizado,
  hashes de inputs y checksums de los otros cinco archivos. No incorpora rutas
  absolutas ni fechas variables; la repetición con iguales inputs/código/entorno
  produce iguales artefactos. `README.md` acompaña cada publicación.

Los JSON tienen `schema_version = flir_labeled_consolidation_candidate_v1`,
`dataset_id` y una lista `records`; el resumen utiliza el mismo `schema_version`.
Se reutiliza `dataset_id_from_manifest`; el versionado distingue este universo
del histórico. El `audit_id` también liga ZIP completos, evidencia, configuración
y versión del código, incluyendo etiquetas huérfanas y ambiguas que no participan
en la receta histórica de `dataset_id`.

El histórico se empareja exactamente por `(split normalizado, basename casefold)`
usando las funciones del constructor oficial; carpetas de imágenes y etiquetas
pueden diferir. Adendo se empareja por basename dentro de su propia cohorte. Su
`original_split` queda nulo, incluso si su empaquetado incluye nombres de split;
`source_folder_split` conserva ese dato sin convertirlo en una asignación.

`label_ids` conserva **todos** los candidatos de emparejamiento. Múltiples imágenes
o etiquetas con una misma clave quedan `ambiguous`; no se elige la primera.
`label_sha256` queda vacío cuando falta una asociación unívoca, y `label_valid`
es falso. Los huérfanos permanecen en la auditoría, fuera de la población de
imágenes. Las cajas inválidas no se corrigen; las etiquetas vacías son válidas.
Las clases ajenas al catálogo se registran separadamente en `unknown_class_ids`;
el validador YOLO reutilizado comprueba sintaxis/geometría, no ontología.

Entradas ZIP repetidas o con aliases normalizados ambiguos, paths absolutos,
traversal y symlinks se rechazan explícitamente antes de publicar: no se fabrican
identidades ni se publica un subconjunto como si fuera completo. ZIP ilegibles o
CRC inválidos también abortan. Una imagen ilegible dentro de un ZIP legible se
conserva con SHA256 y QA fallido, sin pHash.

## Evidencia y revisión

| Estado | Interpretación |
|---|---|
| `exact_bytes` | Igualdad SHA256 calculada sobre los bytes originales; revisión `confirmed_by_bytes`. |
| `visual_candidate` | Candidato perceptual o importado, revisión pendiente. |
| `confirmed_provenance` | Confirmación atribuida a evidencia adicional importada; no validada automáticamente. |
| `rejected` | Relación rechazada mediante revisión explícita importada. |
| `unresolved` | Relación registrada aún sin resolver. |

Todas las relaciones mantienen `split_constraint_applied = false`. Importar una
revisión no elimina evidencias previas ni resuelve revisiones contradictorias.
La futura construcción de grupos necesita una política explícita para resolverlas.
`annotation_conflict` compara la representación canónica del validador únicamente
cuando ambos emparejamientos son unívocos y válidos; en otros casos queda nulo.
La comparación canónica redondea a diez decimales como en el constructor existente;
los bytes, orden y precisión original siguen conservados en la auditoría.

Ejemplo **sintético** de evidencia externa:

```json
{
  "schema_version": "adendo_evidence_v1",
  "relations": [{
    "adendo_basename": "supplement_001",
    "historical_basename": "historical_001",
    "historical_split": "val",
    "state": "visual_candidate",
    "source": "informe previo identificable",
    "method": "inspección visual reportada",
    "score": null,
    "notes": "No establece identidad temporal"
  }]
}
```

Los endpoints deben resolver exactamente una ocurrencia cada uno. Los estados
`confirmed_provenance` y `rejected` requieren `reviewer`, `reviewed_at` y
`additional_evidence` no vacíos. Se conserva la evidencia importada y su SHA256;
no se inventan scores ausentes ni se permite importar `exact_bytes`.

El cribado reutiliza `features.diagnostics._phash`: gris 32×32, DCT-II ortonormal,
bloque 8×8, mediana de `[1:,1:]` y 64 bits. Se registra la implementación y sus
versiones. Compara contenidos Adendo–histórico y conserva todas las parejas bajo
el umbral inclusivo, expandiéndolas a todas las ocurrencias históricas, sin top-1.
La distancia Hamming no es una probabilidad. El valor 8 es una configuración de
cribado, no un umbral de equivalencia validado. La ausencia de candidatos no
demuestra independencia. Las relaciones exactas se registran en cualquier cohorte.

Los seis vínculos aportados por el responsable se importan fuera del repositorio
como candidatos; el desplazamiento reportado se conserva solo en su evidencia,
sin convertirse en regla general. Los `scene`, índices y secuencias derivados de
nombres no equivalen a procedencia ground truth; no se inventan timestamps.

## Reutilización futura

El mapping `frame_id → content_id` permite extraer features una sola vez por
contenido. `label_ids → annotation_audit` permite calcular distribución de clases
sin introducir etiquetas en embeddings. Los endpoints de relaciones permiten
revisar posibles dependencias sin contaminar el clustering con `original_split`.

Este contrato aún **no es entrada directa del splitter existente**: este exige
metadata histórica completa para sus targets. Hay que decidir explícitamente
cómo incorporar Adendo, resolver conflictos de anotación y adjudicar relaciones,
sin asignarle un split histórico ficticio. `split_ready` es siempre falso en esta
fase; `annotation_qa_passed` solo resume lectura/validación del universo de imágenes
y no resuelve conflictos entre anotaciones, huérfanos, ontología ni dependencia.

El resumen distingue `exact_duplicate_groups`, todas las ocurrencias dentro de
esos grupos, las ocurrencias adicionales respecto de un representante y las
parejas exactas. También separa parejas candidatas únicas, registros pHash y
evidencias importadas: varios registros pueden describir una misma pareja.

Pruebas offline con ZIP sintéticos: `uv run pytest tests/test_adendo.py
tests/test_manifest.py tests/test_inventory.py tests/test_identity.py
tests/test_splitting.py -q`. No requieren modelos, Internet ni GPU.
