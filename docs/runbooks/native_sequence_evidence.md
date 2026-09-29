# Ingesta nativa de evidencia histórica de Hypatia

El adaptador `hypatia_legacy_evidence_v1` reconoce las declaraciones `artifact`
de los cuatro reports cuyos esquemas entregó el responsable. Los nombres de
directorios son localizadores; no establecen procedencia. No requiere una
conversión previa ni campos de reviewer/fecha que no existen en los originales.
Las pruebas son sintéticas. La importación de los archivos reales aún debe
ejecutarse y verificarse en Hypatia; no se ejecutó desde este checkout.

| Familia | Archivos consumidos |
|---|---|
| `provenance_decision_v1` | `summary.json` |
| `video11_manual_transition_review_v2` | `summary.json`, `supported_boundary_zones.csv`, `transition_decisions.csv`, `transition_regions.csv` |
| `video11_sequence_structure_v1` | `summary.json`, `boundary_zones.csv`, `index_structure.csv`, `occurrence_structure.parquet`, `sequence_core_candidates.csv`, `structure_summary.csv` |
| `video11_core_recurrence_refined_v1` | `summary.json`, `all_pairs_ranked.csv`, `manual_review_candidates.csv` |

Cada archivo consumido se conserva byte a byte en un snapshot bajo `media/` y
se identifica por SHA256. `native_summaries.json` conserva además los JSON completos;
las tablas `native_*` permiten consultarlos sin modificar los originales. Metadata
registra ruta original, familia, adapter/schema version, query family, checksums,
dataset y variante del source validado, y espacios de features. Las rutas privadas
y los snapshots reales quedan en outputs ignorados; nunca se versionan.

## Validaciones y alcance

- Declaración de artifact y revisión soportada; columnas CSV exactas y las 46
  columnas/tipos Parquet observados. Se admiten los nulos de estadísticas bbox y
  `review_region_id`; los otros campos no admiten nulos en este contrato.
- Flags originales falsos, también anidados y en CSV: ground truth, creación de
  fronteras exactas, instancias, VDGs, splits y confirmaciones. No se promueve
  `assistant_visual_review_with_user_approval`: se conserva verbatim y
  `visual_verification_complete` permanece false en el importado.
- Revisión: 22 transiciones (17 supported/5 unsupported), 16 regiones
  (12 supported/4 unsupported), y 12 zonas compatibles con las regiones.
- Estructura: 910 ocurrencias, 712 índices 1..712, 13 cores candidatos y 12 zonas;
  distribución 870/40, un elemento por índice/ocurrencia, cobertura completa sin
  huecos/overlap y consistencia de IDs/tipos/extremos/regiones entre todas las tablas.
  Counts por elemento, cero contenidos cross-structure y 198 grupos cross-split
  se comprueban contra las filas. Estos números son checks de esta revisión,
  nunca resultados esperados de experimentos futuros.
- Recurrencia: los 78 pares no ordenados de 13 cores, sin pares consigo mismos
  ni duplicados invertidos; 11 candidatos contenidos en la tabla completa.
  Sus valores y ranks CLIP/DINOv2 deben coincidir; no se vuelven a rankear.
- Consistencia conjunta: familia declarada, mismas 12 zonas entre revisión y
  estructura, mismos 13 core IDs entre estructura y recurrencia. Declaraciones
  de dataset incompatibles, fuentes ambiguas y revisiones desconocidas se rechazan.
- Importación ligada al source: cobertura exacta frame/content, índices nominales
  y metadata histórica de identidad. `original_split`, `possible_sequence`,
  `possible_frame_index` y confianza temporal permanecen metadata; no son inputs
  de fitting ni ground truth.

`lineage_supported` para la otra familia permanece en el JSON original, sin
asignaciones de procedencia exacta. La familia no resuelta sigue sin procedencia
asignada. No se deduce procedencia de filenames. Los 13 cores se importan como
`sequence_core_candidate`, con estado candidato y origen legacy; el consumidor
de métricas usa targets candidatos explícitos, nunca `sequence_instance`.
`exhaustive_boundary_review=false`: revisar 22 transiciones no demuestra una
revisión exhaustiva de todos los cortes posibles.

Los reports legacy siguen `unspecified`. El importador rechaza un source marcado
`original_with_hud` u otra variante nombrada; no realiza una migración implícita.
El artefacto conserva `dataset_variant_id` de esa identidad compatible. Una
futura migración explícita requerirá su propia evidencia/versionado.

## Inspección local sin publicación

En PowerShell, desde el repositorio, con una copia local de los reports bajo
`reports` (o cambiando `--root` a su ubicación):

```powershell
uv run --no-sync python -c "from flir_pipeline.cli import app; app()" sequences experiment inspect-real-evidence --family video_11min --root reports
```

Este comando valida los esquemas, archivos y consistencia conjunta sin cargar
features, escribir outputs ni hacer fitting. Si el dataset no está declarado,
`dataset_id`/`dataset_variant_id` quedan null y `canonical_source_verified=false`;
no inventa la identidad de un manifest completo a partir de una familia parcial.
`published=false` y `dry_run=true`. La falta de reports locales es un error
explícito, no un éxito de validación. Para verificar la implementación offline:

```powershell
uv run --no-sync python -m pytest tests/test_native_sequence_evidence.py -q
```

## Importación y verificación en Hypatia

Desde la raíz del repositorio, activar el entorno preparado. Establecer una vez
`FLIR_MANIFEST`, `FLIR_CLIP_FEATURES` y `FLIR_DINOV2_FEATURES` con el manifest
canónico y los stores completos originales. La importación también admite esos
paths en el perfil o descubrimiento por metadata; una ambigüedad exige selección
explícita. `verify` usa las mismas variables de entorno. No relabelar stores.

Para evitar ambigüedades de localización, estos argumentos eligen las cuatro
fuentes exactas sin inferir su semántica de la ruta:

```bash
SOURCES=(
  --select provenance_decision_v1=reports/provenance_diagnostics/provenance_decision_v1
  --select video11_manual_transition_review_v2=reports/provenance_diagnostics/video11_manual_transition_review_v2
  --select video11_sequence_structure_v1=reports/sequence_diagnostics/video11_sequence_structure_v1
  --select video11_core_recurrence_refined_v1=reports/sequence_diagnostics/video11_core_recurrence_refined_v1
)

srun --cpus-per-task=1 --mem=4G --time=00:15:00 \
  flir-pipeline sequences experiment inspect-real-evidence \
  --family video_11min --root reports "${SOURCES[@]}"

srun --cpus-per-task=1 --mem=4G --time=00:15:00 \
  flir-pipeline sequences experiment import-real-evidence \
  --family video_11min --root reports --dataset-variant unspecified \
  "${SOURCES[@]}" --dry-run

EVIDENCE=$(srun --cpus-per-task=1 --mem=4G --time=00:15:00 \
  flir-pipeline sequences experiment import-real-evidence \
  --family video_11min --root reports --dataset-variant unspecified \
  "${SOURCES[@]}" --output reports/sequence_evidence)

srun --cpus-per-task=1 --mem=4G --time=00:15:00 \
  flir-pipeline sequences experiment verify "$EVIDENCE" \
  --family video_11min --dataset-variant unspecified
```

Las lecturas/validaciones Parquet y de features se ejecutan dentro de SLURM,
no en el nodo de login. No hay fitting ni submission de la suite experimental en
estos comandos. Ajustar recursos/partition/account según el sitio si lo exige.

El resultado es `sequence_structure_review_v1` bajo un ID determinista para las
mismas fuentes, localizaciones y versión de software. Una repetición válida
reutiliza la publicación; nunca sobrescribe una publicación distinta/incompleta.
`verify` vuelve a leer los **originales**, compara SHA256 con los snapshots y
reproduce la normalización, intervalos, cores candidatos y máscaras. Si cambia o
desaparece cualquier archivo consumido, falla aun si la copia congelada sigue
intacta. Los consumidores `load_structure` aplican el mismo control. No es una
verificación visual independiente ni una confirmación de dependencia.

Conservar el recibo real y revisar el paquete visual en una revisión nueva e
inmutable antes de cambiar decisiones científicas. La importación no crea
secuencias, VDGs, restricciones de partición ni un split.
