# Reproducción

Python 3.11 y uv. Ejecutar desde la raíz del repositorio con los archivos de esta
rama. Los datos y las publicaciones van a directorios externos nuevos. Los comandos
usan la entrada Python de la CLI para evitar el bloqueo del launcher `.exe` observado
en Windows; `uv run flir-pipeline ...` es equivalente cuando el launcher está habilitado.

## Entorno limpio y pruebas

```powershell
uv sync --locked --extra dev --extra reduction --extra alignment
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:OMP_NUM_THREADS = '1'
$env:NUMBA_NUM_THREADS = '1'
uv run ruff check .
uv run python -m pytest tests/test_general_pipeline.py -q
uv run python -m pytest -q
uv run python -c "from flir_pipeline.cli import app; app()" general --help
```

La instalación descarga paquetes; los tests no descargan datos ni modelos y no
necesitan GPU. El test de decodificación FFV1 requiere el extra `alignment`; sin
PyAV se omite únicamente ese test. El core de ingesta/splitting no requiere PyAV,
CLIP ni DINOv2. La CI preexistente instala core/dev/reduction/alignment en CPU.

## Cuatro casos sintéticos completos

```powershell
$demo = Join-Path $env:TEMP 'general-pipeline-demo-new'
uv run python scripts/general_synthetic_demo.py $demo
uv run python -c "from flir_pipeline.cli import app; app()" general verify `
  "$demo/two_classes_no_video/publication" "$demo/two_classes_no_video/inputs" `
  --report "$demo/two_classes_no_video/replay-independent.json"
```

El directorio `$demo` debe no existir. El script genera imágenes/etiquetas nuevas,
vídeos FFV1 reales en los escenarios pertinentes, 40 resultados de política
(incluidos los inviables), informes y replay. No usa datos FAC.

## Dataset de imágenes propio

```powershell
$dataRoot = 'D:/research/my-dataset'
$publicationRoot = 'D:/research/my-new-publication'
uv run python -c "from flir_pipeline.cli import app; app()" general ingest `
  configs/general_images.example.yaml $dataRoot "$publicationRoot/manifest.json"
uv run python -c "from flir_pipeline.cli import app; app()" general build `
  "$publicationRoot/manifest.json" configs/general_pipeline.yaml $dataRoot `
  "$publicationRoot/proposal"
uv run python -c "from flir_pipeline.cli import app; app()" general verify `
  "$publicationRoot/proposal" $dataRoot --report "$publicationRoot/replay.json"
```

Adaptar la lista de imágenes y el catálogo del ejemplo antes de ejecutar.
Cada ocurrencia necesita un ID propio. Etiquetas ausentes se declaran `null`;
archivos de etiquetas vacíos se conservan como tales. No se supone organización
previa train/val/test. Los ratios son configuración explícita.

Si hay vídeos checksummed y se desea localizar imágenes locales:

```powershell
uv run python -c "from flir_pipeline.cli import app; app()" general localize `
  "$publicationRoot/manifest.json" $dataRoot `
  "$dataRoot/operations/new-localization.json" "$publicationRoot/localized.json"
```

Construir después desde `localized.json`. La salida de evidencia es nueva y se
ubica en un área operacional de la raíz; las imágenes/vídeos originales son
read-only. No haber coincidencia exacta conserva estado desconocido.

## FAC con evidencia existente

El paquete privado de cierre incluye `fac_adapter.yaml`, `fac_manifest.json`,
`fac_adapter_report.json`, `fac-final/`, replay y recibo de publicaciones
anteriores. Las rutas del YAML son relativas a la raíz privada suministrada.

```powershell
$dataRoot = $env:FLIR_DATA_ROOT
$closure = Join-Path $dataRoot 'operations/jp-general-pipeline-closure-20261106'
# Para reproducir la adaptación usar nombres de salida NUEVOS:
uv run python -c "from flir_pipeline.cli import app; app()" general adapt-fac `
  "$closure/fac_adapter.yaml" $dataRoot "$closure/reproduced-manifest.json" `
  "$closure/reproduced-adapter-report.json"
uv run python -c "from flir_pipeline.cli import app; app()" general build `
  "$closure/reproduced-manifest.json" configs/general_pipeline.yaml $dataRoot `
  "$closure/reproduced-proposal"
uv run python -c "from flir_pipeline.cli import app; app()" general verify `
  "$closure/reproduced-proposal" $dataRoot --report "$closure/reproduced-replay.json"
```

`FLIR_DATA_ROOT` debe apuntar al padre de `fac`, `operations` y `audits` del paquete,
o pasar esa raíz explícitamente. El adaptador reusa evidencia temporal existente;
no repite el escaneo largo ni hereda tiempos a las variantes procesadas.
`verify` rehasha vídeos, archivos de evidencia, imágenes y etiquetas declaradas,
reconstruye ambas políticas y compara todo el análisis estable. No entrena.

Los hashes de implementación deben coincidir. Si el código cambió, reproducir con
el snapshot y lock registrados en la publicación; no desactivar esa comprobación.

## Congelamiento funcional: cuatro frentes de evaluación

Usar el snapshot de fuentes del paquete de congelamiento y el mismo `uv.lock`.
El harness de regresión necesita además un checkout Git en el commit registrado
en `regression-final/previous-construction-origin.json`; restaurar el snapshot
sobre ese checkout conservando `.git`. El sellado comprueba la rama aislada
`feat/general-pipeline-closure-20261106`. Un directorio con solo el snapshot y sin
historial Git no basta para reconstruir el control anterior.
La reproducción sintética no requiere FAC; la auditoría retrospectiva sí requiere
los artefactos privados originales, incluidos el manifest/inventory locales y
el control preintegración conservado. Nunca sustituir archivos faltantes por
datos o decisiones inferidos. Todas las salidas indicadas deben ser nuevas.

```powershell
$dataRoot = $env:FLIR_DATA_ROOT
$prior = Join-Path $dataRoot 'operations/jp-general-pipeline-closure-20261106'
$package = Join-Path $dataRoot 'operations/scientific-freeze-reproduction-new'
uv sync --locked --offline --extra dev --extra alignment
uv run --no-sync python scripts/closure/leakage_benchmark.py "$package/benchmark"
uv run --no-sync python scripts/closure/regression_audit.py `
  --repo . --prior $prior --output "$package/regression-final"
uv run --no-sync python scripts/closure/fac_retrospective.py `
  --repo . --data $dataRoot --prior $prior --output "$package/fac-audit"
uv run --no-sync python scripts/closure/verify_closure.py `
  --repo . --data $dataRoot --prior $prior --package $package
uv run --no-sync python scripts/closure/package_closure.py `
  --repo . --data $dataRoot --prior $prior --package $package
```

La sincronización offline requiere paquetes previamente disponibles en caché;
no descarga modelos. El comparador usa un único intérprete/dependencias para
ambos brazos, raíces de trabajo equivalentes, variables offline y un hilo.
Conserva los fallos esperados en JUnit: terminar la auditoría no equivale a que
pytest de organización apruebe. Registra el entorno concreto de la ejecución;
la igualdad del lock no sustituye comparar ese entorno si cambia el sistema.

`verify_closure.py` comprueba los 184 archivos previos, los 221 del cierre anterior
y la igualdad de las 175 fuentes congeladas; ejecuta las 77 pruebas pertinentes,
Ruff, CLI y cuatro replays en procesos separados. No vuelve a correr toda la suite
global. `package_closure.py` exige los resultados esperados, captura fuentes,
documentos y hashes, y genera el reporte de cumplimiento sin alterar publicaciones.

En la ejecución registrada, `regression/` es inválido por setup;
`regression-validated/` solo controla CLI porque se detectó que su solver era
igual al candidato. `regression-final/` restaura el MILP anterior directamente
del commit Git registrado, preservando cambios locales ajenos, y contiene la
comparación final. `sealed/` es un intento incompleto de empaquetado y
`sealed-final/` contiene el reporte definitivo. Los archivos generados, fixtures
e informes quedan fuera del repositorio público.
