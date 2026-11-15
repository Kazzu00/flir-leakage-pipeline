# Preparación curada de desarrollo y auditoría de admisión

Estado: **DEVELOPMENT_NOT_ADMITTED**. Contrato técnico implementado;
extensión de admisión científica propuesta, no implementada ni autorizada.

## Auditoría local del 27 de octubre de 2026

Base inspeccionada: `a6dab04`, rama original `feat/yolo11m-signed-execution`.
Trabajo aislado en `feat/curated-development-preparation`. El directorio local
no versionado `FETCH_HEAD/` era preexistente y no se modificó.
En este backend las fuentes operativas son `docs/status.md` y
`docs/pipeline_traceability.md`; no los nombres equivalentes del repositorio anterior.
La aceptación CUDA de Hypatia fue informada por el usuario; no se repitió aquí.

Se inspeccionaron `common_executor`, `scientific_executor`, `admission`,
`materialization`, CLI, pruebas y contratos existentes. `records_identity` liga
filas completas; `load_publication` exige treinta brazos y 372 ocurrencias de test;
`context` une plan, preparación, pesos y admisión. La admisión exige decisiones
atribuidas, test APPROVED, matriz completa y checkout/entorno congelados.
`check_test_isolation` exige aislamiento del sistema operativo, no solo YAML sin test.
Ninguno de estos controles cambió.

La comparación local de metadata con checksums de la publicación original mostró:

| Propiedad | Original ampliado, seed 0 | Canónico curado |
| --- | ---: | ---: |
| Ocurrencias de desarrollo | 1192 | 1135 |
| Contenidos únicos | 1135 | 1135 |
| S1 train / val | 1093 / 99 | 1044 / 91 |
| S2 train / val | 1093 / 99 | 1065 / 70 |
| S3 train / val | 1093 / 99 | 1052 / 83 |

Se verificaron los contenidos, el representante de menor `frame_id`, los hashes
de imagen/anotación y los localizadores originales de cada representante contra
`development_expanded.parquet`. Las tres asignaciones coinciden por contenido con
`development_content_memberships.parquet` de los brazos ampliados seed 0.
Las 57 ocurrencias redundantes no son nuevos contenidos excluidos. El reporte
de curación describe equivalencia de anotaciones; esta intervención no repitió
la adjudicación visual ni la comparación semántica de todas las copias retiradas.

`protected_records.parquet`, ligado al recibo original, permite comparar solo
`frame_id`, `content_id` e `image_sha256`: 514 ocurrencias, 373 contenidos,
intersección exacta cero. `protected_exclusions.csv` coincide en conjuntos de
identidades; distingue 372 test y 142 cuarentena. Se leyeron metadatos, nunca
imágenes, etiquetas ni galerías protegidas. La comprobación abarca la referencia
suministrada: no certifica su completitud institucional, autorización o aislamiento
ACL/mounts, ni equivale a un replay semántico completo de la publicación.
El test continúa sin aprobación establecida.

Se inspeccionaron `matrix.json`, `provenance.json` y `build_transfer.py` del paquete
de preflight. La matriz propone nueve combinaciones, no índices admitidos. El
builder permite solo pares canónicos y metadata, verifica el recibo fuente y
normaliza metadata TAR. No incluye vistas YAML/listas ni la configuración cuyo
hash declara la matriz; necesita preparación posterior. `protected_data_included=false`
es una declaración de procedencia, no sustituye la comparación con la reserva.
La nueva preparación liga los bytes de esa matriz y procedencia, pero no valida
el checkpoint, el TAR ni el freeze de configuración para entrenamiento.

## Contrato de desarrollo implementado

El comando consume un manifest explícito: `reports/manifest.csv` en el dataset
local o `manifest.csv` tras la transferencia. Conserva sus bytes, IDs, anotaciones
y asignaciones; no reconstruye splits ni inventa columnas de ocurrencias faltantes.
Admite fixtures sintéticos de otro tamaño; el tamaño real se deriva del manifest,
sin usar 1135 como sustituto de una identidad criptográfica.

Reutiliza validación YOLO, ontología original, hashing y escritura comprobada del
backend. Comprueba unicidad de contenido/frame/rutas, identidad content=SHA256,
pares, hashes, decodificación, dimensiones, objetos, etiquetas vacías, cajas
duplicadas y clases. Exige train y val en cada política. La unicidad global de
SHA256 impide cruces exactos. No adjudica etiquetas vacías ni la semántica de SDZI.

Las copias independientes usan `canonical/images/FRAME.ext` y
`canonical/labels/FRAME.txt`. Las tres vistas comparten ese almacén mediante listas;
no usan hardlinks ni symlinks a fuentes. `data.yaml` y listas contienen rutas
POSIX absolutas bajo `--linux-root`, que debe ser el destino real previsto.
No son portables a cualquier ruta sin regenerar la preparación y su recibo.
La generación local valida sintaxis/rutas por construcción; no prueba que exista
ese mount en Hypatia ni ejecuta Ultralytics sobre estas vistas.

`development_receipt.json` incluye contrato versionado, hashes de insumos y todos
los artefactos, hash del módulo implementador, conteos, destino Linux, resultado
de exclusión y bloqueos. `curated_population_id` es SHA256 de la versión y hash
del manifest completo; cambia también si cambia su serialización. No es
`records_id`, `dataset_id` histórico ni ID de admisión. El recibo es determinista,
sin reloj ni rutas locales del host de preparación. No es firma de autoridad.

Una segunda llamada reconstruye el árbol esperado desde las fuentes y compara
todos los archivos sin reescribir. Salidas alteradas, parciales o con archivos
adicionales se preservan y rechazan. Una interrupción puede dejar una salida
parcial no utilizable; usar una ruta nueva después de investigar, no borrar
evidencia automáticamente. El recibo se escribe al final.

Si no se proporcionan ambos argumentos de referencia protegida, el estado es
`PENDING_REFERENCE_MISSING`. Proporcionar solo uno, una referencia vacía, un hash
incorrecto o cualquier cruce exacto bloquea la preparación. Incluso una comparación
exitosa conserva `test_status=NOT_APPROVED` y `ready_for_training=false`.
El mecanismo no sustituye validación de la publicación ni aislamiento operacional.

## Comandos reproducibles

Desde el backend, con variables que apunten a rutas externas existentes:

```powershell
# DATASET: raíz curada; PACKAGE: paquete preflight; PUBLICATION: publicación original.
# OUTPUT: destino local nuevo y externo; LINUX_OUTPUT: ruta absoluta prevista en Linux.
uv run --offline --no-sync flir-pipeline detection common-executor prepare-curated-development `
  "$DATASET" "$DATASET/reports/manifest.csv" "$PACKAGE/matrix.json" `
  "$PACKAGE/provenance.json" "$OUTPUT" --linux-root "$LINUX_OUTPUT" `
  --protected-records "$PUBLICATION/sources/proposal/protected_records.parquet" `
  --protected-receipt "$PUBLICATION/receipt.json"
```

Repetir exactamente el comando verifica idempotencia. Repetir con otra salida
local y el mismo destino Linux reproduce los mismos bytes. Si el launcher no se
puede usar, el entrypoint Python equivalente ejecutado en esta intervención es
`uv run --offline --no-sync python -c 'from flir_pipeline.cli import app; app()'`
seguido de los mismos argumentos. `python -m flir_pipeline.cli` no invoca el CLI
en este checkout y no constituye una comprobación de preparación.

```powershell
uv run --offline --no-sync python -m pytest tests/test_curated_development.py tests/test_common_executor.py tests/test_scientific_executor.py tests/test_detection_cuda_import_order.py tests/test_detection.py -q
uv run --offline --no-sync ruff check .
git diff --check
```

Las pruebas son offline/sintéticas: replay, idempotencia, pairing, CLI, referencias
ausentes, cruces protegidos, manipulación de hashes, cambios de matriz/procedencia,
paths inseguros, test en desarrollo, salidas parciales y rechazo del recibo por la
preparación formal. No descargan modelos ni entrenan.

## Extensión científica propuesta: decisión humana previa

El contrato actual **no admite legítimamente** esta población mediante un cambio
de alias. Deduplicar modifica el peso de las observaciones y los denominadores
de validation; no debe presentarse como el mismo experimento por ocurrencias.
La metadata completa original puede recuperarse de los registros vinculados;
las ocurrencias retiradas solo pueden conservarse como lineage, no fingirse
ejemplos efectivamente entrenados. El manifest solo no reconstruye todos los
registros originales, la aprobación, firmas ni decisiones científicas.

Propuesta mínima futura: contrato versionado y discriminado para población
`curated_content`, integrado en los verificadores existentes tras revisión, con:

1. Manifest/recibo curado, publicación fuente, mapping occurrence→representative,
   reglas de equivalencia de anotaciones y evidencia de asignaciones conservadas,
   todos ligados por SHA256. Identidad nueva del plan y de cada run; rechazo de
   IDs del protocolo anterior aunque compartan contenidos.
2. Decisión atribuida sobre deduplicación y estimando por contenido, conservación
   de vacías, ontología, suficiencia por clase, dependencias y comparabilidad con
   los baselines por ocurrencias. Sin asumir que métricas mejorarán.
3. Test aprobado y congelado con identidad propia explícita, restricciones sobre
   exposición histórica y referencia completa de protección verificada. Mantener
   barrera OS de `check_test_isolation`; no aceptar solo un flag de exclusión.
4. Firma OpenSSH de autoridad externa, vigencia, commit limpio, lock/entorno,
   configuración, pesos/origen, batch/capacidad y presupuesto de exactamente
   S1/S2/S3 × 42/43/44 con split seed 0. El orden de matrix no fija índices.
5. Propósito separado de evaluación: apertura por custodio después de completar
   toda la matriz y congelar checkpoints seleccionados únicamente con validation.
   Resume conserva intentos y vínculo a la admisión nueva.

Pruebas requeridas para esa extensión: firma real válida/tamper/expiración/alcance;
rechazo de publicación o IDs antiguos, deduplicación no aprobada, cambio de
representante/label/split, mapping incompleto, test provisional o accesible al
worker, pesos/config/entorno alterados, matriz incompleta/duplicada y apertura
anticipada del test. Integración Linux con autoridad administrada y aislamiento
real deberá ejecutarse aparte de las pruebas offline. Ninguna de estas decisiones
puede sustituirse por un JSON autofirmado o una modificación del recibo técnico.

No se implementó esta extensión de admisión: falta una decisión científica sobre
la nueva unidad experimental. No hubo entrenamiento, acceso al payload protegido,
commit, push, merge, despliegue ni modificación de fuentes.

## Evidencia de esta ejecución

Preparación real: **1135 contenidos**, conteos de la tabla conservados y exclusión
exacta contra referencia verificada. Replay en otra salida: **2283 archivos
idénticos byte a byte**. Repetición sobre la primera salida: mismos mtimes, ninguna
reescritura. Los **9100 archivos** cubiertos por el recibo fuente verificaron sus
hashes antes y después. Son verificaciones técnicas; no resultados del detector.

Pruebas focalizadas: **91 passed**, incluidas las 23 nuevas; Ruff global PASS;
CLI help y ejecución real PASS; siete fuentes de notebooks limpias y compilables.
Los artefactos reales quedaron fuera de Git en el directorio operacional local
`curated-backend-audit-20261027`: `lineage-audit.json`, `validation.json`, scripts
`audit_lineage.py`/`verify_replay.py` y dos preparaciones con sus recibos.
Los scripts locales contienen localizadores privados y no se versionan.
La ruta `/work/flir/curated-development` es un destino Linux explícito de ejemplo
para esta preparación, no un mount comprobado ni un despliegue ejecutado.

### Trazabilidad de las ejecuciones de pruebas

Las ejecuciones monolíticas iniciadas antes de la reanudación no produjeron un
resumen final recuperable; el segundo log llegó al 42 %. No se contabilizan como
suite aprobada. Para conservar resultados ante interrupciones se ejecutaron los
54 archivos de tests en procesos independientes, con logs y JUnit externos y una
huella conjunta de `src`, `tests`, `configs`, `pyproject.toml` y `uv.lock`.

El primer ejecutor externo omitió crear el padre de `--basetemp` y produjo errores
de setup `FileNotFoundError`. Sus logs se conservan en
`segmented-suite-20261028`; no son evidencia de aprobación ni un fallo de aserción
del backend. El intento corregido usa rutas cortas con padre existente, tres
procesos independientes y reportes en `segmented-suite-20261028-attempt2`.
No se modificaron los tests ni el código científico para corregir ese error.
La colección de referencia contiene 1360 tests en 54 archivos.

**Cierre del 28 de octubre de 2026: 1360 passed, 0 failures, 0 errors, 0 skipped.**
Los 54 archivos terminaron con código de salida cero; duración de la ejecución
por archivos: 786,54 segundos (13 min 6,54 s). El consolidado JUnit coincide con
la colección y se verificaron otra vez los hashes del código registrado. Esto
acredita todos los tests ejecutados por archivo, no una ejecución monolítica
completada. Evidencia externa: `segmented-suite-20261028-attempt2/validation-summary.json`,
`combined-junit.xml` y logs/JUnit individuales. El resumen conserva también hashes
de los reportes y la distinción respecto de intentos interrumpidos o fallidos.

Ruff global, siete fuentes de notebooks y `git diff --check`: PASS. La preparación
real, su replay, idempotencia, integridad de los 9100 archivos del recibo fuente
y comparación contra las asignaciones originales se reverificaron el 28 de octubre.
Las listas Linux también se comprobaron independientemente contra manifest y
pares locales; no se ejecutó el runtime Linux/Hypatia de estas nuevas vistas.

Los scripts externos utilizados para el cierre son `run_segmented_suite_attempt2.py`
y `summarize_tests.py`. Conservan los localizadores privados fuera del repositorio.
Para repetir en el mismo entorno, el primero reutiliza resultados terminados
solo con la misma huella de código; el segundo exige todos los resultados, cero
errores/skips y correspondencia con la colección. Para una ejecución limpia nueva,
usar un directorio nuevo de evidencia y temporales, preservando los anteriores.
