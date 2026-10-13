# Planificación y verificación de test común

Contrato metodológico: [propuesta YOLO11n v1](../protocols/detector_common_test_v1.md).
Estos comandos no usan Ultralytics/torch ni descargan modelos. No producen un
`plan.json` consumible por el detector histórico: publican `proposal.json` en un
namespace nuevo, siempre sin aprobación, freeze ni autorización de entrenamiento.

## Entradas

- Auditoría etiquetada completa: summary, manifest JSON/Parquet, anotaciones y relaciones.
- Suite de splits candidatos: recibo final y todas las membresías verificables.
- Revisión de aceptación: `review_receipt.json`, `results/acceptance.json` y su
  snapshot `before_sha256.json`, ligado a exactamente la auditoría y suite anteriores.
  El dictamen debe ser `APROBADO`; recibos y conteos coherentes de una revisión
  rechazada no habilitan una propuesta nueva.
- Export público `detection-export-v1`: recibo y archivos agregados válidos.
- Configuración histórica `configs/detection/yolo11n.yaml`, solo lectura.
- Opcionalmente ZIP originales y evidencia importada, para revalidar checksums.

Ausencia o corrupción de una fuente aborta sin recibo final. La revisión previa
verifica splits, no otorga aprobación al nuevo protocolo. Los datos originales
siguen fuera del repositorio y las salidas nuevas no pueden solapar sus fuentes.

## CLI

Las variables son rutas locales del operador; no deben versionarse.

```powershell
uv run flir-pipeline detection audit-readiness `
  --audit $audit --splits $splits --review $review --export $verifiedExport `
  --config configs/detection/prospective_test_v1.yaml --output $freshReadiness

uv run flir-pipeline detection plan-common-test `
  --audit $audit --splits $splits --review $review --export $verifiedExport `
  --config configs/detection/common_test_v1.yaml `
  --source-root $env:FLIR_DATA_ROOT --evidence $evidencePath --output $freshProposal

uv run flir-pipeline detection verify-common-test $freshProposal
```

`--reference` puede señalar la configuración histórica a verificar;
`--legacy-artifacts` permite inspeccionar disponibilidad de runs, estadísticas,
views y pesos en un directorio local explícito. Encontrar archivos no equivale a
validarlos: el reporte los marca presentes/no verificados. El export agregado no
provee predicciones por imagen, y no se conecta automáticamente con Hypatia.

Si Windows impide ejecutar el launcher `.exe`, usar la misma entrada Python:
`uv run --no-sync python -c "from flir_pipeline.cli import app; app()" detection ...`.
No requiere desactivar controles de seguridad.

## Exports y transporte Git

Los checksums son sobre bytes exactos. Un checkout con autocrlf puede convertir
LF a CRLF y fallar legítimamente la comprobación del recibo. El verificador **no**
repara ni normaliza estos archivos. Tampoco reescribe los exports científicos.

Si se observa esta situación, comprobar los blobs del commit explícito de Git
contra los hashes publicados. Solo si coinciden, copiar esos bytes exactos a un
directorio externo nuevo, registrar commit, ruta Git, hashes del checkout/blobs
y diferencia de transporte, y usar esa copia como `--export`. Una diferencia
semántica o un blob cuyo hash no coincide sigue bloqueando la fuente. No regenerar
métricas ni cambiar el recibo original para hacer que pase.

En esta intervención los diez archivos referenciados del export coincidieron
con sus blobs en la base `ff461d6`; la diferencia del checkout era exclusivamente
CRLF/LF. La copia y su recibo de transporte permanecen fuera del repositorio.
Esto verifica los agregados exportados, no recupera predicciones ni pesos originales.

## Publicación

`sources/` conserva snapshots exactos de auditoría, suite, revisión, export y
configuración de referencia. Contiene datos privados derivados y nunca se versiona.

- `proposal.json`: identidad, diseño candidato, soporte, evidencia, brazos futuros,
  configuración propuesta del detector y bloqueos. La identidad liga configuración,
  fingerprints de implementación, fuentes y propósito; no depende de la ruta de salida.
- `test_records.parquet`: anchors evaluables candidatos, mismos registros para todos
  los brazos; no son un test aprobado.
- `protected_records.parquet`, `quarantine_records.parquet`: cierre completo y
  resto reservado que no puede liberarse al desarrollo.
- `protection_groups.parquet`: componentes exactas/seleccionadas de toda la población.
- `development_<population>.parquet`: pools fuera de la reserva.
- `development_groups_<population>_<policy>.parquet`: grupos atómicos del diseño,
  todavía sin asignación train/val.
- `provenance.json`: entorno/commit/código, alcance de verificación de ZIP y
  disponibilidad local de evidencia histórica granular.
- `summary.json`: recibo final escrito después de la verificación semántica.

La opción prospectiva sin datos admitidos publica test/reserva/cuarentena vacíos
y estado `UNAVAILABLE`; no fabrica una población externa. Una salida sin
`summary.json` está incompleta, no se reutiliza y se conserva para diagnóstico.

La verificación recalcula fuentes, grupos, cierre, soporte, igualdad del test
declarado entre brazos, pools y blockers; compara todas las tablas persistidas.
Un READY/APPROVED/FROZEN manual se rechaza incluso reparando los checksums del
output. `actual_train_val_assignments_verified=false` distingue el diseño de una
asignación todavía inexistente. No se toma igualdad de test_id como prueba de
que los futuros materializadores preservarán datos: deberán pasar el gate de
asignaciones y un adaptador de entrenamiento nuevo.

## Pruebas

```powershell
uv run --no-sync ruff check .
uv run --no-sync pytest tests/test_common_test.py tests/test_detection.py tests/test_labeled_splits.py tests/test_splitting.py tests/test_detector_final_report.py tests/test_detector_association.py -q --basetemp $freshExternalTemp
```

Fixtures offline y sintéticas; no invocan modelos reales. Cubren test desigual,
contenidos protegidos en train/val, cierre transitivo, anotaciones, clases ausentes,
revisión/freeze falsos, fuentes corruptas, reproducibilidad e incompatibilidad con
el plan histórico. Las regresiones del productor de export usan predicciones
sintéticas identificadas como fixtures, nunca como resultados FLIR.

El estado vigente registra resultados observados y pruebas realmente ejecutadas:
[docs/status.md](../status.md).
