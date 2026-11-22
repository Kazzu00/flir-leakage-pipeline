# Cierre y publicación para revisión — 2026-11-22

Backend: `feat/oe4-closeout-denoised-20261120`,
[PR #10 hacia main](https://github.com/Kazzu00/flir-leakage-pipeline/pull/10).
Frontend independiente: `feat/flir-phase2a-multisplit-alignment`,
[PR #1](https://github.com/Kazzu00/flir-pipeline-explorer/pull/1).
No se fusiona a `main`, no se hace force-push y no se eliminan ramas, stashes,
checkpoints parciales ni antecedentes.

## Alcance publicado

La rama conserva la consolidación previa, implementaciones, contratos, configuración
y scripts Slurm. Se publican los siete commits OE4 posteriores a la candidata
de integración: `fca89dc`, `44b323b`, `c7dda89`, `c0937b8`, `1fd59fd`, `d62ebb1`,
`9d3a1aa`. Se comprobó que `1fd59fd` es la prueba de aislamiento; el cambio de
autorización comunicado está en `c0937b8`. La recuperación permanece en `9d3a1aa`.

Los worktrees y refs se inspeccionaron antes de recuperar cambios. Los archivos
sueltos de evaluación de evidencia y sus documentos del worktree antiguo ya están
incorporados en la candidata; el script actual además parametriza el linkage ID.
Se preservaron esas copias y el checkout anidado `FETCH_HEAD`, sin añadirlo.
Las decisiones previas sobre ramas históricas/equivalencias permanecen en
[final_integration](final_integration.md). El frontend pendiente se publicó en
su propio repositorio; no se copiaron sus nuevos cambios al snapshot embebido.

Se añaden [resultados OE4](results/oe4_20261122/README.md), exportador exacto,
originales CSV/recibo, proyección de seis recibos nativos y pruebas sintéticas.
La [documentación de los tres datasets](final_datasets_20261122.md) conserva
poblaciones/tests/limitaciones distintos y el recibo agregado de empaquetado.

## Evidencia comprobada en este cierre

- SHA oficial del CSV original: `5e690f88ed9a9d9c4f820fd65e660394c99c8ae51109156d2d3646ee15da4bb4`.
- SHA de la tabla final: `4fc8cef72a924103df5d00ee642e24703d74d65498c09da8495f49b3e791c58f`.
- Doce filas originales → seis filas, ocho métricas; 48/48 tokens preservados.
  Auditoría independiente de las cadenas y bytes externos, de Git y de GitHub.
- `records.csv` local: SHA idéntico al registrado por el agregado; 372 ocurrencias,
  983 objetos. Su recibo local anterior tiene bytes distintos del recibo remoto:
  no se confunde la tabla idéntica con la igualdad de los recibos de liberación.
- Seis recibos nativos recuperados: celdas completas, `COMPLETED_EVALUATION`,
  hashes de pesos copiados exactamente, test/freeze/autorización comunes,
  evaluación fijada y selección declarada `last_epoch_50` sin tuning.
- La proyección pública conserva hashes de los originales y excluye mapas privados
  por imagen. No se rehasharon `.pt`, recibos de entrenamiento/freeze ni payloads
  estadísticos. Sus ausencias no se sustituyen por valores inferidos.

## Verificación ejecutada

Entorno local: Windows, Python 3.11.9, pytest 9.1.1, Ruff 0.16.6; uv/lock del
repositorio. Variables HF/Transformers offline y un hilo para OMP/Numba en pytest.

| Comando/comprobación | Resultado observado |
|---|---|
| `uv run --no-sync ruff check .` | Correcto |
| `uv run --no-sync python -m pytest tests/closeout -q` | 97 passed, 6 skipped por Ultralytics opcional |
| `uv run --no-sync python -m pytest tests/test_automatic_executor.py -q` | 28 passed |
| CLI `closeout --help` | Correcto; sin ejecutar sus workers |
| `scripts/check_notebook_source.py` | Siete fuentes limpias y compilables |
| `final_metrics ... --check` | Doce filas → seis, decimales exactos |
| `final_provenance ... --check` | Seis recibos y bindings de metadata correctos |
| Frontend independiente | 115 tests, 61 pruebas de exportadores, E2E 19 passed / 9 skipped; lint/typecheck/build/Prettier/Ruff correctos |

La primera CI global encontró un test que intentaba cambiar una semilla a 43
cuando la semilla seleccionada ya era 43. Se corrigió para mutar el valor real
seleccionado; no se relajó el ejecutor ni se alteraron entrenamientos o métricas.
El resultado global posterior y el estado del commit concreto se consultan en
los checks del PR; esta tabla no los sustituye ni afirma CI global verde.

El workflow `Explorer quality` del snapshot `apps/explorer` conserva un fallo de
Prettier preexistente en siete archivos, documentado en la integración previa.
No se desactiva el check ni se aplica un refactor del snapshot para este cierre.
Sus jobs Docker/exporter aprobaron. La CI del frontend independiente se comprobó
en verde en `f255fdf`; son checkouts distintos y no se intercambian conclusiones.

## Privacidad y límites

Revisados `.gitignore`, archivos trackeados, diffs/staging, extensiones de datos,
patrones de credenciales y tamaños. Los cambios frente a `main` no añaden archivos
de más de 2 MB ni datasets/binarios/entornos. Los originales CSV/recibo conservan
sus bytes por `.gitattributes`; la proyección JSON usa LF determinista.

Permanecen externos: tres ZIP finales, imágenes/vídeos/labels, manifiestos e
inventarios por frame, hashes de imágenes, embeddings, pesos, caches, `.env`,
credenciales, recibos nativos completos (contienen mapas privados), trazas y
outputs operativos. Solo se publican agregados expresamente solicitados, contratos,
código y documentación. No hay enlace de Drive inventado.

No se entrenó, evaluó, reorganizó el dataset ni modificó Hypatia. Los resultados
son una comparación interna descriptiva de una semilla, sin causalidad ni
independencia estadística. Publicación de código/resultados no implica aprobación
científica de las variantes ni integración panóptica completada.
