# FLIR Audit & Human Review v1 — registro de validación

Fecha local: 2026-11-02. Alcance: infraestructura de auditoría y revisión humana;
no experimento de clustering, particionamiento o detección.

## Implementación y procedencia

Rama en ambos repositorios: `feat/audit-human-review-v1`.
Backend desde `8a8100ff6ea1796b2eb694f0ad317e57b01431c8`; frontend desde
`74d29fdbf6c9b67f27b679f28608a4bb2772b49c`.
Los cambios locales previos del backend en `organization_contract.py` y
`docs/analysis/repository_audit.md` se conservaron fuera de los commits de esta tarea.

Archivos principales: `src/flir_pipeline/audit/{models,importers,store,server,cli,demo}.py`,
integración `sequences/cli.py`, JSON Schema, pruebas y runbook. Frontend:
`src/data/audit-client.ts`, `AuditReview.tsx`, integración de `SequenceExplorer`,
indicadores de modo en `AppShell`, estilos acotados y E2E. No se añadieron dependencias.

## Resultados observados

Importación local de los **ocho artefactos prioritarios presentes**, repetida
sin duplicación y verificada contra los hashes de los originales:

| Fuente | Registros/observaciones | Candidatos normalizados |
|---|---:|---:|
| Índices históricos por split | 747 | 0 |
| Perfil temporal JSON | 780 transiciones | 122 |
| Candidatos 5 FPS JSON | 215 | 215 |
| Rachas CSV | 79 | 79 |
| ZIP de revisión inicial | 25 | 25 |
| ZIP de contexto extendido | 9 | 9 |
| Dos CSV compañeros | Conservados y contrastados con JSON | 0 adicionales |
| **Total** | **1.855** | **450** |

El perfil declara 781 frames; su coherencia con 780 transiciones se comprobó.
No se extrajeron 781 nuevos frames. Los splits históricos declarados contienen
467/50/230 índices. La clasificación automática de los 215 candidatos conserva
101 `abrupt_stable_to_stable` y 114 `unresolved_low_similarity`.
**450 es un conteo de registros candidatos de métodos/ventanas diferentes, no
450 eventos físicos distintos.** No se fusionan registros por coincidencia temporal.

Se decodificó una lámina de cada ZIP de revisión: 1920×720 y 2304×720, con lectura
acotada. Esto verifica acceso/decodificación, no el contenido científico de las
imágenes. El ledger real contiene **cero revisiones humanas**. El contexto extendido
y las rachas no declaran fuente de vídeo; permanece desconocida.

Ausentes: `video13_alert_episode_groups_gap3.json`,
`video13_review_batch01_classification.json` y
`video13_extended_context_review_findings.json`.
Un primer intento sobre todos los archivos detectó cinco auxiliares no admitidos
(dos ZIP sin manifiesto, validación distribuida, revisión de fronteras y métricas
subsegundo). Quedaron reportados como errores, sin filas parciales.
La importación `--priority-only` los enumera como excluidos y termina sin errores.
No se inventaron adaptadores para los esquemas de archivos ausentes.

## Comandos y pruebas

- `uv run ruff check .`: aprobado.
- `uv run python -m pytest tests/test_audit_review.py -q`: **30 passed** en la
  verificación final del módulo. Incluye JSON/CSV/ZIP corruptos, ZIP traversal,
  opcionales, idempotencia, procedencia, versiones, incertidumbre, conflictos,
  exportación JSON/CSV y manejo de escritura interrumpida.
- Suite focalizada: `uv run python -m pytest tests/test_audit_review.py
  tests/test_sequences.py tests/test_sequence_experiment_review.py
  tests/test_native_sequence_evidence.py tests/test_smoke.py -q`:
  **133 passed en 310,98 s**. Los endurecimientos posteriores del importador y
  el servidor se revalidaron con las 30 pruebas del módulo y el E2E.
- `uv run pytest` fue bloqueado por Windows Application Control (error 4551);
  `uv run python -m pytest` ejecutó el mismo pytest a través del intérprete.
- Suite Python global: se inició y se interrumpió por ejecución prolongada tras
  alcanzar aproximadamente 30%; **no se declara aprobada ni completa**. Log local
  ignorado: `reports/audit-review-demo/backend-suite.log`.
- CLI `--help`, `demo`, `import`, `summary`, `verify --originals`, persistencia y
  exportación/reimportación se comprobaron. No se usó FAC en pruebas automatizadas.
- Frontend: `npm run lint` sin errores (tres advertencias de hooks en componentes
  existentes); `npm run typecheck`, `npm run test` (**91 passed**) y
  `npm run build` aprobados. Build conserva avisos de tamaño de bundle y del
  script de configuración runtime.
- `npm run test:e2e`: corrida amplia final **14 passed, 8 skipped, 2 failed**.
  Las ocho pruebas reales M02 se omiten sin `FLIR_M02_REAL_ACCEPTANCE=1`.
  La prueba de navegación falló por espera de carga; su repetición aislada con
  un worker pasó. Permanece el fallo de accesibilidad `target-size` en dos
  `summary` de `/organization/splits`, fuera de esta implementación. No se cambió
  esa funcionalidad ni se declara la suite E2E global completamente verde.
- E2E propio final: **1 passed**; prueba conjunta previa con navegación:
  **2 passed**. No usa mocks HTTP. Valida imagen sintética, revisión pendiente,
  corrección `shot_change + same_region`, descarga JSON desde la interfaz,
  CSV, rechazo de Origin remoto, cierre/reinicio del servicio y reapertura
  de navegador con historial/export exactos. Axe no detectó violaciones en la
  nueva sección bajo las reglas WCAG seleccionadas; no equivale a certificación.

## Demostración y artefactos locales

El [runbook](../runbooks/audit_human_review.md) contiene instalación y comandos
reproducibles desde cero. `sequences audit demo` genera evidencia etiquetada
SYNTHETIC, sin archivos FAC. Los ZIP del fixture tienen timestamps fijos.

Artefactos de esta ejecución, excluidos de Git:

- `reports/audit-review-demo/`: fixture, ledger y log parcial global.
- `reports/audit-review-real/audit.sqlite3`: bytes congelados y normalización real.
- `reports/audit-review-real/import-report.json`: informe local de importación.
- Frontend `test-results/audit-review-synthetic.png`: captura sintética inspeccionada.
- Frontend `test-results/` y `playwright-report/`: evidencia de pruebas.
- Directorios temporales `flir-audit-SYNTHETIC-e2e-*`: ledgers de integración.

No se modificaron datos originales, no se subieron datos, no hubo push, merge,
publicación ni despliegue. No se ejecutaron modelos, embeddings, clustering,
splits, YOLO ni evaluación de impacto de leakage.

## Pendientes

Revisión humana real y validación sistemática de alineación val/test; obtención
e inspección de los tres archivos ausentes; soporte de auxiliares si se requieren;
regresión global Python completa y el problema de accesibilidad ajeno en splits.
La interfaz navega láminas disponibles, no reproduce vídeo ni reconstruye contexto
ausente. No se certifica independencia ni se unen paquetes por proximidad.
La siguiente fase debe registrar decisiones reales con protocolo y evidencia
consultada antes de diseñar el grafo de dependencias o nuevos particionamientos.
