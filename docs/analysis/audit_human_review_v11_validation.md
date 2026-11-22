# Validación FLIR Human Review v1.1 — 2026-11-03

Infraestructura implementada y regresiones focalizadas validadas. No se ejecutó
una campaña humana ni un experimento científico. Bases: backend `a63b9e3`,
frontend `49cfa04`; rama `fix/audit-human-review-v1-1` en ambos repositorios.

## Fallos, cambios y regresiones

| Causa | Corrección y archivos/funciones | Regresión |
| --- | --- | --- |
| La validación permitía conclusiones sin referencias | `models.require_evidence`, `DecisionInput.grounded`, `store.Store.save_decision`, validación de medios y registro externo; `server.py`, `cli.py`; UI `AuditReview.tsx` y `audit-client.ts` | Rechazo sin evidencia, referencias inventadas o ajenas; pendientes/indeterminados válidos; imagen externa válida, bytes/metadata alterados; API y UI reales |
| Precedencia JSON dependía del lote de importación | `Store.import_directory`, resolución persistente de compañeros, comparación completa, recibos de conciliación | CSV solo, JSON posterior, repetición, conflictos, IDs e historias conservados; duplicado v1 detectado y conciliación explícita |
| Normalizador descartaba el final de muestra | `importers.py`, `CandidateEvent` y proyección del snapshot | Eventos de varias muestras; metadata ausente; origen ambiguo; extremos y procedencia |
| JSON Schema contenía definiciones sin raíz | `contracts.schema_for/write_schemas`, cinco contratos completos | Documentos válidos y rechazo de decisiones sin evidencia, digest, intervalos y tipos inválidos |
| Duplicados se comprobaban por split aislado | `importers.historical_anomalies`, snapshot/UI | Índices compartidos conservan todas las ocurrencias; advertencia sin inferir leakage |

Regresiones backend nuevas: `tests/test_audit_v11.py`. Compatibilidad mediante
`legacy_models.py` y `legacy_importers.py`, copias de normalizadores v1 para verificar
filas históricas sin reescribirlas. Frontend: `src/test/audit-review.test.ts` y
`e2e/audit-review.spec.ts`, con SQLite/API reales y datos exclusivamente sintéticos.

## Ejecuciones observadas

| Comando/alcance | Resultado |
| --- | --- |
| `uv run ruff check .` | Aprobado |
| `uv run python -m pytest tests/test_audit_review.py tests/test_audit_v11.py tests/test_smoke.py -q` | 49 aprobadas, 9.38 s (46 auditoría, 3 smoke) |
| CLI `sequences audit --help` | Nuevos comandos disponibles |
| Frontend `npm run lint` | 0 errores; 3 advertencias de hooks preexistentes fuera del auditor |
| `npm run typecheck` | Aprobado |
| `npm run test` | 94 aprobadas en 10 archivos |
| `npm run build` | Aprobado; advertencias existentes de tamaño de bundle/script runtime |
| `npm run test:e2e -- e2e/audit-review.spec.ts` | 2 aprobadas; repetición final con control de integridad externo: 33.9 s |
| Frontend suite E2E completa | 16 aprobadas, 1 fallida, 8 omitidas; 2.9 min |
| Repetición aislada de `e2e/explorer.spec.ts:3 --workers=1` | Aprobada; el fallo global de espera de preview no se considera resuelto por esta repetición |
| Backend pytest global | Incompleto: detenido a los 300 s (~17%, sin fallo impreso hasta ese punto); no se declara aprobado |

Las 8 pruebas E2E omitidas requieren `FLIR_M02_REAL_ACCEPTANCE`. El límite global
backend usó un subproceso supervisado y terminó únicamente su árbol de procesos.
Logs y resultados operativos permanecen ignorados por Git.

## Comprobación local acotada

Una copia SQLite del ledger v1, obtenida con origen abierto en modo solo lectura,
verificó 450 eventos, 1855 observaciones y 215 intervalos. El hash del ledger original
permaneció idéntico antes/después. Una importación nueva v1.1 y su repetición conservaron
los mismos IDs: 8 activos, 450 eventos y 1855 observaciones; verificación de originales
válida. No se añadieron decisiones humanas reales. Estos conteos describen los
artefactos locales inspeccionados, no una campaña de anotación.

Los originales de código tenían UTF-8 correcto. La consola PowerShell podía mostrar
mojibake; una escritura intermedia por stdin introdujo signos `?` en cadenas nuevas,
detectados por E2E y corregidos con parche Unicode. La regresión comprueba cadenas
acentuadas originales y nuevas, y E2E verifica texto y persistencia en español.

## Límites pendientes

- Suites globales no completamente aprobadas; estados exactos arriba.
- Ausencia de evidencia visual sigue siendo explícita; no se reconstruyen frames.
- Registro manual aporta trazabilidad, no certificación automática de correspondencia.
- Historias v1 sin evidencia se conservan y advierten; no se validan retroactivamente.
- La reconstrucción desde cero debe conservar el orden canónico o el ledger completo.
- No se modificaron FAC, labels, splits, modelos ni resultados históricos. No hubo
  entrenamiento, clustering, grafo de dependencias, push, merge ni despliegue.

Los commits de entrega se informan al finalizar; este documento se incluye en ellos.
