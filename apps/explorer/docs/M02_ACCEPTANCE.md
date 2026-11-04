# Aceptación M02 con evidencia real

Fecha: 2026-10-21. Estado: **ACCEPTED_WITH_LIMITATIONS** (integración local de contratos; no validación científica de Hypatia). Base frontend: `7d5e3db0cff19c112c10ff005f6a4ab1e58d53c2`. Rama: `fix/m02-real-data-acceptance`. El commit que incorpora este informe identifica la implementación aceptada; sin merge ni publicación.

## Resultado por vista

| Ruta                       | Comprobado                                                                                                                                                   | Limitación explícita                                                                                                        |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------- |
| `/organization`            | Manifest real: 11.305 ocurrencias, 9.552 contenidos, 27.320 pares candidatos y 16 splits; procedencia en drawer                                              | El snapshot separado de video tiene 9.648 ocurrencias / 8.093 contenidos; no representa la población etiquetada             |
| `/organization/explore`    | Selección de encoder/run, membresías C10/C12, contenidos y miniaturas; ausencia de DEMO                                                                      | Coordenadas no exportadas permanecen no disponibles. Membresía congelada no reconstruye el clustering original              |
| `/organization/sequences`  | Selección temporal, siguiente frame, PLAY/PAUSE, miniaturas, linkage y detalle candidato; funciona sin snapshot legacy                                       | Timeline, clúster, secuencia y candidato no son equivalentes; sin confirmación temporal añadida                             |
| `/organization/evaluation` | Contrato histórico YOLO11n independiente, 48 runs; procedencia y limitaciones                                                                                | Tests históricos variables; no comparación causal ni equivalencia con el protocolo nuevo                                    |
| `?view=splits`             | 16 splits históricos y 30 candidatos separados; familia/política/población/semilla, clase/origen/grupo/partición; miembros, relaciones y comparación lateral | Candidatos S1/S2/S3 provisionales, test PROPOSED y entrenamiento bloqueado; 49 contenidos de Adendo sin preview coincidente |
| `?view=reference`          | Resúmenes históricos existentes y cifras 0.721 / 0.663 atribuidas a la propuesta según el brief del investigador                                             | Artículo completo no verificado; sin porcentajes de mejora ni rankings; YOLO11m UNAVAILABLE                                 |

M02 ignora el modo DEMO del workspace. Recursos ausentes muestran error/UNAVAILABLE/PENDING; no hay sustitución sintética. M01/M03 y fixtures de pruebas conservan sus mocks. El registro de variantes distingue original etiquetado, frames sin HUD y videos originales/muestreados, sin atribuirles anotaciones o entrenamientos inexistentes en los contratos conectados.

## Evidencia y trazabilidad

- **Verificado localmente:** `public/runtime/organization-evidence-v2/manifest.json` y sus 15 recursos: esquemas, SHA256 y relaciones frame/content/media/split/timeline. Copia correspondiente a `flir-leakage-pipeline`, publicación `9f763ee`, generador `65428f77323314ae5fd4304efcee5cc12946e029`. Las 9.552 miniaturas locales se verificaron durante esta intervención; no se volvieron a descargar ni generar. Publicación y generador no son el mismo commit.
- **Inspeccionado:** `public/runtime/leakage-snapshot.json`, población de video separada, resúmenes de representación reales. No se inventaron coordenadas para completar la visualización.
- **Inspeccionado y proyectado en lectura:** publicación externa `audits/fixed_test_train_val_v1/five-seeds-20261013-a/{receipt.json,summary.json,suite.json,provenance.json}`, tablas `runs/*/{metadata.json,metrics.json,occurrence_memberships.parquet,atomic_groups.parquet}` y metadatos `sources/audit/{annotation_audit.json,relations.json}`. Procedencia declarada: `8159aac7e90379e2d2b20c0a744c631c884297a8`, **worktree dirty**. Se conservan 30 configuraciones, 25 membresías distintas y 1.706 ocurrencias en su unión; no son 30 entrenamientos.
- `tools/export_candidate_splits.py` comprueba los hashes de los archivos consumidos contra el receipt, identidades y test común, y copia soporte/métricas existentes. No ejecuta auditoría científica ni solver. Su salida privada local `public/runtime/candidate-splits-view-v1.json` es una proyección de presentación, excluida de Git/build. No se inspeccionaron pesos ni se revalidaron experimentos originales de Hypatia.
- **Integridad frontend; ejecución documentada upstream:** `src/features/detection/snapshot/`, contrato publicado en `0a4ea1f`, generador `18d61b35e4c82f41b0c9092e997fb9088f7a10f5`: 48 runs, 16 splits, cuatro estrategias, tres semillas de detector. Véase [DETECTION_CONTRACT.md](DETECTION_CONTRACT.md). El frontend no recertifica las ejecuciones ni convierte asociaciones descriptivas en causalidad.

No se versionan hashes de contenidos, membresías reales, rutas privadas ni capturas con imágenes. La procedencia detallada permanece en los recursos locales y sus drawers.

## Cambios y defectos corregidos

- `src/app/App.tsx`, `src/components/layout/AppShell.tsx`, `src/data/representation.ts`: el gate y modo legacy podían bloquear o sustituir vistas M02. Ahora cada contrato real conserva su disponibilidad independiente.
- `DatasetOverview.tsx`, `VisualExplorer.tsx`, `EvaluationOverview.tsx`: rutas activas sin fallbacks a vistas DEMO. Se mantiene `ArtifactStage` para evidencia validada y `OrganizationEvidenceProvider` para el export de organización.
- `src/data/split-evidence.ts`, `splitting/RealSplits.tsx`: adaptador validado, filtros y comparación de membresías existentes; joins exactos para previews, selección de relacionados y grupos almacenados fragmentados. Conteos de presentación no se etiquetan como nuevas métricas científicas.
- `ReferenceComparison.tsx`, `VariantRegister.tsx`, `src/features/detection/route.ts`: antecedentes, variantes y estado pendiente del nuevo protocolo, sin heredar la procedencia histórica para estas vistas.
- Pruebas de navegación actualizadas al contrato real obligatorio; fixtures sintéticos permanecen exclusivamente en pruebas. No se eliminaron componentes legacy innecesariamente.

## Pruebas ejecutadas

| Comprobación                                                                                      | Resultado                                                                                                                                          |
| ------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| Vitest: `split-evidence`, `navigation`, `organization-clustering`, `timeline-player`, `detection` | 50/50 PASS                                                                                                                                         |
| Vitest: `real-data`                                                                               | 18/18 PASS en ejecución focalizada anterior; junto a split/navigation produjo inicialmente un fallo por timeout del retry, corregido y reejecutado |
| Pytest: `tools/tests/test_candidate_splits.py`                                                    | 5/5 PASS: preservación/inmutabilidad, fuente corrupta, test divergente, grupo ausente y aprobación falsa                                           |
| Ruff sobre importador y sus tests                                                                 | PASS                                                                                                                                               |
| `npm run lint`                                                                                    | PASS, 0 errores; tres warnings de hooks preexistentes en SequenceExplorer/OrganizationLinkageExplorer                                              |
| `npm run typecheck`, `npm run build`                                                              | PASS; warnings de tamaño del chunk de charts y script runtime ya existentes                                                                        |
| Playwright Chromium: `m02-acceptance.spec.ts`                                                     | 8 casos PASS, incluyendo SHA256/joins, recorrido real, recursos ausentes y splits/reference                                                        |
| Playwright: `real-data.spec.ts`, `ux-v2.spec.ts`, caso `split comparison and prediction overlay`  | 7 casos PASS; fixtures sintéticos de regresión separados de aceptación real                                                                        |
| Revisión visual y geometría del comparador                                                        | Captura local `test-results/m02-real-splits.png`; sin overflow horizontal a 390/1440 px                                                            |
| Privacidad de `dist`                                                                              | Sin runtime, imágenes ni artifacts nativos; 11 JSON idénticos por SHA256 al contrato detector autorizado                                           |

Los PASS de Playwright corresponden a ejecuciones focalizadas y reintentos de los casos fallidos, no a una suite completa. Hubo timeouts iniciales de carga dinámica durante HMR y esperas de 5 s: se reinició el servidor de desarrollo, se esperó disponibilidad observable hasta 20 s y se repitieron los casos fallidos en serie sin editar durante la ejecución. No se reemplazó evidencia real por mocks para superarlos. Pruebas y capturas locales son ignoradas y pueden sobrescribirse en próximas ejecuciones.

Reproducir la aceptación real con los recursos existentes montados localmente:

```powershell
$env:FLIR_M02_REAL_ACCEPTANCE = '1'
npx playwright test e2e/m02-acceptance.spec.ts --workers=1
```

La proyección de candidatos solo es necesaria si todavía no existe; nunca sobrescribe la publicación ni el destino:

```powershell
uv run tools/export_candidate_splits.py "$env:FLIR_CANDIDATE_PUBLICATION" public/runtime/candidate-splits-view-v1.json
```

Para esta intervención se reutilizó un entorno Python existente con pandas 3.0.5 y pyarrow 25.0.1, sin instalar dependencias. El script fija esas versiones para usos posteriores. Los tests usan exclusivamente datos temporales sintéticos.

## Límites y pendientes indispensables

1. **Cierre científico YOLO11m:** adjudicar anotaciones/nomenclatura y aprobar población, relaciones admitidas, test y protocolo en backend. Los blockers de la publicación se muestran sin desactivarlos. La aceptación UI no autoriza entrenamiento.
2. **Acceso en otra instalación:** montar exports y media read-only fuera del build. Aprobar explícitamente cualquier distribución de imágenes privadas. No se creó API ni almacenamiento nuevo ni se comprobó Docker local en esta intervención.
3. **Variantes e integración global:** M01/M03 todavía necesitan contratos reales compatibles; HUD-free no tiene versión/anotaciones/resultados admitidos aquí. Los videos no adquieren etiquetas por similitud candidata.
4. **Comparación académica:** verificar el artículo completo antes de atribuir sus cifras como resultados cotejados o declarar protocolos equivalentes.
5. **Ausencias legítimas:** miniaturas adicionales de Adendo, coordenadas y artifacts originales de clustering solo pueden incorporarse desde fuentes existentes autorizadas; no bloquean explorar las membresías actualmente disponibles y no se deben fabricar.

No se ejecutó entrenamiento, extracción de embeddings, reducción, clustering, generación de splits ni nueva auditoría científica. Backend, datos, etiquetas y checkpoints quedaron sin modificaciones. El siguiente trabajo frontend puede concentrarse en pulido UX/UI; la integración científica global permanece condicionada a las decisiones anteriores.
