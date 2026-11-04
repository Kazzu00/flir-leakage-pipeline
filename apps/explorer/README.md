# FLIR Pipeline Explorer

Interfaz de investigación para explorar tres pipelines independientes de visión por computador sobre video aéreo de la Amazonía colombiana. **M02 utiliza exclusivamente contratos reales; los recursos ausentes quedan explícitos. M01/M03 conservan sus modos DEMO. No ejecuta procesamiento científico.**

## Estado

- **Human Review v1.1:** evidencia trazable obligatoria para conclusiones, registro visual externo explícito, filtros de disponibilidad, orden temporal y siguiente pendiente. Conserva historias y muestra anomalías sin inferir leakage. [Operación y resultados de pruebas](docs/AUDIT_REVIEW.md).

- **Aceptación local M02: ACCEPTED_WITH_LIMITATIONS.** Explorador de 16 splits históricos y 30 particiones candidatas S1/S2/S3, filtros, miniaturas existentes y comparación lateral. El test nuevo permanece PROPOSED y no hay resultados YOLO11m. Antecedentes y variantes se distinguen por disponibilidad y procedencia. [Pruebas, evidencia y pendientes](docs/M02_ACCEPTANCE.md).

- **Detector conectado:** `/organization/evaluation` presenta el contrato público `detection-export-v1`, publicación upstream `0a4ea1f`: 48/48 runs, 16 splits, cuatro estrategias y siete asociaciones preespecificadas. Finalización y verificación reportadas upstream; validación de integridad aquí. Comparación descriptiva, no causal. [Contrato, procedencia y actualización](docs/DETECTION_CONTRACT.md).

- **Implementado:** M02 con cuatro secciones: Overview, Visual exploration, Sequences & linkage y Evaluation; Home como pipeline navegable; detalles técnicos en drawers accesibles; selección enlazada entre scatter, contenido y grilla de muestreo.
- **Clustering conectado:** `?view=clustering` consume `organization-evidence-v2` mediante su provider, independiente del snapshot DEMO/REAL del workspace. Configuración y cluster seleccionan contenidos únicos, previews locales y occurrences; grid inicial de 60 contenidos con `Load more`. No usa scatter ni inventa coordenadas. Las membresías preservadas de splits se distinguen del artifact completo; los datos ausentes quedan explícitos.
- **Integración conservada:** V1 y V2 validados por Zod, modo REAL/MIXED, artifacts ausentes pendientes sin fallback. El snapshot local V1 fue aceptado conservando 8.093 contenidos / 9.648 ocurrencias; esta comprobación de infraestructura no revalida resultados científicos.
- **V2:** contratos opcionales para candidatos, zonas, intervalos, revisión, evidencia nativa/legacy, experimentos, recurrencia, linkage y agregación. Exportador read-only con rutas explícitas. Las revisiones de grupos supported/unsupported no confirman enlaces exactos.
- **Docker validado en CI:** imagen multietapa con bases fijadas por digest, nginx, configuración runtime, Compose healthy y snapshot sintético montado read-only. Motor local no disponible; ejecución y health locales pendientes. Ver [DOCKER](docs/DOCKER.md).
- **No conectado:** máscaras reales, APIs, ejecución de jobs, integración real M01/M03. Los previews de Organization se sirven desde recursos locales ignorados; no se incluyen en el build ni se publican. No hay backend, autenticación ni despliegue cloud.
- **No ejecutado:** extracción, clustering, detección de secuencias, linkage, splits, entrenamiento ni nueva ciencia. La UI no certifica eliminación de leakage ni ground truth.

## Relación con los repositorios científicos

| Módulo                           | Repositorio independiente                                                                           | Responsabilidad                                                   |
| -------------------------------- | --------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| 01 Preprocessing                 | [proyecto-FAC](https://github.com/Laura-Martinez-Galindo/proyecto-FAC)                              | Extracción, HUD, inpainting, denoising, calidad                   |
| 02 Representation & organization | [flir-leakage-pipeline](https://github.com/Kazzu00/flir-leakage-pipeline)                           | Identidades, embeddings, similitud, reducción, clustering, splits |
| 03 Panoptic segmentation         | [proyecto-segementacion-panoptica](https://github.com/manugalarza/proyecto-segementacion-panoptica) | Pseudo-anotaciones, modelos, predicciones y evaluación            |

Se inspeccionaron clones `--depth 1` en `.references/`, ignorados por Git y tratados como read-only. No son necesarios para ejecutar la aplicación. Los commits revisados, contratos nativos y contradicciones están documentados en [RESEARCH](docs/RESEARCH.md). En particular, el README de segmentación está desalineado con código/reportes y con el estado actual de organización. La interfaz mantiene la advertencia `inconsistent`.

## Instalación y ejecución

Usar Node.js 24 LTS y npm 11 (versiones exactas de dependencias en `package-lock.json`). No requiere `.env`, GPU, Python ni datos.

```sh
npm ci
npm run dev
```

Abrir `http://127.0.0.1:5173`. Para probar el bundle:

```sh
npm run build
npm run preview
```

`BrowserRouter` tiene fallback a `index.html` en nginx. No se ha realizado un despliegue público. Para contenedor local: `docker compose up -d --build --wait`; ver [DEMO y REAL/MIXED](docs/DOCKER.md).

## Stack y arquitectura

React 19, TypeScript 6 strict, Vite 8, React Router 7, TanStack Query 5, TanStack Table 8, Zod 4, Apache ECharts 6 con echarts-for-react, Tailwind 4, primitivas shadcn/ui y Lucide. ESLint, Prettier, Vitest, Testing Library, Playwright y axe.

Se eligió Table 8.21.3 por compatibilidad con la API estable utilizada; no se mezcla su API con Table 9. No se añadió Zustand: el estado compartido de datos vive en Query y la selección es local al explorador. ECharts carga módulos de scatter/line/bar por su entrada ESM, y los módulos de la aplicación se cargan bajo demanda. React Compiler no está habilitado; su regla específica de incompatibilidad con Table se desactiva, manteniendo las reglas de hooks.

```text
src/app/                       rutas y arranque
src/components/                layout, UI, feedback, visualizaciones
src/features/                  overview, preprocessing, organization,
                               segmentation, experiments
src/contracts/                 esquemas Zod e invariantes
src/data/adapters/             MockDataAdapter, RealLeakageAdapter, composite
src/data/mock/                 fixture determinista
src/data/provider.tsx          frontera React Query / adapter
src/styles/                   tokens, temas y responsive
src/test/                     tests unitarios e integración React
e2e/                          navegador y axe
```

## Rutas

| Ruta                       | Vista                                                            |
| -------------------------- | ---------------------------------------------------------------- |
| `/`                        | Pipeline global navegable                                        |
| `/organization`            | Dataset y disponibilidad de evidencia                            |
| `/organization/explore`    | Encoder para embeddings/similitud/reducción; configuración exportada para clustering |
| `/organization/sequences`  | Sequence structure / Cross-dataset linkage                       |
| `/organization/evaluation` | Grouping/split, dependencia residual y detector                  |
| `/preprocessing/:stage?`   | M01 DEMO, sin integración nueva                                  |
| `/segmentation/:stage?`    | M03 DEMO, sin integración nueva                                  |
| `/experiments`             | Registro y detalle de runs                                       |
| `/evaluation`              | Evaluación global y límites                                      |

Las ocho rutas anteriores de M02 siguen funcionando mediante redirects; por ejemplo `/organization/reduction` → `/organization/explore?view=reduction`. Detalles en [INFORMATION_ARCHITECTURE](docs/INFORMATION_ARCHITECTURE.md).

Clustering presenta agrupación visual almacenada, no identidad temporal, secuencias, leakage confirmado ni ground truth. Noise −1 permanece separado y no representa un único grupo indivisible. Esta vista no contiene BEFORE/AFTER SPLIT; la evidencia de particiones conserva su vista propia. En REAL/MIXED no se reutilizan particiones históricas como resultados del dataset de video muestreado.

## Datos DEMO e integración local

La fixture contiene **144 contenidos sintéticos, 156 ocurrencias, 12 grupos de copias, 6 clústeres y 12 contenidos noise**. No coincide deliberadamente con los conteos del dataset real. Los tres algoritmos muestran fixtures ilustrativas, no resultados comparativos de algoritmos. Las curvas NIQE son sintéticas y no se reconstruyen de resúmenes XLSX. Las métricas no disponibles de evaluación científica son `null`.

`DataAdapter.getSnapshot(signal)` devuelve un snapshot validado. `DataAdapterContext` permite sustituir el adaptador. `RealLeakageAdapter` consume JSON saneado; `CompositeDataAdapter` integra M02 y conserva M01/M03 en DEMO, con indicador MIXED. No existe `APIDataAdapter`, endpoint científico ni acceso desde React a archivos upstream. Las etapas reales ausentes quedan pendientes.

Consultar [REAL_DATA_INTEGRATION](docs/REAL_DATA_INTEGRATION.md) para generar el snapshot con rutas explícitas de manifest, DINOv2/CLIP, similarity v2 y reduction. El exporter acepta runs opcionales de clustering/split/detector compatibles; no ejecuta esas etapas. Los artifacts locales nunca se versionan y `public/runtime` no se copia a `dist`. La única excepción autorizada es el contrato público ligero del detector en `src/features/detection/snapshot/`, independiente del snapshot de video. Se actualiza con `npm run sync:detection -- --repo <pipeline-repo>`; no requiere tener upstream para ejecutar la aplicación.

Tras generar `public/runtime/leakage-snapshot.json`:

```powershell
$env:VITE_DATA_MODE = "real"
$env:VITE_LEAKAGE_SNAPSHOT_URL = "/runtime/leakage-snapshot.json"
npm run dev
```

También puede usarse el selector **Data mode**. `.env.example` documenta el modo predeterminado. Las imágenes reales siguen sin conectarse.

Consultar [DATA_CONTRACTS](docs/DATA_CONTRACTS.md), [ARCHITECTURE](docs/ARCHITECTURE.md), [INFORMATION_ARCHITECTURE](docs/INFORMATION_ARCHITECTURE.md) y [UX_STANDARDS](docs/UX_STANDARDS.md).

## Validación

```sh
npm run lint
npm run typecheck
npm run test
npm run build
npx playwright install chromium
npm run test:e2e
npm run format:check
```

Tests sintéticos y validación del contrato público ligero versionado del detector, sin imágenes FLIR, modelos, GPU o red en la ejecución de tests. La instalación inicial de npm/Chromium requiere descargar dependencias públicas. Las capturas y reportes quedan ignorados en `test-results/` y `playwright-report/`. CI ejecuta las mismas comprobaciones. Axe es un smoke automatizado, no una certificación de WCAG/ISO ni una auditoría exhaustiva con tecnologías de asistencia.

## Seguridad y alcance

No se versionan `.references`, `.env*`, datasets, archivos FLIR, pesos, cachés, reportes ejecutados ni artifacts nativos. Solo se incluye el contrato público del detector expresamente autorizado y documentado. Los repositorios científicos permanecen independientes. No se implementan algoritmos de ciencia de datos en este frontend; visualizar exports no equivale a ejecutar o revalidar experimentos.

## Local human audit review

The existing Sequences & linkage module includes **FLIR Audit & Human Review v1**,
with a separate local evidence service, image context, independent audiovisual
and visual-region judgments, durable version history and JSON/CSV exports.
See [installation, synthetic smoke and review workflow](docs/AUDIT_REVIEW.md).
No sequences, splits, training or scientific confirmations are created.
