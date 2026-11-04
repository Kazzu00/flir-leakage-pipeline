# Validación local

## UX V2 · 2026-10-02

Infraestructura de visualización y exportación, no experimento científico. Se inspeccionó `main` upstream en `34bd631cbff0a825b6b6ccbd7498c825a86a40ae` mediante clon temporal read-only; `.references` no se modificó.

| Comprobación            | Resultado observado                                                                                                  |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------- |
| ESLint / TypeScript     | PASS                                                                                                                 |
| Vitest                  | PASS, 41 tests en 4 archivos                                                                                         |
| Exportador / pytest     | PASS, 23 tests sintéticos y offline                                                                                  |
| Ruff check / format     | PASS, versión fijada 0.16.9                                                                                          |
| Build / Prettier        | PASS                                                                                                                 |
| Playwright Chromium     | PASS, 12 pruebas, incluidos axe en nuevas vistas, ambos temas, teclado y redirects                                   |
| Volumen sintético       | 9.000 puntos en Canvas; tabla acotada y búsqueda de identidad                                                        |
| Compatibilidad V1 local | Snapshot existente aceptado: 8.093 contenidos, 9.648 ocurrencias, 40 runs; lectura sin modificaciones                |
| V2                      | Fixture Python/Zod compartida; datasets, revisiones, cobertura, linkage, checksums y errores de artifacts explícitos |
| CLI                     | `uv run tools/export_leakage_snapshot.py --help` incluye opciones V2                                                 |
| Docker Compose          | `docker compose config --quiet`: PASS                                                                                |
| Docker local            | Build intentado; engine Linux no disponible. Arranque y health locales pendientes                                    |

Se revisaron visualmente Home, reducción y secuencias; screenshots y traces quedan ignorados. El drawer captura/restaura foco y responde a Escape. Las zonas candidatas y revisadas tienen texto y patrones además de color. Axe es una comprobación automatizada, no certificación WCAG ni auditoría completa con lector de pantalla.

Comandos frontend: `npm run lint`, `npm run typecheck`, `npm run test`, `npm run build`, `npm run format:check`, `npm run test:e2e`. Python: `uvx ruff==0.16.9 check tools`, `uvx ruff==0.16.9 format --check tools` y `uv run --with numpy==2.2.6 --with pandas==2.2.3 --with pyarrow==20.0.0 --with pytest==8.4.2 python -m pytest tools/tests -q`. Se usa `python -m pytest` porque la política local de Windows bloquea el ejecutable pytest del entorno temporal.

El build conserva avisos no bloqueantes: chunk compartido ECharts de ~592 kB y script clásico `/runtime-config.js` deliberadamente externo al bundle. El snapshot real no está en `dist`; solo se produce la configuración predeterminada inocua. No se generó un nuevo snapshot privado V2 ni se ejecutaron modelos, secuencias, linkage o splits científicos. La prueba del snapshot existente verifica transporte y estructura, no resultados experimentales.

El [job Docker remoto](https://github.com/Kazzu00/flir-pipeline-explorer/actions/runs/37096999192/job/111128851600) pasó: imagen multietapa y Compose healthy, SPA, cambio DEMO → REAL sin rebuild, snapshot sintético montado y mount read-only. La primera ejecución remota del exportador detectó una diferencia de `sys.path` al lanzar el ejecutable pytest; CI se alineó con `python -m pytest`, usado localmente. El estado global de CI se comprueba sobre el último commit publicado y se reporta aparte; este resultado Docker no implica que los otros jobs hayan pasado.

## Fase 2 · 2026-09-27

Infraestructura de lectura de artifacts probada exclusivamente con datos sintéticos. No se accedió a artifacts privados ni se ejecutaron experimentos científicos.

| Comprobación                  | Resultado observado                                                                                                 |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| ESLint / TypeScript           | PASS                                                                                                                |
| Vitest                        | PASS, 29 tests en 3 archivos                                                                                        |
| Exportador / pytest           | PASS, 13 tests; fixtures temporales NumPy/Parquet; salida compartida con el contrato Zod                            |
| Ruff check / format           | PASS                                                                                                                |
| Build                         | PASS; ECharts ~592 kB minificado / 201 kB gzip, advertencia de tamaño                                               |
| Prettier                      | PASS                                                                                                                |
| Chromium                      | PASS, 10 pruebas, incluidas las seis de DEMO                                                                        |
| Flujo artifact                | Modo MIXED, ambos encoders, coordenadas persistidas, clustering/splits pendientes, errores seguros y retorno a DEMO |
| Volumen sintético             | 9.000 coordenadas persistidas en Canvas; búsqueda de la última identidad y tabla paginada                           |
| Clustering artifact sintético | Ruido explícito, BEFORE/AFTER y AFTER deshabilitado cuando falta split compatible                                   |
| Accesibilidad / responsive    | Axe en vistas demo y reducción artifact; navegación móvil a 390 px; controles etiquetados                           |
| CLI                           | `uv run tools/export_leakage_snapshot.py --help` correcto                                                           |
| Privacidad                    | Runtime y clones ignorados; sin datos reales en fixtures; `dist` contiene solo `index.html` y assets de aplicación  |

El exportador prueba integridad raw/L2, identidades incompatibles, recibos de procedencia temporal, row mapping, preservación multivideo, checksums, benchmark con candidatos declarados, atomicidad de grupos, rechazo de pilotos, solo lectura y no sobrescritura. Los tests de Zod rechazan metadata privada y relaciones incoherentes antes de mostrar datos. La validación de transporte no certifica la calidad científica de métricas upstream.

Se corrigieron durante la validación una etiqueta DEMO residual, clasificación de errores, nombres accesibles de selectores, un desbordamiento móvil y un tipo de fixture JSON. La captura revisada de reducción real usa únicamente tres contenidos sintéticos; el test de volumen usa 9.000. Capturas y traces quedan ignorados. Falta una exportación y revisión con artifacts privados explícitos del usuario.

CI incluye un job de exportador con Python 3.11 y dependencias CPU, además del job frontend. Las instalaciones descargan dependencias públicas; los tests no requieren red, modelos, datasets, GPU ni secretos.

## Fase 1 · 2026-09-26

Entorno: Windows, Node.js 24.11.1, npm 11.6.3. Solo frontend y fixtures sintéticas. No se ejecutaron pipelines científicos ni se importaron datos FLIR.

| Comprobación               | Resultado observado                                                                         |
| -------------------------- | ------------------------------------------------------------------------------------------- |
| `npm run lint`             | PASS, sin errores                                                                           |
| `npm run typecheck`        | PASS, TypeScript strict                                                                     |
| `npm run test`             | PASS, 11 tests en 2 archivos                                                                |
| `npm run build`            | PASS; aviso no bloqueante del chunk ECharts (~579 kB)                                       |
| `npm run test:e2e`         | PASS, 6 pruebas Chromium                                                                    |
| Navegación de rutas        | 24 rutas principales/de etapa cargan sin errores JS                                         |
| Axe                        | Cero violaciones detectadas en cinco vistas y ambos temas, tags WCAG 2 A/AA, 2.1 AA, 2.2 AA |
| Responsive                 | Contención a 1024, 1440 y 1920 px; navegación móvil a 390 px                                |
| Repositorios de referencia | `git status --short` vacío en los tres clones                                               |

Los tests cubren integridad de contenido, grupos indivisibles, singleton noise, asignaciones por ocurrencia, coherencia de conteos, rechazo de secuencias no verificadas, navegación, selección de clúster, BEFORE/AFTER, comparación de splits, filtros de experimentos, overlay y accesibilidad básica. La inspección visual local incluyó Home, clustering y móvil. Screenshots y traces son generados e ignorados por Git.

Los primeros intentos detectaron problemas de semántica de listas de definición, contraste de números y texto del overlay, además de importación CJS/ESM y sincronización de una prueba con rutas lazy; fueron corregidos antes del pase final. No se afirma que la aplicación esté certificada, ni que exista una auditoría exhaustiva con lector de pantalla.

Ruff/pytest no aplican a este repositorio TypeScript. Tampoco se ejecutaron experimentos, extracción de features, entrenamiento, medición de leakage real, integración API ni despliegue web. Las métricas científicas no conectadas permanecen unavailable. GitHub Actions queda configurado; el resultado remoto se verifica por separado del pase local.
