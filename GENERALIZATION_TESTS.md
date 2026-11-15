# Pruebas de generalización — 2026-11-06

La evidencia aquí corresponde a ejecución real de software sobre datos sintéticos.
No es una evaluación poblacional de features o detección FAC.

## Matriz independiente de FAC

`scripts/general_synthetic_demo.py` genera y verifica cuatro publicaciones, cada
una con 12 ocurrencias y 11 contenidos únicos. Los nombres, píxeles, IDs de clase,
etiquetas y vídeos son sintéticos. No usa nombres ni rutas del caso FAC.

| Caso | Configuración | Resultado observado |
|---|---|---|
| Dos clases, sin vídeo | IDs originales dispersos; duplicado exacto; labels vacíos | Diez asignaciones factibles, replay idéntico; sin HTML temporal inventado. |
| Siete clases, cuatro vídeos | FFV1 generado y decodificado realmente; PTS observado | Diez asignaciones factibles y HTML; alternativas exactas preservadas. |
| Sin anotaciones | Cero clases, duplicado exacto y relación candidata | Balance solo de ocurrencias; candidato no une grupos; diez asignaciones y replay. |
| Vacíos, conflictos y múltiples apariciones | Dos vídeos; labels diferentes sobre bytes exactos; todas las unidades unidas por decisiones sintéticas | Cinco asignaciones exactas y cinco estados `infeasible` agrupados; sin fallback; replay también de los resultados inviables. |

Las decisiones aceptadas del último caso son **fixtures explícitas**, no revisión
humana real de FAC. Las cuatro publicaciones suman 40 resultados de política,
35 con asignación y 5 inviables.

## Invariantes verificados

`tests/test_general_pipeline.py` contiene 30 pruebas parametrizadas: identidad,
catálogo variable, labels ausentes/vacíos/inválidos, conservación de bytes y cajas,
atomicidad exacta/aceptada, exclusión común, duplicados cross-split, cobertura,
candidatos separados, contradicciones transitivas y solver sin incumbent.
Incluye corrupción de hashes, geometría, referencias, PTS y estados; rechaza
confirmaciones nominales y aceptaciones candidatas/no atribuidas.

El replay compara todo el análisis estable, además de hashes de membresía.
Se probó permutación de registros, catálogo y relaciones. El ensayo completo
detectó inicialmente una diferencia de orden en IDs de diagnósticos; se corrigió
canonizando la entrada antes del análisis y se agregó regresión específica.
Los intentos anteriores se preservan como historial local, no como entrega final.

Se probaron features L2 sintéticas, rechazo de zero norm, revalidación de fuentes,
publicación inmutable, CLI, localización con decoder simulado y vídeo FFV1 real.
La igualdad del objetivo histórico de cinco clases frente al objetivo explícito
se comprobó para seeds 0–4. Cambiar `original_split` no cambia membresías nuevas.

## Ejecución y regresión

**Actualización de congelamiento:** el control pareado con `cwd` equivalente
obtuvo 55 passed / 34 failed en ambos brazos; todos los fallos originales tienen
la misma causa y no apareció el fallo artificial de ubicación. Las 77 pruebas
pertinentes volvieron a pasar. Se agregaron tres escenarios de benchmark (30
asignaciones), con verdad construida, candidatos falsos y cobertura desconocida;
detalle y límites en [SCIENTIFIC_CLOSURE.md](SCIENTIFIC_CLOSURE.md). No hubo cambios
en el código de producto. Los párrafos posteriores registran la ejecución previa.

Comando focalizado final:

```powershell
uv run --no-sync python -m pytest tests/test_general_pipeline.py tests/test_splitting.py tests/test_paired_delivery.py tests/test_identity.py tests/test_manifest.py tests/test_visual_localization.py -q
```

Resultado: **77 passed**, incluidas las 30 pruebas generales. Ruff global y ayuda
CLI aprobados. Los cuatro demos completos también pasan `verify` con fuentes
rehasheadas. El caso FAC tiene replay independiente en proceso CLI separado.

La suite global terminó con **1.274 passed, 34 failed** en 17.541,38 s. Todos los
fallos pertenecen a `test_organization_export.py`; los mismos 34 se reproducen
sin el comando general y con el solver anterior en una copia aislada. El control
tuvo además un fallo de ubicación del repositorio al validar destinos de medios
(35 failed, 54 passed). No se atribuye ese fallo adicional al pipeline general.
Los tests globales se recogieron antes de las últimas siete pruebas nuevas; las
77 focalizadas sí incluyen las 30 generales finales. No se declara regresión
global limpia. La lista exacta y el control JUnit quedan en el paquete local.

Un entorno independiente creado con `uv sync --locked --offline --extra dev
--extra alignment` pasó las **30 pruebas generales** y reprodujo FAC con hashes
de membresía idénticos. La instalación usó la caché local; no descargó modelos.

La suite offline bloquea conexiones mediante `tests/conftest.py`. No hay descargas
de modelos, GPU ni datos reales en los tests. Solo el adaptador y replay del caso
de estudio acceden a evidencia FAC externa, fuera de pytest.
