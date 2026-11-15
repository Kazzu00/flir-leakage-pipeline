# Fase 2A — ejecución y validación multisplit (2026-11-03)

Base backend `d7d641d`, frontend `e00cf94`. Rama aislada en ambos repositorios:
`feat/flir-phase2a-multisplit-alignment`. Implementación de matching: `a95940a`.
El frontend conserva su código: se eligió el informe HTML local permitido por el
encargo para limitar el alcance de la integración.

Se leyeron la especificación de Fase 2A y el protocolo piloto proporcionados por
el usuario. El [protocolo v0.1](../runbooks/human_review_pilot_protocol_v01.md)
queda como referencia metodológica; se normalizaron únicamente los espacios finales
de Markdown. No se ejecutó una campaña de 24 anotaciones humanas ni se simuló
acuerdo interevaluador.

## Fuentes comprobadas

Se inventariaron 3.659 ocurrencias originales de `video_13min_N`: todas legibles,
1280×720. No se fusionaron identidades entre ZIPs. Las 3.640 ocurrencias con índice
repetido entre empaquetados también presentan diferencias de bytes en algún miembro
de su grupo. Eso describe procedencia, no leakage semántico ni corrupción automática.

| Procedencia | Train | Val | Test | Total |
| --- | ---: | ---: | ---: | ---: |
| Imagenes.zip | 467 | 50 | 230 | 747 |
| Dataset_Balanceado.zip | 722 | 4 | 2 | 728 |
| Dataset_Balanceado_nopre.zip | 722 | 4 | 2 | 728 |
| Dataset_Balanceado_Preproc.zip | 722 | 4 | 2 | 728 |
| dataset_split_completo.zip | 479 | 120 | 129 | 728 |

Se excluyeron por nombre aumentado 898 imágenes de `nopre` y 901 de `Preproc`.
No se leyeron etiquetas YOLO. La selección utiliza exclusivamente `Imagenes.zip`:
12 estratos temporales por split, semilla 20261103, más ocho imágenes diagnósticas.
Total seleccionado: **44 imágenes** (14 train, 14 val, 16 test). El índice 500 está
ausente de esta fuente; no se sustituyó por otra procedencia.

## Ejecución real y estados

**Infraestructura implementada, regresión focalizada aprobada y muestra real
completa ejecutada. Verificación científica humana pendiente.** Paquete de entrega:
`reports/phase2a-complete`; integridad, JSON Schema y equivalencia CSV/JSON válidos.

| Split | Disponibles | Seleccionadas | Candidate | Ambiguous automático | Not found | Errores |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 467 | 14 | 13 | 0 | 1 | 0 |
| Val | 50 | 14 | 14 | 0 | 0 | 0 |
| Test | 230 | 16 | 16 | 0 | 0 | 0 |
| Total | 747 | 44 | 43 | 0 | 1 | 0 |

`human_verified=0`. Cero casos clasificados automáticamente como `ambiguous` no
significa ausencia de incertidumbre: las 43 correspondencias candidatas necesitan
juicio humano y las alternativas conservadas no constituyen una búsqueda exhaustiva.
El caso `not_found` es el índice 630 de train, sin error de lectura/procesamiento.

Costo de matching observado: 485,016 s, CPU, 33.808 cuadros decodificados y 3.113
muestreados/evaluados; ambos conteos incluyen controles y refinamiento, no son
cuadros únicos necesariamente. Presupuesto no agotado. El inventario/hashing previo
es adicional a ese tiempo. Se evaluaron 129 cuadros de control fuera de ventana,
ninguno pasó los umbrales geométricos. No se presenta ese resultado como tasa de
falsos positivos ni prueba de independencia.

Se produjeron 44 láminas de candidatos/vecinos y 44 láminas de controles. Dieciséis
casos tienen referencias de navegación a eventos del ledger. El hash del ledger
original permaneció idéntico y los stats de los archivos FAC no cambiaron.

Los mejores candidatos automáticos tienen diferencias respecto a `1978 + N` entre
−0,528433 s y −0,009367 s, mediana −0,025500 s. Son diferencias de candidatos,
**no errores medidos frente a ground truth** ni una calibración del offset.

### Fronteras diagnósticas

Los ocho cuadros candidatos se reextrajeron desde el vídeo con el script diagnóstico;
sus hashes RGB coincidieron con los del matching. Se decodificaron 176 cuadros para
esa comprobación adicional. Las cuatro láminas están en `reports/phase2a-diagnostics`.

| Frontera | Splits históricos | Segundos de los candidatos | Estado |
| --- | --- | --- | --- |
| 199–200 | train → test | 2176,974800 → 2177,975800 | Ambos candidate |
| 349–350 | test → val | 2326,957967 → 2327,958967 | Ambos candidate |
| 399–400 | val → test | 2376,974600 → 2377,975600 | Ambos candidate |
| 499–501 | test → train | 2476,974500 → 2478,976500 | Ambos candidate; 500 ausente |

Inspección visual de las láminas por el agente, no anotación humana independiente:
en 199–200 persisten construcciones visibles con desplazamiento del encuadre;
en 399–400 se aprecia una disposición similar de camino y tejados. En 349–350 y
499–501 cambia marcadamente la escala/campo de visión. Esa observación admite
explicaciones como zoom o cambio de toma: no se resolvió automáticamente `shot_change`
ni `different_region`. Cada comparación de JPEG con su candidato es visible por
separado. No se asignó ninguna etiqueta de continuidad o dependencia al ledger.

Se conserva un intento interrumpido en `reports/phase2a-final`: 27 casos procesados
y 17 incompletos por presupuesto temporal, con 15.558,64 segundos transcurridos
durante una interrupción larga. Su campo `not_found` junto con `errors` y la razón
"Processing incomplete" no constituye un negativo visual. Ese intento no es la
entrega completa. También existe un ensayo preliminar en `reports/phase2a-real`;
se conserva como registro de desarrollo, no como referencia del código final.

## Criterios e incertidumbre

Las coincidencias automáticas nunca son `human_verified`. SIFT/RANSAC aporta
soporte geométrico; correlación y dHash quedan como diagnósticos independientes.
La ausencia de competidores que pasen los umbrales no demuestra unicidad del cuadro.
Las alternativas top-k y vecinos están disponibles para revisión.

En la inspección preliminar, el índice 630 de train presenta principalmente ruido
y HUD, sin inliers geométricos en el mejor candidato. Esto no demuestra otra región:
la evidencia es insuficiente para identificar un instante. La lámina permite
contrastar el ruido con los vecinos temporales donde vuelve a verse terreno.

Los controles están a ±120 segundos de la hipótesis y al inicio del vídeo.
La inspección del control inicial para el índice 199 muestra nubes y un menú en
pantalla, frente a construcciones y vegetación en el JPEG consultado. Es un control
visual documentado de no correspondencia directa; no se asignó un juicio humano
formal ni una etiqueta de independencia geográfica. Las vistas a ±120 s pueden
compartir región y permanecen controles de búsqueda, sin verdad terreno inventada.

## Tiempo verificable

La fuente AVI declara 1280×720, tasa 30000/1001 y time base 1001/30000.
Se registran PTS de cuadros realmente decodificados, seek solicitado y primer
tiempo decodificado. La tasa efectiva se deriva de los PTS devueltos, no de asumir
5 FPS exactos. El refinamiento inspecciona tasa nativa.

Un contraste independiente con `ffprobe`, para el índice 199, encontró el mismo
timestamp 65244 y segundo 2176,974800 que PyAV, con diferencia 0 a la precisión
serializada y tolerancia de comparación 1e-6 s. `ffprobe` omite `pts` y expone
`best_effort_timestamp` en este AVI: es tiempo del medio reconstruido, no reloj
original de captura. La tolerancia del contraste no es exactitud de alineación JPEG.

El test sintético FFV1 verifica que un seek a 0,11 s devuelve primero 0,12 s en
una grilla conocida de 25 FPS, con error de representación <1e-9 s, sin convertir
esa propiedad sintética en precisión científica sobre FAC.

## Validaciones ejecutadas

| Alcance | Resultado observado |
| --- | --- |
| Ruff global | Aprobado |
| Regresión focalizada ampliada | 214 pruebas aprobadas, 136,19 s |
| Alineación específica | 13 pruebas aprobadas, 9,30 s |
| Integración adicional del informe diagnóstico | 1 prueba aprobada, 7,13 s |
| CLI instalada con extra alignment | Ayuda disponible, exit 0 |
| Smoke sintético persistente | 3 candidatos, cero verificaciones humanas; 132 cuadros decodificados |
| HTML sintético en Chromium | 3 casos, 6 imágenes, primera imagen decodificada, cero errores de runtime |
| Paquete real | 44/44 procesadas, 88 láminas, esquema/hashes/CSV válidos, ledger sin cambios |
| HTML real en Chromium | 44 casos y 88 imágenes decodificadas, cero errores de runtime |
| Diagnósticos en Chromium | 4 pares y 4 imágenes decodificadas, cero errores de runtime |
| Suite global backend completa | No repetida en esta fase; no se declara aprobada |
| Suites frontend globales | No repetidas: no hay cambio de código frontend |

La regresión ampliada incluye `test_alignment.py`, auditoría v1/v1.1, smoke,
vídeo, ingesta de variantes y secuencias. Los tests son sintéticos, offline y CPU.
Se prueban escala/ruido/tono, negativos, modos temporales ambiguos, nombres duplicados
dentro y entre ZIPs, archivos corruptos, vídeo inaccesible, límites, PTS, esquema,
roundtrip CSV/JSON, idempotencia, conservación de SQLite y rechazo de verificación
humana sin evidencia/revisor. Un índice diagnóstico ausente no genera una imagen falsa.

En Windows, el lanzador `pytest.exe` fue bloqueado por Control de aplicaciones;
se usó `uv run python -m pytest`. Una actualización del entorno encontró ocupado
`flir-pipeline.exe` durante el ensayo, por lo que se usó temporalmente `--no-sync`
con las dependencias ya instaladas. La CLI instalada se comprobó después con éxito.
Ninguno de esos fallos operativos se presenta como una suite aprobada.

## Integración y archivos

- `alignment/inventory.py`: manifiesto, procedencia, anomalías y estratificación.
- `alignment/matching.py`: decode/PTS, presupuestos, soporte y alternativas.
- `alignment/models.py`: estados y validación de correspondencias/revisiones.
- `alignment/pipeline.py`: exportación, hashes, HTML, conservación del ledger y
  referencias de navegación por índice a eventos v1.1.
- `alignment/cli.py`, `sequences/cli.py`: `run`, `verify`, `review`.
- Dos esquemas raíz: `alignment-evidence-v1`, `alignment-human-review-v1`.
- `scripts/report_phase2a_diagnostics.py`: comparación de fronteras con reextracción
  independiente y comprobación de hashes RGB, sin anotación automática.
- `tests/test_alignment.py`, extra CPU `alignment`, lock y CI actualizados.

Los enlaces a auditoría por índice son candidatos de navegación, no asociaciones
visuales verificadas. No se llamó a `attach-evidence` atribuyendo una revisión
manual inexistente. No se editaron TemporalObservation ni decisiones anteriores.

## Riesgos y límites

- Muestra dirigida y pequeña: no estima exactitud poblacional ni porcentaje de
  ground truth; los umbrales no están calibrados con adjudicaciones humanas.
- Búsqueda local/ampliada acotada, sin garantía de encontrar repeticiones lejanas
  fuera de controles. No se realizó un barrido exhaustivo del vídeo.
- dHash no descarta cuadros automáticamente; su uso como filtro seguro queda
  pendiente de calibración. HUD residual, zoom, tono y escenas repetidas pueden sesgar.
- Matching y paneles usan 384×216; una revisión detallada puede necesitar volver
  al JPEG original y a los cuadros de resolución nativa mediante sus localizadores.
- Revisión humana, consistencia entre revisores, piloto de 24 casos y decisiones
  de continuidad/dependencia permanecen pendientes.
- HTML local implementado; no se ampliaron acciones de la UI React.
- No se generaron secuencias finales, grafo, splits, embeddings masivos ni
  entrenamiento. No hubo GPU, push, merge, despliegue o subida de datos externos.

Los archivos FAC, imágenes, hashes privados, ledgers e informes ejecutados permanecen
fuera de Git. Se preservaron los cambios ajenos en `organization_contract.py` y
`docs/analysis/repository_audit.md`. El segundo commit de entrega documenta los
resultados finales y añade el informe diagnóstico.
