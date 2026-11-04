# Fase 2B: ejecución técnica y límites

Implementación y piloto real ejecutados; revisión humana de dependencias pendiente.
Rama `feat/flir-phase2b-dependency-pilot`, basada en `079b266` de Fase 2A.1.
No se modificaron los dos cambios locales previos ajenos al alcance, los datos FAC,
las revisiones anteriores, los splits ni el ledger histórico. No hubo push,
entrenamiento, embeddings masivos, tracking, clustering, grafo o secuencias finales.

## Referencia de Fase 2A.1

El JSON recibido coincide byte a byte con el SHA-256 citado en el informe de cierre,
aunque su nombre no incluye `(1)`. Se comprobó la vinculación al piloto, el historial
de 27 decisiones y 24 aceptaciones finales de top-1; 22 confianzas altas y dos medias.
Se preservan referencias y hashes privados dentro del bundle local. La declaración
posterior sobre consulta de alternativas es autorreportada; no altera las casillas
originales. Los controles anteriores tenían un único contenido visual. El 100% de
aceptación por una persona en muestra dirigida no estima precisión poblacional.

## Piloto ejecutado

| Componente | Resultado observado |
|---|---:|
| Pares únicos | 24 |
| Fronteras diagnósticas | 4 |
| Cercanos, distribuidos sobre el rango disponible | 6 |
| No consecutivos | 6 |
| Cambios fotométricos candidatos | 4 |
| Controles negativos candidatos | 4 |
| Imágenes fuente distintas utilizadas | 21 |
| Pares cross-split | 17 |
| Imágenes y contenidos distintos en controles | 8 |
| Comparaciones acotadas calculadas | 276 |
| Fotogramas decodificados para contexto | 858 |
| Recursos PNG del bundle | 129 |
| Pares asignados a segundo revisor | 8 |
| Decisiones humanas de pares | 0 |

Las fronteras son 199–200, 349–350, 399–400 y 499–501; no existe sustitución de 500.
Los seis cercanos tienen separaciones nominales entre 2,002 y 42,009 s. Son cercanos
según una regla operativa ≤60 s, no garantía de misma toma. No hubo déficit de cuota.
Los cambios candidatos se seleccionaron por diferencias de brillo/contraste, no
por observación validada de un evento brusco. Su naturaleza queda por revisar.

La geometría es débil: 16 pares sin inliers, seis con cuatro, uno con seis y uno con
siete. Cuatro inliers pueden ajustar una homografía sin evidencia discriminativa;
no se consideran verificación. Ninguno superó el criterio para superposición válida.
Se conservan correspondencias, errores y máscaras para inspección, sin fabricar
overlays ni inferir `different_region`. Los seis no consecutivos son hipótesis
exploratorias de relación visual con soporte débil, no dependencias confirmadas.

Una primera ejecución conservada en `reports/phase2b-pilot` excluía cualquier píxel
saturado y descartaba demasiado terreno. No se usó para anotación. La ejecución
final `reports/phase2b-final` distingue verde dominante del HUD y distribuye los
pares cercanos sobre el rango temporal disponible. El ajuste fue de QA de máscara
y muestreo, sin etiquetas de dependencia ni calibración con el conjunto de test.

## Validación técnica

- 17 tests propios: selección reproducible, cuotas y controles distintos; ejes
  independientes; incertidumbre; evidencia obligatoria; conflictos de versión;
  persistencia; aislamiento de segunda revisión; adjudicación; hashes; importación
  de historial; contratos positivos/negativos; handler HTTP con origen inválido.
- Regresión final: **93 pruebas aprobadas en 39,85 s**, incluidos los 17 tests propios,
  alineamiento, piloto anterior, auditoría v1/v1.1 y smoke. Ruff global aprobado.
  Una ejecución anterior de 90 pruebas también pasó antes de añadir tres tests.
- Chromium: ocho imágenes sintéticas y 129 reales cargadas; 58 recursos en los ocho
  pares del segundo revisor; cero errores JavaScript. Persistencia exacta tras
  reiniciar, rechazo de evidencia vacía, acción explícita de consulta y aislamiento
  de segundo revisor comprobados. Unicode español conservado en ida y vuelta.
- Real: cero decisiones antes/después de navegar y de reiniciar. No se inventaron
  respuestas humanas para completar la validación.
- Wheel Python generado correctamente, con la nueva vista incluida explícitamente.
- Suites globales Python no ejecutadas en esta entrega; no se declaran aprobadas.
  TypeScript, build React y E2E global frontend no repetidos: ese repositorio no
  cambió. La prueba E2E de esta fase corresponde al servicio local Python/HTML.

Un primer comando de regresión usó un nombre inexistente `tests/test_audit.py`:
terminó sin ejecutar pruebas. Se corrigió a los dos módulos reales de auditoría;
ese intento no se cuenta como validación.

## Archivos y artefactos

Código: `alignment/dependency.py`, `dependency_contract.py`, `dependency_store.py`,
`dependency_cli.py`, `dependency_ui.html`; integración mínima en `alignment/cli.py`,
`pilot_server.py` y empaquetado. Contratos en `docs/contracts/dependency-*`;
tests en `tests/test_dependency_pilot.py`; navegador y generación de esquemas en
`scripts/check_dependency_pilot_browser.cjs` y `scripts/export_dependency_schemas.py`.

Privados locales: `reports/phase2b-final/{pilot.json,checksums.json,reference,media}`,
`reports/phase2b-selection.md`, `reports/phase2b-report.json`, dos SQLite independientes
y `reports/phase2b-browser-*/results.json`. El manifiesto y el reporte local de
selección contienen los 24 pares, índices, splits, tiempos, métricas y hashes.
No se versionan imágenes, hashes reales, anotaciones, bases ni rutas privadas.

## Limitaciones pendientes

La revisión humana de ambos ejes y la segunda revisión independiente todavía no se
han ejecutado. La concordancia permanece nula. El contexto ±1 s alrededor de cada
extremo no documenta todo el intervalo de un par distante. Los tiempos son PTS del
medio y heredan incertidumbre del alineamiento; no son reloj de captura certificado.
El caso de ruido 630 continúa en la referencia de alineación con su advertencia,
pero no fue seleccionado entre estos pares. No se revalida su instante exacto.

La máscara reduce HUD sin garantizar su eliminación; tampoco existe prueba de
que los controles sean negativos verdaderos. Son ocho contenidos diferentes, sin
afirmar independencia estadística. El aislamiento de revisores es de sesiones/API
y bases, no de seguridad frente al propietario del sistema. El reporte es
descriptivo; no demuestra contaminación de entrenamiento ni impacto en métricas.
