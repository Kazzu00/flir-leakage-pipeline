# Congelamiento funcional y cierre científico — 2026-11-06

El pipeline general queda **congelado como candidato funcional**. La evidencia
disponible permite validar sus contratos en fixtures controlados y auditar FAC;
todavía no permite concluir cuánto cambia el rendimiento del detector por leakage.
El cierre global del repositorio sigue condicionado por fallos de organización.

Este trabajo agrega exclusivamente scripts de evaluación y documentación. La
comparación SHA256 con el snapshot anterior confirmó los **175 archivos Python
de `src/` sin cambios ni adiciones**. No se incorporaron funcionalidades, modelos,
algoritmos de partición ni visualizaciones. Los reportes HTML del benchmark usan
el publicador existente. No hubo entrenamiento, modificación del test protegido,
commit, push ni merge.

## 1. Regresión bajo entorno controlado

El comparador `scripts/closure/regression_audit.py` ejecuta las mismas 89 pruebas
de organización en dos raíces aisladas, con el mismo intérprete, dependencias,
tests y variables offline/hilos. Cada proceso usa su propia raíz como `cwd` y
`PYTHONPATH`; se evita el artefacto de ubicación del control anterior.

El brazo anterior restaura la función MILP desde el commit Git registrado y la
CLI preintegración conservada. Elimina la excepción nueva `AssignmentError` y
preserva la asignación aleatoria atómica previa y el cambio local de
`explorer/organization_contract.py` en ambos brazos. **Es un control de la
integración general, no una afirmación de que un checkout limpio de HEAD tenga
estos fallos.** Los módulos generales nuevos permanecen disponibles pero no se
registran en la CLI del control.

La causa examinada es una incompatibilidad previa entre productor y validador:
`organization.py` declara `membership_consistency_verified=False` para artefactos
completos de clustering; el validador local exige `True` y procedencia completa.
La excepción observada es `Clustering membership provenance is incomplete`.
Esto impide publicar ese exportador y en varios tests impide alcanzar la aserción
que pretendía comprobar integridad, referencias o recuperación ante errores.
No se puede interpretar esos tests como validación de sus aserciones posteriores.

La matriz privada tiene una fila por cada uno de los 34 fallos originales:
identificador, resultado anterior/candidato, causa y ubicación, evidencia JUnit,
afectación de contratos, clasificación y acción. No se corrige un fallo previo
para ampliar artificialmente el alcance del congelamiento.

**Resultado observado: anterior 55 passed / 34 failed; candidato 55 passed /
34 failed.** Los 89 identificadores coinciden, no hay cambios de estado y los
34 fallos tienen la misma causa. No se demostró una regresión nueva en este
conjunto. Las 77 pruebas pertinentes aprobaron (43,55 s), junto con Ruff global,
`git diff --check`, ayuda CLI y cuatro replays independientes (tres sintéticos
y FAC). La suite global completa no se repitió: se conserva su resultado previo
de 1.274 passed / 34 failed. **No se declara CI verde.** Los tiempos pareados
no se usan como benchmark de rendimiento: hubo distinta concurrencia de procesos.

El primer intento de este comparador tuvo errores de preparación de directorios
temporales. Se conserva bajo `regression/` como **intento inválido**. El comparador
corregido usa rutas temporales cortas y rechaza resultados con errores de setup;
la evidencia final está en `regression-final/`.

Durante el primer sellado se detectó que el solver del control conservado era
idéntico al candidato. Por eso `regression-validated/` solo acredita el control
de CLI y **no respalda la comparación del solver anterior**. Se preservó ese
intento, se extrajo la función anterior directamente de Git y se repitieron
ambos brazos. Los hashes finales exigen que únicamente difieran los dos archivos
de integración declarados. El directorio `sealed/` es el intento de sellado
incompleto; el paquete definitivo está en `sealed-final/`.

## 2. Benchmark sintético controlado

`scripts/closure/leakage_benchmark.py` fija antes del solver el protocolo y la
verdad construida. Cada caso tiene 12 familias, 36 contenidos visuales y 42
ocurrencias, incluyendo seis repeticiones exactas. Las transformaciones son
brillo/reflexión o desplazamientos de una escena construida. El caso temporal
genera tres vídeos FFV1, con cuatro escenas independientes por vídeo y apariciones
repetidas; la localización usa decodificación y PTS reales. Compartir vídeo no
constituye por sí solo una relación aceptada.

Ambas políticas usan el MILP existente, ratios 0,6/0,2/0,2, seeds 0–4,
`node_limit=32` y `mip_rel_gap=0.001`. No se seleccionaron semillas tras observar
resultados. No hay clases ni etiquetas de entrenamiento en estos fixtures;
el objetivo controla el número de ocurrencias. Los tests y demos anteriores
cubren el balance con catálogos variables.

Cada caso contiene 24 enlaces verdaderos por construcción, 12 candidatos falsos
entre familias y un enlace ambiguo. Los scores sintéticos de los falsos candidatos
no representan similitudes medidas. En los dos primeros casos se aceptan los 24
enlaces verdaderos mediante evidencia explícita del generador. En el tercero
todos permanecen candidatos.

| Caso | Cruces verdaderos, política exacta, seeds 0–4 | Cruces verdaderos, agrupada, seeds 0–4 | Cobertura por ejecución |
|---|---|---|---|
| Transformaciones visuales | 14, 14, 14, 16, 13 | 0, 0, 0, 0, 0 | 22 evaluables; 2 desconocidos |
| Secuencias y repeticiones | 13, 13, 16, 9, 10 | 0, 0, 0, 0, 0 | 22 evaluables; 2 desconocidos |
| Solo candidatos | 13, 12, 11, 14, 15 | 13, 12, 11, 14, 15 | 23 evaluables; 1 desconocido |

Antes de partir se detectan seis grupos exactos cross-split y los 24 enlaces
verdaderos cross-split en cada caso. Después, ambas políticas conservan cada
contenido exacto indivisible. La agrupada conserva indivisibles los componentes
aceptados evaluables. Una reserva deliberada produce cobertura desconocida:
esos enlaces se conservan en el denominador total, no se cuentan como resueltos.
Restringiendo también el diagnóstico histórico a la misma población elegible,
hay 22 cruces previos en cada caso con aceptación y 23 en el caso solo candidatos.
Las diez asignaciones de cada caso comparten exactamente las mismas ocurrencias
elegibles; la exclusión deliberada no explica la diferencia entre políticas.

Las 30 asignaciones resultaron factibles; todos los estados del solver fueron 0,
con gaps cero salvo residuos numéricos de aproximadamente 2,67e-15. Se comprobó:

- conteo independiente usando pares de ocurrencias del generador;
- igualdad de membresías al retirar todos los candidatos y ambiguos;
- igualdad del análisis completo al invertir orden de registros y relaciones;
- replay de publicaciones con fuentes rehasheadas;
- igualdad de ambas políticas en todas las semillas sin restricciones adicionales.

La aceptación es parte explícita del fixture: **no se evalúa la capacidad de
descubrir o adjudicar relaciones automáticamente**. Estos datos validan el manejo
de dependencias declaradas; no miden escenas FAC ni rendimiento de YOLO.

## 3. Auditoría retrospectiva FAC

`scripts/closure/fac_retrospective.py` revalida el manifest histórico, los hashes
de sus ZIP originales y los miembros imagen/label, sin modificarlos. Mantiene
separadas dos poblaciones que no admiten una comparación pareada directa.

### Candidato histórico canónico

Se verificaron 1.657 ocurrencias, 1.459 contenidos: train 1.178, validation 107,
test 372. Hay **198 grupos exactos cross-split y 396 ocurrencias implicadas**:
57 grupos train–validation, 141 train–test y cero validation–test.
57/107 de validation (53,27 %) y 141/372 de test (37,90 %) tienen contenido exacto
en train. Ocho grupos duplicados tienen distintos bytes de anotación; esta
auditoría no adjudica si cada diferencia es un conflicto semántico.

| Tipo de evidencia | Observación y límite |
|---|---|
| Exacta | Igualdad de bytes y cruces verificados; evidencia de overlap exacto histórico. |
| Dependencia no exacta confirmada | Cero relaciones actualmente aceptadas con autoridad suficiente; cruce no establecido, no ausencia demostrada. |
| Candidato directo imagen–vídeo | 12.737 enlaces; cero evaluables dentro del manifest histórico, todos desconocidos por falta de split del extremo vídeo. |
| Soporte compartido de grupo candidato | 1.279 contenidos cubiertos, 180 sin cobertura. 393.034 pares distintos inducidos, 189.743 con algún cruce histórico. Es una hipótesis derivada de recuperación, no 189.743 dependencias confirmadas. |
| Revisión histórica | 25 decisiones: 17 supported, 7 ambiguous, 1 unsupported. No se promueven a aceptación actual; cero asignaciones de secuencia autorizadas. |
| Ambigüedad histórica | Los siete contenidos ambiguos alcanzan siete ocurrencias train y una test; no se inventa un enlace confirmado entre ellas. |

El diagnóstico de soporte compartido deduplica pares entre grupos y pregunta si
existe alguna combinación de splits distinta entre sus ocurrencias. Sus unidades
son **pares de contenidos**, no frames, clusters verdaderos ni aristas revisadas.
No se introduce este diagnóstico como restricción del solver.

### Entrega posterior y propuestas provisionales

La entrega posterior conserva 3.377 ocurrencias; 1.400 son elegibles y 1.977
quedan excluidas. Sus 2.232 primarias con metadata histórica no tienen duplicados
exactos entre sí; esto no contradice los 198 grupos de la población anterior.
Se mantienen 1.116 pares estructurales como candidatos, sin aceptación actual.

Las políticas nuevas coinciden para cada seed porque no hay restricciones
aceptadas adicionales. Los cruces candidatos son 72, 80, 81, 64 y 70 sobre 700
pares evaluables; 416 quedan desconocidos. Seeds 3–4 mantienen incumbents
factibles con gaps 0,0288593 y 0,0117695, sin declararlos óptimos probados.
Son propuestas sobre otra población, **no un ensayo de reducción de leakage
frente al candidato histórico de 1.657 ocurrencias**.

Se reusa la evidencia temporal previa: 953 originales con posición única y 163
con múltiples apariciones. No se transfieren tiempos a variantes procesadas,
ni se convierte cercanía temporal en dependencia confirmada.

## 4. Criterios de cierre y evidencia pendiente

| Criterio | Estado y evidencia |
|---|---|
| Congelamiento de producto | Validado: 175 fuentes Python idénticas al snapshot anterior. |
| Preservación de publicaciones | Validados los 184 hashes previos y los 221 archivos del recibo de cierre anterior. |
| Diagnóstico de los 34 fallos | Comparación pareada y matriz individual en `regression-final/`; no implica CI verde. |
| Contratos generales y regresión pertinente | Tests generales y relacionados, CLI, Ruff y replay registrados en `verification/`. |
| Atomicidad y detección de cruces conocidos | Validadas en benchmark sintético; cobertura desconocida visible. |
| Candidatos sin aceptación | Validados como diagnóstico sin restricción automática. |
| Evidencia empírica de overlap FAC | Exactos históricos confirmados; candidatos y ambiguos separados. |
| Dependencias FAC completas | Pendiente adjudicación actual y estudio de cobertura. |
| Anotaciones listas para comparación | Pendiente validación y resolución documentada de conflictos. |
| Efecto sobre el detector | No evaluado; ningún entrenamiento ni métrica nueva. |
| Regresión global limpia | No cumplido mientras existan los fallos de organización. |

Para comparar finalmente detectores se requiere: adjudicar dependencias con
procedencia y cobertura explícitas; validar anotaciones; fijar la población común,
exclusiones y test protegido legítimo sin usarlo para elegir configuraciones;
congelar baselines histórico/aleatorio/agrupado y medir sus cruces residuales;
registrar entrenamiento, hiperparámetros, recursos, semillas y criterios de
comparación comunes. Solo entonces ejecutar y comparar Precision, Recall,
mAP@50 y mAP@50–95, con variabilidad y límites de interpretación.

## Paquete reproducible

El paquete privado nuevo es
`operations/jp-scientific-freeze-20261106`, bajo `FLIR_DATA_ROOT`. Contiene protocolo,
fixtures y publicaciones, matriz CSV/JSON y JUnit pareados, auditoría FAC con
denominadores, recibos de entradas, verificación y snapshot de fuentes/documentos.
Los datos, IDs, hashes reales y salidas ejecutadas permanecen fuera de Git.
Los comandos están en [REPRODUCE.md](REPRODUCE.md).
