# Train/validation con test común fijo v1

Fase: construcción **provisional** de particiones para revisión. No materializa
datasets, adjudica anotaciones, consulta resultados del detector para asignar,
aprueba/congela test ni entrena. El protocolo histórico queda separado.

## CLI y secuencia obligatoria

Las variables son rutas locales externas; no se versionan.

```powershell
uv run --no-sync python -c "from flir_pipeline.cli import app; app()" splitting fixed-test-train-val `
  --audit $audit --previous $previous --proposal $proposal --review $annotationReview `
  --output $freshPilot

uv run --no-sync python -c "from flir_pipeline.cli import app; app()" splitting verify-fixed-test-train-val `
  $freshPilot --replay

# Solo después de verificar el piloto de seis brazos, sin fallos.
uv run --no-sync python -c "from flir_pipeline.cli import app; app()" splitting fixed-test-train-val `
  --audit $audit --previous $previous --proposal $proposal --review $annotationReview `
  --output $freshFullMatrix --full --pilot $freshPilot

uv run --no-sync python -c "from flir_pipeline.cli import app; app()" splitting verify-fixed-test-train-val `
  $freshFullMatrix --replay
```

Entrada equivalente: `uv run flir-pipeline`. La configuración por defecto es
`configs/splits/fixed_test_train_val_v1.yaml`; `--config` permite otra configuración
explícita. El piloto usa semilla 0 y las seis combinaciones de población/política.
La matriz completa exige exactamente 0–4, las mismas fuentes, código, entorno y
parámetros que un piloto verificado sin fallos. No permite omitir semillas.

## Población y protección

Se validan recibos e identidades de auditoría, splits previos, propuesta y revisión
de anotaciones. Los splits previos son evidencia de procedencia: sus asignaciones
no se reutilizan. La propuesta se reconstruye con su verificador y vuelve a derivar
el cierre transitivo sobre toda la población ampliada antes de intersectar cohortes.
La revisión humana permanece pendiente; sus checksums no equivalen a aprobación.
Esta etapa verifica snapshots y reconstruye las etiquetas desde la auditoría;
no rehasha ZIP originales ni reproduce los PNG de revisión de anotaciones.

La fuente real de revisión contiene un `cases.zip` añadido después de publicar su
recibo. Se verificaron sus 32 miembros contra las evidencias declaradas: todos son
copias byte a byte. El adaptador acepta únicamente ese suplemento de transporte
redundante, con cobertura exacta y sin miembros ambiguos; lo registra por separado
en `suite.json/review_transport` y lo vincula en los hashes de fuentes. No modifica
el recibo de revisión ni añade evidencia de anotación. Otros extras o bytes distintos
se rechazan. La primera invocación se detuvo antes de publicar al detectar el extra.

La propuesta real reserva 514 ocurrencias/373 contenidos. Conserva exactamente
372 ocurrencias de test y 142 en cuarentena global. El brazo histórico contiene
513 reservadas (372 test + 141 cuarentena); la ocurrencia restante de cuarentena
pertenece a Adendo y solo aparece en la población ampliada. Esto no cambia la
protección común: ningún contenido de la reserva global entra en desarrollo.

Se preservan todos los campos del manifest, incluidos bytes/identidades de labels,
frame_id, content_id y procedencia. El verificador compara el manifest completo,
no solo un conteo o test_id. Cero cruces exactos se comprueba entre train/val/test;
test y cuarentena pueden compartir contenido por definición de la reserva.

## Proporciones y construcción

Se aplican al **pool de desarrollo**, sin incluir test o cuarentena:

`train = 1178 / (1178 + 107) = 0.9167315175097276`

`validation = 107 / (1178 + 107) = 0.08326848249027237`

Los números proceden de las ocurrencias históricas verificadas; ambas poblaciones
usan la misma regla. No se reutiliza un vector de tres proporciones ni se infiere
un split histórico para Adendo. La metadata registra conteos y derivación.

- **S1:** permutación reproducible de contenidos exactos y corte cercano al objetivo
  por ocurrencias. No usa clases para asignar, no reintenta semillas ni repara cobertura.
- **S2:** MILP con grupos exactos indivisibles; balance de ocurrencias, instancias de
  las cinco clases y etiquetas vacías.
- **S3:** mismo MILP, con componentes de relaciones seleccionadas dentro del desarrollo.
  Se preservan IDs de evidencia y estados; los candidatos no pasan a procedencia confirmada.

La extensión de `random_assignment` y `milp_assignment` recibe destinos explícitos.
El comportamiento por defecto sigue siendo train/val/test para consumidores históricos.
No se introduce un tercer destino ficticio. Se reutilizan agregación, propagación,
objetivo L1 normalizado y cálculo de balances. El objetivo de dos vías es:

`sum_{s,k} weight_k * abs(actual_sk - target_sk) / (2 * max(target_sk, 1))`

Los pesos de familia son registros=1, clases=1 repartido en cinco y vacíos=1.
El MILP nuevo impone no vaciedad y al menos una instancia de cada clase en cada
destino. Agrupar vectores idénticos conserva esas restricciones lineales; la
expansión aleatoria restaura los grupos atómicos. La factibilidad se comprueba
directamente (bounds, restricciones e integralidad), además de validar las asignaciones.

Presupuesto determinista: 32 nodos, tolerancia relativa 0.001. Una solución factible
no se presenta como óptima si la brecha no lo acredita. Se registran estado/mensaje,
objetivo, cota dual, brecha, nodos y comprobación primal. Tiempo de pared no forma
parte de la identidad ni de la comparación de replay. Se conserva procedencia de
ejecución y versiones NumPy/SciPy/Pandas; replay exige el código/entorno registrados.

S1/S2 no imponen las dependencias internas de desarrollo; las reportan por estado.
S3 sí impone las seleccionadas por el protocolo. La protección contra la reserva
es idéntica para todos. Esta sensibilidad no demuestra independencia temporal.

## Artefactos y fallos

- `sources/`: snapshots inmutables de las cuatro publicaciones de entrada.
- `pilot/`: en la matriz completa, piloto aceptable que autorizó ejecutar las demás semillas.
- `suite.json`: configuración, fuentes, versiones, test/plan IDs y derivación de proporciones.
- `provenance.json`: contexto de ejecución y límites de la verificación de fuentes.
- `runs/<population>-<policy>-<seed>/`: metadata/identidad, grupos, vectores de balance,
  asignaciones completas (train/val/test/quarantine), asignaciones de contenido de
  desarrollo, métricas y soporte por clase con denominadores explícitos.
- `comparison.csv`: comparación de brazos, estados, balance, objetivos y brechas.
- `summary.json`: fallos, membresías distintas, aliases reales y dictamen provisional.
- `receipt.json`: inventario completo con checksums, escrito después de verificación semántica.

Una asignación S1 sin cobertura se conserva como `FAILED_CLASS_COVERAGE`. Un solver
sin incumbente o soporte atómico insuficiente genera `FAILED_CONSTRUCTION`, con
evidencia explícita y sin asignaciones inventadas. Se procesan todos los brazos de
la fase; ningún fallo se elimina del resumen. El verificador reproduce siempre
los fallos de construcción. La CLI retorna código 2 si hay fallos, manteniendo sus
artefactos. Los errores de integridad abortan sin recibo final; no reutilizar esas carpetas.

Un resumen válido no implica que todos sus brazos pasaron: distinguir validez del
registro de fallo y elegibilidad provisional para revisión. Los aliases se calculan
desde asignaciones completas, ordenadas por frame_id; comparar poblaciones distintas
cambia el conjunto de desarrollo aunque el test sea el mismo. No elegir ganadores.

## Estados y restricciones científicas

`PROVISIONAL_VERIFIED` solo acredita invariantes de particionamiento. El estado de
suite es `PROVISIONAL_CANDIDATES`, siempre `ready_for_training=false`. Cambiar ese
estado o las membresías y recalcular checksums no evita la reconstrucción semántica.

Siguen pendientes adjudicación humana, decisión sobre suficiencia por clase y
dependencias, aprobación/freeze del test, adaptador de materialización validado y
configuración/entorno del detector congelados. Ninguna CLI recibe opciones para
saltarse esos requisitos. La propuesta de entrada permanece `PROPOSED`.

## Validación de software

```powershell
uv run --no-sync ruff check .
uv run --no-sync pytest tests/test_fixed_test.py tests/test_splitting.py tests/test_labeled_splits.py tests/test_common_test.py tests/test_annotation_review.py tests/test_detection.py tests/test_detector_final_report.py tests/test_detector_association.py -q --basetemp $freshExternalTemp
git diff --check
git status --short
git diff --cached
```

Fixtures sintéticas, offline, sin modelos ni GPU: destinos, cobertura, restricciones
incompatibles, atomicidad, protección transitiva, aliases, conservación, replay,
checksums y falsa promoción de estados aun con recibos recalculados. Se prueban
también el suplemento de transporte y la publicación explícita de un solver fallido,
con bloqueo de la matriz completa. Los resultados realmente ejecutados se registran
en [status.md](../status.md); no se infiere ejecución por existir código.
