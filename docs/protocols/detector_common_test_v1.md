# Propuesta de evaluación YOLO11n con test común — v1

Estado: **propuesta para revisión, no aprobada ni congelada**. No modifica el
protocolo `controlled_detector_v1`, sus 48 ejecuciones o exports. Los comandos
nuevos no entrenan ni materializan imágenes. No hay ganador seleccionado.

## Preguntas y unidad experimental

Separar dos preguntas: (1) comparar S1/S2/S3 en la misma población de desarrollo
con test idéntico; (2) estudiar la incorporación de Adendo con política, test y
controles del detector constantes. La segunda cambia tamaño y composición del
entrenamiento; no identifica por sí sola el efecto de cada factor.

Los splits candidatos anteriores tienen 30 configuraciones, 25 membresías y 25
tests distintos; las cinco coincidencias son S2/S3 históricos por semilla. Su
intersección de test es vacía. No son entrada directa de este experimento ni
30 observaciones independientes. Las nuevas membresías, una vez construidas,
deberán deduplicarse por asignación completa antes de generar runs. Distintas
semillas del detector dentro de una membresía no son réplicas de particionamiento.

## Opción A: test prospectivo prioritario

No se ha aportado una población anotada con procedencia independiente verificada.
La infraestructura declara **UNAVAILABLE**, no reutiliza frames internos bajo
una etiqueta externa. `prospective_test_v1.yaml` no contiene anchors internos.

Contrato propuesto `prospective_test_admission_v1` para una futura incorporación:

| Campo o evidencia requerida | Comprobación previa a admisión |
|---|---|
| Manifest anotado y recibo SHA256 | IDs únicos, contenido exacto, localizador físico, bytes y cobertura completa de etiquetas |
| Registro de adquisición por secuencia | Fuente atribuida, procedencia comprobable, condiciones de captura y límites de temporalidad conocida |
| Declaración de no exposición | Reviewer y evidencia de que no participó en train/val, selección de splits ni diseño del detector; fechas no inventadas |
| Auditoría contra todas las fuentes previas | Identidades exactas, relaciones visuales/temporales candidatas y revisión de coincidencias; no basta ausencia de pHash cercano |
| QA de anotaciones | Cinco clases canónicas, geometría válida, revisión de conflictos, trazabilidad de cada adjudicación |
| Soporte | Instancias, imágenes, contenidos y unidades de procedencia independientes por clase |
| Registro de decisiones | Revisión atribuida, aprobación de alcance, manifest final y checksums del freeze |

El contrato está diseñado; no se implementa aquí un importador ni se admiten
declaraciones sin evidencia como prueba de independencia. El operador debe aportar
una publicación auditada y revisión antes de una futura extensión de admisión.

## Opción B: holdout interno común candidato

Regla explícita, anterior a nuevas métricas YOLO: los anchors evaluables son las
ocurrencias históricas cuyo `original_split` es test. Se conservan sus frame_id,
content_id, bytes/IDs de anotación y procedencia. Es un holdout retrospectivamente
informado, no un test externo limpio ni una corrección retroactiva del baseline.

El grafo se construye sobre **toda la población ampliada** antes de intersectar
cada brazo. Cada content_id ya es una unidad exacta; se agregan aristas únicas
de `confirmed_provenance`, `visual_candidate` y `unresolved`. Los dos últimos
estados son hipótesis conservadoras. Rechazos y revisiones contradictorias siguen
el tratamiento fail-closed de `sensitivity_groups`; no se adjudican automáticamente.
La reserva toma todas las componentes que tocan anchors. Es idéntica para S1/S2/S3.

Se distinguen cuatro roles:

- **test evaluable candidato**: los anchors históricos comunes, sin modificar labels;
- **reserva protegida**: todo el cierre por contenido y dependencias seleccionadas;
- **cuarentena**: reserva menos anchors; nunca puede pasar a train o validation;
- **desarrollo**: registros de cada población fuera de la reserva.

Las componentes son indivisibles en la frontera reserva/desarrollo. Test y
cuarentena no son dos subconjuntos independientes: la segunda conserva dependencias
fuera del desarrollo sin ponderar repetidamente copias en la métrica. Escoger los
anchors explícitos no resuelve conflictos de labels: la calidad bloquea el freeze.
El hash del test vincula todos los campos conservados y debe ser igual entre brazos.

La evidencia local reconstruida debe consultarse en el estado del proyecto y los
recibos de ejecución. Un cierre hipotético no convierte relaciones perceptuales
en procedencia confirmada. La disponibilidad de clases no acredita potencia
estadística ni representatividad de secuencias.

## Reconstrucción de train/validation en publicación separada

La extensión `fixed_test_train_val_v1` implementa ahora la construcción provisional
y verificación/replay sobre estos pools, con dos destinos explícitos, cobertura
MILP obligatoria y fallos S1 conservados. Véase su
[contrato de publicación](../runbooks/fixed_test_train_val.md). No cambia los bytes
de las propuestas anteriores ni convierte sus campos `NOT_BUILT` en asignaciones:
los resultados nuevos viven en otro namespace y conservan el test `PROPOSED`.

La versión de propuesta publica pools y grupos; **no publica asignaciones train/val** ni
simula que el MILP histórico de tres vías resuelve un problema de dos vías.
Las condiciones necesarias verificadas son al menos dos grupos atómicos y al
menos dos grupos con soporte de cada clase. Si fallan, la combinación se marca
inviable para cobertura positiva en train y validation. Si pasan, la factibilidad
completa permanece `NOT_SOLVED`; requiere construir y comprobar asignaciones.

La adaptación implementada reutiliza `aggregate_groups`, los vectores por ocurrencia,
el objetivo normalizado de `milp_assignment` y `propagate`, introduciendo una
extensión explícita de dos destinos o restricciones fijas de reserva. El test
no se reoptimiza. Nunca se inventa un tercer split para satisfacer una API.

Para S1: permutación de grupos exactos y corte por ocurrencias sobre desarrollo.
Para S2: objetivo MILP de ocurrencias, clases y vacíos sobre los grupos exactos.
Para S3: el mismo objetivo sobre componentes ampliadas por las relaciones
seleccionadas dentro de desarrollo. Los objetivos train/val propuestos se
renormalizan desde los conteos históricos de train y val; no desde Adendo.
La asignación deberá imponer cobertura de clases o declarar incumplimiento,
sin reparar semillas ni sustituir un solver fallido por heurísticas ocultas.

Las dependencias confirmadas fuera de la reserva permanecen identificadas en
la fuente. El borrador conserva la diferencia original S1/S2 exact-content vs
S3 ampliado; antes de aprobar debe decidirse explícitamente si se convierten
en restricciones comunes de desarrollo. No se ocultan dentro de candidatos
perceptuales ni se afirma independencia para S1/S2. En las fuentes reales actuales
no hay relaciones `confirmed_provenance`; esta decisión no se resuelve por ausencia.

`verify_arm_assignments` es un gate reutilizable para el adaptador futuro: exige
todos los brazos, igualdad exacta del test/labels, cobertura íntegra de ocurrencias,
cuarentena, ausencia de contenido protegido en desarrollo, atomicidad y cinco
clases en train/val/test. Calcula aliases de membresía. No autoriza entrenamiento.

## Configuración propuesta del detector

Protocolo nuevo: `common_test_yolo11n_v1`. La configuración histórica se copia
como referencia inmutable y se vincula por SHA256, sin modificarla.

| Control | Propuesta / gate |
|---|---|
| Arquitectura y entorno | YOLO11n; Ultralytics 8.3.203; Torch 2.7.1; resolución 640 |
| Inicialización | Mismo checkpoint pretrained, con SHA256 pendiente; prohibido reutilizar pesos fine-tuned históricos expuestos al test |
| Presupuesto | 50 épocas; registrar imágenes y actualizaciones efectivas, porque igual número de épocas con datasets distintos no iguala cómputo |
| Batch | Un valor explícito común, pendiente de probe que no use el test; no variar por brazo |
| Seeds | Split 0–4; detector 42/43/44; deduplicar membresías antes de formar matriz |
| Optimización/augmentations | Referencia histórica SGD y configuración completa versionada, determinismo FP32; cualquier cambio requiere revisión |
| Selección | Checkpoint por macro mAP@50–95 de validation con cinco clases; criterio de empate debe fijarse en el freeze |
| Test | Mismos frame_id y anotaciones; ninguna consulta para elegir imágenes, semillas, hiperparámetros o ganador |

Batch, entorno operativo, checksum inicial y criterios finales de selección aún
no están congelados. El protocolo aprobado deberá elegir el estimando para el
contraste de Adendo y si necesita un control adicional de presupuesto, sin
convertirlo silenciosamente en una nueva política de particionamiento.

## Métricas e incertidumbre

Primaria macro mAP@50–95 de las cinco clases fijas; secundarias mAP@50, Precision
y Recall, globales y por clase. Mantener los umbrales históricos explícitos:
P/R confidence 0,25 e IoU 0,5; AP confidence mínimo 0,001, IoU 0,50:0,05:0,95,
NMS IoU 0,7 y max_det 300. Clase sin soporte: indefinida; nunca cero artificial
ni macro reducido. Clase con soporte pero sin detecciones: aplicar el contrato
de métricas existente. Conservar soporte por instancias, imágenes y contenidos.

Errores FP/FN/localización requieren predicciones originales, cajas, scores,
etiquetas, reglas de matching, imagen/frame_id y checkpoint. Los exports agregados
no permiten reconstruir esos datos. No se fabrican errores ni intervalos.

Promediar semillas del detector dentro de membresía, distinguir variabilidad de
entrenamiento de variabilidad entre particiones y comparar predicciones pareadas
en el test común. La inferencia debe considerar unidades de procedencia verificadas
cuando existan. Propuesta: bootstrap pareado de bloques independientes, remuestreando
los mismos bloques entre brazos y recalculando AP sobre detecciones agregadas.
No promediar AP por imagen; conservar remuestreos sin soporte como indefinidos.

El bootstrap por imagen anterior supone independencia y puede subestimar
incertidumbre. Duplicados exactos o componentes perceptuales permiten sensibilidades
agrupadas pero no certifican independencia temporal; sin bloques válidos, no
presentar sus intervalos como confirmación de generalización a nuevas secuencias.

## Estados y decisiones pendientes

`PROPOSED` → revisión atribuida → aprobación explícita → freeze verificable →
adaptador de entrenamiento validado. La implementación actual solo produce
`PROPOSED` o `UNAVAILABLE`, siempre `ready_for_training=false`. Ninguna opción CLI
aprueba ni congela. La aceptación de los splits previos no es aprobación del
protocolo del detector. Un cambio manual a READY se rechaza al recalcular el plan.

| Riesgo | Decisión o evidencia pendiente |
|---|---|
| Exposición retrospectiva del antiguo test | Preferir fuente prospectiva; si no existe, limitar inferencia a evaluación interna |
| Conflictos de anotación protegidos | Revisión por ocurrencia/contenido y adjudicación atribuida, sin edición silenciosa |
| Relaciones provisionales | Aprobar alcance de cuarentena como sensibilidad; no llamarlas ground truth |
| Soporte escaso por clase | Definir suficiencia por unidades independientes antes del entrenamiento |
| Train/val aún no construidos | Implementar extensión/adaptador y validar factibilidad, cobertura y aliases |
| Batch/checkpoint/entorno pendientes | Probe sin test y freeze de configuración/checksums |
| Métricas granulares históricas ausentes | Obtener runs, estadísticas, predicciones, materializaciones y pesos originales |
| Test consultado para selección | Separar desarrollo de evaluación final y registrar accesos/decisiones |

El dictamen de esta intervención puede ser **LISTO PARA REVISIÓN DEL PROTOCOLO**
sin que exista autorización para entrenar. Si fallan fuentes/identidades no se
publica una propuesta completa; se informa el bloqueo concreto.
