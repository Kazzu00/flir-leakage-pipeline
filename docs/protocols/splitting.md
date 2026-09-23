# Protocolo de particionamiento

Fijado antes de la primera comparación completa. La entrada son el manifest y
los artefactos verificados existentes; no se ajustan encoders, reducciones ni
clusters. Las asignaciones reales y sus huellas quedan exclusivamente locales.

## Unidades, ruido y anotaciones

Cada contenido exacto es indivisible. En cluster-aware cada cluster no negativo
es una unidad; cada ruido conserva `cluster_id=-1` y recibe un `group_id` singleton
propio. La política `similarity-components` queda reservada, rechazada por la
configuración actual: requiere definir explícitamente un cuantil del mismo
encoder y cualquier compatibilidad temporal. No se ejecuta automáticamente.

Se suman los registros, las instancias de las cinco clases y las anotaciones
vacías por unidad. Las cajas se releen del ZIP en memoria y se contrastan con el
manifest mediante la auditoría existente. Los conflictos de anotación se conservan
por ocurrencia; ni una etiqueta representativa ni una mayoría los reemplaza.
Clases y background solo intervienen en el balance de esta etapa, nunca en fit.

## Baselines y objetivos de tamaño

Los targets train/val/test proceden de las proporciones de **registros** históricos
en el manifest suministrado; se permite una terna explícita configurable.
`original_split` no conserva memberships en ninguna partición nueva.

Historical preserva cada registro y cada conjunto de pertenencias por contenido.
Su tabla de contenidos contiene `new_split=null` si existen varias pertenencias,
y `split_membership_set` explícito. Es la única excepción al invariante de cero
duplicados exactos entre splits; no se presenta como una partición nueva válida.

Random content-level permuta uniformemente contenidos con NumPy, semillas 0–4,
y corta la suma acumulada de registros cerca de los targets. También permuta el
orden de los nombres de split para no fijar su posición en los extremos. No usa
clases ni similitud. Su distribución no es uniforme sobre todas las particiones
de tamaño fijo; es una baseline definida mediante permutación y cortes ponderados.
No se ejecuta naive record-level random.

## Selección preparatoria de clustering

Se reutiliza el frente Pareto B no degenerado. En cada estrato encoder ×
representación se conserva el candidato de menor ruido; empate por ID ascendente.
En un segundo recorrido ordenado se añade otro candidato: primero se prefieren
algoritmos aún no representados globalmente y después se maximiza la distancia
mínima a los ya elegidos. Distancia: media de diferencias absolutas de rangos
percentiles entre métricas disponibles en ambos candidatos: ruido, número de
clusters, mínimos ARI/AMI de parámetros y semillas, retención visual@10 y temporal@5.
La estabilidad de semillas de reducción no se inventa para el espacio original.
La política queda serializada; con seis estratos produce hasta 12 candidatos.
Silhouette no interviene. Un candidato grande o ruidoso no se descarta antes de
evaluar sus restricciones de particionamiento.

## MILP y reproducibilidad

Se utiliza [scipy.optimize.milp/HiGHS](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html),
ya disponible en core. Grupos con el mismo vector de balance se agregan en un
perfil intercambiable. Esto es una reducción exacta del problema de asignación:
todos tienen los mismos coeficientes y no existen restricciones por identidad.

Variable entera `z[p,s]`: cantidad de grupos del perfil p asignados al split s.
Restricciones: `sum_s z[p,s] = multiplicidad[p]`, límites enteros no negativos,
al menos un grupo en cada split, y ecuaciones de balance con holguras positivas
y negativas. Se minimiza L1 normalizada por `max(target,1)`, promediada entre los
tres splits. Las familias registros, clases y vacíos pesan 1 cada una; la familia
clases promedia sus cinco términos. Así una clase frecuente no anula las raras.
No se usa bbox geometry, tiempo, coseno, vecinos ni memberships históricos.

Los perfiles se permutan por semilla y sus grupos se expanden mediante esa misma
semilla. Se ejecutan 0–4. El límite de 32 nodos evita depender de tiempo de pared;
gap relativo solicitado 0.001. Se conserva el incumbente factible junto a status,
gap, cota dual y nodos. No se declara óptimo un run detenido por límite ni se
declara única ninguna solución. La configuración y versiones de NumPy/SciPy
entran en `split_space_id`; paths, fecha y device no. Los límites del solver y
sus posibles diferencias entre versiones son parte de la reproducibilidad.

Durante la ejecución se observó un código HiGHS 16 (`Solution limit reached`)
que SciPy 1.17.1 expuso como status 4, acompañado de un incumbente. Se verifica
directamente finitud, límites, integridad y **todas** las restricciones primales
antes de aceptarlo como factible, conservando el status y gap originales. Esta
corrección de compatibilidad no cambia objetivos, unidades ni regla de selección.

## Evaluación posterior

Se evalúan **ambos encoders por separado** usando matrices y vecinos existentes.
Un par cruza splits si existen dos ocurrencias con pertenencias diferentes,
conservando todos los memberships históricos. Los pares son contenidos únicos
i<j, excluyen self-content, y no se ponderan por multiplicidad. Los duplicados
exactos se contabilizan aparte. El NN cross-split se busca en la matriz completa,
no solo en top-20; ties siguen el orden de contenidos del artefacto verificado.
Se reportan mean, median, Q1/Q3, p90/p95/p99/max, rank-1 y fracciones top-5/10/20.

Los seis cohortes de pares reutilizan los thresholds originales inclusivos,
con empates, de cuantiles .9/.95/.975/.99/.995/.999 propios de cada encoder.
La temporalidad usa consenso de secuencia/índice inferidos, pares de contenidos
distintos y ventanas acumuladas Δ≤1/5/10/25. Se publican denominadores y unknowns.
La fragmentación de secuencias es descriptiva. No se convierten índices a segundos.
Cluster fracture excluye ruido y se compara con historical/random para cada
candidato; todos los grupos usados en nuevos splits deben permanecer íntegros.

## Robustez y shortlist final

Se mide la retención del **mismo nombre de split**, sin permutar train/val/test,
entre los diez pares de semillas. Se reportan mean/std poblacional/min/max de las
métricas. No se seleccionan semillas por menor correlación: seed 0 representa
siempre cada candidato final.

Elegibilidad exige cinco semillas válidas, cero duplicados/fracturas, las cinco
clases presentes en cada split y desviación relativa de registros ≤10% en cada
split y semilla. El 10% es una tolerancia operativa previa de tamaños, no una
afirmación sobre calidad del detector. Se muestra también todo candidato excluido.

Pareto minimiza el peor valor entre semillas de: desviación relativa de registros,
desviación media absoluta de composición de clases (puntos porcentuales), media
NN cross-split y fracción cross-split top 0.1% en cada encoder, y fracción temporal
Δ≤5; además minimiza el máximo std poblacional de media NN entre encoders.
No hay score ponderado de selección. El frente completo se conserva. Hasta tres
anclas finales minimizan respectivamente el peor top 0.1% DINOv2, CLIP y desviación
de clases, con empate por ID; anclas coincidentes se deduplican. No se inventa
diversidad si solo una configuración sostiene todos los extremos.

## Límites de interpretación

Estos datos de similitud ya intervinieron en representación/clustering y coherencia
posterior de selección. Separar construcción de evaluación evita introducirlos
directamente en el solver, pero no convierte la evaluación en un test externo
independiente. La revisión multicriterio sigue siendo exploratoria/adaptativa.
La baseline aleatoria no balancea clases; las diferencias de balance reflejan
también esa elección. Una futura ablación de singleton balanceado aislaría mejor
la contribución exclusiva del agrupamiento. Singleton noise permite correlación
residual. Los resultados no prueban causalidad de leakage ni mejoran por sí mismos
las métricas de un detector. No se materializan imágenes ni se entrena YOLO.
