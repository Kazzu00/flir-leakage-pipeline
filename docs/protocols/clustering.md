# Protocolo de clustering — definido antes del screening

Unidad: content_id único, sin replicar las 1657 ocurrencias. Seis espacios:
L2 original de DINOv2/CLIP como controles y candidatos t-SNE/PaCMAP semilla 0
como ruta principal del pipeline. No se modifican ZIP, embeddings ni reducciones.
La temporalidad disponible sigue inferida; no se declara validada por esta fase.

## Ajuste y escala

Se usa scikit-learn (versión registrada), que ya está en el extra reduction.
No se añade la biblioteca externa hdbscan. Los tres métodos reciben exclusivamente
vectores visuales, ordenados por content_id para fijar decisiones dependientes
del orden; las salidas vuelven al content_index original. Se usa float64 de cálculo,
sin renormalizar ni estandarizar. Un hilo y búsqueda exacta brute hacen explícita
la implementación. Las versiones y esta política forman parte de la procedencia.

En unitarios, ||a−b||²=2(1−cos(a,b)); distancia euclidiana y coseno preservan
orden de vecinos, salvo redondeos de las fuentes float32. Se calculan distancias
euclidianas directamente sobre los L2, sin transformar/clipear el coseno guardado.
En 2D se usa escala euclidiana propia, sin z-score. La densidad 2D no equivale a
densidad original. No hay labels, clases, splits o índices temporales en el fit.

## Fase A

DBSCAN: min_samples 5/10/20 (incluye self); k-distance es la distancia al vecino
distinto número min_samples−1. eps proviene de cuantiles lineales
0.80/0.85/0.90/0.95/0.97/0.99, por espacio. Se registran curvas, cuantiles y eps.
Cuantiles con el mismo eps exacto se identifican como redundantes y se ejecuta
solo el primero; la equivalencia queda guardada. Un eps no positivo se registra
como no ejecutable, sin reemplazarlo por una constante arbitraria.

OPTICS: min_samples 5/10/20 × xi 0.03/0.05/0.10 × min_cluster_size 10/20/30;
extracción xi, max_eps infinito y corrección de predecessor. HDBSCAN:
min_cluster_size 10/20/30/50 × min_samples 5/10/20; EOM, epsilon=0,
alpha=1, allow_single_cluster=false. Principal: hasta 57 runs por espacio, 342
en total. No se hace tuning leaf automáticamente; si EOM degenera se conserva
la evidencia y se puede justificar un análisis secundario separado.

## Métricas y ruido

Noise permanece −1; nunca se convierte en singletons. Se guardan conteos,
cuantiles de tamaños y largest_cluster_fraction respecto de TODOS los contenidos.
Flags: all_noise; single_cluster si n_clusters<=1; nearly_all_noise si noise>=0.95;
dominant_cluster si el mayor contiene >=0.90 del total. Los dos últimos son
avisos descriptivos, no filtros de exclusión. Solo n_clusters<=1 queda fuera
de la shortlist por degeneración; todo run se conserva, incluidos desfavorables.

Silhouette principal: exacta, distancia euclidiana en L2 ORIGINAL, excluyendo
noise, válida cuando 2<=n_clusters<n_clustered. La silhouette del espacio de
clustering es secundaria y se distingue. No se imputa cero a métricas indefinidas.
Medoide: minimiza la suma de distancias euclidianas originales a todos los
miembros; empates por content_id. No se usa un centroide reducido como semántica.

Cohesión visual: pares únicos i<j dentro de cada cluster, coseno original;
media/mediana/Q1/Q3 por cluster. El promedio ponderado global usa cantidad de
pares; además se publica el promedio ponderado por miembros y la mediana de
medias de cluster. Singletons no fabrican pares; sus métricas son nulas.

Temporalidad posterior: consenso de procedencia ya existente. Por cluster,
cobertura de secuencia conocida, cantidad, fracción dominante y entropía Shannon
base 2 entre miembros con secuencia conocida. Desconocidos no forman una secuencia.
Retención temporal: pares de contenidos distintos i<j, misma secuencia válida,
|delta|<=1/5/10 (incluye delta=0 si existe). Denominador: todos los pares elegibles,
incluyendo extremos noise; numerador: ambos en el mismo cluster NO noise.
Se conservan denominadores/cobertura. No son timestamps ni leakage confirmado.

Retención visual @5/10/20: vecinos originales dirigidos. Métrica principal sobre
queries no-noise, denominador n_queries_clustered*k; vecinos noise no coinciden.
También se publica la retención sobre todas las queries y su cobertura. La mezcla
histórica conserva todas las pertenencias y no se penaliza. Clases son interpretación
opcional, no criterio de selección; no se presenta pureza YOLO como calidad de escenas.

## Shortlist sin pesos inventados

Dentro de cada encoder × representación × algoritmo, Fase A conserva el frente
Pareto maximizando silhouette original, cohesión visual ponderada por pares,
retención visual@10 y temporal@5, y minimizando noise_fraction y fracción del
cluster mayor. Número de clusters y composición de secuencia se muestran, sin
objetivos de maximizar su cantidad/pureza. Temporalidad es evaluación posterior
exploratoria; seleccionar con ella no constituye validación temporal independiente.

Hasta tres referencias del frente: extremos de silhouette original, retención
visual@10 y temporal@5, en ese orden. Empates por configuration_id ascendente;
si coinciden se añaden otras filas del frente por ID hasta llegar al máximo.
No hay suma ponderada ni score compuesto. Se conservan frente completo, motivos
de exclusión y métricas. Las tres anclas son preferencias de revisión explícitas,
no demostraciones de superioridad. Métricas sin cobertura no se imputan; se
omite ese criterio solo si está ausente en todo el grupo y se registra.

## Fase B y candidatos

Solo shortlist reducida: misma configuración conceptual en semillas 0/1/2 de la
misma reducción. DBSCAN vuelve a calcular eps con su min_samples y cuantil.
Original L2 no tiene semilla de reducción: esa estabilidad se marca no aplicable,
sin fabricar tres copias de un algoritmo determinista.

ARI y AMI (normalización aritmética) para todos los pares de semillas:
all-points, con −1 como categoría, y common-clustered, intersección no-noise.
Se publican media/mínimo, N y cobertura de la intersección. Si N<2 el acuerdo
es indefinido; si una partición tiene menos de dos grupos se conserva el resultado
definido por sklearn junto con un flag de comparación trivial. No se interpreta
un acuerdo trivial o entre masas de noise como estabilidad útil.

Robustez de parámetros para TODA shortlist: vecinos inmediatos del grid de Fase A
que cambian una sola coordenada, manteniendo las demás: cuantil eps, xi, o
min_samples/min_cluster_size en HDBSCAN. No se comparan algoritmos diferentes como
si fueran perturbaciones. Se conservan comparaciones con vecinos degenerados.

Selección final: Pareto de la shortlist, por encoder × representación, con los
criterios de A más mínimos ARI/AMI common-clustered de robustez local; en reducidos,
también mínimos de estabilidad de semillas. Se exige al menos dos clusters en
cada semilla evaluada y comparaciones common-clustered no triviales. Si no hay
candidatos elegibles se informa, sin relajar reglas. Una referencia para las figuras
por grupo: máxima silhouette original dentro del frente; empate por ID. Todo el
frente sigue siendo candidato, sin elegir un ganador universal ni producir splits.

## Fuentes de implementación

- [DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html).
- [OPTICS](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.OPTICS.html).
- [HDBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html):
  min_samples incluye self; difiere en uno respecto de la biblioteca externa.
- [Silhouette](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html).

No se ejecutan nuevos train/val/test, balanceo final, entrenamiento/evaluación
YOLO ni implementación de cluster-aware splitting en esta fase.
