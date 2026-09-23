# Protocolo de reducción — definido antes del grid completo

Esta fase estudia t-SNE y PaCMAP sobre contenidos únicos, con los embeddings L2
completos de cada encoder por separado. No genera etiquetas de clustering ni
nuevas particiones. IDs temporales, splits y anotaciones se utilizan solo después.

## Configuraciones predefinidas

Por encoder: t-SNE con perplexity 10/30/50, 1000 iteraciones máximas, learning rate
auto, early exaggeration 12, Barnes-Hut angle 0.5 e inicialización PCA; semillas
0/1/2. Nueve ejecuciones. El learning rate efectivo se conserva en metadata.

PaCMAP: n_neighbors=10, FP_ratio=2, MN_ratio=0.2/0.5/1.0, fases 100/100/250,
learning rate=1, métrica euclidiana, FAISS e inicialización PCA; semillas 0/1/2.
Nueve ejecuciones. La base controlada es MN_ratio=0.5. Se varía solo la cantidad
de pares mid-near (2/5/10) manteniendo vecinos y further pairs (10/20); esto
explora el balance de atracciones, sin suponer de antemano qué variante es mejor.

Salida principal 2D; la API acepta 3D, que no se ejecuta en el grid principal.
No se usa PCA previa como técnica de preprocesamiento. La PCA de inicialización
produce la posición inicial en baja dimensión y no sustituye los vectores usados
para construir relaciones. El N actual permite estudiar directamente 384/512D.

PaCMAP 0.9.1 tiene `apply_pca=True` por defecto: se fija explícitamente en false.
Su implementación aplica aún una transformación afín interna: mínimo escalar,
división por el rango escalar y centrado por columna. Bajo distancia euclidiana,
preserva distancias salvo un factor positivo común y redondeo numérico; no es
una nueva normalización L2 ni modifica los archivos originales. Parámetros
efectivos, dimensiones y conteos de pares se verifican tras ajustar el método.

Se fija un hilo nativo/Numba/FAISS y se ejecutan fits secuencialmente. La semilla
controla inicialización y muestreo donde la biblioteca lo expone; no se promete
identidad bit a bit entre plataformas/versiones. Se comprueba repetición sintética
con la misma semilla. Las versiones de implementación forman parte de la identidad.

## Métricas y unidades

La referencia original es la distancia 1−coseno de los embeddings L2 ya
verificados. Sus vecinos se reutilizan desde la fase de similitud y se contrastan
contra los rankings completos. En coordenadas reducidas se usa distancia
euclidiana. Los empates se resuelven por content_id ascendente y se excluye self.

Trustworthiness penaliza vecinos intrusos mediante el rango en el original;
Continuity penaliza vecinos omitidos mediante el rango en la reducción.
Ambas usan la normalización exacta 2/[N·k·(2N−3k−1)], con k<N/2. Se evalúan
k=5/10/20 y se prueban contra la implementación de referencia de scikit-learn
en datos sin empates. Los empates de esta API tienen una convención explícita.

Jaccard@5/10/20 compara conjuntos por contenido y reporta media, mediana, Q1/Q3.
Spearman compara distancias sobre 100000 pares únicos sin reemplazo, semilla 0,
muestreados en orden canónico de IDs; no es la métrica principal y no se reportan
p-valores que supongan independencia de pares. La estabilidad compara todos los
pares de semillas con Jaccard de vecindarios, sin comparar coordenadas crudas.

## Regla de candidatos, fijada antes de observar resultados

Se agrupan las tres semillas de cada configuración. Para cada encoder/método se
exigen las tres ejecuciones válidas y ausencia de colapso completo/rango deficiente.
Se calculan cinco criterios a maximizar: media de trustworthiness en k=5/10/20,
media de continuity en esos k, Jaccard original-reducido medio en esos k,
estabilidad Jaccard media en esos k, y Spearman medio. Cada criterio se agrega
primero por semilla y después entre semillas; estabilidad usa los tres pares.

Se conserva el frente no dominado y se elige un candidato mediante la menor
media de rangos descendentes de los cinco criterios, con igual peso y empates
promediados. Desempate: mayor estabilidad, mayor preservación Jaccard, mayor
trustworthiness, mayor Spearman y configuration_id ascendente. La semilla
representativa es siempre 0; no se elige la proyección más atractiva.

Es una regla exploratoria transparente, no una prueba de superioridad ni de
idoneidad para clustering. Se conservan las alternativas y métricas por run.
Los criterios están relacionados; los pesos/rangos son una decisión metodológica
explícita y no una función objetivo validada para rendimiento de detección.

## Límites para la siguiente fase

t-SNE y PaCMAP son transformaciones no lineales: la densidad en 2D no equivale
a densidad en embeddings. Un candidato no es un clustering validado. La etapa
posterior deberá contrastar estabilidad y coherencia, tratar el ruido y comprobar
relaciones en el original; AMI/ARI requieren asignaciones reales de agrupamiento.
DBSCAN/OPTICS/HDBSCAN, nuevos splits y YOLO permanecen fuera de esta iteración.

Fuentes de implementación consultadas:

- [TSNE de scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html).
- [Trustworthiness de scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.trustworthiness.html).
- [PaCMAP oficial](https://github.com/YingfanWang/PaCMAP), contrastado con el código
  instalado de la versión 0.9.1 y sus parámetros efectivos.
- [Artículo de PaCMAP](https://jmlr.org/papers/v22/20-1061.html).
