# Método: auditoría y partición por dependencias

El objetivo operativo es reducir cruces de **dependencias definidas y aceptadas**,
preservando una comparación reproducible. No se identifica independencia universal
ni causalidad a partir de una ausencia de cruces. El detector constituye una
capa experimental posterior.

## Identidad y población

Cada ocurrencia histórica conserva su `frame_id`; el SHA256 de los bytes de imagen
define `content_id`. Copias exactas forman una unidad indivisible. La representación
y los análisis de similitud trabajan por contenido único; el balance conserva el
peso de todas las ocurrencias y sus propias anotaciones. No se resuelven conflictos
de anotación seleccionando una copia representativa.

Etiquetas ausentes no son fondos. Las etiquetas vacías cuentan como anotaciones
vacías; las inválidas y las reservas explícitas se excluyen de ambas propuestas.
La exclusión se propaga por contenido exacto y por componente aceptado para
mantener una población común. Todos los excluidos permanecen en el manifiesto.
Las diferencias de bytes de etiquetas se reportan; no prueban por sí solas
conflictos semánticos.

## Grafo y políticas

Sea V el conjunto de contenidos exactos. Las relaciones aceptadas forman E+,
y las rechazadas E−. Los componentes conexos de (V,E+) son unidades indivisibles
de `dependency_group_balanced`. Si cualquier arista E− queda dentro de uno de
esos componentes, el sistema informa una contradicción y no publica asignaciones
agrupadas. La política `exact_content_balanced` usa cada contenido como unidad.
Los candidatos y las relaciones ambiguas no afectan esas unidades.

Se reutiliza el MILP de perfiles intercambiables de
`splitting/construction.py`: grupos con vectores de balance idénticos comparten
variables de conteo, sin perder restricciones de atomicidad. Para cada split y
dimensión, se minimiza la desviación absoluta respecto del objetivo proporcional,
ponderada por familia y normalizada por `3 * max(objetivo, 1)`.
Las dimensiones son ocurrencias, objetos por clase y anotaciones vacías. El peso
de clases se reparte entre N clases declaradas. Si no hay anotaciones conocidas,
solo se balancean ocurrencias. Cada split debe recibir al menos un grupo.

Ratios, pesos, límites de nodos, tolerancia y semillas 0–4 son idénticos entre
políticas. El seed permuta perfiles y expande asignaciones de unidades
intercambiables. El orden de entrada se canoniza. Las soluciones se comprueban
directamente contra las restricciones y la integralidad; límites del solver
quedan visibles. No se exige óptimo único ni se aplica un fallback heurístico.
El balance usa slack: desviaciones de ratio/clases no son inviabilidad; menos
de tres grupos sí impide tres particiones no vacías. No se garantiza presencia
de todas las clases en cada split.

## Temporalidad, similitud y evaluación

La localización opcional recorre los fotogramas realmente decodificados. Compara
SHA256 del raster RGB completo y su shape, conservando todas las apariciones.
Esto verifica igualdad del raster, no continuidad, origen único ni identidad de
bytes de archivos codificados. Las imágenes alteradas o recomprimidas pueden
permanecer desconocidas; una búsqueda aproximada requiere revisión posterior.
No hay unión automática por vídeo, timestamp o similitud alta.

DINOv2 y CLIP siguen siendo representaciones configurables de la ruta de
investigación. Sus features son opcionales para este cierre basado en hashes.
La similitud residual es coseno sobre L2 por contenido único, con máximo por
contenido contra otros splits y agregados por partición. No constituye un umbral
de aceptación. t-SNE/PaCMAP son exploración opcional, nunca evidencia temporal ni
verdad de agrupamiento. No se usa Bhattacharyya sobre embeddings arbitrarios.

Los diagnósticos separan duplicados exactos, aristas confirmadas, aceptadas,
candidatas, ambiguas y rechazadas. Incluyen cobertura, denominadores, fragmentación,
tamaños, clases y desconocidos. El split histórico se conserva como referencia
descriptiva; su población cubierta puede diferir y debe leerse junto con cobertura.
La comparación primaria de las dos políticas sí comparte población.

## Evaluación posterior del detector

No se inicia entrenamiento en esta fase. YOLO11m es candidato por continuidad
con el piloto; su checkpoint inicial no puede haber aprendido ejemplos del test.
Antes de comparar: aprobar y congelar un test legítimo común, población de
desarrollo y anotaciones evaluables. Mantener arquitectura/checkpoint, presupuesto,
hiperparámetros, seeds y selección de checkpoints idénticos. El factor es la
política de separación. Medir Precision, Recall, mAP@50, mAP@50–95, por clase e
incertidumbre cuando haya soporte. YOLO11n histórico se informa por separado.
Un test históricamente contaminado solo permite diagnóstico retrospectivo.
