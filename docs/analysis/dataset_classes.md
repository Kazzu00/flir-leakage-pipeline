# Nomenclatura y geometría de anotaciones

Revisión del **2026-09-13**. El catálogo ejecutable único es
`src/flir_pipeline/data/classes.py`. Las tablas y figuras conservan los IDs;
los nombres de presentación no cambian los labels ni el manifest canónico.

## Evidencia del mapping

Se inspeccionaron los cinco ZIP actualmente disponibles bajo `FLIR_DATA_ROOT`
sin extraerlos ni modificarlos. El único YAML de dataset encontrado fue
`dataset_split_completo/dataset.yaml` dentro de `dataset_split_completo.zip`.
Declara `nc: 5` y un diccionario explícito de IDs, no una lista de la que se
supuso un offset. El builder verifica ese diccionario en cada ejecución y
registra el SHA256 del miembro en su metadata local, excluyendo su ruta privada.

| class_id YOLO | Nombre original del YAML | Nombre canónico / presentación |
|---:|---|---|
| 0 | vehicle | Vehicles |
| 1 | building | Buildings |
| 2 | road | Roads |
| 3 | river | Rivers |
| 4 | SDZI | Heavy Machinery |

La nomenclatura canónica procede de la literatura asociada al dataset, cuyo
orden fue confirmado por el responsable del proyecto en esta revisión. Los
primeros cuatro nombres coinciden entre YAML y orden académico; la quinta
posición corresponde a Heavy Machinery. Se documenta por tanto una **validación
por correspondencia de orden**, no una definición lingüística o expansión
demostrada de `SDZI`. Ambos nombres se conservan en `class_catalog.csv`.

La publicación no se ha consultado directamente en esta sesión: todavía faltan
su título/DOI o enlace exactos. Esta limitación bibliográfica queda explícita;
no se inventa una cita ni se atribuyen resultados del paper al proyecto.
La búsqueda pública no permitió resolver inequívocamente la referencia.

Se contrastaron también las 1657 etiquetas asociadas al manifest: IDs observados
0–4, hashes, clases presentes y cantidad de cajas concordantes. Esto verifica
consistencia de los IDs con los labels, pero por sí solo no identifica la
semántica de las clases. El parser YOLO general sigue separado de la ontología.

## Unidades y Background

- **Imágenes que contienen la clase:** una presencia por ocurrencia histórica.
- **Instancias por clase:** todas las cajas de las etiquetas del candidato,
  incluidas varias cajas de una clase en una imagen.
- **Background / background-only:** descripción de anotaciones vacías; no una
  sexta clase ni una caja ficticia. No certifica visualmente ausencia de objetos.

Las 4168 cajas corresponden a 1657 ocurrencias, no a 1459 contenidos deduplicados.
Se conservan los ocho grupos con conflictos de anotación. Las diez etiquetas
huérfanas y sus catorce objetos se auditan por separado. Los 292 labels vacíos
no contribuyen a ninguna distribución geométrica.

## Geometría por instancia

Se conservan ancho y alto YOLO a la precisión leída, antes del redondeo y
ordenamiento usado exclusivamente para comparar conjuntos de anotaciones.
`bbox_instances.csv` mantiene `frame_id`, `content_id`, miembro del label e
índice de caja en orden de parsing (no un número de línea física).

| Métrica | Definición |
|---|---|
| normalized_width | Ancho YOLO normalizado por el ancho de imagen |
| normalized_height | Alto YOLO normalizado por el alto de imagen |
| normalized_area | normalized_width × normalized_height |
| aspect_ratio | normalized_width / normalized_height |

El ratio anterior compara ejes normalizados. En imágenes no cuadradas difiere
del ratio en píxeles, que requeriría multiplicar por ancho_imagen/alto_imagen.
No se presenta área absoluta en píxeles como métrica principal.

`bbox_geometry_by_class.csv` ofrece, para cada clase y métrica: count, mean,
std muestral (`ddof=1`), median, Q1, Q3, min y max. Los cuartiles usan
interpolación lineal; la desviación muestral de una sola caja es indefinida.
Una anotación inválida conserva su QA/conteo y todas sus cajas se excluyen de
geometría; el número excluido se registra. En el candidato actual son cero.

Los boxplots de área normalizada y aspect ratio conservan los extremos y usan
bigotes de 1.5 × IQR. Las diferencias entre clases son descriptivas; no demuestran
causas, independencia estadística ni calidad de un futuro clustering.

La tabla `object_instances_by_class.csv` contiene `class_id`, `class_name`,
`instance_count` y `percentage_of_total_instances`. La tabla combinada anterior
conserva `object_instances` como alias de compatibilidad de `instance_count`.
Todos estos CSV y las figuras se generan localmente en
`reports/feature_engineering/`, sin versionar datos reales.
