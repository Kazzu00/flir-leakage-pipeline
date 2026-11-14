# Contrato del pipeline general v1

El punto de entrada es `flir-pipeline general`. El caso FAC es un adaptador;
no define el catálogo, la cantidad de vídeos, los nombres ni la disponibilidad
de anotaciones del motor. Las publicaciones son propuestas **sin autorización
de entrenamiento**.

## Entradas y esquema

`general ingest CONFIG ROOT OUTPUT` recibe una lista YAML explícita de imágenes
locales, etiquetas YOLO opcionales y vídeos opcionales. No descubre pertenencias
por nombres de archivo. Véase `configs/general_images.example.yaml`.
`general schema` exporta el JSON Schema versionado, también disponible en
`docs/contracts/general-manifest-v1.schema.json`.

El manifiesto `general_manifest_v1` contiene:

| Campo | Contrato |
|---|---|
| `classes` | Lista de `class_id` enteros no negativos y nombre; admite IDs dispersos y cero clases para datos sin etiquetas. |
| `records` | Una fila por ocurrencia, con `frame_id` original, `content_id = image_sha256`, fuente y nombre original. |
| `original_split` | Referencia histórica opcional; nunca alimenta el solver. |
| `annotation_status` | `missing`, `empty`, `valid` o `invalid`. Ausencia no equivale a fondo. |
| `label_source`, `label_sha256`, `boxes` | Ubicación, bytes originales y valores/orden originales; sin corregir ni elegir etiquetas representativas. |
| `positions` | Cero o más observaciones con vídeo, estado, tipo, evidencia y checksum. |
| `videos` | Catálogo arbitrario de IDs, ubicaciones relativas y SHA256. |
| `relations` | Aristas entre contenidos presentes en el manifiesto. |
| `sources` | Evidencias consultadas, método, versión, procedencia y SHA256. |
| `exclusion_reasons` | Motivos explícitos de reserva/cuarentena, propagados a unidades exactas y dependencias aceptadas. |

Los adaptadores pueden declarar miembros como `archivo.tar::miembro` o
`archivo.zip::miembro`, con el archivo contenedor en `sources`. El verificador
lee esos miembros sin extraerlos, comprueba sus hashes y vuelve a parsear las
etiquetas. Rechaza miembros ausentes/ambiguos. La ingesta YAML directa acepta
archivos locales; la ingesta de archivos comprimidos utiliza manifiestos de
adaptadores. No se admiten rutas que escapen de la raíz declarada.

La identidad de dataset reutiliza la receta histórica con la nueva versión
de manifiesto y el marcador explícito `missing` para ausencia de etiquetas.
Cambios en relaciones/temporalidad se registran además mediante el checksum del
snapshot de evidencia; no redefinen la identidad de los bytes de imagen.

## Evidencia y decisiones

Una relación declara `left`, `right`, `kind`, `status`, `accepted`, puntuación
opcional y evidencia. Tipos: `exact`, `validated_variant`, `visual`, `temporal`,
`external`. Estados: `confirmed`, `candidate`, `ambiguous`, `rejected`.

Solo `accepted=true` sobre estado `confirmed` une contenidos distintos. Necesita
revisor y fecha atribuibles. Una relación confirmada pero no aceptada se informa
por separado y no impone coasignación. La identidad exacta ya viene representada
por `content_id`; una arista `exact` no puede unir hashes distintos.

`rejected` significa **cannot-link del agrupamiento**. Si una cadena aceptada
une sus extremos, se publica el conflicto y se bloquea la política agrupada.
No significa que imágenes independientes deban ocupar splits diferentes.
No se elige silenciosamente entre restricciones contradictorias.

Las posiciones distinguen `decoded_pts`, `nominal_sample` y
`external_timestamp`. Una posición decodificada exige ordinal realmente
decodificado, PTS, base de tiempo, SHA256 del raster y correspondencia aritmética
del tiempo. La línea de tiempo solo coloca PTS confirmados. Las apariciones
múltiples se conservan; no se elige una por proximidad nominal. El verificador
revalida la integridad de la evidencia importada, no adjudica su validez humana.

## Etapas y salidas

1. Ingesta/QA no mutante y conservación de todas las ocurrencias.
2. Localización opcional por raster exacto, o importación explícita de evidencia.
3. Validación de aristas, componentes aceptados y contradicciones.
4. Población común: exclusiones propagadas a contenido/componente aceptado.
5. `exact_content_balanced` y `dependency_group_balanced`, semillas 0–4.
6. Diagnósticos con cobertura/denominadores, HTML si existe PTS confirmado.
7. Publicación inmutable y replay con fuentes rehasheadas.

Las dos políticas usan ratios explícitos y el mismo MILP, pesos y presupuesto.
Los grupos son indivisibles. Menos de tres grupos da `infeasible`; contradicciones
dan `blocked_conflicting_constraints`; un solver sin asignación válida da
`solver_no_valid_assignment`. `assigned` significa que existe una asignación
factible comprobada, no necesariamente óptima. Se conservan status, mensaje y gap.

Cada publicación contiene manifiesto, configuración efectiva, relaciones,
asignaciones por política/semilla, análisis, recibos y procedencia. Los informes
de replay se escriben fuera de ella. Los `.partial` fallidos no se sustituyen
por publicaciones completas. Los contratos/esquemas se versionan en Git;
los manifiestos y resultados privados se mantienen fuera del repositorio.

Features opcionales: NPZ con `content_ids`, `embeddings_l2` y `feature_space_id`.
Se validan unicidad, pertenencia, shape, finitud y norma L2. Se informa la cobertura
parcial. No se concatenan labels, diagnósticos ni texto a CLIP/DINOv2.
