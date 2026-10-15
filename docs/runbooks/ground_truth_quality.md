# Ground Truth Quality & Sufficiency Audit v1

La etapa genera evidencia externa para revisión humana del test propuesto y del
soporte de los train/validation existentes. No modifica anotaciones, construye
splits, ejecuta detectores ni aprueba un protocolo. Requiere las publicaciones
verificadas de consolidación, test común, revisión protegida y matriz de 30 brazos,
además de los ZIP originales accesibles fuera de Git.

```powershell
uv run flir-pipeline detection ground-truth-quality `
  --audit $audit --proposal $proposal --review $review `
  --partitions $partitions --source-root $sourceRoot --output $output `
  --observations $observations --evidence $priorEvidence

uv run flir-pipeline detection verify-ground-truth-quality $output `
  --audit $audit --proposal $proposal --review $review `
  --partitions $partitions --source-root $sourceRoot --evidence $priorEvidence
```

Las variables apuntan a directorios externos; `$output` debe ser nuevo y disjunto
de las fuentes. `--observations` es opcional. `--evidence` apunta al documento de
evidencia original cuando la consolidación declara su SHA-256; no se sustituye por
una copia reinterpretada de las relaciones. La verificación reproduce las tablas,
plantillas y galerías en un directorio temporal externo y compara todos sus hashes.
No vuelve a optimizar las particiones. Las versiones de implementación y bibliotecas
quedan ligadas a la identidad del paquete; un cambio requiere una nueva publicación.

El JSON opcional de observaciones tiene esta estructura:

```json
{"observations": [{"frame_ids": ["synthetic-frame"], "source": "solicitud atribuida",
  "observation": "observación preliminar que debe contrastarse",
  "status": "preliminary_not_adjudication"}]}
```

Las observaciones no rellenan decisiones humanas ni pasan por detección automática.
El índice permite abrir todas las imágenes, vehículos, maquinaria pesada, vacías
y conflictos. Cada caso incluye dos paneles derivados del mismo raster nativo,
sin aplicar orientación EXIF, más imagen y YOLO originales descargables sin cambios.
Los paneles son derivados de visualización; no sustituyen el contenido original.

## Artefactos

- `test_inventory.*`, `test_instances.*`: procedencia y SHA-256, dimensiones,
  clases, vacías, cajas a precisión original y áreas; no se redondea el YOLO.
- `support.csv`: 305 filas, cinco clases para test común y los 60 train/validation.
- `descriptive_groups.csv`, `group_selection.json`, `test_relations.json`:
  agrupaciones exactas/candidatas, ámbito inducido y procedencia de relaciones.
  `candidate` usa las relaciones previamente auditadas; `perceptual` incorpora
  un cribado pHash64 dentro de cada subconjunto al umbral registrado en la
  consolidación. Los pares nuevos se conservan en `group_selection.json` y no
  modifican las particiones ni el cierre protegido. `test_relations.json`
  conserva únicamente las relaciones previas incidentes en test.
- `group_concentration.csv`: contribuciones de imágenes, contenidos e instancias.
- `rare_class_sensitivity.csv`: soporte aritmético restante tras retirar un grupo
  positivo de Vehicles o Heavy Machinery, sin publicar nuevas particiones.
- `annotation_sensitivity.csv`: conteos hipotéticos usando cada B o todas las B
  originales; no propone una preferencia ni modifica el test.
- `prior_review/`: copia byte por byte de la evidencia anterior y sus formularios.
- `human_review.template.json` y CSV: todos los casos pendientes, sin revisor,
  fecha, criterios, justificación ni observaciones humanas inventadas.
- `manifest.json`, `receipt.json`, `report.md`, `index.html`: identidad, inventario
  verificable, informe crítico y galerías estáticas sin dependencias de red.

Para registrar observaciones, copiar la plantilla JSON a otro directorio externo.
No editar la publicación recibida. Validar la copia:

```powershell
uv run flir-pipeline detection validate-ground-truth-review $output $humanReview
```

Estados admitidos: `pendiente`, `observaciones_registradas`,
`propuesta_de_correccion`. Los dos últimos requieren atribución, fecha ISO, criterios,
observaciones y justificación. Ninguno adjudica etiquetas ni habilita entrenamiento.
La matriz permite evidencia de vídeo, posibles omisiones, confusiones de clase,
objetos agrupados y cajas parciales; son hipótesis hasta revisión humana.

## Límites metodológicos

Cobertura de cinco clases no acredita suficiencia. Los componentes perceptuales
son hipótesis, no vídeos confirmados ni tamaño efectivo independiente. El análisis
de pHash respeta las relaciones negativas previas; una cadena transitiva no implica
similitud de todos sus pares. El umbral heredado sirve para cribado, no para declarar
suficiencia. El pHash del test se verifica desde los originales; train/validation
utiliza la representación registrada en la consolidación verificada. El análisis
retira aritméticamente grupos para describir concentración; no calcula precisión,
AP, intervalos de confianza ni potencia estadística. La selección de checkpoints
debería usar exclusivamente validation, pero su estabilidad y suficiencia no están
demostradas y la elegibilidad permanece bloqueada. Test no debe guiar esa selección.

La exposición histórica del test se calcula mediante intersección de contenidos
con train histórico. El test es retrospectivo. En la fuente actual se esperan 141
contenidos expuestos, pero el código los calcula y no los impone como constante.

Una eventual capa adjudicada debe conservar hashes originales, nuevos hashes,
revisor, fecha, criterio, justificación y evidencia. Cambiar etiquetas requiere nueva
identidad y verificación del protocolo, sin sustitución silenciosa. Permanecen
pendientes suficiencia, dependencias, aprobación metodológica y freeze.
