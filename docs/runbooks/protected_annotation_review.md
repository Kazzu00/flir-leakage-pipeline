# Revisión humana de anotaciones protegidas v1

Fase: QA de anotaciones del test común **propuesto**, sin adjudicación automática.
No construye splits, entrena, aprueba ni congela el protocolo. El paquete es privado
y debe permanecer fuera del repositorio público.

```powershell
uv run --no-sync python -c "from flir_pipeline.cli import app; app()" detection protected-annotation-review `
  --audit $audit --proposal $proposal --acceptance $acceptance `
  --source-root $env:FLIR_DATA_ROOT --output $freshExternalReview

uv run --no-sync python -c "from flir_pipeline.cli import app; app()" detection verify-annotation-review `
  $freshExternalReview --audit $audit --proposal $proposal --acceptance $acceptance `
  --source-root $env:FLIR_DATA_ROOT

# Copiar la plantilla fuera del paquete y editarla manualmente; no editar la evidencia.
uv run --no-sync python -c "from flir_pipeline.cli import app; app()" detection validate-annotation-decisions `
  $freshExternalReview $humanDecisionFile
```

También se puede utilizar la entrada equivalente `uv run flir-pipeline`.
`--source-root` señala la carpeta de ZIP; las otras entradas son las publicaciones
consolidadas, propuesta interna y revisión de aceptación del protocolo.
La CLI exige los ocho contenidos del contrato v1; el parámetro Python
`expected_cases` permite fixtures sintéticas de menor tamaño.

## Evidencia y límites

El constructor verifica semánticamente la propuesta con el verificador existente,
liga la auditoría a su snapshot y comprueba los hashes de fuente declarados por la
revisión independiente. Contrasta el inventario de conflictos y sus frame/label IDs.
Los archivos de aceptación se conservan vinculados por checksum en la identidad;
estos hashes no son firmas ni aprobación atribuida de anotaciones.

Para cada par se exige una ocurrencia de test (A) y otra de cuarentena (B).
Se verifican los ZIP utilizados completos antes y después; los miembros se leen
sin extraer mediante nombre ZIP original único y offset. Se verifican SHA-256 de
ambas imágenes y labels, igualdad de bytes de imagen y concordancia de los labels
con la auditoría. No se normaliza ni reescribe el texto YOLO.

- `index.html`: ocho comparaciones, vínculos a originales y diferencias descriptivas.
- `cases/conflict-<content_id>/`: imagen original, `A.txt`, `B.txt` y `comparison.png`.
- `manifest.json`: identidad, fuentes/checksums, ocurrencias, labels originales base64,
  cajas a precisión de parsing, dimensiones, diferencias y roles test/cuarentena.
- `differences.csv`: resumen y referencias de índices de cajas (base 0 en manifest).
- `decisions.template.json`: formulario inmutable; copiar fuera para registrar decisiones.
- `receipt.json`: inventario completo y checksums. Su ausencia indica publicación incompleta.

Las cajas se proyectan sobre el raster original sin rotación EXIF, clipping ni cambio
de coordenadas. El render dibuja a resolución nativa; el navegador puede reducir la
visualización. Abrir el PNG completo para inspeccionar detalles. La imagen sin cajas
permite revisar regiones tapadas por texto. El nombre canónico de clase proviene del
catálogo del proyecto; ID 4 conserva su nombre de origen `SDZI` en la documentación
de clases, sin reinterpretación de su significado.

Las diferencias usan multiconjuntos exactos a precisión de parsing, preservando
multiplicidades e índices. Se emparejan cajas idénticas; una geometría exactamente
igual y correspondencia única permite describir un cambio de clase. Un único par
residual de igual clase permite calcular delta XYWH, pero **no prueba identidad de
objeto**. Varios residuos en ambos lados se declaran ambiguos: no se fuerza matching
por IoU ni se decide si una caja se dividió en dos. Las cajas exclusivas se superponen
en el tercer panel incluso cuando la correspondencia no puede establecerse.

Ausencia de anotación no significa ausencia real de objeto. Presencia, identidad,
separación y criterio de inclusión necesitan contexto de vídeo en los conflictos.
El paquete señala esta necesidad, pero no recupera ni inspecciona vídeos; el orden
inferido del nombre de frame sigue siendo heurístico y no un timestamp verificado.

La reproducción vuelve a recuperar las fuentes y compara **todos** los archivos
byte a byte; no basta reparar el recibo. La identidad liga código y versión Pillow;
un cambio de implementación o render requiere otra publicación. No contiene fecha
operativa variable, para permitir reproducción en el mismo entorno y fuentes.

## Decisiones y futura capa adjudicada

Cada formulario identifica conflicto y review_id. Revisor, fecha ISO, justificación
y evidencia adicional se dejan vacíos. Decisión inicial: `pendiente`. Opciones:
`conservar_anotacion_A`, `conservar_anotacion_B`, `proponer_correccion_manual`,
`excluir_de_evaluacion`. Para propuestas no pendientes el validador exige atribución,
fecha y textos completos. Si no hubo evidencia adicional, el humano debe declararlo;
si la hubo, registrar localizador, checksum y alcance de la inspección.

El validador comprueba completitud, **no autenticidad del revisor, calidad de la
justificación ni veracidad de evidencia externa**. No aplica ninguna decisión.
Una corrección manual debe referenciar sus nuevos bytes y checksum; la ingesta y
validación de esa futura capa aún no están implementadas. Se deben conservar los
hashes A/B originales, review_id, decisión atribuida y bytes adjudicados separados.
Cualquier cambio del futuro test exige nueva identidad y reverificación del protocolo.

Resolver estos ocho casos no resuelve suficiencia por clase, dependencias ni aprobación
metodológica, y nunca libera automáticamente test o cuarentena para entrenamiento.

## Pruebas

Comandos ejecutados (temporales de pytest fuera del repositorio):

```powershell
uv run --no-sync ruff check .
uv run --no-sync pytest tests/test_annotation_review.py tests/test_common_test.py tests/test_adendo.py tests/test_labeled_splits.py tests/test_detection.py tests/test_splitting.py tests/test_detector_final_report.py tests/test_detector_association.py -q --basetemp $freshExternalTemp
git diff --check
git status --short
git diff --cached
```

Resultado observado: 159 pruebas focalizadas aprobadas. La suite global no se ejecutó.
Las 12 nuevas también pasaron por separado con el CSV sintético UTF-8 BOM.

`tests/test_annotation_review.py` usa ZIP/imágenes/YOLO sintéticos y offline. Cubre
recuperación y offsets, bytes exactos, checksums incorrectos, UTF-8 con BOM, geometría,
multiconjuntos, cambios de clase, correspondencias ambiguas, decisiones incompletas,
fuentes equivocadas, rechazo de salidas inseguras y reproducción. Una bandera de
entrenamiento alterada falla aun reparando el recibo.
