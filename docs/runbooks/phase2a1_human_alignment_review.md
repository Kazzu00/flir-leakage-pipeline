# Fase 2A.1 — revisión humana de alineamiento

El piloto utiliza exclusivamente correspondencias y controles ya guardados en
`alignment-evidence-v1`. No realiza búsqueda, cambia rankings ni actualiza umbrales.
La unidad es **una imagen original y su top-1 propuesto**, con todas las alternativas
disponibles. Los juicios no equivalen a continuidad de toma, identidad geográfica,
secuencia independiente o ausencia de leakage.

## Abrir el piloto local

```powershell
uv sync --extra dev --extra alignment
uv run flir-pipeline sequences alignment pilot serve reports/phase2a1-pilot reports/phase2a1-reviews.sqlite3 --port 8766
```

Abrir `http://127.0.0.1:8766`. El servicio solo escucha loopback, rechaza Host/Origin
externos y exige un encabezado explícito en escrituras. No usa servicios externos.
La interfaz pertenece al paquete Python y su CLI de alineamiento; no se modificó
la aplicación React ni el ledger histórico Human Audit v1.1.

La ejecución entregada tiene **24 pendientes implícitos y cero anotaciones humanas**.
No hace falta crear revisiones falsas para representar el estado pendiente.
Los experimentos de persistencia escriben únicamente en un piloto sintético separado.

## Selección y preparación reproducibles

```powershell
$facRoot = Read-Host 'Directorio FAC original, solo lectura'
uv run flir-pipeline sequences alignment pilot prepare $facRoot reports/phase2a-complete reports/phase2a1-pilot --seed 20261103
```

La preparación es idempotente para la misma evidencia y selección: un directorio
existente debe verificar sus checksums y pilot ID, o se rechaza. No se sobrescribe
el paquete original de Fase 2A. Para otra selección se requiere otro directorio.

Composición sin duplicación de `image_id`:

| Estrato | n | Regla |
| --- | ---: | --- |
| Diagnósticos | 8 | 199, 200, 349, 350, 399, 400, 499, 501 |
| Adicionales | 12 | Cuatro por train/val/test, semilla fija, excluyendo diagnósticos y 630 |
| Difíciles | 4 | 630 y tres casos restantes con menor margen de inliers entre top-1 y alternativas |

El 630 ocupa una plaza especial dentro de los 24; solo 23 tienen match automático
encontrado. El 500 permanece ausente. No se incorpora una imagen homónima de otro
archivo. El conjunto es dirigido, no representativo de la población.

Los controles negativos **propuestos** se toman del control existente de menor PTS
por caso. No son etiquetas verdaderas automáticas; un humano puede aceptarlos,
rechazarlos o declararlos inciertos. No se agregan como casos duplicados del piloto.

Se reextraen los cinco PTS existentes y el control de cada imagen, comprobando el
hash RGB guardado en Fase 2A. Se exporta una copia PNG visual del original y se
conserva su hash de bytes original. Para las alternativas se reconstruyen SIFT y
RANSAC sobre la región interior CLAHE de 384×216, razón de distancias 0,75,
tolerancia RANSAC 3 px y semilla de la evidencia fuente. Se guardan coordenadas,
máscara de inliers y homografía, sin cambiar las métricas originales ni el ranking.
Las imágenes de líneas muestran inliers verdes y outliers rojos.

La superposición se muestra únicamente con al menos cuatro inliers, matriz finita
no singular y cuadrilátero transformado convexo con área plausible. Ese filtro es
numérico, no una demostración de identidad. Las métricas guardadas de Fase 2A y los
conteos reconstruidos se muestran separados, incluida su concordancia.

## Revisar un caso

1. Comprobar imagen original, split histórico, ID, hash y procedencia desplegable.
2. Comparar top-1 con el original y examinar **todas** las alternativas. Ninguna se
   oculta por tener puntuación cercana; se indica cercanía de soporte por inliers.
3. Consultar métricas, líneas, superposición y PTS. El intervalo es soporte observado,
   no un intervalo de confianza. La precisión del instante de captura JPEG sigue
   desconocida. Los fotogramas candidatos son las representaciones 384×216 de Fase 2A.
4. Marcar explícitamente las imágenes consultadas. Abrir una imagen no marca consulta.
5. Registrar el juicio de top-1: `correct_match`, `incorrect_match` o `uncertain`.
   La incertidumbre sigue siendo una respuesta válida y no exige juicio binario.
6. Indicar confianza humana, revisor y justificación con referencias visuales,
   alternativas y límites. Si se juzga el control, la confianza global debe ser
   explícita para cualquier conclusión, aunque top-1 sea incierto.
7. Si otra alternativa resulta mejor, indicar su rango y consultar su imagen.
   Esto registra preferencia; no verifica automáticamente esa alternativa.
8. Si no se encuentra correspondencia, indicar **en las evidencias presentadas**.
   No significa que sea inexistente en todo el vídeo.
9. Guardar como revisado o mantener pendiente. “Siguiente pendiente” recorre la muestra.

Una revisión conclusiva no se guarda sin original y top-1 consultados, justificación,
revisor y confianza. Una alternativa o control juzgado exige su evidencia propia.
Las referencias ajenas a otro caso o con bytes alterados se rechazan.

`human_verified_top1` solo se deriva de una acción explícita `reviewed` +
`correct_match` que supera esos controles. No modifica el artefacto de matching ni
el antiguo contrato de continuidad/dependencia. Una persona debe realizar la acción;
el servicio valida trazabilidad, no puede certificar la identidad de quien escribe.

## Persistencia, correcciones y exportación

El SQLite del piloto se encuentra fuera del paquete inmutable. Se rechaza usar una
base ajena, incluido el ledger histórico. Los guardados son transaccionales, con
`expected_version`; cambios concurrentes obsoletos se rechazan. Cada corrección
agrega una versión y preserva las anteriores. Volver a pendiente retira ese caso
del análisis vigente, sin borrar su historia.

```powershell
uv run flir-pipeline sequences alignment pilot export reports/phase2a1-pilot reports/phase2a1-reviews.sqlite3 reports/reviews-001.json --format json
uv run flir-pipeline sequences alignment pilot export reports/phase2a1-pilot reports/phase2a1-reviews.sqlite3 reports/reviews-001.csv --format csv
uv run flir-pipeline sequences alignment pilot report reports/phase2a1-pilot reports/phase2a1-reviews.sqlite3 --minimum 5
```

JSON conserva todas las versiones y los pendientes iniciales por separado. CSV usa
`record_type=history` o `initial_pending`, con `decision_json` íntegro para no perder
campos. El pendiente inicial tiene versión 0 y no se atribuye a un revisor.
Los destinos con contenido diferente no se sobrescriben. Respaldar el paquete y
la base del piloto con el servicio detenido. No copiar revisiones a otros paquetes
por semejanza de índices; el ledger está vinculado al hash del piloto.

## Interpretación del análisis

Se usa la última versión de cada caso. Se reportan conteos de correctos, incorrectos,
inciertos, pendientes, alternativas mejores y ausencia de correspondencia presentada.
Se publican dos denominadores explícitos:

- `correct_among_reviewed`: correctos / todos los revisados, incluidos inciertos.
- `correct_among_conclusive`: correctos / (correctos + incorrectos).

Cada proporción permanece `null` hasta reunir al menos cinco observaciones en su
denominador por grupo; `--minimum` permite un corte explícito. Cinco es una barrera
operacional, **no un tamaño muestral que valide calibración o inferencia poblacional**.

Fase 2A no produjo confianza automática calibrada. El desglose correspondiente usa
`not_available_not_calibrated`. Como diagnóstico aparte se muestran bandas de
inliers (`ge40`, `lt40`, abstención), definidas antes de las anotaciones; no son
probabilidades ni se usan para modificar el matching.

La tabla de abstención compara `not_found` automático con referencias humanas
explícitas: top-1 correcto o ausencia en lo presentado. Un top-1 incorrecto por sí
solo no prueba ausencia de toda correspondencia. Se informa también el rechazo,
aceptación e incertidumbre de controles; no se presume su verdad terreno. Con un
solo caso automático de abstención, el piloto tiene capacidad limitada para evaluar
ese comportamiento. No se ajustan umbrales con test ni con las revisiones recibidas.

## Verificación técnica

```powershell
uv run python -m pytest tests/test_alignment_pilot.py tests/test_alignment.py tests/test_audit_review.py tests/test_audit_v11.py tests/test_smoke.py -q
uv run ruff check .
node scripts/check_alignment_pilot_browser.cjs
```

El chequeo de navegador usa Playwright del checkout hermano `flir-pipeline-explorer`.
Crea un piloto sintético aislado, guarda/corrige/reinicia y comprueba las exportaciones.
En el piloto FAC solo hace GET, recorre imágenes y comprueba que el historial no
cambie tras reinicio. No escribe anotaciones reales.

Véase el [reporte de selección y resultados](../analysis/phase2a1_pilot_validation.md).
