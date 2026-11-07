# JP: timeline comparativa y separación provisional

**Corrección posterior:** los ejes nominales descritos abajo documentan la entrega
histórica. No deben localizar imágenes ni sustentar dependencias. Los artefactos
publicados permanecen intactos; aplicar el [contrato de revisión por vídeo](jp_video_grounded_review.md)
para la interpretación científica actual. No ejecutar estos comandos para crear
nuevas particiones durante el cierre de JP; usar el replay en memoria del paquete.

El adaptador consume un inventario previamente validado. No repite la auditoría,
no normaliza labels y no produce materializaciones para entrenamiento.
`eligible` significa elegible **para la simulación provisional**, no admisión
científica. Los IDs de ocurrencia del inventario no sustituyen los `frame_id`
históricos. El contenido sigue identificado por SHA256 de bytes.

## Reproducir desde un entorno limpio

Requiere Python 3.11, uv y las fuentes externas de `configs/jp_delivery.yaml`.
Instalar el entorno con `uv sync --extra dev`; no se necesitan modelos, GPU,
Transformers, Hypatia ni un servidor web. La instalación inicial puede acceder
al registro de paquetes; los tests y la ejecución con el entorno instalado son
offline. El directorio de salida debe ser nuevo y externo a fuentes/repositorios.

```powershell
$evidenceRoot = $env:FLIR_EVIDENCE_ROOT
$delivery = Join-Path $evidenceRoot 'operations/jp-delivery-new'
uv run python -m flir_pipeline.explorer.paired_delivery timeline --evidence-root $evidenceRoot --config configs/jp_delivery.yaml --output $delivery
uv run python -m flir_pipeline.explorer.paired_delivery split --evidence-root $evidenceRoot --config configs/jp_delivery.yaml --output $delivery
uv run python -m flir_pipeline.explorer.paired_delivery verify --evidence-root $evidenceRoot --config configs/jp_delivery.yaml --output $delivery
```

`FLIR_EVIDENCE_ROOT` es el directorio que contiene `fac`, `operations`, `audits`
y `revisiones`; se pasa explícitamente, sin hardcodear rutas personales.
La configuración usa rutas relativas y puede adaptarse a un paquete de evidencia
equivalente. Los originales se abren en lectura. Las imágenes se leen en memoria;
los HTML incorporan miniaturas y gráficos por vídeo. No necesitan red ni acceso
a los TAR para visualizarse después de generados.

El script del productor y su receipt son **opcionales y null por defecto**.
Su ausencia no bloquea el procesamiento. La dimensión intermedia declarada en
CSV y la dimensión final observada son hechos distintos; sin evidencia del
productor, una explicación por reescalado/composición permanece una hipótesis.

## Timeline y muestra

Se reutilizan `ordered_contents` y `comparison_figure` del timeline existente.
Cada vídeo tiene ejes independientes y búsqueda por nombre, filtros por estado,
secuencia y muestra. No se convierten índices nativos del proveedor en índices
muestreados de Hypatia. Una asignación de secuencia desconocida permanece
`unresolved`; no se inventan límites a partir de saltos o nombres.

La muestra se selecciona por orden hash con semilla fija y estratos por vídeo,
resolución, clase, vacío y cajas que intersectan píxeles cambiados. Los píxeles
distintos no equivalen a la máscara del productor. Las láminas permiten revisar
disposición visual y artefactos; no certifican integridad semántica de objetos,
negativos reales o identidad temporal exacta. No se transfieren labels nuevos.

## Dos políticas comparables

Ambas emplean el mismo algoritmo existente de permutación y cortes ponderados,
extendido a unidades atómicas; semilla 20261105 y proporciones 70/15/15.
El peso es un par, para evitar duplicar la contribución de original y NO-HUD.
No se optimizan clases usando métricas de validación.

- Aleatoria: unión por contenido exacto y pareja estructural precautoria.
- Agrupada: las mismas unidades más cierre transitivo de relaciones aceptadas.

Sin relaciones adicionales autorizadas, ambas políticas deben coincidir. Esa
igualdad no demuestra independencia ni reducción de leakage. Relaciones candidatas,
ambiguas o rechazadas nunca generan unión. Se reportan cruces candidatos por
separado; compartir un grupo candidato no confirma una dependencia entre imágenes.

El CSV opcional `relations` requiere `relation_id,left_pair_id,right_pair_id,
kind,status`. Los tipos son `sequence`, `dependency`, `pair`; los estados son
`accepted`, `candidate`, `ambiguous`, `rejected`. Una aceptación requiere además
`reviewer,decision_at,evidence_path,evidence_sha256`. Esos campos documentan la
decisión: no convierten un CSV histórico de soporte en autorización actual.

Las exclusiones por origen, geometría, conflicto de anotación o protección se
propagan por unidades exactas/aceptadas en ambas políticas. La reserva histórica
se conserva **excluyendo** sus referencias y dependencias condicionales: no se
adopta como test aprobado ni se reasigna. El `test` propuesto en esta simulación
es un subconjunto nuevo de la población elegible y no sustituye esa reserva.

## Verificación y límites

La generación ejecuta el algoritmo dos veces, invirtiendo también el orden de
entrada, y compara filas y hashes. El comando `verify` ejecutado por separado
comprueba receipts, cobertura, identidad, parejas, duplicados y aristas aceptadas
directamente, y reproduce las asignaciones desde los inputs congelados. No confía
solo en las columnas `group_id` emitidas por el constructor.

La revisión de octubre conserva D5/D6 pendientes en el paquete documental y
fronteras sin confirmar en la revisión posterior. Los 25 juicios recuperados
se contrastan con su copia histórica; los 902 candidatos BASE no se promueven.
Las secuencias recuperadas carecen de procedencia del productor verificada para
esta nueva población: no se asignan por analogía de índices.

Artefactos: `timeline_comparison.html`, `timeline_manifest.csv`,
`timeline_verification.json`, láminas de muestra; `split_manifest.csv`, repetición,
`split_config.yaml`, `policy_comparison.csv`, `split_verification.json`,
`independent_replay.json`, `provenance.json` y receipts. El HTML
`timeline_with_splits.html` incorpora las asignaciones propuestas y sus cruces
candidatos sin alterar la timeline inicial.

Para cerrar científicamente faltan decisiones atribuidas sobre correspondencias
y objetos procesados, referencias no contrastadas, conflictos de labels/tolerancia,
límites de secuencia y dependencias aceptadas, y alcance/autorización de la reserva.
La disponibilidad del código del productor **no es** un requisito de cierre.

```powershell
uv run python -m pytest tests/test_paired_delivery.py tests/test_splitting.py tests/test_sequences.py tests/test_explorer.py -q
uv run ruff check .
```

No versionar outputs, miniaturas, hashes reales ni manifests privados. Esta ruta
no implementa detector, limpieza de ruido ni segmentación grupal.
