# Ejecución reproducible del clustering

Ejecutar desde la raíz del repositorio con Python 3.11. El extra `reduction`
ya contiene scikit-learn y threadpoolctl; no se añade la librería externa hdbscan.
El extra `reporting` permite ejecutar el notebook y construir su HTML.

```powershell
uv sync --extra dev --extra reduction --extra reporting
```

Crear una especificación local a partir de
`configs/clustering/inputs.example.yaml`, por ejemplo
`reports/clustering/execution/inputs.yaml`. Sustituir sus placeholders por el
manifest y los directorios existentes de features, similitud y benchmark de
reducción de cada encoder. Usar rutas relativas; el archivo con identidades
reales queda ignorado. Las referencias y tres semillas de cada reducción se
resuelven desde metadata verificada del benchmark. No se modifican embeddings,
coordenadas ni ZIP originales.

```powershell
uv run flir-pipeline clustering sweep --inputs reports/clustering/execution/inputs.yaml --config configs/clustering/dbscan_research.yaml --config configs/clustering/optics_research.yaml --config configs/clustering/hdbscan_research.yaml
```

El comando imprime el directorio `artifacts/clustering/screening/<screening_id>`.
Incluye los dos controles L2 y las cuatro reducciones semilla 0. Conservar los
resultados completos, incluso con cero o un grupo. Los cuantiles y curvas
k-distance registran las escalas; aliases idénticos o epsilon no positivo se
registran por separado. El [protocolo](clustering_protocol.md) fija la shortlist
de hasta tres configuraciones por encoder × representación × algoritmo.

```powershell
uv run flir-pipeline clustering compare artifacts/clustering/screening/<screening_id> --inputs reports/clustering/execution/inputs.yaml
uv run flir-pipeline clustering verify artifacts/clustering/comparison/<comparison_id> --inputs reports/clustering/execution/inputs.yaml
uv run flir-pipeline clustering summary artifacts/clustering/comparison/<comparison_id>
```

Reemplazar los placeholders por las rutas impresas, sin editar Python. `compare`
reutiliza vecinos de parámetros del screening y ajusta únicamente las semillas
1/2 de la shortlist reducida. Los controles originales no tienen perturbación
de semilla ficticia. DBSCAN recalcula epsilon con el mismo cuantil en cada seed.
Mantener la misma raíz de artefactos entre A y B. Los runs completos válidos se
reutilizan; las publicaciones incompletas se preservan y se rechazan.

`verify` sin `--inputs` comprueba integridad, identidades, particiones,
selección y acuerdos guardados. Con `--inputs`, comprueba además las fuentes y
recalcula métricas originales, resúmenes y medoides de cada run. Esta validación
puede durar más que algunos ajustes. No equivale a reajustar todos los modelos.

Para una ejecución individual, `--configuration-index` es 0-based dentro del
grid YAML; `--reduction-seed` aplica a t-SNE/PaCMAP.

```powershell
uv run flir-pipeline clustering run --inputs reports/clustering/execution/inputs.yaml --encoder dinov2 --representation original_l2 --config configs/clustering/hdbscan_research.yaml --configuration-index 0
uv run python scripts/build_clustering_review.py --comparison artifacts/clustering/comparison/<comparison_id> --inputs reports/clustering/execution/inputs.yaml
uv run python scripts/check_notebook_source.py
uv run ruff check .
uv run pytest
```

El builder usa `FLIR_DATA_ROOT/Imagenes.zip` desde `.env`, o el argumento
`--images-archive`. Lee imágenes verificadas en memoria y genera composites
locales; no extrae el dataset. El HTML oculta código y el notebook versionado
permanece sin outputs. Se elige una referencia descriptiva por encoder y
representación dentro del frente final, sin declarar ganador. Las otras
configuraciones candidatas permanecen en tablas completas.

Los exemplars usan tamaños mínimo/mediano/máximo, mayor fracción de secuencia y
mayor entropía de secuencia; estos últimos son diagnósticos de procedencia
inferida. Un cluster puede ocupar varios roles. Se muestran medoide y miembros
a distancias cercana, intermedia y lejana en L2 original; el ruido usa una
muestra reproducible con semilla 0 y conserva label −1. Si una categoría no
tiene suficientes miembros, no se inventan imágenes. La vista de un clustering
original reutiliza PaCMAP existente y explicita que el ajuste ocurrió en 384D/512D.

Salidas locales: `artifacts/clustering/` y `reports/clustering/`, incluyendo
`review/clustering_review.executed.ipynb` y `review/clustering_review.html`.
No versionar assignments, content IDs, imágenes, hashes ni outputs ejecutados.
Esta fase produce candidatos; el protocolo de particionamiento se define después.

En Windows, si una política del sistema impide ejecutar el launcher instalado,
la invocación equivalente es `uv run python -c "from flir_pipeline.cli import app; app()"`
seguida de los mismos argumentos. Usar `UV_PROJECT_ENVIRONMENT` para seleccionar
el entorno local existente cuando corresponda.
