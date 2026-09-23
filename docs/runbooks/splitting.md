# Ejecución de particiones y comparación

Desde la raíz, con Python 3.11 y los artefactos completos existentes:

```powershell
uv sync --locked --extra dev --extra reduction --extra reporting
```

Reutilizar el YAML local descrito en `configs/clustering/inputs.example.yaml`:
manifest y directorios de features/similarity de ambos encoders. `--comparison`
señala la colección Pareto **de clustering**, no una colección de splits. Las
etiquetas se leen en memoria desde `FLIR_DATA_ROOT/Etiquetas.zip` o mediante
`--labels-archive`; el ZIP nunca se modifica. No hay paths ni IDs reales en configs.

La comparación completa construye o reutiliza historical, cinco random y cinco
asignaciones para cada candidato seleccionado, y después evalúa y selecciona:

```powershell
uv run flir-pipeline splitting compare --inputs reports/clustering/execution/inputs.yaml --comparison artifacts/clustering/comparison/<clustering_comparison_id>
```

También se pueden ejecutar las construcciones separadamente:

```powershell
uv run flir-pipeline splitting baseline --inputs reports/clustering/execution/inputs.yaml --comparison artifacts/clustering/comparison/<clustering_comparison_id> --config configs/splits/random_baseline.yaml
uv run flir-pipeline splitting build --inputs reports/clustering/execution/inputs.yaml --comparison artifacts/clustering/comparison/<clustering_comparison_id> --config configs/splits/cluster_aware_research.yaml
```

`build --clustering-id <id>` limita a un candidato de la selección reproducible.
No se aceptan IDs ajenos al conjunto evaluado. `compare --cluster-config ...
--random-config ...` permite variar configuraciones explícitas; el protocolo
principal exige semillas 0–4 y los mismos targets. Las raíces por defecto son
`artifacts/splitting/runs/<split_space_id>` y
`artifacts/splitting/comparisons/<comparison_id>`; `--output` permite otra raíz.
Un run final válido se reutiliza; un output incompleto o incompatible se preserva
y se rechaza, sin sobrescribirlo ni borrar checkpoints de otras fases.

Cada run almacena assignments por contenido/ocurrencia, fuente de grupos,
estadísticas de registros, balance por grupos/clases, NN cross-split, cohortes
de cuantiles y relaciones temporales; JSON de métricas, QA y metadata con huellas.
Todas las filas reales quedan ignoradas por Git. El histórico tiene una excepción
documentada: `new_split=null` en contenidos con varias pertenencias y un conjunto
explícito; sus registros retienen las memberships originales sin reparación.

```powershell
uv run flir-pipeline splitting verify artifacts/splitting/comparisons/<comparison_id> --inputs reports/clustering/execution/inputs.yaml --comparison artifacts/clustering/comparison/<clustering_comparison_id>
uv run flir-pipeline splitting evaluate artifacts/splitting/runs/<split_space_id> --inputs reports/clustering/execution/inputs.yaml --comparison artifacts/clustering/comparison/<clustering_comparison_id>
uv run flir-pipeline splitting summary artifacts/splitting/comparisons/<comparison_id>
```

`verify` sin fuentes comprueba integridad, identidad, cobertura, indivisibilidad y
balance guardado. Con fuentes comprueba además origen, labels y todas las métricas
residuales de ambos encoders. Para una colección reconstruye tablas, robustez,
fracturas y Pareto. `evaluate` reevalúa contra fuentes y muestra el resultado;
no cambia silenciosamente una publicación inmutable. Una validación no reajusta
modelos ni vuelve a extraer embeddings.

```powershell
uv run python scripts/build_splitting_review.py --comparison artifacts/splitting/comparisons/<comparison_id>
uv run python scripts/check_notebook_source.py
uv run ruff check .
uv run pytest
```

El builder valida la colección, genera nueve figuras y ejecuta el notebook fuente
de 18 secciones. Salidas locales: `reports/splitting/figures/`, `tables/`, y
`review/splitting_review.executed.ipynb`, `review/splitting_review.html` con código
oculto. El notebook versionado permanece limpio. No se requieren imágenes FLIR
para este reporte; las nueve figuras usan estadísticas ya verificadas.

Si el launcher está bloqueado en Windows, utilizar el equivalente
`uv run python -c "from flir_pipeline.cli import app; app()"` y los mismos argumentos.
`UV_PROJECT_ENVIRONMENT` permite reutilizar un entorno existente; `--no-sync`
evita alterarlo. La ejecución registrada reutilizó un entorno uv aislado, SciPy 1.17.1,
y una instalación local ignorada de uv porque uv no estaba en PATH.

## Exportación posterior, no ejecutada en esta fase

```powershell
uv run flir-pipeline splitting export-lists artifacts/splitting/runs/<split_space_id> --manifest <manifest.parquet> --image-root <existing_materialized_images> --output <new_local_list_directory>
```

Exige un split verificado, el mismo manifest, imágenes ya materializadas y paths
distintos por ocurrencia. Rechaza escapes de directorio, archivos ausentes o
colisiones que perderían conflictos de anotación. Solo escribe train/val/test.txt;
no copia, mueve ni extrae imágenes. La materialización y entrenamiento YOLO siguen
fuera de esta ejecución. Antes de publicar revisar `git status`, `git diff` y
`git diff --cached`, excluyendo datos, hashes de contenido y outputs reales.
