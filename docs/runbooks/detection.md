# Ejecución reproducible de la comparación del detector

Leer primero el [protocolo del detector](../protocols/detection.md). Datos originales fuera del repo,
`FLIR_DATA_ROOT` definido localmente y ZIPs read-only. Nunca publicar artifacts,
pesos, imágenes, labels, hashes reales, manifest, outputs o `.env`.

## Entorno

Python 3.11; el extra opcional evita instalar YOLO/GPU en core/CI:

```powershell
uv sync --locked --extra dev --extra reporting --extra detection
uv run flir-pipeline detection environment
```

Comprobar el build de PyTorch: instalar un wheel compatible con el hardware
real. `torch.cuda.is_available()` es la comprobación operativa; una GPU presente
en otro equipo no habilita CUDA en el portátil. Para CPU, una instalación
explícita permite evitar wheels CUDA innecesarios:

```powershell
uv pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
uv run --no-sync flir-pipeline detection environment
```

Si se reutiliza un entorno aislado, configurar `UV_PROJECT_ENVIRONMENT` con su
ubicación y usar `uv run --no-sync` después de comprobar sus dependencias.
El lock fija dependencias; el freeze registra los wheels concretos, incluido
el sufijo CPU/CUDA. Cambiar de runtime requiere un experimento separado.

## Auditoría, plan y vistas

```powershell
uv run flir-pipeline detection plan artifacts/splitting/comparisons/a19ba503f085f569
uv run flir-pipeline detection materialize
uv run flir-pipeline detection verify
```

El plan valida la colección y registra 16 splits, la auditoría de 30 filas y
las 48 celdas de la matriz. Materialize valida cada fuente, sus hashes y todas
las ocurrencias, preserva labels vacíos/conflictivos y genera YAML/listas.
Las 16 vistas reutilizan un solo almacén por dataset bajo `artifacts/detection`.
No se extraen huérfanos ni archivos ajenos al manifest.

Se puede reconstruir en otra máquina a partir de los ZIPs, manifest y splits
locales validados, sin copiar `.env`. Las vistas contienen rutas operacionales
locales: reconstruirlas en un output nuevo después de mover el proyecto.
No se cambian asignaciones para hacer coincidir rutas o mejorar métricas.

## Capacidad y piloto pequeño

Referencias: **C10 — DINOv2 / PaCMAP / DBSCAN**; **C12 — DINOv2 / t-SNE / HDBSCAN**.

Antes de cualquier entrenamiento real:

```powershell
uv run flir-pipeline detection probe
uv run flir-pipeline detection freeze
uv run flir-pipeline detection pilot-small
```

Probe descarga solo YOLO11n del asset oficial y registra su hash. Prueba
forward/backward/optimizer con datos sintéticos a resolución final, selecciona
un batch ensayado y registra RAM/VRAM. Freeze liga configuración, device,
batch, runtime y pesos antes del piloto real.

El piloto pequeño hace 24/12/20 imágenes, dos epochs y seed 42 en historical,
random seed 0, C10 seed 0 y C12 seed 0. Los subconjuntos se seleccionan dentro
de sus particiones para cubrir clases presentes, sin alterar el split fuente.
El piloto nunca se interpreta ni se agrega como resultado científico final.

Leer `artifacts/detection/compute_budget.json`: tiempos medidos, extrapolación
por entrenamiento y coste de 48 runs. El coste del bootstrap final no está
incluido. **CPU no inicia Stage B automáticamente**; el comando `run` rechaza
esa transición. Un futuro presupuesto reducido requiere preespecificar la
misma matriz/condiciones para todas las estrategias y mantener ≥2 detector seeds.

Smoke adicional puramente sintético, opcional, fuera de pytest/CI:

```powershell
uv run flir-pipeline detection smoke --cpu --output artifacts/detection/synthetic_cpu_smoke
```

Ese smoke usa una epoch a resolución pequeña; no sustituye el probe ni el piloto.

## Stage A y Stage B en un runtime con capacidad suficiente

En CUDA validada y con un freeze separado compatible:

```powershell
uv run flir-pipeline detection run
```

El comando completa primero las cuatro celdas Stage A (seed 0/42, 50 epochs),
valida todas y después recorre Stage B. Cada run nuevo parte de los mismos
pretrained weights. Los cuatro pilotos completos idénticos cuentan una sola
vez dentro de las 48 celdas. `pilot-small` jamás se incorpora a esa matriz.

Resume: repetir `run`. Un run completo se verifica y reutiliza; uno interrumpido
reanuda únicamente su propio `last.pt` si la identidad coincide. Sin checkpoint
reanudable se conserva la carpeta parcial para inspección. No se reutilizan
checkpoints entre estrategias ni se borran `.partial` automáticamente.

Un cambio de protocolo/runtime debe ir a directorios nuevos mediante `--plan`
y `--output`. No sobrescribir freezes previos. Ante OOM real, detener, registrar
fallo y revisar batch común antes del experimento final; nunca reducirlo solo
para una estrategia. El guard impide continuar si Ultralytics cambia batch.

## Artifacts, informe y verificación

Por run final: best.pt, historial/args, `metrics.json`, `metrics_per_class.parquet`,
`training_summary.json`, `metadata.json`, estadísticas de test y bootstrap.
El recibo enlaza dataset, split, pesos, versiones y hashes del código. P/R usan
umbral fijo; los valores best-F1 de consola no son el endpoint preespecificado.

```powershell
uv run python scripts/build_detector_review.py
uv run python scripts/check_notebook_source.py
uv run ruff check .
uv run pytest
git status --short
git diff --check
```

El notebook fuente está limpio. El HTML y notebook ejecutado viven en
`reports/detection/review/`; las nueve figuras, en `reports/detection/figures/`.
Mientras no exista una matriz final completa y controlada, ocho paneles de
métricas muestran **pendiente**, sin ceros ficticios ni métricas del piloto.
La figura de correlación residual usa resultados reales anteriores al detector.

## Detector y similitud residual

```powershell
uv run --extra explorer streamlit run apps/detector_similarity_explorer.py
uv run --extra reporting python scripts/build_detector_review.py
```

La [aplicación separada](../visualization/detector_similarity.md) lee evidencia
local cada 30 segundos. Si el puerto del explorador de clusters está ocupado,
añadir `--server.port 8502`. Sin plan válido muestra evidencia no disponible;
con la matriz incompleta muestra PENDING/PARTIAL y nunca métricas de pilotos.

El builder conserva el nombre de figura 09 y los reportes previos. Genera:

- `tables/experiment_matrix.csv`: las 48 celdas y su estado/identidad.
- `tables/split_context.csv`: los 16 contextos, con fracciones opcionales verificadas.
- `tables/prespecified_associations.csv`: siete especificaciones, estado y número
  de runs con ambos valores definidos; cero no significa performance igual a cero.
- Solo en COMPLETE: `tables/detector_residual_associations.csv`, una fila por
  run × clase/overall, y `tables/strategy_summary.csv`, además de los agregados previos.

Si una reconstrucción queda incompleta, se retiran únicamente los CSV científicos
generados por el reporte anterior para impedir su lectura como resultados actuales.
Los artefactos originales de runs/splits nunca se modifican. Revisar INVALID antes
de continuar: splits/contextos discordantes, duplicados, checksums y diferencias
de protocolo bloquean la tabla científica. RUNNING refleja un estado guardado,
no una conexión ni una comprobación de actividad de la máquina de entrenamiento.
