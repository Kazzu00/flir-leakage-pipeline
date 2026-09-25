# Project status

Documentación reorganizada el **2026-09-23** desde el código y la evidencia local
existente, sin ejecutar experimentos ni cambiar resultados. **Validated** indica
comprobación dentro del protocolo, no validación semántica exhaustiva ni mejora
del detector. **Available** indica infraestructura utilizable; **Experimental**,
incertidumbre interpretativa; **Pending compute**, una ejecución completa pendiente;
**Planned**, trabajo adicional condicionado.

| Componente | Estado | Evidencia y límites |
|---|---|---|
| Auditoría y manifest | Validated | 1657 frame_id, 1459 content_id, 198 grupos duplicados; ocho conflictos de anotación preservados |
| Muestreo de videos fuente | Operationally validated on real data | Smoke real de 5 s y job Hypatia 737719 COMPLETED, ExitCode 0:0; 3 videos fuente → 9648 JPEGs a 1 FPS, Parquet/grilla validados y fuentes read-only; no identifica secuencias ni define splits |
| Puente de frames muestreados a features | Available; synthetic tests only | `data build-video-manifest` y `features extract --images-root`: ocurrencias completas, identidad/deduplicación por bytes, raw/L2 y resume compartidos; aún sin ejecutar en Hypatia sobre las 9648 muestras, sin secuencias ni splits para estos videos |
| Caracterización y diagnostics | Validated | 4168 instancias canónicas, 292 labels vacíos, geometría por clase y 1459 diagnostics; huérfanos separados |
| DINOv2 | Validated | Full 1459 × 384 CLS; raw/L2, revisión resuelta y 1657 mappings |
| CLIP | Validated | Full 1459 × 512 projected-image; raw/L2, revisión resuelta y 1657 mappings |
| Coseno | Validated | Por encoder: 1459 × 1459, 1063611 pares únicos y 29180 vecinos dirigidos top-20 |
| Interpretación temporal | Experimental | Secuencia/índice inferidos por nombre; cero timestamps verificados |
| t-SNE / PaCMAP | Validated with limitations | 18 + 18 runs, preservación y tres semillas; cuatro referencias exploratorias |
| DBSCAN / OPTICS / HDBSCAN | Validated with limitations | 132 / 186 / 96 runs; 204 ARI/AMI, shortlist de 54, 41 Pareto; escenas sin validación exhaustiva |
| Splitting y baselines | Validated with limitations | 66 runs; 65 nuevos sin overlap exacto; 60 cluster-aware sin fracturas; seis Pareto |
| Streamlit / VIKUS | Available | Inspección local; no valida escenas automáticamente |
| Detector YOLO11n | Available; small pilots validated | 16 vistas y cuatro pilotos CPU; no comparación científica final |
| Detector ↔ residual similarity | Available; PENDING | Registro de siete asociaciones, figura 09 y explorador local; 0/48 runs controlados, pilotos excluidos |
| Comparación completa del detector | Pending compute | Stage A/B y 48 runs objetivo aún no ejecutados |
| Bhattacharyya | Planned / conditional | Requiere una representación distribucional explícita |

## Data and representations

El histórico conserva 1178/107/372 registros train/val/test y 57/141/0 contenidos
compartidos train–val/train–test/val–test. El archivo de etiquetas contiene 4182
objetos: 4168 del candidato y 14 de diez etiquetas huérfanas. La referencia
bibliográfica exacta de la nomenclatura sigue pendiente; el orden fue confirmado
por el responsable del proyecto y contrastado con el YAML original.

Las extracciones completas ya existían desde 2026-09-09. Los smoke N=16 se
conservan separados y no caracterizan globalmente el espacio de embeddings.
El inventario registrado conserva cinco ZIP disponibles de seis históricos;
falta `video_13min_778.zip`, sin pérdida de cobertura del candidato canónico.
Véanse [auditoría](analysis/data_quality.md), [clases](analysis/dataset_classes.md)
y [representaciones](analysis/features.md).

## Source-video sampling: real execution confirmed

La evidencia confirmada por el responsable del proyecto registra el job SLURM
**737719** en **Hypatia, Universidad de los Andes**: `COMPLETED`, `ExitCode=0:0`,
`Elapsed=00:15:28`, `MaxRSS` del batch `175016K` y stderr vacío. Se usaron Python
3.11.10, FFmpeg/ffprobe 9.0.2, `sample_fps=1.0` y `jpeg_quality=2`.

Los tres videos produjeron **2770 + 660 + 6218 = 9648 JPEGs**. Se validaron 9648
filas y 9648 `image_path` únicos en `frames.parquet`, índices contiguos desde 0
por video, tiempos coherentes con `sample_index / 1 FPS` y fuentes read-only.
También se validó previamente un smoke real: 5 segundos a 30 FPS, 150 frames
fuente → 5 JPEGs a 1 FPS. El [runbook](runbooks/data.md#evidencia-real-confirmada-en-hypatia)
conserva configuración y resultados agregados; los recibos exactos permanecen locales.

Esta evidencia valida operacionalmente la extracción reproducible de frames.
Los tres videos son **fuentes, no tres secuencias**. No valida límites de escenas,
embeddings de estas muestras, clustering, train/val/test, eliminación de leakage
ni rendimiento del detector. `build-video-manifest` conserva únicamente pruebas
sintéticas y todavía no se ha ejecutado en Hypatia; tampoco se ha establecido
cuántos contenidos exactos únicos hay entre las 9648 muestras. Los resultados
históricos de las demás etapas no cambian.

## Candidate partitions

**C10 — DINOv2 / PaCMAP / DBSCAN** es la referencia visual principal:
seed 0, split `88ccf4e12335a83f`, 1178/107/372 registros, cinco clases por split,
cero duplicados exactos cross-split y 6/7 pares top-0.1% en DINOv2/CLIP.
El histórico tiene 328/277 y las medias random 478/466.8. Su fracción temporal
inferida Δ≤5 es 23.93%, frente a 14.76% histórica y 43.92% random.

**C12 — DINOv2 / t-SNE / HDBSCAN** es el candidato secundario del protocolo del
detector, con menor similitud NN media y temporalidad residual, a costa del
balance. **C01 — CLIP / original L2 / OPTICS** permanece como referencia
descriptiva de balance, con 97.4% noise. Las anclas de splitting (C10/C01) y la
selección previa al detector (C10/C12) corresponden a decisiones distintas;
ninguna constituye un ganador universal.

Las 65 asignaciones nuevas fueron reconstruidas exactamente en la validación
registrada. El MILP balancea clases y random no: ese contraste limita atribuir
un cambio posterior exclusivamente al agrupamiento. Resultados completos en
[clustering](analysis/clustering.md), [splitting](analysis/splitting.md) y
[selección del detector](analysis/detection.md).

## Detector execution boundary

Historical, random_content, **C10 — DINOv2 / PaCMAP / DBSCAN** y
**C12 — DINOv2 / t-SNE / HDBSCAN** se seleccionaron antes de YOLO.
Plan `ace1ffd6bb4775d0`: 16 splits, 48 celdas objetivo, detector seeds 42/43/44
y split seeds 0–4. Runtime CPU congelado: `9897062efdb32450`.

Los cuatro pilotos usan 24/12/20 imágenes, dos epochs, batch 2, imgsz 640 y CPU.
Se revalidaron métricas y 1000 bootstrap por piloto. Training total 140.39 s;
evaluación/bootstrap 57.46 s. La extrapolación registrada de 408–593 horas para
48 runs usa dos métodos, no un intervalo de confianza. El hardware registrado
no expone CUDA; el protocolo largo se detuvo por presupuesto.

Quedan pendientes Stage A completo a 50 epochs, Stage B, comparación multi-seed,
variabilidad e interpretación científica. No hay ejecución GPU ni prueba real de
resume interrumpido en GPU. El HTML conserva una figura de contexto y ocho paneles
pendientes. Véanse [análisis](analysis/detection.md), [protocolo](protocols/detection.md)
y [runbook](runbooks/detection.md).

## Reports and next validation

Los siete notebooks fuente siguen activos, sin outputs versionados. El
[reporte consolidado](runbooks/project_report.md) tiene 22 secciones, 14 figuras
(12 reutilizadas y dos resúmenes temporales) y un diagrama. Los recibos previos
respaldan las validaciones numéricas; una interfaz disponible no cambia el estado
experimental.

El siguiente trabajo metodológico es revisar escenas y conflictos de candidatos,
mejorar la procedencia temporal y ejecutar el protocolo controlado del detector
con presupuesto viable. [Streamlit](visualization/streamlit.md) y
[VIKUS](visualization/vikus.md) apoyan esa revisión. La coherencia visual global,
la eliminación de toda dependencia y la mejora en Precision/Recall/mAP siguen
sin demostrarse.
