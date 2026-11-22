# Tres datasets de la entrega final — 22 de noviembre de 2026

La entrega vigente es `00_ENTREGABLES/FLIR_Entrega_Final_20261122_v2`, en el
almacenamiento externo del responsable. Localización portable en PowerShell:

```powershell
$delivery = Join-Path $env:USERPROFILE 'Desktop/tesis-data/00_ENTREGABLES/FLIR_Entrega_Final_20261122_v2'
```

Esta ruta documenta la entrega existente; el pipeline sigue leyendo sus fuentes
mediante `FLIR_DATA_ROOT`. No hay enlace de Drive verificado. Los ZIP permanecen
fuera de Git y no se regeneraron ni reorganizaron en este cierre.

| ZIP | Train | Val | Test | Estado / uso |
|---|---:|---:|---:|---|
| `FLIR_Historico_Splits.zip` | 1178 | 107 | 372 | Baseline histórico con leakage exacto conocido |
| `FLIR_Sin_HUD_Splits.zip` | 19914 | 839 | 163 | Candidato visual por vídeos completos; no listo para YOLO |
| `FLIR_Sin_Ruido_Splits.zip` | 777 | 172 | 167 | Candidato agrupado, no congelado científicamente |

Los conteos se inspeccionaron en `ENTREGA.md`, `_verification/delivery_counts.json`
y `_verification/final_verification.json` existentes. Se publica únicamente el
[recibo agregado de empaquetado](delivery/final_datasets_20261122_verification.json),
sin identidades de frames, hashes de imágenes, rutas privadas ni anotaciones.
Su `verified=true` documenta la verificación previa del empaquetado; este cierre
no repite la lectura de todos los payloads ZIP ni certifica aprobación científica.

**Histórico:** 1.657 ocurrencias, 1.459 contenidos; preserva 198 grupos exactos
cross-split y ocho conflictos de anotación según la auditoría de entrega. Sus
372 ocurrencias test corresponden al test común, pero su train/val mantiene
contaminación conocida. Los 292 labels vacíos no se rellenan. Los diez labels
huérfanos permanecen fuera de los splits, en su archivo original.

**Sin HUD:** 20.916 imágenes. La política explícita asigna video1 completo a train
(19.800 PNG más 114 imágenes del TAR), video2 a val (839) y video3 a test (163).
Estado `CANDIDATE_WHOLE_VIDEO_VISUAL_NOT_FROZEN`. Solo hay 114/839/163 archivos
label por split, incluidos vacíos recibidos: faltan 19.800 etiquetas train.
Los 1.116 labels existentes requieren revisión geométrica/de objetos procesados.
No convertir imágenes sin label en negativos ni transferir labels históricos
por nombre/similitud. La indivisibilidad por vídeo declarado no prueba ausencia
de similitud entre vídeos ni aislamiento del histórico.

**Sin Ruido:** 1.116 pares imagen/label. Reutiliza las asignaciones existentes
777/172/167, 93 grupos y cero cruces de las 1.170 relaciones registradas, según
la auditoría previa. Estado `CANDIDATE_NOT_FROZEN`: factibilidad del MILP observada,
óptimo no demostrado, sin adjudicación exhaustiva ni freeze científico. Su test
167 es interno y **no equivale al común protegido de 372**. El subconjunto
exploratorio 466/47 y las asignaciones del proveedor 803/151/162 no son esta entrega.

Las transformaciones Sin HUD/Sin Ruido son aportes externos/grupales; su ingesta,
trazabilidad y partición se documentan aquí sin atribuir su algoritmo al componente
individual. No se acredita una integración panóptica final por publicar estos ZIP.

Los tres datasets no son directamente comparables: tienen poblaciones,
anotaciones y tests distintos. Hay 34 contenidos exactos compartidos entre las
dos variantes y 17 con asignación diferente. No mezclar splits entre variantes
ni interpretar sus métricas como comparación pareada o efecto causal del filtrado.
Los [resultados OE4](results/oe4_20261122/README.md) usan únicamente su test común.

Los archivos `memberships.csv`, inventarios, checksums por imagen, conflictos,
correspondencias, etiquetas y demás evidencia detallada se conservan en la entrega
externa y sus ZIP. No se publican como manifiestos distribuibles sin establecer
su permiso de distribución; los esquemas/contratos y productores reproducibles
existentes sí permanecen en el repositorio público.
