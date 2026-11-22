# Recuperación OE4 y denoised — ejecución

## Causa raíz y comprobación

`detection/scientific_runtime.py` llama al `write_view` histórico y luego lee
`development/records.parquet` para obtener `num_objects`. `write_view` cuenta las
etiquetas YOLO verificadas y enriquece **una copia** del DataFrame. El manifiesto
canónico `candidate_manifest.parquet` no tiene esa columna; `development_records`
conserva su esquema y añade los locadores/membresía. El wrapper OE4 consultaba
equivocadamente el DataFrame de entrada después de `write_view`. Por eso pasó el
preflight de integridad, pero falló antes de la primera época. Los tests anteriores
terminaban antes de preparar la vista y no cubrían esa línea.

`closeout/training.py` reutiliza ese mismo `write_view`, lee su salida enriquecida
y mantiene el control `check_loader_annotations` histórico. No usa un conteo
supuesto ni corrige etiquetas. Se comprueban todas las columnas consumidas,
ocurrencias únicas (sin colapsar contenidos repetidos), imágenes, SHA de labels,
configuración completa y loader real de Ultralytics 8.3.203. La preparación CPU
recorre las tres vistas reales antes de reservar GPU y se repite en el worker.
Entrenamiento y exploración denoised comparten una sola llamada YOLO11m.

Replay real local: publicación existente historical-S1-0, 1.144 registros y
3.032 objetos, esquema exacto sin `num_objects`, loader real y llamada YOLO.train
interceptada antes de entrenar. No son los manifiestos remotos de las tres
corridas. Sus argumentos/miembros reales se inspeccionan en el preflight `short`.
La autenticación SSH disponible fue rechazada; no se leyeron los parciales 808329
ni se enviaron trabajos remotamente desde esta sesión.

## Relanzamiento seguro

Continúa `oe4_closeout_v2` y la autorización comunicada ya registrada. No se
introducen firmas ni otro mecanismo de admisión. Freeze, preparación, salidas y
logs nuevos: `$HOME/flir-oe4-recovery-20261122/`. Los parciales del array 808329 y
los tres AUG_ON originales quedan preservados. No se reanuda desde checkpoints
parciales ni se sobrescriben resultados válidos.

Se mantienen datasets de desarrollo por política, pesos iniciales originales,
seed 42, 50 épocas y controles efectivos. Solo cambia aumentación. Índices 0/1/2
son histórica/aleatoria/C10. El test común de 372 permanece como fase posterior;
evaluar exige los tres recibos AUG_OFF completados del freeze nuevo.

## Frente denoised

Se reutiliza el candidato 777/172/167 y sus 93 grupos: no se recalculan embeddings,
clustering ni asignaciones. Se excluyen íntegramente 14 grupos con candidatos
nominales al test protegido o a sus grupos, además de comprobar igualdad exacta
contra contenido protegido. Se retienen 635 registros: 466 train, 47 val y 122
test interno. Solo los 513 train/val se copian al paquete; el test interno queda
como catálogo sin imágenes ni entrada de entrenamiento/evaluación.

Los 342 registros no resueltos restantes impiden afirmar elegibilidad para el
test común. Estas dos corridas son **exploratorias aisladas**, AUG_ON y AUG_OFF,
con misma población denoised y receta base del AUG_ON C10 (no su partición ni su
nombre metodológico). Los recibos tienen otro protocolo y **prohíben evaluar en
el test común**. No son una comparación confirmatoria del denoising, porque
falta el emparejamiento normal/denoised verificado. No se promueven nombres a
anclas. La alternativa que también excluye todos los grupos no resueltos deja
235 train / 1 val sin objetos / 48 test: no admite una evaluación útil sin cambiar
la partición. Se reporta esa limitación, sin ajustar el split para obtener métricas.

Train denoised tiene objetos 335/1449/482/117/165; val 24/14/16/28/35. Las cinco
clases están presentes. Los IDs y bytes no se recodifican; clase 4 usa el alias
maquinaria proporcionado con el dataset. Ambos brazos pasaron preparación real
CPU y alcanzaron la invocación interceptada sin entrenamiento ni inferencia.
Índices denoised 0/1: AUG_ON/AUG_OFF, con recibos y logs separados de OE4.

## Transferir desde PowerShell

```powershell
$Release = Join-Path ([Environment]::GetFolderPath('Desktop')) 'flir-oe4-recovery-20261122'
$Target = '<usuario>@hypatia.uniandes.edu.co'
scp -- "$Release/flir-oe4-recovery-20261122-code.tar.gz" "$Release/flir-oe4-recovery-20261122-code.tar.gz.sha256" "$Release/flir-denoised-exploratory-20261122-data.tar.gz" "$Release/flir-denoised-exploratory-20261122-data.tar.gz.sha256" "${Target}:"
ssh $Target
```

## Preflight CPU en short y enviar ambos arrays

```bash
set -euo pipefail
cd "$HOME"
sha256sum -c flir-oe4-recovery-20261122-code.tar.gz.sha256
sha256sum -c flir-denoised-exploratory-20261122-data.tar.gz.sha256
test ! -e flir-oe4-recovery-20261122-code
test ! -e flir-denoised-exploratory-20261122-data
tar --keep-old-files -xzf flir-oe4-recovery-20261122-code.tar.gz
tar --keep-old-files -xzf flir-denoised-exploratory-20261122-data.tar.gz
cd "$HOME/flir-oe4-recovery-20261122-code"
export FLIR_OE4_CONFIG="$PWD/configs/closeout/oe4_20261115.yaml"
mkdir -p "$HOME/flir-oe4-recovery-20261122/logs"
CPU_JOB=$(sbatch --wait --parsable --output="$HOME/flir-oe4-recovery-20261122/logs/prepare-%j.out" --error="$HOME/flir-oe4-recovery-20261122/logs/prepare-%j.err" scripts/hypatia/preflight_oe4.sbatch "$PWD")
printf 'CPU=%s\n' "$CPU_JOB"
AUG_OFF_JOB=$(bash scripts/hypatia/submit_oe4.sh train | tee "$HOME/flir-oe4-recovery-20261122/submission-aug-off.log" | tail -n 1)
DENOISED_JOB=$(bash scripts/hypatia/submit_denoised.sh | tee "$HOME/flir-oe4-recovery-20261122/submission-denoised.log" | tail -n 1)
printf 'AUG_OFF=%s\nDENOISED=%s\n' "$AUG_OFF_JOB" "$DENOISED_JOB"
```

`sbatch --wait` devuelve fallo si el preflight falla; con `set -e` no se envían
los arrays en ese caso. El código del worker usa el directorio explícito recibido,
no la ubicación del script en el spool. Cada array limita concurrencia a una GPU.
Solo se carga el entorno detector existente `~/.venvs/flir-detection`.

## Confirmar primera época

```bash
squeue -u "$USER"
sacct -j "$CPU_JOB,$AUG_OFF_JOB,$DENOISED_JOB" --format=JobID,State,ExitCode,Elapsed
grep -h 'FLIR_CPU_READY_BOTH' "$HOME/flir-oe4-recovery-20261122/logs/"prepare-*.out
grep -hE 'FLIR_TRAIN_INVOKE|FLIR_EPOCH_STARTED' "$HOME/flir-oe4-recovery-20261122/logs/"train-*.out "$HOME/flir-denoised-exploratory-20261122/logs/"train-*.out
find "$HOME/flir-oe4-recovery-20261122/aug-off" "$HOME/flir-denoised-exploratory-20261122/runs" -name progress.json -exec cat {} \;
```

`FLIR_TRAIN_INVOKE` acredita solo invocación. `FLIR_EPOCH_STARTED ... epoch=1` y
`progress.json` se emiten desde el callback real de inicio de época, después de
verificar argumentos y loaders. Slurm RUNNING por sí solo no acredita primera
época. PENDING implica esperar recursos. Un error antes del callback conserva
diagnóstico y no se presenta como entrenamiento comenzado/completado.

Tras los tres AUG_OFF completados, ejecutar desde este bundle:

```bash
bash scripts/hypatia/submit_oe4.sh evaluate
# Solo después de las seis evaluaciones completas:
bash scripts/hypatia/aggregate_oe4.sh
```

No hay comando de evaluación del test común para denoised exploratorio.
