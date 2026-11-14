# BASE preliminar en Hypatia

Estado 2026-11-04: paquete local validado, entrenamiento todavía no ejecutado.
El usuario aceptó las 950 anotaciones originales no vacías para este piloto,
con revisión visual pendiente. S3 se conserva: 883 train / 67 val. Se excluyen
185 vacías (169 / 16); no se consideran errores automáticamente. Se mantienen
todos los objetos anotados, pero cambia la prevalencia de fondos: estas métricas
no serán directamente comparables con las del conjunto completo.

Cero duplicados exactos entre train/val; dependencia temporal y casi duplicados
no descartados. Exclusión del test vinculada al recibo de preparación previo.
No se abre el test. No es admisión científica final ni ablación emparejada.

## Transferencia desde PowerShell

Definir `DELIVERY` como la carpeta externa entregada y usar una ruta remota nueva,
sin espacios. Sustituir usuario, host y ruta por los de la sesión autorizada.

```powershell
$Delivery = $env:DELIVERY
$SshTarget = 'USUARIO@HOST_HYPATIA'
$Remote = '/RUTA_AUTORIZADA/flir-preliminary'
ssh $SshTarget "mkdir -p '$Remote'"
$Names = @('base_preliminary.tar', 'flir-preliminary.bundle', 'yolo11m.pt', 'weights-provenance.json', 'SHA256SUMS')
foreach ($Name in $Names) { scp (Join-Path $Delivery $Name) "${SshTarget}:${Remote}/" }
```

## Preparación en SSH

Activar primero el entorno de detección existente: Python 3.11, torch 2.7.1,
torchvision 0.22.1, ultralytics 8.3.203. No reinstalar a ciegas ni sustituir versiones.

```bash
set -euo pipefail
export ROOT='/RUTA_AUTORIZADA/flir-preliminary'
cd "$ROOT"
sha256sum -c SHA256SUMS
test ! -e base_preliminary
tar -xf base_preliminary.tar
git clone --branch chore/integrate-hypatia-base flir-preliminary.bundle code
export FLIR_REPO="$ROOT/code"
export FLIR_PYTHON="$(command -v python)"
export PYTHONPATH="$FLIR_REPO/src"
export FLIR_PRELIMINARY_PACKAGE="$ROOT/base_preliminary"
export FLIR_PRELIMINARY_RECEIPT_SHA256="$(sha256sum "$FLIR_PRELIMINARY_PACKAGE/receipt.json" | cut -d' ' -f1)"
export FLIR_WEIGHTS="$ROOT/yolo11m.pt"
export FLIR_PRELIMINARY_OUTPUT="$ROOT/preliminary-runs"
cd "$FLIR_REPO"
"$FLIR_PYTHON" -c 'from flir_pipeline.cli import app; app()' detection preliminary-base verify "$FLIR_PRELIMINARY_PACKAGE" --receipt-sha256 "$FLIR_PRELIMINARY_RECEIPT_SHA256"
```

## Preflight GPU y job real

Definir recursos según la partición/cuenta autorizada de Hypatia. El ejemplo pide
una GPU; añadir `--account` o `--qos` si la cuenta lo exige. El preflight comprueba
ambiente/CUDA, no garantiza memoria o duración del entrenamiento completo.

```bash
export GPU_PARTITION='PARTICION_GPU_AUTORIZADA'
export CPUS=4
export RAM='16G'
export WALLTIME='04:00:00'
srun --partition="$GPU_PARTITION" --gres=gpu:1 --ntasks=1 --cpus-per-task="$CPUS" --mem="$RAM" --time=00:10:00 "$FLIR_PYTHON" -c 'from flir_pipeline.cli import app; app()' detection common-executor preflight "$FLIR_PRELIMINARY_OUTPUT/preflight-$(date +%Y%m%d-%H%M%S).json" --config "$FLIR_PRELIMINARY_PACKAGE/environment-protocol.yaml"
JOB_ID=$(sbatch --parsable --export=ALL --partition="$GPU_PARTITION" --gres=gpu:1 --ntasks=1 --cpus-per-task="$CPUS" --mem="$RAM" --time="$WALLTIME" scripts/hypatia/preliminary_base.sbatch)
JOB_ID="${JOB_ID%%;*}"
squeue -j "$JOB_ID"
tail -f "flir-base-preliminary-${JOB_ID}.log"
# Después del job:
sacct -j "$JOB_ID" --format=JobID,State,Elapsed,MaxRSS,ExitCode
cat "$FLIR_PRELIMINARY_OUTPUT/job-$JOB_ID/execution.json"
```

YOLO11m preentrenado local, 50 epochs, imgsz 640, batch 1, nbs 64, seed 42;
configuración completa congelada en `experiment.json`. Cada job usa una copia
de trabajo para cachés y un directorio nuevo. `completed` exige historial completo
y checkpoints best/last. Conservar fallos; no hay resume en esta ruta preliminar.

## NO-HUD: resultado y condición pendiente

Informe estático externo único: `review.html`, con 185 vacías y 13 pares piloto.
Triaje visual asistido: 67 potencialmente incompletas, 23 posibles negativos,
95 indeterminables; ninguna adjudicación humana ni negativo confirmado.

El índice existente ofrece candidatos para 584 de 950 fuentes; los otros 366
no tienen candidato en ese índice limitado, no prueba de inexistencia de match.
El piloto cubre todas las clases y ambos splits: 7 pares / 11 cajas candidatas,
6 rechazos geométricos. Cero transferencias verificadas y cero población emparejada.
Las propuestas geométricas supervivientes provienen de Adendo; no representan
cobertura verificada de las cinco clases. La ambigüedad temporal sigue pendiente.

No existe aún un comando válido para enviar la ablación emparejada. Primero se
requiere certificar correspondencia temporal, objetos y cajas por par; después
materializar ambos datasets con iguales observaciones/asignaciones y configuración.
Los hashes de rasters distintos se conservan distintos, unidos por identidad de
observación. Alineación aceptada no equivale a transferencia de anotaciones.
Los controles del ejecutor científico final permanecen intactos.
