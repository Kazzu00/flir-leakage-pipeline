#!/usr/bin/env bash
#SBATCH --job-name=flir-base-preliminary
#SBATCH --output=flir-base-preliminary-%j.log
#SBATCH --signal=TERM@120
set -euo pipefail
: "${FLIR_REPO:?Checkout required}"
: "${FLIR_PYTHON:?Activated detection Python required}"
: "${FLIR_PRELIMINARY_PACKAGE:?Verified preliminary package required}"
: "${FLIR_PRELIMINARY_RECEIPT_SHA256:?Expected receipt SHA256 required}"
: "${FLIR_WEIGHTS:?Local pretrained yolo11m.pt required}"
: "${FLIR_PRELIMINARY_OUTPUT:?Fresh external output root required}"
cd "$FLIR_REPO"
export PYTHONPATH="$FLIR_REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled
srun --ntasks=1 "$FLIR_PYTHON" -c 'from flir_pipeline.cli import app; app()' \
  detection preliminary-base train "$FLIR_PRELIMINARY_PACKAGE" "$FLIR_WEIGHTS" \
  "$FLIR_PRELIMINARY_OUTPUT/job-${SLURM_JOB_ID}" \
  --receipt-sha256 "$FLIR_PRELIMINARY_RECEIPT_SHA256"
