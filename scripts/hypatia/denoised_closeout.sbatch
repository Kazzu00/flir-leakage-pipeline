#!/usr/bin/env bash
#SBATCH --job-name=denoised-candidate
set -euo pipefail
ROOT="${FLIR_OE4_REPO:-${SLURM_SUBMIT_DIR:?Submit from the bundle directory}}"
PYTHON="${FLIR_DENOISED_PYTHON:-$HOME/.venvs/flir-detection/bin/python}"
DATASET="${FLIR_DENOISED_DATASET:-$HOME/flir-denoised-20261121/Dataset Sin Split}"
WORK="${FLIR_DENOISED_WORK:-$HOME/flir-denoised-20261121}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
cd "$ROOT"
"$PYTHON" scripts/closeout/verify_bundle.py "$ROOT"
"$PYTHON" scripts/closeout/denoised_job.py "$DATASET" "$WORK" "$ROOT/configs/closeout/denoised_20261120.yaml"
