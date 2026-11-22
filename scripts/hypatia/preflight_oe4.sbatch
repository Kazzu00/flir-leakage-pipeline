#!/usr/bin/env bash
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=flir-cpu-prepare
set -euo pipefail
ROOT="${1:?Pass the absolute bundle directory}"
PYTHON="${FLIR_OE4_PYTHON:-$HOME/.venvs/flir-detection/bin/python}"
export OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
bash scripts/hypatia/preflight_oe4.sh
"$PYTHON" -m flir_pipeline.closeout denoised-prepare-training \
  "${FLIR_OE4_CONFIG:-$ROOT/configs/closeout/oe4_20261115.yaml}" \
  "$HOME/flir-denoised-exploratory-20261122-data" \
  "$HOME/flir-denoised-exploratory-20261122"
echo 'FLIR_CPU_READY_BOTH'
