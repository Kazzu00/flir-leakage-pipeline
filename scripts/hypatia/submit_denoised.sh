#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${FLIR_OE4_PYTHON:-$HOME/.venvs/flir-detection/bin/python}"
CONFIG="${FLIR_OE4_CONFIG:-$ROOT/configs/closeout/oe4_20261115.yaml}"
DATA="$HOME/flir-denoised-exploratory-20261122-data"
WORK="$HOME/flir-denoised-exploratory-20261122"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
"$PYTHON" scripts/closeout/verify_bundle.py "$ROOT"
"$PYTHON" -m flir_pipeline.closeout denoised-prepare-training "$CONFIG" "$DATA" "$WORK"
mkdir -p "$WORK/logs"
sbatch --parsable --partition="${FLIR_GPU_PARTITION:-gpu}" --gres=gpu:1 \
  --cpus-per-task=4 --mem=32G --time="${FLIR_OE4_WALLTIME:-12:00:00}" \
  --array=0-1%1 --job-name=denoised-explore \
  --output="$WORK/logs/train-%A_%a.out" --error="$WORK/logs/train-%A_%a.err" \
  "$ROOT/scripts/hypatia/denoised_train.sbatch" "$ROOT" "$CONFIG" "$PYTHON" "$DATA" "$WORK"
