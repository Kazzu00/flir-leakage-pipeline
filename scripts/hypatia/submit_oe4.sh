#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PHASE="${1:?Usage: submit_oe4.sh train|evaluate}"
case "$PHASE" in train|evaluate) ;; *) echo "Unknown phase" >&2; exit 2 ;; esac
PYTHON="${FLIR_OE4_PYTHON:-$HOME/.venvs/flir-detection/bin/python}"
CONFIG="${FLIR_OE4_CONFIG:-$ROOT/configs/closeout/oe4_20261115.yaml}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
"$PYTHON" scripts/closeout/verify_bundle.py "$ROOT"
"$PYTHON" -m flir_pipeline.closeout oe4-authorize "$CONFIG" "$PHASE"
if [[ "$PHASE" == train ]]; then
  "$PYTHON" -m flir_pipeline.closeout oe4-prepare "$CONFIG"
fi
LOGS="$HOME/flir-oe4-recovery-20261122/logs"
mkdir -p "$LOGS"
sbatch --parsable --partition="${FLIR_GPU_PARTITION:-gpu}" --gres=gpu:1 \
  --cpus-per-task=4 --mem=32G --time="${FLIR_OE4_WALLTIME:-12:00:00}" \
  --array=0-2%1 --job-name="oe4-$PHASE" \
  --output="$LOGS/$PHASE-%A_%a.out" --error="$LOGS/$PHASE-%A_%a.err" \
  "$ROOT/scripts/hypatia/oe4.sbatch" "$PHASE" "$CONFIG" "$PYTHON" "$ROOT"
