#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:?Missing immutable bundle root}"
CONFIG="${2:?Missing config}"
PYTHON="${3:?Missing interpreter}"
DATA="${4:?Missing exploratory data}"
WORK="${5:?Missing operation directory}"
INDEX="${SLURM_ARRAY_TASK_ID:?Submit as array 0-1}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$ROOT"
"$PYTHON" scripts/closeout/verify_bundle.py "$ROOT"
"$PYTHON" -m flir_pipeline.closeout denoised-train "$CONFIG" "$DATA" "$WORK" "$INDEX"
