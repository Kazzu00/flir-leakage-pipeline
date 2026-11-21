#!/usr/bin/env bash
set -euo pipefail
# Slurm spools this script elsewhere; never resolve repository from BASH_SOURCE.
ROOT="${4:?Missing immutable bundle root}"
PHASE="${1:?Missing phase}"
CONFIG="${2:?Missing config}"
PYTHON="${3:?Missing runtime interpreter}"
INDEX="${SLURM_ARRAY_TASK_ID:?Use submit_oe4.sh}"
case "$PHASE" in train|evaluate) ;; *) exit 2 ;; esac
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$ROOT"
"$PYTHON" scripts/closeout/verify_bundle.py "$ROOT"
"$PYTHON" -m flir_pipeline.closeout "oe4-$PHASE" "$CONFIG" "$INDEX"
