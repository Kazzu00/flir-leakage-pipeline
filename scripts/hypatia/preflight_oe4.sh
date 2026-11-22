#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${FLIR_OE4_PYTHON:-$HOME/.venvs/flir-detection/bin/python}"
CONFIG="${FLIR_OE4_CONFIG:-$ROOT/configs/closeout/oe4_20261115.yaml}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
"$PYTHON" scripts/closeout/verify_bundle.py "$ROOT"
mkdir -p "$HOME/flir-oe4-hotfix-20261122/logs"
# CPU metadata/checksums only. Torch/CUDA checks run inside the GPU allocation.
"$PYTHON" -m flir_pipeline.closeout oe4-preflight "$CONFIG"
