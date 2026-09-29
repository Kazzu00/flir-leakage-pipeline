#!/usr/bin/env bash
# Thin transport only: planning/submission in Python, fitting in SLURM workers.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec "${FLIR_PYTHON:-python}" -m flir_pipeline.sequences.experiments.slurm submit "$@"
