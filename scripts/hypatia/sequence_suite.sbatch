#!/usr/bin/env bash
# Scientific settings belong in YAML/Python; adjust SLURM resources after pilots.
#SBATCH --job-name=flir-sequence-evidence
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=sequence-suite-%j.log
set -euo pipefail

module load python/3.11
source "${FLIR_VENV:-$HOME/.venvs/flir-pipeline}/bin/activate"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1
: "${FLIR_MANIFEST:?Export FLIR_MANIFEST}"
: "${FLIR_CLIP_FEATURES:?Export FLIR_CLIP_FEATURES}"
: "${FLIR_DINOV2_FEATURES:?Export FLIR_DINOV2_FEATURES}"
: "${FLIR_SEQUENCE_OUTPUT:?Export FLIR_SEQUENCE_OUTPUT}"

flir-pipeline sequences experiment suite \
  --family "${FLIR_FAMILY:-all}" \
  --profile "${FLIR_SEQUENCE_PROFILE:-configs/hypatia_sequence_experiments.yaml}" "$@"
