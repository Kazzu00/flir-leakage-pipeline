#!/usr/bin/env bash
#SBATCH --job-name=flir-base-hud
#SBATCH --output=flir-base-hud-%j.log
set -euo pipefail
: "${FLIR_REPO:?Repository required}"
export FLIR_VARIANT=base_hud
bash "$FLIR_REPO/scripts/hypatia/hud_comparison.sbatch"
