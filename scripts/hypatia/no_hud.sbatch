#!/usr/bin/env bash
#SBATCH --job-name=flir-no-hud
#SBATCH --output=flir-no-hud-%j.log
set -euo pipefail
: "${FLIR_REPO:?Repository required}"
export FLIR_VARIANT=no_hud
bash "$FLIR_REPO/scripts/hypatia/hud_comparison.sbatch"
