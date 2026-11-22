#!/usr/bin/env bash
# Template only: current curated/no-HUD packages are not admitted to this executor.
#SBATCH --job-name=flir-hud-pair
#SBATCH --output=flir-hud-%j.log
#SBATCH --signal=TERM@120
set -euo pipefail
: "${FLIR_REPO:?Repository checkout required}"
: "${FLIR_VARIANT:?base_hud or no_hud required}"
: "${FLIR_EXECUTOR_OUTPUT:?Separate output directory required}"
case "$FLIR_VARIANT" in base_hud|no_hud) ;; *) exit 2 ;; esac
cd "$FLIR_REPO"
# Reuse the existing environment recorder. No direct YOLO call or admission bypass.
bash scripts/hypatia/common_test_yolo11m.sh preflight \
  "$FLIR_EXECUTOR_OUTPUT/environment-${SLURM_JOB_ID}.json"
echo "BLOCKED: paired curated HUD/no-HUD admission is not implemented; no training started" >&2
exit 2
