#!/usr/bin/env bash
set -euo pipefail
: "${FLIR_REPO:?Export repository checkout}"
: "${FLIR_EXECUTOR_OUTPUT:?Export external output root}"
cd "$FLIR_REPO"
mode="${FLIR_EXECUTOR_MODE:-train}"
concurrency="${FLIR_EXECUTOR_CONCURRENCY:-1}"
[[ "$concurrency" =~ ^[1-9][0-9]*$ ]] || exit 2
case "$mode" in
  train)
    : "${FLIR_EXECUTOR_PLAN:?Export existing plan directory}"
    : "${FLIR_ADMISSION:?Export signed admission document}"
    : "${FLIR_ADMISSION_SIGNATURE:?Export detached signature}"
    # Worker repeats admission; missing trust never reaches sbatch.
    count=$(bash scripts/hypatia/common_test_yolo11m.sh admission-check \
      "$FLIR_EXECUTOR_PLAN" "$FLIR_EXECUTOR_OUTPUT" "$FLIR_ADMISSION" "$FLIR_ADMISSION_SIGNATURE")
    [[ "$count" =~ ^[1-9][0-9]*$ ]] || exit 2 ;;
  synthetic-smoke|probe) count=1 ;;
  *) echo "Only synthetic-smoke, probe or train accepted" >&2; exit 2 ;;
esac
# Optional singleton for first authorized run or an explicit resume.
indices="${FLIR_EXECUTOR_INDICES:-0-$((count - 1))}"
[[ "$indices" =~ ^[0-9]+(-[0-9]+)?$ ]] || exit 2
sbatch --array="$indices%$concurrency" "$@" scripts/hypatia/common_test_yolo11m.sbatch
