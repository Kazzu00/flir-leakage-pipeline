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
    # Admission is checked BEFORE allocating cluster resources. Currently closed.
    uv run --frozen --no-sync python -c \
      'from flir_pipeline.detection.common_executor import require_admission; require_admission()'
    echo "No admitted real-run submission contract implemented" >&2; exit 1 ;;
  synthetic-smoke|probe) count=1 ;;
  *) echo "Only synthetic-smoke, probe or train accepted" >&2; exit 2 ;;
esac
sbatch --array="0-$((count - 1))%$concurrency" "$@" scripts/hypatia/common_test_yolo11m.sbatch
