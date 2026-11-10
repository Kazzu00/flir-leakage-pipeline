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
  train-approved)
    : "${FLIR_EXECUTOR_PLAN:?}"
    : "${FLIR_EXECUTOR_PREPARATION:?}"
    : "${FLIR_EXECUTOR_CHECKPOINT:?}"
    : "${FLIR_EXECUTOR_AUTHORIZATION:?}"
    : "${FLIR_EXECUTOR_SIGNATURE:?}"
    : "${FLIR_EXECUTOR_SIGNERS:?}"
    : "${FLIR_EXECUTOR_REVIEWER:?}"
    # Validate approval and materialization BEFORE sbatch allocation.
    count="$(uv run --frozen --no-sync python -c '
import os
from pathlib import Path
from flir_pipeline.detection.common_admission import verify_admission
a=verify_admission(
    Path(os.environ["FLIR_EXECUTOR_PLAN"]),
    Path(os.environ["FLIR_EXECUTOR_PREPARATION"]),
    Path(os.environ["FLIR_EXECUTOR_CHECKPOINT"]),
    Path(os.environ["FLIR_EXECUTOR_AUTHORIZATION"]),
    Path(os.environ["FLIR_EXECUTOR_SIGNATURE"]),
    Path(os.environ["FLIR_EXECUTOR_SIGNERS"]),
    os.environ["FLIR_EXECUTOR_REVIEWER"],
)
print(len(a["plan"]["runs"]))
')"
    [[ "$count" =~ ^[1-9][0-9]*$ ]] || exit 2
    ;;
  synthetic-smoke|probe) count=1 ;;
  *) echo "Only synthetic-smoke, probe, train-approved or legacy blocked train accepted" >&2; exit 2 ;;
esac
sbatch --array="0-$((count - 1))%$concurrency" "$@" scripts/hypatia/common_test_yolo11m.sbatch
