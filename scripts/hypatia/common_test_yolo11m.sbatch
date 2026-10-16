#!/usr/bin/env bash
# Request partition, memory, CPUs, walltime and GPU explicitly at submission.
#SBATCH --job-name=flir-common-yolo11m
#SBATCH --output=common-yolo11m-%A_%a.log
set -euo pipefail
: "${FLIR_REPO:?Export repository checkout}"
: "${FLIR_EXECUTOR_OUTPUT:?Export external output root}"
cd "$FLIR_REPO"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
mode="${FLIR_EXECUTOR_MODE:-train}"
case "$mode" in
  synthetic-smoke|probe)
    bash scripts/hypatia/common_test_yolo11m.sh "$mode" \
      "$FLIR_EXECUTOR_OUTPUT/${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID:-0}" ;;
  train)
    : "${FLIR_EXECUTOR_PLAN:?Export verified executor plan directory}"
    args=("$FLIR_EXECUTOR_PLAN" "$FLIR_EXECUTOR_OUTPUT" --index "${SLURM_ARRAY_TASK_ID:?Array index required}")
    if [[ "${FLIR_EXECUTOR_RESUME:-0}" == 1 ]]; then args+=(--resume); fi
    bash scripts/hypatia/common_test_yolo11m.sh train "${args[@]}" ;;
  *) echo "Invalid mode" >&2; exit 2 ;;
esac
