#!/usr/bin/env bash
# Request partition, CPUs, walltime and GPU explicitly at submission.
# Provisional host-memory request: 12G yielded RLIMIT_AS=13.2GiB on Hypatia.
# This requests resources through Slurm; it does not remove process limits.
#SBATCH --mem=32G
#SBATCH --job-name=flir-common-yolo11m
#SBATCH --output=common-yolo11m-%A_%a.log
#SBATCH --signal=TERM@120
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
    : "${FLIR_TECHNICAL_FREEZE:?Technical operational freeze required}"
    args=("$FLIR_EXECUTOR_PLAN" "$FLIR_EXECUTOR_OUTPUT" --index "${SLURM_ARRAY_TASK_ID:?Array index required}"
      --technical-freeze "$FLIR_TECHNICAL_FREEZE")
    if [[ "${FLIR_EXECUTOR_RESUME:-0}" == 1 ]]; then
      args+=(--resume)
    fi
    srun --ntasks=1 bash scripts/hypatia/common_test_yolo11m.sh train "${args[@]}" ;;
  pilot)
    : "${FLIR_DATA_GATE:?Technical data gate required}"
    : "${FLIR_WEIGHTS_RECEIPT:?Official checkpoint receipt required}"
    : "${FLIR_GPU_PROBE:?Completed synthetic batch operation directory required}"
    args=("$FLIR_DATA_GATE" "$FLIR_WEIGHTS_RECEIPT" "$FLIR_GPU_PROBE" "$FLIR_EXECUTOR_OUTPUT"
      --batch "${FLIR_COMMON_BATCH:-1}" --index "${FLIR_PILOT_INDEX:-0}")
    [[ "${FLIR_EXECUTOR_RESUME:-0}" != 1 ]] || args+=(--resume)
    srun --ntasks=1 bash scripts/hypatia/common_test_yolo11m.sh pilot "${args[@]}" ;;
  *) echo "Invalid mode" >&2; exit 2 ;;
esac
