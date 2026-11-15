#!/bin/bash -l
# One paired microstep, then at most one official synthetic epoch. Never FAC.
# Only resource intervention versus job 792984: request 32G rather than 12G RAM.
#SBATCH --job-name=flir-cuda-verify
#SBATCH --partition=gpu
#SBATCH --nodelist=nodei-gpu-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=flir-cuda-verify-%j.out
#SBATCH --error=flir-cuda-verify-%j.err
set -euo pipefail
umask 077
: "${SLURM_JOB_ID:?Submit once through Slurm}"
: "${FLIR_REPO:?Existing unchanged executor checkout}"
: "${FLIR_DIAGNOSTIC_SCRIPT:?Audited diagnostic script; computation unchanged from job 792984}"
: "${FLIR_PYTHON:?Existing detection venv Python}"
: "${FLIR_CUDA_OUTPUT:?Fresh external verification directory}"
cd "$FLIR_REPO"
mkdir "$FLIR_CUDA_OUTPUT"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export YOLO_OFFLINE=true
export UV_OFFLINE=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
git rev-parse HEAD > "$FLIR_CUDA_OUTPUT/source-commit.txt"
scontrol show job "$SLURM_JOB_ID" > "$FLIR_CUDA_OUTPUT/slurm-job.txt"
module list > "$FLIR_CUDA_OUTPUT/modules.txt" 2>&1 || true
# Read only: confirm the proposed scheduler explanation, without changing it.
scontrol show config | awk '/VSizeFactor|PropagateResourceLimits|JobAcctGatherType|TaskPlugin/ {print}' \
    > "$FLIR_CUDA_OUTPUT/slurm-memory-policy.txt" || true
run_case() {
    local name="$1"
    shift
    if "$FLIR_PYTHON" -u "$FLIR_DIAGNOSTIC_SCRIPT" "$FLIR_CUDA_OUTPUT/$name" "$@" \
       > "$FLIR_CUDA_OUTPUT/$name.stdout" 2> "$FLIR_CUDA_OUTPUT/$name.stderr"; then
        printf '%s 0\n' "$name" >> "$FLIR_CUDA_OUTPUT/exit-codes.txt"
    else
        local rc=$?
        printf '%s %s\n' "$name" "$rc" >> "$FLIR_CUDA_OUTPUT/exit-codes.txt"
        return "$rc"
    fi
}
# Same computation, seed, targets, size and configured path as the failed baseline.
# The audited script only distinguishes active phase from last checkpoint.
# A new interpreter is used for the subsequent full smoke.
if ! run_case configured-loss-640 --case loss --size 640 --profile configured; then
    printf '%s\n' 'SKIPPED: paired microstep failed; no smoke retry' > "$FLIR_CUDA_OUTPUT/smoke-status.txt"
    exit 1
fi
run_case official-smoke-640 --case smoke --size 640 --profile configured
