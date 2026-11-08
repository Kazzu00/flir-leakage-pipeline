#!/bin/bash -l
# Synthetic diagnostics only. Resources match the supplied failing-job evidence.
#SBATCH --job-name=flir-cuda-diagnosis
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:20:00
#SBATCH --output=flir-cuda-diagnosis-%j.out
#SBATCH --error=flir-cuda-diagnosis-%j.err
set -euo pipefail
umask 077
: "${SLURM_JOB_ID:?Submit one diagnostic job through Slurm}"
: "${FLIR_REPO:?Existing executor checkout}"
: "${FLIR_DIAGNOSTIC_SCRIPT:?Absolute path to diagnostic Python script}"
: "${FLIR_PYTHON:?Absolute existing detection venv Python}"
: "${FLIR_CUDA_OUTPUT:?New external output directory}"
cd "$FLIR_REPO"
mkdir "$FLIR_CUDA_OUTPUT"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export YOLO_OFFLINE=true
export UV_OFFLINE=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
printf '%s\n' 'This job does not execute the official smoke or any FAC training.'
git rev-parse HEAD > "$FLIR_CUDA_OUTPUT/source-commit.txt"
scontrol show job "$SLURM_JOB_ID" > "$FLIR_CUDA_OUTPUT/slurm-job.txt"
module list > "$FLIR_CUDA_OUTPUT/modules-inherited.txt" 2>&1 || true
status=0
run_case() {
    local name="$1"
    shift
    # One new interpreter per comparison; a failed CUDA context is never reused.
    if "$FLIR_PYTHON" -u "$FLIR_DIAGNOSTIC_SCRIPT" "$FLIR_CUDA_OUTPUT/$name" "$@" \
       > "$FLIR_CUDA_OUTPUT/$name.stdout" 2> "$FLIR_CUDA_OUTPUT/$name.stderr"; then
        printf '%s 0\n' "$name" >> "$FLIR_CUDA_OUTPUT/exit-codes.txt"
    else
        local rc=$?
        printf '%s %s\n' "$name" "$rc" >> "$FLIR_CUDA_OUTPUT/exit-codes.txt"
        status=1
    fi
}
run_case inherited-forward-640 --case forward --size 640 --profile bare
run_case inherited-loss-640 --case loss --size 640 --profile bare
run_case configured-loss-640 --case loss --size 640 --profile configured
run_case configured-loss-160 --case loss --size 160 --profile configured
run_case configured-loss-320 --case loss --size 320 --profile configured
# Only this last subprocess environment loads the explicitly reported October modules.
# No driver, library, account limit, allocator fraction or scientific setting is edited.
if (
    status=0
    if ! module load python/3.11 cuda/11.8; then
        printf '%s\n' 'MODULE_LOAD_FAILED' > "$FLIR_CUDA_OUTPUT/modules-october.txt"
        exit 1
    fi
    module list > "$FLIR_CUDA_OUTPUT/modules-october.txt" 2>&1
    run_case october-modules-loss-640 --case loss --size 640 --profile configured
    exit "$status"
); then
    :
else
    status=1
fi
exit "$status"
