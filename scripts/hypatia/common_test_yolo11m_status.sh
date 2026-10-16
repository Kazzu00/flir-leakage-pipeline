#!/usr/bin/env bash
set -euo pipefail
job="${1:?SLURM job ID required}"
[[ "$job" =~ ^[0-9]+$ ]] || exit 2
squeue --jobs="$job"
sacct --jobs="$job" --format=JobID,State,ExitCode,Elapsed,AllocTRES,MaxRSS
