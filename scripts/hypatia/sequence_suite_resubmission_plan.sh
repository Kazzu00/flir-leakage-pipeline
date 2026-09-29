#!/usr/bin/env bash
# Read-only deterministic recovery plan; never resubmits or cancels implicitly.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
exec "${FLIR_PYTHON:-python}" -m flir_pipeline.sequences.experiments.slurm resubmission-plan "$@"
