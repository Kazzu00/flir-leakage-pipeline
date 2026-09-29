#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
exec "${FLIR_PYTHON:-python}" -m flir_pipeline.sequences.experiments.slurm status "$@"
