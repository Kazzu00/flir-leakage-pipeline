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
    : "${FLIR_TECHNICAL_FREEZE:?Export operational technical freeze}"
    # Worker repeats all technical checks; no signature or review flags.
    count=$(bash scripts/hypatia/common_test_yolo11m.sh admission-check "$FLIR_TECHNICAL_FREEZE")
    [[ "$count" =~ ^[1-9][0-9]*$ ]] || exit 2 ;;
  synthetic-smoke|probe|pilot) count=1 ;;
  *) echo "Only synthetic-smoke, probe, pilot or train accepted" >&2; exit 2 ;;
esac
# Optional singleton for first authorized run or an explicit resume.
indices="${FLIR_EXECUTOR_INDICES:-0-$((count - 1))}"
[[ "$indices" =~ ^[0-9]+(-[0-9]+)?$ ]] || exit 2
first="${indices%%-*}"
last="${indices##*-}"
(( first <= last && last < count )) || { echo "Array index outside verified matrix" >&2; exit 2; }
sbatch --array="$indices%$concurrency" "$@" scripts/hypatia/common_test_yolo11m.sbatch
