#!/usr/bin/env bash
# Shared Windows/Linux CLI contract; all paths and resources supplied by caller.
set -euo pipefail
operation="${1:?preflight|synthetic-smoke|probe|dry-run|prepare|verify|status|train}"
shift
case "$operation" in
  preflight|synthetic-smoke|probe|dry-run|prepare|verify|status|train|admission-check|prepare-test|evaluate) ;;
  *) echo "Unknown operation: $operation" >&2; exit 2 ;;
esac
# An already provisioned cluster venv can invoke the same CLI without installation.
if [[ -n "${FLIR_PYTHON:-}" ]]; then
  exec "$FLIR_PYTHON" -c 'from flir_pipeline.cli import app; app()' \
    detection common-executor "$operation" "$@"
fi
exec uv run --frozen --no-sync flir-pipeline detection common-executor "$operation" "$@"
