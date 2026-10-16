#!/usr/bin/env bash
# Shared Windows/Linux CLI contract; all paths and resources supplied by caller.
set -euo pipefail
operation="${1:?preflight|synthetic-smoke|probe|dry-run|prepare|verify|status|train}"
shift
case "$operation" in
  preflight|synthetic-smoke|probe|dry-run|prepare|verify|status|train) ;;
  *) echo "Unknown operation: $operation" >&2; exit 2 ;;
esac
exec uv run --frozen --no-sync flir-pipeline detection common-executor "$operation" "$@"
