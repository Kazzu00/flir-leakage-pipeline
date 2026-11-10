#!/usr/bin/env bash
# Shared Windows/Linux CLI contract; all paths and resources supplied by caller.
set -euo pipefail
# A shared --no-sync environment can point its editable install at another
# worktree. Resolve this script's checkout before importing the CLI.
executor_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "$executor_repo"
export PYTHONPATH="$executor_repo/src${PYTHONPATH:+:$PYTHONPATH}"
operation="${1:?preflight|synthetic-smoke|probe|dry-run|prepare|verify|status|train}"
shift
case "$operation" in
  preflight|synthetic-smoke|probe|dry-run|prepare|verify|status|train|admit|verify-weights|pilot|freeze|admission-check|release-test|prepare-test|evaluate|compare) ;;
  *) echo "Unknown operation: $operation" >&2; exit 2 ;;
esac
exec uv run --frozen --no-sync flir-pipeline detection common-executor "$operation" "$@"
