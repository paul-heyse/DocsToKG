#!/usr/bin/env bash
# Run repository pre-commit hooks in the current virtual environment.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV="$ROOT_DIR/.venv"
PRE_COMMIT_BIN="$VENV/bin/pre-commit"

if [[ ! -d "$VENV" ]]; then
    echo "[pre-commit] error: missing virtual environment at $VENV" >&2
    echo "[pre-commit] hint: run ./scripts/bootstrap_env.sh first." >&2
    exit 1
fi

if [[ ! -x "$PRE_COMMIT_BIN" ]]; then
    echo "[pre-commit] Installing pre-commit into the virtual environment"
    "$VENV/bin/pip" install pre-commit >/dev/null
fi

exec "$PRE_COMMIT_BIN" run --hook-stage manual --all-files "$@"
