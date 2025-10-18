#!/usr/bin/env bash
# Convenience wrapper for running commands inside the project virtual environment.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV="$ROOT_DIR/.venv"

usage() {
    cat <<'EOF'
Usage: ./scripts/dev.sh exec <command> [args...]
       ./scripts/dev.sh python [args...]
       ./scripts/dev.sh pip [args...]

Examples:
  ./scripts/dev.sh exec pytest -q
  ./scripts/dev.sh python -m DocsToKG.DocParsing.cli --help
  ./scripts/dev.sh pip install some-package
EOF
}

if [[ $# -lt 1 ]]; then
    usage
    exit 1
fi

if [[ ! -d "$VENV" ]]; then
    echo "[dev] error: missing virtual environment at $VENV" >&2
    echo "[dev] hint: run ./scripts/bootstrap_env.sh first." >&2
    exit 1
fi

BIN="$VENV/bin"

case "$1" in
    exec)
        shift
        if [[ $# -eq 0 ]]; then
            echo "[dev] error: exec requires a command" >&2
            usage
            exit 1
        fi
        exec "$BIN/env" "PATH=$BIN:$PATH" "VIRTUAL_ENV=$VENV" "$@"
        ;;
    python)
        shift
        exec "$BIN/python" "$@"
        ;;
    pip)
        shift
        exec "$BIN/pip" "$@"
        ;;
    help|-h|--help)
        usage
        ;;
    *)
        echo "[dev] error: unknown subcommand '$1'" >&2
        usage
        exit 1
        ;;
esac
