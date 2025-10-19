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
       ./scripts/dev.sh doctor

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

BIN="$VENV/bin"
SRC="$ROOT_DIR/src"

if [[ ! -x "$BIN/python" ]]; then
    echo "[dev] error: missing virtual environment at $VENV" >&2
    echo "[dev] hint: python3.13 -m venv .venv && source .venv/bin/activate && pip install -e . && pip install -r requirements.txt" >&2
    exit 1
fi

# Compose environment similarly to .envrc
ENV_VARS=( "PATH=$BIN:$PATH" "VIRTUAL_ENV=$VENV" "PYTHONNOUSERSITE=1" )
if [[ -d "$SRC" ]]; then
    if [[ -n "${PYTHONPATH:-}" ]]; then
        ENV_VARS+=( "PYTHONPATH=$SRC:$PYTHONPATH" )
    else
        ENV_VARS+=( "PYTHONPATH=$SRC" )
    fi
fi

case "$1" in
    exec)
        shift
        if [[ $# -eq 0 ]]; then
            echo "[dev] error: exec requires a command" >&2
            usage
            exit 1
        fi
        exec env "${ENV_VARS[@]}" "$@"
        ;;
    python)
        shift
        exec "$BIN/python" "$@"
        ;;
    pip)
        shift
        exec env "${ENV_VARS[@]}" PIP_REQUIRE_VIRTUALENV=1 "$BIN/pip" "$@"
        ;;
    doctor)
        shift || true
        exec "$BIN/python" - <<'PY'
import os
import sys

print("sys.executable:", sys.executable)
print("VIRTUAL_ENV:", os.environ.get("VIRTUAL_ENV"))
print("PATH head:", os.environ.get("PATH", "").split(os.pathsep)[:3])
print("PYTHONPATH head:", os.environ.get("PYTHONPATH", "").split(os.pathsep)[:3])
print("DocsToKG importable:", end=" ")
try:
    import DocsToKG  # noqa: F401
except Exception as exc:  # pragma: no cover - diagnostic aid
    print("no ->", exc.__class__.__name__, exc)
else:
    print("yes")
PY
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
