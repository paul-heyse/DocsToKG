#!/usr/bin/env bash
# Bootstrap the DocsToKG development environment.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV_PATH="$ROOT_DIR/.venv"

# Pick a Python interpreter (default to Python 3.13).
PYTHON_BIN="${PYTHON_BIN:-python3.13}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="python3"
    elif command -v python >/dev/null 2>&1; then
        PYTHON_BIN="python"
    else
        echo "[bootstrap] error: could not find a Python interpreter (expected python3.13)" >&2
        exit 1
    fi
    echo "[bootstrap] warning: falling back to $PYTHON_BIN (set PYTHON_BIN to override)" >&2
fi

if [[ ! -d "$VENV_PATH" ]]; then
    echo "[bootstrap] Creating virtual environment at $VENV_PATH"
    "$PYTHON_BIN" -m venv "$VENV_PATH"
else
    echo "[bootstrap] Reusing existing virtual environment at $VENV_PATH"
fi

echo "[bootstrap] Upgrading pip inside the virtual environment"
"$VENV_PATH/bin/pip" install --upgrade pip

if [[ -f "$ROOT_DIR/requirements.txt" ]]; then
    echo "[bootstrap] Installing project requirements"
    "$VENV_PATH/bin/pip" install --no-cache-dir -r "$ROOT_DIR/requirements.txt"
else
    echo "[bootstrap] warning: requirements.txt not found; skipping dependency install" >&2
fi

echo "[bootstrap] Installing DocsToKG in editable mode"
"$VENV_PATH/bin/pip" install -e .

if [[ -f "docs/build/sphinx/requirements.txt" ]]; then
    echo "[bootstrap] (Optional) Install documentation dependencies with:"
    echo "  $VENV_PATH/bin/pip install -r docs/build/sphinx/requirements.txt"
fi

# Sanity-check LFS backed wheels to catch missing git-lfs pulls.
if command -v head >/dev/null 2>&1; then
    for wheel in "$ROOT_DIR"/ci/wheels/*.whl; do
        [[ -f "$wheel" ]] || continue
        if head -c 32 "$wheel" | grep -q "version https://git-lfs"; then
            echo "[bootstrap] warning: $wheel is a Git LFS pointer; run 'git lfs install && git lfs pull'" >&2
        fi
    done
fi

echo
echo "[bootstrap] Done. If you use direnv, run 'direnv allow' to load the environment."
echo "[bootstrap] Otherwise, activate the virtualenv with:"
echo "  source $VENV_PATH/bin/activate"
