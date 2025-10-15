#!/usr/bin/env bash
# Bootstrap the DocsToKG development environment.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV_PATH="$ROOT_DIR/.venv"

if [[ ! -d "$VENV_PATH" ]]; then
    echo "[bootstrap] Creating virtual environment at $VENV_PATH"
    python -m venv "$VENV_PATH"
else
    echo "[bootstrap] Reusing existing virtual environment at $VENV_PATH"
fi

echo "[bootstrap] Upgrading pip inside the virtual environment"
"$VENV_PATH/bin/pip" install --upgrade pip

echo "[bootstrap] Installing DocsToKG in editable mode"
"$VENV_PATH/bin/pip" install -e .

if [[ -f "docs/build/sphinx/requirements.txt" ]]; then
    echo "[bootstrap] (Optional) Install documentation dependencies with:"
    echo "  $VENV_PATH/bin/pip install -r docs/build/sphinx/requirements.txt"
fi

echo
echo "[bootstrap] Done. If you use direnv, run 'direnv allow' to load the environment."
echo "[bootstrap] Otherwise, activate the virtualenv with:"
echo "  source $VENV_PATH/bin/activate"
